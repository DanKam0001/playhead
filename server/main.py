"""Playhead backend: mints session tokens and serves the one tool that matters.

The shape here follows AssemblyAI's stored-agent + HTTP-tools pattern, so this
backend holds **no** websocket to AssemblyAI, no tool dispatcher, and no session
loop. AssemblyAI calls `/tools/passage_at_playhead` from its own servers; the
browser holds the only websocket, using a short-lived token minted here so the
API key never leaves the server.

What makes this project more than a wrapper is the tool itself: the listener's
playback position selects the passage. "What did that last part mean?" has no
content to search for, so similarity search over the book returns noise.
Position answers it exactly.
"""
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional

import urllib.error
import urllib.request
import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from playhead.library import Library

from . import books
from .store import build_store

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def _setting(name: str, default: str = "") -> str:
    """Read PLAYHEAD_<name>, falling back to the old ECHOREAD_<name>.

    The project was called EchoRead until 2026-09-19. Accepting both means a
    running deployment does not go dark between this code landing and its
    environment variables being renamed. The fallback can be deleted once
    Vercel only has PLAYHEAD_* set.
    """
    return (os.getenv(f"PLAYHEAD_{name}")
            or os.getenv(f"ECHOREAD_{name}")
            or default)


BOOK = _setting("BOOK", "relativity")
# Read lazily-tolerant: a missing key should surface as a clear 500 on the
# one endpoint that needs it, not as an import-time crash that takes the
# whole function down and reports nothing useful.
AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
AGENT_ID = _setting("AGENT_ID", "")

# How much audio may be sent through this deployment, as settings rather than
# constants -- a bigger host, or a self-hosted one, should be able to raise
# them without touching code. Note the upload ceiling is only ours down to
# whatever the platform itself enforces: Vercel rejects a request body over
# ~4.5 MB at the edge, before any of this runs. Raising MAX_UPLOAD_MB past that
# only helps somewhere without that cap.
MAX_UPLOAD_MB = float(_setting("MAX_UPLOAD_MB", "4"))
# 0 means no ceiling of ours. Kept as a setting rather than deleted so a
# deployment that is not sitting on someone's personal API key can put one
# back without a code change.
MAX_SOURCE_MB = float(_setting("MAX_SOURCE_MB", "0"))
# Books added per hour from one address; 0 disables the check entirely.
RATE_PER_HOUR = int(_setting("RATE_PER_HOUR", "0"))

app = FastAPI(title="Playhead")
library = Library(ROOT / "data" / f"{BOOK}.db")
playheads = build_store()


def library_for(session_id: Optional[str]):
    """The book this listener is on, as something with Library's methods.

    The shipped book is a SQLite file in the repo; anything anyone added is in
    Redis. Both answer window() and search(), so neither tool below has to know
    which one it is holding.
    """
    book_id = playheads.get_book(session_id)
    if not book_id or book_id == BOOK:
        return library, BOOK
    rec = books.load(playheads, book_id)
    if not rec or rec.status != "ready":
        return library, BOOK
    return books.RedisLibrary(playheads, rec), rec.title


# ---------- browser -> backend ----------

class Playhead(BaseModel):
    session_id: str
    seconds: float
    book: Optional[str] = None


@app.post("/api/playhead")
def report_playhead(p: Playhead):
    """The browser reports where the audiobook is, roughly once a second.

    AssemblyAI calls the tool server-to-server and has no idea where playback
    is, so the position has to arrive out of band. Cheap, and it means the tool
    stays stateless from the agent's point of view.
    """
    playheads.set(p.session_id, max(0.0, p.seconds))
    # Which book, on the same seam and for the same reason as the position:
    # the browser knows, and the tool call from AssemblyAI cannot see it.
    if p.book:
        playheads.set_book(p.session_id, p.book)
    # The same heartbeat carries jumps back. The agent cannot move the audio
    # itself -- it runs on AssemblyAI's servers -- so go_to_topic leaves a
    # pending position here and the browser collects it within the second.
    seek = playheads.take_seek(p.session_id)
    return {"ok": True, "seek": seek} if seek is not None else {"ok": True}


class PriorQuestion(BaseModel):
    t: float = 0.0
    q: str


class ContextPost(BaseModel):
    session_id: Optional[str] = None
    questions: list[PriorQuestion] = []


@app.post("/api/context")
def set_context(c: ContextPost):
    """What this listener asked in earlier sessions.

    Notes live in the listener's browser, not here -- there are no accounts.
    The browser hands over a short digest at connect time so the agent has some
    continuity, and it expires with the session.
    """
    if not c.questions:
        return {"ok": True, "carried": 0}
    lines = [f"- at {int(q.t)//60}:{int(q.t)%60:02d}, they asked: {q.q.strip()[:160]}"
             for q in c.questions[-8:] if q.q.strip()]
    playheads.set_context(c.session_id, chr(10).join(lines))
    return {"ok": True, "carried": len(lines)}


@app.get("/api/session")
def new_session():
    """Mint a short-lived agent token plus the session id the browser will use."""
    if not AAI_KEY:
        raise HTTPException(500, "ASSEMBLYAI_API_KEY is not set on this deployment.")
    if not AGENT_ID:
        raise HTTPException(500, "PLAYHEAD_AGENT_ID is not set. Run scripts/create_agent.py.")
    url = ("https://agents.assemblyai.com/v1/token"
           "?expires_in_seconds=300&max_session_duration_seconds=1800")
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {AAI_KEY}"})
    try:
        token = json.load(urllib.request.urlopen(req))["token"]
    except urllib.error.HTTPError as e:
        raise HTTPException(502, f"token mint failed: {e.read().decode()[:200]}")
    return {
        "token": token,
        "agent_id": AGENT_ID,
        "session_id": f"s{int(time.time()*1000)}{os.urandom(3).hex()}",
        "book": BOOK,
        "duration": library.duration_hint(),
    }


# ---------- AssemblyAI -> backend (the HTTP tool) ----------

class ToolCall(BaseModel):
    session_id: Optional[str] = None
    search: Optional[str] = None


@app.post("/tools/passage_at_playhead")
def passage_at_playhead(call: ToolCall):
    """The whole idea, as one endpoint.

    Returns prose, not codes: the response is read by a language model that has
    to speak the answer, so a failure explains itself in a sentence it can say.
    """
    t = playheads.get(call.session_id)
    if t is None:
        return {"ok": False,
                "message": "I can't tell where you are in the book yet - "
                           "is it playing?"}
    lib, _title = library_for(call.session_id)
    near = lib.window(t)
    if not near:
        return {"ok": False,
                "message": f"There's no indexed text around {int(t)//60}:{int(t)%60:02d}."}

    parts = [f"The listener is {int(t)//60} minutes {int(t)%60} seconds into the book.",
             "This is what they have just heard:",
             " ".join(c.text for c in near)]

    prior = playheads.get_context(call.session_id)
    if prior:
        parts += ["", "This listener has asked before:", prior,
                  "Only mention this if it is relevant to what they just asked."]

    if call.search:
        far = _search_earlier(lib, call.search, t, exclude={c.id for c in near})
        if far:
            parts += ["", f"Earlier in the book, on '{call.search}':",
                      " ".join(c.text for c in far)]

    return {"ok": True, "playhead_seconds": round(t, 1), "message": "\n".join(parts)}


class TopicCall(BaseModel):
    session_id: Optional[str] = None
    topic: str


@app.post("/tools/go_to_topic")
def go_to_topic(call: TopicCall):
    """Navigation by meaning: the mirror image of passage_at_playhead.

    You cannot skim an audiobook. A sighted reader flips to the right page in
    seconds; by ear the only controls are a scrubber and guesswork. So the
    listener names a topic and the book moves to it.

    The spoiler cap deliberately does NOT apply here. It exists to stop the
    agent volunteering what is ahead; being asked to go there is consent.
    """
    vec = _embed(call.topic)
    if vec is None:
        return {"ok": False,
                "message": "I can't look up topics right now - say roughly where "
                           "you want to go instead."}
    lib, _title = library_for(call.session_id)
    hits = lib.search(vec, k=1)
    if not hits:
        return {"ok": False,
                "message": f"I couldn't find anything about {call.topic} in this book."}

    target = hits[0]
    # Start a little before the passage, so the listener hears it introduced
    # rather than landing mid-sentence.
    start = max(0.0, target.start_s - 8)
    playheads.request_seek(call.session_id, start)
    return {"ok": True, "seek_seconds": round(start, 1),
            "message": (f"Moving the book to {int(start)//60}:{int(start)%60:02d}, "
                        f"where this comes up. Tell the listener where you are taking "
                        f"them and what is there, in one sentence. This is what "
                        f"plays next:" + chr(10) + target.text)}


class OutlineCall(BaseModel):
    session_id: Optional[str] = None


@app.post("/tools/book_outline")
def book_outline(call: OutlineCall):
    """What the whole book covers, so someone knows what they are walking into.

    The spoiler cap exists to stop the agent *volunteering* what is ahead. Being
    asked what a book is about is consent, the same way asking to be taken
    somewhere is -- so this deliberately reads the entire book, not the part
    already heard. Refusing here was the wrong behaviour: someone at 00:05 who
    asks "what is this?" got told it was too early to say, which is useless.

    Excerpts are sampled evenly across the duration rather than summarised by
    us. The agent already has a language model; what it lacks is the text.
    """
    lib, title = library_for(call.session_id)
    if not len(lib):
        return {"ok": False, "message": "There's no indexed book loaded right now."}

    picks = _spread(lib, 10)
    if not picks:
        return {"ok": False, "message": "I couldn't read the book's contents just now."}

    t = playheads.get(call.session_id)
    where = (f"They are {int(t)//60} minutes in." if t else "They are at the start.")
    body = "\n".join(f"[{int(c.start_s)//60:02d}:{int(c.start_s)%60:02d}] {c.text}"
                     for c in picks)
    return {"ok": True, "message": (
        f"Excerpts sampled evenly across the whole book, in order. "
        f"Total length {int(lib.duration_hint())//60} minutes. {where}\n\n{body}\n\n"
        f"Describe what this book covers and how it is organised, in three or "
        f"four sentences, so they know what they are getting into. They asked, "
        f"so telling them the shape of it is not a spoiler -- but if it is a "
        f"story, do not give away how it ends.")}


def _spread(lib, k: int = 10) -> list:
    """K chunks spaced evenly across the book, in reading order."""
    duration = lib.duration_hint()
    if not duration:
        return []
    picked, seen = [], set()
    for i in range(k):
        # Sample at the middle of each band rather than the edges, so the first
        # pick is not the title page and the last is not the licence notice.
        hits = lib.window(duration * (i + 0.5) / k, before=0.0, after=0.0)
        if hits and hits[0].id not in seen:
            seen.add(hits[0].id)
            picked.append(hits[0])
    return picked


def _embed(query: str):
    """One embedding, or None if embeddings are unavailable."""
    try:
        from playhead.brain import GeminiEmbedder
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            return None
        return GeminiEmbedder(key)([query], "query")[0]
    except Exception as exc:
        print(f"[tool] embed failed: {exc}")
        return None


def _search_earlier(lib, query: str, before_s: float, exclude) -> list:
    """Semantic lookback, capped at the playhead so it cannot spoil the book."""
    vec = _embed(query)
    if vec is None:
        return []
    try:
        hits = lib.search(vec, k=5, before_s=before_s)
        return [c for c in hits if c.id not in exclude][:2]
    except Exception as exc:
        print(f"[tool] lookback skipped: {exc}")
        return []


# ---------- bring your own audiobook ----------

class NewBook(BaseModel):
    title: str = ""
    audio_url: str


def _embedder():
    """The Gemini embedder, or None if this deployment has no key."""
    try:
        from playhead.brain import GeminiEmbedder
        key = os.getenv("GEMINI_API_KEY", "")
        return GeminiEmbedder(key) if key else None
    except Exception as exc:
        print(f"[books] embedder unavailable: {exc}")
        return None


def _builtin_card() -> dict:
    return {"id": BOOK, "title": "Relativity: The Special and General Theory",
            "audio_url": f"/audio/{BOOK}.mp3", "status": "ready", "error": "",
            "progress": 100, "chunks": len(library),
            "duration": round(library.duration_hint(), 1), "builtin": True}


# Public domain narration, so anyone landing on the demo has something real to
# try without hunting for a link. These are indexed on demand into the visitor's
# own shelf, not pre-built: it costs nothing until someone wants one, and
# watching it index is the clearest demonstration of what this does.
SUGGESTED = [
    {"title": "Babbage - The first computer, by its inventor",
     "audio_url": "https://archive.org/download/life_of_a_philosopher_1909_librivox/lifeofaphilosopher_09_babbage_64kb.mp3",
     "note": "\"The store\" and \"the mill\" - memory and CPU, described in 1864. "
             "Ask it what they are at 11:18"},
    {"title": "Babbage - The Analytical Engine, part 2",
     "audio_url": "https://archive.org/download/life_of_a_philosopher_1909_librivox/lifeofaphilosopher_10_babbage_64kb.mp3",
     "note": "Punched cards, conditional branching, and the cost of a machine "
             "nobody would fund"},
    {"title": "Russell - Introduction to Mathematical Philosophy, ch. 1",
     "audio_url": "https://archive.org/download/mathematicalphilosophy_1508_librivox/mathematicalphilosophy_01_russell_64kb.mp3",
     "note": "What a number actually is. Sets, logic, foundations"},
    {"title": "Poincare - Science and Hypothesis, ch. 1",
     "audio_url": "https://archive.org/download/science_and_hypothesis_librivox/scienceandhypothesis_01_poincare_64kb.mp3",
     "note": "On the nature of mathematical reasoning - induction and proof"},
    {"title": "Abbott - Flatland, ch. 1-3",
     "audio_url": "https://archive.org/download/flatland3_2603_librivox/flatland_01_abbott_64kb.mp3",
     "note": "Dimensions, from the inside. Short"},
    {"title": "Einstein - Relativity, ch. 10-12",
     "audio_url": "https://archive.org/download/relativity_librivox/relativity_10-12_einstein_64kb.mp3",
     "note": "The Lorentz transformation. Carries on from the shipped chapter"},
]



def _client_id(request: Request) -> str:
    """A random string the browser keeps in localStorage. Scopes a shelf so one
    visitor's books do not appear on another's; not a credential."""
    raw = (request.headers.get("x-client-id") or "").strip()
    return re.sub(r"[^A-Za-z0-9_-]", "", raw)[:48]


def _rate_limit(request: Request) -> None:
    """Per-IP ceiling on new books.

    The endpoint is unauthenticated by design -- there are no accounts -- so
    the thing that needs protecting is the transcription spend behind it.
    Called only once a link has passed validation: a rejected paste costs us
    one HEAD request, and charging someone's quota for a typo is just rude.

    RATE_PER_HOUR = 0 turns this off. The owner of the key gets to decide how
    much of it to leave lying in the road; the SSRF and content-type guards in
    books.py are not optional in the same way and stay on regardless.
    """
    if RATE_PER_HOUR <= 0:
        return
    ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or "unknown"
    try:
        n = playheads.kv_incr(f"playhead:rate:{ip}", 3600)
    except Exception as exc:
        print(f"[rate] check failed, allowing: {exc}")
        return
    if n > RATE_PER_HOUR:
        raise HTTPException(429, f"That's {RATE_PER_HOUR} books in an hour from this "
                                 f"address, which is the limit here. Try again later.")


@app.get("/api/books")
def list_books(request: Request):
    """The shipped book, then this browser's own, then things worth trying."""
    return {"books": [_builtin_card()] + books.shelf(playheads, _client_id(request)),
            "suggested": SUGGESTED,
            # The page reads its own ceilings from here rather than hardcoding
            # them, so raising a setting moves the check and the wording with it.
            "limits": {"upload_mb": MAX_UPLOAD_MB, "source_mb": MAX_SOURCE_MB}}


@app.post("/api/books")
def add_book(new: NewBook, request: Request):
    """Start indexing a book from a URL.

    Returns as soon as the transcription job is queued. The browser then polls
    GET /api/books/{id}, and each poll advances the work by one step -- there
    is no worker here, and a request that waited for a whole audiobook would be
    killed by the platform long before it finished.
    """
    if not AAI_KEY:
        raise HTTPException(500, "ASSEMBLYAI_API_KEY is not set on this deployment.")
    url = new.audio_url.strip()
    try:
        # Checked before anything is fetched or queued: this runs from our own
        # server, so an unvalidated link is a request forgery, and an unbounded
        # one is someone else's transcription bill.
        books.check_source(url, int(MAX_SOURCE_MB * 1_000_000))
        _rate_limit(request)
        rec = books.create(playheads, new.title or _title_from_url(url), url,
                           AAI_KEY, _client_id(request))
    except books.RejectedURL as e:
        raise HTTPException(400, str(e))
    except urllib.error.HTTPError as e:
        detail = e.read().decode()[:200]
        raise HTTPException(400, f"AssemblyAI could not take that link: {detail}")
    return rec.public()


@app.post("/api/books/upload")
async def upload_book(request: Request):
    """Add a book from a file on the listener's machine.

    The audio goes straight to AssemblyAI and is never stored here. The
    platform caps a request body at 4.5 MB, so this path is for a chapter or
    an episode; the page checks the size first and steers longer books to the
    URL form rather than letting them fail at the edge.
    """
    if not AAI_KEY:
        raise HTTPException(500, "ASSEMBLYAI_API_KEY is not set on this deployment.")
    _rate_limit(request)
    title = request.headers.get("x-book-title", "") or "Uploaded book"
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "No audio arrived.")
    if len(raw) > MAX_UPLOAD_MB * 1_000_000:
        raise HTTPException(413, f"That file is {len(raw) / 1e6:.1f} MB and this "
                                 f"deployment accepts {MAX_UPLOAD_MB:g} MB - "
                                 f"paste a link to it instead.")
    try:
        url = books.upload_bytes(raw, AAI_KEY)
        rec = books.create(playheads, title, url, AAI_KEY, _client_id(request))
    except urllib.error.HTTPError as e:
        raise HTTPException(400, f"Upload failed: {e.read().decode()[:200]}")
    # The browser plays its own local copy; this URL is AssemblyAI-only.
    out = rec.public()
    out["audio_url"] = ""
    return out


@app.get("/api/books/{book_id}/contents")
def book_contents(book_id: str):
    """Chapters, read off what the narrator announces.

    An audiobook file carries no structure at all -- it is one opaque stream,
    which is why skipping around is guesswork. The transcript gives it back.
    """
    if book_id == BOOK:
        return {"contents": books.contents(library)}
    rec = books.load(playheads, book_id)
    if not rec or rec.status != "ready":
        return {"contents": []}
    # AssemblyAI's own segmentation first: its headlines are written from the
    # content, so they work on the many recordings that never announce a
    # chapter out loud. Reading the narrator's announcements is the fallback.
    # Two or more, or it is not a table of contents. A short recording often
    # comes back as a single chapter, which is less navigable than even parts.
    stored = books.stored_chapters(playheads, book_id)
    if len(stored) >= 2:
        return {"contents": stored, "source": "auto_chapters"}
    return {"contents": books.contents(books.RedisLibrary(playheads, rec)),
            "source": "transcript"}


@app.get("/api/books/{book_id}")
def book_status(book_id: str):
    """Status, and one slice of work.

    Polling is the scheduler. Each call checks the transcript or embeds the
    next hundred chunks, so progress only moves while someone is watching --
    which is exactly when it matters.
    """
    if book_id == BOOK:
        return _builtin_card()
    rec = books.load(playheads, book_id)
    if not rec:
        raise HTTPException(404, "No such book.")
    rec = books.advance(playheads, rec, AAI_KEY, _embedder())
    return rec.public()


def _title_from_url(url: str) -> str:
    """A readable name out of a file name.

    Archive.org and LibriVox names carry encoding junk ('..._64kb.mp3') that
    would otherwise end up on screen as the title of someone's book.
    """
    name = url.rstrip("/").split("/")[-1].split("?")[0]
    name = re.sub(r"\.(mp3|m4a|m4b|wav|ogg|flac|aac|webm)$", "", name, flags=re.I)
    # No \b before the digits: an underscore is a word character, so "_64kb"
    # has no boundary to anchor to and the junk survives.
    name = re.sub(r"[_\- ]*\d{1,3} ?k(?:b|bps)?(?![a-z0-9])", "", name, flags=re.I)
    name = re.sub(r"[_-]+", " ", name).strip()
    words = [w if (w.isupper() and len(w) > 1) else w.capitalize() for w in name.split()]
    return " ".join(words) or "Untitled book"


@app.get("/api/health")
def health():
    return {"ok": True, "book": BOOK, "chunks": len(library),
            "agent_configured": bool(AGENT_ID), "store": playheads.kind}


# ---------- static ----------

# The repo ships one copy of the book audio, in public/ (that is what Vercel
# serves). audio/ is the working directory for building an index and is not
# committed, so fall back to public/audio for a fresh clone.
AUDIO = ROOT / "audio"
if not AUDIO.exists():
    AUDIO = ROOT / "public" / "audio"
if AUDIO.exists():
    # The browser streams the book from here; range requests come free, which
    # is what lets the <audio> element seek.
    app.mount("/audio", StaticFiles(directory=str(AUDIO)), name="audio")

if WEB.exists():
    app.mount("/static", StaticFiles(directory=str(WEB)), name="static")

    @app.get("/")
    def index():
        return FileResponse(str(WEB / "index.html"))

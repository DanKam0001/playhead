"""EchoRead backend: mints session tokens and serves the one tool that matters.

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

from echoread.library import Library

from . import books
from .store import build_store

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
BOOK = os.getenv("ECHOREAD_BOOK", "relativity")
# Read lazily-tolerant: a missing key should surface as a clear 500 on the
# one endpoint that needs it, not as an import-time crash that takes the
# whole function down and reports nothing useful.
AAI_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
AGENT_ID = os.getenv("ECHOREAD_AGENT_ID", "")

app = FastAPI(title="EchoRead")
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
        raise HTTPException(500, "ECHOREAD_AGENT_ID is not set. Run scripts/create_agent.py.")
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


def _embed(query: str):
    """One embedding, or None if embeddings are unavailable."""
    try:
        from echoread.brain import GeminiEmbedder
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
        from echoread.brain import GeminiEmbedder
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


@app.get("/api/books")
def list_books():
    """The shelf: the shipped book first, then whatever has been added."""
    return {"books": [_builtin_card()] + books.shelf(playheads)}


@app.post("/api/books")
def add_book(new: NewBook):
    """Start indexing a book from a URL.

    Returns as soon as the transcription job is queued. The browser then polls
    GET /api/books/{id}, and each poll advances the work by one step -- there
    is no worker here, and a request that waited for a whole audiobook would be
    killed by the platform long before it finished.
    """
    if not AAI_KEY:
        raise HTTPException(500, "ASSEMBLYAI_API_KEY is not set on this deployment.")
    url = new.audio_url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "That needs to be a direct https link to an audio file.")
    try:
        rec = books.create(playheads, new.title or _title_from_url(url), url, AAI_KEY)
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
    title = request.headers.get("x-book-title", "") or "Uploaded book"
    raw = await request.body()
    if not raw:
        raise HTTPException(400, "No audio arrived.")
    if len(raw) > 4_400_000:
        raise HTTPException(413, "That file is too big to upload here - paste a link to it instead.")
    try:
        url = books.upload_bytes(raw, AAI_KEY)
        rec = books.create(playheads, title, url, AAI_KEY)
    except urllib.error.HTTPError as e:
        raise HTTPException(400, f"Upload failed: {e.read().decode()[:200]}")
    # The browser plays its own local copy; this URL is AssemblyAI-only.
    out = rec.public()
    out["audio_url"] = ""
    return out


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

"""Bring your own audiobook: transcribe it, index it, serve it like the shipped one.

The built-in book is a SQLite file baked into the repo. A book someone adds at
runtime cannot be: the filesystem on a lambda is read-only, and the request that
builds the index is not the request that later reads it. So a user book lives in
the same Redis the playhead uses, behind a class with the same surface as
`playhead.library.Library` -- `window`, `search`, `duration_hint`, `len` -- so
the tools in main.py never learn which kind of book they are holding.

The pipeline is deliberately resumable, one bounded slice of work per HTTP
request:

    transcribing -> indexing -> ready

Nothing here may block. A serverless function is killed at ten seconds, and
transcribing an audiobook takes minutes, so `advance()` does a little work,
writes down where it got to, and returns. The browser polls; each poll pushes
the job one step further. That also gives the page a real progress number
instead of a spinner that means nothing.
"""
import base64
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from playhead.library import Chunk, WINDOW_AFTER_S, WINDOW_BEFORE_S

AAI_BASE = "https://api.assemblyai.com/v2"

# Gemini caps an embedding batch at 100, and one batch is about a second of
# work -- a safe amount to do inside a single request.
SLICE = 100
# 768-d, matching playhead.brain.EMBED_DIM and the shipped .npy.
DIM = 768
# Stored as float16. Cosine similarity over normalised vectors does not care
# about the last few bits of mantissa, and it halves what crosses the wire.
VEC_DTYPE = "float16"

BOOK_TTL = 30 * 24 * 3600
# An upper bound on a single book, in chunks (~1800 is a ten-hour book).
# 0 means no ceiling. Left as a setting rather than deleted because it is the
# only thing standing between one enormous file and the whole store.
MAX_CHUNKS = int(os.getenv("PLAYHEAD_MAX_CHUNKS", "0") or 0)


# A book has to be plausibly a book: long enough to be worth indexing, small
# enough that one link cannot run up an unbounded transcription bill.
MAX_SOURCE_BYTES = 150 * 1024 * 1024
# Speech, not music. Below this many characters per minute of audio, whatever
# was sent is a song, a field recording, or silence with a cough in it.
MIN_CHARS_PER_MINUTE = 250
AUDIO_EXTS = {"mp3", "m4a", "m4b", "wav", "ogg", "oga", "opus", "flac", "aac",
              "wma", "webm", "mp4", "mov", "mkv", "aiff", "aif", "caf"}


class RejectedURL(ValueError):
    """The link is not something we are willing to hand to a transcriber."""


def _is_public_host(host: str) -> bool:
    """Resolve a hostname and refuse anything that is not a public address.

    We make the request from our own server, so an unchecked link is a
    server-side request forgery: 169.254.169.254 is a cloud metadata endpoint,
    and 127.0.0.1 is whatever else happens to be listening next to us.
    """
    import ipaddress
    import socket
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


class _GuardedRedirects(urllib.request.HTTPRedirectHandler):
    """Re-check the host on every hop.

    Validating only the URL someone typed is not enough: a public host is free
    to redirect us straight at a private one.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        host = urllib.parse.urlparse(newurl).hostname or ""
        if not _is_public_host(host):
            raise RejectedURL("that link redirects somewhere we will not follow")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def check_source(url: str, max_bytes: int = MAX_SOURCE_BYTES) -> dict:
    """Decide whether a link is worth handing to AssemblyAI, before we do.

    Returns what the HEAD told us. A server that refuses HEAD is not treated as
    a failure -- plenty of file hosts do -- but then nothing is known about the
    file and MAX_CHUNKS is the only ceiling left.
    """
    parts = urllib.parse.urlparse(url)
    if parts.scheme not in ("http", "https"):
        raise RejectedURL("that needs to be an http or https link")
    if not parts.hostname:
        raise RejectedURL("that link has no host in it")
    if not _is_public_host(parts.hostname):
        raise RejectedURL("that address is not reachable from the public internet")

    opener = urllib.request.build_opener(_GuardedRedirects)
    req = urllib.request.Request(url, method="HEAD",
                                 headers={"User-Agent": "Playhead/1.0"})
    try:
        with opener.open(req, timeout=10) as r:
            ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            length = int(r.headers.get("Content-Length") or 0)
    except RejectedURL:
        raise
    except Exception as exc:
        # Plenty of file hosts refuse HEAD, so this cannot be fatal. But with
        # nothing known about the file, a clearly non-audio extension is the
        # only signal left -- and it catches the common paste-the-wrong-link
        # mistake. No extension at all (CDNs, stream URLs) still goes through.
        print(f"[books] HEAD failed for {url[:80]}: {exc}")
        ext = re.search(r"\.([a-z0-9]{2,5})$", parts.path, re.I)
        if ext and ext.group(1).lower() not in AUDIO_EXTS:
            raise RejectedURL(f"that link ends in .{ext.group(1).lower()}, "
                              f"which is not an audio file")
        return {"content_type": "", "bytes": 0}

    if max_bytes and length and length > max_bytes:
        raise RejectedURL(
            f"that file is {length / 1e6:.0f} MB, and the ceiling here is "
            f"{max_bytes // 1_000_000} MB. Try a single chapter.")
    # Plenty of hosts serve audio as octet-stream, so only a confidently wrong
    # type is rejected.
    if ctype and not (ctype.startswith("audio/") or ctype.startswith("video/")
                      or ctype in ("application/octet-stream", "binary/octet-stream")):
        raise RejectedURL(f"that link is {ctype}, not an audio file")
    return {"content_type": ctype, "bytes": length}


def slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (title or "book").lower()).strip("-")
    return (s[:40] or "book") + "-" + format(int(time.time() * 1000) % 0xFFFFFF, "x")


# ---------- AssemblyAI ----------

def _aai(path: str, key: str, payload: Optional[dict] = None, timeout: int = 15):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        f"{AAI_BASE}{path}",
        data=data,
        headers={"Authorization": key,
                 **({"Content-Type": "application/json"} if data else {})},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def upload_bytes(raw: bytes, key: str) -> str:
    """Hand raw audio to AssemblyAI and get back a URL only they can read.

    Only reachable for small files: the platform caps a request body at 4.5 MB,
    which is a chapter or a podcast episode, not a novel. Longer books come in
    by URL instead, which has no such ceiling.
    """
    req = urllib.request.Request(
        f"{AAI_BASE}/upload", data=raw,
        headers={"Authorization": key, "Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["upload_url"]


def start_transcription(audio_url: str, key: str) -> str:
    """Queue the job and return immediately. Polling happens in advance()."""
    # speech_models is left at its default (universal-3-5-pro, falling back to
    # universal-2). Pinning it here would silently break when the flagship
    # model is renamed, which has already happened once on this project.
    body = {"audio_url": audio_url, "punctuate": True, "format_text": True}
    return _aai("/transcript", key, body)["id"]


# ---------- the record ----------

@dataclass
class BookRecord:
    id: str
    title: str
    audio_url: str
    transcript_id: str = ""
    status: str = "transcribing"       # transcribing | indexing | ready | failed
    error: str = ""
    n_chunks: int = 0
    embedded: int = 0
    duration: float = 0.0
    builtin: bool = False
    created: float = 0.0

    @property
    def progress(self) -> int:
        """0-100. Transcription is the long pole, so it owns most of the bar."""
        if self.status == "ready":
            return 100
        if self.status == "transcribing":
            return 10
        if self.status == "indexing" and self.n_chunks:
            return 40 + int(55 * self.embedded / self.n_chunks)
        return 0

    def public(self) -> dict:
        return {"id": self.id, "title": self.title, "audio_url": self.audio_url,
                "status": self.status, "error": self.error, "progress": self.progress,
                "chunks": self.n_chunks, "duration": round(self.duration, 1),
                "builtin": self.builtin}


def _key(book_id: str, part: str = "meta") -> str:
    return f"playhead:book:{book_id}:{part}"


def load(store, book_id: str) -> Optional[BookRecord]:
    raw = store.kv_get(_key(book_id))
    if not raw:
        return None
    try:
        return BookRecord(**json.loads(raw))
    except Exception:
        return None


def save(store, rec: BookRecord) -> None:
    store.kv_set(_key(rec.id), json.dumps(rec.__dict__), ttl=BOOK_TTL)


def _shelf_key(client_id: str) -> str:
    return f"playhead:books:{client_id or 'anon'}"


def create(store, title: str, audio_url: str, aai_key: str, client_id: str = "") -> BookRecord:
    rec = BookRecord(id=slug(title), title=title.strip() or "Untitled book",
                     audio_url=audio_url, created=time.time())
    rec.transcript_id = start_transcription(audio_url, aai_key)
    save(store, rec)
    store.kv_push(_shelf_key(client_id), rec.id)
    return rec


def shelf(store, client_id: str = "") -> List[dict]:
    """The books this browser added, newest first.

    Deliberately not a shared shelf. There are no accounts, so a global list
    would put whatever a stranger uploaded on the front page of a live demo --
    and this one is being judged in public. The client id is a random string
    the browser keeps in localStorage; it is an identifier, not a credential.
    """
    out = []
    for book_id in store.kv_list(_shelf_key(client_id), 30):
        rec = load(store, book_id)
        if rec:
            out.append(rec.public())
    return out


# ---------- the pipeline ----------

def advance(store, rec: BookRecord, aai_key: str, embedder) -> BookRecord:
    """Push one book one step. Safe to call repeatedly; safe to call on a
    finished book. Every branch returns well inside a request timeout."""
    if rec.status in ("ready", "failed"):
        return rec
    try:
        if rec.status == "transcribing":
            return _poll_transcript(store, rec, aai_key)
        if rec.status == "indexing":
            return _embed_slice(store, rec, embedder)
    except urllib.error.HTTPError as e:
        return _fail(store, rec, f"{e.code}: {e.read().decode()[:160]}")
    except Exception as exc:
        return _fail(store, rec, str(exc)[:200])
    return rec


def _fail(store, rec: BookRecord, message: str) -> BookRecord:
    rec.status, rec.error = "failed", message
    save(store, rec)
    print(f"[books] {rec.id} failed: {message}")
    return rec


def _poll_transcript(store, rec: BookRecord, aai_key: str) -> BookRecord:
    t = _aai(f"/transcript/{rec.transcript_id}", aai_key)
    status = t.get("status")
    if status == "error":
        err = t.get("error", "transcription failed")
        # archive.org in particular will 503 a file it served happily a minute
        # earlier. That is worth saying out loud, because "failed" invites
        # someone to go hunting for a different link when the same one works
        # on a second try.
        if "download" in err.lower() or "unable to" in err.lower():
            err = ("the host would not hand over the file just then. "
                   "That is usually temporary - try it again.")
        return _fail(store, rec, err)
    if status != "completed":
        return rec

    paras = _aai(f"/transcript/{rec.transcript_id}/paragraphs", aai_key)
    payload = {"paragraphs": paras.get("paragraphs", []),
               "text": t.get("text", ""),
               "audio_duration": t.get("audio_duration", 0)}
    chunks = _chunks_from(payload)
    if not chunks:
        return _fail(store, rec, "there is no speech in that recording - "
                                 "Playhead needs someone reading out loud.")

    # Music, ambience and silence all transcribe to almost nothing spread over a
    # long duration. Indexing that produces a book the agent cannot answer from,
    # which reads as the product being broken rather than the input being wrong.
    minutes = max((payload.get("audio_duration") or 0) / 60.0, 0.5)
    density = len(payload.get("text", "")) / minutes
    if density < MIN_CHARS_PER_MINUTE:
        return _fail(store, rec,
                     f"that recording has very little speech in it "
                     f"({density:.0f} characters a minute). Playhead indexes "
                     f"narration - music or ambience gives it nothing to answer from.")

    if MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
    # Times are read on every single question, so they are kept apart from the
    # text: a window lookup then costs one small fetch instead of pulling the
    # whole book across the wire.
    store.kv_set(_key(rec.id, "times"),
                 json.dumps([[round(c.start_s, 2), round(c.end_s, 2)] for c in chunks]),
                 ttl=BOOK_TTL)
    for i in range(0, len(chunks), SLICE):
        store.kv_set(_key(rec.id, f"text:{i // SLICE}"),
                     json.dumps([c.text for c in chunks[i:i + SLICE]]), ttl=BOOK_TTL)

    rec.n_chunks = len(chunks)
    rec.duration = float(t.get("audio_duration") or (chunks[-1].end_s if chunks else 0))
    rec.status, rec.embedded = "indexing", 0
    save(store, rec)
    print(f"[books] {rec.id} transcribed: {rec.n_chunks} chunks")
    return rec


def _chunks_from(payload: dict) -> List[Chunk]:
    from playhead.library import Library
    return Library.chunks_from_transcript(payload)


def _embed_slice(store, rec: BookRecord, embedder) -> BookRecord:
    """Embed the next batch and write it as its own shard.

    One slice per request is the whole trick: a ten-hour book is eighteen
    Gemini calls, which would blow any request timeout done in one go, but is
    eighteen quick polls when spread out -- and the browser gets a real
    percentage out of it.
    """
    if embedder is None:
        return _fail(store, rec, "embeddings are unavailable on this deployment")
    n = rec.embedded // SLICE
    texts = json.loads(store.kv_get(_key(rec.id, f"text:{n}")) or "[]")
    if not texts:
        rec.status = "ready"
        save(store, rec)
        return rec

    vecs = embedder(texts, "document").astype("float32")
    # Normalised at write time so a query is a plain dot product later.
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    store.kv_set(_key(rec.id, f"vecs:{n}"),
                 base64.b64encode(vecs.astype(VEC_DTYPE).tobytes()).decode("ascii"),
                 ttl=BOOK_TTL)

    rec.embedded = min(rec.embedded + len(texts), rec.n_chunks)
    if rec.embedded >= rec.n_chunks:
        rec.status = "ready"
        print(f"[books] {rec.id} ready: {rec.n_chunks} chunks")
    save(store, rec)
    return rec


# ---------- reading it back ----------

# A narrator announces structure out loud: "Section 8: On the Idea of Time in
# Physics", "Chapter Eleven". That announcement is the only table of contents an
# audiobook has, so it is what we read.
_NUMBER = (r"\d{1,3}|[ivxlcdm]{1,7}|one|two|three|four|five|six|seven|eight|nine|ten|"
           r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
           r"nineteen|twenty|twenty[- ]\w+|thirty|forty|fifty")
HEADING_RE = re.compile(
    # A numbered heading must actually carry its number. Without that, "part of
    # his intelligence" and "book of the law" read as chapter openings, which
    # is how Pride and Prejudice ended up with a chapter called "Part: of his
    # intelligence, though unheard by Lydia".
    r"\b(?:(chapter|section|part|book|volume|act|scene|lecture|appendix)\s+"
    rf"({_NUMBER})\b"
    # These stand alone, because they are never numbered.
    # The trailing \b matters: without it "prefaced his speech with a solemn
    # bow" becomes a chapter called "Preface: d his speech with a solemn bow".
    r"|(preface|introduction|prologue|epilogue|conclusion|foreword|afterword)\b)"
    r"\s*[:.\-—]?\s*(.{0,60})", re.I)


def contents(lib) -> List[dict]:
    """A table of contents, derived from what the reader says out loud.

    An audiobook carries no chapter metadata -- the file is one opaque stream.
    But the narrator announces each heading, so the transcript has them, and a
    heading near the start of a chunk is almost always a real one rather than a
    passing mention ("as I said in chapter three").

    Falls back to even parts when a recording announces nothing, which is
    better than an empty panel: it still lets someone jump around.
    """
    duration = lib.duration_hint()
    if not duration:
        return []
    # Both Library and RedisLibrary answer window(), so one huge window is a
    # portable way to walk every chunk without adding a method to either.
    every = lib.window(duration / 2, before=duration, after=duration)
    if not every:
        return []

    found, seen = [], set()
    for c in every:
        head = re.sub(r"\s+", " ", (c.text or "").strip())[:120]
        # A chapter almost always opens with the previous one ending: "End of
        # Section 7. Section 8: ...". Drop that, or we label the new chapter
        # with the old one's number.
        head = re.sub(r"^(?:end of\s+)[^.]{0,40}\.\s*", "", head, flags=re.I)
        m = HEADING_RE.search(head)
        # Only a heading that opens the chunk counts. Mid-paragraph references
        # to other chapters are common and are not structure.
        if not m or m.start() > 12:
            continue

        kind, number, standalone, rest = m.group(1), m.group(2), m.group(3), (m.group(4) or "")
        # The title runs to the end of its own sentence, not for 60 characters.
        rest = re.split(r"[.!?]", rest)[0].strip(" :;,-—")
        label = f"{kind.title()} {number}" if kind else standalone.title()
        if rest:
            label += ": " + rest
        label = label[:70].strip()

        key = re.sub(r"[^a-z0-9]", "", label.lower())
        if not key or key in seen:
            continue
        seen.add(key)
        found.append({"t": round(c.start_s, 1), "title": label})

    if len(found) >= 2:
        return found[:60]

    # Nothing announced: even parts, labelled with how they open.
    n = min(8, max(2, len(every) // 4))
    out = []
    for i in range(n):
        # A zero-width window at t=0 finds nothing when the first chunk starts
        # a half-second in, which silently dropped "Part 1" off the front.
        hits = lib.window(duration * i / n, before=0.0, after=2.0)
        if not hits:
            continue
        opening = re.sub(r"\s+", " ", (hits[0].text or "")).strip()[:52]
        out.append({"t": round(hits[0].start_s, 1),
                    "title": f"Part {i + 1}", "hint": opening + "…"})
    return out


class RedisLibrary:
    """A book held in Redis, with the same methods main.py calls on Library.

    Reads are lazy and shard-scoped. Answering "what did that mean?" needs the
    times plus one text shard; only a named-topic search pays for the vectors.
    """

    def __init__(self, store, rec: BookRecord):
        self._store = store
        self._rec = rec
        self._times: Optional[List[List[float]]] = None
        self._vectors: Optional[np.ndarray] = None
        self._texts: dict = {}

    # -- lazily loaded pieces --

    def _load_times(self) -> List[List[float]]:
        if self._times is None:
            self._times = json.loads(self._store.kv_get(_key(self._rec.id, "times")) or "[]")
        return self._times

    def _text(self, chunk_id: int) -> str:
        n = chunk_id // SLICE
        if n not in self._texts:
            self._texts[n] = json.loads(
                self._store.kv_get(_key(self._rec.id, f"text:{n}")) or "[]")
        shard = self._texts[n]
        i = chunk_id % SLICE
        return shard[i] if i < len(shard) else ""

    def _load_vectors(self) -> Optional[np.ndarray]:
        """Every shard in one round trip.

        A long book is many shards, and fetching them one at a time is that
        many sequential network calls inside a function the platform kills at
        ten seconds. This is the only path that needs the whole index -- the
        window lookup, which runs on every question, never comes here.
        """
        if self._vectors is None:
            n_shards = (self._rec.n_chunks + SLICE - 1) // SLICE
            keys = [_key(self._rec.id, f"vecs:{n}") for n in range(n_shards)]
            blocks = []
            for raw in self._store.kv_mget(keys):
                if not raw:
                    break
                blocks.append(np.frombuffer(base64.b64decode(raw), dtype=VEC_DTYPE)
                              .reshape(-1, DIM).astype("float32"))
            self._vectors = np.vstack(blocks) if blocks else np.zeros((0, DIM), "float32")
        return self._vectors

    def _chunk(self, i: int) -> Chunk:
        start, end = self._load_times()[i]
        return Chunk(i, start, end, self._text(i))

    # -- the Library surface --

    def window(self, timestamp: float,
               before: float = WINDOW_BEFORE_S,
               after: float = WINDOW_AFTER_S) -> List[Chunk]:
        lo, hi = timestamp - before, timestamp + after
        ids = [i for i, (s, e) in enumerate(self._load_times()) if e >= lo and s <= hi]
        return [self._chunk(i) for i in ids]

    def search(self, query_vec: np.ndarray, k: int = 4,
               before_s: Optional[float] = None) -> List[Chunk]:
        vecs = self._load_vectors()
        if vecs is None or not len(vecs):
            return []
        q = query_vec.astype("float32").ravel()
        q /= np.linalg.norm(q) + 1e-9
        scores = vecs @ q

        if before_s is not None:
            # The spoiler cap, same rule as the shipped book: nothing from
            # further along than the listener has actually reached.
            times = self._load_times()
            mask = np.array([times[i][0] <= before_s for i in range(len(scores))])
            scores = np.where(mask, scores, -np.inf)

        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [self._chunk(int(i)) for i in top]

    def duration_hint(self) -> float:
        if self._rec.duration:
            return self._rec.duration
        times = self._load_times()
        return float(times[-1][1]) if times else 0.0

    def __len__(self) -> int:
        return self._rec.n_chunks

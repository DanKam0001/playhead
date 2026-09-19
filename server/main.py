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
import time
from pathlib import Path
from typing import Dict, Optional

import urllib.error
import urllib.request
import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from echoread.library import Library

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


# ---------- browser -> backend ----------

class Playhead(BaseModel):
    session_id: str
    seconds: float


@app.post("/api/playhead")
def report_playhead(p: Playhead):
    """The browser reports where the audiobook is, roughly once a second.

    AssemblyAI calls the tool server-to-server and has no idea where playback
    is, so the position has to arrive out of band. Cheap, and it means the tool
    stays stateless from the agent's point of view.
    """
    playheads.set(p.session_id, max(0.0, p.seconds))
    return {"ok": True}


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
    near = library.window(t)
    if not near:
        return {"ok": False,
                "message": f"There's no indexed text around {int(t)//60}:{int(t)%60:02d}."}

    parts = [f"The listener is {int(t)//60} minutes {int(t)%60} seconds into the book.",
             "This is what they have just heard:",
             " ".join(c.text for c in near)]

    if call.search:
        far = _search_earlier(call.search, t, exclude={c.id for c in near})
        if far:
            parts += ["", f"Earlier in the book, on '{call.search}':",
                      " ".join(c.text for c in far)]

    return {"ok": True, "playhead_seconds": round(t, 1), "message": "\n".join(parts)}


def _search_earlier(query: str, before_s: float, exclude) -> list:
    """Semantic lookback, capped at the playhead so it cannot spoil the book."""
    try:
        from echoread.brain import GeminiEmbedder
        key = os.getenv("GEMINI_API_KEY", "")
        if not key:
            return []
        vec = GeminiEmbedder(key)([query], "query")[0]
        hits = library.search(vec, k=5, before_s=before_s)
        return [c for c in hits if c.id not in exclude][:2]
    except Exception as exc:
        print(f"[tool] lookback skipped: {exc}")
        return []


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

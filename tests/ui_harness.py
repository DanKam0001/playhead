"""A real browser, a stub backend, and a fake agent socket.

The web client has never had a test. Everything in it is state that changes on
an event -- a book swap, a part handover, a websocket message -- and those are
exactly the bugs that do not show up in a careful read: the playback rate that
silently resets, the position saved under the wrong key, the reply that pauses
a book the listener just asked to resume.

So: serve `web/` for real, serve short real audio files so the <audio> element
behaves like an audio element, replace the websocket with something the test
drives, and click the actual buttons.

Nothing here talks to AssemblyAI, Gemini, Upstash or the network. The browser
comes from the Playwright install; when that is missing the tests skip rather
than fail, because CI has no browser.
"""
from __future__ import annotations

import http.server
import json
import math
import struct
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"


def wav_bytes(seconds: float, rate: int = 8000) -> bytes:
    """A real, decodable WAV of a known length.

    Chromium needs actual media to fire loadedmetadata and to report a
    duration, and duration is what the spine, the marks and every seek are
    computed from. A quiet tone is enough and keeps the files tiny.
    """
    n = int(seconds * rate)
    frames = b"".join(
        struct.pack("<h", int(3000 * math.sin(i * 2 * math.pi * 220 / rate)))
        for i in range(n)
    )
    return (b"RIFF" + struct.pack("<I", 36 + len(frames)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(frames)) + frames)


# Two books. The first is a single file; the second is three files laid end to
# end, which is the shape every real audiobook here has.
PART_SECONDS = [6.0, 5.0, 7.0]
# Longer than the 10 s floor offerResume() applies, so the resume bar
# can actually be tested.
SINGLE_SECONDS = 20.0

SINGLE = {
    "id": "single-book",
    "title": "Single File Book",
    "audio_url": "/audio/single.wav",
    "status": "ready", "error": "", "progress": 100,
    "chunks": 12, "duration": SINGLE_SECONDS,
    "builtin": True, "featured": False,
    "parts": [], "part_count": 1, "parts_done": 1,
}

MULTI = {
    "id": "multi-book",
    "title": "Three Part Book",
    "audio_url": "/audio/part0.wav",
    "status": "ready", "error": "", "progress": 100,
    "chunks": 40, "duration": sum(PART_SECONDS),
    "builtin": False, "featured": True,
    "parts": [
        {"url": "/audio/part0.wav", "offset_s": 0.0, "duration_s": PART_SECONDS[0]},
        {"url": "/audio/part1.wav", "offset_s": PART_SECONDS[0], "duration_s": PART_SECONDS[1]},
        {"url": "/audio/part2.wav", "offset_s": PART_SECONDS[0] + PART_SECONDS[1],
         "duration_s": PART_SECONDS[2]},
    ],
    "part_count": 3, "parts_done": 3,
}

# Still indexing: every part listed, only the first measured. This is exactly
# what the server returns mid-run, and the shape that used to break seeking.
INDEXING = {
    "id": "indexing-book",
    "title": "Still Indexing Book",
    "audio_url": "/audio/part0.wav",
    "status": "transcribing", "error": "", "progress": 5,
    "chunks": 0, "duration": 0.0,
    "builtin": False, "featured": False,
    "parts": [
        {"url": "/audio/part0.wav", "offset_s": 0.0, "duration_s": PART_SECONDS[0]},
        {"url": "/audio/part1.wav", "offset_s": PART_SECONDS[0], "duration_s": 0.0},
        {"url": "/audio/part2.wav", "offset_s": 0.0, "duration_s": 0.0},
    ],
    "part_count": 3, "parts_done": 1,
}

# A long book, only so the clock has to render hours.
LONG = dict(SINGLE, id="long-book", title="Very Long Book",
            duration=54 * 3600 + 16 * 60 + 21, builtin=False, featured=True)


class Handler(http.server.SimpleHTTPRequestHandler):
    books = [SINGLE, MULTI]
    playhead_calls: list = []

    def log_message(self, *_):      # keep the test output readable
        pass

    def _send(self, payload, ctype="application/json"):
        body = json.dumps(payload).encode() if ctype == "application/json" else payload
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            return self._send((WEB / "index.html").read_bytes(), "text/html")
        if path == "/static/app.js" or path == "/app.js":
            return self._send((WEB / "app.js").read_bytes(), "text/javascript")
        if path.startswith("/audio/"):
            name = path.rsplit("/", 1)[1]
            if name.startswith("part"):
                secs = PART_SECONDS[int(name[4])]
            else:
                secs = SINGLE_SECONDS
            return self._send(wav_bytes(secs), "audio/wav")
        if path == "/api/books":
            return self._send({"books": self.books, "suggested": [],
                               "limits": {"upload_mb": 4, "source_mb": 150}})
        if path.startswith("/api/books/") and path.endswith("/contents"):
            return self._send({"contents": [
                {"t": 0, "title": "Chapter one"},
                {"t": PART_SECONDS[0] + 1, "title": "Chapter two"},
            ]})
        if path.startswith("/api/books/"):
            wanted = path.rsplit("/", 1)[1]
            for b in self.books:
                if b["id"] == wanted:
                    return self._send(b)
            return self._send({"detail": "no such book"})
        if path == "/api/session":
            return self._send({"token": "t", "agent_id": "a", "session_id": "s1"})
        return self._send({"detail": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if self.path.split("?")[0] == "/api/playhead":
            try:
                Handler.playhead_calls.append(json.loads(raw))
            except Exception:
                pass
        return self._send({"ok": True})


class Server:
    def __init__(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.httpd.shutdown()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}/"


# Replaces the websocket before any of the page's own script runs, and gives
# the test a way to speak as the agent. Also records what the page sent, which
# is how the handshake shape is checked.
FAKE_WS = """
window.__sent = [];
window.__sockets = 0;
class FakeWS {
  constructor(url) {
    window.__sockets++;
    this.url = url; this.readyState = 1;
    window.__ws = this;
    setTimeout(() => { if (this.onopen) this.onopen(); }, 0);
  }
  send(data) { window.__sent.push(data); }
  close() { this.readyState = 3; if (this.onclose) this.onclose(); }
}
FakeWS.OPEN = 1;
window.WebSocket = FakeWS;
window.__agent = (msg) => {
  if (window.__ws && window.__ws.onmessage) {
    window.__ws.onmessage({ data: JSON.stringify(msg) });
  }
};
// Count heartbeats without waiting a real second for each.
window.__intervals = [];
const realSetInterval = window.setInterval;
window.setInterval = function (fn, ms) {
  const id = realSetInterval(fn, ms);
  window.__intervals.push({ id, ms });
  return id;
};
const realClearInterval = window.clearInterval;
window.clearInterval = function (id) {
  window.__intervals = window.__intervals.filter((i) => i.id !== id);
  return realClearInterval(id);
};
"""

LAUNCH_ARGS = [
    "--autoplay-policy=no-user-gesture-required",
    "--use-fake-device-for-media-stream",
    "--use-fake-ui-for-media-stream",
    "--mute-audio",
]

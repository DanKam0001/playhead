"""A stand-in for Upstash REST, so the Redis path can be tested without one.

Speaks only the commands `server/store.py` sends, in both the URL-path form and
the JSON-body form, and answers with Upstash's `{"result": ...}` envelope. That
is enough to exercise the parts that a dict cannot: command encoding, a 200 KB
shard surviving an HTTP round trip, and MGET returning things in order.
"""
import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

DB: dict = {}
LISTS: dict = {}


def reset() -> None:
    DB.clear()
    LISTS.clear()


def run(cmd):
    op = cmd[0].lower()
    if op == "set":
        DB[cmd[1]] = cmd[2]
        return "OK"
    if op == "get":
        return DB.get(cmd[1])
    if op == "mget":
        return [DB.get(k) for k in cmd[1:]]
    if op == "getdel":
        return DB.pop(cmd[1], None)
    if op == "del":
        return 1 if DB.pop(cmd[1], None) is not None else 0
    if op == "incr":
        DB[cmd[1]] = str(int(DB.get(cmd[1], 0)) + 1)
        return int(DB[cmd[1]])
    if op == "expire":
        return 1
    if op == "lpush":
        LISTS.setdefault(cmd[1], []).insert(0, cmd[2])
        return len(LISTS[cmd[1]])
    if op == "ltrim":
        LISTS[cmd[1]] = LISTS.get(cmd[1], [])[int(cmd[2]):int(cmd[3]) + 1]
        return "OK"
    if op == "lrange":
        lst = LISTS.get(cmd[1], [])
        stop = int(cmd[3])
        return lst[int(cmd[2]):(stop + 1 if stop >= 0 else None)]
    raise ValueError(f"unsupported command {op}")


class _Handler(BaseHTTPRequestHandler):
    def _reply(self, obj):
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            self._reply({"result": run(json.loads(self.rfile.read(n)))})
        except Exception as exc:
            self._reply({"error": str(exc)})

    def do_GET(self):
        parts = [urllib.parse.unquote(p) for p in self.path.lstrip("/").split("/")]
        try:
            self._reply({"result": run(parts)})
        except Exception as exc:
            self._reply({"error": str(exc)})

    def log_message(self, *args):
        pass  # the test output is the interesting thing, not the request log


def serve(port: int = 8199) -> HTTPServer:
    server = HTTPServer(("127.0.0.1", port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server

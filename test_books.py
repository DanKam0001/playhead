"""Tests for the bring-your-own-book path, with no Upstash and no API keys.

The shipped book is a SQLite file; a book someone adds lives in Redis. That
second path could not be covered by anything here before -- MemoryStore does not
exercise the command encoding, and a 200 KB index shard is exactly the kind of
payload that works in a dict and fails over HTTP. So this stands up a stand-in
for Upstash REST, speaking the same {"result": ...} envelope, and runs the real
RedisStore against it.

    python test_books.py

Must pass with no network, no key, and no audio hardware.
"""


import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------- a stand-in for Upstash REST ----------

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

DB = {}
LISTS = {}


def run(cmd):
    op = cmd[0].lower()
    if op == "set":
        DB[cmd[1]] = cmd[2]
        return "OK"
    if op == "get":
        return DB.get(cmd[1])
    if op == "getdel":
        return DB.pop(cmd[1], None)
    if op == "del":
        return 1 if DB.pop(cmd[1], None) is not None else 0
    if op == "lpush":
        LISTS.setdefault(cmd[1], []).insert(0, cmd[2])
        return len(LISTS[cmd[1]])
    if op == "ltrim":
        lst = LISTS.get(cmd[1], [])
        LISTS[cmd[1]] = lst[int(cmd[2]):int(cmd[3]) + 1]
        return "OK"
    if op == "lrange":
        lst = LISTS.get(cmd[1], [])
        stop = int(cmd[3])
        return lst[int(cmd[2]):(stop + 1 if stop >= 0 else None)]
    raise ValueError("unsupported command " + op)


class H(BaseHTTPRequestHandler):
    def _reply(self, obj):
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        cmd = json.loads(self.rfile.read(n))
        try:
            self._reply({"result": run(cmd)})
        except Exception as exc:
            self._reply({"error": str(exc)})

    def do_GET(self):
        parts = [urllib.parse.unquote(p) for p in self.path.lstrip("/").split("/")]
        try:
            self._reply({"result": run(parts)})
        except Exception as exc:
            self._reply({"error": str(exc)})

    def log_message(self, *a):
        pass


def serve(port):
    s = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


if False:
    serve(8199)
    print("fake upstash on 8199")
    threading.Event().wait()


# ---------- the tests ----------


serve(8199)
os.environ["UPSTASH_REDIS_REST_URL"] = "http://127.0.0.1:8199"
os.environ["UPSTASH_REDIS_REST_TOKEN"] = "fake"

import numpy as np
from server import books
from server.store import build_store

store = build_store()
assert store.kind == "redis", store.kind
print("store kind:", store.kind)

# --- a big value through the JSON-body path ---
big = "x" * 210_000
store.kv_set("echoread:test:big", big, ttl=60)
back = store.kv_get("echoread:test:big")
assert back == big, f"round trip lost data: {len(back or '')} vs {len(big)}"
print("PASS 200KB value round trip")

# --- the shelf list ---
store.kv_push("echoread:test:list", "a")
store.kv_push("echoread:test:list", "b")
assert store.kv_list("echoread:test:list", 10) == ["b", "a"], store.kv_list("echoread:test:list", 10)
print("PASS lpush/lrange ordering")

# --- a whole book, built the way the pipeline builds one ---
N = 250                      # spans three shards at SLICE=100
rng = np.random.default_rng(7)
texts = [f"passage number {i} about topic {i % 7}" for i in range(N)]
times = [[i * 20.0, i * 20.0 + 20.0] for i in range(N)]

rec = books.BookRecord(id="testbook", title="Test", audio_url="http://x/y.mp3",
                       status="indexing", n_chunks=N)
books.save(store, rec)
store.kv_set(books._key("testbook", "times"), __import__("json").dumps(times))
for i in range(0, N, books.SLICE):
    store.kv_set(books._key("testbook", f"text:{i // books.SLICE}"),
                 __import__("json").dumps(texts[i:i + books.SLICE]))

vecs = rng.standard_normal((N, books.DIM)).astype("float32")
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)


class FakeEmbedder:
    """Hands back the slice of vectors the real one would have produced."""
    def __init__(self):
        self.calls = 0

    def __call__(self, texts_, task="document"):
        lo = self.calls * books.SLICE
        self.calls += 1
        return vecs[lo:lo + len(texts_)]


emb = FakeEmbedder()
rec = books.load(store, "testbook")
steps = 0
while rec.status == "indexing" and steps < 10:
    rec = books.advance(store, rec, "no-key", emb)
    steps += 1
    print(f"  step {steps}: {rec.status} {rec.progress}% embedded={rec.embedded}")
assert rec.status == "ready", (rec.status, rec.error)
assert rec.embedded == N, rec.embedded
print(f"PASS pipeline reached ready in {steps} slices ({emb.calls} embed calls)")

# --- read it back ---
lib = books.RedisLibrary(store, rec)
assert len(lib) == N

w = lib.window(1000.0)
assert w, "empty window"
assert all(c.end_s >= 1000 - 90 and c.start_s <= 1000 + 15 for c in w), [(c.start_s, c.end_s) for c in w]
assert all(c.text for c in w), "a chunk came back with no text"
print(f"PASS window at 16:40 -> {len(w)} chunks, first={w[0].cite()} {w[0].text[:34]!r}")

# search must find the exact vector it was given back
probe = 137
hits = lib.search(vecs[probe], k=3)
assert hits[0].id == probe, (hits[0].id, probe)
print(f"PASS search recovered chunk {probe} as top hit (float16 round trip intact)")

# the spoiler cap
capped = lib.search(vecs[probe], k=3, before_s=500.0)
assert all(c.start_s <= 500.0 for c in capped), [c.start_s for c in capped]
assert probe not in [c.id for c in capped], "spoiler cap let a later chunk through"
print(f"PASS spoiler cap held: {len(capped)} hits, all before 08:20")

# the shelf is scoped to a client, not shared
store.kv_push(books._shelf_key("client-a"), "testbook")
assert [b["title"] for b in books.shelf(store, "client-a")] == ["Test"]
assert books.shelf(store, "client-b") == [], "one client can see another's books"
print("PASS shelf is per client, not global")

# --- the URL guard ---
for bad, why in [
    ("ftp://example.com/a.mp3", "scheme"),
    ("http://127.0.0.1/a.mp3", "loopback"),
    ("http://169.254.169.254/latest/meta-data/", "cloud metadata"),
    ("http://localhost:8199/a.mp3", "localhost"),
    ("http://10.0.0.5/a.mp3", "private range"),
]:
    try:
        books.check_source(bad)
    except books.RejectedURL:
        pass
    else:
        raise AssertionError(f"check_source allowed {why}: {bad}")
print("PASS url guard refused scheme, loopback, metadata, localhost, private range")

# a real public audio link still passes, when there is a network to check it on
try:
    info = books.check_source(
        "https://archive.org/download/art_of_war_librivox/art_of_war_01-02_sun_tzu_64kb.mp3")
    print(f"PASS url guard allowed a real LibriVox link ({info['bytes'] / 1e6:.1f} MB)")
except books.RejectedURL as exc:
    raise AssertionError(f"guard rejected a legitimate link: {exc}")
except Exception as exc:
    print(f"SKIP live URL check (no network): {exc}")

print()
print("ALL REDIS BOOK TESTS PASSED")

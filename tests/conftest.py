"""Shared fixtures.

Everything here runs with no API key, no network and no audio hardware. That
is a hard rule: a test suite that needs credentials is a test suite nobody
runs, and CI has none.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests import fake_upstash  # noqa: E402

PORT = 8199


@pytest.fixture(scope="session", autouse=True)
def upstash_stand_in():
    """Point RedisStore at a local stand-in for the whole session."""
    server = fake_upstash.serve(PORT)
    os.environ["UPSTASH_REDIS_REST_URL"] = f"http://127.0.0.1:{PORT}"
    os.environ["UPSTASH_REDIS_REST_TOKEN"] = "not-a-real-token"
    yield server
    server.shutdown()


@pytest.fixture
def store(upstash_stand_in):
    """A fresh RedisStore, talking to an empty stand-in."""
    from server.store import build_store
    fake_upstash.reset()
    s = build_store()
    assert s.kind == "redis", "the stand-in was not picked up"
    return s


@pytest.fixture
def fake_embedder():
    """Hands back a slice of a fixed matrix, the way Gemini would.

    Deterministic, so a test can assert that a specific chunk comes back as the
    top hit, which is what proves the float16 round trip did not lose it.
    """
    class Embedder:
        def __init__(self, vectors):
            self.vectors = vectors
            self.calls = 0

        def __call__(self, texts, task="document"):
            from server.books import SLICE
            lo = self.calls * SLICE
            self.calls += 1
            return self.vectors[lo:lo + len(texts)]

    def build(n, dim=768, seed=7):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal((n, dim)).astype("float32")
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return Embedder(v)

    return build


@pytest.fixture
def indexed_book(store, fake_embedder):
    """Build a book through the real pipeline and hand back its library.

    Returns (RedisLibrary, record, vectors) so a test can search for a vector
    it knows the answer to.
    """
    import json
    from server import books

    def build(n_chunks=250, seconds_each=20.0):
        emb = fake_embedder(n_chunks)
        texts = [f"passage number {i} about topic {i % 7}" for i in range(n_chunks)]
        times = [[i * seconds_each, (i + 1) * seconds_each] for i in range(n_chunks)]

        rec = books.BookRecord(id=f"book{n_chunks}", title="Test Book",
                               audio_url="http://example.com/b.mp3",
                               status="indexing", n_chunks=n_chunks,
                               duration=n_chunks * seconds_each)
        books.save(store, rec)
        store.kv_set(books._key(rec.id, "times"), json.dumps(times))
        for i in range(0, n_chunks, books.SLICE):
            store.kv_set(books._key(rec.id, f"text:{i // books.SLICE}"),
                         json.dumps(texts[i:i + books.SLICE]))

        rec = books.load(store, rec.id)
        guard = 0
        while rec.status == "indexing" and guard < 200:
            rec = books.advance(store, rec, "no-key", emb)
            guard += 1
        assert rec.status == "ready", (rec.status, rec.error)
        return books.RedisLibrary(store, rec), rec, emb.vectors

    return build

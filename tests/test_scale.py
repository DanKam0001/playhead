"""How large a book the retrieval layer can carry, and what it costs.

The size ceilings are settings now, and this deployment has them off, so
"how big can it get" stopped being rhetorical. These measure the layer that
would break first -- not AssemblyAI, which transcribed a 47-minute file in
about fifteen seconds, but the index that has to live in Redis and be read
back inside a function the platform kills at ten seconds.

Run the slow one explicitly:

    pytest tests/test_scale.py -m slow -s
"""
import json
import time

import numpy as np
import pytest

from server import books

# ~1800 chunks is a ten-hour book at the median chunk length measured on real
# audio (50 s / 630 characters). 2000 is a shade past that.
TEN_HOUR_BOOK = 2000


def build_at_scale(store, n_chunks, dim=books.DIM):
    """Write an index of n_chunks straight to the store, skipping embedding."""
    each = 20.0
    times = [[i * each, (i + 1) * each] for i in range(n_chunks)]
    rec = books.BookRecord(id=f"scale{n_chunks}", title="Scale", audio_url="",
                           status="ready", n_chunks=n_chunks,
                           embedded=n_chunks, duration=n_chunks * each)
    books.save(store, rec)
    store.kv_set(books._key(rec.id, "times"), json.dumps(times))

    rng = np.random.default_rng(11)
    vectors = rng.standard_normal((n_chunks, dim)).astype("float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    import base64
    for i in range(0, n_chunks, books.SLICE):
        sl = slice(i, i + books.SLICE)
        store.kv_set(books._key(rec.id, f"text:{i // books.SLICE}"),
                     json.dumps([f"passage {j} concerning topic {j % 40}"
                                 for j in range(sl.start, min(sl.stop, n_chunks))]))
        store.kv_set(books._key(rec.id, f"vecs:{i // books.SLICE}"),
                     base64.b64encode(
                         vectors[sl].astype(books.VEC_DTYPE).tobytes()).decode("ascii"))
    return books.RedisLibrary(store, rec), vectors


def test_window_does_not_read_the_whole_book(store):
    """The hot path -- every question goes through it.

    A window lookup must stay cheap no matter how long the book is, which is
    why times are stored apart from text: it reads one small key plus the one
    text shard it lands in, never the whole index.
    """
    lib, _ = build_at_scale(store, TEN_HOUR_BOOK)

    t0 = time.perf_counter()
    hits = lib.window(20_000.0)
    elapsed = time.perf_counter() - t0

    shards = (TEN_HOUR_BOOK + books.SLICE - 1) // books.SLICE
    assert hits, "empty window on a large book"
    # One or two shards -- two only when the window straddles a boundary --
    # never all twenty. The count must not grow with the length of the book.
    assert len(lib._texts) <= 2, f"pulled {len(lib._texts)} of {shards} text shards"
    assert lib._vectors is None, "a window lookup loaded the vectors"
    print(f"\n  window over {TEN_HOUR_BOOK} chunks: {elapsed * 1000:.0f} ms, "
          f"{len(hits)} chunks, {len(lib._texts)} of {shards} shards read")


def test_vectors_load_in_one_round_trip(store):
    """Fetched one shard at a time, a ten-hour book is twenty sequential
    network calls inside a ten-second budget. MGET makes it one."""
    lib, vectors = build_at_scale(store, TEN_HOUR_BOOK)

    calls = {"n": 0}
    real_mget = store.kv_mget
    store.kv_mget = lambda keys: (calls.__setitem__("n", calls["n"] + 1), real_mget(keys))[1]

    t0 = time.perf_counter()
    loaded = lib._load_vectors()
    elapsed = time.perf_counter() - t0

    assert calls["n"] == 1, f"{calls['n']} round trips, expected 1"
    assert loaded.shape == (TEN_HOUR_BOOK, books.DIM)
    print(f"\n  loaded {TEN_HOUR_BOOK}x{books.DIM} vectors in {elapsed * 1000:.0f} ms, "
          f"{loaded.nbytes / 1e6:.1f} MB in memory, "
          f"{(TEN_HOUR_BOOK + books.SLICE - 1) // books.SLICE} shards, 1 round trip")


def test_search_stays_fast_at_ten_hours(store):
    """Brute-force cosine was chosen over a vector DB on the claim that it is
    negligible next to the model call. This is that claim, checked."""
    lib, vectors = build_at_scale(store, TEN_HOUR_BOOK)
    lib._load_vectors()

    t0 = time.perf_counter()
    for probe in (0, 500, 1200, 1999):
        hits = lib.search(vectors[probe], k=4)
        assert hits[0].id == probe, (probe, hits[0].id)
    elapsed = (time.perf_counter() - t0) / 4

    print(f"\n  search over {TEN_HOUR_BOOK} chunks: {elapsed * 1000:.1f} ms per query")
    # Generous: the point is that it is nowhere near the ~2 s model call.
    assert elapsed < 0.15, f"{elapsed * 1000:.0f} ms is too slow to be free"


def test_the_spoiler_cap_holds_at_scale(store):
    lib, vectors = build_at_scale(store, TEN_HOUR_BOOK)
    capped = lib.search(vectors[1900], k=5, before_s=2000.0)
    assert capped and all(c.start_s <= 2000.0 for c in capped)


def test_shard_size_stays_under_the_request_cap(store):
    """Upstash rejects a request body over 1 MB on the free tier. One shard is
    100 chunks of float16, which is the reason SLICE is 100 and not 1000."""
    build_at_scale(store, 300)
    raw = store.kv_get(books._key("scale300", "vecs:0"))
    size = len(raw.encode())
    print(f"\n  one shard ({books.SLICE} chunks x {books.DIM}d float16): "
          f"{size / 1024:.0f} KB base64")
    assert size < 1_000_000, f"shard is {size} bytes, past Upstash's cap"


@pytest.mark.slow
def test_find_the_ceiling(store):
    """Grow a book until something gives, and report where.

    Not run by default: it is a measurement, not an assertion about
    correctness. `pytest tests/test_scale.py -m slow -s` prints the table.
    """
    print("\n  chunks     hours   vec MB   load ms   search ms")
    for n in (500, 1000, 2000, 4000, 8000):
        lib, vectors = build_at_scale(store, n)
        t0 = time.perf_counter()
        lib._load_vectors()
        load_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        lib.search(vectors[n // 2], k=4)
        search_ms = (time.perf_counter() - t0) * 1000
        print(f"  {n:6d}   {n * 20 / 3600:6.1f}   {n * books.DIM * 2 / 1e6:6.1f}"
              f"   {load_ms:7.0f}   {search_ms:9.1f}")

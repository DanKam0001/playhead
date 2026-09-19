"""The bring-your-own-book pipeline, end to end against the stand-in.

Covers the part that has no equivalent in the shipped SQLite book: sharded
storage, float16 vectors, resumable indexing, and a shelf scoped to a browser.
"""
import numpy as np
import pytest

from server import books


def test_indexing_completes_in_bounded_slices(store, fake_embedder, indexed_book):
    """The pipeline must never do a whole book in one request.

    A function is killed at ten seconds; 250 chunks is three Gemini batches,
    and each one has to be its own call.
    """
    lib, rec, _ = indexed_book(n_chunks=250)
    assert rec.status == "ready"
    assert rec.embedded == 250
    assert len(lib) == 250


def test_progress_is_monotonic_and_ends_at_100(store, fake_embedder):
    """The bar is driven by real work, so it must never go backwards."""
    import json
    emb = fake_embedder(250)
    rec = books.BookRecord(id="prog", title="P", audio_url="http://x/y.mp3",
                           status="indexing", n_chunks=250, duration=5000)
    books.save(store, rec)
    store.kv_set(books._key("prog", "times"),
                 json.dumps([[i * 20.0, i * 20.0 + 20] for i in range(250)]))
    for i in range(0, 250, books.SLICE):
        store.kv_set(books._key("prog", f"text:{i // books.SLICE}"),
                     json.dumps([f"t{j}" for j in range(i, min(i + books.SLICE, 250))]))

    seen = []
    rec = books.load(store, "prog")
    while rec.status == "indexing" and len(seen) < 20:
        rec = books.advance(store, rec, "no-key", emb)
        seen.append(rec.progress)
    assert seen == sorted(seen), seen
    assert seen[-1] == 100


def test_window_returns_what_was_just_heard(indexed_book):
    lib, _, _ = indexed_book(n_chunks=250)
    hits = lib.window(1000.0)
    assert hits, "empty window"
    # -90s / +15s around the playhead, weighted backwards on purpose.
    assert all(c.end_s >= 910 and c.start_s <= 1015 for c in hits)
    assert all(c.text for c in hits), "a chunk came back with no text"


def test_float16_still_recovers_the_exact_chunk(indexed_book):
    """Vectors are halved to fit the store. If that cost recall, the top hit
    for a vector taken straight from the index would stop being its own chunk.
    """
    lib, _, vectors = indexed_book(n_chunks=250)
    for probe in (0, 137, 249):
        hits = lib.search(vectors[probe], k=3)
        assert hits[0].id == probe, (probe, hits[0].id)


def test_spoiler_cap_excludes_everything_ahead(indexed_book):
    """Answering a chapter 3 question with chapter 12 material is a real
    failure for a book, not a rounding error."""
    lib, _, vectors = indexed_book(n_chunks=250)
    probe = 200                      # sits at 4000s, far past the cap
    capped = lib.search(vectors[probe], k=5, before_s=500.0)
    assert capped, "the cap removed everything"
    assert all(c.start_s <= 500.0 for c in capped)
    assert probe not in [c.id for c in capped]


def test_search_with_no_vectors_is_empty_not_an_error(store):
    rec = books.BookRecord(id="empty", title="E", audio_url="", n_chunks=0)
    lib = books.RedisLibrary(store, rec)
    assert lib.search(np.zeros(books.DIM, dtype="float32")) == []


def test_shelf_is_scoped_to_one_browser(store, indexed_book):
    """A global shelf would put whatever a stranger added on the front page of
    a live demo. The client id is an identifier, not a credential."""
    indexed_book(n_chunks=100)
    store.kv_push(books._shelf_key("client-a"), "book100")
    assert [b["title"] for b in books.shelf(store, "client-a")] == ["Test Book"]
    assert books.shelf(store, "client-b") == []


def test_a_finished_book_is_not_advanced_again(store, indexed_book):
    lib, rec, _ = indexed_book(n_chunks=100)
    again = books.advance(store, rec, "no-key", None)
    assert again.status == "ready"
    assert again.embedded == rec.embedded


def test_indexing_without_an_embedder_fails_loudly(store):
    """No Gemini key should produce a book marked failed with a reason, not a
    book marked ready with no vectors behind it."""
    import json
    rec = books.BookRecord(id="noemb", title="N", audio_url="", status="indexing",
                           n_chunks=10, duration=200)
    books.save(store, rec)
    store.kv_set(books._key("noemb", "times"),
                 json.dumps([[i * 20.0, i * 20.0 + 20] for i in range(10)]))
    store.kv_set(books._key("noemb", "text:0"),
                 json.dumps([f"t{i}" for i in range(10)]))
    out = books.advance(store, books.load(store, "noemb"), "no-key", None)
    assert out.status == "failed"
    assert "embed" in out.error.lower()

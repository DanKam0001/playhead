"""Books made of many files.

A real audiobook is published one file per chapter, so a book is a list of
parts laid end to end on a single timeline. The arithmetic that shifts each
part's chunks onto that timeline is the part worth testing: get it wrong and a
question at 4:12:30 quietly returns the wrong chapter, which nothing else would
catch.
"""
import json

import pytest

from playhead.library import Chunk
from server import books


def make(store, book_id, part_lengths, chunks_each=5, chunk_len=20.0):
    """Absorb N parts the way _poll_transcript does, and hand back the record."""
    rec = books.BookRecord(
        id=book_id, title="Many Parts", audio_url="http://x/1.mp3",
        parts=[{"url": f"http://x/{i}.mp3", "transcript_id": f"t{i}",
                "offset_s": 0.0, "duration_s": 0.0, "done": False}
               for i in range(len(part_lengths))])
    books.save(store, rec)

    offset = 0.0
    for i, length in enumerate(part_lengths):
        rec.parts[i]["offset_s"] = offset
        chunks = [Chunk(j, j * chunk_len, (j + 1) * chunk_len,
                        f"part {i} chunk {j}") for j in range(chunks_each)]
        books._absorb_part(store, rec, chunks, offset, [])
        rec.parts[i]["duration_s"] = length
        rec.part_index += 1
        offset += length
    rec.duration = offset

    tail = json.loads(store.kv_get(books._key(book_id, "tail")) or "[]")
    if tail:
        store.kv_set(books._key(book_id, f"text:{rec.chunk_cursor // books.SLICE}"),
                     json.dumps(tail))
    rec.n_chunks = rec.chunk_cursor + len(tail)
    rec.status = "ready"
    books.save(store, rec)
    return rec


def test_each_part_is_shifted_onto_the_book_timeline(store):
    rec = make(store, "three", [100.0, 200.0, 150.0])
    times = json.loads(store.kv_get(books._key("three", "times")))
    assert len(times) == 15

    # Part 0 starts at 0, part 1 at 100, part 2 at 300.
    assert times[0][0] == 0.0
    assert times[5][0] == 100.0
    assert times[10][0] == 300.0
    assert rec.duration == 450.0


def test_times_never_go_backwards_across_a_boundary(store):
    make(store, "mono", [100.0, 200.0, 150.0])
    times = json.loads(store.kv_get(books._key("mono", "times")))
    starts = [t[0] for t in times]
    assert starts == sorted(starts), starts


def test_a_window_late_in_the_book_finds_the_right_part(store):
    """The failure this guards against is silent: a question at 4:12:30 comes
    back with chapter one because the offsets were never applied."""
    rec = make(store, "late", [600.0, 600.0, 600.0], chunks_each=30, chunk_len=20.0)
    lib = books.RedisLibrary(store, rec)

    # Well inside the third part: nothing from earlier chapters should appear.
    deep = lib.window(1500.0)
    assert deep and all("part 2" in c.text for c in deep), [c.text for c in deep]

    # Just after the boundary, the -90s window reaches back into part 2 -- and
    # should, since that is the run-up to what was just heard. What matters is
    # that it is the *adjacent* part, not chapter one.
    edge = lib.window(1250.0)
    assert edge
    assert {t.split(" chunk")[0] for t in (c.text for c in edge)} == {"part 1", "part 2"}


def test_shards_line_up_when_parts_do_not_divide_evenly(store):
    """Parts almost never end on a shard boundary. Text is buffered in a tail
    until a shard fills, so an off-by-one here scrambles every later chunk."""
    rec = make(store, "ragged", [100.0] * 7, chunks_each=30)   # 210 chunks, SLICE 100
    lib = books.RedisLibrary(store, rec)
    assert rec.n_chunks == 210
    # Walk every chunk and check its text matches the part it claims to be in.
    for i in range(rec.n_chunks):
        expected_part, expected_j = divmod(i, 30)
        assert lib._text(i) == f"part {expected_part} chunk {expected_j}", i


def test_chapters_are_shifted_too(store):
    rec = books.BookRecord(id="chap", title="C", audio_url="",
                           parts=[{"url": "u", "offset_s": 0.0}])
    books.save(store, rec)
    chunks = [Chunk(0, 0.0, 20.0, "text")]
    books._absorb_part(store, rec, chunks, 0.0,
                       [{"start": 5000, "headline": "First"}])
    books._absorb_part(store, rec, chunks, 600.0,
                       [{"start": 5000, "headline": "Second"}])
    got = json.loads(store.kv_get(books._key("chap", "chapters")))
    assert [c["t"] for c in got] == [5.0, 605.0]


def test_a_single_file_book_still_works(store):
    rec = make(store, "solo", [300.0])
    lib = books.RedisLibrary(store, rec)
    assert rec.duration == 300.0
    assert lib.window(50.0)


def test_create_accepts_one_url_or_many(store, monkeypatch):
    monkeypatch.setattr(books, "start_transcription", lambda url, key: "fake")
    one = books.create(store, "One", "http://x/a.mp3", "k", book_id="b1")
    assert len(one.parts) == 1 and one.audio_url == "http://x/a.mp3"

    many = books.create(store, "Many", ["http://x/a.mp3", "http://x/b.mp3"], "k",
                        book_id="b2")
    assert len(many.parts) == 2
    # audio_url stays the first file, so anything that only understands one
    # file still has something to play.
    assert many.audio_url == "http://x/a.mp3"


def test_create_refuses_an_empty_list(store):
    with pytest.raises(ValueError):
        books.create(store, "None", [], "k", book_id="b3")

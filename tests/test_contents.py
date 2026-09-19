"""Chapter detection.

An audiobook file carries no structure at all -- it is one opaque stream, which
is why skipping around one is guesswork. The narrator announces the headings
out loud, so the transcript is the only table of contents that exists.
"""
from playhead.library import Chunk

from server import books


class FakeLibrary:
    """Just enough of the Library surface for contents() to walk a book."""

    def __init__(self, chunks):
        self._chunks = chunks

    def duration_hint(self):
        return self._chunks[-1].end_s if self._chunks else 0.0

    def window(self, t, before=0.0, after=0.0):
        lo, hi = t - before, t + after
        return [c for c in self._chunks if c.end_s >= lo and c.start_s <= hi]


def lib_from(texts, each=30.0):
    return FakeLibrary([Chunk(i, i * each, (i + 1) * each, t)
                        for i, t in enumerate(texts)])


def test_reads_headings_the_narrator_announces():
    lib = lib_from([
        "Section 7: On the Relativity of Simultaneity. Consider a train.",
        "the train moves along the embankment at constant velocity",
        "End of Section 7. Section 8: On the Idea of Time in Physics. Lightning struck.",
        "two events which are simultaneous with respect to the embankment",
    ])
    out = books.contents(lib)
    assert [c["title"] for c in out] == [
        "Section 7: On the Relativity of Simultaneity",
        "Section 8: On the Idea of Time in Physics",
    ]
    assert out[1]["t"] == 60.0


def test_the_end_of_the_previous_chapter_does_not_steal_the_label():
    """A chapter almost always opens with the last one ending. Matching the
    first heading found would number every chapter one behind."""
    lib = lib_from([
        "Chapter One: Beginnings. It was a start.",
        "End of Chapter One. Chapter Two: Middles. It continued.",
    ])
    assert books.contents(lib)[1]["title"] == "Chapter Two: Middles"


def test_a_passing_mention_is_not_structure():
    """'As I said in chapter three' is a reference, not a heading."""
    lib = lib_from([
        "Chapter One: Beginnings. It was a start.",
        "As I explained back in chapter three, the argument depends on this.",
        "Chapter Two: Middles. It continued.",
    ])
    assert len(books.contents(lib)) == 2


def test_titles_stop_at_the_end_of_their_own_sentence():
    lib = lib_from([
        "Chapter One: A Short Title. Then a great deal of body text follows here "
        "which must not end up inside the chapter title itself.",
        "Chapter Two: Another. More body text.",
    ])
    assert books.contents(lib)[0]["title"] == "Chapter One: A Short Title"


def test_falls_back_to_even_parts_when_nothing_is_announced():
    """A recording that announces nothing should still be navigable. An empty
    panel is worse than approximate one."""
    lib = lib_from(["just some prose with no structure at all"] * 20)
    out = books.contents(lib)
    assert len(out) >= 2
    assert out[0]["title"] == "Part 1"
    assert "hint" in out[0]
    assert [c["t"] for c in out] == sorted(c["t"] for c in out)


def test_an_empty_book_has_no_contents():
    assert books.contents(FakeLibrary([])) == []


def test_duplicate_headings_are_not_listed_twice():
    lib = lib_from([
        "Introduction. This is a LibriVox recording.",
        "Introduction. This is a LibriVox recording.",
        "Chapter One: Real. Body.",
    ])
    titles = [c["title"] for c in books.contents(lib)]
    assert len(titles) == len(set(titles))


def test_a_numbered_word_in_prose_is_not_a_heading():
    """Real bug: Pride and Prejudice produced a chapter called
    "Part: of his intelligence, though unheard by Lydia". A heading that takes
    a number has to actually carry one."""
    lib = lib_from([
        "part of his intelligence, though unheard by Lydia, was caught by her sister",
        "book of the law was opened before them and they read from it",
        "Chapter 19: The next day opened a new scene at Longbourn.",
        "and so the day continued much as it had begun for everyone",
    ])
    titles = [c["title"] for c in books.contents(lib)]
    # Neither prose line becomes a chapter. Only one real heading survives,
    # which is below the threshold, so even parts take over -- correct, and
    # still better than one wrong chapter title.
    assert not any("intelligence" in t or "law" in t for t in titles), titles
    assert all(t.startswith("Part ") for t in titles), titles


def test_unnumbered_headings_that_never_take_a_number_still_count():
    lib = lib_from([
        "Preface. The author wishes to thank his correspondents.",
        "Chapter 1: Beginnings. It was a start.",
    ])
    assert [c["title"] for c in books.contents(lib)] == [
        "Preface: The author wishes to thank his correspondents",
        "Chapter 1: Beginnings",
    ]


def test_a_word_containing_a_heading_is_not_a_heading():
    """Real bug: "prefaced his speech with a solemn bow" became a chapter
    called "Preface: d his speech with a solemn bow"."""
    lib = lib_from([
        "Chapter 1: Real. Body text.",
        "prefaced his speech with a solemn bow, and though she could not hear",
        "Chapter 2: Also real. More body.",
    ])
    titles = [c["title"] for c in books.contents(lib)]
    assert titles == ["Chapter 1: Real", "Chapter 2: Also real"], titles


def test_the_fallback_starts_at_the_beginning():
    """Real bug: Part 1 was dropped because a zero-width window at t=0 misses
    a first chunk that starts a half-second in."""
    chunks = [Chunk(i, i * 30.0 + 0.5, (i + 1) * 30.0, "prose with no structure")
              for i in range(20)]
    out = books.contents(FakeLibrary(chunks))
    assert out[0]["title"] == "Part 1", [c["title"] for c in out]
    assert out[0]["t"] < 5.0

"""What the buttons actually do, in a real browser.

Every test here failed against the client as it stood before 2026-09-21. They
are written as "press this, then assert the thing a listener would notice",
not as assertions about internals, because the bugs were all cases where the
internals were consistent and the product was not.

Skipped when Playwright or its browser is missing, which is the case in CI.
"""
from __future__ import annotations

import json

import pytest

from tests.ui_harness import (FAKE_WS, LAUNCH_ARGS, INDEXING, LONG, MULTI,
                              PART_SECONDS, SINGLE, Handler, Server)

sync_playwright = pytest.importorskip(
    "playwright.sync_api", reason="playwright not installed").sync_playwright


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(args=LAUNCH_ARGS)
        except Exception as exc:                       # no browser downloaded
            pytest.skip(f"no chromium available: {exc}")
        yield b
        b.close()


@pytest.fixture
def page(browser):
    """A fresh page with a fresh profile, a stub backend and a fake socket."""
    Handler.books = [SINGLE, MULTI]
    Handler.playhead_calls = []
    with Server() as server:
        ctx = browser.new_context(permissions=["microphone"])
        pg = ctx.new_page()
        pg.add_init_script(FAKE_WS)
        pg.goto(server.url)
        pg.wait_for_function("window.__ready === undefined")
        # The shelf arrives over fetch; wait for a book to be selected.
        pg.wait_for_function("document.getElementById('booktitle').textContent.length > 0")
        pg.wait_for_function("document.getElementById('book').readyState >= 1")
        yield pg
        ctx.close()


def open_library(page):
    """The shelf lives in a panel that starts collapsed."""
    if page.get_attribute("#library", "hidden") is not None:
        page.click("#librarybtn")
    page.wait_for_selector("#library:not([hidden])")


def shelf_row(page, title):
    """The title button of a shelf row, not the little remove cross beside it."""
    return page.locator("#shelf li").filter(
        has=page.locator("span.t", has_text=title)).locator("button").first


def pick_book(page, title):
    open_library(page)
    shelf_row(page, title).click()
    page.wait_for_function(
        "t => document.getElementById('booktitle').textContent === t", arg=title)


def start_session(page):
    """Press the start button and bring the fake agent up to `session.ready`."""
    page.click("#start")
    page.wait_for_function("window.__ws !== undefined")
    page.evaluate("window.__agent({type:'session.ready', session_id:'s1'})")
    page.wait_for_function("document.getElementById('start').textContent.includes('Stop')")


# --------------------------------------------------------------------------
# the one that was reported
# --------------------------------------------------------------------------

def test_speed_survives_a_book_change(page):
    """3x on one book must still be 3x on the next.

    load() resets playbackRate to defaultPlaybackRate, which nothing set, so
    the audio dropped to 1x while the slider still read 3.00x.
    """
    page.fill("#speed", "3")
    page.dispatch_event("#speed", "input")
    assert page.evaluate("document.getElementById('book').playbackRate") == 3

    pick_book(page, "Three Part Book")
    page.wait_for_function("document.getElementById('book').readyState >= 1")

    assert page.evaluate("document.getElementById('book').playbackRate") == 3
    assert page.inner_text("#speedval").startswith("3.00")


def test_speed_survives_a_part_handover(page):
    """A 54-hour book changes file many times; the rate must not reset."""
    pick_book(page, "Three Part Book")
    page.fill("#speed", "2")
    page.dispatch_event("#speed", "input")

    # Jump into part three: a real handover, not just a currentTime change.
    page.evaluate(f"window.__seek = {PART_SECONDS[0] + PART_SECONDS[1] + 1}")
    page.evaluate("document.querySelector('#toclist button:nth-child(1)')")
    page.evaluate("seek(window.__seek)")
    page.wait_for_function("partIndex === 2")
    page.wait_for_function("document.getElementById('book').readyState >= 1")

    assert page.evaluate("document.getElementById('book').playbackRate") == 2


def test_speed_persists_across_a_reload(page):
    page.fill("#speed", "1.5")
    page.dispatch_event("#speed", "input")
    page.reload()
    page.wait_for_function("document.getElementById('book').readyState >= 1")
    assert page.evaluate("document.getElementById('book').playbackRate") == 1.5


# --------------------------------------------------------------------------
# the clock
# --------------------------------------------------------------------------

def test_clock_shows_hours_on_a_long_book(page):
    """54 hours used to render as 3256:21."""
    assert page.evaluate("clockText(54*3600 + 16*60 + 21)") == "54:16:21"
    assert page.evaluate("clockText(9*60 + 12)") == "09:12"      # short stays short
    assert page.evaluate("clockText(0)") == "00:00"
    assert page.evaluate("clockText(-5)") == "00:00"
    assert page.evaluate("clockText(NaN)") == "00:00"


def test_shelf_duration_reads_in_hours(page):
    Handler.books = [SINGLE, LONG]
    page.reload()
    open_library(page)
    row = shelf_row(page, "Very Long Book").inner_text()
    assert "54:16:21" in row


# --------------------------------------------------------------------------
# books that are not ready
# --------------------------------------------------------------------------

def test_indexing_book_is_never_auto_selected(page):
    """Opening on an unplayable book reads as the whole page being broken."""
    Handler.books = [INDEXING, SINGLE]
    page.reload()
    page.wait_for_function("document.getElementById('booktitle').textContent.length > 0")
    assert page.inner_text("#booktitle") == "Single File Book"


def test_indexing_book_parts_are_not_treated_as_a_timeline(page):
    """Zeroed offsets are a wrong timeline, not a short one.

    With the zeros trusted, every seek past the first second resolved to the
    last part -- touch the spine anywhere and you landed in the final chapter.
    """
    assert page.evaluate("usableParts(%s).length" % json.dumps(INDEXING)) == 0
    assert page.evaluate("usableParts(%s).length" % json.dumps(MULTI)) == 3


def test_clicking_an_indexing_book_explains_rather_than_selecting(page):
    Handler.books = [SINGLE, INDEXING]
    page.reload()
    page.wait_for_function("document.getElementById('booktitle').textContent.length > 0")
    open_library(page)
    shelf_row(page, "Still Indexing Book").click()
    assert page.inner_text("#booktitle") == "Single File Book"
    assert "indexing" in page.inner_text("#addstatus").lower()


# --------------------------------------------------------------------------
# resume position
# --------------------------------------------------------------------------

def test_switching_books_does_not_move_the_other_books_position(page):
    """posKey() follows currentBook, which changes before the audio does."""
    page.evaluate("seek(7); rememberPosition();")
    page.wait_for_function("Math.abs(document.getElementById('book').currentTime - 7) < 1")
    page.evaluate("rememberPosition()")

    pick_book(page, "Three Part Book")
    page.wait_for_function("document.getElementById('book').readyState >= 1")
    page.wait_for_timeout(300)

    stored = page.evaluate("JSON.parse(localStorage.getItem('playhead:pos:multi-book') || 'null')")
    assert stored in (None, 0), f"the single book's position leaked in as {stored}"


def test_resume_clicked_immediately_still_seeks(page):
    """currentTime set before metadata is silently ignored."""
    page.evaluate("localStorage.setItem('playhead:pos:single-book', '12')")
    page.reload()
    page.wait_for_selector("#resume:not([hidden])")
    assert page.inner_text("#resumeat") == "00:12"
    # Deliberately as early as possible: before the fix, a click here set
    # currentTime on an element with no metadata and was silently dropped.
    page.click("#resumebtn")
    page.wait_for_function(
        "Math.abs(document.getElementById('book').currentTime - 12) < 1.5")


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------

def test_handshake_nests_the_agent_id(page):
    """Sent flat, the socket still says ready and no agent ever loads."""
    start_session(page)
    sent = [json.loads(s) for s in page.evaluate("window.__sent")]
    update = next(s for s in sent if s["type"] == "session.update")
    assert update["session"]["agent_id"] == "a", "agent_id must be nested under session"


def test_reconnecting_does_not_stack_heartbeats(page):
    """Every reconnect used to add another report per second, forever."""
    start_session(page)
    page.wait_for_function("window.__intervals.length === 1")

    page.evaluate("window.__ws.close()")
    page.wait_for_function("document.getElementById('start').disabled === false")
    start_session(page)

    assert page.evaluate("window.__intervals.length") == 1


def test_stop_listening_releases_the_microphone(page):
    start_session(page)
    assert page.evaluate("live") is True
    page.click("#start")
    page.wait_for_function("live === false")
    assert page.evaluate("micStream === null")
    assert page.evaluate("window.__intervals.length") == 0
    assert "Enable" in page.inner_text("#start")


def test_stopping_does_not_start_a_paused_book_playing(page):
    start_session(page)
    page.evaluate("document.getElementById('book').pause()")
    page.click("#start")
    page.wait_for_function("live === false")
    assert page.evaluate("document.getElementById('book').paused") is True
    assert page.evaluate("document.getElementById('book').volume") == 1


# --------------------------------------------------------------------------
# commands and the reply that follows them
# --------------------------------------------------------------------------

def test_carry_on_is_not_undone_by_the_agents_acknowledgement(page):
    """The book came back for a second, then stopped again."""
    start_session(page)
    page.evaluate("document.getElementById('book').play()")
    # Ask something, so the book is paused the way it would be.
    page.evaluate("window.__agent({type:'transcript.user', text:'what did that mean?'})")
    page.evaluate("window.__agent({type:'reply.started'})")
    page.wait_for_function("document.getElementById('book').paused === true")
    page.evaluate("window.__agent({type:'reply.done', status:'completed'})")

    # Now say "carry on", and let the agent answer it the way it really does.
    page.evaluate("window.__agent({type:'transcript.user', text:'okay, carry on'})")
    page.wait_for_function("document.getElementById('book').paused === false")
    page.evaluate("window.__agent({type:'reply.started'})")
    page.wait_for_timeout(150)

    assert page.evaluate("document.getElementById('book').paused") is False, \
        "the acknowledgement re-paused a book the listener asked to resume"

    page.evaluate("window.__agent({type:'reply.done', status:'completed'})")
    assert page.evaluate("suppressReply") is False


def test_a_command_is_not_filed_as_a_note(page):
    start_session(page)
    page.evaluate("window.__agent({type:'transcript.user', text:'carry on'})")
    page.evaluate("window.__agent({type:'reply.done', status:'completed'})")
    assert page.evaluate("notes.length") == 0


def test_a_real_question_containing_continue_is_not_a_command(page):
    start_session(page)
    page.evaluate(
        "window.__agent({type:'transcript.user',"
        " text:'why does he continue with the second argument?'})")
    assert page.evaluate("suppressReply") is False
    assert page.evaluate("openTurn !== null")


# --------------------------------------------------------------------------
# seeking and the rewind flag
# --------------------------------------------------------------------------

def test_a_stale_jump_does_not_cancel_the_next_rewind(page):
    """justSeeked survived from one interaction to the next."""
    start_session(page)
    page.evaluate("applySeek(4)")                 # the agent moved us
    assert page.evaluate("justSeeked") is True
    page.evaluate("window.__agent({type:'input.speech.started'})")
    assert page.evaluate("justSeeked") is False, "a new question clears the flag"


def test_a_jump_during_an_answer_is_not_rewound(page):
    start_session(page)
    page.evaluate("document.getElementById('book').play()")
    page.evaluate("window.__agent({type:'input.speech.started'})")
    page.evaluate("window.__agent({type:'transcript.user', text:'take me to the part about x'})")
    page.evaluate("window.__agent({type:'reply.started'})")
    page.evaluate("applySeek(5)")
    page.evaluate("window.__agent({type:'reply.done', status:'completed'})")
    page.wait_for_timeout(120)
    at = page.evaluate("pos()")
    assert at > 4.0, f"the deliberate jump was undone by the rewind (landed at {at})"


def test_seek_crosses_into_the_right_part(page):
    pick_book(page, "Three Part Book")
    target = PART_SECONDS[0] + PART_SECONDS[1] + 2      # inside part three
    page.evaluate(f"seek({target})")
    page.wait_for_function("partIndex === 2")
    page.wait_for_function("document.getElementById('book').readyState >= 1")
    assert abs(page.evaluate("pos()") - target) < 1.0


def test_the_spine_reports_book_time_not_file_time(page):
    pick_book(page, "Three Part Book")
    page.evaluate(f"seek({PART_SECONDS[0] + 1})")
    page.wait_for_function("partIndex === 1")
    page.wait_for_function("document.getElementById('book').readyState >= 1")
    # The element is one second into file two; the book is seven seconds in.
    assert page.evaluate("document.getElementById('book').currentTime") < 3
    assert abs(page.evaluate("pos()") - (PART_SECONDS[0] + 1)) < 1.0
    assert page.inner_text("#clock") == "00:07"


# --------------------------------------------------------------------------
# notes and the shelf
# --------------------------------------------------------------------------

def test_notes_are_per_book(page):
    page.evaluate("notes = [{t: 3, q: 'first book question', a: 'answer'}];"
                  "storageSet(notesKey(), notes); renderNotes();")
    pick_book(page, "Three Part Book")
    assert page.evaluate("notes.length") == 0
    pick_book(page, "Single File Book")
    assert page.evaluate("notes.length") == 1


def test_clearing_notes_also_clears_the_conversation(page):
    page.evaluate("notes = [{t: 1, q: 'q', a: 'a'}]; storageSet(notesKey(), notes);"
                  "renderNotes(); turn('you','something');")
    page.on("dialog", lambda d: d.accept())
    page.click("#clearnotes")
    page.wait_for_function("notes.length === 0")
    assert page.query_selector("#transcript .steps") is not None


def test_the_playhead_report_carries_the_selected_book(page):
    start_session(page)
    pick_book(page, "Three Part Book")
    page.wait_for_function("window.__intervals.length >= 1")
    page.evaluate("reportPlayhead()")
    page.wait_for_timeout(250)
    assert Handler.playhead_calls, "no playhead was reported"
    assert Handler.playhead_calls[-1]["book"] == "multi-book"


# --------------------------------------------------------------------------
# what counts as a command
# --------------------------------------------------------------------------

@pytest.mark.parametrize("said", [
    "okay, carry on",            # the natural phrasing, and the one that failed
    "ok",
    "alright, keep going",
    "got it, thanks",
    "yeah okay carry on",
    "carry on",
    "continue",
    "never mind",
    "that's it",
    "Thanks!",
    "all right, go on",
    "resume",
])
def test_these_are_resume_commands(page, said):
    assert page.evaluate("s => COMMANDS[0].match.test(s)", said), \
        f"{said!r} should hand the book back, not be answered"


@pytest.mark.parametrize("said", [
    "why does he continue with the second argument?",
    "okay so what did that mean",
    "carry on about the train and the embankment",
    "right, but what about simultaneity?",
    "play the part about induction",
    "continue explaining that",
    "what did that mean",
    "go on about sense-data",
])
def test_these_are_questions_not_commands(page, said):
    assert not page.evaluate("s => COMMANDS[0].match.test(s)", said), \
        f"{said!r} is a question and must reach the agent"


# --------------------------------------------------------------------------
# the transport
# --------------------------------------------------------------------------

def test_play_button_reflects_actual_state(page):
    """The icon comes from the element's own events, not from what we clicked.

    Waiting on `paused` is the wrong signal and made this flaky: `paused`
    flips synchronously inside play(), while the `play` event that repaints
    the icon fires a tick later. Wait for the label, which is the thing under
    test.
    """
    page.click("#play")
    page.wait_for_function(
        "document.getElementById('play').getAttribute('aria-label').includes('Pause')")
    assert page.evaluate("document.getElementById('book').paused") is False
    page.click("#play")
    page.wait_for_function(
        "document.getElementById('play').getAttribute('aria-label').includes('Play')")
    assert page.evaluate("document.getElementById('book').paused") is True


def test_the_spine_scrubs_to_where_you_click(page):
    box = page.locator("#spine").bounding_box()
    # Three-quarters of the way down the vertical axis.
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] * 0.75)
    page.mouse.down()
    page.mouse.up()
    page.wait_for_function("pos() > 10")
    assert abs(page.evaluate("pos()") - 15) < 3      # 75% of a 20 s book


def test_a_cancelled_pointer_does_not_leave_the_spine_dragging(page):
    box = page.locator("#spine").bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + 5)
    page.mouse.down()
    page.dispatch_event("#spine", "pointercancel")
    page.mouse.up()
    assert page.evaluate("dragging") is False


def test_chapters_jump_and_mark_where_you_are(page):
    pick_book(page, "Three Part Book")
    page.wait_for_selector("#toclist button")
    page.locator("#toclist button").nth(1).click()
    page.wait_for_function("partIndex === 1")
    # partIndex flips as soon as the file is chosen; the seek inside it lands
    # only once that file reports metadata. Wait for the position, not the
    # part -- otherwise pos() is still the part's bare offset.
    page.wait_for_function(
        "t => Math.abs(pos() - t) < 0.6", arg=PART_SECONDS[0] + 1, timeout=15000)
    page.wait_for_function(
        "document.querySelectorAll('#toclist button')[1].getAttribute('aria-current') === 'true'")


def test_marks_on_the_spine_play_from_where_the_question_was_asked(page):
    page.evaluate("notes = [{t: 14, q: 'what did that mean?', a: 'an answer'}];"
                  "storageSet(notesKey(), notes); renderNotes();")
    page.wait_for_selector("#spinemarks .mark")
    page.click("#spinemarks .mark")
    page.wait_for_function("Math.abs(pos() - 14) < 1.5")


# --------------------------------------------------------------------------
# the idle hang-up
# --------------------------------------------------------------------------

def test_an_idle_session_releases_itself(page):
    """A session bills per minute while it is open, talking or not."""
    start_session(page)
    assert page.evaluate("live") is True
    # Fire the timer now rather than waiting five real minutes.
    page.evaluate("clearTimeout(idleTimer); idleTimer = setTimeout(() => {"
                  " if (live) { stopSession();"
                  " setStatus('still listening to the book - tap to ask again','idle'); } }, 50)")
    page.wait_for_function("live === false")
    assert page.evaluate("micStream === null")
    assert page.evaluate("window.__intervals.length") == 0


def test_speech_keeps_an_idle_session_alive(page):
    start_session(page)
    page.evaluate("window.__idleBefore = idleTimer")
    page.evaluate("window.__agent({type:'input.speech.started'})")
    assert page.evaluate("idleTimer !== window.__idleBefore"),         "a question should reset the idle clock"


def test_the_idle_timer_does_not_outlive_the_session(page):
    """A timer that fires after a manual stop would stomp on the next session."""
    start_session(page)
    page.click("#start")
    page.wait_for_function("live === false")
    assert page.evaluate("idleTimer === null")

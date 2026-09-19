"""The state machine that ties ear, book and (later) brain together."""
import enum
import re
import threading
import time
from typing import Callable, List, Optional, Protocol

from .brain import is_hold_command, is_resume_command
from .ears import Ears
from .mic import Microphone
from .player import AudiobookPlayer

# How long a locally-detected speech onset gets to be confirmed by the server
# (a SpeechStarted event or any transcript) before we call it road noise.
CONFIRM_WINDOW_S = 1.2
# Rewind on resume so the user re-hears the run-up to what they asked about.
RESUME_REWIND_S = 3.0
# A mid-question pause ("Wait, hold on... what did that mean?") arrives as two
# separate end_of_turn events. Wait this long after one before deciding the
# person has actually finished, and stitch anything that arrives meanwhile.
TURN_STITCH_S = 1.1
# ...but a turn that already looks like a finished question does not need the
# full wait. Measured: the stitch is the largest single delay we control, more
# than the LLM call. Fragments ("Wait, hold on.") still get the long window.
TURN_STITCH_COMPLETE_S = 0.45
# After a hold phrase ("hold on..."), how long to stay paused and listening
# before giving up and resuming the book.
LISTEN_TIMEOUT_S = 20.0


def looks_complete(text: str) -> bool:
    """Does this read as a whole question rather than a run-up to one?

    Cheap and deliberately conservative: being wrong costs a chopped question,
    which is much worse than waiting an extra 0.65s.
    """
    t = text.strip()
    if not t:
        return False
    if t.endswith(("?",)):
        return True
    # Trailing conjunctions/fillers mean more is coming.
    if re.search(r"\b(and|but|so|because|like|um+|uh+|hold on|wait|hang on)[\s.,]*$",
                 t, re.I):
        return False
    return len(t.split()) >= 6 and t.endswith((".", "!"))


class State(enum.Enum):
    PLAYING = "playing"        # book running, nobody talking
    DUCKED = "ducked"          # local VAD fired, waiting for confirmation
    LISTENING = "listening"    # confirmed: book paused, user is asking
    THINKING = "thinking"      # question sent to the brain
    SPEAKING = "speaking"      # agent replying through TTS


class Brain(Protocol):
    """Implemented in step 2 (Gemini + timestamp-scoped RAG)."""

    def answer(self, question: str, book_timestamp: float) -> str: ...


class Voice(Protocol):
    """Implemented in step 3 (ElevenLabs). Must block until playback finishes."""

    def speak(self, text: str) -> None: ...


class Session:
    def __init__(
        self,
        player: AudiobookPlayer,
        ears: Ears,
        mic: Microphone,
        brain: Optional[Brain] = None,
        voice: Optional[Voice] = None,
        on_state_change: Optional[Callable[[State], None]] = None,
    ):
        self.player = player
        self.ears = ears
        self.mic = mic
        self.brain = brain
        self.voice = voice
        self._on_state_change = on_state_change

        self._state = State.PLAYING
        self._lock = threading.RLock()
        # Book position at the moment of interruption. This is the RAG key --
        # captured at barge-in, not at question-end, because by the time the
        # user finishes talking the "last sentence" they mean is already behind us.
        self.interrupt_timestamp: float = 0.0
        self._ducked_at: float = 0.0
        # Buffered fragments of the question still being stitched together.
        self._pending: List[str] = []
        self._stitch_timer: Optional[threading.Timer] = None
        self._listen_timer: Optional[threading.Timer] = None
        self._last_partial = ""

    # ---------- state ----------

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def _set_state(self, state: State) -> None:
        with self._lock:
            if self._state is state:
                return
            self._state = state
        print(f"[session] -> {state.value}")
        if self._on_state_change:
            self._on_state_change(state)

    # ---------- barge-in path ----------

    def on_local_speech_start(self) -> None:
        """Local VAD (~50 ms). Duck optimistically; we might be wrong."""
        with self._lock:
            if self._state is not State.PLAYING:
                return
            self.interrupt_timestamp = self.player.duck()
            self._ducked_at = time.monotonic()
            self._set_state(State.DUCKED)
        print(f"[session] possible barge-in at {self.interrupt_timestamp:.1f}s in book")

    def on_local_speech_end(self) -> None:
        """Speech stopped before anything confirmed it -- treat as noise."""
        with self._lock:
            if self._state is not State.DUCKED:
                return
            if time.monotonic() - self._ducked_at < CONFIRM_WINDOW_S:
                return  # still inside the window; give the server a chance
            self.player.resume()
            self._set_state(State.PLAYING)
        print("[session] false alarm, unducking")

    def _confirm_barge_in(self) -> None:
        """Server agrees there is speech: commit to a full pause."""
        with self._lock:
            if self._state not in (State.DUCKED, State.PLAYING):
                return
            if self._state is State.PLAYING:
                # Server beat our local VAD (possible in a quiet cabin).
                self.interrupt_timestamp = self.player.position_seconds
            self.player.pause()
            self._set_state(State.LISTENING)

    def on_server_speech_started(self, _timestamp_ms: int) -> None:
        self._confirm_barge_in()

    def on_partial_turn(self, text: str) -> None:
        # Any real words are also confirmation, in case SpeechStarted is missed.
        self._confirm_barge_in()
        # Repaint in place only when the text actually changed. The API resends
        # an unchanged partial repeatedly during a long turn, and reprinting it
        # every time floods the console -- which is the demo's visual.
        if text == self._last_partial:
            return
        pad = max(0, len(self._last_partial) - len(text))
        self._last_partial = text
        print(f"\r[user...] {text}{' ' * pad}", end="", flush=True)

    # ---------- question path ----------

    def on_final_turn(self, text: str) -> None:
        """One end_of_turn. NOT necessarily the whole question -- see TURN_STITCH_S."""
        self._last_partial = ""
        print(f"\n[user] {text}")
        with self._lock:
            if self._state is not State.LISTENING:
                return
            self._pending.append(text.strip())
            if self._stitch_timer is not None:
                self._stitch_timer.cancel()
            delay = (TURN_STITCH_COMPLETE_S
                     if looks_complete(" ".join(self._pending))
                     else TURN_STITCH_S)
            self._stitch_timer = threading.Timer(delay, self._flush_question)
            self._stitch_timer.daemon = True
            self._stitch_timer.start()

    def _flush_question(self) -> None:
        """Fired once the person has actually stopped talking."""
        with self._lock:
            if self._state is not State.LISTENING or not self._pending:
                return
            question = " ".join(x for x in self._pending if x).strip()
            self._pending = []
            self._stitch_timer = None

        if is_hold_command(question):
            # "Hold on." is a placeholder, not a question. Stay paused and keep
            # listening. Answering it interrupts the person mid-thought and
            # their actual question arrives while we are already talking.
            print(f"[session] '{question}' -> holding, still listening")
            self._arm_listening_timeout()
            return

        if is_resume_command(question):
            print(f"[session] '{question}' -> resuming, not a question")
            self.resume_book()
            return

        threading.Thread(target=self._handle_question, args=(question,), daemon=True).start()

    def _arm_listening_timeout(self) -> None:
        """Don't sit paused forever if the held-for question never arrives."""
        with self._lock:
            if self._listen_timer is not None:
                self._listen_timer.cancel()
            self._listen_timer = threading.Timer(LISTEN_TIMEOUT_S, self._listen_timed_out)
            self._listen_timer.daemon = True
            self._listen_timer.start()

    def _listen_timed_out(self) -> None:
        with self._lock:
            if self._state is not State.LISTENING:
                return
        print("[session] nothing followed the hold -- resuming")
        self.resume_book()

    def _handle_question(self, question: str) -> None:
        # Speak sentence-by-sentence only when both halves support it; otherwise
        # generate the whole reply first. Decided up front so the speaking block
        # below cannot reference an unset name.
        streamed = (
            self.brain is not None
            and self.voice is not None
            and hasattr(self.brain, "answer_stream")
            and hasattr(self.voice, "speak_stream")
        )
        try:
            self._set_state(State.THINKING)
            if self.brain is None:
                reply = (f"(no brain wired yet - you asked '{question}' "
                         f"at {self.interrupt_timestamp:.1f}s)")
            elif streamed:
                reply = None  # produced below, while speaking
            else:
                reply = self.brain.answer(question, self.interrupt_timestamp)
            if reply is not None:
                print(f"[agent] {reply}")

            self._set_state(State.SPEAKING)
            if self.voice is not None:
                # Mute barge-in detection or the agent hears its own reply
                # through the speakers and interrupts itself.
                self.mic.vad_enabled = False
                try:
                    if streamed:
                        reply = self.voice.speak_stream(
                            self.brain.answer_stream(question, self.interrupt_timestamp)
                        )
                        print(f"[agent] {reply}")
                    else:
                        self.voice.speak(reply)
                finally:
                    self.mic.reset_vad()
                    self.mic.vad_enabled = True

            # Tell the STT model what we just said, so short follow-ups
            # ("yeah, that one") are recognised against the right context.
            # After speaking, because in the streaming path the full reply does
            # not exist until then.
            try:
                if self.ears is not None and reply:
                    self.ears.set_agent_context(reply)
            except Exception as exc:
                print(f"[session] agent_context update skipped: {exc}")
        except Exception as exc:  # never leave the book paused forever
            print(f"[session] question failed: {exc}")
        finally:
            self.resume_book()

    def resume_book(self) -> None:
        with self._lock:
            self._pending = []
            if self._listen_timer is not None:
                self._listen_timer.cancel()
                self._listen_timer = None
            if self._stitch_timer is not None:
                self._stitch_timer.cancel()
                self._stitch_timer = None
        self.player.rewind(RESUME_REWIND_S)
        self.player.resume()
        self._set_state(State.PLAYING)

    # ---------- run ----------

    def warm(self) -> None:
        """Pay every cold start while the book is still playing.

        Measured: the first Gemini call in a process costs ~3s more than the
        rest. Without this the first question on camera is the slow one.
        """
        for component in (self.brain, self.voice):
            if component is not None and hasattr(component, "warm"):
                component.warm()

    def run(self) -> None:
        """Blocks until the mic stream ends (Ctrl-C)."""
        # Two-stage calibration, and the order matters in both directions.
        #
        # Stage 1, before any sound: the true room floor.
        self.mic.start()
        self.ears.connect()
        self.player.start()
        # Stage 2, with the book audible: raises the floor above whatever leaks
        # from the speakers back into the mic. Skip this and an open desk mic
        # hears the narrator and barge-ins fire continuously. Do *only* this and
        # a quiet moment in the book sets the floor too low. Hence max of both.
        time.sleep(0.4)  # let playback reach steady state
        print("[session] measuring playback leakage -- stay quiet another moment...")
        self.mic.calibrate(1.2, keep_max=True)
        try:
            self.ears.stream(self.mic.frames())
        finally:
            self.ears.close()
            self.mic.close()
            self.player.close()

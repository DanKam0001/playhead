"""AssemblyAI Universal-Streaming wrapper (wss://streaming.assemblyai.com/v3/ws).

Verified against assemblyai==1.3.0. Note the v3 streaming API is a different
surface from the old v2 `aai.RealtimeTranscriber` you will find in most blog
posts -- v3 is turn-based (Turn events with end_of_turn), not
partial/final-transcript based.
"""
from typing import Callable, Iterable, Optional, Sequence

from assemblyai.streaming.v3 import (
    BeginEvent,
    Encoding,
    SpeechStartedEvent,
    StreamingClient,
    StreamingClientOptions,
    StreamingError,
    StreamingEvents,
    StreamingMode,
    StreamingParameters,
    StreamingSessionParameters,
    SpeechModel,
    TerminationEvent,
    TurnEvent,
)

from .config import SAMPLE_RATE


class Ears:
    """Owns the websocket and translates AssemblyAI events into our callbacks."""

    def __init__(
        self,
        api_key: str,
        on_speech_started: Optional[Callable[[int], None]] = None,
        on_partial_turn: Optional[Callable[[str], None]] = None,
        on_final_turn: Optional[Callable[[str], None]] = None,
        keyterms: Optional[Sequence[str]] = None,
        mode: StreamingMode = StreamingMode.balanced,
    ):
        self._mode = mode
        self._client = StreamingClient(
            StreamingClientOptions(api_key=api_key, api_host="streaming.assemblyai.com")
        )
        self._on_speech_started = on_speech_started
        self._on_partial_turn = on_partial_turn
        self._on_final_turn = on_final_turn
        # Domain vocabulary from the book. A technical audiobook is full of terms
        # a general model will mangle ("eigenvector", "Kubernetes", author names).
        # Feeding them here measurably improves recognition of the user's question.
        # Streaming caps this at 100 terms (pre-recorded allows 1000).
        self._keyterms = list(keyterms or [])[:100]
        self._connected = False

        self._client.on(StreamingEvents.Begin, self._handle_begin)
        self._client.on(StreamingEvents.Turn, self._handle_turn)
        self._client.on(StreamingEvents.SpeechStarted, self._handle_speech_started)
        self._client.on(StreamingEvents.Termination, self._handle_termination)
        self._client.on(StreamingEvents.Error, self._handle_error)

    # ---------- event handlers (called on the SDK's socket thread) ----------

    def _handle_begin(self, _client, event: BeginEvent):
        print(f"[ears] session {event.id} open (expires {event.expires_at})")
        self._connected = True

    def _handle_speech_started(self, _client, event: SpeechStartedEvent):
        # Server-side confirmation of speech. Slower than our local VAD, but
        # authoritative -- use it to confirm a barge-in the local VAD guessed at.
        if self._on_speech_started:
            self._on_speech_started(event.timestamp)

    def _handle_turn(self, _client, event: TurnEvent):
        if not event.transcript:
            return
        if event.end_of_turn:
            # NOTE: a mid-sentence pause splits one spoken question into several
            # end_of_turn events -- "Wait, hold on." arrives as a complete turn
            # before "what did that mean?" does. Verified against the live API in
            # probe_streaming.py. Session debounces these; do not answer on the
            # first one. (turn_is_formatted is always True on U3.5 Pro, so it is
            # not a useful gate.)
            if self._on_final_turn:
                self._on_final_turn(event.transcript)
        elif self._on_partial_turn:
            self._on_partial_turn(event.transcript)

    def _handle_termination(self, _client, event: TerminationEvent):
        print(
            f"[ears] session closed after {event.audio_duration_seconds:.1f}s audio"
        )
        self._connected = False

    def _handle_error(self, _client, error: StreamingError):
        print(f"[ears] ERROR: {error}")

    # ---------- transport ----------

    def connect(self) -> None:
        self._client.connect(
            StreamingParameters(
                sample_rate=SAMPLE_RATE,
                encoding=Encoding.pcm_s16le,
                # Optional (server defaults to this) but pin it so a server-side
                # default change cannot silently alter behaviour mid-hackathon.
                speech_model=SpeechModel.universal_3_5_pro,
                # NOT min_latency, deliberately: barge-in speed comes from the
                # local VAD in mic.py, which beats any network round trip. That
                # frees the server to optimise for turn-detection quality, which
                # is what actually stops questions being chopped in half.
                mode=self._mode,
                # Server-side noise suppression. The mic is in a car and is also
                # hearing the audiobook through the speakers.
                voice_focus="near-field",
                # Ride out natural mid-question pauses before calling end-of-turn.
                min_turn_silence=560,
                max_turn_silence=2000,
                keyterms_prompt=self._keyterms or None,
            )
        )
        # Omitted deliberately, verified against the live API:
        #   format_turns                        -- U3.5 Pro always formats
        #   end_of_turn_confidence_threshold    -- confidence is binary (0.0/1.0)
        #   min_end_of_turn_silence_when_confident -- older-model parameter

    def stream(self, frames: Iterable[bytes]) -> None:
        """Blocks, pumping mic frames into the socket until the iterator ends."""
        self._client.stream(frames)

    def force_endpoint(self) -> None:
        """Cut the current turn short -- e.g. the user pressed a 'done' button."""
        self._client.force_endpoint()

    def update_keyterms(self, keyterms: Sequence[str]) -> None:
        """Swap vocabulary mid-session as the book moves into a new chapter."""
        self._keyterms = list(keyterms)[:100]
        self._client.set_params(StreamingSessionParameters(keyterms_prompt=self._keyterms))

    def set_agent_context(self, last_reply: str) -> None:
        """Tell the model what the agent just said out loud.

        Biases the next user turn -- short follow-ups ("yeah, that one", "no,
        the other thing") are much better recognised with this set.
        """
        self._client.set_params(
            StreamingSessionParameters(agent_context=last_reply[:1500])
        )

    def close(self) -> None:
        if self._connected:
            self._client.disconnect(terminate=True)

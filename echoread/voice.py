"""ElevenLabs TTS. Implements the Voice protocol in session.py.

speak() must BLOCK until playback finishes -- session.py mutes barge-in
detection for its duration, and returning early would let the agent hear the
tail of its own reply and interrupt itself.
"""
from typing import Iterator, Optional

import numpy as np
import sounddevice as sd
from elevenlabs.client import ElevenLabs

# Flash is the low-latency model (~75ms TTFB). In a car, a snappy adequate
# voice beats a beautiful slow one.
TTS_MODEL = "eleven_flash_v2_5"
TTS_SAMPLE_RATE = 22050
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"  # George - warm storyteller, suits a book


class ElevenLabsVoice:
    def __init__(
        self,
        api_key: str,
        voice_id: str = "",
        model: str = TTS_MODEL,
        sample_rate: int = TTS_SAMPLE_RATE,
    ):
        self._client = ElevenLabs(api_key=api_key)
        self._voice_id = voice_id or DEFAULT_VOICE
        self._model = model
        self._sample_rate = sample_rate
        self._format = f"pcm_{sample_rate}"
        self._cancelled = False

    def _synthesize(self, text: str, previous_text: str = "") -> Iterator[bytes]:
        kwargs = {}
        if previous_text:
            # Gives the model the run-up, so a sentence synthesized on its own
            # does not restart the intonation from cold.
            kwargs["previous_text"] = previous_text[-400:]
        return self._client.text_to_speech.convert(
            voice_id=self._voice_id,
            text=text,
            model_id=self._model,
            output_format=self._format,
            # Start returning audio before the whole sentence is synthesized.
            optimize_streaming_latency=3,
            **kwargs,
        )

    def speak(self, text: str) -> None:
        """Synthesize and play, blocking until the last sample is out."""
        if not text.strip():
            return
        self._cancelled = False
        stream = sd.RawOutputStream(
            samplerate=self._sample_rate, channels=1, dtype="int16"
        )
        stream.start()
        try:
            # Writing straight into the OS buffer lets the device drain at
            # exactly the right rate and absorbs network jitter. Sleep-based
            # chunk scheduling drifts and produces audible pops.
            leftover = b""
            for chunk in self._synthesize(text):
                if self._cancelled:
                    break
                buf = leftover + chunk
                # RawOutputStream wants whole frames (2 bytes, mono int16).
                usable = len(buf) - (len(buf) % 2)
                leftover = buf[usable:]
                if usable:
                    stream.write(buf[:usable])
        finally:
            stream.stop()
            stream.close()

    def speak_stream(self, sentences) -> str:
        """Speak sentences as they arrive. Blocks until the last one is out.

        One output stream for the whole reply, not one per sentence -- opening
        a device per sentence produces an audible click between them.
        `previous_text` keeps prosody continuous across the joins.
        """
        self._cancelled = False
        stream = sd.RawOutputStream(
            samplerate=self._sample_rate, channels=1, dtype="int16"
        )
        stream.start()
        said: list[str] = []
        try:
            leftover = b""
            for sentence in sentences:
                if self._cancelled:
                    break
                for chunk in self._synthesize(sentence, previous_text=" ".join(said)):
                    if self._cancelled:
                        break
                    buf = leftover + chunk
                    usable = len(buf) - (len(buf) % 2)
                    leftover = buf[usable:]
                    if usable:
                        stream.write(buf[:usable])
                said.append(sentence)
        finally:
            stream.stop()
            stream.close()
        return " ".join(said)

    def warm(self) -> None:
        """Pay the TLS/connection cost before the first real reply."""
        try:
            next(iter(self._synthesize("Ready.")))
        except Exception as exc:
            print(f"[voice] warm-up skipped: {exc}")

    def cancel(self) -> None:
        """Stop mid-reply, e.g. the user interrupted the agent."""
        self._cancelled = True

    def to_wav(self, text: str, path: str) -> str:
        """Render to a file. Used to build demo audiobook assets."""
        import soundfile as sf

        pcm = b"".join(self._synthesize(text))
        pcm = pcm[: len(pcm) - (len(pcm) % 2)]
        sf.write(path, np.frombuffer(pcm, dtype=np.int16), self._sample_rate)
        return path

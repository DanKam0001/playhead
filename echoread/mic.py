"""Microphone capture + local VAD.

Two consumers of the same audio:
  1. AssemblyAI, over the websocket, for words.
  2. A local energy VAD, in-process, for *when speech starts*.

The local VAD exists because a network round trip is too slow for barge-in.
By the time SpeechStarted comes back from the server the narrator has talked
over the user for a few hundred ms. Locally we can duck the book in ~50 ms.
The server's opinion still arrives and is used to confirm or cancel.
"""
import queue
import threading
import time
from typing import Callable, Iterator, Optional

import numpy as np
import sounddevice as sd
from assemblyai.streaming.v3.extras import EnergyVad

from .config import (ATTACK_FRAMES, CHANNELS, FRAME_SAMPLES, MIN_TRIGGER_RMS,
                     SAMPLE_RATE)


class Microphone:
    def __init__(
        self,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[], None]] = None,
        threshold_ratio: float = 4.0,
        hangover_frames: int = 8,
        device: Optional[int | str] = None,
        min_trigger_rms: float = MIN_TRIGGER_RMS,
        attack_frames: int = ATTACK_FRAMES,
    ):
        self._q: queue.Queue[Optional[bytes]] = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._closed = threading.Event()
        self._device = device

        # NOTE: EnergyVad only adapts its noise floor while INACTIVE. If the
        # ambient level sits above the initial floor, every frame reads as
        # speech, hangover keeps it active, and the floor never updates -- it
        # latches on forever. In a moving car that means the book ducks once
        # and never comes back. So we measure the cabin before arming it.
        self._threshold_ratio = threshold_ratio
        self._hangover_frames = hangover_frames
        self._vad = EnergyVad(
            threshold_ratio=threshold_ratio,
            hangover_frames=hangover_frames,
        )
        self._calibrating = False
        self._calib_rms: list[float] = []
        self._floor = 0.0
        self._min_trigger = min_trigger_rms
        self._attack_frames = attack_frames
        self._active_run = 0
        self._on_speech_start = on_speech_start
        self._on_speech_end = on_speech_end
        self._speaking = False

        # Set to False to stop the VAD from firing barge-ins -- e.g. while the
        # agent's own TTS is coming out of the speakers. Without this the agent
        # interrupts itself, because the mic hears the reply.
        self.vad_enabled = True

    def _callback(self, indata, _frames, _time, status):
        if status:
            print(f"[mic] {status}")
        pcm16 = (indata[:, 0] * 32767).astype(np.int16)
        self._q.put(pcm16.tobytes())

        if self._calibrating:
            self._calib_rms.append(float(np.sqrt(np.mean(indata[:, 0] ** 2))))
            return
        if not self.vad_enabled:
            return
        result = self._vad.process(indata[:, 0])
        # Require a sustained run, not a single hot frame. Without this, one
        # transient -- a click, a chair, a consonant burst from the speakers --
        # is enough to duck the book.
        self._active_run = self._active_run + 1 if result.active else 0
        speaking_now = self._active_run >= self._attack_frames
        if speaking_now and not self._speaking:
            self._speaking = True
            if self._on_speech_start:
                self._on_speech_start()
        elif not speaking_now and self._speaking:
            self._speaking = False
            if self._on_speech_end:
                self._on_speech_end()

    def start(self, calibrate_seconds: float = 1.5) -> None:
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="float32",
            blocksize=FRAME_SAMPLES,
            callback=self._callback,
            device=self._device,
        )
        self._stream.start()
        info = sd.query_devices(self._stream.device)
        print(f"[mic] listening on [{self._stream.device}] {info['name']}")
        if calibrate_seconds > 0:
            self.calibrate(calibrate_seconds)

    def frames(self) -> Iterator[bytes]:
        """Blocking generator of 16-bit PCM chunks, fed straight to the socket."""
        while not self._closed.is_set():
            chunk = self._q.get()
            if chunk is None:
                break
            yield chunk

    def calibrate(self, seconds: float = 1.5, keep_max: bool = False) -> float:
        """Measure the ambient level and set the VAD's noise floor to it.

        Called twice, and both matter:

        1. Before the book starts -> the true room floor.
        2. Just after it starts, with keep_max -> the floor *including* the
           audiobook leaking back into the microphone.

        Stage 2 is what stops the book from interrupting itself. On an isolated
        headset mic the two readings are nearly identical and barge-in stays
        sensitive; on an open desk mic that hears the speakers, stage 2 raises
        the bar above the leakage so only real speech gets through.

        Nobody should be talking during either pass.
        """
        self._calib_rms = []
        self._calibrating = True
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(0.05)
        self._calibrating = False

        samples = sorted(self._calib_rms)
        if not samples:
            print("[mic] WARNING: no audio frames arrived during calibration -- "
                  "check the device is not muted. Run mic_check.py.")
            floor = 1e-4
        else:
            # The 90th percentile, not the median: leakage from the audiobook is
            # speech, so it is bursty. The median sits in the gaps between words
            # and would leave the threshold under the loud parts, which is
            # exactly what makes the book interrupt itself.
            floor = max(samples[int(len(samples) * 0.9)], 1e-5)
        if keep_max:
            floor = max(floor, self._floor)
        # However quiet the room was, never let the trigger drop to a level that
        # room tone can clear. This is the difference between "sensitive" and
        # "fires on everything".
        measured = floor
        floor = max(floor, self._min_trigger / self._threshold_ratio)
        if floor > measured:
            print(f"[mic] measured floor {measured:.5f} was below the minimum; "
                  f"trigger clamped to {self._min_trigger:.5f}")
        self._floor = floor
        self._vad = EnergyVad(
            threshold_ratio=self._threshold_ratio,
            hangover_frames=self._hangover_frames,
            initial_noise_floor=floor,
        )
        self._speaking = False
        self._active_run = 0
        print(f"[mic] calibrated: noise floor {floor:.5f}, "
              f"barge-in above {floor * self._threshold_ratio:.5f}")
        if floor > 0.02:
            print("[mic] WARNING: noise floor is high -- barge-in will need a "
                  "raised voice. Headphones, or a mic further from the speakers, "
                  "would fix this.")
        return floor

    def reset_vad(self) -> None:
        """Re-baseline to the last calibration, e.g. after the agent stops talking."""
        self._vad.reset()
        self._speaking = False
        self._active_run = 0

    def close(self) -> None:
        self._closed.set()
        self._q.put(None)
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def __enter__(self) -> "Microphone":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

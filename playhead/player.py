"""Audiobook playback with sample-accurate timestamp tracking.

The timestamp is the load-bearing part of this project: it is the retrieval
key for RAG. So we do not trust a wall-clock timer (it drifts against the
audio device and lies while paused). We count frames the sound card has
actually consumed.
"""
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf


class AudiobookPlayer:
    """Pausable / duckable player that always knows where it is in the book."""

    def __init__(self, path: str | Path, duck_volume: float = 0.15):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Audiobook file not found: {self.path}")

        self._file = sf.SoundFile(str(self.path))
        self.samplerate = self._file.samplerate
        self.channels = self._file.channels
        self.duration = len(self._file) / self.samplerate

        self._duck_volume = duck_volume
        self._gain = 1.0
        self._paused = False
        self._finished = threading.Event()
        # Frames the device has actually played. Guarded because the audio
        # callback runs on PortAudio's thread, not ours.
        self._frames_played = 0
        self._lock = threading.Lock()
        self._stream: Optional[sd.OutputStream] = None

    # ---------- position ----------

    @property
    def position_seconds(self) -> float:
        with self._lock:
            return self._frames_played / self.samplerate

    def seek(self, seconds: float) -> None:
        with self._lock:
            frame = max(0, min(int(seconds * self.samplerate), len(self._file)))
            self._file.seek(frame)
            self._frames_played = frame

    # ---------- transport ----------

    def _callback(self, outdata, frames, _time, status):
        if status:
            print(f"[player] {status}")
        with self._lock:
            if self._paused:
                outdata[:] = 0
                return  # do NOT advance the clock while paused
            data = self._file.read(frames, dtype="float32", always_2d=True)
            n = len(data)
            self._frames_played += n
            outdata[:n] = data * self._gain
            if n < frames:
                outdata[n:] = 0
                raise sd.CallbackStop

    def start(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype="float32",
            callback=self._callback,
            finished_callback=self._finished.set,
            blocksize=1024,
        )
        self._stream.start()

    def pause(self) -> float:
        """Hard stop. Returns the timestamp we stopped at."""
        with self._lock:
            self._paused = True
            return self._frames_played / self.samplerate

    def duck(self) -> float:
        """Drop to background volume but keep playing.

        Used for the optimistic barge-in: if the trigger turns out to be road
        noise, unducking is far less jarring than an audible stop/start.
        """
        with self._lock:
            self._gain = self._duck_volume
            return self._frames_played / self.samplerate

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._gain = 1.0

    def rewind(self, seconds: float) -> None:
        """Back up a little before resuming, so the user re-hears their context."""
        self.seek(max(0.0, self.position_seconds - seconds))

    @property
    def is_playing(self) -> bool:
        return self._stream is not None and self._stream.active and not self._paused

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._finished.wait(timeout)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
        self._file.close()

    def __enter__(self) -> "AudiobookPlayer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

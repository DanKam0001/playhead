"""Central settings, loaded once from .env."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# AssemblyAI streaming requires mono PCM. 16 kHz is the sweet spot for
# latency vs accuracy; the mic and the WS session must agree on this.
SAMPLE_RATE = 16_000
CHANNELS = 1
# 50 ms frames. Small enough that local VAD reacts fast, large enough
# that we are not spamming the socket with tiny writes.
FRAME_MS = 50
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000

# The trigger level may never fall below this, however quiet calibration was.
# A silent room calibrates to ~0.00002; times any ratio that is still ~0, so
# room tone (~0.0002) clears it and the book ducks constantly. Measured on this
# machine: speech peaks ~0.024, so 0.004 leaves 6x headroom for speech while
# sitting 22x above room tone.
MIN_TRIGGER_RMS = 0.004
# Speech must stay above the trigger for this many consecutive frames before
# it counts. At 50 ms/frame, 3 frames = 150 ms: shorter than any real word,
# longer than a keyboard tap, a chair creak, or a click.
ATTACK_FRAMES = 3


def _req(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return val


@dataclass(frozen=True)
class Settings:
    assemblyai_api_key: str
    gemini_api_key: str
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    vad_threshold_ratio: float
    hangover_frames: int
    min_trigger_rms: float
    attack_frames: int
    duck_volume: float

    @classmethod
    def load(cls, require_brain: bool = False) -> "Settings":
        """require_brain=False lets step-1 (ear only) run without Gemini/11Labs keys."""
        getter = _req if require_brain else (lambda n: os.getenv(n, ""))
        return cls(
            assemblyai_api_key=_req("ASSEMBLYAI_API_KEY"),
            gemini_api_key=getter("GEMINI_API_KEY"),
            elevenlabs_api_key=getter("ELEVENLABS_API_KEY"),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
            vad_threshold_ratio=float(os.getenv("BARGEIN_VAD_THRESHOLD_RATIO", "4.0")),
            hangover_frames=int(os.getenv("BARGEIN_HANGOVER_FRAMES", "8")),
            min_trigger_rms=float(os.getenv("BARGEIN_MIN_TRIGGER", str(MIN_TRIGGER_RMS))),
            attack_frames=int(os.getenv("BARGEIN_ATTACK_FRAMES", str(ATTACK_FRAMES))),
            duck_volume=float(os.getenv("DUCK_VOLUME", "0.15")),
        )

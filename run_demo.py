"""Playhead: play an audiobook, interrupt it with your voice, discuss it.

    python run_demo.py audio/relativity.wav --device 3 --start 9:20

Looks for data/<name>.db (built by build_index.py). With an index and keys it
answers out loud; without, it runs ear-only and just prints the question.
"""
import sys
from pathlib import Path

from playhead.config import Settings
from playhead.ears import Ears
from playhead.mic import Microphone
from playhead.player import AudiobookPlayer
from playhead.session import Session

# Fallback only. Real keyterms are derived from the book by build_index.py and
# loaded from data/<name>.keyterms.json -- hardcoding does not survive a
# second book.
FALLBACK_KEYTERMS = ["eigenvector", "eigenvalue", "principal component analysis"]


def load_keyterms(stem: str) -> list:
    path = Path("data") / f"{stem}.keyterms.json"
    if path.exists():
        import json
        terms = json.loads(path.read_text(encoding="utf-8"))
        print(f"[demo] {len(terms)} keyterms from {path.name}")
        return terms
    print(f"[demo] no {path.name} -- using fallback keyterms")
    return FALLBACK_KEYTERMS


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2

    settings = Settings.load(require_brain=False)
    player = AudiobookPlayer(sys.argv[1], duck_volume=settings.duck_volume)
    print(f"[demo] {player.path.name}: {player.duration:.0f}s @ {player.samplerate} Hz")

    # Brain and voice are optional: without them this still demonstrates
    # barge-in, which is the part that has to work on stage.
    brain = voice = None
    db = Path("data") / f"{player.path.stem}.db"
    if db.exists() and settings.gemini_api_key:
        from playhead.brain import Brain
        from playhead.library import Library
        lib = Library(db)
        brain = Brain(settings.gemini_api_key, lib)
        print(f"[demo] index: {len(lib)} chunks from {db}")
    else:
        print(f"[demo] no index at {db} -- run build_index.py first. Ear-only mode.")
    if settings.elevenlabs_api_key:
        from playhead.voice import ElevenLabsVoice
        voice = ElevenLabsVoice(settings.elevenlabs_api_key, settings.elevenlabs_voice_id)
    else:
        print("[demo] no ElevenLabs key -- replies will be printed, not spoken.")

    session_ref = {}

    # --device N picks an input; find N with: python mic_check.py --list
    device = None
    if "--device" in sys.argv:
        device = int(sys.argv[sys.argv.index("--device") + 1])
    # --start SECONDS jumps the playhead, so filming a take at 9:40 does not
    # mean sitting through nine minutes of Einstein first. Accepts 580 or 9:40.
    if "--start" in sys.argv:
        raw = sys.argv[sys.argv.index("--start") + 1]
        start = (int(raw.split(":")[0]) * 60 + float(raw.split(":")[1])
                 if ":" in raw else float(raw))
        player.seek(start)
        print(f"[demo] starting at {int(start) // 60}:{int(start) % 60:02d}")
    mic = Microphone(
        on_speech_start=lambda: session_ref["s"].on_local_speech_start(),
        on_speech_end=lambda: session_ref["s"].on_local_speech_end(),
        threshold_ratio=settings.vad_threshold_ratio,
        hangover_frames=settings.hangover_frames,
        device=device,
        min_trigger_rms=settings.min_trigger_rms,
        attack_frames=settings.attack_frames,
    )
    ears = Ears(
        api_key=settings.assemblyai_api_key,
        on_speech_started=lambda ts: session_ref["s"].on_server_speech_started(ts),
        on_partial_turn=lambda t: session_ref["s"].on_partial_turn(t),
        on_final_turn=lambda t: session_ref["s"].on_final_turn(t),
        keyterms=load_keyterms(player.path.stem),
    )
    session_ref["s"] = Session(player=player, ears=ears, mic=mic,
                                brain=brain, voice=voice)

    if brain is not None or voice is not None:
        print("[demo] warming up API connections (saves ~3s on the first question)...")
        session_ref["s"].warm()
    print("[demo] calibrating mic -- STAY QUIET for ~1.5s (book starts after)...")
    print("[demo] then: talk to interrupt. Ctrl-C to quit.")
    print("[demo] not ducking when you talk? run: python mic_check.py")
    try:
        session_ref["s"].run()
    except KeyboardInterrupt:
        print("\n[demo] bye")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Empirical check of what the streaming API actually sends back.

Docs disagree with each other on turn_is_formatted / speech_model / which
turn-detection params U3.5 Pro honors. This settles it against the live API.
"""
import os, time
import numpy as np, soundfile as sf
from dotenv import load_dotenv

load_dotenv()

QUESTION = "Wait, hold on. What did that last sentence about eigenvectors actually mean?"
WAV = "audio/_probe_question.wav"


def synth():
    if os.path.exists(WAV):
        return
    from playhead.voice import ElevenLabsVoice
    v = ElevenLabsVoice(os.environ["ELEVENLABS_API_KEY"],
                        os.getenv("ELEVENLABS_VOICE_ID", ""))
    v.to_wav(QUESTION, WAV)
    print(f"[probe] synthesized: {WAV}")


def main():
    synth()
    data, sr = sf.read(WAV, dtype="float32", always_2d=True)
    mono = data[:, 0]
    # Resample to 16k by linear interpolation (good enough for a probe).
    n = int(len(mono) * 16000 / sr)
    mono = np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono)
    pcm = (mono * 32767).astype("<i2").tobytes()
    print(f"[probe] {len(mono)/16000:.1f}s of audio at 16kHz")

    from assemblyai.streaming.v3 import (
        StreamingClient, StreamingClientOptions, StreamingParameters,
        StreamingEvents, StreamingMode, Encoding, SpeechModel,
    )
    seen = []
    client = StreamingClient(StreamingClientOptions(
        api_key=os.environ["ASSEMBLYAI_API_KEY"]))
    client.on(StreamingEvents.Begin, lambda c, e: print(f"[Begin] id={e.id}"))
    client.on(StreamingEvents.SpeechStarted,
              lambda c, e: print(f"[SpeechStarted] {e.model_dump()}"))
    client.on(StreamingEvents.Error, lambda c, e: print(f"[Error] {e}"))

    def on_turn(c, e):
        seen.append(e)
        print(f"[Turn] order={e.turn_order} eot={e.end_of_turn} "
              f"formatted={e.turn_is_formatted} conf={e.end_of_turn_confidence} "
              f"text={e.transcript!r}")
    client.on(StreamingEvents.Turn, on_turn)

    client.connect(StreamingParameters(
        sample_rate=16000,
        encoding=Encoding.pcm_s16le,
        speech_model=SpeechModel.universal_3_5_pro,
        mode=StreamingMode.min_latency,
        keyterms_prompt=["eigenvector", "eigenvalue"],
    ))

    def frames():
        step = 16000 * 2 // 10  # 100 ms of int16
        for i in range(0, len(pcm), step):
            yield pcm[i:i + step]
            time.sleep(0.1)      # must not send faster than realtime
        time.sleep(2.5)          # let end-of-turn fire

    client.stream(frames())
    client.disconnect(terminate=True)

    print("\n=== VERDICT ===")
    finals = [e for e in seen if e.end_of_turn]
    print(f"turn events: {len(seen)}, end_of_turn events: {len(finals)}")
    for e in finals:
        print(f"  eot turn_is_formatted={e.turn_is_formatted} conf={e.end_of_turn_confidence}")
    if finals and not any(e.turn_is_formatted for e in finals):
        print("  !! no end_of_turn event had turn_is_formatted=True")
        print("  !! ears.py gating on turn_is_formatted would DROP every question")


if __name__ == "__main__":
    main()

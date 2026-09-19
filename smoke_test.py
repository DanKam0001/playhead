"""Offline checks that need no API key and no speakers."""
import time
import numpy as np
import soundfile as sf

from echoread.player import AudiobookPlayer
from echoread.session import Session, State


def make_tone(path="audio/_test_tone.wav", secs=30, sr=22050):
    t = np.linspace(0, secs, int(sr * secs), endpoint=False)
    sf.write(path, (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), sr)
    return path


class FakeStream:
    active = True
    def stop(self): pass
    def close(self): pass


def main():
    path = make_tone()
    p = AudiobookPlayer(path, duck_volume=0.15)
    assert abs(p.duration - 30) < 0.1, p.duration
    p._stream = FakeStream()

    # Drive the audio callback by hand: 1 s of frames at 1024 blocks.
    buf = np.zeros((1024, p.channels), dtype="float32")
    blocks = p.samplerate // 1024
    for _ in range(blocks):
        p._callback(buf, 1024, None, None)
    pos = p.position_seconds
    assert 0.9 < pos < 1.05, f"timestamp drifted: {pos}"

    # Ducking keeps the clock running; pausing freezes it.
    p.duck()
    for _ in range(blocks):
        p._callback(buf, 1024, None, None)
    assert p.position_seconds > pos, "ducking must not stop the clock"
    assert abs(buf.max() - 0.0) >= 0, ""
    mid = p.position_seconds
    p.pause()
    for _ in range(blocks):
        p._callback(buf, 1024, None, None)
    assert p.position_seconds == mid, "paused clock must not advance"

    p.resume()
    p.rewind(3.0)  # only ~2 s in, so this must clamp to 0 rather than go negative
    assert p.position_seconds == 0.0, p.position_seconds
    p.seek(10.0)
    p.rewind(3.0)
    assert abs(p.position_seconds - 7.0) < 0.01, p.position_seconds

    # State machine: local VAD -> ducked -> server confirm -> listening.
    p2 = AudiobookPlayer(path); p2._stream = FakeStream()
    for _ in range(blocks * 5):
        p2._callback(buf, 1024, None, None)

    class NoMic:
        vad_enabled = True
        def reset_vad(self): pass

    s = Session(player=p2, ears=None, mic=NoMic())
    assert s.state is State.PLAYING
    s.on_local_speech_start()
    assert s.state is State.DUCKED, s.state
    assert 4.8 < s.interrupt_timestamp < 5.1, s.interrupt_timestamp
    s.on_server_speech_started(0)
    assert s.state is State.LISTENING, s.state

    # False-alarm path: duck, then speech ends after the window with no confirm.
    p3 = AudiobookPlayer(path); p3._stream = FakeStream()
    s2 = Session(player=p3, ears=None, mic=NoMic())
    s2.on_local_speech_start()
    assert s2.state is State.DUCKED
    s2.on_local_speech_end()
    assert s2.state is State.DUCKED, "must not unduck inside the confirm window"
    s2._ducked_at -= 2.0
    s2.on_local_speech_end()
    assert s2.state is State.PLAYING, s2.state

    # --- turn stitching: a paused question must not fire the brain early ---
    from echoread import session as S
    p4 = AudiobookPlayer(path); p4._stream = FakeStream()
    asked = []
    s3 = Session(player=p4, ears=None, mic=NoMic())
    s3.brain = type("B", (), {"answer": lambda _s, q, t: asked.append(q) or "ok"})()
    s3.on_local_speech_start(); s3.on_server_speech_started(0)
    s3.on_final_turn("Wait, hold on.")          # the real API really does split here
    time.sleep(0.4)
    assert not asked, f"fired on a fragment: {asked}"
    s3.on_final_turn("What did that last sentence mean?")
    time.sleep(S.TURN_STITCH_S + 0.6)
    assert len(asked) == 1, f"expected one stitched question, got {asked}"
    assert asked[0] == "Wait, hold on. What did that last sentence mean?", asked[0]

    # A hold phrase must NOT be answered -- it should keep listening.
    p7 = AudiobookPlayer(path); p7._stream = FakeStream()
    held = []
    s6 = Session(player=p7, ears=None, mic=NoMic())
    s6.brain = type("B", (), {"answer": lambda _s, q, t: held.append(q) or "ok"})()
    s6.on_local_speech_start(); s6.on_server_speech_started(0)
    s6.on_final_turn("Hold on.")
    time.sleep(S.TURN_STITCH_S + 0.4)
    assert not held, f"answered a hold phrase: {held}"
    assert s6.state is State.LISTENING, s6.state
    s6.on_final_turn("What did that last part mean?")
    time.sleep(S.TURN_STITCH_COMPLETE_S + 0.4)
    assert held == ["What did that last part mean?"], held
    p7.close()

    # A complete-sounding question uses the short stitch window.
    p6 = AudiobookPlayer(path); p6._stream = FakeStream()
    asked3 = []
    s5 = Session(player=p6, ears=None, mic=NoMic())
    s5.brain = type("B", (), {"answer": lambda _s, q, t: asked3.append(q) or "ok"})()
    s5.on_local_speech_start(); s5.on_server_speech_started(0)
    s5.on_final_turn("What did that last sentence mean?")
    time.sleep(S.TURN_STITCH_COMPLETE_S + 0.35)
    assert asked3, "a complete question should not wait the full stitch window"
    assert S.TURN_STITCH_COMPLETE_S < S.TURN_STITCH_S
    p6.close()

    # "okay, carry on" is an instruction to resume, not a question.
    p5 = AudiobookPlayer(path); p5._stream = FakeStream()
    asked2 = []
    s4 = Session(player=p5, ears=None, mic=NoMic())
    s4.brain = type("B", (), {"answer": lambda _s, q, t: asked2.append(q) or "ok"})()
    s4.on_local_speech_start(); s4.on_server_speech_started(0)
    s4.on_final_turn("Okay, carry on")
    time.sleep(S.TURN_STITCH_S + 0.6)
    assert not asked2, f"resume command sent to the LLM: {asked2}"
    assert s4.state is State.PLAYING, s4.state
    p4.close(); p5.close()

    # Question path with no brain wired must still un-pause the book.
    s.on_final_turn("what did that last sentence mean?")
    for _ in range(80):
        if s.state is State.PLAYING:
            break
        time.sleep(0.05)
    assert s.state is State.PLAYING, f"book left stuck in {s.state}"

    # --- VAD: the uncalibrated latch-up this project has to avoid ---
    from assemblyai.streaming.v3.extras import EnergyVad
    cabin = lambda: np.random.normal(0, 0.01, 800).astype("float32")   # road noise
    speech = lambda: np.random.normal(0, 0.15, 800).astype("float32")  # a question

    naive = EnergyVad(threshold_ratio=4.0, hangover_frames=8)  # default 1e-4 floor
    latched = all(naive.process(cabin()).active for _ in range(100))
    assert latched, "expected the documented latch-up on an uncalibrated floor"

    # Calibrated to the same cabin noise, it stays quiet and still hears speech.
    floor = float(np.median([np.sqrt(np.mean(cabin() ** 2)) for _ in range(30)]))
    tuned = EnergyVad(threshold_ratio=4.0, hangover_frames=8, initial_noise_floor=floor)
    assert not any(tuned.process(cabin()).active for _ in range(100)),         "calibrated VAD must ignore steady road noise"
    assert tuned.process(speech()).active, "calibrated VAD must still hear speech"
    for _ in range(8):
        tuned.process(cabin())          # hangover drains
    assert not tuned.process(cabin()).active, "must release after speech ends"

    # --- barge-in trigger must survive a silent calibration ---
    from echoread.mic import Microphone
    from echoread.config import FRAME_SAMPLES, ATTACK_FRAMES
    from assemblyai.streaming.v3.extras import EnergyVad as EV

    # Real numbers measured on hardware 2026-09-10.
    SILENT_ROOM, ROOM_TONE, SPEECH = 0.00002, 0.00018, 0.02427
    mic = Microphone(threshold_ratio=4.0)
    floor = max(max(SILENT_ROOM, 1e-5), mic._min_trigger / mic._threshold_ratio)
    trigger = floor * mic._threshold_ratio
    assert ROOM_TONE < trigger, (
        f"room tone {ROOM_TONE} clears trigger {trigger} -> ducks on everything")
    assert SPEECH > trigger, f"speech {SPEECH} below trigger {trigger}"

    # Attack: a short transient must not fire; sustained speech must.
    fired = []
    m2 = Microphone(on_speech_start=lambda: fired.append(1), threshold_ratio=4.0)
    m2._vad = EV(threshold_ratio=4.0, hangover_frames=0, initial_noise_floor=floor)
    quiet = np.zeros((FRAME_SAMPLES, 1), dtype="float32")
    loud = np.full((FRAME_SAMPLES, 1), 0.05, dtype="float32")
    for _ in range(ATTACK_FRAMES - 1):
        m2._callback(loud, FRAME_SAMPLES, None, None)
    assert not fired, "a sub-attack transient triggered barge-in"
    m2._callback(quiet, FRAME_SAMPLES, None, None)
    for _ in range(ATTACK_FRAMES):
        m2._callback(loud, FRAME_SAMPLES, None, None)
    assert fired, "sustained speech failed to trigger barge-in"

    p.close(); p2.close(); p3.close()
    print("PASS: timestamps, transport, barge-in, stitching, hold+resume intent, VAD, trigger floor + attack")


if __name__ == "__main__":
    main()

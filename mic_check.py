"""See what the microphone is actually hearing, and whether barge-in would fire.

    python mic_check.py                 # list devices, then meter the default
    python mic_check.py --list          # just list input devices
    python mic_check.py --device 3      # meter a specific device
    python mic_check.py --ratio 2.5     # try a different trigger threshold

The meter shows live RMS against the calibrated noise floor and the trigger
threshold. If TALKING never lights up while you speak, barge-in cannot work --
and this tells you whether the problem is the device, the floor, or the ratio.
"""
import argparse
import sys
import time

import numpy as np
import sounddevice as sd
from assemblyai.streaming.v3.extras import EnergyVad

from playhead.config import (ATTACK_FRAMES, CHANNELS, FRAME_SAMPLES,
                             MIN_TRIGGER_RMS, SAMPLE_RATE)

BAR_W = 34
# Full-scale for the bar. Normal speech into a headset mic sits well under this.
BAR_MAX = 0.25


def list_devices() -> None:
    print("Input devices:")
    default_in = sd.default.device[0]
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        mark = " <- default" if i == default_in else ""
        print(f"  [{i:2}] {d['name']}  ({d['max_input_channels']}ch, "
              f"{int(d['default_samplerate'])}Hz){mark}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--ratio", type=float, default=4.0)
    ap.add_argument("--hangover", type=int, default=8)
    ap.add_argument("--min-trigger", type=float, default=MIN_TRIGGER_RMS)
    ap.add_argument("--attack", type=int, default=ATTACK_FRAMES)
    ap.add_argument("--calibrate", type=float, default=1.5)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="auto-stop after N seconds instead of Ctrl-C")
    args = ap.parse_args()

    list_devices()
    if args.list:
        return 0

    dev = args.device if args.device is not None else sd.default.device[0]
    info = sd.query_devices(dev)
    print(f"\nUsing: [{dev}] {info['name']}")

    levels: list[float] = []
    state = {"calibrating": True, "vad": None, "rms": 0.0, "active": False,
             "triggers": 0, "peak": 0.0, "run": 0}

    def cb(indata, _frames, _t, status):
        if status:
            print(f"\n[mic] {status}")
        mono = indata[:, 0]
        rms = float(np.sqrt(np.mean(mono ** 2)))
        state["rms"] = rms
        if state["calibrating"]:
            levels.append(rms)
            return
        # Peak must exclude calibration, or an idle room "peaks" above its own
        # threshold and the verdict below misreads silence as failed speech.
        state["peak"] = max(state["peak"], rms)
        was = state["active"]
        # Mirror the session's attack requirement, or the meter lights up on
        # transients the real thing would ignore.
        hot = state["vad"].process(mono).active
        state["run"] = state["run"] + 1 if hot else 0
        state["active"] = state["run"] >= args.attack
        if state["active"] and not was:
            state["triggers"] += 1

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32",
                        blocksize=FRAME_SAMPLES, callback=cb, device=dev):
        print(f"\nCalibrating for {args.calibrate}s -- STAY QUIET...")
        time.sleep(args.calibrate)
        state["calibrating"] = False

        if not levels:
            print("\nNo audio frames arrived at all. The device is not delivering "
                  "input -- check Windows mic permissions and that it isn't muted.")
            return 1

        floor = max(float(np.median(levels)), 1e-5)
        # Same clamp the live session applies: a silent room must not produce a
        # trigger that room tone can clear.
        measured = floor
        floor = max(floor, args.min_trigger / args.ratio)
        thresh = floor * args.ratio
        if floor > measured:
            print(f"  measured floor {measured:.5f} was below the minimum; "
                  f"clamped so the trigger is {thresh:.5f}")
        state["vad"] = EnergyVad(threshold_ratio=args.ratio,
                                 hangover_frames=args.hangover,
                                 initial_noise_floor=floor)
        quiet_peak = max(levels)
        print(f"  noise floor : {floor:.5f}  (peak while quiet {quiet_peak:.5f})")
        print(f"  trigger at  : {thresh:.5f}  (ratio {args.ratio})")
        print(f"  attack      : {args.attack} frames "
              f"({args.attack * 1000 // (SAMPLE_RATE // FRAME_SAMPLES)} ms of sustained speech)")
        print("\nNow TALK. Ctrl-C to stop.\n")

        deadline = time.monotonic() + args.seconds if args.seconds else None
        try:
            while deadline is None or time.monotonic() < deadline:
                rms = state["rms"]
                filled = min(BAR_W, int(BAR_W * rms / BAR_MAX))
                mark = min(BAR_W - 1, int(BAR_W * thresh / BAR_MAX))
                bar = ["-"] * BAR_W
                for i in range(filled):
                    bar[i] = "#"
                bar[mark] = "|" if bar[mark] == "-" else "!"
                tag = "TALKING" if state["active"] else "       "
                sys.stdout.write(
                    f"\r[{''.join(bar)}] rms={rms:.4f} thr={thresh:.4f} "
                    f"{tag} triggers={state['triggers']}  "
                )
                sys.stdout.flush()
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass

    peak = state["peak"]
    print(f"\n\nPeak level while listening: {peak:.5f}")
    print(f"Trigger threshold:          {thresh:.5f}")

    # Absolute floor for "someone actually spoke". Normal speech into any mic
    # clears this easily; an idle room never does. Comparing peak to the
    # threshold alone is meaningless when the peak is just room tone.
    SPEECH_FLOOR = 0.005
    if peak < SPEECH_FLOOR:
        print(f"\nDIAGNOSIS: no speech-level audio arrived "
              f"(peak {peak:.5f} is room tone).")
        print("  Either nobody spoke, or this is not the mic you talk into.")
        print("  - Check Windows > Settings > System > Sound > Input: is this")
        print("    device selected, and is its volume above 0 and unmuted?")
        print("  - Try another one:  python mic_check.py --device N")
    elif state["triggers"] == 0:
        head = thresh / max(peak, 1e-9)
        suggested = max(1.5, args.ratio / head * 0.6)
        print(f"\nDIAGNOSIS: speech was audible but never crossed the threshold "
              f"({head:.1f}x short).")
        print(f"  - Retry lower:  python mic_check.py --ratio {suggested:.1f}")
        print(f"  - Then set BARGEIN_VAD_THRESHOLD_RATIO={suggested:.1f} in .env")
    else:
        print(f"\nOK: barge-in fired {state['triggers']} times -- this device "
              f"and ratio work.")
        print(f"  Set BARGEIN_VAD_THRESHOLD_RATIO={args.ratio} in .env"
              + (f", and run the demo with --device {dev}"
                 if args.device is not None else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

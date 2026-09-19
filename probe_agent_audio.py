"""Send real speech to the agent and record the exact event names it replies with.

The browser client was written from prose. This settles the user-transcript and
speech-detection event names, and the audio format the session expects.
"""
import asyncio, base64, json, os, urllib.request
from collections import Counter

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
import websockets

load_dotenv()
KEY = os.environ["ASSEMBLYAI_API_KEY"]
AGENT_ID = os.environ["ECHOREAD_AGENT_ID"]
RATE = 24000


def mint():
    req = urllib.request.Request(
        "https://agents.assemblyai.com/v1/token?expires_in_seconds=120",
        headers={"Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(req))["token"]


def question_pcm():
    data, sr = sf.read("audio/_probe_question.wav", dtype="float32", always_2d=True)
    mono = data[:, 0]
    n = int(len(mono) * RATE / sr)
    mono = np.interp(np.linspace(0, len(mono) - 1, n), np.arange(len(mono)), mono)
    return (mono * 32767).astype("<i2").tobytes()


async def main():
    pcm = question_pcm()
    print(f"sending {len(pcm)/2/RATE:.1f}s of speech at {RATE} Hz")
    seen, interesting = Counter(), []

    async with websockets.connect(f"wss://agents.assemblyai.com/v1/ws?token={mint()}",
                                  max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update",
                                  "session": {"agent_id": AGENT_ID}}))
        ready = asyncio.Event()

        async def reader():
            async for raw in ws:
                m = json.loads(raw)
                t = m.get("type")
                seen[t] += 1
                if t == "session.ready":
                    print("\nsession.ready config:")
                    print(json.dumps(m.get("config", {}), indent=1)[:700])
                    ready.set()
                elif t and not t.startswith("reply.audio"):
                    if t.startswith(("input.", "transcript.user", "tool", "session.error", "error")):
                        interesting.append(m)
                        print(f"  [{t}] {json.dumps({k: v for k, v in m.items() if k != 'type'})[:220]}")

        task = asyncio.create_task(reader())
        await asyncio.wait_for(ready.wait(), timeout=15)
        await asyncio.sleep(3)          # let the greeting finish

        print("\n--- streaming the question ---")
        step = int(RATE * 0.05) * 2     # 50 ms frames
        for i in range(0, len(pcm), step):
            await ws.send(json.dumps({"type": "input.audio",
                                      "audio": base64.b64encode(pcm[i:i+step]).decode()}))
            await asyncio.sleep(0.05)
        await asyncio.sleep(12)         # let it transcribe, call the tool, answer
        task.cancel()

    print("\n=== event tally ===")
    for t, n in seen.most_common():
        print(f"  {n:4}  {t}")


asyncio.run(main())

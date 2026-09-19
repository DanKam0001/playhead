"""Watch what the Voice Agent websocket actually says.

The browser client was written from a prose description of the protocol, and the
mic never starts -- which points at session setup, not at the microphone. This
connects the same way the browser does and prints every frame, so the real
handshake settles it.
"""
import asyncio, json, os, urllib.request
from dotenv import load_dotenv
import websockets

load_dotenv()
KEY = os.environ["ASSEMBLYAI_API_KEY"]
AGENT_ID = os.getenv("ECHOREAD_AGENT_ID", "")


def mint():
    req = urllib.request.Request(
        "https://agents.assemblyai.com/v1/token?expires_in_seconds=120",
        headers={"Authorization": f"Bearer {KEY}"})
    return json.load(urllib.request.urlopen(req))["token"]


async def try_shape(label: str, payload: dict, listen_s: float = 6.0):
    token = mint()
    url = f"wss://agents.assemblyai.com/v1/ws?token={token}"
    print(f"\n=== {label} ===")
    print(f"  sending: {json.dumps(payload)[:150]}")
    try:
        async with websockets.connect(url, max_size=None) as ws:
            await ws.send(json.dumps(payload))
            end = asyncio.get_event_loop().time() + listen_s
            while asyncio.get_event_loop().time() < end:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=end - asyncio.get_event_loop().time())
                except asyncio.TimeoutError:
                    break
                m = json.loads(raw)
                t = m.get("type")
                keys = [k for k in m if k != "type"]
                extra = ""
                if t in ("error", "session.error"):
                    extra = " <- " + json.dumps(m)[:220]
                elif t == "session.ready":
                    extra = " <- KEYS: " + ",".join(keys)
                print(f"  [{t}] {keys}{extra}")
                if t in ("error", "session.error"):
                    break
    except Exception as e:
        print(f"  connection failed: {type(e).__name__}: {str(e)[:200]}")


async def main():
    print(f"agent: {AGENT_ID or '(none configured)'}")
    await try_shape("A: agent_id at top level (what the browser sends now)",
                    {"type": "session.update", "agent_id": AGENT_ID})
    await try_shape("B: agent_id nested under session",
                    {"type": "session.update", "session": {"agent_id": AGENT_ID}})
    await try_shape("C: no session.update at all (does it auto-start?)",
                    {"type": "ping"}, listen_s=4.0)


asyncio.run(main())

"""Can this account use the Voice Agent API? (LLM Gateway was gated, so check.)

Note the auth difference: Voice Agent takes `Authorization: Bearer <key>`,
while STT and LLM Gateway take the raw key with no prefix.
"""
import json, os, urllib.error, urllib.request
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["ASSEMBLYAI_API_KEY"]
BEARER = {"Authorization": f"Bearer {KEY}"}


def get(url, headers):
    req = urllib.request.Request(url, headers=headers)
    return json.load(urllib.request.urlopen(req))


def show(label, fn):
    try:
        print(f"{label:28} OK   {fn()}")
    except urllib.error.HTTPError as e:
        print(f"{label:28} HTTP {e.code}  {e.read().decode()[:130]}")
    except Exception as e:
        print(f"{label:28} FAIL {type(e).__name__}: {str(e)[:110]}")


if __name__ == "__main__":
    show("voices (Bearer)", lambda: f"{len(get('https://agents.assemblyai.com/v1/voices', BEARER))} entries"
         if isinstance(get('https://agents.assemblyai.com/v1/voices', BEARER), list)
         else str(get('https://agents.assemblyai.com/v1/voices', BEARER))[:120])
    show("voices (raw key)", lambda: str(get('https://agents.assemblyai.com/v1/voices',
                                             {"Authorization": KEY}))[:90])
    show("token mint", lambda: "token issued" if "token" in get(
        "https://agents.assemblyai.com/v1/token?expires_in_seconds=60", BEARER) else "no token field")
    show("stored agents list", lambda: str(get("https://agents.assemblyai.com/v1/agents", BEARER))[:120])

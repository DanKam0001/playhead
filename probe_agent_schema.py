"""Learn the stored-agent schema by walking the API's own validator.

The docs URL 404s, so the validator is the authoritative source.
"""
import json, os, urllib.error, urllib.request
from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["ASSEMBLYAI_API_KEY"]
URL = "https://agents.assemblyai.com/v1/agents"


def post(body):
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        return "OK", json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:600]


if __name__ == "__main__":
    for label, body in [
        ("empty", {}),
        ("name only", {"name": "echoread-probe"}),
        ("name+prompt", {"name": "echoread-probe", "system_prompt": "You help."}),
    ]:
        code, detail = post(body)
        print(f"--- {label}: {code}")
        print(json.dumps(detail, indent=1)[:700] if isinstance(detail, dict) else detail)
        print()

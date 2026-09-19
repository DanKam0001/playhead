"""Publish server/agent.json to AssemblyAI as a stored agent.

    python scripts/create_agent.py https://your-host.example.com

The agent definition lives on AssemblyAI's side, so the browser only ever sends
{agent_id} -- no prompt and no tool definitions travel to the client. The tool
URL must be publicly reachable, because AssemblyAI calls it from its servers:
for local development use a tunnel (ngrok / cloudflared), not localhost.
"""
import json, os, sys, urllib.error, urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
KEY = os.environ["ASSEMBLYAI_API_KEY"]
API = "https://agents.assemblyai.com/v1/agents"
SPEC = Path(__file__).resolve().parent.parent / "server" / "agent.json"


def call(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {url} -> {e.code}\n{e.read().decode()[:900]}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    base = sys.argv[1].rstrip("/")
    if base.startswith("http://localhost") or base.startswith("http://127."):
        print("WARNING: AssemblyAI calls the tool from its own servers, so a "
              "localhost URL will never be reachable. Use a tunnel.")

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for tool in spec.get("tools", []):
        tool["http"]["url"] = tool["http"]["url"].replace("__TOOL_BASE_URL__", base)
        print(f"  tool {tool['name']} -> {tool['http']['url']}")

    existing = call("GET", API).get("agents", [])
    match = next((a for a in existing if a.get("name") == spec["name"]), None)
    if match:
        # PUT, not PATCH: the API answers PATCH with 405. PUT merges the fields
        # sent, so sending the whole spec overwrites every field we define.
        agent = call("PUT", f"{API}/{match['id']}", spec)
        print(f"updated existing agent {match['id']}")
    else:
        agent = call("POST", API, spec)
        print(f"created agent {agent.get('id')}")

    agent_id = agent.get("id") or (match or {}).get("id")
    print(f"\nPLAYHEAD_AGENT_ID={agent_id}")
    print("Put that in .env, then start the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

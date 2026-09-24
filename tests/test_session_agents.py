"""Per-listener routing: every tool call must say whose playhead it is about.

AssemblyAI calls an HTTP tool anonymously, so the only way a call identifies
its listener is the session id pinned in the tool URL of that listener's own
agent. These tests hold that seam: the URLs carry the id, a tool reads it back,
two listeners stay apart, and a failed agent create still leaves a working one.
No keys, no network: the AssemblyAI calls are replaced.
"""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from server import main  # noqa: E402


@pytest.fixture
def created(monkeypatch):
    """Capture every agent the backend creates or deletes."""
    log = {"posted": [], "deleted": []}

    def fake(method, url, body=None):
        if method == "POST":
            log["posted"].append(body)
            return {"id": f"agent_{len(log['posted'])}"}
        if method == "DELETE":
            log["deleted"].append(url.rsplit("/", 1)[1])
        return {"agents": []}

    monkeypatch.setattr(main, "_agents_call", fake)
    return log


def test_session_agent_pins_the_session_in_every_tool_url(created):
    agent_id = main._session_agent("s123")
    spec = created["posted"][0]
    assert agent_id == "agent_1"
    assert spec["name"] == "playhead-session-s123"
    assert spec["tools"], "the agent must carry the tools"
    for tool in spec["tools"]:
        assert tool["http"]["url"].startswith(main.PUBLIC_URL + "/tools/")
        assert tool["http"]["url"].endswith("?session_id=s123")
    # The shared template is not mutated for the next listener.
    assert "__TOOL_BASE_URL__" in main._AGENT_SPEC["tools"][0]["http"]["url"]


def test_a_failed_create_falls_back_to_the_shared_agent(monkeypatch):
    def boom(*a, **k):
        raise OSError("AssemblyAI is down")

    monkeypatch.setattr(main, "_agents_call", boom)
    assert main._session_agent("s9") == main.AGENT_ID


def test_two_listeners_get_their_own_passages(created):
    client = TestClient(main.app)
    # Two people, different places in the same book. B reported last, which
    # is exactly the case the old "freshest playhead" fallback got wrong for A.
    client.post("/api/playhead", json={"session_id": "sA", "seconds": 60.0})
    client.post("/api/playhead", json={"session_id": "sB", "seconds": 900.0})
    a = client.post("/tools/passage_at_playhead?session_id=sA", json={}).json()
    b = client.post("/tools/passage_at_playhead?session_id=sB", json={}).json()
    assert a["ok"] and b["ok"]
    assert a["playhead_seconds"] == 60.0
    assert b["playhead_seconds"] == 900.0


def test_answer_length_is_the_first_thing_the_model_reads(created):
    client = TestClient(main.app)
    client.post("/api/playhead", json={"session_id": "sS", "seconds": 300.0, "brevity": "short"})
    msg = client.post("/tools/passage_at_playhead?session_id=sS", json={}).json()["message"]
    assert msg.startswith("ANSWER LENGTH: ONE sentence")


def test_ending_a_session_deletes_its_agent(created):
    client = TestClient(main.app)
    main._session_agent("sEnd")
    client.post("/api/session/end", json={"session_id": "sEnd"})
    assert created["deleted"] == ["agent_1"]


def test_lookback_is_the_listeners_choice_within_the_cap(created):
    client = TestClient(main.app)
    client.post("/api/playhead", json={"session_id": "sL", "seconds": 900.0, "lookback": 300})
    wide = client.post("/tools/passage_at_playhead?session_id=sL", json={}).json()["message"]
    client.post("/api/playhead", json={"session_id": "sL", "seconds": 900.0, "lookback": 30})
    narrow = client.post("/tools/passage_at_playhead?session_id=sL", json={}).json()["message"]
    assert "the last 300 seconds" in wide and "the last 30 seconds" in narrow
    assert len(wide) > len(narrow)
    # A tampered page cannot ask for more than the cap.
    client.post("/api/playhead", json={"session_id": "sL", "seconds": 900.0, "lookback": 99999})
    assert "the last 300 seconds" in client.post("/tools/passage_at_playhead?session_id=sL", json={}).json()["message"]


def test_a_long_passage_is_trimmed_to_fit_the_tool_response_cap():
    text = " ".join(f"word{i}" for i in range(5000))
    out = main._trim_passage(text)
    assert len(out) <= main.PASSAGE_BUDGET_CHARS + 3
    assert out.endswith("word4999"), "the newest words are what 'that' points at"

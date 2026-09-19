"""Where the listener's playback position lives.

This exists because of how the architecture splits. The browser knows where the
audiobook is; AssemblyAI calls the tool from its own servers and does not. So
the position has to be handed over out of band -- and on a serverless host the
two requests may not even reach the same instance, which makes a plain dict in
process memory silently wrong: the browser writes to one lambda, the tool reads
from another and finds nothing.

Redis when it is configured, process memory when it isn't, so local development
needs no infrastructure.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Protocol

# Playhead entries expire rather than accumulating. Comfortably longer than a
# session, short enough that abandoned ones clear themselves out.
TTL_SECONDS = 3600
LATEST_KEY = "echoread:latest"
SEEK_LATEST_KEY = "echoread:seek:latest"
CTX_LATEST_KEY = "echoread:ctx:latest"
# A requested jump is consumed once. Leaving it set would drag the
# listener back to the same spot on every heartbeat.
SEEK_TTL_SECONDS = 30


class PlayheadStore(Protocol):
    def set(self, session_id: str, seconds: float) -> None: ...
    def get(self, session_id: Optional[str]) -> Optional[float]: ...
    def request_seek(self, session_id: Optional[str], seconds: float) -> None: ...
    def take_seek(self, session_id: Optional[str]) -> Optional[float]: ...
    def set_context(self, session_id: Optional[str], text: str) -> None: ...
    def get_context(self, session_id: Optional[str]) -> Optional[str]: ...


class MemoryStore:
    """Correct on a single long-lived process. Used for local development."""

    def __init__(self):
        self._d: dict[str, tuple[float, float]] = {}
        self._seeks: dict[str, tuple[float, float]] = {}
        self._ctx: dict[str, str] = {}

    def set(self, session_id: str, seconds: float) -> None:
        self._prune()
        self._d[session_id] = (seconds, time.time())

    def get(self, session_id: Optional[str]) -> Optional[float]:
        self._prune()
        if session_id and session_id in self._d:
            return self._d[session_id][0]
        if not self._d:
            return None
        # The agent does not always pass the session id through. One listener
        # at a time is the normal case here, so fall back to the freshest.
        return max(self._d.values(), key=lambda v: v[1])[0]

    def _prune(self) -> None:
        cutoff = time.time() - TTL_SECONDS
        for k in [k for k, v in self._d.items() if v[1] < cutoff]:
            self._d.pop(k, None)

    def request_seek(self, session_id: Optional[str], seconds: float) -> None:
        self._seeks[session_id or "latest"] = (seconds, time.time())

    def take_seek(self, session_id: Optional[str]) -> Optional[float]:
        for key in ([session_id] if session_id else []) + ["latest"]:
            hit = self._seeks.pop(key, None)
            if hit and time.time() - hit[1] < SEEK_TTL_SECONDS:
                self._seeks.pop("latest", None)
                return hit[0]
        return None

    def set_context(self, session_id: Optional[str], text: str) -> None:
        self._ctx[session_id or "latest"] = text

    def get_context(self, session_id: Optional[str]) -> Optional[str]:
        for key in ([session_id] if session_id else []) + ["latest"]:
            if key in self._ctx:
                return self._ctx[key]
        return None

    @property
    def kind(self) -> str:
        return "memory"


class RedisStore:
    """Upstash REST. Works across lambdas, which process memory cannot."""

    def __init__(self, url: str, token: str):
        self._url = url.rstrip("/")
        self._auth = {"Authorization": f"Bearer {token}"}

    def _cmd(self, *parts: str):
        req = urllib.request.Request(
            f"{self._url}/{'/'.join(urllib.parse.quote(str(p), safe='') for p in parts)}",
            headers=self._auth)
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.load(r).get("result")

    def set(self, session_id: str, seconds: float) -> None:
        value = str(seconds)
        self._cmd("set", f"echoread:ph:{session_id}", value, "EX", str(TTL_SECONDS))
        # Mirror to a well-known key so a tool call with no session id still
        # has something correct to read.
        self._cmd("set", LATEST_KEY, value, "EX", str(TTL_SECONDS))

    def get(self, session_id: Optional[str]) -> Optional[float]:
        for key in ([f"echoread:ph:{session_id}"] if session_id else []) + [LATEST_KEY]:
            try:
                raw = self._cmd("get", key)
            except Exception as exc:
                print(f"[store] redis get failed: {exc}")
                return None
            if raw is not None:
                try:
                    return float(raw)
                except ValueError:
                    continue
        return None

    def request_seek(self, session_id: Optional[str], seconds: float) -> None:
        value = str(seconds)
        keys = ([f"echoread:seek:{session_id}"] if session_id else []) + [SEEK_LATEST_KEY]
        for key in keys:
            try:
                self._cmd("set", key, value, "EX", str(SEEK_TTL_SECONDS))
            except Exception as exc:
                print(f"[store] redis seek set failed: {exc}")

    def take_seek(self, session_id: Optional[str]) -> Optional[float]:
        # GETDEL, so the jump happens once and the listener keeps control
        # afterwards.
        for key in ([f"echoread:seek:{session_id}"] if session_id else []) + [SEEK_LATEST_KEY]:
            try:
                raw = self._cmd("getdel", key)
            except Exception as exc:
                print(f"[store] redis seek take failed: {exc}")
                return None
            if raw is not None:
                try:
                    seconds = float(raw)
                except ValueError:
                    continue
                if key != SEEK_LATEST_KEY:
                    try:
                        self._cmd("del", SEEK_LATEST_KEY)
                    except Exception:
                        pass
                return seconds
        return None

    def set_context(self, session_id: Optional[str], text: str) -> None:
        for key in ([f"echoread:ctx:{session_id}"] if session_id else []) + [CTX_LATEST_KEY]:
            try:
                self._cmd("set", key, text, "EX", str(TTL_SECONDS))
            except Exception as exc:
                print(f"[store] redis context set failed: {exc}")

    def get_context(self, session_id: Optional[str]) -> Optional[str]:
        for key in ([f"echoread:ctx:{session_id}"] if session_id else []) + [CTX_LATEST_KEY]:
            try:
                raw = self._cmd("get", key)
            except Exception as exc:
                print(f"[store] redis context get failed: {exc}")
                return None
            if raw:
                return str(raw)
        return None

    @property
    def kind(self) -> str:
        return "redis"


def build_store() -> PlayheadStore:
    url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    if url and token:
        return RedisStore(url, token)
    print("[store] no Upstash config - using process memory. "
          "Correct locally; WRONG on a serverless host.")
    return MemoryStore()

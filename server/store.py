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
LATEST_KEY = "playhead:latest"
SEEK_LATEST_KEY = "playhead:seek:latest"
CTX_LATEST_KEY = "playhead:ctx:latest"
# A requested jump is consumed once. Leaving it set would drag the
# listener back to the same spot on every heartbeat.
SEEK_TTL_SECONDS = 30


BOOK_LATEST_KEY = "playhead:bk:latest"
# A book someone added is theirs for a month. Long enough to come back to,
# short enough that the store does not grow forever on a free tier.
BOOK_TTL_SECONDS = 30 * 24 * 3600


class PlayheadStore(Protocol):
    def set(self, session_id: str, seconds: float) -> None: ...
    def get(self, session_id: Optional[str]) -> Optional[float]: ...
    def request_seek(self, session_id: Optional[str], seconds: float) -> None: ...
    def take_seek(self, session_id: Optional[str]) -> Optional[float]: ...
    def set_context(self, session_id: Optional[str], text: str) -> None: ...
    def get_context(self, session_id: Optional[str]) -> Optional[str]: ...
    # Which book this listener is on. Same out-of-band seam as the playhead:
    # the browser knows, the agent's tool call does not.
    def set_book(self, session_id: Optional[str], book_id: str) -> None: ...
    def get_book(self, session_id: Optional[str]) -> Optional[str]: ...
    # A general key/value surface, used by server/books.py to hold user-added
    # books and their indexes. Values can be a few hundred KB.
    def kv_set(self, key: str, value: str, ttl: Optional[int] = None) -> None: ...
    def kv_get(self, key: str) -> Optional[str]: ...
    def kv_push(self, key: str, value: str) -> None: ...
    def kv_list(self, key: str, n: int = 50) -> list: ...
    def kv_incr(self, key: str, ttl: int) -> int: ...
    def kv_mget(self, keys: list) -> list: ...


class MemoryStore:
    """Correct on a single long-lived process. Used for local development."""

    def __init__(self):
        self._d: dict[str, tuple[float, float]] = {}
        self._seeks: dict[str, tuple[float, float]] = {}
        self._ctx: dict[str, str] = {}
        self._books: dict[str, str] = {}
        self._kv: dict[str, str] = {}
        self._lists: dict[str, list] = {}
        self._counts: dict[str, int] = {}

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

    def set_book(self, session_id: Optional[str], book_id: str) -> None:
        self._books[session_id or "latest"] = book_id
        self._books["latest"] = book_id

    def get_book(self, session_id: Optional[str]) -> Optional[str]:
        for key in ([session_id] if session_id else []) + ["latest"]:
            if key in self._books:
                return self._books[key]
        return None

    def kv_set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        self._kv[key] = value

    def kv_get(self, key: str) -> Optional[str]:
        return self._kv.get(key)

    def kv_push(self, key: str, value: str) -> None:
        self._lists.setdefault(key, []).insert(0, value)

    def kv_list(self, key: str, n: int = 50) -> list:
        return list(self._lists.get(key, []))[:n]

    def kv_incr(self, key: str, ttl: int) -> int:
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def kv_mget(self, keys: list) -> list:
        return [self._kv.get(k) for k in keys]

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

    def _post(self, *parts, timeout: int = 10):
        """Same commands, sent as a JSON body instead of a URL path.

        An index shard is a few hundred KB of base64. That does not fit in a
        URL, so anything that carries a payload goes through here.
        """
        body = json.dumps([str(p) for p in parts]).encode("utf-8")
        req = urllib.request.Request(
            self._url, data=body,
            headers={**self._auth, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.load(r)
        # Upstash reports a rejected command in the body, not the status line.
        # Swallowing that would leave a book marked ready with no vectors
        # behind it, which fails later and somewhere less obvious.
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"redis: {payload['error']}")
        return payload.get("result") if isinstance(payload, dict) else None

    def set(self, session_id: str, seconds: float) -> None:
        value = str(seconds)
        self._cmd("set", f"playhead:ph:{session_id}", value, "EX", str(TTL_SECONDS))
        # Mirror to a well-known key so a tool call with no session id still
        # has something correct to read.
        self._cmd("set", LATEST_KEY, value, "EX", str(TTL_SECONDS))

    def get(self, session_id: Optional[str]) -> Optional[float]:
        for key in ([f"playhead:ph:{session_id}"] if session_id else []) + [LATEST_KEY]:
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
        keys = ([f"playhead:seek:{session_id}"] if session_id else []) + [SEEK_LATEST_KEY]
        for key in keys:
            try:
                self._cmd("set", key, value, "EX", str(SEEK_TTL_SECONDS))
            except Exception as exc:
                print(f"[store] redis seek set failed: {exc}")

    def take_seek(self, session_id: Optional[str]) -> Optional[float]:
        # GETDEL, so the jump happens once and the listener keeps control
        # afterwards.
        for key in ([f"playhead:seek:{session_id}"] if session_id else []) + [SEEK_LATEST_KEY]:
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
        for key in ([f"playhead:ctx:{session_id}"] if session_id else []) + [CTX_LATEST_KEY]:
            try:
                self._cmd("set", key, text, "EX", str(TTL_SECONDS))
            except Exception as exc:
                print(f"[store] redis context set failed: {exc}")

    def get_context(self, session_id: Optional[str]) -> Optional[str]:
        for key in ([f"playhead:ctx:{session_id}"] if session_id else []) + [CTX_LATEST_KEY]:
            try:
                raw = self._cmd("get", key)
            except Exception as exc:
                print(f"[store] redis context get failed: {exc}")
                return None
            if raw:
                return str(raw)
        return None

    def set_book(self, session_id: Optional[str], book_id: str) -> None:
        keys = ([f"playhead:bk:{session_id}"] if session_id else []) + [BOOK_LATEST_KEY]
        for key in keys:
            try:
                self._cmd("set", key, book_id, "EX", str(TTL_SECONDS))
            except Exception as exc:
                print(f"[store] redis book set failed: {exc}")

    def get_book(self, session_id: Optional[str]) -> Optional[str]:
        for key in ([f"playhead:bk:{session_id}"] if session_id else []) + [BOOK_LATEST_KEY]:
            try:
                raw = self._cmd("get", key)
            except Exception as exc:
                print(f"[store] redis book get failed: {exc}")
                return None
            if raw:
                return str(raw)
        return None

    def kv_set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        args = ["set", key, value] + (["EX", str(ttl)] if ttl else [])
        self._post(*args)

    def kv_get(self, key: str) -> Optional[str]:
        raw = self._post("get", key)
        return str(raw) if raw is not None else None

    def kv_push(self, key: str, value: str) -> None:
        self._post("lpush", key, value)
        # Keep the shelf bounded; nobody scrolls past fifty books.
        self._post("ltrim", key, "0", "49")

    def kv_list(self, key: str, n: int = 50) -> list:
        raw = self._post("lrange", key, "0", str(n - 1))
        return list(raw or [])

    def kv_incr(self, key: str, ttl: int) -> int:
        """Count something in a window. The EX only lands on the first hit,
        so the window is fixed from the first request rather than sliding
        forward forever as more arrive."""
        n = int(self._post("incr", key) or 0)
        if n == 1:
            self._post("expire", key, str(ttl))
        return n

    def kv_mget(self, keys: list) -> list:
        """Every key in one round trip.

        A ten-hour book is eighteen index shards. Fetched one at a time that is
        eighteen sequential network calls inside a function that is killed at
        ten seconds; as one MGET it is one.
        """
        if not keys:
            return []
        raw = self._post("mget", *keys, timeout=20)
        return list(raw or [None] * len(keys))

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

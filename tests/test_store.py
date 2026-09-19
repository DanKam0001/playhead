"""The Redis layer: command encoding, big payloads, and the seek handshake.

MemoryStore cannot cover any of this. A dict does not care how a command is
encoded, and a 200 KB index shard is exactly the kind of value that works in
process memory and fails over HTTP.
"""
import pytest


def test_large_value_survives_a_round_trip(store):
    """An index shard is ~205 KB of base64, far past what fits in a URL."""
    payload = "x" * 210_000
    store.kv_set("playhead:test:big", payload, ttl=60)
    assert store.kv_get("playhead:test:big") == payload


def test_mget_returns_values_in_order(store):
    """Shards are reassembled positionally, so order is not cosmetic."""
    for i in range(5):
        store.kv_set(f"playhead:test:k{i}", f"value-{i}")
    keys = [f"playhead:test:k{i}" for i in range(5)]
    assert store.kv_mget(keys) == [f"value-{i}" for i in range(5)]


def test_mget_of_nothing_is_empty(store):
    assert store.kv_mget([]) == []


def test_missing_keys_come_back_as_none(store):
    store.kv_set("playhead:test:present", "here")
    assert store.kv_mget(["playhead:test:present", "playhead:test:absent"]) == ["here", None]


def test_shelf_is_newest_first(store):
    for name in ["oldest", "middle", "newest"]:
        store.kv_push("playhead:test:shelf", name)
    assert store.kv_list("playhead:test:shelf", 10) == ["newest", "middle", "oldest"]


def test_playhead_round_trip(store):
    store.set("s1", 123.4)
    assert store.get("s1") == pytest.approx(123.4)


def test_playhead_falls_back_to_the_freshest(store):
    """The agent does not reliably pass a session id, and one listener at a
    time is the normal case, so an unknown id reads the latest position."""
    store.set("s1", 99.0)
    assert store.get("unknown-session") == pytest.approx(99.0)


def test_a_seek_is_consumed_exactly_once(store):
    """Left set, it would drag the listener back on every heartbeat."""
    store.request_seek("s1", 42.0)
    assert store.take_seek("s1") == pytest.approx(42.0)
    assert store.take_seek("s1") is None


def test_seek_reaches_a_session_that_did_not_request_it(store):
    """go_to_topic often arrives with no session id, so it writes a latest
    key too; the browser must still collect it."""
    store.request_seek(None, 7.5)
    assert store.take_seek("some-session") == pytest.approx(7.5)


def test_context_and_book_binding(store):
    store.set_context("s1", "- at 1:00, they asked: what is entropy")
    assert "entropy" in store.get_context("s1")
    store.set_book("s1", "book-abc")
    assert store.get_book("s1") == "book-abc"


def test_rate_counter_increments(store):
    assert [store.kv_incr("playhead:test:rate", 3600) for _ in range(3)] == [1, 2, 3]


def test_a_redis_error_is_raised_not_swallowed(store):
    """Upstash reports a rejected command in the body with a 200 status.
    Swallowing that would leave a book marked ready with no vectors behind it.
    """
    with pytest.raises(Exception):
        store._post("definitely-not-a-command", "x")

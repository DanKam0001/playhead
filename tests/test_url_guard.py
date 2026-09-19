"""The books endpoint is unauthenticated by design, so the guards are the API.

We make the HEAD request ourselves, which means an unvalidated link is our own
server-side request forgery rather than somebody else's problem.
"""
import pytest

from server import books


@pytest.mark.parametrize("url, why", [
    ("ftp://example.com/a.mp3", "scheme"),
    ("file:///etc/passwd", "file scheme"),
    ("http://127.0.0.1/a.mp3", "loopback"),
    ("http://localhost:8199/a.mp3", "localhost by name"),
    ("http://169.254.169.254/latest/meta-data/", "cloud metadata"),
    ("http://10.0.0.5/a.mp3", "private range"),
    ("http://192.168.1.1/a.mp3", "private range"),
    ("http://[::1]/a.mp3", "ipv6 loopback"),
    ("https:///no-host.mp3", "no host"),
])
def test_refused(url, why):
    with pytest.raises(books.RejectedURL):
        books.check_source(url)


def test_a_wrong_extension_is_refused_when_head_tells_us_nothing(monkeypatch):
    """Plenty of hosts refuse HEAD. With nothing known about the file, the
    path is the only signal left, and it catches the usual wrong paste."""
    monkeypatch.setattr(books, "_is_public_host", lambda host: True)

    def explode(*a, **k):
        raise OSError("no HEAD for you")
    monkeypatch.setattr(books.urllib.request, "build_opener",
                        lambda *a: type("O", (), {"open": explode})())

    with pytest.raises(books.RejectedURL, match="html"):
        books.check_source("https://example.com/index.html")


def test_no_extension_is_allowed_when_head_tells_us_nothing(monkeypatch):
    """CDN and stream URLs often have no extension. Refusing those would be
    worse than letting the transcriber decide."""
    monkeypatch.setattr(books, "_is_public_host", lambda host: True)

    def explode(*a, **k):
        raise OSError("no HEAD for you")
    monkeypatch.setattr(books.urllib.request, "build_opener",
                        lambda *a: type("O", (), {"open": explode})())

    assert books.check_source("https://cdn.example.com/stream/abc123")["bytes"] == 0


def test_a_size_ceiling_of_zero_means_no_ceiling(monkeypatch):
    """0 is how the setting is switched off, so it must not read as 'reject
    everything larger than nothing'."""
    monkeypatch.setattr(books, "_is_public_host", lambda host: True)

    class Response:
        headers = {"Content-Type": "audio/mpeg", "Content-Length": "999999999"}
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(books.urllib.request, "build_opener",
                        lambda *a: type("O", (), {"open": lambda s, r, timeout=0: Response()})())
    assert books.check_source("https://example.com/huge.mp3", max_bytes=0)["bytes"] > 0


def test_public_hosts_resolve(monkeypatch):
    assert books._is_public_host("example.com") is True
    assert books._is_public_host("127.0.0.1") is False
    assert books._is_public_host("nonexistent.invalid") is False

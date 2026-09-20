"""Transcribe and index the featured books, once, against a real store.

    python scripts/seed_featured.py            # index anything missing
    python scripts/seed_featured.py --force    # rebuild them all

These are the ready-made books every visitor sees. They are seeded from here
rather than through the API because the ids have to be fixed -- FEATURED_IDS in
server/main.py names them -- and an endpoint that lets a caller choose its own
id is an endpoint that lets a caller overwrite someone else's book.

Needs the production credentials in the environment:

    ASSEMBLYAI_API_KEY, GEMINI_API_KEY,
    UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN

Without the Upstash pair the store falls back to process memory and this seeds
a dictionary that disappears when the script exits, so it refuses to run.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from server import books                      # noqa: E402
from server.store import build_store          # noqa: E402

# id -> what to index. The ids match FEATURED_IDS in server/main.py; changing
# one here without changing it there quietly drops the book off the shelf.
FEATURED = {
    # The whole book, all fifteen chapters, laid end to end on one
    # timeline -- nearly five hours. This is the one that shows the
    # difference between indexing a chapter and indexing a book.
    "featured-russell-problems-full": {
        "title": "Russell - The Problems of Philosophy (complete)",
        "audio_urls": [

            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_01_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_02_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_03_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_04_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_05_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_06_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_07_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_08_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_09_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_10_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_11_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_12_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_13_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_14_russell_64kb.mp3",
            "https://archive.org/download/problems_of_philosophy_librivox/problemsofphilosophy_15_russell_64kb.mp3"
],
    },
    "featured-bennett-24hours-1": {
        "title": "Bennett - How to Live on 24 Hours a Day, ch. 1",
        "audio_url": "https://archive.org/download/howtoliveon24hoursaday_1504_librivox/"
                     "howtoliveon24hoursaday_01_bennett_64kb.mp3",
    },
    "featured-wittgenstein-tractatus-1": {
        "title": "Wittgenstein - Tractatus Logico-Philosophicus, part 1",
        "audio_url": "https://archive.org/download/tractatus_ge_librivox/"
                     "tractatus_1_wittgenstein_64kb.mp3",
    },
}


def embedder():
    from playhead.brain import GeminiEmbedder
    key = os.environ["GEMINI_API_KEY"]
    return GeminiEmbedder(key)


def seed(store, book_id: str, spec: dict, aai_key: str, force: bool) -> str:
    existing = books.load(store, book_id)
    if existing and existing.status == "ready" and not force:
        return f"already ready ({existing.n_chunks} chunks)"

    urls = spec.get("audio_urls") or [spec["audio_url"]]
    for u in urls:
        books.check_source(u, 0)
    rec = books.create(store, spec["title"], urls, aai_key,
                       client_id="", book_id=book_id)
    emb = embedder()

    # Same bounded-slice loop the browser drives, just without a browser.
    started = time.time()
    while rec.status in ("transcribing", "indexing"):
        if time.time() - started > 2700:
            return "gave up after 45 minutes"
        time.sleep(8)
        rec = books.advance(store, rec, aai_key, emb)
        where = f"  part {rec.part_index}/{len(rec.parts)}" if len(rec.parts) > 1 else ""
        print(f"      {rec.status} {rec.progress}%{where}   ", end="\r", flush=True)
    print(" " * 40, end="\r")
    return (f"ready: {rec.n_chunks} chunks, {rec.duration / 60:.0f} min"
            if rec.status == "ready" else f"FAILED: {rec.error}")


def main() -> int:
    if not (os.getenv("UPSTASH_REDIS_REST_URL") and os.getenv("UPSTASH_REDIS_REST_TOKEN")):
        print("Refusing to run: no Upstash configured, so this would seed a dict "
              "that vanishes when the script exits.\n"
              "Export UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN first.")
        return 2
    aai_key = os.environ["ASSEMBLYAI_API_KEY"]
    force = "--force" in sys.argv
    store = build_store()
    print(f"store: {store.kind}\n")

    failed = 0
    for book_id, spec in FEATURED.items():
        print(f"  {spec['title']}")
        try:
            outcome = seed(store, book_id, spec, aai_key, force)
        except Exception as exc:
            outcome, failed = f"FAILED: {exc}", failed + 1
        if outcome.startswith(("FAILED", "gave up")):
            failed += 1
        print(f"      {outcome}\n")

    print("Add these to FEATURED_IDS in server/main.py:")
    for book_id in FEATURED:
        print(f'    "{book_id}",')
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

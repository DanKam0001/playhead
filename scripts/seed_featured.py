"""Transcribe and index the featured shelf, once, against a real store.

    python scripts/seed_featured.py                  # index anything missing
    python scripts/seed_featured.py --only walden    # just the ones that match
    python scripts/seed_featured.py --force          # rebuild them all
    python scripts/seed_featured.py --list           # show the shelf, seed nothing

These are the ready-made books every visitor sees. They are seeded from here
rather than through the API because the ids have to be fixed -- FEATURED_IDS in
server/main.py names them -- and an endpoint that lets a caller choose its own
id is an endpoint that lets a caller overwrite someone else's book.

Each entry names an **archive.org identifier**, not a list of URLs. The file
list is fetched from archive.org's metadata API at run time and sorted, because
these books run to 117 files and a hand-typed list is 117 chances to put chapter
40 before chapter 4. Part order is the one mistake nothing downstream can
detect: get it wrong and every timestamp in the book is wrong, the index builds
cleanly, and the answers are confidently about the wrong passage.

Books are seeded shortest first, so the shelf is usable early and the longest
book is the only thing still running at the end.

Needs the production credentials in the environment:

    ASSEMBLYAI_API_KEY, GEMINI_API_KEY,
    UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN

Without the Upstash pair the store falls back to process memory and this seeds
a dictionary that disappears when the script exits, so it refuses to run.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# The shelf has to outlive a judging window that opens after submissions close.
# Set before importing books, which reads it at import time.
os.environ.setdefault("PLAYHEAD_BOOK_TTL_DAYS", "180")

from server import books                      # noqa: E402
from server.store import build_store          # noqa: E402

# id -> what to index, shortest first. The ids match FEATURED_IDS in
# server/main.py; changing one here without changing it there quietly drops the
# book off the shelf.
#
# All public domain, all LibriVox. Durations are from archive.org and are here
# as a sanity check on the fetched file list -- if the total comes back wildly
# different, the identifier is serving something other than what we think.
FEATURED = [
    ("featured-allen-as-a-man-thinketh", {
        "title": "Allen - As a Man Thinketh (complete)",
        "identifier": "as_a_man_thinketh_mc_librivox", "hours": 0.90}),
    ("featured-bennett-24hours", {
        "title": "Bennett - How to Live on 24 Hours a Day (complete)",
        "identifier": "howtoliveon24hoursaday_1504_librivox", "hours": 1.61}),
    ("featured-wittgenstein-tractatus", {
        "title": "Wittgenstein - Tractatus Logico-Philosophicus (complete)",
        "identifier": "tractatus_ge_librivox", "hours": 4.24}),
    ("featured-russell-problems-full", {
        "title": "Russell - The Problems of Philosophy (complete)",
        "identifier": "problems_of_philosophy_librivox", "hours": 4.84}),
    ("featured-wilde-dorian-gray", {
        "title": "Wilde - The Picture of Dorian Gray (complete)",
        "identifier": "dorian_gray_librivox", "hours": 6.19}),
    ("featured-thompson-calculus", {
        "title": "Thompson - Calculus Made Easy (complete)",
        "identifier": "calculus_made_easy_1608_librivox", "hours": 10.12}),
    # Spanish. Kept deliberately: it is the proof that "any audiobook" is not
    # quietly "any English audiobook". Needs language_detection in books.py.
    ("featured-shelley-frankenstein-es", {
        "title": "Shelley - Frankenstein (complete, Spanish)",
        "identifier": "frankenstein_el_moderno_prometeo_1611_librivox",
        "hours": 11.55}),
    ("featured-hume-treatise", {
        "title": "Hume - A Treatise of Human Nature, vol. 1 (complete)",
        "identifier": "treatise_human_nature_1_0911_librivox", "hours": 13.95}),
    ("featured-thoreau-walden", {
        "title": "Thoreau - Walden (complete)",
        "identifier": "walden_librivox", "hours": 14.30}),
    ("featured-locke-understanding", {
        "title": "Locke - An Essay Concerning Human Understanding, vol. 2",
        "identifier": "essay_concerning_human_understanding2_1904_librivox",
        "hours": 14.66}),
    ("featured-dostoevsky-crime-punishment", {
        "title": "Dostoevsky - Crime and Punishment (complete)",
        "identifier": "crime_and_punishment_0902_librivox", "hours": 23.44}),
    # 117 files, 54 hours, one timeline. Last on purpose: it is a third of the
    # total transcription cost and the only book long enough that failing it
    # late still leaves a full shelf.
    ("featured-dumas-monte-cristo", {
        "title": "Dumas - The Count of Monte Cristo (complete)",
        "identifier": "count_montecristo_1308_librivox", "hours": 54.27}),
]


def duration_seconds(value) -> float:
    """archive.org reports length as seconds, or as MM:SS, or as HH:MM:SS."""
    if not value:
        return 0.0
    text = str(value)
    if ":" in text:
        parts = [float(p) for p in text.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    try:
        return float(text)
    except ValueError:
        return 0.0


def part_urls(identifier: str):
    """Every audio file in an archive.org item, in reading order.

    Sorted by filename, which is correct here because LibriVox zero-pads to a
    fixed width within an item (001..117, not 1..117). That is checked rather
    than assumed: a run of names whose numbers are not consecutive means the
    padding is inconsistent and the sort is lying.
    """
    url = f"https://archive.org/metadata/{identifier}"
    with urllib.request.urlopen(url, timeout=60) as response:
        meta = json.load(response)
    files = [f for f in meta.get("files", [])
             if f.get("name", "").endswith("_64kb.mp3")]
    if not files:
        files = [f for f in meta.get("files", [])
                 if f.get("name", "").endswith(".mp3")]
    if not files:
        raise RuntimeError(f"{identifier}: no audio files in the item")
    files.sort(key=lambda f: f["name"])
    total = sum(duration_seconds(f.get("length")) for f in files)
    urls = [f"https://archive.org/download/{identifier}/{f['name']}"
            for f in files]
    return urls, total


def embedder():
    from playhead.brain import GeminiEmbedder
    key = os.environ["GEMINI_API_KEY"]
    return GeminiEmbedder(key)


def seed(store, book_id: str, spec: dict, aai_key: str, force: bool) -> str:
    existing = books.load(store, book_id)
    if existing and existing.status == "ready" and not force:
        return (f"already ready ({existing.n_chunks} chunks, "
                f"{existing.duration / 3600:.1f} h)")

    # A book that failed partway is worth repairing rather than rebuilding.
    # It failed on ONE part; every part before it is transcribed and absorbed,
    # and re-running from scratch pays for all of them again. Clear the bad
    # part's job so it is resubmitted, put the record back in progress, and
    # carry on from where it stopped.
    # `part_index > 0` was the wrong test and it cost real money: Dorian Gray
    # had absorbed nothing but had NINE transcripts already bought and
    # waiting, so it failed this check and was rebuilt from scratch, paying
    # for all nine a second time. What makes a book repairable is having any
    # work to salvage at all -- absorbed parts or submitted jobs.
    if (existing and existing.status == "failed" and not force
            and (existing.part_index > 0
                 or any(p.get("transcript_id") for p in existing.parts))):
        rec = existing
        # Clear only the jobs that actually failed.
        #
        # SUBMIT_BATCH queues eight parts ahead of absorption, so when a book
        # dies at part 14 the parts after it have often already transcribed --
        # and been paid for. Those transcripts are still sitting on
        # AssemblyAI's side and are free to collect. Blindly resubmitting the
        # tail would buy all of them a second time; leaving a *failed* job in
        # place would fail the book again the moment it reached that part.
        # So: ask, and only reset what is genuinely broken.
        cleared = kept = 0
        for part in rec.parts[rec.part_index:]:
            tid = part.get("transcript_id")
            if not tid:
                continue
            try:
                state = books._aai(f"/transcript/{tid}", aai_key).get("status")
            except Exception:
                state = "error"
            if state == "error":
                part["transcript_id"] = ""
                part["submitted_at"] = 0
                part["resubmits"] = 0
                cleared += 1
            else:
                kept += 1
        rec.status, rec.error, rec.transient = "transcribing", "", 0
        books.save(store, rec)
        print(f"      repairing: failed at part {rec.part_index + 1}/{len(rec.parts)}; "
              f"keeping {rec.part_index} absorbed + {kept} already-paid transcripts, "
              f"resubmitting {cleared}")
    # Resume rather than restart. These runs are hours long; losing a
    # half-transcribed 117-part book to a dropped connection and paying for it
    # twice is the failure this guards against.
    elif existing and existing.status in ("transcribing", "indexing") and not force:
        rec = existing
        done = sum(1 for p in rec.parts if p.get("done"))
        print(f"      resuming at part {done}/{len(rec.parts)}")
    else:
        urls, total = part_urls(spec["identifier"])
        claimed = spec.get("hours", 0)
        if claimed and abs(total / 3600 - claimed) > max(0.5, claimed * 0.1):
            return (f"REFUSED: expected ~{claimed:.1f} h, the item serves "
                    f"{total / 3600:.1f} h across {len(urls)} files")
        print(f"      {len(urls)} files, {total / 3600:.2f} h")
        books.check_source(urls[0], 0)
        rec = books.create(store, spec["title"], urls, aai_key,
                           client_id="", book_id=book_id)

    emb = embedder()

    # The same bounded-slice loop the browser drives, just without a browser.
    # Budget scales with the book: a 54-hour book cannot finish in the 45
    # minutes that was right for a single chapter.
    budget = max(2700, len(rec.parts) * 120)
    started = time.time()
    last = ""
    while rec.status in ("transcribing", "indexing"):
        if time.time() - started > budget:
            return (f"gave up after {budget / 60:.0f} minutes at "
                    f"part {rec.part_index}/{len(rec.parts)} "
                    f"(re-run to resume)")
        time.sleep(5)
        try:
            rec = books.advance(store, rec, aai_key, emb)
        except Exception as exc:
            # One transient archive.org 503 should not cost an hour of work.
            print(f"\n      advance failed, retrying in 30s: {exc}")
            time.sleep(30)
            rec = books.load(store, book_id)
            continue
        where = (f"  part {rec.part_index}/{len(rec.parts)}"
                 if len(rec.parts) > 1 else "")
        mins = (time.time() - started) / 60
        last = f"      {rec.status} {rec.progress}%{where}  {mins:.0f}m   "
        print(last, end="\r", flush=True)
    print(" " * max(40, len(last)), end="\r")
    if rec.status != "ready":
        return f"FAILED: {rec.error}"
    return (f"ready: {rec.n_chunks} chunks, {rec.duration / 3600:.2f} h, "
            f"{len(rec.parts)} files, {(time.time() - started) / 60:.0f} min")


def main() -> int:
    # Comma-separated, because the fast way to seed this shelf is several
    # processes over disjoint slices of it rather than one long serial run.
    only = []
    if "--only" in sys.argv:
        only = [p.strip().lower()
                for p in sys.argv[sys.argv.index("--only") + 1].split(",")
                if p.strip()]
    shelf = [(i, s) for i, s in FEATURED
             if not only or any(p in i.lower() or p in s["title"].lower()
                                for p in only)]
    if only and not shelf:
        print(f"nothing on the shelf matches {only}")
        return 2

    if "--list" in sys.argv:
        total = sum(s.get("hours", 0) for i, s in shelf)
        for book_id, spec in shelf:
            print(f"  {spec['hours']:6.2f} h  {spec['title']}")
        print(f"\n  {total:6.2f} h total across {len(shelf)} books")
        return 0

    if not (os.getenv("UPSTASH_REDIS_REST_URL")
            and os.getenv("UPSTASH_REDIS_REST_TOKEN")):
        print("Refusing to run: no Upstash configured, so this would seed a dict "
              "that vanishes when the script exits.\n"
              "Export UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN first.")
        return 2
    aai_key = os.environ["ASSEMBLYAI_API_KEY"]
    force = "--force" in sys.argv
    store = build_store()
    ttl_days = int(os.environ["PLAYHEAD_BOOK_TTL_DAYS"])
    print(f"store: {store.kind}   ttl: {ttl_days} days   books: {len(shelf)}\n")

    failed, began = 0, time.time()
    for book_id, spec in shelf:
        print(f"  {spec['title']}")
        try:
            outcome = seed(store, book_id, spec, aai_key, force)
        except Exception as exc:
            outcome = f"FAILED: {type(exc).__name__}: {exc}"
        if outcome.startswith(("FAILED", "REFUSED", "gave up")):
            failed += 1
        print(f"      {outcome}\n", flush=True)

    print(f"done in {(time.time() - began) / 60:.0f} minutes, "
          f"{failed} of {len(shelf)} not ready")
    print("\nFEATURED_IDS in server/main.py should read:")
    for book_id, _ in FEATURED:
        print(f'    "{book_id}",')
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

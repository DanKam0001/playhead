"""What this project has actually cost on AssemblyAI, from their records.

    python scripts/spend.py              # totals, by day, by book
    python scripts/spend.py --refresh    # ignore the local cache
    python scripts/spend.py --what-if    # what the same audio would have cost
                                         # under other settings

Written after the account went to a negative balance mid-seed, with nothing
anywhere counting what was being spent. The lesson was not "we used too much",
it was "nobody could see it": there is **no balance or usage endpoint** on the
API (`/v2/account` returns `{}`), so the dashboard is the only place the real
number lives, and nothing local was tracking the inputs to it.

So this reconstructs spend from the transcript history, which is ground truth
on AssemblyAI's side rather than our own bookkeeping -- our records cannot
drift from it, because they are not consulted. Every completed job is billed
for its audio duration, whether or not the book it belonged to ever finished
indexing; that gap is the thing worth watching, and it is printed.

Prices are from https://www.assemblyai.com/pricing, read 2026-09-21. They are
a constant here rather than a lookup because there is no pricing endpoint --
**check them before quoting a number to anyone.**
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

CACHE = ROOT / "data" / ".spend_cache.json"

# $ per audio hour. Async transcription only; see VOICE_PER_MIN for sessions.
PRICES = {
    "universal-3-5-pro": 0.21,
    "universal-2": 0.15,
}
# Add-ons stack on top of the base model, per audio hour.
ADDONS = {
    "auto_chapters": 0.08,      # deprecated, and English-only -- see below
}
# The Voice Agent is billed per minute of **session**, including the time
# nobody is speaking, because it is a websocket held open. That is the number
# that matters while the demo is public.
VOICE_PER_MIN = 0.075


def _save(cache: dict) -> None:
    """Write the cache atomically, so an interrupted run leaves it readable."""
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache), encoding="utf-8")
    tmp.replace(CACHE)


def api(url: str, key: str):
    req = urllib.request.Request(url, headers={"Authorization": key})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def all_transcripts(key: str, refresh: bool) -> list:
    """Every job on the account, with durations, cached between runs.

    The list endpoint does not carry `audio_duration`, so each job needs its
    own fetch -- hundreds of them. They never change once finished, so they
    are cached and only new ids are fetched.
    """
    cache = {}
    if CACHE.exists() and not refresh:
        try:
            cache = json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    url, listing = "https://api.assemblyai.com/v2/transcript?limit=100", []
    while url:
        page = api(url, key)
        rows = page.get("transcripts", [])
        listing.extend(rows)
        nxt = (page.get("page_details") or {}).get("prev_url")
        if not nxt or not rows:
            break
        url = nxt

    fresh = 0
    for row in listing:
        tid = row["id"]
        if tid in cache:
            continue
        if row.get("status") != "completed":
            # Errored jobs are not billed, so they need no detail fetch.
            cache[tid] = {"status": row.get("status"), "seconds": 0.0,
                          "created": row.get("created", ""),
                          "audio_url": row.get("audio_url", ""),
                          "error": (row.get("error") or "")[:80]}
            continue
        try:
            full = api(row["resource_url"], key)
        except Exception:
            continue
        cache[tid] = {
            "status": "completed",
            "seconds": float(full.get("audio_duration") or 0),
            "created": row.get("created", ""),
            "audio_url": row.get("audio_url", ""),
            "language": full.get("language_code"),
            "chapters": len(full.get("chapters") or []),
            "auto_chapters": bool(full.get("auto_chapters")),
        }
        fresh += 1
        # Save as we go. Several hundred detail fetches take minutes, and
        # writing the cache only at the end means an interruption throws away
        # every one of them and starts from zero next time -- exactly the
        # failure this project spent the morning removing from the indexer.
        if fresh % 25 == 0:
            _save(cache)
            print(f"  fetched {fresh} new job details...", end="\r", flush=True)

    _save(cache)
    if fresh:
        print(f"  fetched {fresh} new job details      ")
    return [dict(v, id=k) for k, v in cache.items()]


def book_of(audio_url: str) -> str:
    if "/download/" in audio_url:
        return audio_url.split("/download/", 1)[1].split("/")[0]
    return "(upload or probe)"


def money(hours: float, base: str, chapters: bool) -> float:
    rate = PRICES[base] + (ADDONS["auto_chapters"] if chapters else 0.0)
    return hours * rate


def main() -> int:
    key = os.environ["ASSEMBLYAI_API_KEY"]
    jobs = all_transcripts(key, "--refresh" in sys.argv)
    done = [j for j in jobs if j.get("status") == "completed"]
    billed_h = sum(j["seconds"] for j in done) / 3600
    errored = [j for j in jobs if j.get("status") != "completed"]

    print(f"\n{'=' * 66}")
    print(f"  {len(done)} billed jobs, {billed_h:.2f} audio hours")
    print(f"  {len(errored)} errored (not billed)")
    print(f"{'=' * 66}\n")

    # The model is not reported back on the transcript, so the base rate is a
    # range rather than a figure. Say so instead of picking one.
    chaptered = sum(j["seconds"] for j in done if j.get("auto_chapters")) / 3600
    lo = money(billed_h, "universal-2", False) + chaptered * ADDONS["auto_chapters"]
    hi = money(billed_h, "universal-3-5-pro", False) + chaptered * ADDONS["auto_chapters"]
    print(f"  ESTIMATED TRANSCRIPTION SPEND: ${lo:.2f} - ${hi:.2f}")
    print(f"    base {billed_h:.1f} h at $0.15-0.21/h"
          f"   + auto_chapters on {chaptered:.1f} h at $0.08/h")
    print(f"    (the API does not report which model ran, hence the range;")
    print(f"     and there is no balance endpoint, so confirm on the dashboard)\n")

    per_day = defaultdict(float)
    for j in done:
        per_day[j.get("created", "")[:10]] += j["seconds"] / 3600
    print("  BY DAY")
    for d in sorted(per_day):
        print(f"    {d}   {per_day[d]:7.2f} h   ~${per_day[d] * 0.23:6.2f}")

    per_book = defaultdict(float)
    for j in done:
        per_book[book_of(j.get("audio_url", ""))] += j["seconds"] / 3600
    print("\n  BY SOURCE (what was paid for, not what got indexed)")
    for b, h in sorted(per_book.items(), key=lambda x: -x[1]):
        print(f"    {b[:46]:48s} {h:7.2f} h   ~${h * 0.23:6.2f}")

    # Non-English audio charged for a feature that cannot run on it.
    wasted = sum(j["seconds"] for j in done
                 if j.get("auto_chapters") and j.get("language") not in (None, "en")
                 and not j.get("chapters")) / 3600
    if wasted > 0.05:
        print(f"\n  NOTE: auto_chapters was billed on {wasted:.2f} h of non-English")
        print(f"        audio and returned no chapters "
              f"(~${wasted * ADDONS['auto_chapters']:.2f} for nothing).")

    if "--what-if" in sys.argv:
        print(f"\n{'=' * 66}\n  WHAT THE SAME {billed_h:.1f} HOURS WOULD COST\n{'=' * 66}")
        for label, base, ch in (
                ("as run: U3.5 Pro + auto_chapters", "universal-3-5-pro", True),
                ("U3.5 Pro, no auto_chapters      ", "universal-3-5-pro", False),
                ("Universal-2 + auto_chapters     ", "universal-2", True),
                ("Universal-2, no auto_chapters   ", "universal-2", False)):
            c = money(billed_h, base, ch)
            print(f"    {label}  ${c:7.2f}")
        cheapest = money(billed_h, "universal-2", False)
        print(f"\n    dropping auto_chapters alone saves "
              f"${billed_h * ADDONS['auto_chapters']:.2f} on this volume.")
        print(f"    It is deprecated, English-only, and books.contents() already")
        print(f"    derives a table of contents without it.")
        print(f"\n  VOICE AGENT (billed per minute of session, idle time included)")
        for mins, what in ((2, "one 2-minute demo take"),
                           (40, "filming, ~20 takes"),
                           (250, "50 judges x 5 minutes"),
                           (60, "ONE tab left open an hour")):
            print(f"    {what:34s} {mins:4d} min   ${mins * VOICE_PER_MIN:6.2f}")
        print(f"\n    A session costs money while it is open, not while it is")
        print(f"    talking. That is what the Stop listening button is for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

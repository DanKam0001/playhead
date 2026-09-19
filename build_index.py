"""Turn an audiobook file into a timestamp-indexed, embedded library.

    python build_index.py audio/chapter1.wav

Transcription is cached to data/<name>.transcript.json -- re-transcribing a
10-hour book every time you tweak chunking is slow and costs money.
"""
import json
import os
import sys
from pathlib import Path

import assemblyai as aai
from dotenv import load_dotenv

from playhead.brain import GeminiEmbedder
from playhead.library import Library

load_dotenv()

DATA = Path("data")


def transcribe(audio_path: Path, force: bool = False) -> dict:
    cache = DATA / f"{audio_path.stem}.transcript.json"
    if cache.exists() and not force:
        print(f"[index] using cached transcript {cache}")
        return json.loads(cache.read_text(encoding="utf-8"))

    aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]
    print(f"[index] transcribing {audio_path.name} (this takes a while)...")
    config = aai.TranscriptionConfig(
        # Ordered fallback list: flagship first, broadly-available model second.
        speech_models=["universal-3-5-pro", "universal-2"],
        punctuate=True,
        format_text=True,
    )
    t = aai.Transcriber(config=config).transcribe(str(audio_path))
    if t.status == aai.TranscriptStatus.error:
        raise RuntimeError(t.error)

    # Paragraphs are the useful unit: they follow the reader's pauses.
    paragraphs = [
        {"text": p.text, "start": p.start, "end": p.end}
        for p in t.get_paragraphs()
    ]
    payload = {
        "id": t.id,
        "text": t.text,
        "audio_duration": t.audio_duration,
        "paragraphs": paragraphs,
    }
    DATA.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"[index] cached transcript -> {cache}")
    return payload


KEYTERM_PROMPT = """From this audiobook transcript, list the technical terms, proper nouns, and domain jargon that a general speech-recognition model would be likely to mis-hear -- and that a listener might say out loud when asking a question about it.

One term per line, no numbering, no commentary. At most six words each. Prefer specific terms over generic ones. Maximum 90 terms.

TRANSCRIPT:
"""


def extract_keyterms(payload: dict, api_key: str) -> list[str]:
    """Derive the STT vocabulary from the book itself.

    Hardcoding these does not survive a second book. The terms are fed to the
    live stream via keyterms_prompt so jargon in the *question* is recognised.
    Streaming caps at 100.
    """
    from google import genai
    from google.genai import types

    text = payload.get("text", "")[:120_000]
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=KEYTERM_PROMPT + text,
        config=types.GenerateContentConfig(
            max_output_tokens=2000, temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    terms, seen = [], set()
    for line in (resp.text or "").splitlines():
        t = line.strip().lstrip("-*0123456789. ").strip()
        if t and len(t.split()) <= 6 and t.lower() not in seen:
            seen.add(t.lower())
            terms.append(t)
    return terms[:100]


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    audio = Path(sys.argv[1])
    force = "--force" in sys.argv
    payload = transcribe(audio, force=force)

    chunks = Library.chunks_from_transcript(payload)
    print(f"[index] {len(chunks)} chunks from {len(payload['paragraphs'])} paragraphs")
    if not chunks:
        print("[index] nothing to index -- is the audio silent?")
        return 1

    lib = Library(DATA / f"{audio.stem}.db")
    embedder = GeminiEmbedder(os.environ["GEMINI_API_KEY"])
    print(f"[index] embedding {len(chunks)} chunks...")
    lib.build(chunks, embedder)
    print(f"[index] done: {len(lib)} chunks -> {lib.db_path}")

    kt_path = DATA / f"{audio.stem}.keyterms.json"
    if not kt_path.exists() or force:
        try:
            terms = extract_keyterms(payload, os.environ["GEMINI_API_KEY"])
            kt_path.write_text(json.dumps(terms, indent=1), encoding="utf-8")
            print(f"[index] {len(terms)} keyterms -> {kt_path}")
            print(f"[index] e.g. {', '.join(terms[:8])}")
        except Exception as exc:
            print(f"[index] keyterm extraction skipped: {exc}")

    c = chunks[len(chunks) // 2]
    print(f"[index] sample {c.cite()} {c.text[:90]}...")
    lib.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

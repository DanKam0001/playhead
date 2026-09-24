"""Side-by-side: ordinary RAG vs position-first retrieval. The demo's money shot.

    python compare_rag.py 576 "Wait, what did that last part mean?"
    python compare_rag.py --all          # run the scripted demo questions

The naive side is NOT a straw man. It is what a competent developer would
build: embed the question, take the top-k most similar chunks from the whole
book, hand them to the same model with a standard RAG prompt. That is the
default architecture for "chat with your document", and it is genuinely good at
questions that contain their own subject.

It fails here for a structural reason, not a quality one. "What did that last
part mean?" is deictic -- it points at the listener's position instead of naming
a topic. There is nothing in it to match, so similarity search returns whatever
happens to sit nearest a contentless query, which is noise. Position answers it
exactly, in under a millisecond, with no model involved.
"""
import argparse
import os
import sys

from dotenv import load_dotenv

from playhead.brain import Brain, GeminiEmbedder, is_deictic
from playhead.library import Library

load_dotenv()

NAIVE_PROMPT = """Answer the question using the excerpts below. Two or three \
sentences, spoken aloud, no markdown.

EXCERPTS:
{context}

QUESTION: {question}"""

# The scripted demo questions, with the book position each is asked at.
SCRIPT = [
    (576, "Wait, what did that last part mean?"),
    (1045, "Why does it matter which one you pick?"),
    (700, "Hold on, what was the train and embankment thing again?"),
]


class NaiveRag:
    """Textbook vector RAG: top-k over the whole book, no position, no guards."""

    def __init__(self, api_key: str, library: Library, k: int = 4):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._embedder = GeminiEmbedder(api_key)
        self._library = library
        self._k = k
        self.last_chunks = []

    def answer(self, question: str, _book_timestamp: float) -> str:
        from google.genai import types

        qvec = self._embedder([question], "query")[0]
        # No before_s cap: ordinary RAG has no notion of where the reader is,
        # so it will happily answer from later in the book.
        chunks = self._library.search(qvec, k=self._k)
        self.last_chunks = chunks
        context = "\n\n".join(f"{c.cite()} {c.text}" for c in chunks) or "(nothing found)"
        resp = self._client.models.generate_content(
            model="gemini-3.8-flash",
            contents=NAIVE_PROMPT.format(context=context, question=question),
            config=types.GenerateContentConfig(
                max_output_tokens=400, temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (resp.text or "").strip()


def run_one(naive: NaiveRag, brain: Brain, t: float, question: str) -> None:
    mins = f"{int(t) // 60}:{int(t) % 60:02d}"
    print("=" * 78)
    print(f"  ASKED AT {mins} IN THE BOOK")
    print(f"  \"{question}\"")
    print(f"  deictic: {is_deictic(question)}")
    print("=" * 78)

    print("\n-- ORDINARY RAG (vector search over the whole book) " + "-" * 25)
    try:
        a = naive.answer(question, t)
        print(f"   retrieved: {[c.cite() for c in naive.last_chunks]}")
        print(f"   answer:    {a}")
    except Exception as exc:
        print(f"   FAILED: {type(exc).__name__}: {str(exc)[:140]}")

    print("\n-- PLAYHEAD (position window, capped at the playhead) " + "-" * 23)
    try:
        a = brain.answer(question, t)
        near, far = brain.last_context
        print(f"   retrieved: {[c.cite() for c in near]} near"
              + (f" + {[c.cite() for c in far]} elsewhere" if far else ""))
        print(f"   answer:    {a}")
    except Exception as exc:
        print(f"   FAILED: {type(exc).__name__}: {str(exc)[:140]}")
    print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("timestamp", nargs="?", type=float)
    ap.add_argument("question", nargs="*")
    ap.add_argument("--db", default="data/calculus.db")
    ap.add_argument("--all", action="store_true", help="run the scripted questions")
    args = ap.parse_args()

    lib = Library(args.db)
    if not len(lib):
        print(f"{args.db} is empty -- run build_index.py first.")
        return 1
    key = os.environ["GEMINI_API_KEY"]
    naive, brain = NaiveRag(key, lib), Brain(key, lib)

    if args.all:
        cases = SCRIPT
    elif args.timestamp is not None and args.question:
        cases = [(args.timestamp, " ".join(args.question))]
    else:
        print(__doc__)
        return 2

    for t, q in cases:
        brain.clear_memory()
        run_one(naive, brain, float(t), q)
    lib.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

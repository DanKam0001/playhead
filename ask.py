"""Exercise the brain without audio: ask a question at a book timestamp.

    python ask.py 30 "what did that last sentence mean?"
"""
import os, sys
from dotenv import load_dotenv
from echoread.brain import Brain
from echoread.library import Library

load_dotenv()


def make_brain(db="data/chapter4.db"):
    lib = Library(db)
    return Brain(os.environ["GEMINI_API_KEY"], lib), lib


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); raise SystemExit(2)
    brain, lib = make_brain()
    t = float(sys.argv[1]); q = " ".join(sys.argv[2:])
    print(f"[{t:.0f}s] Q: {q}")
    print("A:", brain.answer(q, t))
    near, far = brain.last_context
    print(f"   (context: {[c.cite() for c in near]} near, {[c.cite() for c in far]} elsewhere)")
    lib.close()

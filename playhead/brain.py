"""Gemini + timestamp-scoped retrieval, with conversational memory."""
import re
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import logging

from google import genai
from google.genai import types

# gemini-3.x returns an opaque "thought_signature" part alongside the text
# even with thinking disabled, and the SDK logs a warning for every single
# call. The concatenated text is exactly what we want, so this is noise --
# and it would be on screen during the demo.
logging.getLogger("google_genai.types").setLevel(logging.ERROR)
logging.getLogger("google.genai.types").setLevel(logging.ERROR)

from .library import Chunk, Library

EMBED_MODEL = "gemini-embedding-001"
CHAT_MODEL = "gemini-3.8-flash"  # 2.2s with thinking off; pinned, not a -latest alias
EMBED_DIM = 768                   # truncated from 3072: 4x less RAM, no measurable
                                  # recall loss at book scale
# Gemini 3.x flash models think before answering. Left on, reasoning tokens eat
# the output budget and the spoken reply truncates mid-sentence ("It means
# that"). Off: 2.2s instead of 2.7s and complete answers. Verified on this key;
# note gemini-3.6-flash and flash-lite reject budget=0 outright.
THINKING_BUDGET = 0
MAX_REPLY_TOKENS = 400
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")

SYSTEM_PROMPT = """You are a study companion riding along with someone driving \
while they listen to an audiobook. They have paused it to ask you something.

Rules:
- They are DRIVING. Two or three sentences. No lists, no headings, no markdown \
- your reply is spoken aloud, not read.
- Answer from the BOOK CONTEXT below. It is what they just heard.
- When they say "that", "the last bit", "he", they mean the JUST HEARD passage. \
Resolve it from there, not from the wider book.
- Never mention material from later in the book than where they are. No spoilers.
- If the context does not contain the answer, say so in a few words and then \
answer from your own knowledge anyway, flagged briefly ("not in the book, but..."). \
Refusing outright is unhelpful; pretending it came from the book is worse. \
Never state a fact you are not confident in.
- Speak plainly. Define jargon in passing rather than assuming it."""

# Things a listener says to end the conversation rather than ask something.
# Real speech chains these -- "okay, cool, carry on" -- so match a sequence.
_RESUME_PHRASE = (
    r"(?:ok(?:ay)?|alright|right|sure|cool|nice|thanks?|thank you|got it|"
    r"(?:that )?makes sense|fair enough|i see|gotcha|never ?mind|nvm|"
    r"yep|yeah|yes|mhm|uh[- ]huh|perfect|great|"
    r"carry on|keep going|go on|continue|resume|play|unpause|"
    r"back to (?:the )?book|that'?s (?:it|all)|no(?:pe)?)"
)
_RESUME_RE = re.compile(
    rf"^\W*{_RESUME_PHRASE}(?:[\s,.!-]+{_RESUME_PHRASE})*[\s.!,]*$", re.I
)


# "What did THAT mean?" points at the playhead, not at a topic. Embedding it
# costs ~0.9s and returns noise, because there is no content in it to match.
_DEICTIC_RE = re.compile(
    r"\b(?:that|this|those|these|it|he|she|they|the last (?:bit|part|sentence|line))\b", re.I
)
_CONTENT_RE = re.compile(r"\b(?:earlier|before|previously|again|back when|chapter|remind|stand for)\b", re.I)


def is_deictic(text: str) -> bool:
    """True if the question refers to the current position rather than a topic.

    Deliberately conservative: an explicit look-back cue ("earlier", "again")
    always wins, because those are the questions semantic search exists for.
    """
    if _CONTENT_RE.search(text):
        return False
    words = re.findall(r"[a-z']+", text.lower())
    return bool(_DEICTIC_RE.search(text)) and len(words) <= 14


# The opposite of a resume command: "keep waiting, I am still composing the
# question". Answering one of these is worse than useless -- the agent replies
# to a fragment while the person is still mid-sentence, and their real question
# gets clipped. Observed live 2026-09-10 with "Hold on."
_HOLD_RE = re.compile(
    r"^\W*(?:hold on|hang on|wait(?: a (?:sec|second|moment|minute))?|one (?:sec|second|moment|minute)|just a (?:sec|second|moment|minute)|give me a (?:sec|second|moment|minute)|um+|uh+|hmm+|let me think)[\s.,!]*$",
    re.I,
)


def is_hold_command(text: str) -> bool:
    """True if this is a placeholder, not a question. Keep listening."""
    return bool(_HOLD_RE.match(text.strip()))


def is_resume_command(text: str) -> bool:
    """True if this is 'stop talking and play the book', not a question."""
    return bool(_RESUME_RE.match(text.strip()))


class GeminiEmbedder:
    def __init__(self, api_key: str, model: str = EMBED_MODEL, dim: int = EMBED_DIM):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._dim = dim

    def __call__(self, texts: Sequence[str], task: str = "document") -> np.ndarray:
        task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
        out: List[List[float]] = []
        # The API caps batch size; 100 is comfortably under it and keeps the
        # progress granular when indexing a long book.
        for i in range(0, len(texts), 100):
            resp = self._client.models.embed_content(
                model=self._model,
                contents=list(texts[i : i + 100]),
                config=types.EmbedContentConfig(
                    task_type=task_type, output_dimensionality=self._dim
                ),
            )
            out.extend(e.values for e in resp.embeddings)
        return np.array(out, dtype="float32")


class Brain:
    """Implements the Brain protocol in session.py."""

    def __init__(
        self,
        api_key: str,
        library: Library,
        embedder: Optional[GeminiEmbedder] = None,
        model: str = CHAT_MODEL,
        max_history_turns: int = 6,
    ):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._library = library
        self._embedder = embedder if embedder is not None else GeminiEmbedder(api_key)
        self._history: List[Tuple[str, str]] = []
        self._max_history = max_history_turns
        self.last_context: Tuple[List[Chunk], List[Chunk]] = ([], [])

    def clear_memory(self) -> None:
        """Called when a fresh interruption starts, so old threads don't bleed in."""
        self._history.clear()

    def _build_prompt(self, question: str, near: Sequence[Chunk], far: Sequence[Chunk]) -> str:
        parts = ["BOOK CONTEXT — what they just heard:"]
        parts.append("\n".join(f"{c.cite()} {c.text}" for c in near) or "(nothing indexed here)")
        if far:
            parts.append("\nEARLIER IN THE BOOK — possibly relevant:")
            parts.append("\n".join(f"{c.cite()} {c.text}" for c in far))
        if self._history:
            parts.append("\nCONVERSATION SO FAR:")
            parts.extend(f"{'Them' if r == 'user' else 'You'}: {t}" for r, t in self._history)
        parts.append(f"\nTHEY ASK: {question}")
        return "\n".join(parts)

    def warm(self) -> None:
        """Burn the cold start before the user ever asks anything.

        Measured: the first call in a process costs ~3.3s more than the rest
        (TLS, DNS, client init). On camera the first question is the one that
        matters, so pay that cost while the book is still playing.
        """
        try:
            self._client.models.generate_content(
                model=self._model,
                contents="hi",
                config=types.GenerateContentConfig(
                    max_output_tokens=8,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
                ),
            )
            self._embedder(["warm"], "query")
        except Exception as exc:
            print(f"[brain] warm-up skipped: {exc}")

    def answer_stream(self, question: str, book_timestamp: float) -> Iterator[str]:
        """Yield the reply sentence by sentence, so TTS can start early.

        Waiting for the whole reply before speaking wastes ~1s of the budget.
        """
        qvec = None if is_deictic(question) else self._embed_or_none(question)
        near, far = self._library.context_for(book_timestamp, qvec)
        self.last_context = (near, far)

        stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=self._build_prompt(question, near, far),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=MAX_REPLY_TOKENS,
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            ),
        )
        buf, spoken = "", []
        for chunk in stream:
            buf += chunk.text or ""
            # Emit on sentence boundaries. The minimum length stops a stray
            # "Yes." going to TTS as its own clip, which sounds clipped.
            while True:
                m = _SENTENCE_END.search(buf)
                if not m or m.end() < 25:
                    break
                sentence, buf = buf[: m.end()].strip(), buf[m.end():]
                spoken.append(sentence)
                yield sentence
        if buf.strip():
            spoken.append(buf.strip())
            yield buf.strip()

        reply = " ".join(spoken)
        self._history.append(("user", question))
        self._history.append(("agent", reply))
        del self._history[: -self._max_history * 2]

    def answer(self, question: str, book_timestamp: float) -> str:
        qvec = None
        if is_deictic(question):
            # Fast path: position alone answers it. Skips a round trip.
            print("[brain] deictic question, skipping semantic search")
        else:
            qvec = self._embed_or_none(question)

        return self._generate(question, book_timestamp, qvec)

    def _embed_or_none(self, question: str):
        try:
            return self._embedder([question], "query")[0]
        except Exception as exc:
            # Semantic search is the optional half. Losing it must not cost us
            # the positional context, which is what answers most questions.
            print(f"[brain] embedding failed, positional context only: {exc}")
            return None

    def _generate(self, question: str, book_timestamp: float, qvec) -> str:
        near, far = self._library.context_for(book_timestamp, qvec)
        self.last_context = (near, far)

        resp = self._client.models.generate_content(
            model=self._model,
            contents=self._build_prompt(question, near, far),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=MAX_REPLY_TOKENS,
                temperature=0.3,
                thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET),
            ),
        )
        reply = (resp.text or "").strip() or "Sorry, I didn't catch that — say again?"

        self._history.append(("user", question))
        self._history.append(("agent", reply))
        del self._history[: -self._max_history * 2]
        return reply

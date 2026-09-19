"""The indexed book: text chunks anchored to playback timestamps.

Two retrieval paths, deliberately:

  window(t)  -- what was being read at timestamp t. Deterministic, exact.
  search(q)  -- semantically similar passages anywhere in the book.

The window is the load-bearing one. The signature query for this product,
"wait, what did that last sentence mean?", is deictic: it contains no content
to embed, so vector search over it returns noise. Position answers it; the
vector index is for "where did they define X earlier?" -- questions that do
carry content and reach outside the current window.

Storage is SQLite + a .npy of embeddings. A 10-hour book is ~1800 chunks;
brute-force cosine over that is ~1.3 ms, against an LLM call of 500-1500 ms.
A vector DB would add a dependency tree and a failure mode to save nothing.
"""
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np

Embedder = Callable[[Sequence[str], str], np.ndarray]
"""(texts, task) -> (n, dim) float32. task is 'document' or 'query'."""

# How much book to hand the LLM around the interrupt point. Weighted backwards:
# the user is asking about what they just heard.
WINDOW_BEFORE_S = 90.0
WINDOW_AFTER_S = 15.0


@dataclass
class Chunk:
    id: int
    start_s: float
    end_s: float
    text: str

    def cite(self) -> str:
        m, s = divmod(int(self.start_s), 60)
        return f"[{m:02d}:{s:02d}]"


class Library:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vec_path = self.db_path.with_suffix(".npy")
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS chunks (
                   id INTEGER PRIMARY KEY,
                   start_s REAL NOT NULL,
                   end_s REAL NOT NULL,
                   text TEXT NOT NULL)"""
        )
        # Window lookup is a range scan on start_s; index it.
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_start ON chunks(start_s)")
        self._conn.commit()
        self._vectors: Optional[np.ndarray] = None
        if self.vec_path.exists():
            self._vectors = np.load(self.vec_path)

    # ---------- build ----------

    @staticmethod
    def chunks_from_transcript(transcript_json: dict, target_seconds: float = 25.0) -> List[Chunk]:
        """Turn an AssemblyAI transcript into timestamped chunks.

        Prefers the API's own paragraph boundaries -- they follow the reader's
        pauses, which beat fixed-width windows that guillotine mid-sentence.
        Short paragraphs are merged up to target_seconds so a chunk carries
        enough context to be worth retrieving.
        """
        paras = transcript_json.get("paragraphs") or []
        if not paras:
            # Fall back to sentences, then to one blob.
            paras = transcript_json.get("sentences") or []
        if not paras:
            text = transcript_json.get("text", "")
            dur = transcript_json.get("audio_duration", 0) or 0
            return [Chunk(0, 0.0, float(dur), text)] if text else []

        chunks: List[Chunk] = []
        buf_text: List[str] = []
        buf_start: Optional[float] = None
        buf_end = 0.0
        for p in paras:
            # AssemblyAI reports milliseconds.
            start, end = p["start"] / 1000.0, p["end"] / 1000.0
            if buf_start is None:
                buf_start = start
            buf_text.append(p["text"].strip())
            buf_end = end
            if buf_end - buf_start >= target_seconds:
                chunks.append(Chunk(len(chunks), buf_start, buf_end, " ".join(buf_text)))
                buf_text, buf_start = [], None
        if buf_text and buf_start is not None:
            chunks.append(Chunk(len(chunks), buf_start, buf_end, " ".join(buf_text)))
        return chunks

    def build(self, chunks: Sequence[Chunk], embedder: Optional[Embedder] = None) -> None:
        self._conn.execute("DELETE FROM chunks")
        self._conn.executemany(
            "INSERT INTO chunks (id, start_s, end_s, text) VALUES (?,?,?,?)",
            [(c.id, c.start_s, c.end_s, c.text) for c in chunks],
        )
        self._conn.commit()
        if embedder is not None and chunks:
            vecs = embedder([c.text for c in chunks], "document").astype("float32")
            # Pre-normalise so query time is a plain dot product.
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            np.save(self.vec_path, vecs)
            self._vectors = vecs

    # ---------- retrieval ----------

    def _rows_to_chunks(self, rows) -> List[Chunk]:
        return [Chunk(r["id"], r["start_s"], r["end_s"], r["text"]) for r in rows]

    def window(
        self,
        timestamp: float,
        before: float = WINDOW_BEFORE_S,
        after: float = WINDOW_AFTER_S,
    ) -> List[Chunk]:
        """Chunks overlapping [t-before, t+after], in reading order."""
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE end_s >= ? AND start_s <= ? ORDER BY start_s",
            (timestamp - before, timestamp + after),
        ).fetchall()
        return self._rows_to_chunks(rows)

    def search(self, query_vec: np.ndarray, k: int = 4, before_s: Optional[float] = None) -> List[Chunk]:
        """Top-k semantically similar chunks.

        before_s caps results at the listener's current position -- retrieving
        chapter 12 to answer a chapter 3 question is a spoiler, which for a
        book is a real failure, not a nitpick.
        """
        if self._vectors is None or not len(self._vectors):
            return []
        q = query_vec.astype("float32").ravel()
        q /= np.linalg.norm(q) + 1e-9
        scores = self._vectors @ q

        if before_s is not None:
            allowed = {r["id"] for r in self._conn.execute(
                "SELECT id FROM chunks WHERE start_s <= ?", (before_s,))}
            mask = np.zeros(len(scores), dtype=bool)
            mask[list(allowed)] = True
            scores = np.where(mask, scores, -np.inf)

        k = min(k, int(np.isfinite(scores).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE id IN ({','.join('?' * len(top))})",
            [int(i) for i in top],
        ).fetchall()
        by_id = {c.id: c for c in self._rows_to_chunks(rows)}
        return [by_id[int(i)] for i in top if int(i) in by_id]

    def context_for(
        self,
        timestamp: float,
        query_vec: Optional[np.ndarray] = None,
        k: int = 3,
    ) -> tuple[List[Chunk], List[Chunk]]:
        """(what they just heard, elsewhere in the book). Deduped."""
        near = self.window(timestamp)
        near_ids = {c.id for c in near}
        far: List[Chunk] = []
        if query_vec is not None:
            far = [c for c in self.search(query_vec, k=k + len(near_ids), before_s=timestamp)
                   if c.id not in near_ids][:k]
        return near, far

    def duration_hint(self) -> float:
        """End of the last indexed chunk. Lets the browser size its scrubber
        without needing the audio file's own metadata."""
        row = self._conn.execute("SELECT MAX(end_s) FROM chunks").fetchone()
        return float(row[0] or 0.0)

    def __len__(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

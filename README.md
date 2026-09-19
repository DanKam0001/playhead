# EchoRead

**An audiobook you can interrupt.** Say *"wait, what did that last part mean?"*
and it answers about the passage you just heard — then picks up where it left off.

**Live: https://echoread-alpha.vercel.app** — press start, let it play, talk over it.

Built for the AssemblyAI Voice Agent Hackathon.

---

## The problem this solves

An audiobook is the one teacher you cannot raise your hand at. For people who
learn by ear — commuters, people with dyslexia or low vision, anyone whose
reading time is walking time — a sentence that doesn't land is simply lost. You
can rewind, but rewinding replays the same words that already failed.

Asking is the natural repair, and it has never been available.

## Why this is hard, and what's actually new here

Ask a normal retrieval system *"what did that mean?"* and it has nothing to work
with. **The question contains no topic.** There is no phrase to search for, no
entity to match, no keyword to embed. Every naive RAG pipeline answers this
question badly, because the thing being asked about is not in the question.

EchoRead retrieves by **position, not by words**. The listener's playhead — where
they are in the audio — is the query. The passage they just heard is the answer's
context, whether or not they managed to name it.

We call this **position-first (deictic) retrieval**. "That", "this", "the last
bit", "he" — the words linguists call deictic, which point at context rather than
carry it — are exactly the words people use when something doesn't land. They are
the words search is worst at and position is best at.

Two rules fall out of it:

- **The playhead is the query.** A window of **−90 s / +15 s** around the
  listener's position, so it covers the run-up as well as the sentence itself.
- **Nothing after the playhead exists.** Semantic lookback is capped at the
  listener's position. Answering a chapter 3 question with chapter 12 material
  would spoil the book — a real failure, not a rounding error.

`compare_rag.py` runs the two approaches side by side on the same questions.

## Architecture

```
Browser                    AssemblyAI                   This backend
───────                    ──────────                   ────────────
mic ──24 kHz PCM16──▶  Voice Agent API
                       (STT + LLM + TTS,
                        one websocket)
                            │
                            │ HTTP tool call (server-to-server)
                            ▼
                                                  /tools/passage_at_playhead
                                                  reads the playhead from Redis,
                                                  returns the passage window
                            │
◀──── reply audio ──────────┘
   playhead POSTed once a second ───────────────▶ /api/playhead
```

Three things worth noting:

- **The backend holds no websocket.** The browser connects straight to
  AssemblyAI with a five-minute token this server mints, so the API key never
  reaches the client and the server stays stateless enough for serverless.
- **The agent lives on AssemblyAI's side** as a stored agent, with one HTTP
  tool. The client sends only an agent id — no prompt, no tool definitions.
- **The playhead travels out of band.** AssemblyAI calls the tool from its own
  servers and has no idea where playback is, so the browser reports position to
  Redis, and the tool reads it there. This is the seam that makes a
  *position-first* agent possible on a serverless deployment.

**Echo cancellation is why this is a browser app.** The book and the agent's own
voice come out of the speakers and back into the mic; the browser's native AEC
removes them, which a plain desktop capture cannot do.

## Stack

| Part | Choice |
|---|---|
| Voice agent (STT + LLM + TTS) | **AssemblyAI Voice Agent API**, one websocket, 24 kHz PCM16 both ways |
| Building the index | **AssemblyAI async transcription** (`universal-3-5-pro`), paragraph chunks with timestamps |
| Retrieval | SQLite time window + brute-force cosine over Gemini embeddings |
| Backend | FastAPI on Vercel |
| Shared state | Upstash Redis (memory fallback for local runs) |

**No vector database.** Brute-force cosine over a 10-hour book takes **1.3 ms**,
against ~2200 ms for the model call it feeds. A vector DB would have added
~200 MB of dependencies to save nothing measurable.

## Measured

- Einstein chapter: **20.6 min → 23 chunks**, median 50 s / 630 chars
- Brute-force cosine: **1.3 ms** for 1800 chunks × 3072-d
- Demo book: LibriVox *Relativity: The Special and General Theory*, §7–9 — public domain

## Run it locally

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows
pip install -r requirements-dev.txt
cp .env.example .env                                 # add your keys

python scripts/create_agent.py https://your-public-url    # publish the agent
uvicorn server.main:app --reload
```

The tool URL must be publicly reachable — AssemblyAI calls it from its own
servers, so for local development use a tunnel (ngrok, cloudflared), not
localhost. `GET /api/health` reports whether the agent is configured and which
store is in use.

To index a different book:

```bash
python build_index.py path/to/book.mp3 --name mybook    # transcribe + chunk + embed
python compare_rag.py                                   # position-first vs naive RAG
```

## Layout

| Path | What it is |
|---|---|
| `web/`, `public/` | The browser client. **`public/` is what deploys** — copy `web/` into it |
| `server/main.py` | FastAPI: session tokens, playhead, the HTTP tool |
| `server/agent.json` | The stored agent: system prompt + the one tool |
| `echoread/library.py` | The retrieval core — time window, capped semantic search |
| `build_index.py` | AssemblyAI transcription → timestamped chunks → embeddings |
| `compare_rag.py` | Side-by-side: naive vector RAG vs position-first |

## How we got here

EchoRead began as a desktop app with local voice-activity detection, streaming
STT, and a separate TTS vendor. That version worked, and the tuning taught us
things worth keeping: speech reads at 0.024 RMS against room tone at 0.0002;
local barge-in detection beat a network round trip ~50 ms to ~300 ms; a
mid-question pause splits into two turns unless you stitch them.

It was retired when AssemblyAI's Voice Agent API collapsed that entire pipeline
into one websocket — and, more importantly, when a browser could do the echo
cancellation that the desktop build kept fighting. The retrieval core survived
the rewrite unchanged, which is the part that was ever novel.

## Licence

MIT — see [LICENSE](LICENSE).

# EchoRead — agent handoff

Interactive audiobook voice agent. Listen to a technical audiobook; interrupt it
by talking ("wait, what did that last sentence mean?"); discuss; resume.
Built for the **AssemblyAI Voice Agent Hackathon** (lablab.ai).

**Read `PROGRESS.md` for current state, the plan, and hackathon requirements.**
This file is the engineering context that is expensive to rediscover.

---

## ⚠ Read this first: there are TWO codebases, and only one is the product

On 2026-09-13 the project **pivoted** from a local Python desktop app to a
browser app on AssemblyAI's **Voice Agent API**, because the submission form
requires a live Application URL. Both still exist in this repo.

| | **Web app — THE PRODUCT** | Desktop app — retired |
|---|---|---|
| Live at | https://echoread-alpha.vercel.app | local only |
| Code | `server/`, `web/`, `public/`, `api/`, `scripts/` | `echoread/{mic,ears,session,voice,player,brain}.py`, `run_demo.py` |
| STT / LLM / TTS | AssemblyAI Voice Agent (all three) | AssemblyAI streaming + Gemini + ElevenLabs |
| Barge-in | Voice Agent `input.speech.started` + browser echo cancellation | local EnergyVad + two-stage calibration |

**Shared by both, and the actual novelty:** `echoread/library.py` (position-first
retrieval + spoiler cap) and `build_index.py` (the timestamped index).

**Do not spend time on mic calibration, `MIN_TRIGGER_RMS`, `ATTACK_FRAMES`, turn
stitching, or Gemini/ElevenLabs tuning** unless the user explicitly revives the
desktop app. The "Barge-in tuning" and much of the "Architecture decisions"
sections below describe the retired client. They are kept because the findings
are real and some carry over — but the browser's native echo cancellation is
what solved the problem they were fighting.

### The `web/` vs `public/` trap

Vercel serves the browser client from **`public/`**, not `web/`. `web/` is what
the local uvicorn server serves. They are copies. **Edit `web/`, then copy to
`public/` before deploying** — otherwise the deploy succeeds and nothing
changes:

```bash
cp web/index.html web/app.js public/
```

Verify after deploying: `curl -s https://echoread-alpha.vercel.app/static/app.js | grep <your change>`.

---

## Always do this before writing AssemblyAI code

Fetch https://www.assemblyai.com/docs/llms.txt first. The API has changed — do
not rely on memorized parameter names. Optional docs MCP server:
`claude mcp add assemblyai-docs --transport http https://mcp.assemblyai.com/docs`

---

## Verified API facts (checked against the live API, not the docs)

The docs and AssemblyAI's own integration guide contradict each other. These
were confirmed by running `probe_streaming.py` against a real key on 2026-09-09.
**Trust these over any doc snapshot.**

- `speech_model` on streaming is **optional**, defaulting to `universal-3-5-pro`.
  (Their integration guide says required; the model-selection page says optional.)
- `turn_is_formatted` is **always True** on U3.5 Pro, including partials. Not a
  useful gate for "is this the final text".
- `end_of_turn_confidence` is **binary** — 0.0 partial, 1.0 end-of-turn. So
  `end_of_turn_confidence_threshold` does nothing.
- **A mid-sentence pause splits one spoken question into multiple `end_of_turn`
  events.** "Wait, hold on." arrives complete before "what did that mean?" does.
  Anything treating the first `end_of_turn` as the whole question is broken.
  `session.py` stitches them — do not remove that.
- The SDK (1.3.0) **does** expose `SpeechModel.universal_3_5_pro` enums, despite
  guides claiming they don't exist. `universal-3-6-pro` also exists.
- Gemini 3.x flash models **think by default**, and reasoning tokens consume
  `max_output_tokens` — replies truncate mid-sentence ("It means that"). Set
  `thinking_config=ThinkingConfig(thinking_budget=0)`. Note `gemini-3.6-flash`
  and `gemini-flash-lite-latest` **reject** budget=0.
- `gemini-2.5-flash` is retired for new API keys (404). Using `gemini-3.8-flash`.
- **LLM Gateway**: endpoint and auth work, but this account returns *"does not
  have access to this LLM Gateway model"* — a credits/plan gate, not an
  architecture problem. See `probe_gateway.py`. Worth asking for in Discord.

## Barge-in tuning (settled over three hardware runs, 2026-09-10)

Measured on this machine: **speech ~0.024 RMS, room tone ~0.0002 RMS.**
Three guards, all necessary — the ratio alone cannot do the job:

- **Two-stage calibration.** Stage 1 before playback = room floor. Stage 2
  (`keep_max=True`) with the book audible raises the floor above speaker
  leakage. Only stage 1 -> an open desk mic hears the narrator and barge-in
  fires every ~2 s. Only stage 2 -> a quiet passage sets the floor too low.
  Take the max. On speakers stage 2 reads ~24x stage 1; on headphones identical.
- **`MIN_TRIGGER_RMS = 0.004`.** A quiet room calibrates to ~0.00002; times any
  ratio that is still under room tone. **Raising `BARGEIN_VAD_THRESHOLD_RATIO`
  does not fix this** — 7x near-zero is near-zero. The floor is clamped instead.
- **`ATTACK_FRAMES = 3`** (150 ms). One hot frame is a click, a chair, or a
  consonant burst from the speakers. Speech sustains; transients don't.

The noise floor uses the **90th percentile**, not the median — leaked speech is
bursty and a median sits in the gaps between words.

`EnergyVad` only adapts its floor while **inactive**: if ambient sits above the
initial floor, every frame reads as speech, hangover keeps it active, the floor
never updates, and it latches on forever. Hence calibrating before arming.

## Architecture decisions (don't re-litigate these)

| Decision | Why |
|---|---|
| No vector DB | Brute-force cosine = 1.3 ms for a 10-hour book vs ~2200 ms for the LLM call. Chroma would add ~200 MB of deps to save nothing. |
| Position-first retrieval | "What did *that* mean?" is deictic — nothing to embed. The SQLite timestamp window is load-bearing; vector search only serves content-bearing questions. |
| Spoiler guard | Semantic search is capped at the listener's position. Answering a ch.3 question with ch.12 material is a real failure for a book. |
| Local VAD owns barge-in latency | Network round trip too slow (~300 ms vs ~50 ms). That frees the streaming session to run `balanced`, **not** `min_latency` — do not "speed up barge-in" by changing that; it only degrades turn detection. |
| Interrupt timestamp captured at barge-in | By question-end, the sentence they meant has passed. |
| Warm APIs at startup | First Gemini call costs ~3 s more than the rest; that's the on-camera one. |
| Adaptive stitch | 0.45 s for a complete-sounding question, 1.1 s for a fragment. The largest delay we control — bigger than the LLM call. |
| Hold intent | "Hold on." is a placeholder, not a question. Answering it talks over the user and their real question gets dropped. |
| Two tools, mirrored | `passage_at_playhead` turns position into meaning; `go_to_topic` turns meaning into position. Same insight, both directions. Don't add a third without a reason this good. |
| The seek rides the playhead heartbeat | The agent runs on AssemblyAI's servers and cannot touch the page. `go_to_topic` leaves a position in Redis; the browser's once-a-second report collects it. GETDEL, so a jump happens once. |
| Spoiler cap does NOT apply to `go_to_topic` | The cap stops the agent *volunteering* what is ahead. Being asked to go there is consent. |
| Notes live in the browser, not the server | No accounts, no auth, no storage to secure, 11 days before a deadline. The export file is how they move between machines; `/api/context` carries a digest of past *questions* (not answers) for continuity. |
| A user's book lives in Redis, not SQLite | The lambda filesystem is read-only, and the request that builds an index is not the request that reads it. `RedisLibrary` answers `window`/`search`/`duration_hint`/`len`, so the tools never learn which kind of book they hold. |
| Indexing is driven by polling, not a worker | A function is killed at 10 s; transcribing a book takes minutes. Each `GET /api/books/{id}` advances the job one bounded step (poll the transcript, or embed the next 100 chunks) and saves where it got to. The progress bar is a side effect of that, not a fake. |
| Vectors sharded, float16, 100 per key | 100 x 768 x 2 B = 154 KB, ~205 KB base64 — comfortably under Upstash's 1 MB request cap, and one shard is one Gemini batch. `test_books.py` proves float16 still recovers the exact top hit. |
| Times kept apart from text | A window lookup runs on every question and only needs `[[start,end],...]` (~20 B/chunk) plus the one text shard it lands in. Storing them together would drag a whole book across the wire per question. |
| Uploads capped at 4 MB, links uncapped | The platform caps a request body at 4.5 MB. The browser checks size *before* sending so an audiobook gets a sentence, not an edge-level failure. AssemblyAI has no browser-safe upload token (checked 2026-09-19), so the key cannot move to the client. |
| The spine is the UI | Everything here is anchored to a position in time, so the page is a time axis with marks on it, rather than a stack of cards. Structure carries the information. |
| Streaming TTS kept but **measured no win** | 2.35 s vs 2.49 s; replies are ~250 chars so time-to-first-token dominates. Settled — don't redo this experiment. |

## Editing gotchas on this machine (cost real debugging time)

`pathlib.read_text()` / `write_text()` default to **cp1252** on Windows. A patch
script that reads a file containing an em-dash, searches for it, and writes back
will silently fail to match or write mojibake (`â€”`). One such patch "succeeded"
but changed nothing, producing a `NameError` that only surfaced mid-demo.

- Always pass `encoding="utf-8"` to both.
- **Assert the anchor matched before writing.** Never print success blindly.
- The bash heredoc layer eats backslashes: a regex `\b` can arrive as nothing,
  or as a literal backspace (0x08). Build them with `chr(92)`, or use the Edit
  tool — and read the line back afterwards.
- Prefer ASCII in Python source. Keep em-dashes in Markdown.

## Conventions

- Keys live in `Desktop/Bots/master_env`, synced by `Bots/sync_env.py`.
  `check_keys.py` verifies all three vendors — run it before any demo.
  **The Gemini keys in `Desktop/ENV/API.txt` are all dead. Don't use that file.**
- `smoke_test.py` must pass with no API key and no audio hardware.
- Three input devices here: default is the G433 headset mic; `--device 3` is a
  FDUCE M160 desk mic. `mic_check.py --list` enumerates them.
- Demo on headphones. Book + reply out of speakers re-enter the mic; real
  acoustic echo cancellation is unbuilt and is a documented limitation.

## File map

**Web app (the product):**

| File | Role |
|---|---|
| `server/main.py` | FastAPI: `/api/session` (token mint), `/api/playhead`, `/api/context`, `/api/books` (+ `/upload`, `/{id}`), `/tools/passage_at_playhead`, `/tools/go_to_topic`, `/api/health` |
| `server/books.py` | Bring-your-own-book: AssemblyAI transcription, chunking, embedding, and `RedisLibrary` (same surface as `Library`) |
| `server/store.py` | Shared state — playhead, pending seek, prior-session context, current book, and a general kv surface for book indexes. Upstash Redis in prod, memory locally |
| `server/agent.json` | The stored agent: system prompt + the one HTTP tool |
| `scripts/create_agent.py` | Publish/update the agent. Tool URL host must resolve, so deploy first |
| `web/index.html`, `web/app.js` | Browser client (source of truth — copy to `public/`) |
| `public/` | What Vercel actually serves, incl. `audio/relativity.mp3` |
| `api/index.py` | Vercel entry point, re-exports `server.main.app` |
| `vercel.json`, `.vercelignore` | Legacy builds/routes form, on purpose |

**Shared — the novel part:**

| File | Role |
|---|---|
| `echoread/library.py` | Position window + semantic search capped at the playhead |
| `build_index.py` | Audio -> transcript -> chunks -> embeddings -> keyterms |
| `compare_rag.py` | Naive vector RAG vs EchoRead side by side — the video's money shot |

**Probes — how the undocumented API was learned:**

| File | Settles |
|---|---|
| `probe_agent_ws.py` | The `session.update` handshake shape |
| `probe_agent_audio.py` | Every event name, sending real speech |
| `probe_agent_schema.py` | Stored-agent schema, from the validator |
| `probe_voiceagent.py` | Account access to the Voice Agent API |
| `probe_streaming.py` | Streaming v3 turn behaviour (desktop era) |
| `probe_gateway.py` | LLM Gateway access (gated on this account) |

**Retired desktop client** (see the two-codebases note at the top):
`echoread/{player,mic,ears,session,brain,voice,config}.py`, `run_demo.py`,
`mic_check.py`, `ask.py`, `bench_latency.py`, `make_demo_audiobook.py`.

`check_keys.py` verifies all vendors. `smoke_test.py` covers the desktop client
and must pass with no key or hardware; **the web app has no automated tests** —
verify it with `/api/health` and the page's event log.

## Measured numbers (real — use these in the writeup, don't invent others)

- Local barge-in ~50 ms vs ~300 ms for the server round trip
- Speech 0.024 RMS vs room tone 0.0002 RMS
- Brute-force cosine: 1.3 ms for 1800 chunks x 3072-d (a 10-hour book)
- Gemini warm 1.3-2.5 s; cold start ~3 s extra; TTS first byte ~0.8 s
- Total to first spoken word: ~3.1-3.9 s warm (was ~5.4 s cold)
- Einstein chapter: 20.6 min -> 23 chunks, median 50 s / 630 chars
- Demo book: LibriVox `relativity_librivox`, sections 7-9, public domain

## Deployment (live 2026-09-13)

**https://echoread-alpha.vercel.app** — Vercel project `echoread`, scope
`dankam0001s-projects`, agent `agent_b7b84b2a03254f86b3b6f468878ebaa8`.

```bash
TOKEN=$(grep -oE '^VERCEL_API_KEY=.*' ~/Desktop/Bots/master_env | cut -d= -f2-)
vercel deploy --prod --yes --token "$TOKEN" --scope dankam0001s-projects
```

Env vars are set on the Vercel project (ASSEMBLYAI_API_KEY, GEMINI_API_KEY,
UPSTASH_REDIS_REST_URL/TOKEN, ECHOREAD_AGENT_ID). `/api/health` reports whether
the agent is configured and which store is in use — check it after any deploy.

### Four things that will bite on this deployment

1. **Use the ALIAS, never the deployment URL.** `ssoProtection` is
   `all_except_custom_domains`, so `echoread-<hash>-....vercel.app` 302s to
   `vercel.com/sso-api` while `echoread-alpha.vercel.app` serves the real app.
   A tool URL or submission link pointing at a deployment URL will silently
   land on a Vercel login page.
2. **Serverless breaks in-memory state.** The browser POSTs the playhead on one
   lambda; AssemblyAI's tool call lands on another. `server/store.py` keeps it
   in Upstash Redis for exactly this reason, falling back to a dict locally.
   If `/api/health` says `"store":"memory"` in production, the Upstash env vars
   are missing and the tool will intermittently claim it can't find the reader.
3. **`.vercelignore` patterns are not root-anchored.** `audio/` also matched
   `public/audio/` and silently 404'd the audiobook. It is `/audio/` now.
4. **Vercel installs the root `requirements.txt`.** It is deliberately minimal
   (fastapi, pydantic, numpy, google-genai, dotenv). Local tooling —
   assemblyai, sounddevice, soundfile, elevenlabs, uvicorn — lives in
   `requirements-dev.txt`. Adding `sounddevice` to the root file would break
   the build, because there is no PortAudio on a lambda.

`vercel.json` uses the legacy `builds`/`routes` form on purpose: it forwards the
original request path to the function, which the newer `rewrites` form does not,
and FastAPI needs the real path to route.

## Voice Agent websocket protocol (observed live, 2026-09-14)

The docs page 404s, so this was learned from the socket itself via
`probe_agent_ws.py` (handshake shapes) and `probe_agent_audio.py` (real speech
in, every event name out). Trust this over any prose description.

**The handshake, and the bug it caused:**

```js
// CORRECT - agent nested under `session`
{ type: "session.update", session: { agent_id: "agent_..." } }

// WRONG - still answers session.ready, but NO agent is loaded:
// no greeting, no STT, and the microphone appears completely dead
{ type: "session.update", agent_id: "agent_..." }
```

Both return `session.updated` then `session.ready`. Only the nested form makes
the agent greet. This cost a debugging round: the symptom was "the mic isn't
picking up anything", and the mic was fine.

**Events, with the fields that actually carry the payload:**

| Event | Field | Notes |
|---|---|---|
| `session.updated`, `session.ready` | `config`, `session_id`, `resume_token` | `config` echoes the stored agent, useful for verifying the right agent loaded |
| `input.speech.started` / `input.speech.stopped` | `timestamp` | the barge-in signal |
| `transcript.user.delta` / `transcript.user` | **`text`** | not `transcript` |
| `transcript.agent.delta` / `transcript.agent` | **`text`** | `.delta` also has `start_ms`/`end_ms` |
| `reply.started`, `reply.done` | `reply_id`, `status` | `status: "interrupted"` on barge-in |
| `reply.audio` | **`data`** | base64 PCM16 @24 kHz. Note the asymmetry: outbound is `input.audio` with `audio`, inbound is `reply.audio` with `data` |
| `session.error` | `code`, `message` | e.g. `invalid_format` for an unknown message type |

**HTTP tool calls are invisible on the websocket** — no `tool.*` events appear,
because AssemblyAI calls the endpoint server-to-server. To confirm a tool fired,
check the backend logs or `/api/health`, not the client.

A mid-question pause still splits into separate user turns here, exactly as it
did on the streaming API, and the agent may answer the fragment. Tune via the
agent's `turn_detection` rather than re-implementing the old stitching.

**The page has a visible event log** (`web/index.html`, `#log`). Every stall in
this project traced back to not being able to see what the machine was doing;
keep it.

### Three more that cost a round each (2026-09-15)

- **Updating a stored agent is `PUT /v1/agents/{id}`**, which merges the fields
  you send. PATCH returns 405, and OPTIONS reports `Allow: GET`, which is wrong;
  don't trust it. Never send a stripped-down body while probing: send the full
  spec. `scripts/create_agent.py` does this.
- **Never give the tool a parameter the model can't fill.** It used to take a
  `session_id` "given to you at the start of the conversation". Nothing gives
  it one, so the agent asked the *listener* for their session ID, out loud. The
  tool now takes only `search`; the backend uses the freshest playhead. The
  real fix for multiple listeners is out-of-band, not a parameter.
- (Browser) `createScriptProcessor` accepts only powers of two; `1200` threw
  `IndexSizeError` and read as "microphone blocked".

## Bring your own audiobook (built 2026-09-19)

The shipped Einstein chapter is now a demo, not the whole product. Library panel
in the page: paste a direct audio link, or drop a file under 4 MB.

**Verified end to end against a real LibriVox URL** (`relativity_10-12`,
deliberately a *different* section from the shipped 07-09, so a wrong index is
obvious): transcribed, 14 chunks, ready in ~2.5 min; `passage_at_playhead` at
10:00 returned chapter 11 material that is **not in the shipped index**;
`go_to_topic("the Lorentz transformation")` resolved to 7:04; the spoiler cap
correctly returned no lookback at t=120.

`test_books.py` covers the Redis path with **no keys and no network** — it runs
the real `RedisStore` against a stand-in for Upstash REST. It is the first
automated test the web app has had. Run it after touching `store.py` or
`books.py`.

Notes and the resume point are now **per book** (`echoread:notes:<book id>`).
Questions about one book were noise against another.

## Open and untested (as of 2026-09-19)

- **The bring-your-own-book work is NOT DEPLOYED.** It was built, run and
  verified locally on 2026-09-19, but the production deploy was blocked by a
  permission prompt and never ran. `echoread-alpha.vercel.app` is still serving
  the previous build. Nothing about the feature is live until someone runs the
  deploy in the Deployment section and re-checks `/api/health`. The Upstash
  round trip is covered by `test_books.py` against a stand-in, **not** against
  real Upstash — confirm a book reaches `ready` on the live site before filming
  it.
- **Speaker mode (no headphones) has never been tested.** The old "headphones
  are non-negotiable" line is a *desktop-era* constraint that got repeated by
  mistake: the browser cancels its own output (the book and the reply both play
  through the page, and the mic asks for `echoCancellation`), and the book ducks
  to 12% on speech. So it *should* work on speakers. Nobody has watched it. Two
  failure modes to look for: the agent replying to its own voice, or ducking +
  cancellation eating the listener's voice too. The user was asked to test this
  before filming - find out what they saw.
- **Notes, export/import, resume and `/api/context` shipped 2026-09-19 and have
  been verified by curl, not by a human in a browser.** The continuity path
  (browser POSTs past questions -> Redis -> appended to the tool response) is
  confirmed working server-side.
- **The agent still may answer a fragment** if the listener pauses mid-question.
  Fix via the stored agent's turn detection, not client-side stitching.
- The agent does not reliably pass `session_id` to tools, so the store falls
  back to the freshest playhead. Correct for one listener; wrong for many.

## Demo / submission material

- Shot list, positioning and paste-ready submission copy:
  https://claude.ai/artifact/9nXiQ5BQPHngK2FCpMJMsR
- The video is **2:00**, seven shots. Shot 4 (`compare_rag.py`, the 17:25 case)
  is the one that carries the entry.
- **Do not pitch the driving use case.** It invites "isn't that illegal with
  headphones?" and it is the weakest example. Lead with learning by ear:
  blindness, dyslexia, or reading being work.

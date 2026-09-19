// EchoRead browser client.
//
// The websocket runs here, not in the backend: the server mints a short-lived
// token and the browser connects straight to AssemblyAI. The API key never
// reaches this file.
//
// The browser owns two things the agent cannot see: the audiobook playhead,
// which it reports to the backend so the tool can read it, and the ducking of
// playback the moment someone starts talking.

const WS_URL = "wss://agents.assemblyai.com/v1/ws";
const RATE = 24000;          // Voice Agent API is 24 kHz PCM16 both ways
// ScriptProcessor only accepts powers of two (256..16384); anything else throws
// IndexSizeError before a frame is captured. 2048 samples is ~85 ms at 24 kHz,
// inside the 50-1000 ms chunk range AssemblyAI streaming asks for.
const FRAME = 2048;

const el = (id) => document.getElementById(id);
const book = el("book");
const statusEl = el("status");
const transcriptEl = el("transcript");
const startBtn = el("start");

let ws, session, micCtx, micNode, outCtx, playCursor = 0;
let ducked = false, agentSpeaking = false, framesSent = 0;

function setStatus(text, state) {
  statusEl.textContent = text;
  statusEl.dataset.state = state || "idle";
}

// A visible event log. Every time this project has stalled, the cause was not
// being able to see what the machine was doing, so the page shows it.
const debugLines = [];
function debug(msg) {
  const log = document.getElementById("log");
  if (!log) return;
  const t = new Date().toLocaleTimeString([], { hour12: false });
  debugLines.unshift(t + "  " + msg);
  debugLines.length = Math.min(debugLines.length, 60);
  log.textContent = debugLines.join(String.fromCharCode(10));
}

function line(who, text, cls) {
  const p = document.createElement("p");
  p.className = "line " + (cls || who);
  p.innerHTML = `<span class="who">${who}</span>${text}`;
  transcriptEl.appendChild(p);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
  return p;
}

// ---------- the book ----------

function duck() {
  if (ducked) return;
  ducked = true;
  book.volume = 0.12;
  document.body.classList.add("listening");
}

function pauseBook() {
  book.pause();
  document.body.classList.add("listening");
}

function resumeBook() {
  ducked = false;
  book.volume = 1;
  document.body.classList.remove("listening");
  if (book.paused) book.play().catch(() => {});
}

// The tool runs on AssemblyAI's servers and has no idea where playback is, so
// the position has to be reported out of band.
async function reportPlayhead() {
  if (!session) return;
  try {
    await fetch("/api/playhead", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: session.session_id, seconds: book.currentTime }),
    });
  } catch (_) { /* a dropped report is harmless; the next one is a second away */ }
}

// ---------- audio plumbing ----------

function floatToPCM16(input) {
  const out = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function toBase64(bytes) {
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

async function startMic() {
  // Echo cancellation is why this runs in a browser at all: the book and the
  // agent's own reply come out of the speakers and back into the mic. The
  // native AEC removes them, which a plain desktop capture cannot do.
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      sampleRate: RATE,
      channelCount: 1,
    },
  });
  micCtx = new AudioContext({ sampleRate: RATE });
  const src = micCtx.createMediaStreamSource(stream);
  micNode = micCtx.createScriptProcessor(FRAME, 1, 1);
  micNode.onaudioprocess = (e) => {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const pcm = floatToPCM16(e.inputBuffer.getChannelData(0));
    ws.send(JSON.stringify({
      type: "input.audio",
      audio: toBase64(new Uint8Array(pcm.buffer)),
    }));
    framesSent++;
    if (framesSent % 100 === 0) debug(`${framesSent} audio frames sent`);
  };
  src.connect(micNode);
  micNode.connect(micCtx.destination);
}

function playReply(b64) {
  // Schedule each chunk after the previous one. The device drains at exactly
  // 24 kHz, so consecutive buffers join without gaps; sleep-based timing drifts
  // and produces clicks.
  if (!outCtx) outCtx = new AudioContext({ sampleRate: RATE });
  const bin = atob(b64);
  const pcm = new Int16Array(bin.length / 2);
  for (let i = 0; i < pcm.length; i++) {
    pcm[i] = (bin.charCodeAt(i * 2 + 1) << 8) | bin.charCodeAt(i * 2);
  }
  const buf = outCtx.createBuffer(1, pcm.length, RATE);
  const ch = buf.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;

  const node = outCtx.createBufferSource();
  node.buffer = buf;
  node.connect(outCtx.destination);
  const now = outCtx.currentTime;
  if (playCursor < now) playCursor = now;
  node.start(playCursor);
  playCursor += buf.duration;
}

function flushReply() {
  playCursor = 0;
}

// ---------- session ----------

async function start() {
  startBtn.disabled = true;
  setStatus("connecting", "busy");
  try {
    session = await (await fetch("/api/session")).json();
    if (session.detail) throw new Error(session.detail);
  } catch (e) {
    setStatus("backend error: " + e.message, "error");
    startBtn.disabled = false;
    return;
  }

  ws = new WebSocket(`${WS_URL}?token=${encodeURIComponent(session.token)}`);

  ws.onopen = () => {
    // The agent id MUST be nested under `session`. Sent at the top level the
    // socket still answers session.ready, but no agent is loaded: no greeting,
    // no STT, and the microphone appears dead. Verified against the live
    // socket in probe_agent_ws.py.
    ws.send(JSON.stringify({
      type: "session.update",
      session: { agent_id: session.agent_id },
    }));
    debug("sent session.update for " + session.agent_id);
    setStatus("connected", "busy");
  };

  ws.onerror = () => setStatus("websocket error", "error");
  ws.onclose = () => {
    // Keep an error on screen rather than replacing it with "disconnected".
    if (statusEl.dataset.state !== "error") setStatus("disconnected", "idle");
    startBtn.disabled = false;
  };

  let partial = null;

  ws.onmessage = async (ev) => {
    const m = JSON.parse(ev.data);
    if (m.type !== "reply.audio") debug(m.type);
    switch (m.type) {
      case "session.ready":
        try {
          await startMic();
          debug("microphone live at " + micCtx.sampleRate + " Hz");
        } catch (err) {
          debug("MIC FAILED: " + err.name + " - " + err.message);
          setStatus("microphone blocked: " + err.name, "error");
          ws.close();
          return;
        }
        await reportPlayhead();
        setInterval(reportPlayhead, 1000);
        book.play().catch(() => {});
        setStatus("listening - just talk", "live");
        break;

      case "input.speech.started":
        // Duck immediately; commit to a full pause once words arrive.
        duck();
        setStatus("you're talking", "live");
        break;

      case "transcript.user.delta":
        if (!partial) partial = line("you", "", "you partial");
        partial.innerHTML = `<span class="who">you</span>${m.text || ""}`;
        pauseBook();
        break;

      case "transcript.user":
        if (partial) partial.remove();
        partial = null;
        line("you", m.text || "");
        pauseBook();
        setStatus("thinking", "busy");
        break;

      case "input.speech.stopped":
        setStatus("thinking", "busy");
        break;

      case "reply.started":
        agentSpeaking = true;
        flushReply();
        setStatus("answering", "busy");
        break;

      case "reply.audio":
        playReply(m.data || m.audio);
        break;

      case "transcript.agent":
        line("echoread", m.text || "");
        break;

      case "reply.done":
        agentSpeaking = false;
        if (m.status === "interrupted") flushReply();
        // Back up slightly so the run-up to the question is re-heard.
        book.currentTime = Math.max(0, book.currentTime - 3);
        resumeBook();
        setStatus("listening - just talk", "live");
        break;

      case "error":
      case "session.error":
        setStatus("agent error: " + (m.message || ""), "error");
        break;
    }
  };
}

startBtn.addEventListener("click", start);

book.addEventListener("timeupdate", () => {
  const t = book.currentTime;
  el("clock").textContent =
    `${String(Math.floor(t / 60)).padStart(2, "0")}:${String(Math.floor(t % 60)).padStart(2, "0")}`;
});

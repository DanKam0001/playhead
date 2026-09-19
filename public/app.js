// Playhead browser client.
//
// The websocket runs here, not in the backend: the server mints a short-lived
// token and the browser connects straight to AssemblyAI. The API key never
// reaches this file.
//
// The browser owns three things the agent cannot see: the audiobook playhead,
// which book is loaded, and the ducking of playback the moment someone starts
// talking. The first two are reported to the backend so the tools can read
// them; the third never leaves the page.

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
let ducked = false, agentSpeaking = false, framesSent = 0, justSeeked = false;

function setStatus(text, state) {
  statusEl.textContent = text;
  statusEl.dataset.state = state || "idle";
}

// A visible event log. Every time this project has stalled, the cause was not
// being able to see what the machine was doing, so the page shows it.
const debugLines = [];
function debug(msg) {
  const log = el("log");
  if (!log) return;
  const t = new Date().toLocaleTimeString([], { hour12: false });
  debugLines.unshift(t + "  " + msg);
  debugLines.length = Math.min(debugLines.length, 60);
  log.textContent = debugLines.join(String.fromCharCode(10));
}

function clockText(t) {
  if (!isFinite(t) || t < 0) t = 0;
  const m = String(Math.floor(t / 60)).padStart(2, "0");
  const s = String(Math.floor(t % 60)).padStart(2, "0");
  return `${m}:${s}`;
}

// ---------- the conversation ----------

// The opening copy, kept in JS so clearing the conversation can put it back
// rather than leaving an empty panel.
const LEDE = [
  'Press <b>Start listening</b>, let it play, then just talk over it. Try '
  + '<b>&ldquo;wait, what did that last part mean?&rdquo;</b> &mdash; the question names no '
  + 'topic, so there is nothing to search for. Playhead answers it from where you are '
  + 'in the audio.',
  'Or say <b>&ldquo;take me to the part about the train&rdquo;</b> and the book moves there.',
];

let ledeCleared = false;
function turn(who, text, cls) {
  if (!ledeCleared) {
    transcriptEl.querySelectorAll(".lede").forEach((n) => n.remove());
    ledeCleared = true;
  }
  const wrap = document.createElement("div");
  wrap.className = "turn " + (cls || who);
  const at = document.createElement("span");
  at.className = "at";
  at.textContent = clockText(book.currentTime);
  const body = document.createElement("div");
  body.className = "body";
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who;
  const p = document.createElement("p");
  p.textContent = text;
  body.append(label, p);
  wrap.append(at, body);
  transcriptEl.appendChild(wrap);
  wrap.scrollIntoView({ block: "nearest", behavior: "smooth" });
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

// The tool runs on AssemblyAI's servers and has no idea where playback is, or
// which book is loaded, so both are reported out of band.
async function reportPlayhead() {
  if (!session) return;
  try {
    const res = await fetch("/api/playhead", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: session.session_id,
        seconds: book.currentTime,
        book: currentBook ? currentBook.id : null,
      }),
    });
    // The reply can carry a jump. go_to_topic runs on AssemblyAI's servers and
    // cannot touch this page, so it leaves a position behind and we collect it.
    const data = await res.json();
    if (typeof data.seek === "number") applySeek(data.seek);
  } catch (_) { /* a dropped report is harmless; the next one is a second away */ }
}

function applySeek(seconds) {
  book.currentTime = seconds;
  // Suppress the usual rewind-on-resume: the listener asked to be here, and
  // backing up 3 s from a deliberate jump is just wrong.
  justSeeked = true;
  debug(`jumped to ${clockText(seconds)}`);
  turn("playhead", "Jumped to " + clockText(seconds), "system");
}

// ---------- the spine ----------
//
// The whole book as one vertical axis: how far in the listener is, and where
// each question was asked. Doubles as the scrubber, so position is read and
// set in the same place.

// On a phone the rail sits above the conversation rather than beside it, so the
// axis lies flat and the questions become a list under it. Same information,
// turned ninety degrees.
const narrow = () => window.matchMedia("(max-width: 820px)").matches;

function paintSpine() {
  const dur = duration();
  const frac = dur ? Math.min(1, book.currentTime / dur) : 0;
  const pct = (frac * 100).toFixed(2) + "%";
  const fill = el("spinefill").style;
  const cur = el("spinecursor").style;
  if (narrow()) {
    fill.width = pct; fill.height = "100%";
    cur.left = pct; cur.top = "50%";
  } else {
    fill.height = pct; fill.width = "100%";
    cur.top = pct; cur.left = "50%";
  }
  el("clock").textContent = clockText(book.currentTime);
  el("total").textContent = "/ " + clockText(dur);
  const spine = el("spine");
  spine.setAttribute("aria-valuemax", Math.round(dur));
  spine.setAttribute("aria-valuenow", Math.round(book.currentTime));
  spine.setAttribute("aria-valuetext", clockText(book.currentTime));
}

function duration() {
  if (isFinite(book.duration) && book.duration > 0) return book.duration;
  return currentBook && currentBook.duration ? currentBook.duration : 0;
}

function renderMarks() {
  const holder = el("spinemarks");
  const dur = duration();
  // Without a duration there is nowhere on the axis to put a mark -- but
  // imported notes arrive before the audio has reported its length, and
  // dropping them silently makes an import look like it did nothing. Fall
  // back to the same flat list the phone layout uses until the length lands.
  const flow = narrow() || !dur;
  holder.classList.toggle("flow", flow);
  holder.textContent = "";
  notes.forEach((n) => {
    const b = document.createElement("button");
    b.className = "mark";
    b.type = "button";
    // Beside the axis at the right height on a wide screen; a plain list on a
    // phone, where there is no room to hang labels off a bar.
    if (!flow) b.style.top = Math.min(99, (n.t / dur) * 100).toFixed(2) + "%";
    b.setAttribute("aria-label", `Play from ${clockText(n.t)}: ${n.q}`);
    if (flow) {
      const at = document.createElement("i");
      at.className = "markat";
      at.textContent = clockText(n.t);
      b.append(at);
    }
    const label = document.createElement("span");
    label.textContent = n.q;
    b.append(label);
    b.addEventListener("click", () => {
      book.currentTime = n.t;
      book.play().catch(() => {});
    });
    holder.appendChild(b);
  });
}

function seekFromEvent(e) {
  const rect = el("spine").getBoundingClientRect();
  const dur = duration();
  if (!dur) return;
  const frac = narrow()
    ? (e.clientX - rect.left) / rect.width
    : (e.clientY - rect.top) / rect.height;
  book.currentTime = Math.max(0, Math.min(dur, frac * dur));
  paintSpine();
}

// ---------- notes ----------
//
// The questions someone asks are a map of where the book lost them, so they
// are worth keeping. This is all per-browser: no account, no server copy. The
// export file is how notes move between machines, and it is also what the
// import reads back. Notes are per book -- questions about one book are noise
// against another.

let notes = [];
let pendingQ = null;

const notesKey = () => "playhead:notes:" + (currentBook ? currentBook.id : "relativity");
const posKey = () => "playhead:pos:" + (currentBook ? currentBook.id : "relativity");

// The product was called EchoRead until 2026-09-19. Anyone who used it before
// then has notes under the old prefix, and a rename that silently eats them is
// the worst kind of bug: invisible, and only to the people who used it most.
function migrateOldKeys() {
  try {
    Object.keys(localStorage)
      .filter((k) => k.startsWith("echoread:"))
      .forEach((k) => {
        const moved = "playhead:" + k.slice("echoread:".length);
        if (localStorage.getItem(moved) === null) {
          localStorage.setItem(moved, localStorage.getItem(k));
        }
        localStorage.removeItem(k);
      });
  } catch (_) { /* no storage at all is fine; there is nothing to migrate */ }
}

// Private windows and blocked site data make these throw rather than return
// empty, so every access is guarded and the page works with no storage at all.
function storageGet(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw ? JSON.parse(raw) : fallback;
  } catch (_) { return fallback; }
}

function storageSet(key, value) {
  try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* ignore */ }
}

function noteQuestion(text) {
  // Capture the position now: by the time the answer arrives the book may have
  // moved, and the note is only useful if it points where the question was asked.
  pendingQ = { t: book.currentTime, q: text };
}

function noteAnswer(text) {
  if (!pendingQ || !text) return;
  notes.push({ t: Math.round(pendingQ.t), q: pendingQ.q, a: text, at: new Date().toISOString() });
  pendingQ = null;
  notes = notes.slice(-200);
  storageSet(notesKey(), notes);
  renderNotes();
}

function renderNotes() {
  el("noteactions").hidden = notes.length === 0;
  el("railnote").textContent = notes.length
    ? `${notes.length} question${notes.length > 1 ? "s" : ""} on this book. Click one to play from there.`
    : "Your questions get pinned to the spine at the moment you asked them — a map of where the book lost you.";
  renderMarks();
}

function exportNotes() {
  const lines = ["# Playhead notes", "",
                 "Book: " + (currentBook ? currentBook.title : "relativity"),
                 "Exported: " + new Date().toLocaleString(), ""];
  notes.forEach((n) => {
    lines.push("## " + clockText(n.t), "", "**You asked:** " + n.q, "", n.a, "");
  });
  // A machine-readable copy rides along in a comment, so one file is both
  // pleasant to read and importable.
  lines.push("<!-- playhead:data " + JSON.stringify(notes) + " -->");

  const blob = new Blob([lines.join(String.fromCharCode(10))], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "playhead-notes.md";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function importNotes(file) {
  const text = await file.text();
  // Files exported before the rename carry the old marker; still read them.
  let marker = "<!-- playhead:data ";
  let i = text.indexOf(marker);
  if (i < 0) {
    marker = "<!-- echoread:data ";
    i = text.indexOf(marker);
  }
  let incoming = null;
  try {
    incoming = i >= 0
      ? JSON.parse(text.slice(i + marker.length, text.lastIndexOf("-->")))
      : JSON.parse(text);
  } catch (_) { incoming = null; }

  if (!Array.isArray(incoming)) {
    setStatus("no Playhead notes in that file", "error");
    return;
  }
  const seen = new Set(notes.map((n) => n.t + "|" + n.q));
  incoming.forEach((n) => {
    if (n && n.q && !seen.has(n.t + "|" + n.q)) notes.push(n);
  });
  notes.sort((x, y) => x.t - y.t);
  storageSet(notesKey(), notes);
  renderNotes();
  // Imported questions are only useful to the agent if it hears about them.
  // They are normally handed over when a session starts, so importing into a
  // session already running has to push them itself -- otherwise the import
  // does nothing until the next reload, which is the opposite of the point.
  pushContext();
  debug("imported " + incoming.length + " notes");
  setStatus(`imported ${incoming.length} notes`, "idle");
}

// Hand this listener's past questions to the backend, so the tool can remind
// the agent what they have already asked about. Questions only: enough for
// continuity, and far less to confuse it than whole past conversations. Called
// when a session starts, and again after an import during one.
function pushContext() {
  if (!session || !notes.length) return;
  fetch("/api/context", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: session.session_id,
      questions: notes.slice(-8).map((n) => ({ t: n.t, q: n.q })),
    }),
  }).catch(() => { /* continuity is a bonus, never a blocker */ });
}

// Where they stopped, so the next visit can pick it up.
function rememberPosition() {
  if (book.currentTime > 5) storageSet(posKey(), Math.round(book.currentTime));
}

function offerResume() {
  const at = storageGet(posKey(), 0);
  const bar = el("resume");
  if (!at || at < 10) { bar.hidden = true; return; }
  el("resumeat").textContent = clockText(at);
  bar.hidden = false;
  el("resumebtn").onclick = () => {
    book.currentTime = at;
    bar.hidden = true;
  };
  el("resumedismiss").onclick = () => {
    bar.hidden = true;
    storageSet(posKey(), 0);
  };
}

// ---------- the library ----------
//
// A book someone adds is transcribed, chunked at the reader's own pauses and
// embedded against its timestamps -- the same index the shipped book uses, so
// both tools work on it without knowing the difference. Indexing is driven by
// polling: each poll advances the job one step, which is also where the
// progress number comes from.

let currentBook = null;
let shelf = [];
let suggested = [];
let limits = { upload_mb: 4, source_mb: 150 };   // replaced by the server's own
let localAudio = {};     // book id -> object URL, for files added this session
let needsFile = false;   // the selected book came from a file we no longer hold

// Scopes the shelf to this browser. There are no accounts, and a single shared
// shelf would put whatever a stranger added on the front page of a live demo.
// An identifier, not a credential -- it guards tidiness, not secrets.
function clientId() {
  let id = storageGet("playhead:client", null);
  if (!id) {
    id = "c" + Math.random().toString(36).slice(2) + Date.now().toString(36);
    storageSet("playhead:client", id);
  }
  return id;
}

const withClient = (extra) => Object.assign({ "x-client-id": clientId() }, extra || {});

function addStatus(msg, bad) {
  const p = el("addstatus");
  p.textContent = msg || "";
  if (bad) p.dataset.bad = "1"; else delete p.dataset.bad;
}

async function loadShelf() {
  try {
    const data = await (await fetch("/api/books", { headers: withClient() })).json();
    shelf = data.books || [];
    suggested = data.suggested || [];
    if (data.limits) limits = data.limits;
  } catch (_) { shelf = []; }
  renderShelf();
  renderSuggested();
  renderLimits();
  if (!currentBook) {
    const saved = storageGet("playhead:book", null);
    const pick = shelf.find((b) => b.id === saved) || shelf[0];
    if (pick) selectBook(pick, true);
  }
  // Anything still indexing keeps getting nudged until it is done.
  shelf.filter((b) => b.status === "transcribing" || b.status === "indexing")
       .forEach((b) => pollBook(b.id));
}

function renderShelf() {
  const list = el("shelf");
  list.textContent = "";
  shelf.forEach((b) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    if (currentBook && b.id === currentBook.id) btn.setAttribute("aria-current", "true");

    const t = document.createElement("span");
    t.className = "t";
    t.textContent = b.title;
    const m = document.createElement("span");
    m.className = "m";
    m.textContent = b.status === "ready"
      ? `${clockText(b.duration)} · ${b.chunks}`
      : (b.status === "failed" ? "failed" : b.progress + "%");
    btn.append(t, m);

    if (b.status !== "ready" && b.status !== "failed") {
      const bar = document.createElement("span");
      bar.className = "bar";
      const i = document.createElement("i");
      i.style.width = b.progress + "%";
      bar.appendChild(i);
      btn.appendChild(bar);
    }
    if (b.status === "failed" && b.error) btn.title = b.error;

    btn.addEventListener("click", () => {
      if (b.status !== "ready") {
        addStatus(b.status === "failed" ? "That one failed: " + b.error : "Still indexing — give it a moment.", b.status === "failed");
        return;
      }
      selectBook(b);
    });
    li.appendChild(btn);

    // A failed book is usually a host that would not hand the file over, so
    // the useful control is "try again", not a dead row.
    if (b.status === "failed" && b.audio_url) {
      const again = document.createElement("button");
      again.className = "retry";
      again.type = "button";
      again.textContent = "Try again";
      again.addEventListener("click", (e) => {
        e.stopPropagation();
        shelf = shelf.filter((x) => x.id !== b.id);
        renderShelf();
        addByUrl(b.audio_url, b.title);
      });
      li.appendChild(again);
    }
    list.appendChild(li);
  });
}

function selectBook(b, quiet) {
  currentBook = b;
  storageSet("playhead:book", b.id);
  el("booktitle").textContent = b.title;

  const src = b.audio_url || localAudio[b.id] || "";
  needsFile = !src;
  if (src) {
    book.src = src;
    book.load();
  }
  book.pause();
  setPlayIcon(false);

  // Notes and the resume point belong to the book, not the browser.
  notes = storageGet(notesKey(), []);
  pendingQ = null;
  renderNotes();
  offerResume();
  paintSpine();
  renderShelf();

  if (needsFile) {
    addStatus("“" + b.title + "” was added from a file on this device. Choose it again below to play it — the index is already built.", false);
    el("librarybtn").setAttribute("aria-expanded", "true");
    el("library").hidden = false;
  } else if (!quiet) {
    addStatus("");
  }
  // Tell the backend straight away, so a question asked before the first
  // heartbeat still reaches the right book.
  if (session) reportPlayhead();
}

function renderLimits() {
  const drop = el("droplimit");
  if (drop) drop.textContent = `(up to ${limits.upload_mb} MB — longer books need a link)`;
  const src = el("sourcelimit");
  if (src) src.textContent = `${limits.source_mb} MB`;
}

function renderSuggested() {
  const list = el("suggested");
  if (!list) return;
  list.textContent = "";
  const have = new Set(shelf.map((b) => b.audio_url));
  suggested.filter((s) => !have.has(s.audio_url)).forEach((s) => {
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    const t = document.createElement("span");
    t.className = "t";
    t.textContent = s.title;
    const note = document.createElement("span");
    note.className = "m";
    note.textContent = s.note || "";
    btn.append(t, note);
    btn.addEventListener("click", () => addByUrl(s.audio_url, s.title));
    li.appendChild(btn);
    list.appendChild(li);
  });
  el("suggestedwrap").hidden = list.children.length === 0;
}

async function pollBook(id) {
  for (let i = 0; i < 240; i++) {
    await new Promise((r) => setTimeout(r, 2500));
    let rec;
    try {
      rec = await (await fetch("/api/books/" + encodeURIComponent(id),
                               { headers: withClient() })).json();
    } catch (_) { continue; }
    const at = shelf.findIndex((b) => b.id === rec.id);
    if (at >= 0) shelf[at] = rec; else shelf.push(rec);
    if (currentBook && currentBook.id === rec.id) currentBook = rec;
    renderShelf();
    if (rec.status === "ready") {
      addStatus("“" + rec.title + "” is ready — " + rec.chunks + " passages indexed.");
      debug("indexed " + rec.id + ": " + rec.chunks + " chunks");
      return rec;
    }
    if (rec.status === "failed") {
      addStatus("Indexing failed: " + rec.error, true);
      return rec;
    }
    addStatus(rec.status === "transcribing"
      ? "Transcribing — this takes a minute or two for a chapter."
      : `Indexing passages … ${rec.progress}%`);
  }
}

async function addByUrl(presetUrl, presetTitle) {
  const url = (presetUrl || el("addurl").value).trim();
  if (!url) return;
  el("addbtn").disabled = true;
  addStatus("Sending it off to be transcribed …");
  try {
    const res = await fetch("/api/books", {
      method: "POST",
      headers: withClient({ "Content-Type": "application/json" }),
      body: JSON.stringify({ audio_url: url, title: presetTitle || "" }),
    });
    const rec = await res.json();
    if (!res.ok) throw new Error(rec.detail || "could not add that");
    if (!presetUrl) el("addurl").value = "";
    shelf.push(rec);
    renderShelf();
    renderSuggested();
    pollBook(rec.id);
  } catch (e) {
    addStatus(e.message, true);
  } finally {
    el("addbtn").disabled = false;
  }
}

async function addByFile(file) {
  if (!file) return;
  // Re-binding audio to a book whose index already exists: no upload, no
  // second transcription, just give the player something to play.
  if (needsFile && currentBook) {
    localAudio[currentBook.id] = URL.createObjectURL(file);
    needsFile = false;
    book.src = localAudio[currentBook.id];
    book.load();
    addStatus("Playing “" + currentBook.title + "” from your copy.");
    return;
  }
  // Checked here so an oversized file gets a sentence instead of an
  // edge-level failure. The ceiling comes from the deployment, not from this
  // file, so raising the setting moves the check and the wording together.
  if (file.size > limits.upload_mb * 1e6) {
    addStatus("That file is " + (file.size / 1e6).toFixed(1) + " MB, and uploads here stop at "
      + limits.upload_mb + " MB. Put it somewhere with a direct link "
      + "(archive.org, Dropbox, S3) and paste the link above.", true);
    return;
  }
  addStatus("Uploading …");
  try {
    const res = await fetch("/api/books/upload", {
      method: "POST",
      headers: withClient({ "Content-Type": "application/octet-stream",
                            "x-book-title": file.name.replace(/\.[^.]+$/, "") }),
      body: file,
    });
    const rec = await res.json();
    if (!res.ok) throw new Error(rec.detail || "upload failed");
    // The browser plays the listener's own copy; AssemblyAI got its own.
    localAudio[rec.id] = URL.createObjectURL(file);
    shelf.push(rec);
    renderShelf();
    pollBook(rec.id);
  } catch (e) {
    addStatus(e.message, true);
  }
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
  // Already opened by unlockAudio() inside the tap; reuse it rather than
  // building a second one, which iOS would hand back suspended.
  if (!micCtx) micCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RATE });
  if (micCtx.state === "suspended") await micCtx.resume();
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
  if (!outCtx) outCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RATE });
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

// iOS will not let a script start audio unless it can trace the call back to a
// real tap, and that trace is lost across the first `await`. So everything that
// needs unlocking is opened here, synchronously, at the top of the click
// handler: the two AudioContexts, and the <audio> element itself, which is
// played and immediately paused purely to mark it as user-started. Without
// this the agent works perfectly on an iPhone and the book stays silent.
function unlockAudio() {
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!outCtx) outCtx = new Ctx({ sampleRate: RATE });
    if (!micCtx) micCtx = new Ctx({ sampleRate: RATE });
    if (outCtx.state === "suspended") outCtx.resume();
    if (micCtx.state === "suspended") micCtx.resume();
    const p = book.play();
    if (p && p.then) p.then(() => book.pause()).catch((e) => debug("unlock: " + e.name));
  } catch (err) {
    debug("audio unlock failed: " + err.name);
  }
}

async function start() {
  if (needsFile) {
    addStatus("This book has no audio loaded — choose the file below first.", true);
    return;
  }
  unlockAudio();
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

  pushContext();

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
        setPlayIcon(true);
        setStatus("listening - just talk", "live");
        break;

      case "input.speech.started":
        // Duck immediately; commit to a full pause once words arrive.
        duck();
        setStatus("you're talking", "live");
        break;

      case "transcript.user.delta":
        if (!partial) partial = turn("you", "", "you partial");
        partial.textContent = m.text || "";
        pauseBook();
        break;

      case "transcript.user":
        if (partial) partial.closest(".turn").remove();
        partial = null;
        turn("you", m.text || "");
        noteQuestion(m.text || "");
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
        turn("playhead", m.text || "");
        noteAnswer(m.text || "");
        break;

      case "reply.done":
        agentSpeaking = false;
        if (m.status === "interrupted") flushReply();
        // Back up slightly so the run-up to the question is re-heard — unless
        // the agent just moved us somewhere on purpose.
        if (justSeeked) justSeeked = false;
        else book.currentTime = Math.max(0, book.currentTime - 3);
        resumeBook();
        setPlayIcon(true);
        setStatus("listening - just talk", "live");
        break;

      case "error":
      case "session.error":
        setStatus("agent error: " + (m.message || ""), "error");
        break;
    }
  };
}

// ---------- wiring ----------

const PLAY_PATH = "M0 0l12 7-12 7z";
const PAUSE_PATH = "M0 0h4v14H0zM8 0h4v14H8z";
function setPlayIcon(playing) {
  el("playicon").firstElementChild.setAttribute("d", playing ? PAUSE_PATH : PLAY_PATH);
  el("play").setAttribute("aria-label", playing ? "Pause the book" : "Play the book");
}

startBtn.addEventListener("click", start);

el("play").addEventListener("click", () => {
  if (book.paused) book.play().catch(() => {}); else book.pause();
});
book.addEventListener("play", () => setPlayIcon(true));
book.addEventListener("pause", () => setPlayIcon(false));
book.addEventListener("loadedmetadata", () => { paintSpine(); renderMarks(); });

let lastRemembered = 0;
book.addEventListener("timeupdate", () => {
  paintSpine();
  if (book.currentTime - lastRemembered > 5 || book.currentTime < lastRemembered) {
    lastRemembered = book.currentTime;
    rememberPosition();
  }
});
book.addEventListener("pause", rememberPosition);
window.addEventListener("beforeunload", rememberPosition);

// Spine as scrubber: press and drag anywhere along the axis.
let dragging = false;
el("spine").addEventListener("pointerdown", (e) => {
  dragging = true;
  el("spine").setPointerCapture(e.pointerId);
  seekFromEvent(e);
});
el("spine").addEventListener("pointermove", (e) => { if (dragging) seekFromEvent(e); });
el("spine").addEventListener("pointerup", () => { dragging = false; });
el("spine").addEventListener("keydown", (e) => {
  const step = e.key === "ArrowUp" || e.key === "ArrowLeft" ? -15
             : e.key === "ArrowDown" || e.key === "ArrowRight" ? 15 : 0;
  if (!step) return;
  e.preventDefault();
  book.currentTime = Math.max(0, Math.min(duration(), book.currentTime + step));
  paintSpine();
});

el("librarybtn").addEventListener("click", () => {
  const open = el("library").hidden;
  el("library").hidden = !open;
  el("librarybtn").setAttribute("aria-expanded", String(open));
});

el("addbtn").addEventListener("click", () => addByUrl());
el("addurl").addEventListener("keydown", (e) => { if (e.key === "Enter") addByUrl(); });

// The spine changes axis with the layout, so it has to be repainted when the
// layout changes under it -- a rotated phone, or a resized window.
window.matchMedia("(max-width: 820px)").addEventListener("change", () => {
  paintSpine();
  renderMarks();
});
el("filepick").addEventListener("change", (e) => {
  if (e.target.files[0]) addByFile(e.target.files[0]);
  e.target.value = "";
});
["dragenter", "dragover"].forEach((t) =>
  el("drop").addEventListener(t, (e) => { e.preventDefault(); el("drop").classList.add("over"); }));
["dragleave", "drop"].forEach((t) =>
  el("drop").addEventListener(t, () => el("drop").classList.remove("over")));
el("drop").addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files[0]) addByFile(e.dataTransfer.files[0]);
});

el("export").addEventListener("click", exportNotes);
el("importfile").addEventListener("change", (e) => {
  if (e.target.files[0]) importNotes(e.target.files[0]);
  e.target.value = "";
});
el("clearnotes").addEventListener("click", () => {
  if (!confirm("Delete all " + notes.length + " notes for this book? The export file is the only copy.")) return;
  notes = [];
  storageSet(notesKey(), notes);
  renderNotes();
  // The conversation on screen is the same material as the notes, so clearing
  // one while the other stays put looks like the button did nothing.
  clearConversation();
});

// Wipe the transcript back to its opening state, lede and all.
function clearConversation() {
  transcriptEl.textContent = "";
  ledeCleared = false;
  LEDE.forEach((html) => {
    const p = document.createElement("p");
    p.className = "lede";
    p.innerHTML = html;
    transcriptEl.appendChild(p);
  });
}

migrateOldKeys();
notes = storageGet(notesKey(), []);
renderNotes();
paintSpine();
loadShelf();

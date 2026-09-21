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

let ws, session, micCtx, micNode, micStream, outCtx, playCursor = 0;
let ducked = false, agentSpeaking = false, framesSent = 0, justSeeked = false;
let scheduled = [];   // agent audio already queued on the device
let heartbeat = null; // the once-a-second playhead report
let live = false;     // a session is up and the microphone is open

// ---------- spoken commands ----------
//
// A few things are better handled here than by the agent. "Carry on" should
// stop the talking *now*; waiting for a model to decide it was an instruction
// means another second of speech over someone who has asked for silence. These
// match only a whole short utterance, so a real question containing the word
// "continue" is never swallowed.

// People do not say "okay." They say "okay, carry on" -- an acknowledgement
// and an instruction, in one breath. The old pattern matched exactly one of
// these words and then demanded the end of the utterance, so the single most
// natural way to ask for the book back was the one phrasing it ignored.
// Now any run of them counts, which also covers "alright thanks", "got it,
// keep going" and "yeah okay carry on".
const RESUME_WORD =
  "(?:ok(?:ay)?|alright|all right|right|yeah|yep|yes|thanks?|thank you|" +
  "got it|gotcha|understood|i see|makes sense|cool|nice|carry on|keep going|" +
  "go on|continue|resume|play|unpause|back to (?:the )?book|never ?mind|" +
  "that'?s (?:it|all))";

const COMMANDS = [
  {
    match: new RegExp(
      `^\\W*${RESUME_WORD}(?:[\\s,.!]+${RESUME_WORD})*[\\s.!,]*$`, "i"),
    run: () => {
      flushReply();
      resumeBook();
      setStatus("listening - just talk", "live");
      debug("command: carry on");
    },
  },
  {
    match: /^\W*(?:go back|back up|rewind|skip back)(?: a bit| a little)?[\s.!,]*$/i,
    run: () => {
      flushReply();
      seek(pos() - 30);
      resumeBook();
      turn("playhead", "Back 30 seconds", "system");
      debug("command: go back");
    },
  },
];

// The agent hears "okay, carry on" too, and answers it -- "Sure, let me know
// if you have more questions" -- which fires reply.started, which re-pauses
// the book we have just resumed. So the book came back for a second and then
// stopped again, which reads as the command not working.
//
// A command is handled here and the reply it provokes is dropped: not played,
// not shown, and above all not allowed to pause anything. Cleared when that
// reply finishes, with a timer in case the agent chooses not to answer at all.
let suppressReply = false;
let suppressTimer = null;

function suppressNextReply() {
  suppressReply = true;
  if (suppressTimer) clearTimeout(suppressTimer);
  suppressTimer = setTimeout(() => { suppressReply = false; }, 6000);
}

// Returns true when the utterance was a command and has been handled here.
function handleCommand(text) {
  const hit = COMMANDS.find((c) => c.match.test(text || ""));
  if (!hit) return false;
  suppressNextReply();
  hit.run();
  return true;
}

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

// Books here run to fifty-four hours, so minutes:seconds is not enough -- it
// printed 3256:21 for a position three hours in, which is both unreadable and
// wrong-looking. Hours appear only when there are some, so a short chapter
// still reads 09:12 rather than 0:09:12.
function clockText(t) {
  if (!isFinite(t) || t < 0) t = 0;
  const h = Math.floor(t / 3600);
  const m = String(Math.floor((t % 3600) / 60)).padStart(2, "0");
  const s = String(Math.floor(t % 60)).padStart(2, "0");
  return h ? `${h}:${m}:${s}` : `${m}:${s}`;
}

// ---------- the conversation ----------

// The opening steps, kept here so clearing the conversation can put them back
// rather than leaving an empty panel.
const STEPS = [
  'Pick a book from the <b>library</b>',
  '<b>Enable asking questions</b>',
  'Talk over it whenever something doesn&rsquo;t land',
];

let ledeCleared = false;
function turn(who, text, cls) {
  if (!ledeCleared) {
    transcriptEl.querySelectorAll(".steps").forEach((n) => n.remove());
    ledeCleared = true;
  }
  const wrap = document.createElement("div");
  wrap.className = "turn " + (cls || who);
  const at = document.createElement("span");
  at.className = "at";
  at.textContent = clockText(pos());
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

// ---------- one thought, one turn ----------
//
// A pause mid-sentence ends a turn, so "I think... physical intuition is a...
// limitation when it comes to..." arrives as five separate transcripts. The
// agent handles that fine -- it answers the whole thought -- but the page was
// printing five stub lines and filing five notes for one question.
//
// The agent's own turn_detection is set to wait longer (max_silence 2000 ms),
// which is the real fix. This is the second half: while nothing has been
// answered yet, consecutive fragments are the same question, so they are shown
// and remembered as one.

let openTurn = null;   // { p, node } of a user turn still being added to

function addUserSpeech(text) {
  text = (text || "").trim();
  if (!text) return;
  if (openTurn) {
    // Join with a space unless the fragment starts with punctuation.
    const joiner = /^[,.;:!?]/.test(text) ? "" : " ";
    openTurn.p.textContent = (openTurn.p.textContent + joiner + text).trim();
    openTurn.node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  } else {
    const p = turn("you", text);
    openTurn = { p, node: p.closest(".turn") };
    // A new thought starts a new note. noteQuestion() deliberately keeps the
    // first fragment's timestamp while a thought is still being assembled --
    // but a question that never got answered (an agent error, a dropped
    // socket) left its timestamp behind, and the next question inherited it.
    // The mark then pinned to where the previous question was asked.
    pendingQ = null;
  }
  // The note tracks the whole thought, not the first stub of it.
  noteQuestion(openTurn.p.textContent);
}

// ---------- one timeline across many files ----------
//
// A real audiobook is published one file per chapter, so a book here is a list
// of parts laid end to end. Everything above this line works in *book* time --
// 4:12:30 means four hours twelve into the book, not into whichever file is
// loaded. Only these three functions know that more than one file exists.

let parts = [];        // [{url, offset_s, duration_s}], empty for a single file
let partIndex = 0;

// The part list is only a timeline when every part has been measured. An
// unfinished book reports all of its parts with zeroed offsets, which is not a
// shorter timeline -- it is a wrong one.
function usableParts(b) {
  if (!b || b.status !== "ready") return [];
  const ps = b.parts || [];
  if (ps.length < 2) return [];
  return ps.every((p) => p.duration_s > 0) ? ps : [];
}

function partOffset() {
  return parts.length ? (parts[partIndex] ? parts[partIndex].offset_s : 0) : 0;
}

// Where we are in the whole book.
function pos() {
  const t = book.currentTime;
  return partOffset() + (isFinite(t) ? t : 0);
}

function loadPart(i, localTime, play) {
  partIndex = Math.max(0, Math.min(parts.length - 1, i));
  // This call carries its own destination; a seek queued against the file we
  // are leaving must not be replayed into the one we are loading.
  pendingSeek = null;
  book.src = parts[partIndex].url;
  book.load();
  const go = () => {
    book.removeEventListener("loadedmetadata", go);
    book.currentTime = Math.max(0, localTime || 0);
    if (play) book.play().catch(() => {});
    paintSpine();
  };
  book.addEventListener("loadedmetadata", go);
}

// Setting currentTime before the element has metadata is silently ignored --
// no error, no seek, and the caller has no way to tell. That is how clicking
// "Pick up there" the instant a book opened did nothing at all. Hold the
// request and apply it when the file is ready.
let pendingSeek = null;

function setLocalTime(local) {
  if (book.readyState < 1) { pendingSeek = local; return; }
  book.currentTime = local;
}

function drainPendingSeek() {
  if (pendingSeek === null) return;
  const t = pendingSeek;
  pendingSeek = null;
  book.currentTime = t;
}

// Move to a point in the book, changing file if that is where it lands.
function seek(t) {
  const total = duration();
  t = Math.max(0, total ? Math.min(total - 0.25, t) : t);
  if (parts.length < 2) {
    setLocalTime(t);
    return;
  }
  let i = 0;
  while (i < parts.length - 1 && t >= parts[i].offset_s + parts[i].duration_s) i++;
  const local = Math.max(0, t - parts[i].offset_s);
  if (i === partIndex) setLocalTime(local);
  else loadPart(i, local, !book.paused);
}

// ---------- the book ----------

function duck() {
  if (ducked) return;
  ducked = true;
  book.volume = 0.12;
  document.body.classList.add("listening");
}

function pauseBook() {
  // Named for what it usually does. In "keep it playing" mode the book never
  // actually stops -- it drops to 12% and carries on underneath, which is the
  // whole point of that choice.
  if (answerMode === "pause") book.pause();
  else duck();
  document.body.classList.add("listening");
}

function resumeBook() {
  unduck();
  if (book.paused) book.play().catch(() => {});
}

// Undo the ducking without deciding whether the book should be playing. Ending
// a session should give the volume back, not start playback someone paused on
// purpose a minute ago.
function unduck() {
  ducked = false;
  book.volume = 1;
  document.body.classList.remove("listening");
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
        seconds: pos(),
        book: currentBook ? currentBook.id : null,
        brevity,
      }),
    });
    // The reply can carry a jump. go_to_topic runs on AssemblyAI's servers and
    // cannot touch this page, so it leaves a position behind and we collect it.
    const data = await res.json();
    if (typeof data.seek === "number") applySeek(data.seek);
  } catch (_) { /* a dropped report is harmless; the next one is a second away */ }
}

function applySeek(seconds) {
  seek(seconds);
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
  const frac = dur ? Math.min(1, pos() / dur) : 0;
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
  el("clock").textContent = clockText(pos());
  el("total").textContent = "/ " + clockText(dur);
  const spine = el("spine");
  spine.setAttribute("aria-valuemax", Math.round(dur));
  spine.setAttribute("aria-valuenow", Math.round(pos()));
  spine.setAttribute("aria-valuetext", clockText(pos()));
}

function duration() {
  // With parts, the element only knows the file it holds; the book's length
  // comes from the server. Without parts the element is the better source,
  // since it is exact and available before anything else loads.
  if (parts.length > 1 && currentBook && currentBook.duration) return currentBook.duration;
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
    // Position in pixels, not percent, so overlapping marks can be pushed
    // apart below. Two questions asked a minute apart in a long book land on
    // the same few pixels and print on top of each other.
    if (!flow) b.dataset.ideal = String((n.t / dur));
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
      seek(n.t);
      // Deliberate, like a chapter jump: if this happens while an answer is
      // running, do not undo it with the usual three-second rewind.
      justSeeked = true;
      book.play().catch(() => {});
    });
    holder.appendChild(b);
  });
  if (!flow) spreadMarks(holder);
}

// Push overlapping marks apart.
//
// Two questions a minute apart in a long book land on the same few pixels, and
// the labels print straight on top of each other. Each mark keeps its true
// position unless something is already there, then it steps down just enough
// to be readable -- so the ones that are actually far apart stay honest.
const MARK_GAP = 19;   // px: label line-height plus a little air

function spreadMarks(holder) {
  const height = holder.getBoundingClientRect().height;
  if (!height) return;
  const marks = [...holder.children];
  let floor = 0;
  marks.forEach((b) => {
    const ideal = Number(b.dataset.ideal || 0) * height;
    const y = Math.max(ideal, floor);
    b.style.top = Math.min(y, height - 2) + "px";
    // A displaced mark says so, so the dot beside the axis is not read as the
    // exact moment when it has been nudged.
    b.classList.toggle("shifted", y - ideal > 2);
    floor = y + MARK_GAP;
  });
}

function seekFromEvent(e) {
  const rect = el("spine").getBoundingClientRect();
  const dur = duration();
  if (!dur) return;
  const frac = narrow()
    ? (e.clientX - rect.left) / rect.width
    : (e.clientY - rect.top) / rect.height;
  seek(frac * dur);
  justSeeked = true;      // dragging the spine is as deliberate as it gets
  paintSpine();
}

// ---------- speed ----------
//
// Anyone who listens to books seriously listens fast, and the rate has to
// survive changing book and reloading or it is a toy.

let rate = 1;

function applyRate(r) {
  rate = Math.min(3, Math.max(1, Number(r) || 1));
  // BOTH, and this is the whole bug: load() runs the media load algorithm,
  // which resets playbackRate to defaultPlaybackRate. Setting only
  // playbackRate meant every book change and every part handover silently
  // dropped back to 1x while the slider still read 3x -- the control and the
  // audio disagreeing, which is worse than either being wrong.
  book.defaultPlaybackRate = rate;
  book.playbackRate = rate;
  // Keep voices sounding like voices rather than chipmunks at 2x. Three
  // spellings because the unprefixed one is recent.
  book.preservesPitch = true;
  book.mozPreservesPitch = true;
  book.webkitPreservesPitch = true;
  storageSet("playhead:rate", rate);
  el("speed").value = String(rate);
  el("speedval").textContent = rate.toFixed(2) + "×";
  el("speedreset").hidden = rate === 1;
}

// Re-assert after any load, because defaultPlaybackRate is only a default:
// a browser that ignores it still gets the rate put back here.
function reassertRate() {
  if (book.playbackRate !== rate) book.playbackRate = rate;
  book.preservesPitch = true;
  book.mozPreservesPitch = true;
  book.webkitPreservesPitch = true;
}

function buildSpeeds() {
  const slider = el("speed");
  slider.addEventListener("input", () => applyRate(slider.value));
  el("speedreset").addEventListener("click", () => applyRate(1));
  applyRate(Number(storageGet("playhead:rate", 1)) || 1);
}

// ---------- what the book does while the agent answers ----------
//
// People genuinely differ here. Pausing means you hear the answer cleanly but
// lose the thread of the book; ducking keeps the book moving underneath but
// you half-follow both. Neither is correct, so it is a choice rather than a
// default someone has to work around.

let answerMode = "pause";
let brevity = "full";

function applyAnswerMode(mode) {
  answerMode = mode === "duck" ? "duck" : "pause";
  storageSet("playhead:answermode", answerMode);
  el("answermode").querySelectorAll("button").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.mode === answerMode));
  });
}

function applyBrevity(mode) {
  brevity = mode === "short" ? "short" : "full";
  storageSet("playhead:brevity", brevity);
  el("brevity").querySelectorAll("button").forEach((b) => {
    b.setAttribute("aria-checked", String(b.dataset.brief === brevity));
  });
  // Takes effect on the next heartbeat, so a change mid-session lands within
  // a second rather than waiting for a reconnect.
  reportPlayhead();
}

function buildBrevity() {
  el("brevity").querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => applyBrevity(b.dataset.brief));
  });
  applyBrevity(storageGet("playhead:brevity", "full"));
}

function buildAnswerMode() {
  el("answermode").querySelectorAll("button").forEach((b) => {
    b.addEventListener("click", () => applyAnswerMode(b.dataset.mode));
  });
  applyAnswerMode(storageGet("playhead:answermode", "pause"));
}

// ---------- table of contents ----------
//
// An audiobook file has no structure in it -- it is one opaque stream, which
// is why skipping around one is guesswork. The backend reads the chapters back
// out of the transcript, because the narrator announces them out loud.

let chapters = [];

async function loadContents(bookId) {
  chapters = [];
  renderContents();
  el("tocnote").textContent = "Reading the book’s structure…";
  el("tocnote").hidden = false;
  try {
    const data = await (await fetch("/api/books/" + encodeURIComponent(bookId) + "/contents",
                                    { headers: withClient() })).json();
    chapters = data.contents || [];
  } catch (_) { chapters = []; }
  renderContents();
}

function renderContents() {
  const list = el("toclist");
  const note = el("tocnote");
  list.textContent = "";
  if (!chapters.length) {
    note.textContent = "This recording doesn’t announce chapters, so there are none to list.";
    note.hidden = false;
    return;
  }
  note.hidden = true;
  chapters.forEach((c) => {
    const li = document.createElement("li");
    const b = document.createElement("button");
    b.type = "button";
    b.dataset.t = String(c.t);
    const at = document.createElement("span");
    at.className = "at";
    at.textContent = clockText(c.t);
    const t = document.createElement("span");
    t.className = "t";
    t.textContent = c.title;
    if (c.hint) t.title = c.hint;
    b.append(at, t);
    b.addEventListener("click", () => {
      seek(c.t);
      justSeeked = true;
      book.play().catch(() => {});
    });
    li.appendChild(b);
    list.appendChild(li);
  });
  markCurrentChapter();
}

function markCurrentChapter() {
  if (!chapters.length) return;
  const now = pos();
  let active = -1;
  chapters.forEach((c, i) => { if (c.t <= now + 0.5) active = i; });
  el("toclist").querySelectorAll("button").forEach((b, i) => {
    if (i === active) b.setAttribute("aria-current", "true");
    else b.removeAttribute("aria-current");
  });
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

// "relativity" was the shipped book two renames ago. Keying anything to it now
// files notes under a book that does not exist, so the fallback is explicit:
// nothing is selected yet, and nothing should be written.
const NO_BOOK = "unselected";
const notesKey = () => "playhead:notes:" + (currentBook ? currentBook.id : NO_BOOK);
const posKey = () => "playhead:pos:" + (currentBook ? currentBook.id : NO_BOOK);

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
  // moved, and the note is only useful if it points where the question was
  // asked. A stitched question keeps the position of its first fragment.
  pendingQ = { t: pendingQ ? pendingQ.t : pos(), q: text };
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
  // Always shown. Import in particular has to be reachable when there are no
  // notes yet -- that is exactly the moment someone arriving on a second
  // machine needs it, and hiding it made the whole section look missing.
  el("noteactions").hidden = false;
  el("export").disabled = notes.length === 0;
  el("clearnotes").disabled = notes.length === 0;
  el("notecount").textContent = notes.length ? String(notes.length) : "";
  el("railnote").textContent = notes.length
    ? `${notes.length} question${notes.length > 1 ? "s" : ""} on this book. Click a mark to play from there.`
    : "Questions you ask get pinned to the spine where you asked them. Import a file to bring notes from another machine.";
  renderMarks();
}

function exportNotes() {
  const lines = ["# Playhead notes", "",
                 "Book: " + (currentBook ? currentBook.title : "none"),
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
//
// Suppressed while a book is being swapped in. posKey() follows currentBook,
// which changes the instant the new book is chosen -- but the <audio> element
// still holds the old file for a few more ticks, and the pause and timeupdate
// events it fires during the swap were writing the OLD book's position under
// the NEW book's key. The symptom is opening a book you have never played and
// being offered to resume at somebody else's timestamp.
let switching = false;

function rememberPosition() {
  if (switching || !currentBook) return;
  if (pos() > 5) storageSet(posKey(), Math.round(pos()));
}

function offerResume() {
  const at = storageGet(posKey(), 0);
  const bar = el("resume");
  if (!at || at < 10) { bar.hidden = true; return; }
  el("resumeat").textContent = clockText(at);
  bar.hidden = false;
  el("resumebtn").onclick = () => {
    seek(at);
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
    // Never open on a book that cannot be played. Landing on one that is
    // still indexing gives a dead transport and an empty spine, which reads
    // as the whole page being broken rather than as one book not being ready.
    const saved = storageGet("playhead:book", null);
    const ready = shelf.filter((b) => b.status === "ready");
    const pick = ready.find((b) => b.id === saved) || ready[0];
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

    // Anything added can be taken off again. The index is left to expire on
    // its own -- someone else may be holding the same book.
    if (!b.builtin && !b.featured) {
      const x = document.createElement("button");
      x.className = "drop";
      x.type = "button";
      x.textContent = "×";
      x.title = "Remove “" + b.title + "” from your shelf";
      x.setAttribute("aria-label", x.title);
      x.addEventListener("click", async (e) => {
        e.stopPropagation();
        shelf = shelf.filter((x2) => x2.id !== b.id);
        if (currentBook && currentBook.id === b.id) {
          const first = shelf[0];
          if (first) selectBook(first, true);
        }
        renderShelf();
        renderSuggested();
        try {
          await fetch("/api/books/" + encodeURIComponent(b.id),
                      { method: "DELETE", headers: withClient() });
        } catch (_) { /* it is already gone from the page */ }
      });
      li.appendChild(x);
    }

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
  // Bank the old book's position while posKey() still points at it.
  if (currentBook && currentBook.id !== b.id) rememberPosition();
  switching = true;
  // A backstop: a source that never reports metadata (a dead link, a revoked
  // object URL) would otherwise leave saving switched off for the session.
  setTimeout(() => { switching = false; }, 4000);
  lastRemembered = 0;
  currentBook = b;
  storageSet("playhead:book", b.id);
  el("booktitle").textContent = b.title;

  // A book of many files becomes one timeline; a single file is just itself.
  //
  // Only trust the part list once the book is ready. While it is still being
  // absorbed the record carries every part but only the finished ones have a
  // real offset -- the rest are 0.0 -- and seek() walks those offsets looking
  // for the file a timestamp lands in. With a run of zeroes every seek past
  // the first second resolves to the last part, so a half-indexed book plays
  // chapter 117 wherever you touch the spine. Treated as a single file until
  // the offsets mean something.
  parts = usableParts(b);
  partIndex = 0;
  pendingSeek = null;     // belongs to the book we are leaving
  const src = b.audio_url || localAudio[b.id] || "";
  needsFile = !src && !parts.length;
  if (parts.length) loadPart(0, 0, false);
  else if (src) {
    book.src = src;
    book.load();
  }
  book.pause();
  setPlayIcon(false);

  // Notes and the resume point belong to the book, not the browser.
  notes = storageGet(notesKey(), []);
  pendingQ = null;
  renderNotes();
  loadContents(b.id);
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

// Each poll advances the job one step server-side, so two pollers on the same
// book do not go twice as fast -- they just double the requests and fight over
// the same record. loadShelf nudges everything unfinished on every load, which
// is exactly how a second one gets started.
const polling = new Set();

async function pollBook(id) {
  if (polling.has(id)) return;
  polling.add(id);
  try {
    return await pollLoop(id);
  } finally {
    polling.delete(id);
  }
}

async function pollLoop(id) {
  // Ten minutes was the old ceiling and it was written when a book was one
  // chapter. A fifteen-part book takes about that long and a hundred-part one
  // takes hours, so the budget follows the book; the poll also slows down
  // once it is clear this is a long one, rather than asking every 2.5 s for
  // an afternoon.
  const started = Date.now();
  let budgetMs = 15 * 60 * 1000;
  for (;;) {
    const mins = (Date.now() - started) / 60000;
    const gap = mins < 2 ? 2500 : mins < 10 ? 6000 : 15000;
    await new Promise((r) => setTimeout(r, gap));
    if (Date.now() - started > budgetMs) {
      const b = shelf.find((x) => x.id === id);
      addStatus("Still working on “" + (b ? b.title : id) + "”. It is safe to leave "
                + "this page — reopen it and indexing carries on from where it "
                + "got to.");
      return;
    }
    let rec;
    try {
      rec = await (await fetch("/api/books/" + encodeURIComponent(id),
                               { headers: withClient() })).json();
    } catch (_) { continue; }
    // Now that the record is in hand, size the budget to the actual book:
    // two minutes a part, which comfortably covers a 117-file novel.
    if (rec && rec.part_count) {
      budgetMs = Math.max(budgetMs, rec.part_count * 120 * 1000);
    }
    const at = shelf.findIndex((b) => b.id === rec.id);
    if (at >= 0) shelf[at] = rec; else shelf.push(rec);
    // Re-select rather than just swapping the record in. The player's parts,
    // duration and contents were all derived from the old one, and a book
    // that finished indexing while it was open otherwise stayed unplayable
    // until a reload -- with a full progress bar, which looks like a lie.
    if (currentBook && currentBook.id === rec.id) {
      const wasReady = currentBook.status === "ready";
      currentBook = rec;
      if (!wasReady && rec.status === "ready") selectBook(rec, true);
    }
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
    // A multi-file book says which file it is on: "transcribing" sitting at
    // 5% for half an hour on a 117-part novel looks stuck when it is working.
    const many = rec.part_count > 1
      ? ` — file ${Math.min(rec.parts_done + 1, rec.part_count)} of ${rec.part_count}`
      : "";
    addStatus(rec.status === "transcribing"
      ? `Transcribing${many} — a minute or two per file.`
      : `Indexing passages … ${rec.progress}%${many}`);
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
  // Held so the tracks can be stopped later. Without the handle the only way
  // to release the microphone is to close the tab.
  micStream = stream;
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
  // Keep the handle. Resetting the cursor alone does not silence anything --
  // these buffers are already scheduled on the device and will play out
  // regardless, which is why "interrupted" used to keep talking.
  scheduled.push(node);
  node.onended = () => { scheduled = scheduled.filter((n) => n !== node); };
}

function flushReply() {
  scheduled.forEach((n) => { try { n.stop(); } catch (_) { /* already done */ } });
  scheduled = [];
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

// Give the microphone back and stop the session.
//
// There was no way to do this short of closing the tab, which is the wrong
// answer to "stop listening to me" -- and it also made a bad session
// unrecoverable without losing the page. Stopping the tracks is what turns the
// browser's recording indicator off; closing the socket alone does not.
function stopSession() {
  live = false;
  if (heartbeat) { clearInterval(heartbeat); heartbeat = null; }
  try { if (micNode) micNode.disconnect(); } catch (_) { /* already gone */ }
  micNode = null;
  if (micStream) {
    micStream.getTracks().forEach((t) => { try { t.stop(); } catch (_) { /* ok */ } });
    micStream = null;
  }
  if (ws) {
    // Drop the handlers first: onclose would otherwise re-enter the UI reset
    // while this one is half-done.
    ws.onclose = null;
    ws.onerror = null;
    try { ws.close(); } catch (_) { /* already closing */ }
    ws = null;
  }
  flushReply();
  session = null;
  agentSpeaking = false;
  openTurn = null;
  unduck();
  setListenIcon(false);
  startBtn.disabled = false;
  setStatus("not listening — the book plays on", "idle");
  debug("session stopped, microphone released");
}

function setListenIcon(on) {
  live = on;
  startBtn.textContent = on ? "Stop listening" : "Enable asking questions";
  startBtn.classList.toggle("live", on);
}

async function start() {
  if (live) { stopSession(); return; }
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
    // The socket going away has to tear down the rest too, or the microphone
    // stays open and the heartbeat keeps reporting for a session that no
    // longer exists.
    if (heartbeat) { clearInterval(heartbeat); heartbeat = null; }
    if (micStream) {
      micStream.getTracks().forEach((t) => { try { t.stop(); } catch (_) { /* ok */ } });
      micStream = null;
    }
    try { if (micNode) micNode.disconnect(); } catch (_) { /* already gone */ }
    micNode = null;
    session = null;
    setListenIcon(false);
    unduck();
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
        // Replace, never stack. Pressing start again after a dropped socket
        // used to leave the old timer running, so every reconnect added
        // another report per second -- four sessions in and the backend was
        // being told the playhead four times a second by four dead sessions.
        if (heartbeat) clearInterval(heartbeat);
        heartbeat = setInterval(reportPlayhead, 1000);
        live = true;
        setListenIcon(true);
        // Re-enable it: the button was disabled to stop a second connection
        // being opened while this one was still handshaking, and it now means
        // "stop listening". Left disabled it read as a dead control, and the
        // only way to release the microphone was to close the tab.
        startBtn.disabled = false;
        book.play().catch(() => {});
        setPlayIcon(true);
        setStatus("listening - just talk", "live");
        break;

      case "input.speech.started":
        // A new question starts here, so anything that moved the playhead
        // before it is history. Without this a chapter click half an hour ago
        // still suppressed the three-second rewind on the next answer, and
        // the flag could sit true for the rest of the session.
        justSeeked = false;
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
        // A command is an instruction, not a question: it gets acted on here
        // and is deliberately not kept as a note. "Carry on" is not a thing
        // anyone wants on their map of where the book lost them.
        if (handleCommand(m.text)) {
          turn("you", m.text || "");
          openTurn = null;
          break;
        }
        addUserSpeech(m.text || "");
        pauseBook();
        setStatus("thinking", "busy");
        break;

      case "input.speech.stopped":
        setStatus("thinking", "busy");
        break;

      case "reply.started":
        // The thought has been answered; the next thing said is a new one.
        openTurn = null;
        // A reply to "carry on" is not an answer to anything. Let it run its
        // course on the server, but do not stop the book for it.
        if (suppressReply) { flushReply(); break; }
        agentSpeaking = true;
        // Hold the book for THIS answer. Resuming happens on reply.done, so
        // when the agent answers twice in a row the book was playing
        // underneath the second one -- which read as the pause setting being
        // ignored. Every answer re-asserts it.
        pauseBook();
        flushReply();
        setStatus("answering", "busy");
        break;

      case "reply.audio":
        if (suppressReply) break;
        playReply(m.data || m.audio);
        break;

      case "transcript.agent":
        if (suppressReply) break;
        turn("playhead", m.text || "");
        noteAnswer(m.text || "");
        break;

      case "reply.done":
        if (suppressReply) {
          suppressReply = false;
          if (suppressTimer) { clearTimeout(suppressTimer); suppressTimer = null; }
          flushReply();
          break;      // the book was never stopped; leave it alone
        }
        agentSpeaking = false;
        if (m.status === "interrupted") flushReply();
        // Back up slightly so the run-up to the question is re-heard -- unless
        // the agent just moved us somewhere on purpose, or the book never
        // stopped, in which case rewinding would undo the continuity that was
        // the reason for choosing that mode.
        if (justSeeked) justSeeked = false;
        else if (answerMode === "pause") seek(pos() - 3);
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
// The end of a file is only the end of the book if it was the last one.
book.addEventListener("ended", () => {
  if (parts.length && partIndex < parts.length - 1) {
    debug(`part ${partIndex + 2} of ${parts.length}`);
    loadPart(partIndex + 1, 0, true);
  }
});
book.addEventListener("play", () => setPlayIcon(true));
book.addEventListener("pause", () => setPlayIcon(false));
book.addEventListener("loadedmetadata", () => {
  switching = false;          // the element now holds the book we think it does
  reassertRate();
  drainPendingSeek();
  paintSpine();
  renderMarks();
});
// loadeddata fires for sources that never report metadata the same way; both
// are cheap and the rate has to survive every path into a new file.
book.addEventListener("loadeddata", reassertRate);

let lastRemembered = 0;
book.addEventListener("timeupdate", () => {
  paintSpine();
  markCurrentChapter();
  if (pos() - lastRemembered > 5 || pos() < lastRemembered) {
    lastRemembered = pos();
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
// pointercancel as well as pointerup: the browser takes the pointer away on a
// scroll gesture or an incoming call, and without this the spine stayed in
// drag mode, so the next stray move over it threw the listener across the book.
["pointerup", "pointercancel", "lostpointercapture"].forEach((t) =>
  el("spine").addEventListener(t, () => { dragging = false; }));
el("spine").addEventListener("keydown", (e) => {
  const step = e.key === "ArrowUp" || e.key === "ArrowLeft" ? -15
             : e.key === "ArrowDown" || e.key === "ArrowRight" ? 15 : 0;
  if (!step) return;
  e.preventDefault();
  seek(pos() + step);
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

// Wipe the transcript back to its opening state, steps and all.
function clearConversation() {
  transcriptEl.textContent = "";
  ledeCleared = false;
  const ol = document.createElement("ol");
  ol.className = "steps";
  STEPS.forEach((html) => {
    const li = document.createElement("li");
    li.innerHTML = html;
    ol.appendChild(li);
  });
  transcriptEl.appendChild(ol);
}

migrateOldKeys();
buildSpeeds();
buildAnswerMode();
buildBrevity();
notes = storageGet(notesKey(), []);
renderNotes();
paintSpine();
loadShelf();

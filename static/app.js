// Shhhhhhh AI — silent speech from your Mac camera
// Face tracking (MediaPipe) runs in the browser; lip reading, Voice Memory,
// AI correction and voices run in the local Python server (server/app.py).

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const isHindi = (t) => /[ऀ-ॿ]/.test(t || "");
const DEV_MOCK = new URLSearchParams(location.search).has("mock");

// ------------------------------------------------------------- API
async function api(path, opts = {}) {
  const res = await fetch(path, {
    method: opts.method || (opts.body ? "POST" : "GET"),
    headers: opts.body ? { "Content-Type": "application/json" } : {},
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  let data = null;
  try { data = await res.json(); } catch { /* empty */ }
  if (!res.ok) throw new Error((data && data.detail) || `HTTP ${res.status}`);
  return data;
}

function toast(html, kind = "") {
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.innerHTML = html;
  $("#toasts").appendChild(el);
  setTimeout(() => { el.style.transition = "opacity .4s"; el.style.opacity = "0"; }, 4200);
  setTimeout(() => el.remove(), 4700);
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

// ------------------------------------------------------------- face geometry
const OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146];
const INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95];
const JAW = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288];
const BLEND = ["jawOpen", "jawForward", "jawLeft", "jawRight", "mouthClose", "mouthFunnel", "mouthPucker",
  "mouthLeft", "mouthRight", "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
  "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight", "mouthRollLower",
  "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthPressLeft", "mouthPressRight",
  "mouthLowerDownLeft", "mouthLowerDownRight", "mouthUpperUpLeft", "mouthUpperUpRight", "cheekPuff"];
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152,
  148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];

const mid = (a, b) => [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
const meanPts = (ps) => [ps.reduce((s, p) => s + p[0], 0) / ps.length, ps.reduce((s, p) => s + p[1], 0) / ps.length];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function facePoints(lm, w, h) {
  const P = (i) => [lm[i].x * w, lm[i].y * h];
  return {
    P,
    rEye: mid(P(33), P(133)),          // subject's right eye (image left)
    lEye: mid(P(362), P(263)),
    nose: meanPts([P(2), P(98), P(327)]),
    mouth: meanPts([P(61), P(291), P(0), P(17), P(13), P(14)]),
  };
}

// Per-frame features for Voice Memory (layout must match server/memory.py GROUPS)
function mouthFeatures(fp, blendCats) {
  const { P, rEye, lEye } = fp;
  const o = mid(rEye, lEye);
  const dx = lEye[0] - rEye[0], dy = lEye[1] - rEye[1];
  const s = Math.hypot(dx, dy) || 1, c = dx / s, sn = dy / s;
  const N = (p) => { const x = p[0] - o[0], y = p[1] - o[1]; return [(x * c + y * sn) / s, (-x * sn + y * c) / s]; };
  const f = [];
  for (const i of OUTER) f.push(...N(P(i)));
  for (const i of INNER) f.push(...N(P(i)));
  for (const i of JAW) f.push(...N(P(i)));
  f.push(dist(P(13), P(14)) / s, dist(P(61), P(291)) / s, dist(P(1), P(152)) / s);
  const bmap = {};
  for (const cat of blendCats || []) bmap[cat.categoryName] = cat.score;
  for (const k of BLEND) f.push(bmap[k] ?? 0);
  return f.map((v) => Math.round(v * 10000) / 10000);
}

// ------------------------------------------------------------- state
const state = {
  settings: {},
  status: {},
  page: "live",
  landmarker: null,
  ready: false,
  face: null,           // latest {lm, fp, feat, blend}
  box: null,            // fixed crop box for the current clip {x,y,s,v}
  boxVersion: 0,
  activity: 0, floor: 0.004, prevFeat: null,
  rec: null,            // active recording
  ring: [],             // pre-roll buffer (hands-free)
  autoOn: true,
  busy: false,
  lastClip: null,
  lastResult: null,
  memory: { phrases: [] },
  accuracy: {},
  train: null,
};
const CROP = 320;
const cropCanvas = Object.assign(document.createElement("canvas"), { width: CROP, height: CROP });
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: false });

// ------------------------------------------------------------- MediaPipe
async function loadLandmarker() {
  const local = "/static/vendor/mediapipe";
  const cdn = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0";
  let mod, base;
  try { mod = await import(`${local}/vision_bundle.mjs`); base = local; }
  catch { mod = await import(`${cdn}/vision_bundle.mjs`); base = cdn; }
  const fileset = await mod.FilesetResolver.forVisionTasks(`${base}/wasm`);
  let modelPath = "/static/models/face_landmarker.task";
  try { const r = await fetch(modelPath, { method: "HEAD" }); if (!r.ok) throw 0; }
  catch { modelPath = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"; }
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: modelPath, delegate },
    runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true,
    minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  try { return await mod.FaceLandmarker.createFromOptions(fileset, opts("GPU")); }
  catch { return await mod.FaceLandmarker.createFromOptions(fileset, opts("CPU")); }
}

// Dev-only synthetic face so the interface can be previewed without a camera (?mock)
function mockFace(t) {
  const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
  const set = (i, x, y) => { lm[i] = { x, y, z: 0 }; };
  const open = 0.012 + 0.03 * Math.max(0, Math.sin(t / 180)) * (Math.sin(t / 900) > -0.2 ? 1 : 0.1);
  set(33, 0.40, 0.42); set(133, 0.46, 0.42); set(362, 0.54, 0.42); set(263, 0.60, 0.42);
  set(1, 0.5, 0.52); set(2, 0.5, 0.56); set(98, 0.47, 0.56); set(327, 0.53, 0.56);
  const ring = (ids, rx, ry, cy) => ids.forEach((id, k) => {
    const a = Math.PI - (k / ids.length) * 2 * Math.PI;
    set(id, 0.5 + rx * Math.cos(a), cy - ry * Math.sin(a));
  });
  ring(OUTER, 0.075, 0.022 + open, 0.655);
  ring(INNER, 0.06, 0.004 + open, 0.655);
  JAW.forEach((id, k) => { const a = Math.PI * (k / (JAW.length - 1)); set(id, 0.5 - 0.15 * Math.cos(a), 0.66 + 0.1 * Math.sin(a) + open); });
  FACE_OVAL.forEach((id, k) => { const a = (k / FACE_OVAL.length) * 2 * Math.PI - Math.PI / 2; if (!JAW.includes(id)) set(id, 0.5 + 0.17 * Math.cos(a), 0.5 + 0.24 * Math.sin(a)); });
  set(13, 0.5, 0.655 - open); set(14, 0.5, 0.655 + open); set(0, 0.5, 0.633 - open); set(17, 0.5, 0.677 + open);
  set(152, 0.5, 0.77 + open);
  return { lm, blend: [{ categoryName: "jawOpen", score: open * 8 }] };
}
function mockStream() {
  const c = Object.assign(document.createElement("canvas"), { width: 640, height: 480 });
  const g = c.getContext("2d");
  const draw = () => {
    const gr = g.createRadialGradient(320, 220, 30, 320, 240, 420);
    gr.addColorStop(0, "#5b4a42"); gr.addColorStop(0.5, "#2b2522"); gr.addColorStop(1, "#0e0d0c");
    g.fillStyle = gr; g.fillRect(0, 0, 640, 480);
    g.fillStyle = "#8a6a58"; g.beginPath(); g.ellipse(320, 245, 112, 150, 0, 0, Math.PI * 2); g.fill();
    requestAnimationFrame(draw);
  };
  draw();
  return c.captureStream(30);
}

// ------------------------------------------------------------- camera loop
const video = $("#video");
const overlay = $("#overlay");
const octx = overlay.getContext("2d");
const mouthCtx = $("#mouthCanvas").getContext("2d");

async function startCamera() {
  setCamStatus("Starting camera…", "");
  try {
    video.srcObject = DEV_MOCK ? mockStream() : await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: "user" }, audio: false,
    });
    await video.play();
  } catch (e) {
    showCamMsg("<b>Camera blocked</b>Allow camera access for this page (click the camera icon in the address bar), then reload.");
    return;
  }
  setCamStatus("Loading face tracker…", "busy");
  if (!DEV_MOCK) {
    try { state.landmarker = await loadLandmarker(); }
    catch (e) {
      showCamMsg(`<b>Face tracker didn't load</b>${esc(e.message || e)}<br>Run <code>./setup.sh</code> once with internet so it is stored offline.`);
      return;
    }
  }
  state.ready = true;
  setCamStatus("Ready", "ready");
  requestAnimationFrame(loop);
}

function showCamMsg(html) { const m = $("#camMsg"); m.innerHTML = `<div>${html}</div>`; m.classList.remove("hidden"); setCamStatus("Camera off", ""); }
function setCamStatus(txt, cls) { const p = $("#camStatus"); p.className = `pill glass ${cls}`; p.querySelector(".txt").textContent = txt; }

let lastVideoTime = -1, lastCapture = 0;
function loop(now) {
  requestAnimationFrame(loop);
  if (video.readyState < 2) return;
  const w = video.videoWidth, h = video.videoHeight;
  if (overlay.width !== w) { overlay.width = w; overlay.height = h; }
  if (video.currentTime === lastVideoTime && !DEV_MOCK) return;
  lastVideoTime = video.currentTime;

  let lm = null, blend = null;
  if (DEV_MOCK) { const m = mockFace(now); lm = m.lm; blend = m.blend; }
  else {
    const res = state.landmarker.detectForVideo(video, now);
    if (res.faceLandmarks && res.faceLandmarks.length) {
      lm = res.faceLandmarks[0];
      blend = res.faceBlendshapes && res.faceBlendshapes[0] ? res.faceBlendshapes[0].categories : [];
    }
  }

  if (lm) {
    const fp = facePoints(lm, w, h);
    const feat = mouthFeatures(fp, blend);
    state.face = { lm, fp, feat, w, h };
    updateActivity(feat);
    if (!state.rec) updateBox(lm, w, h);
  } else {
    state.face = null;
    state.activity *= 0.8;
  }
  drawOverlay();
  drawMouthInset();
  if (state.train) drawTrainCam();
  $("#meterBar").style.width = `${Math.min(100, (state.activity / Math.max(onThreshold(), 1e-4)) * 70)}%`;

  // capture every camera frame (~30 fps; the server resamples to 25) while recording, or into the pre-roll buffer when hands-free
  const needFrames = state.rec || (state.settings.capture_mode === "auto" && state.autoOn && !state.busy && state.page === "live" && !state.train);
  if (needFrames && now - lastCapture >= 20 && state.box) {
    lastCapture = now;
    const entry = captureEntry(now);
    if (state.rec) pushEntry(state.rec, entry);
    else { state.ring.push(entry); if (state.ring.length > 15) state.ring.shift(); }
  }
  tickAuto(now);
  if (state.rec && state.rec.onTick) state.rec.onTick(now);
}

function updateBox(lm, w, h) {
  let minx = 1, miny = 1, maxx = 0, maxy = 0;
  for (const i of FACE_OVAL) { const p = lm[i]; minx = Math.min(minx, p.x); maxx = Math.max(maxx, p.x); miny = Math.min(miny, p.y); maxy = Math.max(maxy, p.y); }
  const fw = (maxx - minx) * w, fh = (maxy - miny) * h;
  let s = Math.min(Math.max(fw, fh) * 2.0, Math.min(w, h));
  const cx = ((minx + maxx) / 2) * w, cy = ((miny + maxy) / 2) * h;
  const b = state.box;
  if (b) {
    const bcx = b.x + b.s / 2, bcy = b.y + b.s / 2;
    if (Math.hypot(cx - bcx, cy - bcy) < 0.1 * b.s && Math.abs(s - b.s) < 0.15 * b.s) return;
  }
  const x = Math.min(Math.max(cx - s / 2, 0), w - s), y = Math.min(Math.max(cy - s / 2, 0), h - s);
  state.box = { x, y, s, v: ++state.boxVersion };
}

function captureEntry(now) {
  const b = state.box;
  cropCtx.drawImage(video, b.x, b.y, b.s, b.s, 0, 0, CROP, CROP);
  const frame = cropCanvas.toDataURL("image/jpeg", 0.82);
  let kp = null, feat = null;
  if (state.face) {
    const k = CROP / b.s, { rEye, lEye, nose, mouth } = state.face.fp;
    kp = [rEye, lEye, nose, mouth].map((p) => [Math.round((p[0] - b.x) * k * 10) / 10, Math.round((p[1] - b.y) * k * 10) / 10]);
    feat = state.face.feat;
  }
  return { frame, kp, feat, t: now, v: b.v };
}
function newRec() { return { frames: [], keypoints: [], timestamps: [], features: [], start: performance.now(), spoke: false, quietSince: null }; }
function pushEntry(rec, e) {
  rec.frames.push(e.frame); rec.keypoints.push(e.kp); rec.timestamps.push(e.t);
  if (e.feat) rec.features.push(e.feat);
}

// ------------------------------------------------------------- speech activity
function updateActivity(feat) {
  const p = state.prevFeat;
  state.prevFeat = feat;
  if (!p) return;
  const iOpen = 110, iWidth = 111, iJaw = 113; // scalars + first blendshape (jawOpen)
  const a = Math.abs(feat[iOpen] - p[iOpen]) + 0.6 * Math.abs(feat[iWidth] - p[iWidth]) + Math.abs(feat[iJaw] - p[iJaw]);
  state.activity = 0.65 * state.activity + 0.35 * a;
  if (!state.rec) state.floor = 0.995 * state.floor + 0.005 * Math.min(state.activity, 0.02);
}
const onThreshold = () => Math.max(0.014, state.floor * 3.2);
const offThreshold = () => Math.max(0.0065, state.floor * 1.7);

let aboveCount = 0;
function tickAuto(now) {
  if (state.settings.capture_mode !== "auto" || !state.autoOn || state.busy || state.page !== "live" || state.train) return;
  if (!state.rec) {
    aboveCount = state.activity > onThreshold() ? aboveCount + 1 : 0;
    if (aboveCount >= 3 && state.face) {
      const rec = newRec();
      const v = state.box && state.box.v;
      for (const e of state.ring) if (e.v === v) pushEntry(rec, e);
      state.ring = [];
      beginRecording(rec, "auto");
    }
    return;
  }
  if (state.rec.kind !== "auto") return;
  const r = state.rec, dur = now - r.start;
  if (state.activity < offThreshold()) { r.quietSince = r.quietSince ?? now; } else r.quietSince = null;
  if ((r.quietSince && now - r.quietSince > 750) || dur > 9000) finishRecording();
}

// ------------------------------------------------------------- drawing
function drawPath(ids, P, close = true) {
  octx.beginPath();
  ids.forEach((id, k) => { const p = P(id); k ? octx.lineTo(p[0], p[1]) : octx.moveTo(p[0], p[1]); });
  if (close) octx.closePath();
}
function drawOverlay() {
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!state.face) return;
  const { fp } = state.face;
  const recCol = state.rec ? "255,69,58" : "255,255,255";
  const scale = overlay.width / 640;
  octx.lineJoin = "round";
  // face oval (very faint)
  drawPath(FACE_OVAL, fp.P);
  octx.strokeStyle = `rgba(255,255,255,0.10)`; octx.lineWidth = 1.2 * scale; octx.stroke();
  // jaw dots
  octx.fillStyle = `rgba(${recCol},0.55)`;
  for (const id of JAW) { const p = fp.P(id); octx.beginPath(); octx.arc(p[0], p[1], 2.2 * scale, 0, Math.PI * 2); octx.fill(); }
  // lips
  octx.shadowColor = `rgba(${recCol},0.9)`; octx.shadowBlur = 12 * scale;
  drawPath(OUTER, fp.P); octx.strokeStyle = `rgba(${recCol},0.95)`; octx.lineWidth = 2.2 * scale; octx.stroke();
  drawPath(INNER, fp.P); octx.strokeStyle = `rgba(${recCol},0.6)`; octx.lineWidth = 1.4 * scale; octx.stroke();
  octx.shadowBlur = 0;
  // alignment anchors
  octx.fillStyle = "rgba(100,210,255,0.9)";
  for (const p of [fp.rEye, fp.lEye, fp.nose]) { octx.beginPath(); octx.arc(p[0], p[1], 2.6 * scale, 0, Math.PI * 2); octx.fill(); }
}
function drawMouthInset() {
  const W = 132, H = 88;
  mouthCtx.fillStyle = "#111"; mouthCtx.fillRect(0, 0, W, H);
  if (!state.face) return;
  const { fp } = state.face;
  const mw = dist(fp.P(61), fp.P(291)) * 2.1, mh = mw * (H / W);
  mouthCtx.save();
  mouthCtx.filter = "grayscale(1) contrast(1.25)";
  mouthCtx.translate(W, 0); mouthCtx.scale(-1, 1);
  mouthCtx.drawImage(video, fp.mouth[0] - mw / 2, fp.mouth[1] - mh / 2, mw, mh, 0, 0, W, H);
  mouthCtx.restore();
}
function drawTrainCam() {
  const c = $("#trainCam"), g = c.getContext("2d");
  if (c.width !== 400) { c.width = 400; c.height = 400; }
  const b = state.box;
  if (!b) return;
  g.drawImage(video, b.x, b.y, b.s, b.s, 0, 0, 400, 400);
  if (state.face) {
    const k = 400 / b.s, P = (i) => { const p = state.face.fp.P(i); return [(p[0] - b.x) * k, (p[1] - b.y) * k]; };
    const col = state.rec ? "255,69,58" : "255,255,255";
    g.lineWidth = 2.2; g.strokeStyle = `rgba(${col},0.95)`; g.shadowColor = `rgba(${col},.9)`; g.shadowBlur = 10;
    for (const ids of [OUTER, INNER]) { g.beginPath(); ids.forEach((id, j) => { const p = P(id); j ? g.lineTo(...p) : g.moveTo(...p); }); g.closePath(); g.stroke(); }
    g.shadowBlur = 0;
  }
}

// ------------------------------------------------------------- recording (live)
function beginRecording(rec = newRec(), kind = "hold") {
  if (!state.ready || state.busy || state.rec) return;
  rec.kind = kind;
  state.rec = rec;
  $("#recBtn").classList.add("on");
  setCamStatus(kind === "auto" ? "Listening to your lips…" : "Recording — mouth your sentence", "rec");
}
async function finishRecording() {
  const rec = state.rec;
  if (!rec) return;
  state.rec = null;
  $("#recBtn").classList.remove("on");
  const dur = (rec.timestamps.at(-1) || 0) - (rec.timestamps[0] || 0);
  if (rec.frames.length < 12 || dur < 500) {
    setCamStatus("Too short — hold a bit longer", "ready");
    return;
  }
  if (rec.features.length < 8) { setCamStatus("No face seen — face the camera", "ready"); return; }
  state.lastClip = rec;
  await recognize(rec);
}

async function recognize(rec) {
  state.busy = true;
  setCamStatus("Reading lips…", "busy");
  $("#readingBar").classList.remove("hidden");
  try {
    const r = await api("/api/recognize", { body: { frames: rec.frames, keypoints: rec.keypoints, timestamps: rec.timestamps, features: rec.features } });
    (r.fallbacks || []).forEach((n) => toast(`<b>Switched engine</b> ${esc(n)}`));
    showResult(r);
    if (r.text && state.settings.speak_automatically) speak(r.text);
    setCamStatus("Ready", "ready");
  } catch (e) {
    toast(`<b>Couldn't read that</b> ${esc(e.message)}`);
    setCamStatus("Ready", "ready");
  } finally {
    state.busy = false;
    $("#readingBar").classList.add("hidden");
    state.ring = [];
  }
}

function showResult(r) {
  state.lastResult = r;
  const t = $("#heroText");
  t.classList.remove("pop", "placeholder", "lang-hi");
  void t.offsetWidth;
  if (r.text) {
    t.textContent = r.text;
    t.classList.add("pop");
    if (isHindi(r.text)) t.classList.add("lang-hi");
  } else {
    t.textContent = r.message || "Didn't catch that.";
    t.classList.add("placeholder");
  }
  $("#heroRaw").textContent = r.raw ? `raw · ${r.raw}` + (r.ctc_raw && r.ctc_raw !== r.raw ? `  ·  lips only · ${r.ctc_raw}` : "") : "";
  const chips = [];
  if (r.source === "memory") chips.push(`<span class="chip memory">Voice Memory · ${Math.round((r.confidence || 0) * 100)}%</span>`);
  if (r.source === "live") chips.push(`<span class="chip live">Live lip reading · ${r.vsr_engine === "colab" ? "Colab GPU" : "This Mac"}</span>`);
  if (r.cleanup_engine) chips.push(`<span class="chip">${{ claude: "Corrected by Claude", openai: "Corrected by OpenAI", sarvam: "Corrected by Sarvam", basic: "No AI correction" }[r.cleanup_engine] || r.cleanup_engine}</span>`);
  if (r.seconds != null) chips.push(`<span class="chip">${r.seconds.toFixed(1)} s</span>`);
  if ((r.fallbacks || []).length) chips.push(`<span class="chip warn">fallback used</span>`);
  $("#heroChips").innerHTML = chips.join("");
  $("#heroActions").classList.toggle("hidden", !r.text);
  $("#learnBtn").disabled = !state.lastClip || !state.lastClip.features.length;
  $("#learnBtn").innerHTML = `<svg viewBox="0 0 24 24"><path d="M5 12l4 4 10-10"/></svg>${r.source === "memory" ? "Correct — reinforce memory" : "Save to Voice Memory"}`;
  if (r.text) addHistory(r);
}

function addHistory(r) {
  const el = document.createElement("div");
  el.className = "h-item";
  const src = r.source === "memory" ? "Voice Memory" : `Live · ${r.vsr_engine === "colab" ? "Colab" : "Mac"}${r.cleanup_engine ? " · " + r.cleanup_engine : ""}`;
  el.innerHTML = `<div class="h-text ${isHindi(r.text) ? "lang-hi" : ""}">${esc(r.text)}</div><div class="h-time">${new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</div><div class="h-meta">${esc(src)}${r.raw ? " · raw: " + esc(r.raw) : ""}</div>`;
  $("#history").prepend(el);
}

async function speak(text) {
  try {
    const r = await api("/api/speak", { body: { text } });
    (r.notes || []).forEach((n) => toast(`<b>Voice fallback</b> ${esc(n)}`));
    if (r.audio_b64) await new Audio(`data:${r.mime || "audio/wav"};base64,${r.audio_b64}`).play();
    else if (r.engine === "browser") browserSpeak(text);
  } catch (e) { browserSpeak(text); }
}
function browserSpeak(text) {
  if (!("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = isHindi(text) ? "hi-IN" : "en-IN";
  const v = speechSynthesis.getVoices().find((x) => x.lang === u.lang) || speechSynthesis.getVoices().find((x) => x.lang.startsWith(u.lang.slice(0, 2)));
  if (v) u.voice = v;
  speechSynthesis.cancel(); speechSynthesis.speak(u);
}

// hold-to-talk: Space or the round button
function isTyping() { const a = document.activeElement; return a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.tagName === "SELECT" || a.isContentEditable); }
document.addEventListener("keydown", (e) => {
  if (e.code === "Escape" && document.body.classList.contains("present")) togglePresent(false);
  if (e.code !== "Space" || e.repeat || isTyping() || state.page !== "live" || state.train) return;
  e.preventDefault();
  if (state.settings.capture_mode === "hold") beginRecording(newRec(), "hold");
});
document.addEventListener("keyup", (e) => {
  if (e.code !== "Space" || isTyping() || state.train) return;
  if (state.rec && state.rec.kind === "hold") { e.preventDefault(); finishRecording(); }
});
const recBtn = $("#recBtn");
recBtn.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  if (state.settings.capture_mode === "auto") {
    state.autoOn = !state.autoOn;
    if (!state.autoOn && state.rec) state.rec = null;
    applyCaptureUI();
    return;
  }
  recBtn.setPointerCapture(e.pointerId);
  beginRecording(newRec(), "hold");
});
const endHold = () => { if (state.rec && state.rec.kind === "hold") finishRecording(); };
recBtn.addEventListener("pointerup", endHold);
recBtn.addEventListener("pointercancel", endHold);

function applyCaptureUI() {
  const auto = state.settings.capture_mode === "auto";
  recBtn.classList.toggle("auto", auto && state.autoOn);
  $("#recHint").innerHTML = auto
    ? (state.autoOn ? "Hands-free — just start mouthing. Tap to pause." : "Paused — tap to resume hands-free")
    : "Hold <kbd>Space</kbd> and mouth your sentence";
}

// result actions
$("#speakAgain").onclick = () => { const t = $("#heroText").textContent.trim(); if (t) speak(t); };
$("#editBtn").onclick = () => {
  const t = $("#heroText");
  const on = t.getAttribute("contenteditable") !== "true";
  t.setAttribute("contenteditable", on ? "true" : "false");
  if (on) { t.focus(); document.getSelection().selectAllChildren(t); }
};
$("#heroText").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); e.target.setAttribute("contenteditable", "false"); } });
$("#learnBtn").onclick = async () => {
  const text = $("#heroText").textContent.trim();
  if (!text || !state.lastClip) return;
  $("#heroText").setAttribute("contenteditable", "false");
  try {
    const r = await api("/api/memory/learn", { body: { text, features: state.lastClip.features } });
    toast(`<b>Learned</b> “${esc(text)}” — ${r.samples} sample${r.samples > 1 ? "s" : ""} in your Voice Memory`, "ok");
    $("#learnBtn").disabled = true;
    refreshMemory();
  } catch (e) { toast(`<b>Couldn't save</b> ${esc(e.message)}`); }
};
$("#clearHist").onclick = () => { $("#history").innerHTML = ""; };

// ------------------------------------------------------------- navigation
function go(page) {
  state.page = page;
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${page}`));
  if (page === "memory") refreshMemory();
  if (page === "settings") refreshStatus(true);
}
$$(".nav-item").forEach((b) => (b.onclick = () => go(b.dataset.page)));
function togglePresent(on) {
  document.body.classList.toggle("present", on);
  if (on) go("live");
}
$("#presentBtn").onclick = () => togglePresent(true);
$("#exitPresent").onclick = () => togglePresent(false);

// ------------------------------------------------------------- settings binding
function applySettingsUI() {
  const s = state.settings;
  $$("[data-setting]").forEach((el) => {
    const k = el.dataset.setting, v = s[k];
    if (el.classList.contains("segmented")) {
      $$("button", el).forEach((b) => b.classList.toggle("on", String(b.dataset.v) === String(v)));
    } else if (el.type === "checkbox") el.checked = !!v;
    else if (document.activeElement !== el) el.value = v ?? "";
  });
  $("#strictV").textContent = Number(s.memory_strictness ?? 0.5).toFixed(2);
  $("#modeDesc").textContent = { smart: "Voice Memory first, then live lip reading", live: "Always use the open-vocabulary lip-reading model", memory: "Only your trained phrases (fastest, any language)" }[s.mode] || "";
  applyCaptureUI();
}
async function saveSetting(key, value) {
  try {
    state.settings = await api("/api/settings", { body: { [key]: value } });
    applySettingsUI();
    refreshStatus();
  } catch (e) { toast(`<b>Setting not saved</b> ${esc(e.message)}`); }
}
$$("[data-setting]").forEach((el) => {
  const k = el.dataset.setting;
  if (el.classList.contains("segmented")) {
    el.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      saveSetting(k, el.dataset.num ? Number(b.dataset.v) : b.dataset.v);
    });
  } else if (el.type === "checkbox") el.addEventListener("change", () => saveSetting(k, el.checked));
  else if (el.type === "range") {
    el.addEventListener("input", () => ($("#strictV").textContent = Number(el.value).toFixed(2)));
    el.addEventListener("change", () => saveSetting(k, Number(el.value)));
  } else el.addEventListener("change", () => saveSetting(k, el.value));
});
$$("[data-test]").forEach((b) => (b.onclick = async () => {
  const comp = b.dataset.test;
  b.disabled = true; const label = b.textContent; b.textContent = "Testing…";
  try {
    const r = await api(`/api/test/${comp}`, { method: "POST" });
    if (comp === "voice") {
      if (r.audio_b64) new Audio(`data:${r.mime};base64,${r.audio_b64}`).play();
      else if (r.engine === "browser") browserSpeak("Hello! Shhhhhhh AI voice is working.");
      $("#voiceDetail").textContent = r.ok ? `Played with ${r.engine}` : r.detail;
    }
    toast(`${r.ok ? "<b>Works</b>" : "<b>Problem</b>"} ${esc(comp)} — ${esc(r.detail || "")}${r.seconds ? ` (${r.seconds}s)` : ""}`, r.ok ? "ok" : "");
  } catch (e) { toast(`<b>Problem</b> ${esc(comp)} — ${esc(e.message)}`); }
  b.disabled = false; b.textContent = label;
  refreshStatus(true);
}));
$("#eraseBtn").onclick = async () => {
  if (!confirmTwice($("#eraseBtn"))) return;
  await api("/api/memory", { method: "DELETE" });
  toast("<b>Voice Memory erased</b>", "ok");
  refreshMemory();
};
function confirmTwice(btn) {
  if (btn.dataset.armed) { delete btn.dataset.armed; btn.textContent = "Erase"; return true; }
  btn.dataset.armed = "1"; btn.textContent = "Tap again to erase";
  setTimeout(() => { delete btn.dataset.armed; btn.textContent = "Erase"; }, 3000);
  return false;
}

// ------------------------------------------------------------- status
async function refreshStatus(force = false) {
  try { state.status = await api(`/api/status${force ? "?force=1" : ""}`); } catch { return; }
  const st = state.status, s = state.settings;
  const localOk = st.local?.state === "ready", colabOk = !!st.colab?.ok;
  const lipOk = s.vsr_engine === "local" ? localOk : colabOk;
  const lipAlt = s.vsr_fallback && (s.vsr_engine === "local" ? colabOk : localOk);
  setEngine("lip", lipOk ? "ok" : lipAlt ? "warn" : "bad", (s.vsr_engine === "local" ? "Mac" : "Colab") + (lipOk ? "" : st.local?.state === "loading" && s.vsr_engine === "local" ? " · loading" : " · offline"));
  const cl = s.cleanup_engine;
  const clOk = cl === "off" || st[cl]?.ok;
  setEngine("clean", cl === "off" ? "" : clOk ? "ok" : (s.cleanup_fallback && (st.claude?.ok || st.openai?.ok || st.sarvam?.ok)) ? "warn" : "bad", { claude: "Claude", openai: "OpenAI", sarvam: "Sarvam", off: "Off" }[cl]);
  const ve = s.voice_engine;
  const vOk = ve === "browser" || (ve === "sarvam" && st.sarvam?.ok) || (ve === "mac" && st.mac_voice?.ok);
  setEngine("voice", ve === "off" ? "" : vOk ? "ok" : s.voice_fallback ? "warn" : "bad", { sarvam: "Sarvam", mac: "Mac", browser: "Browser", off: "Off" }[ve]);
  setEngine("mem", st.memory > 0 ? "ok" : "", `${st.memory} samples`);
  const L = st.local || {};
  $("#localDetail").textContent = { ready: `Ready · ${L.device || "cpu"}`, loading: "Loading model… (first time takes ~30 s)", missing: "Not installed — run ./setup.sh", error: L.error || "Error", idle: "Idle" }[L.state] || "—";
  $("#localDot").className = `dot ${localOk ? "ok" : L.state === "loading" ? "warn" : "bad"}`;
  $("#colabDot").className = `dot ${colabOk ? "ok" : s.colab_url ? "bad" : ""}`;
  $("#colabDetail").textContent = s.colab_url ? (colabOk ? `Connected · ${st.colab.gpu || "GPU"}` : st.colab?.detail || "offline") : "Paste the link the Colab notebook prints";
}
function setEngine(k, cls, val) {
  const row = $(`.engine-row[data-k="${k}"]`);
  row.querySelector(".dot").className = `dot ${cls}`;
  row.querySelector(".val").textContent = val;
}

// ------------------------------------------------------------- Voice Memory page
async function refreshMemory() {
  try { state.memory = await api("/api/memory"); } catch { return; }
  const ps = state.memory.phrases;
  $("#stPhrases").textContent = ps.length;
  $("#stSamples").textContent = state.memory.total_samples;
  const langs = [...new Set(ps.map((p) => p.lang))];
  $("#stLangs").textContent = langs.length ? langs.map((l) => (l === "hi" ? "हिन्दी" : "English")).join(" + ") : "—";
  $("#memEmpty").classList.toggle("hidden", ps.length > 0);
  $("#phraseList").innerHTML = ps.map((p) => {
    const acc = state.accuracy[p.id];
    const accHtml = acc == null ? `<span class="acc none">${p.samples < 2 ? "train" : "—"}</span>`
      : `<span class="acc ${acc >= 0.85 ? "good" : acc >= 0.6 ? "mid" : "bad"}">${Math.round(acc * 100)}%</span>`;
    const dots = Array.from({ length: 6 }, (_, i) => `<i class="${i < p.samples ? "f" : ""}"></i>`).join("");
    return `<div class="p-row${p.active ? "" : " off"}" data-id="${p.id}">
      <div><div class="p-text ${p.lang}">${esc(p.text)}</div>${p.hint ? `<div class="p-hint">${esc(p.hint)}</div>` : ""}</div>
      <div><span class="lang ${p.lang}">${p.lang === "hi" ? "हिन्दी" : "English"}</span></div>
      <div class="sdots">${dots}<small>${p.samples}</small></div>
      <div>${accHtml}</div>
      <div class="p-actions">
        <button class="btn small" data-active="${p.id}" data-on="${p.active ? 1 : 0}" title="Off phrases are kept but never matched">${p.active ? "On" : "Off"}</button>
        <button class="btn small ${p.samples < 3 ? "primary" : ""}" data-train="${p.id}">${p.samples ? "Train more" : "Train"}</button>
        <button class="icon-btn" data-del="${p.id}" title="Delete"><svg viewBox="0 0 24 24"><path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/></svg></button>
      </div></div>`;
  }).join("");
  $$("[data-train]").forEach((b) => (b.onclick = () => openTrain(b.dataset.train)));
  $$("[data-active]").forEach((b) => (b.onclick = async () => {
    await api(`/api/memory/phrases/${b.dataset.active}/active`, { body: { active: b.dataset.on !== "1" } });
    refreshMemory();
  }));
  $$("[data-del]").forEach((b) => (b.onclick = async () => {
    if (b.dataset.armed) { await api(`/api/memory/phrases/${b.dataset.del}`, { method: "DELETE" }); refreshMemory(); return; }
    b.dataset.armed = "1"; b.style.color = "var(--red)"; b.title = "Click again to delete";
    setTimeout(() => { delete b.dataset.armed; b.style.color = ""; }, 2500);
  }));
}
$("#newPhrase").addEventListener("keydown", async (e) => {
  if (e.key !== "Enter") return;
  const text = e.target.value.trim(); if (!text) return;
  try {
    await api("/api/memory/phrases", { body: { text, hint: $("#newHint").value.trim() } });
    e.target.value = ""; $("#newHint").value = "";
    refreshMemory();
  } catch (err) { toast(`<b>Couldn't add</b> ${esc(err.message)}`); }
});
$("#newHint").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#newPhrase").dispatchEvent(new KeyboardEvent("keydown", { key: "Enter" })); });
$("#starterBtn").onclick = async () => { await api("/api/memory/starter", { method: "POST" }); toast("<b>Added</b> 10 English + 10 Hindi starter phrases", "ok"); refreshMemory(); };
$("#healthBtn").onclick = async () => {
  const btn = $("#healthBtn"); btn.disabled = true; btn.textContent = "Checking…";
  let r;
  try { r = await api("/api/memory/evaluate"); } finally { btn.disabled = false; btn.textContent = "Check memory health"; }
  state.accuracy = {};
  r.phrases.forEach((p) => (state.accuracy[p.id] = p.accuracy));
  $("#stHealth").textContent = r.overall == null ? "—" : `${Math.round(r.overall * 100)}%`;
  toast(r.overall == null ? "<b>Train at least 2 samples</b> per phrase first" : `<b>Memory health ${Math.round(r.overall * 100)}%</b> — replayed ${r.tested} samples against your memory`, r.overall >= 0.85 ? "ok" : "");
  refreshMemory();
};

// ------------------------------------------------------------- training sheet
const TARGET = 6;
function openTrain(pid) {
  const p = state.memory.phrases.find((x) => x.id === pid);
  if (!p) return;
  state.train = { pid, p, samples: p.samples, running: false };
  $("#trainPhrase").textContent = p.text;
  $("#trainPhrase").className = `train-phrase ${p.lang}`;
  $("#trainHint").textContent = p.hint || "";
  $("#trainSheet").classList.remove("hidden");
  renderTrain("Ready");
}
function renderTrain(label, cls = "", progress = 0) {
  const t = state.train; if (!t) return;
  $("#trainStep").textContent = t.samples >= TARGET ? `Trained · ${t.samples} samples` : `Sample ${t.samples + 1} of ${TARGET}`;
  $("#trainDots").innerHTML = Array.from({ length: Math.max(TARGET, t.samples) }, (_, i) => `<i class="${i < t.samples ? "f" : ""}"></i>`).join("");
  $("#cdNum").textContent = label;
  $("#countdown").className = `countdown ${cls}`;
  $("#cdProg").style.strokeDashoffset = String(327 * (1 - progress));
  $("#trainGo").textContent = t.running ? "Stop" : t.samples ? "Record another" : "Record";
}
async function trainOnce() {
  const t = state.train;
  if (!state.face) { t.running = false; renderTrain("No face"); toast("<b>No face seen</b> — face the camera"); return; }
  const rec = await recordTrainSample(t);
  t.running = false;
  if (state.train !== t) return;
  if (!rec) { renderTrain("Again"); toast("<b>No lip movement seen</b> — mouth the phrase a little bigger"); return; }
  try {
    const r = await api(`/api/memory/phrases/${t.pid}/samples`, { body: { features: rec.features, frames: rec.frames, keypoints: rec.keypoints, timestamps: rec.timestamps } });
    t.samples = r.samples;
    renderTrain("✓", "done", 1);
    if (t.samples === TARGET) refreshMemory();
  } catch (e) { renderTrain("Again"); toast(`<b>Sample rejected</b> ${esc(e.message)}`); }
}
function recordTrainSample(t) {
  return new Promise((resolve) => {
    const rec = newRec();
    rec.kind = "train";
    const WAIT = 6000, MAX = 9000;  // give up if no lips move in 6 s; cap a sample at 9 s
    rec.onTick = (now) => {
      const dur = now - rec.start;
      if (state.activity > onThreshold()) rec.spoke = rec.spoke || now;
      if (state.activity < offThreshold()) rec.quietSince = rec.quietSince ?? now; else rec.quietSince = null;
      renderTrain(rec.spoke ? "Listening" : "Go", "rec", rec.spoke ? Math.min(1, (now - rec.spoke) / MAX) : 0);
      const done = (rec.spoke && rec.quietSince && now - rec.quietSince > 750 && now - rec.spoke > 500)
        || (!rec.spoke && dur > WAIT) || dur > MAX || !t.running || state.train !== t;
      if (done) {
        rec.onTick = null;
        if (state.rec === rec) state.rec = null;
        resolve(rec.spoke && rec.features.length >= 8 ? rec : null);
      }
    };
    state.rec = rec;
  });
}
$("#trainGo").onclick = () => {
  const t = state.train; if (!t) return;
  if (t.running) { t.running = false; return; }  // Stop: ends the current sample early and keeps it
  t.running = true;
  renderTrain("Go", "rec");
  trainOnce();
};
$("#trainUndo").onclick = async () => {
  const t = state.train; if (!t) return;
  const r = await api(`/api/memory/phrases/${t.pid}/samples/last`, { method: "DELETE" });
  t.samples = r.samples; renderTrain("Undone");
};
function closeTrain() {
  if (state.train) state.train.running = false;
  state.train = null; state.rec = null;
  $("#trainSheet").classList.add("hidden");
  refreshMemory();
}
$("#trainClose").onclick = closeTrain;
$("#trainSheet").addEventListener("click", (e) => { if (e.target.id === "trainSheet") closeTrain(); });

// ------------------------------------------------------------- boot
(async function boot() {
  try { state.settings = await api("/api/settings"); } catch (e) { toast(`<b>Server not reachable</b> start it with ./run.sh`); }
  applySettingsUI();
  refreshStatus(true);
  setInterval(refreshStatus, 4000);
  refreshMemory();
  startCamera();
})();

if (DEV_MOCK) window.__maun = { showResult, state, go, openTrain, refreshMemory };

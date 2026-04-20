// robo-classifier UI
// Single-page vanilla JS: screen switcher + pipeline runner + grid + detail.

const state = {
  inputDir: "",
  profile: null,
  images: [],       // session images
  filteredImages: [], // after grid filter/sort
  currentIndex: 0,   // detail view index into filteredImages
  cropMode: false,
  cropRect: null,    // {l,t,r,b} in normalized (0-1) coords on the shown image
  cropDrag: null,    // {startX, startY} during drag
  cropPreview: false, // when true, detail view shows just the cropped region
};

// ====== Screen management ======
const screens = {
  run:       document.getElementById("screen-run"),
  threshold: document.getElementById("screen-threshold"),
  grid:      document.getElementById("screen-grid"),
  detail:    document.getElementById("screen-detail"),
};
function showScreen(name) {
  for (const [k, el] of Object.entries(screens)) el.hidden = k !== name;
  document.getElementById("crumbs").textContent =
    state.inputDir ? `${state.inputDir}` : "";
}

// ====== Tiny toast ======
let toastTimer = null;
function flashToast(message, ms = 2000) {
  let el = document.getElementById("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.classList.add("visible");
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("visible"), ms);
}

// ====== API helpers ======
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}
function apiPost(path, body) {
  return api(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// ====== Init ======
async function init() {
  // Load profiles
  const { profiles } = await api("/api/profiles");
  const select = document.getElementById("profile");
  select.innerHTML = "";
  if (profiles.length === 0) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "(no models found in models/ — using model.pt)";
    select.appendChild(opt);
  } else {
    for (const p of profiles) {
      const opt = document.createElement("option");
      opt.value = p.name === "(default)" ? "" : p.name;
      opt.textContent = p.description ? `${p.name} — ${p.description}` : p.name;
      select.appendChild(opt);
    }
  }

  // Restore last-used directory if any
  const lastDir = localStorage.getItem("robo.lastInputDir");
  if (lastDir) document.getElementById("input_dir").value = lastDir;

  // Run form
  document.getElementById("run-form").addEventListener("submit", onRun);
  document.getElementById("browse-existing-btn").addEventListener("click", onBrowseExisting);

  // Folder browser
  document.getElementById("browse-btn").addEventListener("click", openBrowser);
  document.getElementById("browser-up").addEventListener("click", browserUp);
  document.getElementById("browser-use").addEventListener("click", browserUseCurrent);
  document.getElementById("browser-close").addEventListener("click", closeBrowser);

  // Thresholds
  document.getElementById("threshold-back").addEventListener("click", () => showScreen("run"));
  document.getElementById("threshold-continue").addEventListener("click", () => {
    showScreen("grid");
    renderGrid();
  });
  document.getElementById("high-slider").addEventListener("input", onThresholdChange);
  document.getElementById("low-slider").addEventListener("input", onThresholdChange);
  document.getElementById("threshold-filter").addEventListener("change", renderThresholdGrid);
  document.getElementById("dry-run-btn").addEventListener("click", () => runWriteKeywords(true));
  document.getElementById("write-btn").addEventListener("click", () => runWriteKeywords(false));

  // Grid
  document.getElementById("back-to-run").addEventListener("click", () => showScreen("run"));
  document.getElementById("back-to-threshold").addEventListener("click", () => showScreen("threshold"));
  document.getElementById("grid-filter").addEventListener("change", renderGrid);
  document.getElementById("grid-sort").addEventListener("change", renderGrid);

  // Detail
  document.getElementById("back-to-grid").addEventListener("click", () => showScreen("grid"));
  document.querySelectorAll(".label-btn").forEach(btn =>
    btn.addEventListener("click", () => applyLabel(btn.dataset.label))
  );
  document.getElementById("crop-btn").addEventListener("click", enterCropMode);
  document.getElementById("crop-save").addEventListener("click", saveCrop);
  document.getElementById("crop-cancel").addEventListener("click", exitCropMode);
  document.getElementById("crop-preview-btn").addEventListener("click", toggleCropPreview);
  document.getElementById("crop-clear").addEventListener("click", clearCrop);
  document.getElementById("junk-btn").addEventListener("click", junkCurrent);

  document.addEventListener("keydown", onKey);

  showScreen("run");
}

// ====== Folder browser ======
// Server-side directory picker. The browser can't hand us real filesystem
// paths (security), so we drive navigation via /api/ls.

let browserCurrentPath = null;

async function openBrowser() {
  const panel = document.getElementById("browser");
  panel.hidden = false;
  // Seed from the input field if it looks like a path, otherwise server default.
  const hint = document.getElementById("input_dir").value.trim();
  await loadBrowserAt(hint || "");
}

function closeBrowser() {
  document.getElementById("browser").hidden = true;
}

async function loadBrowserAt(path) {
  let data;
  try {
    data = await api(`/api/ls?path=${encodeURIComponent(path)}`);
  } catch (e) {
    // If the path doesn't exist, fall back to server default (home).
    try { data = await api(`/api/ls`); }
    catch { alert(`Cannot list: ${e.message}`); return; }
  }
  browserCurrentPath = data.path;
  document.getElementById("browser-path").textContent = data.path;
  const stats = [];
  if (data.raw_count) stats.push(`${data.raw_count} RAW`);
  if (data.jpg_count) stats.push(`${data.jpg_count} JPG`);
  document.getElementById("browser-stats").textContent = stats.join(" · ");

  const list = document.getElementById("browser-list");
  list.innerHTML = "";
  if (data.dirs.length === 0) {
    const li = document.createElement("li");
    li.className = "empty";
    li.textContent = "(no subdirectories)";
    list.appendChild(li);
  } else {
    for (const d of data.dirs) {
      const li = document.createElement("li");
      li.innerHTML = `<span class="icon">📁</span><span>${d.name}</span>`;
      li.addEventListener("click", () => loadBrowserAt(d.path));
      list.appendChild(li);
    }
  }
}

async function browserUp() {
  if (!browserCurrentPath) return;
  // Let the server compute the parent so we don't have to deal with Windows
  // separators etc. client-side — /api/ls returns `parent` in its response.
  const data = await api(`/api/ls?path=${encodeURIComponent(browserCurrentPath)}`);
  if (data.parent) await loadBrowserAt(data.parent);
}

function browserUseCurrent() {
  if (!browserCurrentPath) return;
  document.getElementById("input_dir").value = browserCurrentPath;
  localStorage.setItem("robo.lastInputDir", browserCurrentPath);
  closeBrowser();
}

// ====== Run pipeline ======
async function onRun(ev) {
  ev.preventDefault();
  const inputDir = document.getElementById("input_dir").value.trim();
  if (!inputDir) return;
  const profile = document.getElementById("profile").value || null;
  const skip = !document.getElementById("run_junk_filter").checked;
  const burstEl = document.getElementById("burst_threshold").value;
  const burst = burstEl === "" ? null : parseFloat(burstEl);
  const dry = document.getElementById("dry_run").checked;

  state.inputDir = inputDir;
  localStorage.setItem("robo.lastInputDir", inputDir);
  document.getElementById("run-btn").disabled = true;
  document.getElementById("progress-box").hidden = false;
  document.getElementById("progress-log").textContent = "";
  document.getElementById("progress-bar").value = 0;
  document.getElementById("progress-stage").textContent = "Starting…";

  try {
    const { job_id } = await apiPost("/api/run", {
      input_dir: inputDir,
      profile,
      skip_junk_filter: skip,
      burst_threshold: burst,
      dry_run: dry,
    });
    streamProgress(job_id);
  } catch (e) {
    document.getElementById("progress-stage").textContent = `Error: ${e.message}`;
    document.getElementById("run-btn").disabled = false;
  }
}

function streamProgress(jobId) {
  const es = new EventSource(`/api/progress/${jobId}`);
  const stageEl = document.getElementById("progress-stage");
  const barEl = document.getElementById("progress-bar");
  const textEl = document.getElementById("progress-text");
  const logEl = document.getElementById("progress-log");

  // Log buffer: capped, flushed on a timer so high-frequency progress events
  // (6000+ on a full shoot) don't thrash the DOM.
  const LOG_CAP_LINES = 400;
  const LOG_FLUSH_MS = 150;
  let logBuffer = [];
  let logTimer = null;

  // Timing: remember the job start and each stage's start so we can show
  // both the wall-clock-since-start and the previous stage's duration.
  const jobStartMs = Date.now();
  let currentStage = null;
  let currentStageStartMs = null;
  const fmtElapsed = (ms) => {
    const s = ms / 1000;
    if (s < 60) return `${s.toFixed(1)}s`;
    const m = Math.floor(s / 60);
    const r = (s - m * 60).toFixed(1);
    return `${m}m${r}s`;
  };
  const stamp = () => `[+${fmtElapsed(Date.now() - jobStartMs)}]`;

  // Live ticker: update the header with running elapsed so the user sees
  // the clock move while a long stage is crunching.
  const tickerEl = document.getElementById("progress-elapsed");
  const headerTicker = setInterval(() => {
    if (!tickerEl) return;
    tickerEl.textContent = fmtElapsed(Date.now() - jobStartMs);
  }, 250);

  const scheduleLogFlush = () => {
    if (logTimer) return;
    logTimer = setTimeout(() => {
      logTimer = null;
      if (!logBuffer.length) return;
      const existing = logEl.textContent.split("\n");
      const combined = existing.concat(logBuffer);
      // Keep the tail only
      const trimmed = combined.slice(-LOG_CAP_LINES);
      logEl.textContent = trimmed.join("\n");
      logBuffer = [];
      logEl.scrollTop = logEl.scrollHeight;
    }, LOG_FLUSH_MS);
  };

  // Throttle progress bar + text updates to animation frames.
  let pendingBar = null;
  const scheduleBarUpdate = () => {
    if (pendingBar == null) return;
    barEl.max = pendingBar.total;
    barEl.value = pendingBar.current;
    textEl.textContent = `${pendingBar.current} / ${pendingBar.total}`;
    pendingBar = null;
  };

  es.onmessage = (ev) => {
    let event;
    try { event = JSON.parse(ev.data); } catch { return; }

    if (event.type === "stage") {
      // Close out the previous stage with its duration.
      if (currentStage !== null && currentStageStartMs !== null) {
        const dur = Date.now() - currentStageStartMs;
        logBuffer.push(`${stamp()} [${currentStage}] done in ${fmtElapsed(dur)}`);
      }
      currentStage = event.stage;
      currentStageStartMs = Date.now();
      stageEl.textContent = event.message;
      barEl.value = 0;
      barEl.max = 1;
      textEl.textContent = "";
      logBuffer.push(`${stamp()} [${event.stage}] ${event.message || ""}`);
      scheduleLogFlush();
    } else if (event.type === "progress") {
      // Coalesce: only keep the latest progress update, render on next frame.
      const wasPending = pendingBar != null;
      pendingBar = { current: event.current, total: event.total };
      if (!wasPending) requestAnimationFrame(scheduleBarUpdate);
      // Log sparingly — every 10% or so.
      if (event.total > 0 && event.current % Math.max(1, Math.floor(event.total / 10)) === 0) {
        logBuffer.push(`${stamp()}   ${event.stage}: ${event.current}/${event.total}`);
        scheduleLogFlush();
      }
    } else if (event.type === "status") {
      es.close();
      clearInterval(headerTicker);
      document.getElementById("run-btn").disabled = false;
      // Close out the last stage.
      if (currentStage !== null && currentStageStartMs !== null) {
        const dur = Date.now() - currentStageStartMs;
        logBuffer.push(`${stamp()} [${currentStage}] done in ${fmtElapsed(dur)}`);
      }
      const total = fmtElapsed(Date.now() - jobStartMs);
      if (event.status === "done") {
        stageEl.textContent = `Done in ${total}.`;
        logBuffer.push(`${stamp()} pipeline complete (total ${total})`);
        scheduleLogFlush();
        openSession();
      } else {
        stageEl.textContent = `Error after ${total}: ${event.error || "unknown"}`;
        logBuffer.push(`${stamp()} ERROR: ${event.error || "unknown"}`);
        scheduleLogFlush();
      }
    }
  };
  es.onerror = () => {
    es.close();
    document.getElementById("run-btn").disabled = false;
  };
}

async function onBrowseExisting() {
  const inputDir = document.getElementById("input_dir").value.trim();
  if (!inputDir) return;
  state.inputDir = inputDir;
  localStorage.setItem("robo.lastInputDir", inputDir);
  openSession();
}

// ====== Grid ======
async function openSession() {
  const data = await api(`/api/session?input_dir=${encodeURIComponent(state.inputDir)}`);
  state.images = data.images;
  state.inputDir = data.input_dir;

  // If classification has run (results.csv present), land on Thresholds first
  // so the user can tune before committing keywords. The Threshold screen has
  // a "Continue to grid" button for label/crop review afterwards.
  if (data.has_results) {
    await enterThresholdScreen();
    return;
  }
  showScreen("grid");
  renderGrid();
}

function renderGrid() {
  const filter = document.getElementById("grid-filter").value;
  const sort = document.getElementById("grid-sort").value;

  let imgs = state.images.slice();
  if (filter === "winners") imgs = imgs.filter(i => i.is_winner);
  else if (filter === "selects") imgs = imgs.filter(i => i.classification === "select");

  if (sort === "confidence_desc") {
    imgs.sort((a, b) => (b.confidence_select || 0) - (a.confidence_select || 0));
  } else {
    imgs.sort((a, b) => a.filename.localeCompare(b.filename));
  }
  state.filteredImages = imgs;

  document.getElementById("grid-count").textContent =
    `${imgs.length} of ${state.images.length} images`;

  const grid = document.getElementById("grid");
  grid.innerHTML = "";
  for (const [idx, img] of imgs.entries()) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.addEventListener("click", () => openDetail(idx));
    // Use backend default size so pregenerated cache hits.
    const src = `/api/thumb?input_dir=${encodeURIComponent(state.inputDir)}&filename=${encodeURIComponent(img.filename)}`;
    tile.innerHTML = `
      <img loading="lazy" src="${src}" alt="" />
      <div class="meta">
        <span>${img.filename}</span>
        <span class="conf">${img.confidence_select != null ? img.confidence_select.toFixed(3) : "-"}</span>
      </div>
      ${img.label ? `<div class="label-dot dot-${img.label.toLowerCase()}"></div>` : ""}
      ${img.crop ? `<div class="crop-badge">✂</div>` : ""}
    `;
    grid.appendChild(tile);
  }
}

// ====== Detail ======
function openDetail(idx) {
  state.currentIndex = idx;
  state.cropMode = false;
  state.cropRect = null;
  showScreen("detail");
  renderDetail();
}

async function renderDetail() {
  const img = state.filteredImages[state.currentIndex];
  if (!img) return;
  const imgEl = document.getElementById("detail-img");
  const newSrc = `/api/image?input_dir=${encodeURIComponent(state.inputDir)}&filename=${encodeURIComponent(img.filename)}`;
  // Reset any previous transform so the new image starts in a clean state.
  resetImgTransform();
  imgEl.src = newSrc;
  // Once the image has loaded, apply preview transform if requested.
  imgEl.onload = () => applyCropPreviewIfActive();

  document.getElementById("detail-filename").textContent = img.filename;
  document.getElementById("detail-info").textContent =
    `${state.currentIndex + 1}/${state.filteredImages.length}` +
    (img.confidence_select != null ? ` · select=${img.confidence_select.toFixed(3)}` : "") +
    (img.is_winner ? " · winner" : "");

  // Lazy-load XMP state if we haven't yet
  if (img.label === undefined || img.label === null || img._stateLoaded !== true) {
    try {
      const s = await api(`/api/state?input_dir=${encodeURIComponent(state.inputDir)}&filename=${encodeURIComponent(img.filename)}`);
      img.label = s.label;
      img.crop = s.crop;
      img._stateLoaded = true;
      // State might now include a crop — reapply preview if it's active.
      applyCropPreviewIfActive();
    } catch {
      /* ignore — leave nulls */
    }
  }

  // Label button active state
  for (const btn of document.querySelectorAll(".label-btn")) {
    btn.classList.toggle("active", btn.dataset.label === (img.label || ""));
  }

  // Preview-mode button visual state
  document.getElementById("crop-preview-btn").classList.toggle("active", state.cropPreview);

  // Enable/disable crop-related buttons honestly based on whether a crop exists.
  const hasCrop = !!img.crop;
  document.getElementById("crop-preview-btn").disabled = !hasCrop && !state.cropPreview;
  document.getElementById("crop-clear").disabled = !hasCrop;

  // Crop overlay state
  document.getElementById("crop-overlay").hidden = true;
  document.getElementById("crop-save").hidden = true;
  document.getElementById("crop-cancel").hidden = true;
  document.getElementById("crop-btn").hidden = false;
  document.getElementById("crop-hint").hidden = true;
}

function navigateDetail(delta) {
  const n = state.filteredImages.length;
  if (n === 0) return;
  state.currentIndex = (state.currentIndex + delta + n) % n;
  if (state.cropMode) exitCropMode();
  // Preview mode carries across navigation — if the next image has a crop,
  // the preview transform is re-applied in renderDetail. If not, it no-ops.
  renderDetail();
}

async function applyLabel(color) {
  const img = state.filteredImages[state.currentIndex];
  if (!img) return;
  await apiPost("/api/label", {
    input_dir: state.inputDir,
    filename: img.filename,
    color,
  });
  img.label = color || null;
  renderDetail();
}

async function junkCurrent() {
  const img = state.filteredImages[state.currentIndex];
  if (!img) return;
  if (!confirm(`Move ${img.filename} to junk?`)) return;
  await apiPost("/api/junk", {
    input_dir: state.inputDir,
    filename: img.filename,
  });
  // Remove from local lists and advance
  state.images = state.images.filter(i => i.filename !== img.filename);
  state.filteredImages.splice(state.currentIndex, 1);
  if (state.filteredImages.length === 0) { showScreen("grid"); renderGrid(); return; }
  if (state.currentIndex >= state.filteredImages.length) state.currentIndex = 0;
  renderDetail();
}

// ====== Crop ======
function enterCropMode() {
  state.cropMode = true;
  state.cropRect = null;
  const overlay = document.getElementById("crop-overlay");
  overlay.hidden = false;
  document.getElementById("crop-save").hidden = false;
  document.getElementById("crop-save").disabled = true;   // enabled once a rect is drawn
  document.getElementById("crop-cancel").hidden = false;
  document.getElementById("crop-btn").hidden = true;
  document.getElementById("crop-hint").hidden = false;

  overlay.onmousedown = startCropDrag;
}

function exitCropMode() {
  state.cropMode = false;
  state.cropRect = null;
  state.cropDrag = null;
  const overlay = document.getElementById("crop-overlay");
  overlay.hidden = true;
  overlay.onmousedown = null;
  document.getElementById("crop-save").hidden = true;
  document.getElementById("crop-cancel").hidden = true;
  document.getElementById("crop-btn").hidden = false;
  document.getElementById("crop-hint").hidden = true;
  document.getElementById("crop-rect").style.display = "none";
}

// Crop drag math works in the image's displayed-pixel space then normalizes
// to 0-1 over the rendered image bounds (which correspond 1:1 to the original).
function imageBounds() {
  const img = document.getElementById("detail-img");
  const rect = img.getBoundingClientRect();
  return rect;
}

function startCropDrag(ev) {
  const bounds = imageBounds();
  // Clamp start point to image bounds so clicks in letterbox still work —
  // the drag will begin at the nearest image edge.
  const x = Math.max(bounds.left, Math.min(bounds.right, ev.clientX));
  const y = Math.max(bounds.top,  Math.min(bounds.bottom, ev.clientY));
  state.cropDrag = { startX: x, startY: y, bounds };
  ev.preventDefault();
  document.addEventListener("mousemove", onCropDrag);
  document.addEventListener("mouseup", endCropDrag, { once: true });
}

function onCropDrag(ev) {
  if (!state.cropDrag) return;
  const { startX, startY, bounds } = state.cropDrag;
  const cx = Math.max(bounds.left, Math.min(bounds.right, ev.clientX));
  const cy = Math.max(bounds.top, Math.min(bounds.bottom, ev.clientY));
  const left = Math.min(startX, cx), right = Math.max(startX, cx);
  const top  = Math.min(startY, cy), bottom = Math.max(startY, cy);

  const rect = document.getElementById("crop-rect");
  rect.style.display = "block";
  rect.style.left   = (left - bounds.left) + "px";
  rect.style.top    = (top - bounds.top) + "px";
  rect.style.width  = (right - left) + "px";
  rect.style.height = (bottom - top) + "px";

  // Position relative to the overlay, which covers the full stage.
  // We offset by the image's position within the overlay.
  const overlay = document.getElementById("crop-overlay");
  const overlayRect = overlay.getBoundingClientRect();
  rect.style.left = (left - overlayRect.left) + "px";
  rect.style.top  = (top  - overlayRect.top ) + "px";

  state.cropRect = {
    l: (left - bounds.left) / bounds.width,
    t: (top - bounds.top) / bounds.height,
    r: (right - bounds.left) / bounds.width,
    b: (bottom - bounds.top) / bounds.height,
  };
}

function endCropDrag() {
  document.removeEventListener("mousemove", onCropDrag);
  state.cropDrag = null;
  // Enable the Save button now that a rect exists.
  if (state.cropRect) {
    document.getElementById("crop-save").disabled = false;
    document.getElementById("crop-hint").hidden = true;
  }
}

async function saveCrop() {
  if (!state.cropRect) {
    flashToast("Drag a rectangle on the image first.");
    return;
  }
  const img = state.filteredImages[state.currentIndex];
  await apiPost("/api/crop", {
    input_dir: state.inputDir,
    filename: img.filename,
    left: state.cropRect.l,
    top: state.cropRect.t,
    right: state.cropRect.r,
    bottom: state.cropRect.b,
  });
  img.crop = {
    left:   state.cropRect.l,
    top:    state.cropRect.t,
    right:  state.cropRect.r,
    bottom: state.cropRect.b,
    angle:  0,
  };
  exitCropMode();
  renderDetail();
}

// ====== Crop preview ======
// Toggle between "whole image" and "zoomed to crop rect" view. Uses a CSS
// transform on the <img> so we don't need a server round-trip for a rendered
// crop — the XMP metadata is unchanged either way.

function resetImgTransform() {
  const imgEl = document.getElementById("detail-img");
  imgEl.style.transform = "";
  imgEl.style.maxWidth = "";
  imgEl.style.maxHeight = "";
  imgEl.style.width = "";
  imgEl.style.height = "";
}

function applyCropPreviewIfActive() {
  if (!state.cropPreview) { resetImgTransform(); return; }
  const img = state.filteredImages[state.currentIndex];
  const imgEl = document.getElementById("detail-img");
  if (!img || !img.crop || !imgEl.naturalWidth) { resetImgTransform(); return; }

  const crop = img.crop;
  const stage = document.getElementById("detail-stage");
  const natW = imgEl.naturalWidth, natH = imgEl.naturalHeight;
  const cropWpx = (crop.right - crop.left) * natW;
  const cropHpx = (crop.bottom - crop.top) * natH;
  const stageW = stage.clientWidth, stageH = stage.clientHeight;

  // Scale so the cropped region fits inside the stage, preserving aspect.
  const scale = Math.min(stageW / cropWpx, stageH / cropHpx);
  const dispW = cropWpx * scale;
  const dispH = cropHpx * scale;

  // Letterbox: center the cropped region inside the stage.
  const offsetX = (stageW - dispW) / 2;
  const offsetY = (stageH - dispH) / 2;

  // Translate so the crop's top-left lands at (offsetX, offsetY), then scale.
  const tx = offsetX - crop.left * natW * scale;
  const ty = offsetY - crop.top  * natH * scale;

  // Override the default object-fit:contain sizing so transform math starts
  // from the image's natural pixel dimensions.
  imgEl.style.maxWidth = "none";
  imgEl.style.maxHeight = "none";
  imgEl.style.width = natW + "px";
  imgEl.style.height = natH + "px";
  imgEl.style.transformOrigin = "0 0";
  imgEl.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
}

function toggleCropPreview() {
  const img = state.filteredImages[state.currentIndex];
  if (!img) return;
  if (!img.crop && !state.cropPreview) {
    flashToast("No crop on this image yet — draw one with Crop (C) first.");
    return;
  }
  state.cropPreview = !state.cropPreview;
  document.getElementById("crop-preview-btn").classList.toggle("active", state.cropPreview);
  applyCropPreviewIfActive();
}

// Reapply preview transform on window resize so the letterboxing stays correct.
window.addEventListener("resize", () => {
  if (state.cropPreview) applyCropPreviewIfActive();
});


async function clearCrop() {
  const img = state.filteredImages[state.currentIndex];
  if (!img) return;
  if (!img.crop) {
    flashToast("No crop to clear.");
    return;
  }
  await apiPost("/api/crop/clear", {
    input_dir: state.inputDir,
    filename: img.filename,
    color: "",  // unused but matches LabelRequest shape
  });
  img.crop = null;
  // Nothing to preview anymore — exit preview mode so the full image shows.
  if (state.cropPreview) {
    state.cropPreview = false;
    resetImgTransform();
  }
  renderDetail();
}

// ====== Keyboard ======
function onKey(ev) {
  // Only act on detail screen (except Escape always)
  const onDetail = !screens.detail.hidden;
  if (ev.key === "Escape" && state.cropMode) { exitCropMode(); return; }
  if (!onDetail) return;
  if (ev.target.tagName === "INPUT" || ev.target.tagName === "TEXTAREA") return;

  switch (ev.key) {
    case "ArrowLeft":  navigateDetail(-1); ev.preventDefault(); break;
    case "ArrowRight": navigateDetail(1);  ev.preventDefault(); break;
    case "1": applyLabel("Green"); break;
    case "2": applyLabel("Yellow"); break;
    case "3": applyLabel("Red"); break;
    case "4": applyLabel("Blue"); break;
    case "5": applyLabel("Purple"); break;
    case "0": applyLabel(""); break;
    case "c": case "C":
      if (state.cropMode) saveCrop(); else enterCropMode();
      break;
    case "p": case "P": toggleCropPreview(); break;
    case "x": case "X": junkCurrent(); break;
    case "Escape": showScreen("grid"); break;
  }
}

// ====== Thresholds screen ======
// Mirrors the recovered review.py: two sliders, live histogram, tier chips,
// filtered thumb grid, write-keywords button with clear-first for safe re-runs.

const thresholdState = {
  high: 0.95,
  low: 0.85,
  winners: [],   // subset of state.images where is_winner == true
  drawTimer: null,
};

async function enterThresholdScreen() {
  // openSession() has already populated state.images. Filter down to winners
  // with confidence data (the set the tier threshold applies to).
  thresholdState.winners = state.images.filter(
    i => i.is_winner && i.confidence_select != null
  );

  document.getElementById("threshold-summary").textContent =
    `${state.images.length} images · ${thresholdState.winners.length} burst winners`;

  // Seed slider values from persisted state if we have any.
  const savedHigh = parseFloat(localStorage.getItem("robo.high") || "");
  const savedLow  = parseFloat(localStorage.getItem("robo.low")  || "");
  thresholdState.high = Number.isFinite(savedHigh) ? savedHigh : 0.95;
  thresholdState.low  = Number.isFinite(savedLow)  ? savedLow  : 0.85;
  document.getElementById("high-slider").value = thresholdState.high;
  document.getElementById("low-slider").value  = thresholdState.low;
  document.getElementById("high-val").textContent = thresholdState.high.toFixed(2);
  document.getElementById("low-val").textContent  = thresholdState.low.toFixed(2);

  showScreen("threshold");
  document.getElementById("back-to-threshold").hidden = false;  // make it visible on grid too
  refreshThresholdView();
}

function onThresholdChange(ev) {
  const el = ev.target;
  if (el.id === "high-slider") {
    thresholdState.high = parseFloat(el.value);
    if (thresholdState.high <= thresholdState.low) {
      thresholdState.low = Math.max(0.50, thresholdState.high - 0.01);
      document.getElementById("low-slider").value = thresholdState.low;
      document.getElementById("low-val").textContent = thresholdState.low.toFixed(2);
    }
  } else {
    thresholdState.low = parseFloat(el.value);
    if (thresholdState.low >= thresholdState.high) {
      thresholdState.high = Math.min(1.00, thresholdState.low + 0.01);
      document.getElementById("high-slider").value = thresholdState.high;
      document.getElementById("high-val").textContent = thresholdState.high.toFixed(2);
    }
  }
  document.getElementById("high-val").textContent = thresholdState.high.toFixed(2);
  document.getElementById("low-val").textContent  = thresholdState.low.toFixed(2);
  localStorage.setItem("robo.high", String(thresholdState.high));
  localStorage.setItem("robo.low",  String(thresholdState.low));
  refreshThresholdView();
}

// Coalesce the (potentially expensive) grid redraw onto the next frame so
// dragging the slider feels responsive.
function refreshThresholdView() {
  updateThresholdStats();
  drawHistogram();
  if (thresholdState.drawTimer) cancelAnimationFrame(thresholdState.drawTimer);
  thresholdState.drawTimer = requestAnimationFrame(renderThresholdGrid);
}

function updateThresholdStats() {
  const { high, low, winners } = thresholdState;
  let nAuto = 0, nReview = 0, nBelow = 0;
  const tierCounts = {};
  for (const w of winners) {
    const c = w.confidence_select;
    if (c >= high) nAuto++;
    else if (c >= low) nReview++;
    else nBelow++;
    if (c >= low && c >= 0.90) {
      // Tier keywords only start at 0.90 (see classify.get_tier_keyword).
      const tier = Math.min(99, Math.floor(c * 100));
      const key = `robo_${tier}`;
      tierCounts[key] = (tierCounts[key] || 0) + 1;
    }
  }
  document.getElementById("st-auto").textContent   = nAuto;
  document.getElementById("st-review").textContent = nReview;
  document.getElementById("st-below").textContent  = nBelow;
  document.getElementById("st-total").textContent  = winners.length;

  const chipsEl = document.getElementById("tier-chips");
  chipsEl.innerHTML = "";
  for (const [k, v] of Object.entries(tierCounts).sort().reverse()) {
    const chip = document.createElement("span");
    chip.className = "tier-chip";
    chip.textContent = `${k}: ${v}`;
    chipsEl.appendChild(chip);
  }
}

function drawHistogram() {
  const canvas = document.getElementById("histogram");
  const wrap = canvas.parentElement;
  const dpr = window.devicePixelRatio || 1;
  const W = wrap.clientWidth - 16;
  const H = 80;
  canvas.width = W * dpr;
  canvas.height = H * dpr;
  canvas.style.width = W + "px";
  canvas.style.height = H + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const N_BINS = 50;  // 0.50 → 1.00 at 0.01 per bin
  const bins = new Array(N_BINS).fill(0);
  for (const w of thresholdState.winners) {
    const c = w.confidence_select;
    if (c < 0.50 || c > 1.00) continue;
    const idx = Math.min(N_BINS - 1, Math.floor((c - 0.50) * 100));
    bins[idx]++;
  }
  const maxVal = Math.max(...bins, 1);
  const barW = W / N_BINS;
  const { high, low } = thresholdState;

  for (let i = 0; i < N_BINS; i++) {
    const conf = 0.50 + (i + 0.5) / 100;
    const hPx = Math.ceil((bins[i] / maxVal) * (H - 4));
    const x = i * barW;
    const y = H - hPx;
    ctx.fillStyle = confColor(conf, low, high);
    ctx.fillRect(x + 0.5, y, Math.max(barW - 1, 1), hPx);
  }

  // Dashed threshold lines
  function drawLine(threshold, color, label) {
    const x = Math.round(((threshold - 0.50) / 0.50) * W);
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, 0); ctx.lineTo(x, H);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = "10px system-ui";
    const lw = ctx.measureText(label).width;
    ctx.fillText(label, Math.min(x + 3, W - lw - 2), 10);
  }
  drawLine(low,  "hsl(220,80%,65%)", `low ${low.toFixed(2)}`);
  drawLine(high, "hsl(10,80%,65%)",  `high ${high.toFixed(2)}`);
}

function confColor(conf, low, high) {
  if (conf < low) return "hsl(230,25%,30%)";
  const t = (conf - low) / (1.0 - low);
  const hue = Math.round(240 * (1 - t));
  const lit = conf >= high ? 48 : 58;
  return `hsl(${hue},85%,${lit}%)`;
}

function renderThresholdGrid() {
  const filter = document.getElementById("threshold-filter").value;
  const { high, low, winners } = thresholdState;
  let visible = winners;
  if (filter === "range")     visible = winners.filter(w => w.confidence_select >= low && w.confidence_select < high);
  else if (filter === "above_low") visible = winners.filter(w => w.confidence_select >= low);
  // sort by confidence desc
  visible = visible.slice().sort((a, b) => b.confidence_select - a.confidence_select);

  document.getElementById("threshold-grid-count").textContent =
    `${visible.length} of ${winners.length} winners shown`;

  const grid = document.getElementById("threshold-grid");
  grid.innerHTML = "";
  // Cap to something sensible — the DOM doesn't love 10k tiles.
  const CAP = 500;
  for (const img of visible.slice(0, CAP)) {
    const tile = document.createElement("div");
    tile.className = "tile";
    tile.style.borderColor = confColor(img.confidence_select, low, high);
    tile.style.borderWidth = "2px";
    tile.style.borderStyle = "solid";
    tile.addEventListener("click", () => {
      // Jump to the detail view for this image. Match filteredImages to winners
      // so ←/→ navigation on detail walks the winner set.
      state.filteredImages = winners.slice().sort((a, b) => b.confidence_select - a.confidence_select);
      state.currentIndex = state.filteredImages.findIndex(x => x.filename === img.filename);
      showScreen("detail");
      renderDetail();
    });
    const src = `/api/thumb?input_dir=${encodeURIComponent(state.inputDir)}&filename=${encodeURIComponent(img.filename)}`;
    tile.innerHTML = `
      <img loading="lazy" src="${src}" alt="" />
      <div class="meta">
        <span>${img.filename}</span>
        <span class="conf">${img.confidence_select.toFixed(3)}</span>
      </div>
    `;
    grid.appendChild(tile);
  }
  if (visible.length > CAP) {
    const note = document.createElement("div");
    note.style.gridColumn = "1 / -1";
    note.style.color = "var(--fg-dim)";
    note.style.padding = "8px";
    note.textContent = `Showing first ${CAP} of ${visible.length}. Tighten filters to see more targeted results.`;
    grid.appendChild(note);
  }
}

async function runWriteKeywords(dryRun) {
  const statusEl = document.getElementById("write-status");
  const { low, high } = thresholdState;
  if (!dryRun) {
    const nAuto = document.getElementById("st-auto").textContent;
    const nReview = document.getElementById("st-review").textContent;
    if (!confirm(
      `Write keywords for ${nAuto} auto + ${nReview} review winners?\n` +
      `  low=${low.toFixed(2)}  high=${high.toFixed(2)}`
    )) return;
  }
  statusEl.className = "write-status run";
  statusEl.textContent = dryRun ? "Running dry run…" : "Writing keywords…";
  document.getElementById("write-btn").disabled = true;
  document.getElementById("dry-run-btn").disabled = true;
  try {
    const body = {
      input_dir: state.inputDir,
      low: thresholdState.low,
      dry_run: dryRun,
      clear_first: document.getElementById("clear-first").checked,
    };
    const data = await apiPost("/api/write_keywords", body);
    const tiers = Object.entries(data.tier_counts || {})
      .sort().reverse().map(([k, v]) => `${k}:${v}`).join(" ");
    const prefix = dryRun ? "[dry run] " : "";
    statusEl.className = "write-status ok";
    statusEl.textContent = `${prefix}${data.n_tagged} tier keywords, ${data.n_select} select tags${tiers ? " — " + tiers : ""}`;
  } catch (e) {
    statusEl.className = "write-status err";
    statusEl.textContent = `Error: ${e.message}`;
  } finally {
    document.getElementById("write-btn").disabled = false;
    document.getElementById("dry-run-btn").disabled = false;
  }
}

window.addEventListener("resize", () => {
  if (!screens.threshold.hidden) drawHistogram();
});

init();

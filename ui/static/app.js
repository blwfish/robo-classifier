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
  run:    document.getElementById("screen-run"),
  grid:   document.getElementById("screen-grid"),
  detail: document.getElementById("screen-detail"),
};
function showScreen(name) {
  for (const [k, el] of Object.entries(screens)) el.hidden = k !== name;
  document.getElementById("crumbs").textContent =
    state.inputDir ? `${state.inputDir}` : "";
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

  // Run form
  document.getElementById("run-form").addEventListener("submit", onRun);
  document.getElementById("browse-existing-btn").addEventListener("click", onBrowseExisting);

  // Grid
  document.getElementById("back-to-run").addEventListener("click", () => showScreen("run"));
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
      stageEl.textContent = event.message;
      barEl.value = 0;
      barEl.max = 1;
      textEl.textContent = "";
      // Stage transitions are rare enough to log verbatim and immediately.
      logBuffer.push(`[${event.stage}] ${event.message || ""}`);
      scheduleLogFlush();
    } else if (event.type === "progress") {
      // Coalesce: only keep the latest progress update, render on next frame.
      const wasPending = pendingBar != null;
      pendingBar = { current: event.current, total: event.total };
      if (!wasPending) requestAnimationFrame(scheduleBarUpdate);
      // Log sparingly — every 10% or so.
      if (event.total > 0 && event.current % Math.max(1, Math.floor(event.total / 10)) === 0) {
        logBuffer.push(`  ${event.stage}: ${event.current}/${event.total}`);
        scheduleLogFlush();
      }
    } else if (event.type === "status") {
      es.close();
      document.getElementById("run-btn").disabled = false;
      if (event.status === "done") {
        stageEl.textContent = "Done.";
        logBuffer.push("done.");
        scheduleLogFlush();
        openSession();
      } else {
        stageEl.textContent = `Error: ${event.error || "unknown"}`;
        logBuffer.push(`ERROR: ${event.error || "unknown"}`);
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
  openSession();
}

// ====== Grid ======
async function openSession() {
  const data = await api(`/api/session?input_dir=${encodeURIComponent(state.inputDir)}`);
  state.images = data.images;
  state.inputDir = data.input_dir;
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

  // Crop overlay state
  document.getElementById("crop-overlay").hidden = true;
  document.getElementById("crop-save").hidden = true;
  document.getElementById("crop-cancel").hidden = true;
  document.getElementById("crop-btn").hidden = false;
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
  document.getElementById("crop-cancel").hidden = false;
  document.getElementById("crop-btn").hidden = true;

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
  const x = ev.clientX, y = ev.clientY;
  if (x < bounds.left || x > bounds.right || y < bounds.top || y > bounds.bottom) return;
  state.cropDrag = { startX: x, startY: y, bounds };
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
}

async function saveCrop() {
  if (!state.cropRect) { exitCropMode(); return; }
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
    // Nothing to preview; silently no-op (could flash a toast later).
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

init();

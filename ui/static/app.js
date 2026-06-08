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
  masterLevel: true,         // auto-level master toggle (persisted in localStorage)
  levelOverrides: new Set(), // filenames where the master is reversed for that image
};

// ====== Screen management ======
const screens = {
  library:   document.getElementById("screen-library"),
  settings:  document.getElementById("screen-settings"),
  train:     document.getElementById("screen-train"),
  ingest:    document.getElementById("screen-ingest"),
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

// ====== Theme ======
(function () {
  const THEMES = ["modern", "deco", "full-deco"];
  const saved = localStorage.getItem("robo.theme") || "modern";
  const active = THEMES.includes(saved) ? saved : "modern";
  document.documentElement.setAttribute("data-theme", active);

  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem("robo.theme", t);
    document.querySelectorAll(".theme-opt").forEach(btn =>
      btn.classList.toggle("checked", btn.dataset.themeVal === t)
    );
  }
  applyTheme(active);

  document.querySelectorAll(".theme-opt").forEach(btn =>
    btn.addEventListener("click", () => applyTheme(btn.dataset.themeVal))
  );
})();

// ====== Menubar ======
(function () {
  const menubar = document.getElementById("menubar");
  if (!menubar) return;

  function closeAll() {
    menubar.querySelectorAll(".menu.open").forEach(m => m.classList.remove("open"));
  }

  menubar.addEventListener("click", e => {
    const label = e.target.closest(".menu-label");
    if (!label) return;
    const menu = label.closest(".menu");
    const wasOpen = menu.classList.contains("open");
    closeAll();
    if (!wasOpen) menu.classList.add("open");
    e.stopPropagation();
  });

  // Close when a menu item is activated
  menubar.addEventListener("click", e => {
    if (e.target.closest(".menu-item")) closeAll();
  });

  document.addEventListener("click", closeAll);
  document.addEventListener("keydown", e => { if (e.key === "Escape") closeAll(); });
})();

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

  // Load tuning presets
  const { presets } = await api("/api/presets");
  const presetSelect = document.getElementById("preset");
  presetSelect.innerHTML = "";
  const noneOpt = document.createElement("option");
  noneOpt.value = "";
  noneOpt.textContent = "(code defaults)";
  presetSelect.appendChild(noneOpt);
  for (const p of presets) {
    const opt = document.createElement("option");
    opt.value = p.name;
    opt.textContent = p.description ? `${p.name} — ${p.description}` : p.name;
    presetSelect.appendChild(opt);
  }
  // Remember the last-picked preset across sessions.
  const savedPreset = localStorage.getItem("robo.preset") || "";
  if ([...presetSelect.options].some(o => o.value === savedPreset)) {
    presetSelect.value = savedPreset;
  }
  presetSelect.addEventListener("change", () => {
    localStorage.setItem("robo.preset", presetSelect.value);
  });

  // Restore last-used directory if any
  const lastDir = localStorage.getItem("robo.lastInputDir");
  if (lastDir) document.getElementById("input_dir").value = lastDir;

  // Perf panel — reload when opened, and once on init
  const perfPanel = document.getElementById("perf-panel");
  if (perfPanel) {
    perfPanel.addEventListener("toggle", () => { if (perfPanel.open) loadPerfPanel(); });
    loadPerfPanel();
  }

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

  // Auto-level: master toggle (persisted) + per-image override
  const masterLevelEl = document.getElementById("master-level-chk");
  state.masterLevel = localStorage.getItem("robo.masterLevel") !== "false";
  masterLevelEl.checked = state.masterLevel;
  masterLevelEl.addEventListener("change", () => {
    state.masterLevel = masterLevelEl.checked;
    localStorage.setItem("robo.masterLevel", state.masterLevel);
    updateLevelUI(state.filteredImages[state.currentIndex]);
  });
  document.getElementById("img-level-chk").addEventListener("change", (e) => {
    const img = state.filteredImages[state.currentIndex];
    if (!img) return;
    // Checkbox shows effective level state. Clicking flips it → toggle override.
    const wasEffective = getEffectiveLevelAngle(img) !== 0;
    if (e.target.checked !== wasEffective) {
      if (state.levelOverrides.has(img.filename)) state.levelOverrides.delete(img.filename);
      else state.levelOverrides.add(img.filename);
    }
    updateLevelUI(img);
  });

  // Train screen
  document.getElementById("train-back").addEventListener("click", () => showScreen("ingest"));
  document.getElementById("train-go-btn").addEventListener("click", startTraining);
  document.getElementById("train-again-btn").addEventListener("click", resetTrainScreen);

  trainSelectBrowser = makeBrowser({
    panelId: "train-select-browser", pathId: "train-select-path",
    listId: "train-select-list", upId: "train-select-up",
    useId: "train-select-use", closeId: "train-select-close",
    onSelect(path) {
      document.getElementById("train-select-dir").value = path;
      updateTrainDataCount("select", path);
      updateTrainGoBtn();
      saveTrainRecentDir("select", path);
    },
  });
  trainRejectBrowser = makeBrowser({
    panelId: "train-reject-browser", pathId: "train-reject-path",
    listId: "train-reject-list", upId: "train-reject-up",
    useId: "train-reject-use", closeId: "train-reject-close",
    onSelect(path) {
      document.getElementById("train-reject-dir").value = path;
      updateTrainDataCount("reject", path);
      updateTrainGoBtn();
      saveTrainRecentDir("reject", path);
    },
  });
  document.getElementById("train-select-browse").addEventListener("click", () => {
    const hint = document.getElementById("train-select-dir").value.trim()
      || localStorage.getItem("robo.train.recentSelect") || "";
    trainSelectBrowser.open(hint);
  });
  document.getElementById("train-reject-browse").addEventListener("click", () => {
    const hint = document.getElementById("train-reject-dir").value.trim()
      || localStorage.getItem("robo.train.recentReject") || "";
    trainRejectBrowser.open(hint);
  });
  document.getElementById("train-model-name").addEventListener("input", () => {
    updateTrainGoBtn();
    autofillTrainKeywords();
  });
  document.getElementById("train-accept-kw").addEventListener("input", () => {
    document.getElementById("train-accept-kw").dataset.userEdited = "1";
  });
  document.getElementById("train-reject-kw").addEventListener("input", () => {
    document.getElementById("train-reject-kw").dataset.userEdited = "1";
  });
  ["train-epochs", "train-lr", "train-batch", "train-test-size"].forEach(id => {
    document.getElementById(id).addEventListener("change", saveTrainHyperparams);
  });
  document.getElementById("train-use-model-btn").addEventListener("click", () => {
    // Reload profiles and switch to run screen
    refreshProfiles().then(() => showScreen("run"));
  });

  // Library screen
  document.getElementById("library-btn").addEventListener("click", openLibrary);
  document.getElementById("library-back").addEventListener("click", () => showScreen(state.prevScreen || "ingest"));
  document.getElementById("library-go-settings").addEventListener("click", openSettings);

  // Settings screen
  document.getElementById("settings-btn").addEventListener("click", openSettings);
  document.getElementById("settings-back").addEventListener("click", () => showScreen(state.prevScreen || "ingest"));

  // Menu nav items
  document.getElementById("menu-open-ingest").addEventListener("click", () => showScreen("ingest"));
  document.getElementById("menu-open-run").addEventListener("click", () => showScreen("run"));
  document.getElementById("menu-open-train").addEventListener("click", () => showScreen("train"));
  document.getElementById("menu-shortcuts").addEventListener("click", () =>
    flashToast("← → nav  •  C crop  •  L level  •  1–5 labels  •  Esc grid", 4000)
  );
  document.getElementById("cfg-save-btn").addEventListener("click", saveConfig);
  document.querySelectorAll(".cfg-browse-btn").forEach(btn =>
    btn.addEventListener("click", () => cfgBrowser.open(""))
  );
  cfgBrowser = makeBrowser({
    panelId: "cfg-browser", pathId: "cfg-browser-path",
    listId: "cfg-browser-list", upId: "cfg-browser-up",
    useId: "cfg-browser-use", closeId: "cfg-browser-close",
    onSelect(path) {
      const field = document._cfgBrowseField;
      if (field === "model_library")  document.getElementById("cfg-model-library").value = path;
      if (field === "dataset_scratch") document.getElementById("cfg-dataset-scratch").value = path;
    },
  });
  // Attach field tracking to browse buttons
  document.querySelectorAll(".cfg-browse-btn").forEach(btn =>
    btn.addEventListener("click", () => { document._cfgBrowseField = btn.dataset.field; })
  );

  // Ingest screen
  document.getElementById("ingest-to-train").addEventListener("click", () => openTrainScreen());
  document.getElementById("ingest-to-run").addEventListener("click", () => showScreen("run"));
  document.getElementById("refresh-cards-btn").addEventListener("click", loadCards);
  document.getElementById("add-local-btn").addEventListener("click", () => localBrowser.open(""));
  document.getElementById("ingest-go-btn").addEventListener("click", startIngest);
  document.getElementById("ingest-dest").addEventListener("input", updateIngestGoBtn);
  document.getElementById("ingest-again-btn").addEventListener("click", () => {
    document.getElementById("ingest-progress-box").hidden = true;
    document.getElementById("ingest-done-box").hidden = true;
    document.getElementById("ingest-go-btn").disabled = false;
    initIngest();
  });

  // Browsers for the ingest screen
  localBrowser = makeBrowser({
    panelId: "local-browser", pathId: "local-browser-path",
    listId: "local-browser-list", upId: "local-browser-up",
    useId: "local-browser-use", closeId: "local-browser-close",
    onSelect(path) {
      const label = path.split("/").pop() || path;
      ingestState.localSources.push({ path, label });
      renderLocalSources();
      updateIngestGoBtn();
    },
  });
  destBrowser = makeBrowser({
    panelId: "dest-browser", pathId: "dest-browser-path",
    listId: "dest-browser-list", upId: "dest-browser-up",
    useId: "dest-browser-use", closeId: "dest-browser-close",
    onSelect(path) {
      document.getElementById("ingest-dest").value = path;
      ingestState.destDir = path;
      updateIngestGoBtn();
    },
  });
  document.getElementById("ingest-dest-browse-btn").addEventListener("click", () => {
    const hint = document.getElementById("ingest-dest").value.trim();
    destBrowser.open(hint || "/Volumes");
  });

  document.addEventListener("keydown", onKey);

  showScreen("ingest");
  initIngest();
  checkConfigBanner();
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
  state.profile = profile;
  localStorage.setItem("robo.lastInputDir", inputDir);
  document.getElementById("run-btn").disabled = true;
  document.getElementById("progress-box").hidden = false;
  document.getElementById("progress-log").textContent = "";
  document.getElementById("progress-bar").value = 0;
  document.getElementById("progress-stage").textContent = "Starting…";

  try {
    const preset = document.getElementById("preset").value || null;
    const { job_id } = await apiPost("/api/run", {
      input_dir: inputDir,
      profile,
      preset,
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
        loadPerfPanel();
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
// ── Auto-level helpers ──────────────────────────────────────────────────────

function getEffectiveLevelAngle(img) {
  if (!img) return 0;
  const roll = img.roll_angle || 0;
  if (roll === 0) return 0;
  const isOverridden = state.levelOverrides.has(img.filename);
  // master XOR override: override reverses master for this image
  return (state.masterLevel !== isOverridden) ? roll : 0;
}

function updateLevelUI(img) {
  const roll = img ? (img.roll_angle || 0) : 0;
  const effectiveAngle = getEffectiveLevelAngle(img);
  const chk = document.getElementById("img-level-chk");
  const lbl = document.getElementById("img-level-label");
  const rollEl = document.getElementById("detail-roll");

  if (roll === 0) {
    chk.disabled = true;
    chk.checked = false;
    lbl.textContent = "Level";
    rollEl.textContent = "";
    rollEl.className = "detail-roll";
  } else {
    chk.disabled = false;
    chk.checked = effectiveAngle !== 0;
    lbl.textContent = "Level";
    rollEl.textContent = `${roll > 0 ? "+" : ""}${roll}°`;
    rollEl.className = "detail-roll" + (effectiveAngle !== 0 ? " active" : "");
  }
}

// ────────────────────────────────────────────────────────────────────────────

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
      applyCropPreviewIfActive();
    } catch {
      /* ignore — leave nulls */
    }
  }

  // Lazy-load roll angle for auto-level (cached on img object after first fetch)
  if (img.roll_angle === undefined) {
    img.roll_angle = 0; // show UI immediately while fetch is in flight
    updateLevelUI(img);
    try {
      const meta = await api(`/api/image_meta?input_dir=${encodeURIComponent(state.inputDir)}&filename=${encodeURIComponent(img.filename)}`);
      img.roll_angle = meta.roll_angle;
    } catch {
      img.roll_angle = 0;
    }
  }
  updateLevelUI(img);

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
  const angle = getEffectiveLevelAngle(img);
  await apiPost("/api/crop", {
    input_dir: state.inputDir,
    filename: img.filename,
    left:  state.cropRect.l,
    top:   state.cropRect.t,
    right: state.cropRect.r,
    bottom: state.cropRect.b,
    angle,
  });
  img.crop = {
    left:   state.cropRect.l,
    top:    state.cropRect.t,
    right:  state.cropRect.r,
    bottom: state.cropRect.b,
    angle,
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
      model_name: state.profile || null,
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

// ====== Perf panel ======

async function loadPerfPanel() {
  const wrap = document.getElementById("perf-table-wrap");
  const summary = document.getElementById("perf-summary");
  if (!wrap) return;
  try {
    const { runs } = await api("/api/perf/recent?n=15");
    if (!runs || runs.length === 0) {
      wrap.innerHTML = "<p class='placeholder'>No runs recorded yet.</p>";
      return;
    }
    summary.textContent = `Recent runs (${runs.length})`;
    wrap.innerHTML = "";
    const table = document.createElement("table");
    table.className = "perf-table";
    table.innerHTML = `<thead><tr>
      <th>Date</th><th>Host</th><th>Type</th><th>Device</th>
      <th>Storage</th><th>Preset</th><th>Total</th><th>Stages</th>
    </tr></thead>`;
    const tbody = document.createElement("tbody");

    for (const run of runs) {
      const tr = document.createElement("tr");
      const dt = new Date(run.ts * 1000);
      const date = dt.toLocaleDateString() + " " + dt.toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
      const stages = (run.stages || []);
      const maxDur = Math.max(...stages.map(s => s.duration_s), 0.001);

      const stageCells = stages.map(s => {
        const pct = Math.round(s.duration_s / maxDur * 100);
        const isBottleneck = s.duration_s === maxDur;
        const rate = s.mb_per_s  ? `${s.mb_per_s} MB/s` :
                     s.files_per_s ? `${s.files_per_s} f/s` : "";
        const dur = fmtDur(s.duration_s);
        return `<span class="perf-stage${isBottleneck ? " bottleneck" : ""}"
          title="${s.name}: ${dur}${rate ? "  " + rate : ""}  (${s.files||0} files)"
          style="--pct:${pct}%">${s.name}</span>`;
      }).join("");

      tr.innerHTML = `
        <td class="perf-date">${date}</td>
        <td>${run.hostname}</td>
        <td class="perf-type">${run.run_type}</td>
        <td title="${run.device_name}">${run.device}</td>
        <td class="perf-storage perf-storage-${run.storage_class || 'unknown'}">${run.storage_class || "?"}</td>
        <td>${run.preset || "—"}</td>
        <td>${fmtDur(run.total_duration_s)}</td>
        <td class="perf-stages">${stageCells}</td>
      `;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    wrap.appendChild(table);
  } catch (e) {
    wrap.innerHTML = `<p class='placeholder'>Error loading perf data: ${e.message}</p>`;
  }
}

function fmtDur(s) {
  if (!s) return "—";
  if (s < 60)   return `${s.toFixed(0)}s`;
  if (s < 3600) return `${(s/60).toFixed(1)}m`;
  return `${(s/3600).toFixed(1)}h`;
}

// ====== Generic folder browser controller ======
// Drives any {panel, path-display, list, up, use, close} set of elements.
function makeBrowser({ panelId, pathId, listId, upId, useId, closeId, onSelect }) {
  const panel   = document.getElementById(panelId);
  const pathEl  = document.getElementById(pathId);
  const listEl  = document.getElementById(listId);
  let currentPath = null;

  async function loadAt(path) {
    let data;
    try {
      data = await api(`/api/ls?path=${encodeURIComponent(path)}`);
    } catch {
      try { data = await api("/api/ls"); } catch { return; }
    }
    currentPath = data.path;
    pathEl.textContent = data.path;
    listEl.innerHTML = "";
    if (data.dirs.length === 0) {
      const li = document.createElement("li");
      li.className = "empty";
      li.textContent = "(no subdirectories)";
      listEl.appendChild(li);
    } else {
      for (const d of data.dirs) {
        const li = document.createElement("li");
        li.innerHTML = `<span class="icon">📁</span><span>${d.name}</span>`;
        li.addEventListener("click", () => loadAt(d.path));
        listEl.appendChild(li);
      }
    }
  }

  document.getElementById(upId).addEventListener("click", async () => {
    if (!currentPath) return;
    const data = await api(`/api/ls?path=${encodeURIComponent(currentPath)}`);
    if (data.parent) await loadAt(data.parent);
  });
  document.getElementById(useId).addEventListener("click", () => {
    if (currentPath) onSelect(currentPath);
    panel.hidden = true;
  });
  document.getElementById(closeId).addEventListener("click", () => { panel.hidden = true; });

  return {
    open(hint = "") { panel.hidden = false; loadAt(hint || ""); },
    close() { panel.hidden = true; },
  };
}

// ====== Train screen ======

let trainSelectBrowser = null;
let trainRejectBrowser = null;

// Per-machine persistence for train dirs and hyperparams
function saveTrainRecentDir(which, path) {
  localStorage.setItem(`robo.train.recent${which.charAt(0).toUpperCase() + which.slice(1)}`, path);
}

function saveTrainHyperparams() {
  localStorage.setItem("robo.train.hyperparams", JSON.stringify({
    epochs:    document.getElementById("train-epochs").value,
    lr:        document.getElementById("train-lr").value,
    batch:     document.getElementById("train-batch").value,
    testSplit: document.getElementById("train-test-size").value,
  }));
}

function updateTrainGoBtn() {
  const sel  = document.getElementById("train-select-dir").value.trim();
  const rej  = document.getElementById("train-reject-dir").value.trim();
  const name = document.getElementById("train-model-name").value.trim();
  document.getElementById("train-go-btn").disabled = !(sel && rej && name);
}

function autofillTrainKeywords() {
  const name = document.getElementById("train-model-name").value.trim();
  const acceptEl = document.getElementById("train-accept-kw");
  const rejectEl = document.getElementById("train-reject-kw");
  // Only auto-fill if the field is empty or was previously auto-filled
  // (don't overwrite something the user typed manually)
  const slug = name.replace(/[^a-z0-9_-]/gi, "_").toLowerCase();
  if (!acceptEl.dataset.userEdited) {
    acceptEl.value = slug ? `robo_${slug}_select` : "";
  }
  // Reject keyword stays blank by default — user opts in explicitly
}


async function updateTrainDataCount(which, path) {
  const el = document.getElementById(`train-${which}-count`);
  el.textContent = "…";
  try {
    const sel = document.getElementById("train-select-dir").value.trim();
    const rej = document.getElementById("train-reject-dir").value.trim();
    if (!sel || !rej) return;
    const data = await api(`/api/train/data_stats?select_dir=${encodeURIComponent(sel)}&reject_dir=${encodeURIComponent(rej)}`);
    document.getElementById("train-select-count").textContent = data.select >= 0 ? `${data.select.toLocaleString()} images` : "not found";
    document.getElementById("train-reject-count").textContent = data.reject >= 0 ? `${data.reject.toLocaleString()} images` : "not found";
  } catch { el.textContent = ""; }
}

async function openTrainScreen() {
  // Check config first
  const data = await api("/api/config");
  const missing = data.missing_required || [];
  document.getElementById("train-config-banner").hidden = missing.length === 0;
  document.getElementById("train-go-btn").disabled = missing.length > 0;

  // Restore last-used dirs
  const lastSel = localStorage.getItem("robo.train.recentSelect") || "";
  const lastRej = localStorage.getItem("robo.train.recentReject") || "";
  if (lastSel) { document.getElementById("train-select-dir").value = lastSel; updateTrainDataCount("select", lastSel); }
  if (lastRej) { document.getElementById("train-reject-dir").value = lastRej; updateTrainDataCount("reject", lastRej); }

  // Restore last-used hyperparams
  const hp = JSON.parse(localStorage.getItem("robo.train.hyperparams") || "{}");
  if (hp.epochs    != null) document.getElementById("train-epochs").value    = hp.epochs;
  if (hp.lr        != null) document.getElementById("train-lr").value        = hp.lr;
  if (hp.batch     != null) document.getElementById("train-batch").value     = hp.batch;
  if (hp.testSplit != null) document.getElementById("train-test-size").value = hp.testSplit;

  resetTrainScreen(/*keepDirs=*/true);
  showScreen("train");
}

function resetTrainScreen(keepDirs = false) {
  if (!keepDirs) {
    document.getElementById("train-select-dir").value = "";
    document.getElementById("train-reject-dir").value = "";
    document.getElementById("train-select-count").textContent = "";
    document.getElementById("train-reject-count").textContent = "";
  }
  document.getElementById("train-progress-box").hidden = true;
  document.getElementById("train-done-box").hidden = true;
  document.getElementById("train-status-line").textContent = "";
  document.getElementById("train-go-btn").disabled = false;
  trainChartData = { trainAcc: [], testAcc: [], trainLoss: [] };
  updateTrainGoBtn();
}

// ---- Chart ----

let trainChartData = { trainAcc: [], testAcc: [], trainLoss: [] };

function drawTrainChart() {
  const canvas = document.getElementById("train-chart");
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const W = canvas.parentElement.clientWidth - 16;
  const H = 180;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.width = W + "px"; canvas.style.height = H + "px";
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, W, H);

  const { trainAcc, testAcc, trainLoss } = trainChartData;
  const n = Math.max(trainAcc.length, 1);

  function plotLine(values, color, scale = 1) {
    if (!values.length) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    values.forEach((v, i) => {
      const x = (i / Math.max(n - 1, 1)) * (W - 20) + 10;
      const y = H - 10 - ((v * scale) / 100) * (H - 20);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Grid lines at 25%, 50%, 75%, 100% accuracy
  ctx.strokeStyle = "rgba(255,255,255,0.06)";
  ctx.lineWidth = 1;
  [25, 50, 75, 100].forEach(pct => {
    const y = H - 10 - (pct / 100) * (H - 20);
    ctx.beginPath(); ctx.moveTo(10, y); ctx.lineTo(W - 10, y); ctx.stroke();
    ctx.fillStyle = "rgba(255,255,255,0.25)";
    ctx.font = "9px system-ui";
    ctx.fillText(pct + "%", 0, y + 3);
  });

  plotLine(trainAcc,  "hsl(150,70%,55%)");
  plotLine(testAcc,   "hsl(200,80%,65%)");
  plotLine(trainLoss, "hsl(30,80%,60%)", 10);  // loss scaled ×10 for visibility
}

// ---- Stream ----

async function startTraining() {
  const selDir     = document.getElementById("train-select-dir").value.trim();
  const rejDir     = document.getElementById("train-reject-dir").value.trim();
  const name       = document.getElementById("train-model-name").value.trim();
  const desc       = document.getElementById("train-model-desc").value.trim();
  const acceptKw   = document.getElementById("train-accept-kw").value.trim();
  const rejectKw   = document.getElementById("train-reject-kw").value.trim();
  const epochs  = parseInt(document.getElementById("train-epochs").value)   || 15;
  const lr      = parseFloat(document.getElementById("train-lr").value)      || 1e-4;
  const batch   = parseInt(document.getElementById("train-batch").value)     || 32;
  const tsize   = parseFloat(document.getElementById("train-test-size").value) || 0.2;

  if (!selDir || !rejDir || !name) return;

  document.getElementById("train-go-btn").disabled = true;
  document.getElementById("train-progress-box").hidden = false;
  document.getElementById("train-done-box").hidden = true;
  document.getElementById("train-epoch-log").textContent = "";
  document.getElementById("train-phase-label").textContent = "Starting…";
  document.getElementById("train-progress-bar").value = 0;
  trainChartData = { trainAcc: [], testAcc: [], trainLoss: [] };
  drawTrainChart();

  try {
    const { job_id, model_output } = await apiPost("/api/train/start", {
      select_dir: selDir, reject_dir: rejDir,
      model_name: name, description: desc,
      accept_keyword: acceptKw, reject_keyword: rejectKw,
      test_size: tsize, epochs, learning_rate: lr, batch_size: batch,
    });
    document.getElementById("train-done-path").textContent = model_output;
    streamTrainProgress(job_id, epochs);
  } catch (e) {
    document.getElementById("train-phase-label").textContent = `Error: ${e.message}`;
    document.getElementById("train-go-btn").disabled = false;
  }
}

function streamTrainProgress(jobId, totalEpochs) {
  const es = new EventSource(`/api/train/progress/${jobId}`);
  const phaseEl = document.getElementById("train-phase-label");
  const barEl   = document.getElementById("train-progress-bar");
  const textEl  = document.getElementById("train-progress-text");
  const logEl   = document.getElementById("train-epoch-log");

  let logLines = [];
  let logTimer = null;
  function appendLog(line) {
    logLines.push(line);
    if (!logTimer) logTimer = setTimeout(() => {
      logTimer = null;
      logEl.textContent = logLines.slice(-100).join("\n");
      logEl.scrollTop = logEl.scrollHeight;
    }, 150);
  }

  es.onmessage = (ev) => {
    let event;
    try { event = JSON.parse(ev.data); } catch { return; }
    const t = event.type;

    if (t === "phase") {
      phaseEl.textContent = event.message;
      appendLog(`\n[${event.phase}] ${event.message}`);
    } else if (t === "scan") {
      textEl.textContent = `${event.select.toLocaleString()} select · ${event.reject.toLocaleString()} reject`;
    } else if (t === "split") {
      textEl.textContent =
        `Train: ${event.train_select} sel / ${event.train_reject} rej  ·  ` +
        `Test: ${event.test_select} sel / ${event.test_reject} rej`;
      appendLog(`Split: train ${event.train_select}+${event.train_reject}  test ${event.test_select}+${event.test_reject}`);
    } else if (t === "copy") {
      barEl.max = event.total; barEl.value = event.done;
    } else if (t === "setup") {
      phaseEl.textContent = `Training on ${event.device} · ${event.train_total} train / ${event.test_total} test`;
      barEl.max = totalEpochs; barEl.value = 0;
    } else if (t === "epoch") {
      barEl.value = event.epoch;
      phaseEl.textContent = `Epoch ${event.epoch} / ${event.epochs}`;
      trainChartData.trainAcc.push(event.train_acc);
      trainChartData.testAcc.push(event.test_acc);
      trainChartData.trainLoss.push(event.train_loss);
      drawTrainChart();
      appendLog(
        `  Epoch ${String(event.epoch).padStart(2)}/${event.epochs}` +
        `  train ${event.train_acc.toFixed(1)}%  test ${event.test_acc.toFixed(1)}%` +
        `  loss ${event.train_loss.toFixed(4)}` +
        (event.saved ? "  ✓ saved" : "") +
        `  (${event.elapsed_s}s)`
      );
    } else if (t === "status") {
      es.close();
      if (logTimer) { clearTimeout(logTimer); logEl.textContent = logLines.join("\n"); }
      if (event.status === "done" && event.summary) {
        barEl.value = barEl.max;
        phaseEl.textContent = "Training complete.";
        document.getElementById("train-best-acc").textContent =
          `${event.summary.best_acc}%`;
        document.getElementById("train-done-box").hidden = false;
        loadPerfPanel();
      } else {
        phaseEl.textContent = `Error: ${event.error || "unknown"}`;
        document.getElementById("train-go-btn").disabled = false;
      }
    } else if (t === "error") {
      appendLog(`ERROR: ${event.message}`);
    }
  };

  es.onerror = () => {
    es.close();
    document.getElementById("train-go-btn").disabled = false;
  };
}

// ====== Settings screen ======

let cfgBrowser = null;

// ── Library ──────────────────────────────────────────────────────────────────

async function openLibrary() {
  state.prevScreen = Object.entries(screens).find(([, el]) => !el.hidden)?.[0] || "ingest";
  showScreen("library");
  await loadLibrary();
}

async function loadLibrary() {
  const listEl  = document.getElementById("library-list");
  const emptyEl = document.getElementById("library-empty");
  const bannerEl = document.getElementById("library-missing-banner");
  listEl.innerHTML = "";
  try {
    const { models, default: defaultName } = await api("/api/library");
    bannerEl.hidden = true;
    emptyEl.hidden = models.length > 0;
    listEl.hidden  = models.length === 0;
    models.forEach(m => listEl.appendChild(makeModelCard(m, defaultName)));
  } catch (e) {
    if (e.message && e.message.includes("not configured")) {
      bannerEl.hidden = false;
      emptyEl.hidden = true;
      listEl.hidden = true;
    } else {
      listEl.innerHTML = `<p class="placeholder">Error: ${e.message}</p>`;
    }
  }
}

function makeModelCard(m, defaultName) {
  const card = document.createElement("div");
  card.className = "model-card" + (m.is_default ? " is-default" : "");
  card.dataset.name = m.name;

  const acc  = m.best_acc != null ? `${m.best_acc.toFixed(1)}%` : "—";
  const date = m.trained_at ? m.trained_at.slice(0, 10) : "";

  card.innerHTML = `
    <div class="model-card-top">
      <span class="model-name" title="Double-click to rename">${esc(m.name)}</span>
      <span class="model-acc">${acc}</span>
      <span class="model-date">${date}</span>
      <button class="model-default-btn${m.is_default ? " active" : ""}" title="${m.is_default ? "Default model" : "Set as default"}">★</button>
      <button class="model-delete-btn" title="Delete">✕</button>
    </div>
    ${m.description ? `<div class="model-desc">${esc(m.description)}</div>` : ""}
    <div class="model-notes-row">
      <textarea class="model-notes" placeholder="Notes…" rows="2">${esc(m.notes || "")}</textarea>
      <button class="model-notes-save small" hidden>Save</button>
    </div>`;

  // Rename on double-click
  const nameEl = card.querySelector(".model-name");
  nameEl.addEventListener("dblclick", () => startRename(card, m.name));

  // Default button
  card.querySelector(".model-default-btn").addEventListener("click", async () => {
    await apiPost(`/api/library/${encodeURIComponent(m.name)}/default`, {});
    await loadLibrary();
    await refreshProfiles();
  });

  // Delete button
  card.querySelector(".model-delete-btn").addEventListener("click", async () => {
    if (!confirm(`Delete model "${m.name}"? This cannot be undone.`)) return;
    await apiDelete(`/api/library/${encodeURIComponent(m.name)}`);
    await loadLibrary();
    await refreshProfiles();
  });

  // Notes save
  const notesEl = card.querySelector(".model-notes");
  const saveBtn = card.querySelector(".model-notes-save");
  notesEl.addEventListener("input", () => { saveBtn.hidden = false; });
  saveBtn.addEventListener("click", async () => {
    await apiPatch(`/api/library/${encodeURIComponent(m.name)}`, { notes: notesEl.value });
    saveBtn.hidden = true;
    saveBtn.textContent = "Saved";
    setTimeout(() => { saveBtn.textContent = "Save"; }, 1500);
  });

  return card;
}

function startRename(card, oldName) {
  const nameEl = card.querySelector(".model-name");
  const orig = nameEl.textContent;
  nameEl.contentEditable = "true";
  nameEl.focus();
  const range = document.createRange();
  range.selectNodeContents(nameEl);
  window.getSelection().removeAllRanges();
  window.getSelection().addRange(range);

  async function commit() {
    nameEl.contentEditable = "false";
    const newName = nameEl.textContent.trim();
    if (!newName || newName === oldName) { nameEl.textContent = orig; return; }
    try {
      await apiPost(`/api/library/${encodeURIComponent(oldName)}/rename`, { new_name: newName });
      await loadLibrary();
      await refreshProfiles();
    } catch (e) {
      nameEl.textContent = orig;
      flashToast(e.message || "Rename failed");
    }
  }
  nameEl.addEventListener("blur", commit, { once: true });
  nameEl.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); nameEl.blur(); }
    if (e.key === "Escape") { nameEl.textContent = orig; nameEl.contentEditable = "false"; }
  }, { once: true });
}

// Generic DELETE and PATCH wrappers (api/apiPost already exist)
async function apiPatch(path, body) {
  const r = await fetch(path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
  return r.json();
}
async function apiDelete(path) {
  const r = await fetch(path, { method: "DELETE" });
  if (!r.ok) { const e = await r.json().catch(() => ({})); throw new Error(e.detail || r.statusText); }
  return r.json();
}

function esc(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

// ─────────────────────────────────────────────────────────────────────────────

async function openSettings() {
  state.prevScreen = Object.entries(screens).find(([, el]) => !el.hidden)?.[0] || "ingest";
  showScreen("settings");
  await loadConfig();
}

async function loadConfig() {
  try {
    const data = await api("/api/config");
    const fields = data.fields || {};
    document.getElementById("cfg-model-library").value  = fields.model_library?.value  || "";
    document.getElementById("cfg-dataset-scratch").value = fields.dataset_scratch?.value || "";
    const missing = data.missing_required || [];
    document.getElementById("config-missing-banner").hidden = missing.length === 0;
  } catch (e) {
    document.getElementById("cfg-save-status").textContent = `Load error: ${e.message}`;
  }
}

async function saveConfig() {
  const statusEl = document.getElementById("cfg-save-status");
  statusEl.textContent = "Saving…";
  statusEl.className = "cfg-save-status";
  try {
    const updates = {
      model_library:   document.getElementById("cfg-model-library").value.trim(),
      dataset_scratch: document.getElementById("cfg-dataset-scratch").value.trim(),
    };
    const data = await apiPost("/api/config", { updates });
    const missing = data.missing_required || [];
    document.getElementById("config-missing-banner").hidden = missing.length === 0;
    statusEl.textContent = "Saved.";
    statusEl.className = "cfg-save-status ok";
    // Refresh profile list in case model_library changed
    await refreshProfiles();
    setTimeout(() => { statusEl.textContent = ""; }, 2000);
  } catch (e) {
    statusEl.textContent = `Error: ${e.message}`;
    statusEl.className = "cfg-save-status err";
  }
}

// Reload profile dropdown (called after config save)
async function refreshProfiles() {
  try {
    const { profiles } = await api("/api/profiles");
    const select = document.getElementById("profile");
    const cur = select.value;
    select.innerHTML = "";
    if (profiles.length === 0) {
      const opt = document.createElement("option");
      opt.value = ""; opt.textContent = "(no models in library)";
      select.appendChild(opt);
    } else {
      for (const p of profiles) {
        const opt = document.createElement("option");
        opt.value = p.name === "(default)" ? "" : p.name;
        opt.textContent = p.description ? `${p.name} — ${p.description}` : p.name;
        select.appendChild(opt);
      }
    }
    // Restore previous selection; fall back to default profile, then first
    if ([...select.options].some(o => o.value === cur)) {
      select.value = cur;
    } else {
      const def = profiles.find(p => p.is_default);
      if (def) select.value = def.name === "(default)" ? "" : def.name;
    }
  } catch { /* ignore */ }
}

// Check config on startup and show banner on run screen if misconfigured
async function checkConfigBanner() {
  try {
    const data = await api("/api/config");
    const missing = data.missing_required || [];
    let banner = document.getElementById("run-config-banner");
    if (!banner) {
      banner = document.createElement("div");
      banner.id = "run-config-banner";
      banner.className = "config-banner";
      banner.innerHTML = `Paths not configured. <button type="button" onclick="openSettings()">Open Settings →</button>`;
      document.getElementById("screen-run").prepend(banner);
    }
    banner.hidden = missing.length === 0;
  } catch { /* ignore */ }
}

// ====== Ingest screen ======

let localBrowser = null;
let destBrowser  = null;

const ingestState = {
  selectedCards: new Set(),   // card paths currently checked
  forcedCards:   new Set(),   // card paths with "re-ingest" checked
  localSources:  [],          // [{path, label}] added via local folder picker
};

function getRecentDests() {
  try { return JSON.parse(localStorage.getItem("robo.recentDests") || "[]"); }
  catch { return []; }
}
function addRecentDest(path) {
  const list = getRecentDests().filter(p => p !== path);
  list.unshift(path);
  localStorage.setItem("robo.recentDests", JSON.stringify(list.slice(0, 5)));
}

async function initIngest() {
  ingestState.selectedCards.clear();
  ingestState.forcedCards.clear();
  ingestState.localSources = [];
  await loadCards();
  renderRecentDests();
  // Pre-populate dest with most recently used path
  const recents = getRecentDests();
  if (recents.length > 0 && !document.getElementById("ingest-dest").value.trim()) {
    document.getElementById("ingest-dest").value = recents[0];
  }
  updateIngestGoBtn();
}

async function loadCards() {
  const listEl = document.getElementById("cards-list");
  listEl.innerHTML = "<p class='placeholder'>Scanning for cards…</p>";
  try {
    const { cards } = await api("/api/ingest/cards");
    renderCards(cards);
  } catch (e) {
    listEl.innerHTML = `<p class='placeholder'>Error: ${e.message}</p>`;
  }
}

function renderCards(cards) {
  const listEl = document.getElementById("cards-list");
  listEl.innerHTML = "";

  if (cards.length === 0) {
    listEl.innerHTML = "<p class='placeholder'>No camera cards detected. Insert a card or use + Add local folder below.</p>";
  }

  for (const card of cards) {
    const row = document.createElement("div");
    row.className = "card-row";
    row.innerHTML = `
      <label class="card-label">
        <input type="checkbox" class="card-check" />
        <span class="card-name">${card.name}</span>
        <span class="card-count">${card.file_count.toLocaleString()} files</span>
      </label>
      <label class="force-label" title="Re-copy files already in the manifest">
        <input type="checkbox" class="card-force" />
        re-ingest
      </label>
    `;
    const check = row.querySelector(".card-check");
    const force = row.querySelector(".card-force");
    check.addEventListener("change", () => {
      if (check.checked) ingestState.selectedCards.add(card.path);
      else { ingestState.selectedCards.delete(card.path); ingestState.forcedCards.delete(card.path); }
      force.disabled = !check.checked;
      updateIngestGoBtn();
    });
    force.disabled = true;
    force.addEventListener("change", () => {
      if (force.checked) ingestState.forcedCards.add(card.path);
      else ingestState.forcedCards.delete(card.path);
    });
    listEl.appendChild(row);
  }
}

function renderLocalSources() {
  // Re-render the local-sources list below the cards list.
  let localEl = document.getElementById("local-sources-list");
  if (!localEl) {
    localEl = document.createElement("div");
    localEl.id = "local-sources-list";
    document.getElementById("cards-list").after(localEl);
  }
  localEl.innerHTML = "";
  for (const [i, src] of ingestState.localSources.entries()) {
    const row = document.createElement("div");
    row.className = "card-row local-source-row";
    row.innerHTML = `
      <span class="card-name">📂 ${src.label}</span>
      <button type="button" class="remove-local small" data-idx="${i}">✕</button>
    `;
    row.querySelector(".remove-local").addEventListener("click", () => {
      ingestState.localSources.splice(i, 1);
      renderLocalSources();
      updateIngestGoBtn();
    });
    localEl.appendChild(row);
  }
}

function renderRecentDests() {
  const el = document.getElementById("ingest-recent");
  const recent = getRecentDests();
  el.innerHTML = "";
  if (recent.length === 0) return;
  const label = document.createElement("div");
  label.className = "recent-label";
  label.textContent = "Recent:";
  el.appendChild(label);
  for (const path of recent) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "recent-item";
    btn.textContent = path;
    btn.title = path;
    btn.addEventListener("click", () => {
      document.getElementById("ingest-dest").value = path;
      ingestState.destDir = path;
      updateIngestGoBtn();
    });
    el.appendChild(btn);
  }
}

function updateIngestGoBtn() {
  const hasSources = ingestState.selectedCards.size > 0 || ingestState.localSources.length > 0;
  const hasDest = !!document.getElementById("ingest-dest").value.trim();
  document.getElementById("ingest-go-btn").disabled = !(hasSources && hasDest);

  const total = ingestState.selectedCards.size + ingestState.localSources.length;
  document.getElementById("ingest-source-summary").textContent =
    total > 0 ? `${total} source${total === 1 ? "" : "s"} selected` : "";
}

async function startIngest() {
  const dest = document.getElementById("ingest-dest").value.trim();
  if (!dest) return;

  const sources = [];
  for (const path of ingestState.selectedCards) {
    sources.push({
      path,
      label: path.split("/").pop() || path,
      force: ingestState.forcedCards.has(path),
    });
  }
  for (const src of ingestState.localSources) {
    sources.push({ path: src.path, label: src.label, force: false });
  }
  if (sources.length === 0) return;

  addRecentDest(dest);
  ingestState.destDir = dest;

  document.getElementById("ingest-go-btn").disabled = true;
  document.getElementById("ingest-progress-box").hidden = false;
  document.getElementById("ingest-done-box").hidden = true;
  document.getElementById("ingest-log").textContent = "";
  document.getElementById("ingest-progress-bar").value = 0;
  document.getElementById("ingest-progress-stage").textContent = "Starting…";

  try {
    const { job_id } = await apiPost("/api/ingest/start", { sources, dest_dir: dest });
    streamIngestProgress(job_id, dest);
  } catch (e) {
    document.getElementById("ingest-progress-stage").textContent = `Error: ${e.message}`;
    document.getElementById("ingest-go-btn").disabled = false;
  }
}

function streamIngestProgress(jobId, destDir) {
  const es = new EventSource(`/api/ingest/progress/${jobId}`);
  const stageEl = document.getElementById("ingest-progress-stage");
  const barEl   = document.getElementById("ingest-progress-bar");
  const logEl   = document.getElementById("ingest-log");

  const LOG_CAP = 300;
  let logLines = [];
  let logTimer = null;

  function flushLog() {
    logTimer = null;
    if (logLines.length > LOG_CAP) logLines = logLines.slice(-LOG_CAP);
    logEl.textContent = logLines.join("\n");
    logEl.scrollTop = logEl.scrollHeight;
  }
  function appendLog(line) {
    logLines.push(line);
    if (!logTimer) logTimer = setTimeout(flushLog, 150);
  }

  let skippedCount = 0;

  es.onmessage = (ev) => {
    let event;
    try { event = JSON.parse(ev.data); } catch { return; }
    const t = event.type;

    if (t === "scan_start") {
      stageEl.textContent = `Scanning ${event.source}…`;
    } else if (t === "scan_done") {
      appendLog(`Scanned ${event.source}: ${event.count.toLocaleString()} files`);
    } else if (t === "exif_start") {
      stageEl.textContent = `Reading EXIF from ${event.total.toLocaleString()} files…`;
    } else if (t === "progress" && event.stage === "copy") {
      stageEl.textContent = `Copying ${event.done.toLocaleString()} / ${event.total.toLocaleString()}`;
      barEl.max   = event.total;
      barEl.value = event.done;
    } else if (t === "file_result") {
      if (event.status === "copied") {
        appendLog(`→ ${event.dest}`);
      } else if (event.status === "skipped") {
        skippedCount++;
        if (skippedCount <= 3) appendLog(`  skip ${event.filename} (dup)`);
        else if (skippedCount === 4) appendLog(`  … (more skipped)`);
      } else if (event.status === "error") {
        appendLog(`ERROR ${event.filename}: ${event.reason}`);
      }
    } else if (t === "status") {
      es.close();
      if (logTimer) { clearTimeout(logTimer); flushLog(); }
      if (event.status === "done" && event.summary) {
        const s = event.summary;
        barEl.value = barEl.max;
        stageEl.textContent =
          `Done — ${s.copied.toLocaleString()} copied` +
          (s.skipped ? `, ${s.skipped.toLocaleString()} skipped` : "") +
          (s.errors  ? `, ${s.errors} errors` : "");
        showIngestDone(s, destDir);
      } else {
        stageEl.textContent = `Error: ${event.error || "unknown"}`;
        document.getElementById("ingest-go-btn").disabled = false;
      }
    }
  };

  es.onerror = () => {
    es.close();
    document.getElementById("ingest-go-btn").disabled = false;
  };
}

function showIngestDone(summary, destDir) {
  document.getElementById("ingest-n-copied").textContent  = summary.copied.toLocaleString();
  document.getElementById("ingest-n-skipped").textContent = summary.skipped.toLocaleString();
  document.getElementById("ingest-n-errors").textContent  = summary.errors;
  document.getElementById("ingest-done-box").hidden = false;

  document.getElementById("ingest-run-pipeline-btn").onclick = () => {
    document.getElementById("input_dir").value = destDir;
    localStorage.setItem("robo.lastInputDir", destDir);
    state.inputDir = destDir;
    showScreen("run");
  };
}

init();

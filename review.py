#!/usr/bin/env python3
"""
review.py — Interactive threshold browser for robo-classifier

Loads results.csv, opens a local web UI to tune thresholds visually,
then writes keywords when ready.

Usage:
    python review.py results.csv [--nef_dir PATH] [--port 5000]
    python review.py results.csv --high 0.93 --low 0.82

Auto-detects workflow:
  NEF-heavy  → extracts embedded previews from NEFs for thumbnails (background)
  JPG-heavy  → looks for a parallel small-JPG directory for faster thumbnails
"""

import argparse
import csv
import hashlib
import io
import subprocess
import sys
import tempfile
import threading
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path

try:
    from flask import Flask, request, jsonify, send_file, Response
except ImportError:
    print("Flask required: pip install flask")
    sys.exit(1)

try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Import utilities from classify.py
sys.path.insert(0, str(Path(__file__).parent))
try:
    from classify import write_keywords, parse_burst_base, get_tier_keyword, RAW_EXTENSIONS
except ImportError as e:
    print(f"Error importing from classify.py: {e}")
    sys.exit(1)

THUMB_W, THUMB_H = 280, 190

# Directory names that suggest small/preview JPGs (Vic's second-card workflow)
SMALL_DIR_KEYWORDS = {"small", "sraw", "basic", "low", "thumb", "thumbs",
                      "thumbnail", "thumbnails", "preview", "previews",
                      "web", "reduced", "mini", "s-raw", "jpg_small"}

app = Flask(__name__)
_state: dict = {}


# ─── Data Loading ──────────────────────────────────────────────────────────────

def resolve_path(path_str: str, csv_dir: Path) -> Path:
    """
    Resolve a path from results.csv. Tries absolute first, then relative to
    the CSV file's directory, then relative to cwd.
    """
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return p
    rel_to_csv = csv_dir / p
    if rel_to_csv.exists():
        return rel_to_csv
    return p  # return as-is; thumbnail serving will just 404


def load_results(csv_path: str) -> list[dict]:
    csv_dir = Path(csv_path).parent.resolve()
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            row["confidence_select"] = float(row["confidence_select"])
            row["confidence_reject"] = float(row.get("confidence_reject", 0.0))
            # Resolve path so thumbnail serving works regardless of where we're run from
            row["path"] = str(resolve_path(row["path"], csv_dir))
            rows.append(row)
    return rows


def compute_bursts_and_winners(results: list[dict]) -> tuple[list[dict], dict]:
    """
    Group all results by burst (filename-based), find best frame per burst.
    No threshold filter — that happens at display/write time.
    """
    bursts: dict[str, list] = defaultdict(list)
    for r in results:
        base = parse_burst_base(r["filename"])
        bursts[base].append(r)

    winners = []
    for frames in bursts.values():
        best = max(frames, key=lambda x: x["confidence_select"])
        winners.append(best)

    winners.sort(key=lambda x: x["confidence_select"], reverse=True)
    return winners, dict(bursts)


# ─── Workflow Auto-Detection ───────────────────────────────────────────────────

def detect_workflow(results: list[dict]) -> dict:
    """
    Inspect results to determine workflow type and useful paths.

    Returns dict with:
        workflow      'nef' | 'jpg' | 'mixed'
        raw_count     int
        jpg_count     int
        nef_dir       Path | None  – inferred from RAW file locations
        small_jpg_dir Path | None  – parallel small-JPG dir (Vic's workflow)
    """
    paths = [Path(r["path"]) for r in results]
    raw_exts = RAW_EXTENSIONS | {e.upper() for e in RAW_EXTENSIONS}
    raw_paths = [p for p in paths if p.suffix.lower() in RAW_EXTENSIONS]
    jpg_paths = [p for p in paths if p.suffix.lower() not in RAW_EXTENSIONS]
    raw_count, jpg_count = len(raw_paths), len(jpg_paths)

    if raw_count >= jpg_count * 3:
        workflow = "nef"
    elif jpg_count >= raw_count * 3:
        workflow = "jpg"
    else:
        workflow = "mixed"

    # Auto-detect NEF directory: most common parent of RAW files
    nef_dir = None
    if raw_paths:
        counts = Counter(p.parent for p in raw_paths if p.exists())
        if counts:
            nef_dir = counts.most_common(1)[0][0]

    # Auto-detect small-JPG directory (Vic's workflow)
    small_jpg_dir = None
    if jpg_paths:
        jpg_dirs = {p.parent for p in jpg_paths if p.exists()}
        # Collect all stems from the main JPG set for matching
        main_stems = {p.stem.lower() for p in jpg_paths}

        for jpg_dir in jpg_dirs:
            candidate = _find_small_jpg_dir(jpg_dir, main_stems)
            if candidate:
                small_jpg_dir = candidate
                break

    return {
        "workflow": workflow,
        "raw_count": raw_count,
        "jpg_count": jpg_count,
        "nef_dir": nef_dir,
        "small_jpg_dir": small_jpg_dir,
    }


def _find_small_jpg_dir(jpg_dir: Path, main_stems: set[str]) -> Path | None:
    """
    Search for a sibling or child directory containing small JPGs whose
    filenames match main_stems. Returns the first strong match or None.
    """
    search_dirs = []

    # Siblings of the JPG directory
    try:
        search_dirs += [d for d in jpg_dir.parent.iterdir() if d.is_dir() and d != jpg_dir]
    except PermissionError:
        pass

    # Immediate children of the JPG directory
    try:
        search_dirs += [d for d in jpg_dir.iterdir() if d.is_dir()]
    except PermissionError:
        pass

    best: tuple[int, Path] | None = None

    for candidate in search_dirs:
        name_lower = candidate.name.lower()
        name_keywords_match = any(kw in name_lower for kw in SMALL_DIR_KEYWORDS)

        # Collect JPG stems in this candidate directory
        try:
            candidate_stems = {
                p.stem.lower()
                for p in candidate.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg"}
            }
        except PermissionError:
            continue

        if not candidate_stems:
            continue

        overlap = len(main_stems & candidate_stems)
        if overlap == 0:
            continue

        # Score: overlap count + bonus for matching directory name
        score = overlap + (1000 if name_keywords_match else 0)

        if best is None or score > best[0]:
            best = (score, candidate)

    # Only accept if at least 10% of main images have a match
    if best and best[0] > 0:
        overlap_count = best[0] % 1000 if best[0] >= 1000 else best[0]
        if overlap_count >= max(1, len(main_stems) * 0.10):
            return best[1]

    return None


def build_small_jpg_index(small_jpg_dir: Path) -> dict[str, Path]:
    """Build lowercase-stem → Path index for a small-JPG directory."""
    index: dict[str, Path] = {}
    for p in small_jpg_dir.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg"}:
            index[p.stem.lower()] = p
    return index


# ─── NEF Preview Pre-Extraction ───────────────────────────────────────────────

def preextract_nef_previews(raw_paths: list[Path], thumb_dir: str) -> None:
    """
    Background thread: pre-extract embedded previews from all RAW files
    so that the first page load doesn't stall waiting for exiftool.
    """
    thumb_dir_path = Path(thumb_dir)
    pending = []
    for p in raw_paths:
        cache_key = hashlib.md5(str(p).encode()).hexdigest()
        thumb_path = thumb_dir_path / f"{cache_key}.jpg"
        if not thumb_path.exists():
            pending.append(p)

    if not pending:
        return

    print(f"  [background] Pre-extracting {len(pending)} NEF previews…")
    done = 0
    for p in pending:
        try:
            result = subprocess.run(
                ["exiftool", "-b", "-PreviewImage", str(p)],
                capture_output=True, timeout=15
            )
            if result.returncode == 0 and result.stdout:
                cache_key = hashlib.md5(str(p).encode()).hexdigest()
                thumb_path = thumb_dir_path / f"{cache_key}.jpg"
                if HAS_PIL:
                    img = PILImage.open(io.BytesIO(result.stdout)).convert("RGB")
                    img.thumbnail((THUMB_W, THUMB_H), PILImage.LANCZOS)
                    img.save(thumb_path, "JPEG", quality=80)
                else:
                    thumb_path.write_bytes(result.stdout)
                done += 1
        except Exception:
            pass

    print(f"  [background] NEF preview extraction done ({done}/{len(pending)})")


# ─── Thumbnail Serving ─────────────────────────────────────────────────────────

def get_or_create_thumb(file_path: str) -> Path | None:
    """
    Return path to a resized thumbnail JPEG.
    Resolution order:
      1. Cached thumbnail (already generated this session)
      2. Small-JPG index match (Vic's workflow)
      3. Exiftool preview extraction (NEF workflow)
      4. Direct PIL resize of original JPG
    """
    thumb_dir = Path(_state["thumb_dir"])
    cache_key = hashlib.md5(file_path.encode()).hexdigest()
    thumb_path = thumb_dir / f"{cache_key}.jpg"

    if thumb_path.exists():
        return thumb_path

    source = Path(file_path)

    # ── 1. Check small-JPG index (fast path for Vic's workflow) ──────────────
    small_index: dict[str, Path] = _state.get("small_jpg_index", {})
    if small_index:
        small_match = small_index.get(source.stem.lower())
        if small_match and small_match.exists():
            return _make_thumb(small_match, thumb_path)

    # ── 2. RAW files: extract embedded preview ────────────────────────────────
    if source.suffix.lower() in RAW_EXTENSIONS:
        try:
            result = subprocess.run(
                ["exiftool", "-b", "-PreviewImage", str(source)],
                capture_output=True, timeout=15,
            )
            if not (result.returncode == 0 and result.stdout):
                return None
            if HAS_PIL:
                img = PILImage.open(io.BytesIO(result.stdout)).convert("RGB")
                img.thumbnail((THUMB_W, THUMB_H), PILImage.LANCZOS)
                img.save(thumb_path, "JPEG", quality=80)
                return thumb_path
            else:
                thumb_path.write_bytes(result.stdout)
                return thumb_path
        except Exception:
            return None

    # ── 3. Regular JPG / PNG ──────────────────────────────────────────────────
    if not source.exists():
        return None
    return _make_thumb(source, thumb_path)


def _make_thumb(source: Path, dest: Path) -> Path | None:
    """Resize source image to thumb size and save to dest. Returns dest or None."""
    try:
        if HAS_PIL:
            img = PILImage.open(source).convert("RGB")
            img.thumbnail((THUMB_W, THUMB_H), PILImage.LANCZOS)
            img.save(dest, "JPEG", quality=80)
            return dest
        else:
            import shutil
            shutil.copy(source, dest)
            return dest
    except Exception:
        return None


# ─── Flask Routes ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


@app.route("/thumb")
def thumb():
    path = request.args.get("path", "")
    if not path:
        return Response(status=400)
    p = get_or_create_thumb(path)
    if p is None:
        return Response(status=404)
    return send_file(str(p), mimetype="image/jpeg")


@app.route("/api/data")
def api_data():
    high = float(request.args.get("high", 0.95))
    low = float(request.args.get("low", 0.85))
    show_all = request.args.get("show_all", "false") == "true"
    page = int(request.args.get("page", 0))
    per_page = int(request.args.get("per_page", 100))

    all_winners = _state["burst_winners"]

    # Filter for display
    visible = all_winners if show_all else [
        w for w in all_winners if w["confidence_select"] >= low
    ]

    # Summary stats
    n_auto   = sum(1 for w in all_winners if w["confidence_select"] >= high)
    n_review = sum(1 for w in all_winners if low <= w["confidence_select"] < high)
    n_below  = sum(1 for w in all_winners if w["confidence_select"] < low)

    # Tier counts for frames above low threshold
    tier_counts: dict[str, int] = {}
    for w in all_winners:
        c = w["confidence_select"]
        if c >= low:
            kw = get_tier_keyword(c)
            if kw:
                tier_counts[kw] = tier_counts.get(kw, 0) + 1

    # Histogram: 50 bins from 0.50 to 1.00
    bins = [0] * 50
    for w in all_winners:
        c = w["confidence_select"]
        if 0.50 <= c <= 1.0:
            bins[min(49, int((c - 0.50) * 100))] += 1

    # Paginate
    total = len(visible)
    page_items = visible[page * per_page : (page + 1) * per_page]

    items = []
    for r in page_items:
        c = r["confidence_select"]
        items.append({
            "path": r["path"],
            "filename": r["filename"],
            "confidence": round(c, 4),
            "tier": get_tier_keyword(c),
            "zone": "auto" if c >= high else ("review" if c >= low else "below"),
        })

    return jsonify({
        "stats": {
            "total_bursts": len(all_winners),
            "n_auto": n_auto,
            "n_review": n_review,
            "n_below": n_below,
            "tier_counts": tier_counts,
        },
        "histogram": {"bins": bins, "bin_start": 0.50, "bin_width": 0.01},
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": max(1, (total + per_page - 1) // per_page),
        },
        "items": items,
        "workflow_info": _state.get("workflow_info", {}),
    })


@app.route("/api/write_keywords", methods=["POST"])
def api_write_keywords():
    data = request.get_json()
    low     = float(data.get("low", 0.85))
    dry_run = bool(data.get("dry_run", False))

    all_winners = _state["burst_winners"]
    bursts      = _state["bursts"]
    nef_dir     = _state["nef_dir"]

    selected = [w for w in all_winners if w["confidence_select"] >= low]
    if not selected:
        return jsonify({"ok": False, "error": "No images above threshold"}), 400

    try:
        tier_counts, winner_written, select_written, errors = write_keywords(
            selected, bursts, nef_dir
        )
        prefix = "(dry run) " if dry_run else ""
        print(f"{prefix}Wrote {winner_written} robo_9x keywords, "
              f"{select_written} select keywords ({errors} errors)")
        return jsonify({
            "ok": True,
            "n_tagged": winner_written,
            "n_select": select_written,
            "errors": errors,
            "tier_counts": {k: v for k, v in tier_counts.items() if v > 0},
        })
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ─── HTML / JS Frontend ────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Robo-Classifier Review</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #111;
    color: #ddd;
    font-family: system-ui, sans-serif;
    font-size: 13px;
  }

  /* ── Controls header ── */
  #header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: #1c1c1c;
    border-bottom: 1px solid #333;
    padding: 10px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .ctrl-row {
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
  }

  .slider-group {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .slider-group label {
    white-space: nowrap;
    color: #aaa;
    min-width: 155px;
  }
  .slider-group input[type=range] {
    width: 220px;
    accent-color: #888;
    cursor: pointer;
  }
  .slider-val {
    font-weight: 600;
    min-width: 36px;
    font-variant-numeric: tabular-nums;
  }
  #high-val { color: #f55; }
  #low-val  { color: #55f; }

  .stats-row {
    display: flex;
    gap: 20px;
    align-items: center;
    flex-wrap: wrap;
  }
  .stat { color: #aaa; }
  .stat strong { color: #eee; }
  .stat.auto strong   { color: #f55; }
  .stat.review strong { color: #fa0; }
  .stat.below strong  { color: #559; }

  .tier-summary {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
    align-items: center;
    color: #777;
    font-size: 11px;
  }
  .tier-chip {
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
  }

  .btn-row {
    display: flex;
    gap: 8px;
    align-items: center;
  }
  button {
    padding: 5px 14px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
    font-size: 12px;
    font-weight: 600;
    transition: opacity 0.15s;
  }
  button:hover { opacity: 0.85; }
  button:active { opacity: 0.7; }
  #toggle-btn  { background: #2a3a4a; color: #8af; }
  #toggle-btn.active { background: #1a3a5a; color: #5af; }
  #write-btn   { background: #2d5a1e; color: #8f8; }
  #dry-run-btn { background: #2a2a1e; color: #aa8; }
  #status-msg  { color: #888; font-size: 11px; }

  #workflow-badge {
    font-size: 10px;
    padding: 2px 7px;
    border-radius: 10px;
    background: #252525;
    color: #888;
    border: 1px solid #333;
  }

  /* ── Histogram ── */
  #histogram-wrap {
    background: #161616;
    border-bottom: 1px solid #2a2a2a;
    padding: 8px 16px 4px;
    user-select: none;
  }
  #histogram-wrap canvas {
    display: block;
    width: 100%;
    height: 70px;
  }
  .hist-axis {
    display: flex;
    justify-content: space-between;
    color: #555;
    font-size: 10px;
    margin-top: 1px;
    padding: 0 1px;
  }

  /* ── Image Grid ── */
  #grid-wrap { padding: 12px 14px; }

  #grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 8px;
  }

  .thumb {
    position: relative;
    border-radius: 3px;
    overflow: hidden;
    border: 4px solid transparent;
    cursor: default;
    transition: border-color 0.2s;
    background: #222;
  }
  .thumb:hover { filter: brightness(1.1); }

  .thumb img {
    display: block;
    width: 100%;
    aspect-ratio: 3/2;
    object-fit: cover;
    background: #1a1a1a;
  }

  .thumb-info {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    padding: 3px 6px;
    background: rgba(0,0,0,0.65);
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
  }
  .conf-val { font-weight: 700; }
  .tier-badge {
    font-size: 10px;
    padding: 1px 4px;
    border-radius: 2px;
    background: rgba(255,255,255,0.12);
  }

  /* ── Pagination ── */
  #pagination {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 10px;
    padding: 14px;
    color: #777;
  }
  #pagination button { background: #252525; color: #aaa; padding: 4px 12px; }
  #pagination button:disabled { opacity: 0.3; cursor: default; }
  #page-info { font-size: 12px; }

  #loading {
    text-align: center;
    padding: 40px;
    color: #555;
    display: none;
  }
</style>
</head>
<body>

<div id="header">
  <div class="ctrl-row">
    <div class="slider-group">
      <label>Auto-select threshold</label>
      <input type="range" id="high-slider" min="0.50" max="1.00" step="0.01" value="0.95">
      <span class="slider-val" id="high-val">0.95</span>
    </div>
    <div class="slider-group">
      <label>Review threshold (lower cutoff)</label>
      <input type="range" id="low-slider" min="0.50" max="1.00" step="0.01" value="0.85">
      <span class="slider-val" id="low-val">0.85</span>
    </div>
    <div class="btn-row">
      <span id="workflow-badge">—</span>
      <button id="toggle-btn">Show All Bursts</button>
      <button id="dry-run-btn">Dry Run</button>
      <button id="write-btn">Write Keywords</button>
      <span id="status-msg"></span>
    </div>
  </div>
  <div class="stats-row">
    <div class="stat auto">Auto: <strong id="st-auto">—</strong></div>
    <div class="stat review">Review: <strong id="st-review">—</strong></div>
    <div class="stat below">Below: <strong id="st-below">—</strong></div>
    <div class="stat">Total bursts: <strong id="st-total">—</strong></div>
    <div class="tier-summary" id="tier-summary"></div>
  </div>
</div>

<div id="histogram-wrap">
  <canvas id="histogram"></canvas>
  <div class="hist-axis">
    <span>0.50</span><span>0.55</span><span>0.60</span><span>0.65</span>
    <span>0.70</span><span>0.75</span><span>0.80</span><span>0.85</span>
    <span>0.90</span><span>0.95</span><span>1.00</span>
  </div>
</div>

<div id="grid-wrap">
  <div id="loading">Loading…</div>
  <div id="grid"></div>
</div>

<div id="pagination">
  <button id="prev-btn" disabled>‹ Prev</button>
  <span id="page-info"></span>
  <button id="next-btn" disabled>Next ›</button>
</div>

<script>
// ── State ──────────────────────────────────────────────────────────────────
const params = new URLSearchParams(window.location.search);
const state = {
  high: parseFloat(params.get('high') || '0.95'),
  low:  parseFloat(params.get('low')  || '0.85'),
  showAll: false,
  page: 0,
  perPage: 100,
  totalPages: 1,
  histBins: [],
};

// ── Color helpers ──────────────────────────────────────────────────────────
function confColor(conf, low, high) {
  if (conf < low) return 'hsl(230,25%,30%)'; // dim blue-gray for below cutoff
  const t = (conf - low) / (1.0 - low);
  const hue = Math.round(240 * (1 - t));
  const sat = 85;
  const lit = conf >= high ? 48 : 58;
  return `hsl(${hue},${sat}%,${lit}%)`;
}

function tierBgColor(conf, low) {
  if (conf < low) return 'rgba(255,255,255,0.1)';
  const t = (conf - low) / (1.0 - low);
  const hue = Math.round(240 * (1 - t));
  return `hsl(${hue},70%,30%)`;
}

// ── Histogram ──────────────────────────────────────────────────────────────
function drawHistogram() {
  const canvas = document.getElementById('histogram');
  const wrap = document.getElementById('histogram-wrap');
  canvas.width = wrap.clientWidth - 32;
  canvas.height = 70;
  const ctx = canvas.getContext('2d');
  const bins = state.histBins;
  if (!bins.length) return;

  const W = canvas.width, H = canvas.height - 4;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const maxVal = Math.max(...bins, 1);
  const nBins = bins.length;
  const barW = W / nBins;

  for (let i = 0; i < nBins; i++) {
    const conf = 0.50 + (i + 0.5) * 0.01;
    const h = Math.ceil((bins[i] / maxVal) * H);
    const x = i * barW;
    const y = H - h + 4;
    ctx.fillStyle = confColor(conf, state.low, state.high);
    ctx.fillRect(x + 0.5, y, Math.max(barW - 1, 1), h);
  }

  function drawLine(threshold, color, label) {
    const x = Math.round(((threshold - 0.50) / 0.50) * W);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = color;
    ctx.font = '10px system-ui';
    const lw = ctx.measureText(label).width;
    const lx = Math.min(x + 3, W - lw - 2);
    ctx.fillText(label, lx, 10);
  }

  drawLine(state.low,  'hsl(230,85%,65%)', `low ${state.low.toFixed(2)}`);
  drawLine(state.high, 'hsl(0,85%,65%)',   `high ${state.high.toFixed(2)}`);
}

// ── Grid rendering ─────────────────────────────────────────────────────────
function renderGrid(items) {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  for (const item of items) {
    const bc = confColor(item.confidence, state.low, state.high);
    const tc = tierBgColor(item.confidence, state.low);
    const div = document.createElement('div');
    div.className = 'thumb';
    div.style.borderColor = bc;

    const img = document.createElement('img');
    img.loading = 'lazy';
    img.src = `/thumb?path=${encodeURIComponent(item.path)}`;
    img.alt = item.filename;

    const info = document.createElement('div');
    info.className = 'thumb-info';

    const confSpan = document.createElement('span');
    confSpan.className = 'conf-val';
    confSpan.textContent = item.confidence.toFixed(3);
    confSpan.style.color = bc;

    const tierSpan = document.createElement('span');
    tierSpan.className = 'tier-badge';
    tierSpan.style.background = tc;
    tierSpan.textContent = item.tier || '—';

    info.appendChild(confSpan);
    info.appendChild(tierSpan);
    div.appendChild(img);
    div.appendChild(info);
    grid.appendChild(div);
  }
}

// ── Stats & workflow badge ─────────────────────────────────────────────────
function updateStats(stats, workflowInfo) {
  document.getElementById('st-auto').textContent   = stats.n_auto;
  document.getElementById('st-review').textContent = stats.n_review;
  document.getElementById('st-below').textContent  = stats.n_below;
  document.getElementById('st-total').textContent  = stats.total_bursts;

  if (workflowInfo && workflowInfo.label) {
    document.getElementById('workflow-badge').textContent = workflowInfo.label;
  }

  const tierWrap = document.getElementById('tier-summary');
  tierWrap.innerHTML = '';
  const tiers = Object.entries(stats.tier_counts || {})
    .sort((a, b) => b[0].localeCompare(a[0]));
  for (const [kw, cnt] of tiers) {
    const conf = parseInt(kw.replace('robo_', '')) / 100;
    const chip = document.createElement('span');
    chip.className = 'tier-chip';
    chip.style.background = tierBgColor(conf + 0.005, state.low);
    chip.textContent = `${kw}: ${cnt}`;
    tierWrap.appendChild(chip);
  }
}

// ── Pagination ─────────────────────────────────────────────────────────────
function updatePagination(pag) {
  const info = document.getElementById('page-info');
  const start = pag.page * pag.per_page + 1;
  const end   = Math.min((pag.page + 1) * pag.per_page, pag.total);
  info.textContent = pag.total ? `${start}–${end} of ${pag.total}` : 'No results';
  document.getElementById('prev-btn').disabled = pag.page <= 0;
  document.getElementById('next-btn').disabled = pag.page >= pag.pages - 1;
  state.totalPages = pag.pages;
}

// ── Main data fetch ────────────────────────────────────────────────────────
let fetchTimer = null;
function scheduleFetch(immediate = false) {
  clearTimeout(fetchTimer);
  fetchTimer = setTimeout(doFetch, immediate ? 0 : 120);
}

async function doFetch() {
  document.getElementById('loading').style.display = 'block';
  document.getElementById('grid').innerHTML = '';

  const qs = new URLSearchParams({
    high:     state.high,
    low:      state.low,
    show_all: state.showAll,
    page:     state.page,
    per_page: state.perPage,
  });

  try {
    const resp = await fetch(`/api/data?${qs}`);
    const data = await resp.json();
    state.histBins = data.histogram.bins;
    drawHistogram();
    updateStats(data.stats, data.workflow_info);
    updatePagination(data.pagination);
    renderGrid(data.items);
  } catch (e) {
    console.error(e);
    setStatus('Fetch error — is the server running?', '#f55');
  } finally {
    document.getElementById('loading').style.display = 'none';
  }
}

// ── Controls ───────────────────────────────────────────────────────────────
const highSlider = document.getElementById('high-slider');
const lowSlider  = document.getElementById('low-slider');
const highVal    = document.getElementById('high-val');
const lowVal     = document.getElementById('low-val');

highSlider.value = state.high;
lowSlider.value  = state.low;
highVal.textContent = state.high.toFixed(2);
lowVal.textContent  = state.low.toFixed(2);

highSlider.addEventListener('input', () => {
  state.high = parseFloat(highSlider.value);
  if (state.high <= state.low) {
    state.low = Math.max(0.50, state.high - 0.01);
    lowSlider.value = state.low;
    lowVal.textContent = state.low.toFixed(2);
  }
  highVal.textContent = state.high.toFixed(2);
  state.page = 0;
  scheduleFetch();
});

lowSlider.addEventListener('input', () => {
  state.low = parseFloat(lowSlider.value);
  if (state.low >= state.high) {
    state.high = Math.min(1.00, state.low + 0.01);
    highSlider.value = state.high;
    highVal.textContent = state.high.toFixed(2);
  }
  lowVal.textContent = state.low.toFixed(2);
  state.page = 0;
  scheduleFetch();
});

document.getElementById('toggle-btn').addEventListener('click', function() {
  state.showAll = !state.showAll;
  state.page = 0;
  this.textContent = state.showAll ? 'Above Review Threshold' : 'Show All Bursts';
  this.classList.toggle('active', state.showAll);
  scheduleFetch(true);
});

document.getElementById('prev-btn').addEventListener('click', () => {
  if (state.page > 0) { state.page--; scheduleFetch(true); }
});
document.getElementById('next-btn').addEventListener('click', () => {
  if (state.page < state.totalPages - 1) { state.page++; scheduleFetch(true); }
});

// ── Keyword Writing ────────────────────────────────────────────────────────
function setStatus(msg, color = '#aaa') {
  const el = document.getElementById('status-msg');
  el.textContent = msg;
  el.style.color = color;
}

async function callWriteKeywords(dry_run) {
  setStatus(dry_run ? 'Running dry run…' : 'Writing keywords…', '#aa8');
  try {
    const resp = await fetch('/api/write_keywords', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ low: state.low, dry_run }),
    });
    const data = await resp.json();
    if (!data.ok) { setStatus(`Error: ${data.error}`, '#f55'); return; }
    const prefix = dry_run ? '[dry run] ' : '';
    const tiers = Object.entries(data.tier_counts || {})
      .sort((a, b) => b[0].localeCompare(a[0]))
      .map(([k, v]) => `${k}:${v}`).join(' ');
    setStatus(
      `${prefix}${data.n_tagged} keywords written, ${data.n_select} select tags${tiers ? ' — ' + tiers : ''}`,
      dry_run ? '#aa8' : '#8f8'
    );
  } catch (e) {
    setStatus(`Network error: ${e}`, '#f55');
  }
}

document.getElementById('write-btn').addEventListener('click', () => {
  const n = document.getElementById('st-auto').textContent;
  const r = document.getElementById('st-review').textContent;
  if (!confirm(
    `Write keywords for ${n} auto + ${r} review images (${parseInt(n)+parseInt(r)} total)?\n` +
    `Low threshold: ${state.low.toFixed(2)}  |  High threshold: ${state.high.toFixed(2)}`
  )) return;
  callWriteKeywords(false);
});

document.getElementById('dry-run-btn').addEventListener('click', () => {
  callWriteKeywords(true);
});

window.addEventListener('resize', drawHistogram);
scheduleFetch(true);
</script>
</body>
</html>
"""


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Interactive threshold browser for robo-classifier results"
    )
    parser.add_argument("results_csv", help="Path to results.csv from classify.py")
    parser.add_argument("--nef_dir",
                        help="NEF directory for XMP sidecar writing (overrides auto-detect)")
    parser.add_argument("--small_jpg_dir",
                        help="Directory of small JPGs for thumbnails (overrides auto-detect)")
    parser.add_argument("--port", type=int, default=5100)
    parser.add_argument("--no_browser", action="store_true")
    parser.add_argument("--high", type=float, default=0.95, metavar="THRESHOLD",
                        help="Initial auto-select threshold (default: 0.95)")
    parser.add_argument("--low", type=float, default=0.85, metavar="THRESHOLD",
                        help="Initial review/lower threshold (default: 0.85)")
    args = parser.parse_args()

    csv_path = Path(args.results_csv)
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    if not HAS_PIL:
        print("Note: Pillow not installed — thumbnails will be full-size (slower). "
              "Install with: pip install pillow")

    print(f"Loading {csv_path}…")
    results = load_results(str(csv_path))
    burst_winners, bursts = compute_bursts_and_winners(results)
    print(f"  {len(results)} images → {len(burst_winners)} burst winners")

    # ── Workflow detection ───────────────────────────────────────────────────
    print("Detecting workflow…")
    wf = detect_workflow(results)

    # CLI overrides take priority
    nef_dir     = Path(args.nef_dir)      if args.nef_dir      else wf["nef_dir"]
    small_dir   = Path(args.small_jpg_dir) if args.small_jpg_dir else wf["small_jpg_dir"]

    # Build human-readable label for UI badge
    wf_parts = []
    if wf["raw_count"]:
        wf_parts.append(f"{wf['raw_count']} NEF")
    if wf["jpg_count"]:
        wf_parts.append(f"{wf['jpg_count']} JPG")
    workflow_label = f"{wf['workflow'].upper()}: {' + '.join(wf_parts)}"

    print(f"  Workflow : {workflow_label}")
    if nef_dir:
        print(f"  NEF dir  : {nef_dir}")
    if small_dir:
        print(f"  Small JPG: {small_dir}  (using for thumbnails)")
    elif wf["workflow"] == "jpg":
        print(f"  Small JPG: not found — will resize originals")

    # Build small JPG index
    small_jpg_index: dict[str, Path] = {}
    if small_dir and small_dir.is_dir():
        small_jpg_index = build_small_jpg_index(small_dir)
        print(f"  Small JPG index: {len(small_jpg_index)} files")

    thumb_dir = tempfile.mkdtemp(prefix="robo_review_")

    _state.update({
        "results": results,
        "burst_winners": burst_winners,
        "bursts": bursts,
        "nef_dir": str(nef_dir) if nef_dir else None,
        "thumb_dir": thumb_dir,
        "small_jpg_index": small_jpg_index,
        "workflow_info": {
            "label": workflow_label,
            "workflow": wf["workflow"],
            "raw_count": wf["raw_count"],
            "jpg_count": wf["jpg_count"],
            "nef_dir": str(wf["nef_dir"]) if wf["nef_dir"] else None,
            "small_jpg_dir": str(wf["small_jpg_dir"]) if wf["small_jpg_dir"] else None,
        },
    })

    # ── Background NEF preview extraction ────────────────────────────────────
    if wf["workflow"] in ("nef", "mixed") and wf["raw_count"] > 0:
        raw_paths = [Path(r["path"]) for r in results
                     if Path(r["path"]).suffix.lower() in RAW_EXTENSIONS]
        t = threading.Thread(
            target=preextract_nef_previews,
            args=(raw_paths, thumb_dir),
            daemon=True,
        )
        t.start()

    url = f"http://localhost:{args.port}/?high={args.high}&low={args.low}"
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()

    print(f"Server: {url}  (Ctrl+C to stop)")
    app.run(port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

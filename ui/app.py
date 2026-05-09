"""
FastAPI entrypoint for the robo-classifier review UI.

Run with:
    python -m ui.app
    python ui/app.py
    uvicorn ui.app:app --reload

Serves the SPA at http://localhost:8765/ by default.
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make sibling modules importable when running via `python ui/app.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_config import config as app_config
from ui import ingest_runner, pipeline_runner, review, thumbs, train_runner
from image_utils import RAW_EXTENSIONS, JPG_EXTENSIONS


app = FastAPI(title="robo-classifier")


STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

class RunRequest(BaseModel):
    input_dir: str
    profile: str | None = None
    model: str = "model.pt"
    preset: str | None = None   # hardware-tuning preset name
    skip_junk_filter: bool = False
    burst_threshold: float | None = None
    no_keywords: bool = False
    dry_run: bool = False


class LabelRequest(BaseModel):
    input_dir: str
    filename: str
    color: str  # "Red" | "Yellow" | "Green" | "Blue" | "Purple" | ""


class CropRequest(BaseModel):
    input_dir: str
    filename: str
    left: float
    top: float
    right: float
    bottom: float
    angle: float = 0.0


class JunkMoveRequest(BaseModel):
    input_dir: str
    filename: str
    junk_dir_name: str = "junk"


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _resolve_input_dir(input_dir: str) -> Path:
    p = Path(input_dir).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(400, f"Not a directory: {p}")
    return p


def _resolve_image(input_dir: Path, filename: str) -> Path:
    # Guard against path escape
    if "/" in filename or ".." in filename or "\\" in filename:
        raise HTTPException(400, "invalid filename")
    p = input_dir / filename
    if not p.exists() or not p.is_file():
        raise HTTPException(404, f"Not found: {filename}")
    return p


def _load_results_csv(input_dir: Path) -> list[dict]:
    """Load classification results CSV written by classify.run_pipeline."""
    csv_path = input_dir / "results.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def _load_winners_csv(input_dir: Path) -> set[str]:
    """Return set of winner paths (as strings) from winners.csv."""
    csv_path = input_dir / "winners.csv"
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {row['path'] for row in csv.DictReader(f)}


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/ls")
def list_dir(path: str = ""):
    """
    List subdirectories of `path` for the UI's folder picker.

    Security: resolves to absolute path; no path traversal beyond what the
    filesystem already enforces. This is a local-only tool (localhost bind),
    so we don't restrict to any specific root — the user can navigate anywhere
    they have read access to.
    """
    if not path:
        p = Path.home()
    else:
        p = Path(path).expanduser()
    try:
        p = p.resolve()
    except Exception:
        raise HTTPException(400, f"bad path: {path}")
    if not p.exists() or not p.is_dir():
        raise HTTPException(404, f"not a directory: {p}")

    dirs = []
    jpg_here = raw_here = 0
    try:
        for entry in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if entry.name.startswith('.'):
                continue  # hide dot-dirs and dot-files
            try:
                if entry.is_dir():
                    dirs.append({"name": entry.name, "path": str(entry)})
                elif entry.is_file():
                    ext = entry.suffix.lower()
                    if ext in RAW_EXTENSIONS:
                        raw_here += 1
                    elif ext in JPG_EXTENSIONS:
                        jpg_here += 1
            except OSError:
                continue  # permission denied etc.
    except PermissionError:
        raise HTTPException(403, f"permission denied: {p}")

    return {
        "path": str(p),
        "parent": str(p.parent) if p.parent != p else None,
        "dirs": dirs,
        "jpg_count": jpg_here,
        "raw_count": raw_here,
    }


@app.get("/api/presets")
def get_presets():
    """List hardware-tuning presets available in presets/."""
    from presets_loader import list_presets
    return {"presets": list_presets()}


@app.get("/api/profiles")
def list_profiles():
    """List available model profiles from the configured model library."""
    # Primary: configured model_library
    search_dirs = []
    lib = app_config.model_library
    if lib and lib.exists():
        search_dirs.append(lib)
    # Fallback for dev: ROOT/models if it exists and library isn't configured
    fallback = ROOT / "models"
    if fallback.exists() and fallback not in search_dirs:
        search_dirs.append(fallback)

    profiles = []
    seen = set()
    for models_dir in search_dirs:
        for pt in sorted(models_dir.glob("*.pt")):
            if pt.stem in seen:
                continue
            seen.add(pt.stem)
            meta = {}
            sidecar = pt.with_suffix(".json")
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text())
                except Exception:
                    pass
            profiles.append({
                "name": pt.stem,
                "path": str(pt),
                "description": meta.get("description", ""),
                "trained_at": meta.get("trained_at", ""),
                "library": str(models_dir),
            })

    # Legacy: bare model.pt in repo root
    root_pt = ROOT / "model.pt"
    if root_pt.exists() and "model" not in seen:
        profiles.insert(0, {
            "name": "(default)",
            "path": str(root_pt),
            "description": "legacy model.pt in repo root",
            "trained_at": "",
            "library": str(ROOT),
        })
    return {"profiles": profiles}


@app.post("/api/run")
def run_pipeline(req: RunRequest):
    input_dir = _resolve_input_dir(req.input_dir)
    options = {
        "model": req.model,
        "profile": req.profile,
        "skip_junk_filter": req.skip_junk_filter,
        "burst_threshold": req.burst_threshold,
        "no_keywords": req.no_keywords,
        "dry_run": req.dry_run,
    }
    # Merge in tuning preset if one was selected (UI has no inline overrides
    # beyond the fields above, so preset values flow straight through).
    if req.preset:
        try:
            from presets_loader import load_preset
            options.update(load_preset(req.preset))
        except FileNotFoundError as e:
            raise HTTPException(400, str(e))
    job = pipeline_runner.MANAGER.create(str(input_dir), options)
    pipeline_runner.MANAGER.start(job)
    return {"job_id": job.id}


@app.get("/api/progress/{job_id}")
async def progress(job_id: str, request: Request):
    job = pipeline_runner.MANAGER.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")

    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = job.events.get(timeout=0.25)
            except Exception:
                # Heartbeat / check status
                if job.status in ("done", "error"):
                    # Drain any straggler then exit
                    while not job.events.empty():
                        ev = job.events.get_nowait()
                        yield f"data: {json.dumps(ev, default=str)}\n\n"
                        if ev.get("type") == "__end__":
                            return
                    # Final status snapshot
                    yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'summary': job.summary, 'error': job.error}, default=str)}\n\n"
                    return
                # Keep connection alive
                yield ": heartbeat\n\n"
                continue
            if event.get("type") == "__end__":
                yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'summary': job.summary, 'error': job.error}, default=str)}\n\n"
                return
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/job/{job_id}")
def job_status(job_id: str):
    job = pipeline_runner.MANAGER.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {
        "id": job.id,
        "status": job.status,
        "summary": job.summary,
        "error": job.error,
    }


@app.get("/api/session")
def session(input_dir: str, only_winners: bool = False):
    """
    Return the reviewable image set for a directory:
      - all images found at the top level (non-recursive)
      - their classification + winner status (if pipeline has run)
      - their current label + crop state from XMP
    """
    d = _resolve_input_dir(input_dir)
    results = _load_results_csv(d)
    winners = _load_winners_csv(d)

    # Build an index of classifier results by filename
    by_name = {r['filename']: r for r in results}

    # Enumerate top-level images
    images = []
    for f in sorted(d.iterdir()):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        if ext not in (RAW_EXTENSIONS | JPG_EXTENSIONS):
            continue
        r = by_name.get(f.name, {})
        is_winner = str(f) in winners
        if only_winners and not is_winner:
            continue

        images.append({
            "filename": f.name,
            "is_raw": ext in RAW_EXTENSIONS,
            "classification": r.get("classification"),
            "confidence_select": float(r['confidence_select']) if r.get('confidence_select') else None,
            "is_winner": is_winner,
            # label + crop are lazy-loaded via /api/state when the detail view
            # opens — reading XMP per image at list time is prohibitively slow
            # on big shoots (one exiftool subprocess per file).
            "label": None,
            "crop": None,
        })

    return {
        "input_dir": str(d),
        "count": len(images),
        "images": images,
        "has_results": bool(results),
    }


@app.get("/api/state")
def image_state(input_dir: str, filename: str):
    """Read label + crop for a single image (XMP sidecar or embedded)."""
    d = _resolve_input_dir(input_dir)
    source = _resolve_image(d, filename)
    return review.read_state(source)


@app.get("/api/thumb")
def thumb(input_dir: str, filename: str, size: int = 512):
    d = _resolve_input_dir(input_dir)
    source = _resolve_image(d, filename)
    try:
        path = thumbs.get_thumb(d, source, size=size)
    except Exception as e:
        raise HTTPException(500, f"thumb error: {e}")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/image")
def full_image(input_dir: str, filename: str):
    d = _resolve_input_dir(input_dir)
    source = _resolve_image(d, filename)
    try:
        path = thumbs.get_full(d, source)
    except Exception as e:
        raise HTTPException(500, f"image error: {e}")
    media = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "application/octet-stream"
    return FileResponse(path, media_type=media)


@app.post("/api/label")
def set_label(req: LabelRequest):
    d = _resolve_input_dir(req.input_dir)
    source = _resolve_image(d, req.filename)
    ok, msg = review.set_label(source, req.color)
    if not ok:
        raise HTTPException(500, msg)
    return {"ok": True}


@app.post("/api/crop")
def set_crop(req: CropRequest):
    d = _resolve_input_dir(req.input_dir)
    source = _resolve_image(d, req.filename)
    ok, msg = review.set_crop(
        source, req.left, req.top, req.right, req.bottom, angle=req.angle
    )
    if not ok:
        raise HTTPException(500, msg)
    return {"ok": True}


@app.post("/api/crop/clear")
def clear_crop(req: LabelRequest):
    """Reuse LabelRequest for the {input_dir, filename} shape."""
    d = _resolve_input_dir(req.input_dir)
    source = _resolve_image(d, req.filename)
    ok, msg = review.clear_crop(source)
    if not ok:
        raise HTTPException(500, msg)
    return {"ok": True}


class WriteKeywordsRequest(BaseModel):
    input_dir: str
    low: float = 0.90  # only winners with confidence_select >= low get tagged
    dry_run: bool = False
    clear_first: bool = True  # clear stale robo_*/select tags before writing
    nef_dir: str | None = None


@app.post("/api/write_keywords")
def write_keywords_endpoint(req: WriteKeywordsRequest):
    """
    Write tier + select keywords to XMP/JPEG based on a user-tuned threshold.
    Reads results.csv for per-image classifications and reconstructs the bursts
    so we can re-tag select-siblings consistently.
    """
    from classify import (
        burst_dedup, clear_robo_keywords, get_capture_times,
        get_tier_keyword, write_keywords as _write_keywords,
    )

    d = _resolve_input_dir(req.input_dir)
    results = _load_results_csv(d)
    if not results:
        raise HTTPException(400, "No results.csv found — run the pipeline first.")

    # Rebuild bursts + winners the same way classify does (filename-based; the
    # time-based grouping only runs when the pipeline is invoked with
    # burst_threshold). That matches what's already in winners.csv.
    for r in results:
        r['confidence_select'] = float(r['confidence_select'])
        r['confidence_reject'] = float(r.get('confidence_reject', 0.0))
    winners_all, bursts = burst_dedup(results)

    selected = [w for w in winners_all if w['confidence_select'] >= req.low]
    if not selected:
        raise HTTPException(400, f"No winners at or above low={req.low:.2f}")

    # Dry run: just compute what WOULD happen.
    if req.dry_run:
        path_to_burst = {
            frame['path']: bk for bk, frames in bursts.items() for frame in frames
        }
        qualifying_bursts = set()
        tier_counts = {f"robo_{i}": 0 for i in range(90, 100)}
        tier_counts["below_threshold"] = 0
        for w in selected:
            kw = get_tier_keyword(w['confidence_select'])
            if kw:
                tier_counts[kw] += 1
                bk = path_to_burst.get(w['path'])
                if bk:
                    qualifying_bursts.add(bk)
            else:
                tier_counts["below_threshold"] += 1
        select_count = sum(len(bursts[b]) for b in qualifying_bursts)
        return {
            "ok": True,
            "dry_run": True,
            "n_tagged": sum(v for k, v in tier_counts.items() if k.startswith("robo_")),
            "n_select": select_count,
            "errors": 0,
            "tier_counts": {k: v for k, v in tier_counts.items() if v > 0},
        }

    # Real write: optionally clear stale keywords from the universe of files
    # we might touch (all winners + all burst siblings of qualifying bursts).
    if req.clear_first:
        to_clear: set[str] = set()
        path_to_burst = {
            frame['path']: bk for bk, frames in bursts.items() for frame in frames
        }
        for w in selected:
            kw = get_tier_keyword(w['confidence_select'])
            if kw is None:
                continue
            to_clear.add(w['path'])
            bk = path_to_burst.get(w['path'])
            if bk:
                for frame in bursts[bk]:
                    to_clear.add(frame['path'])
        for path in to_clear:
            clear_robo_keywords(path, req.nef_dir)

    tier_counts, winner_written, select_written, errors = _write_keywords(
        selected, bursts, req.nef_dir,
    )
    return {
        "ok": True,
        "dry_run": False,
        "n_tagged": winner_written,
        "n_select": select_written,
        "errors": errors,
        "tier_counts": {k: v for k, v in tier_counts.items() if v > 0},
    }


@app.post("/api/junk")
def move_to_junk(req: JunkMoveRequest):
    d = _resolve_input_dir(req.input_dir)
    source = _resolve_image(d, req.filename)
    junk_dir = d / req.junk_dir_name
    junk_dir.mkdir(parents=True, exist_ok=True)
    dest = junk_dir / source.name
    shutil.move(str(source), str(dest))
    # Move companion XMP if present
    xmp = source.with_suffix('.xmp')
    if xmp.exists():
        shutil.move(str(xmp), str(junk_dir / xmp.name))
    return {"ok": True, "moved_to": str(dest)}


# -----------------------------------------------------------------------------
# App config
# -----------------------------------------------------------------------------

class ConfigUpdateRequest(BaseModel):
    updates: dict[str, str]


@app.get("/api/config")
def get_config():
    return app_config.as_dict()


@app.post("/api/config")
def update_config(req: ConfigUpdateRequest):
    app_config.set_many(req.updates)
    return app_config.as_dict()


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

class TrainStartRequest(BaseModel):
    select_dir: str
    reject_dir: str
    model_name: str
    description: str = ""
    test_size: float = 0.2
    epochs: int = 15
    learning_rate: float = 1e-4
    batch_size: int = 32


@app.get("/api/train/data_stats")
def train_data_stats(select_dir: str, reject_dir: str):
    """Count images in select/reject dirs without starting a job."""
    def _count(d: str) -> int:
        p = Path(d).expanduser()
        if not p.is_dir():
            return -1
        exts = {".jpg", ".jpeg", ".JPG", ".JPEG"}
        return sum(1 for f in p.iterdir() if f.is_file() and f.suffix in exts)

    return {
        "select": _count(select_dir),
        "reject": _count(reject_dir),
    }


@app.post("/api/train/start")
def train_start(req: TrainStartRequest):
    lib = app_config.model_library
    if not lib:
        raise HTTPException(400, "model_library not configured — open Settings first")
    scratch = app_config.dataset_scratch
    if not scratch:
        raise HTTPException(400, "dataset_scratch not configured — open Settings first")

    lib.mkdir(parents=True, exist_ok=True)
    model_output = str(lib / f"{req.model_name}.pt")

    job = train_runner.MANAGER.create(
        select_dir=req.select_dir,
        reject_dir=req.reject_dir,
        dataset_dir=str(scratch),
        test_size=req.test_size,
        model_output=model_output,
        model_name=req.model_name,
        description=req.description,
        epochs=req.epochs,
        learning_rate=req.learning_rate,
        batch_size=req.batch_size,
    )
    train_runner.MANAGER.start(job)
    return {"job_id": job.id, "model_output": model_output}


@app.get("/api/train/progress/{job_id}")
async def train_progress(job_id: str, request: Request):
    job = train_runner.MANAGER.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown train job")

    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = job.events.get(timeout=0.25)
            except Exception:
                if job.status in ("done", "error"):
                    while not job.events.empty():
                        ev = job.events.get_nowait()
                        yield f"data: {json.dumps(ev, default=str)}\n\n"
                        if ev.get("type") == "__end__":
                            return
                    yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'summary': job.summary, 'error': job.error}, default=str)}\n\n"
                    return
                yield ": heartbeat\n\n"
                continue
            if event.get("type") == "__end__":
                yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'summary': job.summary, 'error': job.error}, default=str)}\n\n"
                return
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# Perf log
# -----------------------------------------------------------------------------

@app.get("/api/perf/recent")
def perf_recent(n: int = 20):
    from perf import load_recent
    return {"runs": load_recent(n)}


# -----------------------------------------------------------------------------
# Ingest
# -----------------------------------------------------------------------------

class IngestStartRequest(BaseModel):
    sources: list[dict]   # [{path: str, label: str, force: bool}]
    dest_dir: str


@app.get("/api/ingest/cards")
def ingest_cards():
    """Return mounted volumes that contain a DCIM/ folder (camera cards)."""
    from ingest import detect_cards
    return {"cards": detect_cards()}


@app.get("/api/ingest/scan")
def ingest_scan(path: str):
    """Return file count for a source path (card or local folder)."""
    from ingest import scan_source
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise HTTPException(404, f"path not found: {p}")
    files = scan_source(p)
    return {"path": str(p), "count": len(files)}


@app.get("/api/ingest/aliases")
def ingest_aliases():
    """Return the camera model alias table so the UI can display it."""
    from ingest import _MODEL_ALIASES
    return {"aliases": _MODEL_ALIASES}


@app.post("/api/ingest/start")
def ingest_start(req: IngestStartRequest):
    dest = Path(req.dest_dir).expanduser()
    job = ingest_runner.MANAGER.create(req.sources, str(dest))
    ingest_runner.MANAGER.start(job)
    return {"job_id": job.id}


@app.get("/api/ingest/progress/{job_id}")
async def ingest_progress(job_id: str, request: Request):
    job = ingest_runner.MANAGER.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown ingest job")

    async def event_stream():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = job.events.get(timeout=0.25)
            except Exception:
                if job.status in ("done", "error"):
                    while not job.events.empty():
                        ev = job.events.get_nowait()
                        yield f"data: {json.dumps(ev, default=str)}\n\n"
                        if ev.get("type") == "__end__":
                            return
                    yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'summary': job.summary, 'error': job.error}, default=str)}\n\n"
                    return
                yield ": heartbeat\n\n"
                continue
            if event.get("type") == "__end__":
                yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'summary': job.summary, 'error': job.error}, default=str)}\n\n"
                return
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------

def serve(host: str = "127.0.0.1", port: int = 8765, reload: bool = False):
    import uvicorn
    uvicorn.run("ui.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    serve(args.host, args.port, args.reload)

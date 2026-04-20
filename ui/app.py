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

from ui import pipeline_runner, review, thumbs
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


@app.get("/api/profiles")
def list_profiles():
    """List available model profiles (models/*.pt with optional .json sidecar)."""
    models_dir = ROOT / "models"
    profiles = []
    if models_dir.exists():
        for pt in sorted(models_dir.glob("*.pt")):
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
            })

    # Also surface the default model.pt in repo root if present
    root_pt = ROOT / "model.pt"
    if root_pt.exists() and not any(p['name'] == "model" for p in profiles):
        profiles.insert(0, {
            "name": "(default)",
            "path": str(root_pt),
            "description": "legacy model.pt in repo root",
            "trained_at": "",
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
                        yield f"data: {json.dumps(ev)}\n\n"
                        if ev.get("type") == "__end__":
                            return
                    # Final status snapshot
                    yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'summary': job.summary, 'error': job.error})}\n\n"
                    return
                # Keep connection alive
                yield ": heartbeat\n\n"
                continue
            if event.get("type") == "__end__":
                yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'summary': job.summary, 'error': job.error})}\n\n"
                return
            yield f"data: {json.dumps(event)}\n\n"

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

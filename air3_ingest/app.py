"""
FastAPI entrypoint for the Air3 ingest tool.

Run with:
    python air3_ingest/app.py

Serves the UI at http://localhost:8766/ by default.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import merge

app = FastAPI(title="air3-ingest")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

PREFS_PATH = Path(__file__).parent / ".prefs.json"
DEFAULT_PREFS = {
    "source_dir": "/Volumes/Air3_mSD/DCIM",
    "destination_dir": str(Path.home() / "Desktop" / "Air3 Merged"),
    "gap_threshold_s": merge.GAP_THRESHOLD_DEFAULT_S,
}


def load_prefs() -> dict:
    if PREFS_PATH.exists():
        try:
            return {**DEFAULT_PREFS, **json.loads(PREFS_PATH.read_text())}
        except (json.JSONDecodeError, OSError):
            pass
    return dict(DEFAULT_PREFS)


def save_prefs(prefs: dict) -> None:
    PREFS_PATH.write_text(json.dumps(prefs, indent=2))


def _applescript_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def choose_folder_dialog(prompt: str, default_dir: str | None) -> str | None:
    script = f'POSIX path of (choose folder with prompt "{_applescript_escape(prompt)}"'
    if default_dir and Path(default_dir).is_dir():
        script += f' default location (POSIX file "{_applescript_escape(default_dir)}")'
    script += ")"
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if result.returncode != 0:
        return None  # user cancelled, or dialog failed
    return result.stdout.strip()


class ChooseDirRequest(BaseModel):
    which: str  # "source" | "destination"


class ScanRequest(BaseModel):
    source_dir: str
    gap_threshold_s: float = merge.GAP_THRESHOLD_DEFAULT_S


class ProcessRequest(BaseModel):
    source_dir: str
    destination_dir: str
    gap_threshold_s: float = merge.GAP_THRESHOLD_DEFAULT_S
    # Groups to process, identified by their exact clip filename list (as
    # returned by /api/scan's group_summary "clip_names") rather than a
    # positional index -- /api/process re-runs discovery from scratch, and if
    # the source folder's contents changed since the scan (e.g. still
    # copying off the SD card, or a file removed), positional indices could
    # silently point at a different group than what the user selected.
    # None = process every discovered group.
    selected_groups: list[list[str]] | None = None


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/prefs")
def get_prefs():
    return load_prefs()


@app.post("/api/choose_dir")
def choose_dir(req: ChooseDirRequest):
    if req.which not in ("source", "destination"):
        raise HTTPException(400, "which must be 'source' or 'destination'")
    prefs = load_prefs()
    key = f"{req.which}_dir"
    prompt = (
        "Choose Air3 source folder (SD card or DCIM folder)"
        if req.which == "source"
        else "Choose destination folder for merged videos"
    )
    chosen = choose_folder_dialog(prompt, prefs.get(key))
    if chosen is None:
        return {"cancelled": True, "path": prefs.get(key)}
    prefs[key] = chosen
    save_prefs(prefs)
    return {"cancelled": False, "path": chosen}


def _validated_gap_threshold(gap_threshold_s: float) -> float:
    if gap_threshold_s <= 0:
        raise HTTPException(400, f"gap_threshold_s must be > 0, got {gap_threshold_s}")
    return gap_threshold_s


def _discover_and_group(source_dir: str, gap_threshold_s: float):
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise HTTPException(400, f"source_dir does not exist: {source_dir}")
    try:
        clips, warnings = merge.discover_clips(source_path)
        groups = merge.group_clips(clips, gap_threshold_s)
        summaries = [merge.group_summary(g) for g in groups]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"discovery/grouping failed: {e}")
    return groups, summaries, warnings


@app.post("/api/scan")
def scan(req: ScanRequest):
    gap_threshold_s = _validated_gap_threshold(req.gap_threshold_s)
    prefs = load_prefs()
    prefs["source_dir"] = req.source_dir
    prefs["gap_threshold_s"] = gap_threshold_s
    save_prefs(prefs)

    _groups, summaries, warnings = _discover_and_group(req.source_dir, gap_threshold_s)
    return {"groups": summaries, "warnings": warnings}


@app.post("/api/process")
def process(req: ProcessRequest):
    gap_threshold_s = _validated_gap_threshold(req.gap_threshold_s)
    prefs = load_prefs()
    prefs["source_dir"] = req.source_dir
    prefs["destination_dir"] = req.destination_dir
    prefs["gap_threshold_s"] = gap_threshold_s
    save_prefs(prefs)

    if not req.destination_dir:
        raise HTTPException(400, "destination_dir is required")
    dest_path = Path(req.destination_dir)

    groups, _summaries, discovery_warnings = _discover_and_group(req.source_dir, gap_threshold_s)
    groups_by_names = {tuple(c.mp4_path.name for c in g.clips): g for g in groups}

    if req.selected_groups is not None:
        # dict.fromkeys, not set(): de-dupes while preserving the order the
        # user selected groups in, so results come back in a stable order.
        requested = list(dict.fromkeys(tuple(names) for names in req.selected_groups))
    else:
        requested = list(groups_by_names.keys())

    results = []
    for names in requested:
        group = groups_by_names.get(names)
        if group is None:
            results.append({
                "source_files": list(names),
                "ok": False,
                "error": "this exact set of clips is no longer one discovered group -- "
                         "the source folder likely changed since the last scan; re-scan and retry",
            })
            continue
        try:
            r = merge.merge_group(group, dest_path)
        except Exception as e:
            results.append({"source_files": list(names), "ok": False, "error": str(e)})
            continue
        results.append({
            "ok": r.ok,
            "output_path": str(r.output_path) if r.output_path else None,
            "source_files": r.source_files,
            "error": r.error,
            "warnings": r.warnings,
        })
    return {"results": results, "discovery_warnings": discovery_warnings}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)

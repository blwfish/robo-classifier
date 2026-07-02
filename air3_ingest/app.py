"""
FastAPI entrypoint for the Air3 ingest tool.

Run with:
    python air3_ingest/app.py

Serves the UI at http://localhost:8767/ by default.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import audio_merge
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


def _prefs_value_ok(value, default) -> bool:
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, str):
        return isinstance(value, str)
    return type(value) is type(default)


def load_prefs() -> dict:
    if not PREFS_PATH.exists():
        return dict(DEFAULT_PREFS)
    try:
        raw = json.loads(PREFS_PATH.read_text())
    except (json.JSONDecodeError, OSError) as e:
        print(f"air3_ingest: {PREFS_PATH} is unreadable/corrupt ({e}); using defaults")
        return dict(DEFAULT_PREFS)
    if not isinstance(raw, dict):
        print(f"air3_ingest: {PREFS_PATH} does not contain a JSON object "
              f"(got {type(raw).__name__}); using defaults")
        return dict(DEFAULT_PREFS)

    prefs = dict(DEFAULT_PREFS)
    for key, default in DEFAULT_PREFS.items():
        if key not in raw:
            continue
        value = raw[key]
        if _prefs_value_ok(value, default):
            prefs[key] = value
        else:
            print(f"air3_ingest: {PREFS_PATH} key {key!r} has wrong type "
                  f"({type(value).__name__}, expected {type(default).__name__}); "
                  f"ignoring, using default")
    return prefs


def save_prefs(prefs: dict) -> None:
    PREFS_PATH.write_text(json.dumps(prefs, indent=2))


def _applescript_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def choose_folder_dialog(prompt: str, default_dir: str | None) -> tuple[str | None, str | None]:
    """Returns (chosen_path, error).

    Exactly one of these is meaningful at a time:
      - success: (path, None)
      - user cancelled the dialog: (None, None) -- not an error
      - genuine failure (permission denial, malformed prompt, osascript
        missing, unexpected output): (None, "diagnostic message")

    A prior version conflated every non-zero osascript exit with "user
    cancelled" and discarded stderr entirely, so a broken dialog looked
    identical to normal user behavior.
    """
    script = f'POSIX path of (choose folder with prompt "{_applescript_escape(prompt)}"'
    if default_dir and Path(default_dir).is_dir():
        script += f' default location (POSIX file "{_applescript_escape(default_dir)}")'
    script += ")"
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    except FileNotFoundError:
        return None, "osascript is not available on this system"

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "-128" in stderr:  # AppleScript's "User canceled." error number
            return None, None
        return None, stderr or f"osascript exited {result.returncode} with no diagnostic output"

    chosen = result.stdout.strip()
    if not chosen.startswith("/"):
        return None, f"osascript returned an unexpected (non-path) value: {chosen!r}"
    return chosen, None


class ChooseDirRequest(BaseModel):
    which: str  # "source" | "destination"


class ScanRequest(BaseModel):
    source_dir: str
    gap_threshold_s: float = merge.GAP_THRESHOLD_DEFAULT_S


class ProcessRequest(BaseModel):
    source_dir: str
    destination_dir: str
    gap_threshold_s: float = merge.GAP_THRESHOLD_DEFAULT_S
    # Groups to process, identified by their exact clip full-path list (as
    # returned by /api/scan's group_summary "clip_paths") rather than a
    # positional index or bare filename -- /api/process re-runs discovery
    # from scratch, and if the source folder's contents changed since the
    # scan (e.g. still copying off the SD card, or a file removed),
    # positional indices could silently point at a different group than
    # what the user selected. Full paths (not just names) matter because
    # DJI cameras paginate onto multiple *MEDIA folders and can restart
    # file numbering per folder, so two clips in different subfolders can
    # share a filename.
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
    chosen, error = choose_folder_dialog(prompt, prefs.get(key))
    if error is not None:
        raise HTTPException(502, f"folder picker failed: {error}")
    if chosen is None:
        return {"cancelled": True, "path": prefs.get(key)}
    prefs[key] = chosen
    save_prefs(prefs)
    return {"cancelled": False, "path": chosen}


def _validated_gap_threshold(gap_threshold_s: float) -> float:
    if gap_threshold_s <= 0:
        raise HTTPException(400, f"gap_threshold_s must be > 0, got {gap_threshold_s}")
    return gap_threshold_s


def _detect_source_kind(source_path: Path) -> str:
    """"video" if any *.mp4 exists anywhere under source_path, else "audio"
    if any *.wav exists, else "empty". A folder is assumed to hold exactly
    one kind of source in practice (a drone SD card vs. an audio recorder's
    card are never the same folder) -- checked in this order so a folder
    that somehow has both is treated as video, not silently as a mix."""
    has_mp4 = has_wav = False
    for p in source_path.rglob("*"):
        if not p.is_file():
            continue
        suffix = p.suffix.lower()
        if suffix == ".mp4":
            has_mp4 = True
        elif suffix == ".wav":
            has_wav = True
        if has_mp4:
            break
    if has_mp4:
        return "video"
    if has_wav:
        return "audio"
    return "empty"


def _discover_and_group(source_dir: str, gap_threshold_s: float):
    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise HTTPException(400, f"source_dir does not exist: {source_dir}")
    try:
        kind = _detect_source_kind(source_path)
        if kind == "video":
            clips, warnings = merge.discover_clips(source_path)
            groups = merge.group_clips(clips, gap_threshold_s)
            summaries = [merge.group_summary(g) for g in groups]
            extra = {}
        elif kind == "audio":
            clips, disc_warnings = audio_merge.discover_audio_clips(source_path)
            groups, trims, group_warnings = audio_merge.group_audio_clips(clips)
            summaries = [audio_merge.audio_group_summary(g) for g in groups]
            warnings = disc_warnings + group_warnings
            extra = {"trims": trims}
        else:
            groups, summaries, warnings, extra = [], [], [], {}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"discovery/grouping failed: {e}")
    return kind, groups, summaries, warnings, extra


@app.post("/api/scan")
def scan(req: ScanRequest):
    gap_threshold_s = _validated_gap_threshold(req.gap_threshold_s)
    prefs = load_prefs()
    prefs["source_dir"] = req.source_dir
    prefs["gap_threshold_s"] = gap_threshold_s
    save_prefs(prefs)

    kind, _groups, summaries, warnings, _extra = _discover_and_group(req.source_dir, gap_threshold_s)
    return {"kind": kind, "groups": summaries, "warnings": warnings}


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

    kind, groups, _summaries, discovery_warnings, extra = _discover_and_group(req.source_dir, gap_threshold_s)
    path_attr = "mp4_path" if kind == "video" else "wav_path"
    groups_by_paths = {tuple(str(getattr(c, path_attr)) for c in g.clips): g for g in groups}

    if req.selected_groups is not None:
        # dict.fromkeys, not set(): de-dupes while preserving the order the
        # user selected groups in, so results come back in a stable order.
        requested = list(dict.fromkeys(tuple(paths) for paths in req.selected_groups))
    else:
        requested = list(groups_by_paths.keys())

    results = []
    for paths in requested:
        group = groups_by_paths.get(paths)
        if group is None:
            results.append({
                "source_files": [Path(p).name for p in paths],
                "ok": False,
                "error": "this exact set of clips is no longer one discovered group -- "
                         "the source folder likely changed since the last scan; re-scan and retry",
            })
            continue
        try:
            if kind == "video":
                r = merge.merge_group(group, dest_path)
            else:
                r = audio_merge.merge_audio_group(group, extra["trims"], dest_path)
        except Exception as e:
            results.append({"source_files": [Path(p).name for p in paths], "ok": False, "error": str(e)})
            continue
        results.append({
            "ok": r.ok,
            "output_path": str(r.output_path) if r.output_path else None,
            "source_files": r.source_files,
            "error": r.error,
            "warnings": r.warnings,
        })
    return {
        "results": results,
        "discovery_warnings": discovery_warnings,
        "failed_count": sum(1 for r in results if not r["ok"]),
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()
    uvicorn.run(app, host="127.0.0.1", port=args.port)

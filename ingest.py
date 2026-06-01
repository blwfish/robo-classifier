"""
ingest.py — pre-shoot ingestion: card detection, EXIF rename, hash-dedup copy.

Rename format: YYYYMMDDHHmmss-{ALIAS}_{OriginalFilename.ext}
  e.g.  20260228103758-Z9_BLW4730.NEF
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Camera model alias table
# ---------------------------------------------------------------------------
# Maps raw EXIF CameraModelName → filename-safe slug.
# Keys compared case-insensitively after stripping whitespace.
# DJI bodies report part numbers (FC####) instead of marketing names.

_MODEL_ALIASES: dict[str, str] = {
    # Nikon Z-series
    "NIKON Z 9":    "Z9",
    "NIKON Z 8":    "Z8",
    "NIKON Z 6_3":  "Z6III",   # placeholder — verify against real file
    "NIKON Z 6III": "Z6III",
    "NIKON Z 7_2":  "Z7II",
    "NIKON Z 7II":  "Z7II",
    "NIKON Z 6_2":  "Z6II",
    "NIKON Z 6II":  "Z6II",
    "NIKON Z 6":    "Z6",
    "NIKON Z 5_2":  "Z5II",
    "NIKON Z 5":    "Z5",
    "NIKON Z 50":   "Z50",
    "NIKON Z 30":   "Z30",
    "NIKON Z FC":   "ZFC",
    # Nikon D-series
    "NIKON D850":   "D850",
    "NIKON D810":   "D810",
    "NIKON D800":   "D800",
    "NIKON D750":   "D750",
    "NIKON D780":   "D780",
    "NIKON D500":   "D500",
    "NIKON D6":     "D6",
    # Canon R-series
    "CANON EOS R5":           "R5",
    "CANON EOS R5 C":         "R5C",
    "CANON EOS R6":           "R6",
    "CANON EOS R6 MARK II":   "R6II",
    "CANON EOS R3":           "R3",
    "CANON EOS-1D X MARK III":"1DXIII",
    # Sony
    "ILCE-9M3":  "A9III",
    "ILCE-9M2":  "A9II",
    "ILCE-1":    "A1",
    "ILCE-7M4":  "A7IV",
    "ILCE-7RM5": "A7RV",
    # DJI (FC part numbers)
    "FC8284": "Air3",       # DJI Air 3
    "FC3582": "Air3",
    "FC3170": "Mini3",      # DJI Mini 3
    "FC3411": "Mini3Pro",   # DJI Mini 3 Pro
    "FC7203": "Mini2",      # DJI Mini 2
    "FC2103": "MavicAir2",
    "FC220":  "Mavic",
    "FC350":  "Phantom3",
    "FC6310": "Phantom4Pro",
    "FC330":  "Phantom4",
}

_ALIAS_LOOKUP: dict[str, str] = {k.lower(): v for k, v in _MODEL_ALIASES.items()}


def model_alias(exif_model: str) -> str:
    """Return the filename-safe alias for a raw EXIF camera model string."""
    key = exif_model.strip().lower()
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]
    # Fallback: strip whitespace and characters unsafe in filenames.
    slug = "".join(c for c in exif_model.strip() if c not in ' /\\:*?"<>|')
    return slug or "Unknown"


# ---------------------------------------------------------------------------
# Supported file extensions
# ---------------------------------------------------------------------------

PHOTO_EXTENSIONS = frozenset({
    ".nef", ".cr2", ".cr3", ".arw", ".dng", ".orf", ".raf", ".rw2",
    ".jpg", ".jpeg",
})
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".avi", ".mkv"})
ALL_EXTENSIONS = PHOTO_EXTENSIONS | VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Card detection
# ---------------------------------------------------------------------------

def detect_cards() -> list[dict]:
    """
    Return mounted volumes that look like camera cards (contain DCIM/).
    macOS: scans /Volumes/*. Windows: scans removable drive letters.
    Each entry: {path, name, dcim, file_count}.
    """
    cards = []

    if platform.system() == "Darwin":
        volumes = Path("/Volumes")
        if not volumes.exists():
            return []
        for vol in sorted(volumes.iterdir()):
            if vol.name.startswith("."):
                continue
            dcim = vol / "DCIM"
            if dcim.is_dir():
                count = _count_files(dcim)
                cards.append({
                    "path": str(vol),
                    "name": vol.name,
                    "dcim": str(dcim),
                    "file_count": count,
                })

    elif platform.system() == "Windows":
        import string
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            dcim = drive / "DCIM"
            if dcim.is_dir():
                count = _count_files(dcim)
                cards.append({
                    "path": str(drive),
                    "name": f"{letter}:",
                    "dcim": str(dcim),
                    "file_count": count,
                })

    return cards


def _count_files(root: Path) -> int:
    return sum(
        1 for f in root.rglob("*")
        if f.is_file() and f.suffix.lower() in ALL_EXTENSIONS
    )


# ---------------------------------------------------------------------------
# File scanning
# ---------------------------------------------------------------------------

def scan_source(source_path: str | Path) -> list[dict]:
    """
    Enumerate all ingestable files under source_path.

    If source_path has a DCIM/ subdir, only files under DCIM/ are returned
    (ignores camera firmware dirs, MISC/, etc.).  For a plain local folder
    with no DCIM/, the whole tree is scanned.

    Returns list of {path: str, ext: str, size: int}, sorted by path.
    """
    root = Path(source_path)
    dcim = root / "DCIM"
    search_root = dcim if dcim.is_dir() else root

    files = []
    for f in search_root.rglob("*"):
        if not f.is_file():
            continue
        # Skip hidden dirs (e.g. .Spotlight-V100 on macOS cards)
        if any(part.startswith(".") for part in f.parts):
            continue
        if f.suffix.lower() in ALL_EXTENSIONS:
            files.append({
                "path": str(f),
                "ext": f.suffix.lower(),
                "size": f.stat().st_size,
            })

    return sorted(files, key=lambda x: x["path"])


# ---------------------------------------------------------------------------
# EXIF batch reading
# ---------------------------------------------------------------------------

def _read_exif_batch(paths: list[Path]) -> dict[str, dict]:
    """
    Read DateTimeOriginal and CameraModelName for a batch of files.
    Returns {str(path): {datetime: str|None, model: str}}.
    """
    if not paths:
        return {}

    cmd = [
        "exiftool", "-json", "-q",
        "-DateTimeOriginal",
        "-CreateDate",      # DJI uses CreateDate, not DateTimeOriginal
        "-CameraModelName",
        "-Model",           # fallback tag name
        "--",
        *[str(p) for p in paths],
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        records = json.loads(result.stdout)
    except Exception:
        return {}

    out: dict[str, dict] = {}
    for rec in records:
        src = rec.get("SourceFile", "")
        dt = rec.get("DateTimeOriginal") or rec.get("CreateDate") or ""
        model = rec.get("CameraModelName") or rec.get("Model") or ""
        out[src] = {"datetime": dt, "model": model}
    return out


def _parse_dt(dt_str: str) -> Optional[str]:
    """
    Convert exiftool datetime '2026:02:28 10:37:58' → '20260228103758'.
    Returns None if unparseable.
    """
    if not dt_str:
        return None
    # Both EXIF ("2026:02:28 10:37:58") and ISO ("2026-02-28T10:37:58") formats
    # have their datetime in the first 19 characters; anything beyond is timezone.
    dt_str = dt_str[:19] if len(dt_str) > 19 else dt_str
    cleaned = dt_str.replace(":", "").replace(" ", "").replace("T", "")
    digits = "".join(c for c in cleaned if c.isdigit())
    if len(digits) >= 14:
        return digits[:14]
    return None


# ---------------------------------------------------------------------------
# Hash manifest (SQLite, global per user)
# ---------------------------------------------------------------------------

_MANIFEST_PATH = Path.home() / ".robo-classifier" / "ingest.db"


def _open_manifest(db_path: Path = _MANIFEST_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingested (
            hash          TEXT PRIMARY KEY,
            dest_path     TEXT NOT NULL,
            camera_alias  TEXT,
            orig_filename TEXT,
            ingest_ts     REAL,
            source_label  TEXT
        )
    """)
    conn.commit()
    return conn


def _file_hash(path: Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _known_hash(conn: sqlite3.Connection, hash_: str) -> Optional[str]:
    row = conn.execute(
        "SELECT dest_path FROM ingested WHERE hash=?", (hash_,)
    ).fetchone()
    return row[0] if row else None


def _record(
    conn: sqlite3.Connection,
    hash_: str,
    dest_path: str,
    camera_alias: str,
    orig_filename: str,
    source_label: str,
):
    conn.execute(
        """INSERT OR REPLACE INTO ingested
           (hash, dest_path, camera_alias, orig_filename, ingest_ts, source_label)
           VALUES (?,?,?,?,?,?)""",
        (hash_, dest_path, camera_alias, orig_filename, time.time(), source_label),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Rename helper
# ---------------------------------------------------------------------------

def build_dest_name(dt_str: Optional[str], model: str, orig_name: str) -> str:
    """Build the renamed filename: YYYYMMDDHHmmss-{ALIAS}_{orig_name}."""
    ts = dt_str or "00000000000000"
    alias = model_alias(model) if model else "Unknown"
    return f"{ts}-{alias}_{orig_name}"


# ---------------------------------------------------------------------------
# Core ingest
# ---------------------------------------------------------------------------

def ingest(
    sources: list[dict],
    dest_dir: str | Path,
    progress_cb: Optional[Callable[[dict], None]] = None,
    manifest_path: Path = _MANIFEST_PATH,
) -> dict:
    """
    Copy + rename files from source cards/dirs into dest_dir.

    sources: list of dicts, each with:
        path  (str)  — card root or local directory
        label (str)  — human name shown in progress (card name / volume name)
        force (bool) — if True, re-copy even if hash already in manifest

    Progress events emitted via progress_cb:
        {type: "scan_start",  source: label}
        {type: "scan_done",   source: label, count: n}
        {type: "exif_start",  total: n}
        {type: "progress",    stage: "copy", done: n, total: N,
                              filename: orig, new_name: renamed}
        {type: "file_result", filename: orig, status: "copied"|"skipped"|"error",
                              dest: new_name, reason: str}
        {type: "__end__",     summary: {copied, skipped, errors, total, dest}}

    Returns the summary dict.
    """

    def emit(event: dict):
        if progress_cb:
            progress_cb(event)

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    conn = _open_manifest(manifest_path)

    # ---- Phase 1: scan all sources ----------------------------------------
    all_files: list[dict] = []
    for src in sources:
        label = src.get("label", src["path"])
        emit({"type": "scan_start", "source": label})
        files = scan_source(src["path"])
        for f in files:
            f["source_label"] = label
            f["force"] = src.get("force", False)
        all_files.extend(files)
        emit({"type": "scan_done", "source": label, "count": len(files)})

    total = len(all_files)
    if total == 0:
        summary = {"copied": 0, "skipped": 0, "errors": 0, "total": 0, "dest": str(dest)}
        emit({"type": "__end__", "summary": summary})
        conn.close()
        return summary

    # ---- Phase 2: batch EXIF read (photos only) ---------------------------
    photo_paths = [
        Path(f["path"]) for f in all_files if f["ext"] in PHOTO_EXTENSIONS
    ]
    emit({"type": "exif_start", "total": len(photo_paths)})
    exif_map = _read_exif_batch(photo_paths)

    # ---- Phase 3: hash-checked copy ---------------------------------------
    copied = skipped = errors = 0

    for i, f in enumerate(all_files):
        src_path = Path(f["path"])
        exif = exif_map.get(str(src_path), {})
        dt = _parse_dt(exif.get("datetime", ""))
        model = exif.get("model", "")

        new_name = build_dest_name(dt, model, src_path.name)
        dest_path = dest / new_name

        emit({
            "type": "progress",
            "stage": "copy",
            "done": i,
            "total": total,
            "filename": src_path.name,
            "new_name": new_name,
        })

        try:
            file_hash = _file_hash(src_path)
            existing = _known_hash(conn, file_hash)

            if existing and not f["force"]:
                skipped += 1
                emit({
                    "type": "file_result",
                    "filename": src_path.name,
                    "status": "skipped",
                    "reason": f"already ingested → {existing}",
                })
                continue

            # Resolve name collision (same filename, different content)
            if dest_path.exists():
                stem, suffix = dest_path.stem, dest_path.suffix
                n = 1
                while dest_path.exists():
                    dest_path = dest / f"{stem}_c{n}{suffix}"
                    n += 1

            shutil.copy2(str(src_path), str(dest_path))
            _record(
                conn, file_hash, str(dest_path),
                model_alias(model) if model else "Unknown",
                src_path.name,
                f["source_label"],
            )
            copied += 1
            emit({
                "type": "file_result",
                "filename": src_path.name,
                "status": "copied",
                "dest": dest_path.name,
            })

        except OSError as e:
            errors += 1
            emit({
                "type": "file_result",
                "filename": src_path.name,
                "status": "error",
                "reason": str(e),
            })

    conn.close()
    summary = {
        "copied": copied,
        "skipped": skipped,
        "errors": errors,
        "total": total,
        "dest": str(dest),
    }
    emit({"type": "__end__", "summary": summary})
    return summary


# ---------------------------------------------------------------------------
# CLI (quick test / standalone use)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Ingest camera cards into a dated directory.")
    parser.add_argument("dest", help="Destination directory")
    parser.add_argument("sources", nargs="*", help="Source paths (default: auto-detect cards)")
    parser.add_argument("--force", action="store_true", help="Re-ingest already-known files")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report without copying")
    args = parser.parse_args()

    if args.sources:
        srcs = [{"path": p, "label": Path(p).name, "force": args.force} for p in args.sources]
    else:
        cards = detect_cards()
        if not cards:
            print("No camera cards detected.", file=sys.stderr)
            sys.exit(1)
        for c in cards:
            print(f"Found card: {c['name']} ({c['file_count']} files)")
        srcs = [{"path": c["path"], "label": c["name"], "force": args.force} for c in cards]

    if args.dry_run:
        total = 0
        for src in srcs:
            files = scan_source(src["path"])
            print(f"{src['label']}: {len(files)} files")
            total += len(files)
        print(f"Total: {total} files → {args.dest}")
        sys.exit(0)

    def cb(ev):
        t = ev.get("type")
        if t == "scan_done":
            print(f"  Scanned {ev['source']}: {ev['count']} files")
        elif t == "exif_start":
            print(f"  Reading EXIF from {ev['total']} photos…")
        elif t == "file_result":
            status = ev["status"]
            if status == "copied":
                print(f"  → {ev['dest']}")
            elif status == "skipped":
                print(f"  skip {ev['filename']} ({ev['reason']})")
            elif status == "error":
                print(f"  ERROR {ev['filename']}: {ev['reason']}", file=sys.stderr)
        elif t == "__end__":
            s = ev["summary"]
            print(f"\nDone: {s['copied']} copied, {s['skipped']} skipped, {s['errors']} errors")

    ingest(srcs, args.dest, progress_cb=cb)

"""Subprocess/path helpers shared between merge.py (video) and audio_merge.py (audio)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run(cmd: list[str], warnings: list[str] | None = None) -> str:
    # errors="replace" (not the subprocess default "strict"): stderr from
    # ffmpeg/ffprobe/exiftool/bwfmetaedit is diagnostic text we only ever log,
    # never parse, so a stray non-UTF-8 byte should degrade gracefully rather
    # than raising UnicodeDecodeError -- which callers can't catch via
    # `except RuntimeError`.
    result = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\n{result.stderr.strip()}")
    stderr = result.stderr.strip()
    if stderr and warnings is not None:
        warnings.append(f"{cmd[0]} warning: {stderr}")
    return result.stdout


def quote_concat_path(p: Path) -> str:
    return "file '" + str(p.resolve()).replace("'", "'\\''") + "'"


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem}_v{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1

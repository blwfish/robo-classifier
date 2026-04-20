"""
Review-state I/O: color labels and crop rectangles.

Writes Lightroom/Bridge-compatible XMP metadata via exiftool. For RAW files,
writes into the .xmp sidecar; for JPEGs, embeds directly.

Fields written:
  - xmp:Label — "Red" | "Yellow" | "Green" | "Blue" | "Purple" | ""  (clears)
  - crs:HasCrop, crs:CropLeft/Top/Right/Bottom (0-1), crs:CropAngle
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional

from image_utils import RAW_EXTENSIONS


LABEL_COLORS = {"Red", "Yellow", "Green", "Blue", "Purple", ""}


def _target_for_write(source: Path) -> tuple[Path, bool]:
    """
    Return (target_path, needs_sidecar_creation).
    For RAWs, target is the .xmp next to the file; for JPEGs, the file itself.
    """
    ext = source.suffix.lower()
    if ext in RAW_EXTENSIONS:
        xmp = source.with_suffix('.xmp')
        return xmp, not xmp.exists()
    return source, False


def _run_exiftool(args: list[str]) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ['exiftool', '-overwrite_original', *args],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0, (result.stderr or result.stdout)
    except FileNotFoundError:
        return False, "exiftool not found"


def set_label(source: Path, color: str) -> tuple[bool, str]:
    """Set or clear the XMP color label on a file. color='' clears."""
    if color not in LABEL_COLORS:
        return False, f"invalid color: {color!r}"

    target, needs_create = _target_for_write(source)

    if needs_create:
        # Create sidecar seeded from RAW metadata, then set label.
        ok, msg = _run_exiftool([
            '-tagsfromfile', str(source),
            f'-xmp:Label={color}',
            '-o', str(target),
        ])
        return ok, msg

    return _run_exiftool([f'-xmp:Label={color}', str(target)])


def set_crop(source: Path, left: float, top: float, right: float, bottom: float,
             angle: float = 0.0) -> tuple[bool, str]:
    """
    Set a Lightroom-compatible crop rectangle (all values 0-1 normalized).

    left/top/right/bottom are fractions of the original image dimensions.
    angle is in degrees (positive = counter-clockwise).
    """
    for v, name in [(left, 'left'), (top, 'top'), (right, 'right'), (bottom, 'bottom')]:
        if not 0.0 <= v <= 1.0:
            return False, f"{name} out of [0,1]: {v}"
    if left >= right or top >= bottom:
        return False, "degenerate crop rect"

    target, needs_create = _target_for_write(source)

    args = [
        '-crs:HasCrop=true',
        f'-crs:CropLeft={left:.6f}',
        f'-crs:CropTop={top:.6f}',
        f'-crs:CropRight={right:.6f}',
        f'-crs:CropBottom={bottom:.6f}',
        f'-crs:CropAngle={angle:.2f}',
    ]

    if needs_create:
        return _run_exiftool([
            '-tagsfromfile', str(source),
            *args,
            '-o', str(target),
        ])

    return _run_exiftool([*args, str(target)])


def clear_crop(source: Path) -> tuple[bool, str]:
    """Remove any existing crop metadata."""
    target, needs_create = _target_for_write(source)
    if needs_create:
        return True, "no crop to clear (no sidecar)"
    return _run_exiftool([
        '-crs:HasCrop=',
        '-crs:CropLeft=',
        '-crs:CropTop=',
        '-crs:CropRight=',
        '-crs:CropBottom=',
        '-crs:CropAngle=',
        str(target),
    ])


def read_state(source: Path) -> dict:
    """
    Read back label + crop from XMP (sidecar for RAW, embedded for JPEG).
    Returns {'label': str|None, 'crop': {...}|None}.
    """
    target, needs_create = _target_for_write(source)
    if needs_create:
        return {'label': None, 'crop': None}

    try:
        result = subprocess.run(
            [
                'exiftool', '-json',
                '-xmp:Label',
                '-crs:HasCrop',
                '-crs:CropLeft', '-crs:CropTop',
                '-crs:CropRight', '-crs:CropBottom',
                '-crs:CropAngle',
                str(target),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {'label': None, 'crop': None}
        data = json.loads(result.stdout)
        if not data:
            return {'label': None, 'crop': None}
        item = data[0]
    except (FileNotFoundError, json.JSONDecodeError):
        return {'label': None, 'crop': None}

    label = item.get('Label') or None
    crop = None
    if item.get('HasCrop'):
        crop = {
            'left': float(item.get('CropLeft', 0)),
            'top': float(item.get('CropTop', 0)),
            'right': float(item.get('CropRight', 1)),
            'bottom': float(item.get('CropBottom', 1)),
            'angle': float(item.get('CropAngle', 0)),
        }
    return {'label': label, 'crop': crop}

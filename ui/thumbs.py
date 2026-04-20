"""
Thumbnail cache for UI image browsing.

Thumbnails go in <input_dir>/.robo-classifier/thumbs/ keyed by source filename.
For RAW files, we pull the embedded preview via exiftool; for JPEGs we
downsample with PIL.
"""

from __future__ import annotations

import hashlib
import io
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from PIL import Image, ImageOps

from image_utils import RAW_EXTENSIONS, ExiftoolProcess, default_workers


THUMB_MAX = 512  # longest edge in px
FULL_MAX = 2048  # for "detail view" rendering of RAWs


def cache_dir(input_dir: Path) -> Path:
    d = input_dir / ".robo-classifier" / "thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(input_dir: Path, source: Path, size: int, suffix: str = ".jpg") -> Path:
    # Hash the full path to avoid collisions if files from different subdirs
    # ever share names. Keep the stem for human-readability.
    h = hashlib.sha1(str(source.resolve()).encode()).hexdigest()[:10]
    return cache_dir(input_dir) / f"{source.stem}_{h}_{size}{suffix}"


def get_thumb(input_dir: Path, source: Path, size: int = THUMB_MAX) -> Path:
    """
    Return path to a cached thumbnail for source, creating it if needed.
    Raises FileNotFoundError if source doesn't exist.
    """
    if not source.exists():
        raise FileNotFoundError(source)

    out = _cache_path(input_dir, source, size)
    if out.exists() and out.stat().st_mtime >= source.stat().st_mtime:
        return out

    ext = source.suffix.lower()
    if ext in RAW_EXTENSIONS:
        # Extract embedded preview and downsample
        with ExiftoolProcess() as et:
            preview = et.execute('-b', '-PreviewImage', str(source))
        if not preview:
            raise RuntimeError(f"No embedded preview in {source}")
        img = Image.open(io.BytesIO(preview))
    else:
        img = Image.open(source)

    img = ImageOps.exif_transpose(img)  # honor orientation tag
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    img = img.convert("RGB")
    img.save(out, "JPEG", quality=85, optimize=True)
    return out


def get_full(input_dir: Path, source: Path) -> Path:
    """
    Return a path to a renderable JPEG for the detail view.
    For JPEGs, that's just the source. For RAWs, a cached downsampled preview.
    """
    ext = source.suffix.lower()
    if ext not in RAW_EXTENSIONS:
        return source
    return get_thumb(input_dir, source, size=FULL_MAX)


def _write_thumb_from_input(
    input_dir: Path, source: Path, input_image_path: Path, size: int
) -> Optional[Path]:
    """
    Create a cached thumbnail for `source` by reading pixels from
    `input_image_path` (which may be the source itself, or an already-extracted
    RAW preview JPEG). Returns None on failure.
    """
    out = _cache_path(input_dir, source, size)
    try:
        if out.exists() and out.stat().st_mtime >= source.stat().st_mtime:
            return out
        img = Image.open(input_image_path)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        img = img.convert("RGB")
        img.save(out, "JPEG", quality=85, optimize=True)
        return out
    except Exception:
        return None


def pregenerate_thumbs(
    input_dir: Path,
    sources,
    preview_map: Optional[dict] = None,
    workers: Optional[int] = None,
    size: int = THUMB_MAX,
    progress_cb=None,
):
    """
    Batch-create thumbnails for a list of source paths, using a thread pool.

    If `preview_map` (source -> already-extracted preview JPEG path) is given,
    RAW sources decode their thumbnails from the cached preview rather than
    re-invoking exiftool. JPEG sources always decode directly.

    PIL releases the GIL for encode/decode, so threads are effective here.
    Skips RAW sources that aren't in preview_map (they'll lazy-load later).
    """
    input_dir = Path(input_dir)
    sources = [Path(s) for s in sources]
    preview_map = preview_map or {}
    workers = workers or default_workers()

    # Build (source, input_path) pairs, skipping RAWs without a preview.
    tasks = []
    for s in sources:
        ext = s.suffix.lower()
        if ext in RAW_EXTENSIONS:
            prev = preview_map.get(s)
            if prev is None:
                continue  # lazy fallback in the UI will handle it
            tasks.append((s, Path(prev)))
        else:
            tasks.append((s, s))

    if not tasks:
        return

    # Ensure cache dir exists once, then let workers write to it concurrently.
    cache_dir(input_dir)

    total = len(tasks)
    done = [0]
    lock = threading.Lock()

    def _one(pair):
        src, inp = pair
        _write_thumb_from_input(input_dir, src, inp, size)
        with lock:
            done[0] += 1
            n = done[0]
            if n % 200 == 0 or n == total:
                print(f"\r  Pregenerated {n}/{total} thumbnails...", end="", flush=True)
            if progress_cb is not None:
                progress_cb(n, total)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(_one, t) for t in tasks]))

    print()

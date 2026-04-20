#!/usr/bin/env python3
"""
image_utils.py

Shared utilities for finding images and extracting RAW previews.
Used by classify.py, junk_filter.py, and (future) UI code.
"""

import os
import select
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


RAW_EXTENSIONS = {'.nef', '.cr2', '.cr3', '.arw', '.orf', '.raf', '.dng', '.rw2'}
JPG_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | JPG_EXTENSIONS

# rawpy (libraw bindings) is much faster than shelling out to exiftool for
# preview extraction because it avoids subprocess/pipe overhead and its C code
# releases the GIL, so threads actually parallelize.
try:
    import rawpy as _rawpy  # noqa: F401
    HAS_RAWPY = True
except ImportError:
    HAS_RAWPY = False


def default_workers():
    """Default worker count for parallelism: cpu_count - 2, clamped to >=2."""
    return max(2, (os.cpu_count() or 4) - 2)


def find_images(input_dir, recursive=False):
    """
    Find image files in a directory.

    Args:
        input_dir: directory to search
        recursive: if True, include subdirectories (except dot-dirs)

    Returns:
        (jpg_files, raw_files) tuple — both sorted lists of Path
    """
    input_dir = Path(input_dir)
    jpg_files = []
    raw_files = []

    iterator = input_dir.rglob('*') if recursive else input_dir.iterdir()
    for f in iterator:
        if not f.is_file():
            continue
        # Skip hidden/dot directories when recursive
        if recursive and any(part.startswith('.') for part in f.relative_to(input_dir).parts[:-1]):
            continue
        ext = f.suffix.lower()
        if ext in JPG_EXTENSIONS:
            jpg_files.append(f)
        elif ext in RAW_EXTENSIONS:
            raw_files.append(f)

    return sorted(jpg_files), sorted(raw_files)


class ExiftoolProcess:
    """
    Persistent exiftool process using -stay_open mode.
    Avoids process spawn overhead for each file.

    For binary output (like -b -PreviewImage), we read in chunks and look
    for the ready marker since binary data doesn't have nice line endings.
    """

    def __init__(self):
        self.process = None
        self._seq = 0
        self._start()

    def _start(self):
        self.process = subprocess.Popen(
            ['exiftool', '-stay_open', 'True', '-@', '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self._seq = 0

    def execute(self, *args):
        """Execute an exiftool command and return stdout bytes."""
        if self.process is None or self.process.poll() is not None:
            self._start()

        cmd = '\n'.join(args) + f'\n-execute{self._seq}\n'
        ready_marker = f'{{ready{self._seq}}}'.encode('utf-8')
        self._seq += 1

        self.process.stdin.write(cmd.encode('utf-8'))
        self.process.stdin.flush()

        output = b''
        chunk_size = 65536
        while True:
            readable, _, _ = select.select([self.process.stdout], [], [], 30)
            if not readable:
                break

            chunk = self.process.stdout.read1(chunk_size) if hasattr(self.process.stdout, 'read1') else self.process.stdout.read(chunk_size)
            if not chunk:
                break
            output += chunk

            if output.rstrip().endswith(ready_marker):
                output = output.rstrip()
                if output.endswith(ready_marker):
                    output = output[:-len(ready_marker)].rstrip()
                break

        return output

    def close(self):
        if self.process and self.process.poll() is None:
            self.process.stdin.write(b'-stay_open\nFalse\n')
            self.process.stdin.flush()
            self.process.wait()
        self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _extract_one_rawpy(raw_path, temp_dir, max_edge=None):
    """
    Extract the embedded preview JPEG from one RAW file using rawpy/libraw.

    If max_edge is set and the embedded preview is larger than that on its
    longest edge, decode + downsample + re-encode before writing to disk.
    Z9 NEFs (for example) embed full-resolution ~5MB previews that starve
    downstream models on JPEG-decode time; a 2048px cap shrinks the file
    ~18× and leaves plenty of detail for 640px (YOLO) and 224px (ResNet)
    input sizes.

    Returns (raw_path, preview_path) on success or (raw_path, None) on failure.
    """
    import rawpy
    preview_path = Path(temp_dir) / f"{raw_path.stem}_preview.jpg"
    try:
        with rawpy.imread(str(raw_path)) as raw:
            thumb = raw.extract_thumb()
        if thumb.format != rawpy.ThumbFormat.JPEG:
            # Some formats embed a bitmap (BMP) preview; we'd need to encode it.
            # Uncommon for modern cameras; fall back caller-side.
            return raw_path, None

        data = thumb.data
        if max_edge is not None:
            from PIL import Image, ImageOps
            import io
            img = Image.open(io.BytesIO(data))
            if max(img.size) > max_edge:
                img = ImageOps.exif_transpose(img)
                img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=90, optimize=True)
                data = buf.getvalue()

        with open(preview_path, 'wb') as f:
            f.write(data)
        return raw_path, preview_path
    except Exception:
        return raw_path, None


def _extract_one_exiftool_worker(chunk, temp_dir, done, lock, total, progress_cb):
    """Serial exiftool worker for one chunk — used by fallback path only."""
    local_map = {}
    local_failed = []
    with ExiftoolProcess() as et:
        for raw_path in chunk:
            preview_path = Path(temp_dir) / f"{raw_path.stem}_preview.jpg"
            preview_data = et.execute('-b', '-PreviewImage', str(raw_path))
            if preview_data:
                with open(preview_path, 'wb') as f:
                    f.write(preview_data)
                local_map[raw_path] = preview_path
            else:
                local_failed.append(raw_path)
            with lock:
                done[0] += 1
                if progress_cb is not None:
                    progress_cb(done[0], total)
    return local_map, local_failed


def extract_raw_previews(raw_files, temp_dir, progress_label="Extracting previews",
                         workers=1, progress_cb=None, use_rawpy=True,
                         max_preview_edge=512):
    """
    Extract embedded JPEG previews from RAW files into temp_dir.

    Uses rawpy (libraw bindings) by default — it calls into C with the GIL
    released, so threads scale across cores. Falls back to parallel exiftool
    subprocesses if rawpy isn't installed (or use_rawpy=False).

    Args:
        raw_files: list of RAW Paths
        temp_dir: output directory for preview .jpg files
        progress_label: prefix for the human-readable progress line
        workers: number of parallel worker threads (1 = serial)
        progress_cb: optional callable(current, total) for UI progress events
        use_rawpy: prefer rawpy when available (default True)
        max_preview_edge: if set and backend is rawpy, downsample any preview
            whose longest edge exceeds this, re-encoding at quality 90. Makes
            downstream JPEG decode (YOLO, classifier, thumbnailer) an order
            of magnitude faster on cameras that embed full-resolution
            previews. Default 512 (sufficient for 640px YOLO, 224px ResNet,
            and 512px UI thumbs; caller accepted that partially-clipped cars
            may get more aggressively flagged at this size). Pass None to
            preserve original preview bytes, or a larger value (e.g. 1024,
            2048) for slower but more detection-conservative behavior.

    Returns:
        dict mapping original RAW path -> preview path
    """
    if not raw_files:
        return {}

    workers = max(1, int(workers))
    backend = "rawpy" if (use_rawpy and HAS_RAWPY) else "exiftool"
    print(f"{progress_label} from {len(raw_files)} RAW files "
          f"({backend}, {'serial' if workers == 1 else f'{workers} workers'})...")

    preview_map: dict = {}
    failed: list = []
    lock = threading.Lock()
    done = [0]
    total = len(raw_files)

    def _progress_tick():
        # Caller holds `lock`.
        done[0] += 1
        n = done[0]
        if n % 100 == 0 or n == total:
            print(f"\r  Extracted {n}/{total} previews...", end="", flush=True)
        if progress_cb is not None:
            progress_cb(n, total)

    if backend == "rawpy":
        # rawpy: one file per task — libraw opens/closes per call, so there's
        # no persistent-handle benefit to chunking. Keep the pool simple.
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_extract_one_rawpy, rp, temp_dir, max_preview_edge)
                for rp in raw_files
            ]
            for fut in as_completed(futs):
                raw_path, preview_path = fut.result()
                with lock:
                    if preview_path is not None:
                        preview_map[raw_path] = preview_path
                    else:
                        failed.append(raw_path)
                    _progress_tick()
    else:
        # exiftool fallback: chunk so each worker keeps a persistent process.
        chunks = [raw_files] if workers == 1 else [raw_files[i::workers] for i in range(workers)]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(_extract_one_exiftool_worker, c, temp_dir, done, lock, total, progress_cb)
                for c in chunks if c
            ]
            for fut in as_completed(futs):
                local_map, local_failed = fut.result()
                preview_map.update(local_map)
                failed.extend(local_failed)

    print()
    if failed:
        print(f"  WARNING: Failed to extract {len(failed)} previews")

    return preview_map


def check_exiftool():
    """Return True if exiftool is available, False otherwise."""
    try:
        subprocess.run(['exiftool', '-ver'], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

#!/usr/bin/env python3
"""
junk_filter.py

Detect and remove "junk" frames from an auto-capture directory.

Junk = frames with no usable vehicle in them. Two failure modes:
  - No vehicle detected at all (sky, pit wall, grandstand, empty track)
  - Only vehicles present are badly edge-clipped (Z9 fired at the wrong moment)

Uses YOLOv8 (ultralytics) with COCO classes: car, truck, bus, motorcycle.

Usage (standalone):
    python junk_filter.py /path/to/images
    python junk_filter.py /path/to/images --dry_run
    python junk_filter.py /path/to/images --junk_dir_name excess

Usage (module):
    from junk_filter import filter_directory
    result = filter_directory(input_dir)
"""

import argparse
import csv
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from image_utils import (
    RAW_EXTENSIONS,
    JPG_EXTENSIONS,
    find_images,
    extract_raw_previews,
    default_workers,
)


# COCO class names considered "vehicles" for racing use
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}

DEFAULT_YOLO_MODEL = 'yolov8n.pt'


@dataclass
class Detection:
    cls: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height


@dataclass
class JunkResult:
    path: Path
    is_junk: bool
    reason: str  # 'no_vehicle' | 'all_edge_chopped' | 'keep'
    detections: list = field(default_factory=list)
    usable_detections: list = field(default_factory=list)
    img_width: int = 0
    img_height: int = 0


def _touches_edge(det: Detection, img_w: int, img_h: int, margin: float) -> bool:
    return (
        det.x1 <= margin
        or det.y1 <= margin
        or det.x2 >= img_w - margin
        or det.y2 >= img_h - margin
    )


def _is_edge_chopped(
    det: Detection,
    img_w: int,
    img_h: int,
    edge_margin_px: float,
    min_visible_frac: float,
    edge_min_area_frac: float = 0.0,
) -> bool:
    """
    A detection is "edge-chopped" (i.e. unusable) if it touches an image edge
    AND any of:
      - visible width is below `min_visible_frac` of image width (on L/R edges)
      - visible height is below `min_visible_frac` of image height (on T/B edges)
      - total visible area is below `edge_min_area_frac` of image area

    The area rule catches "slivers" that pass the width/height gate because
    they're elongated along the edge but still represent only a fragment of a
    vehicle (e.g. a car leaning out of the frame at the side).
    """
    touches_left   = det.x1 <= edge_margin_px
    touches_right  = det.x2 >= img_w - edge_margin_px
    touches_top    = det.y1 <= edge_margin_px
    touches_bottom = det.y2 >= img_h - edge_margin_px

    if not (touches_left or touches_right or touches_top or touches_bottom):
        return False

    if (touches_left or touches_right) and det.width < min_visible_frac * img_w:
        return True
    if (touches_top or touches_bottom) and det.height < min_visible_frac * img_h:
        return True
    if edge_min_area_frac > 0 and det.area < edge_min_area_frac * img_w * img_h:
        return True

    return False


def _select_device(device: Optional[str]) -> str:
    if device:
        return device
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def detect_junk(
    image_paths,
    model=None,
    model_path: str = DEFAULT_YOLO_MODEL,
    device: Optional[str] = None,
    min_conf: float = 0.3,
    edge_margin_px: float = 5.0,
    min_visible_frac: float = 0.1,
    min_area_frac: float = 0.002,
    edge_min_area_frac: float = 0.05,
    batch_size: int = 32,
    imgsz: int = 640,
    progress_cb=None,
) -> list[JunkResult]:
    """
    Run YOLO on each image and classify as junk/keep.

    Args:
        image_paths: iterable of Paths to readable images (JPEG/PNG — not RAW).
        model: pre-loaded ultralytics YOLO instance (optional, avoids re-loading).
        model_path: path/name of YOLO weights if model not provided.
        device: 'cpu' | 'mps' | 'cuda' (auto-detect if None).
        min_conf: YOLO detection confidence threshold.
        edge_margin_px: px within image edge to count as "touching".
        min_visible_frac: a vehicle touching an edge whose visible dimension
            along that axis is below this fraction of the image is "chopped".
        min_area_frac: ignore vehicles smaller than this fraction of image area
            (filters tiny background cars to keep the filter conservative).
        batch_size: YOLO batch size.
        imgsz: YOLO inference image size.

    Returns:
        list of JunkResult, one per input path (failed loads are skipped silently).
    """
    from ultralytics import YOLO

    device = _select_device(device)
    if model is None:
        model = YOLO(model_path)

    paths = [Path(p) for p in image_paths]
    results_out: list[JunkResult] = []

    for i in range(0, len(paths), batch_size):
        batch = paths[i:i + batch_size]
        preds = model.predict(
            [str(p) for p in batch],
            conf=min_conf,
            device=device,
            verbose=False,
            imgsz=imgsz,
        )

        for path, pred in zip(batch, preds):
            img_h, img_w = pred.orig_shape

            detections: list[Detection] = []
            usable: list[Detection] = []

            for b in pred.boxes:
                cls_idx = int(b.cls.item())
                cls_name = pred.names[cls_idx]
                if cls_name not in VEHICLE_CLASSES:
                    continue
                conf = float(b.conf.item())
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())

                det = Detection(cls=cls_name, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2)

                # Filter tiny detections (background traffic, distant cars)
                if det.area < min_area_frac * img_w * img_h:
                    continue

                detections.append(det)

                if not _is_edge_chopped(det, img_w, img_h, edge_margin_px,
                                        min_visible_frac, edge_min_area_frac):
                    usable.append(det)

            if not detections:
                reason = 'no_vehicle'
                is_junk = True
            elif not usable:
                reason = 'all_edge_chopped'
                is_junk = True
            else:
                reason = 'keep'
                is_junk = False

            results_out.append(JunkResult(
                path=path,
                is_junk=is_junk,
                reason=reason,
                detections=detections,
                usable_detections=usable,
                img_width=img_w,
                img_height=img_h,
            ))

        scanned = min(i + batch_size, len(paths))
        print(
            f"\r  Scanned {scanned}/{len(paths)} images...",
            end="",
            flush=True,
        )
        if progress_cb is not None:
            progress_cb(scanned, len(paths))

    print()
    return results_out


def _companion_xmp(path: Path) -> Optional[Path]:
    """Return an XMP sidecar next to path if one exists, else None."""
    xmp = path.with_suffix('.xmp')
    if xmp.exists():
        return xmp
    xmp = path.with_suffix('.XMP')
    if xmp.exists():
        return xmp
    return None


def _move_with_sidecar(src: Path, dst_dir: Path, dry_run: bool = False):
    """
    Move src into dst_dir. If an .xmp sidecar exists next to src, move it too.
    Returns list of (src, dst) pairs actually moved (or that would be moved).
    """
    moves = []
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / src.name
    moves.append((src, dst))
    if not dry_run:
        shutil.move(str(src), str(dst))

    xmp = _companion_xmp(src)
    if xmp is not None:
        dst_xmp = dst_dir / xmp.name
        moves.append((xmp, dst_xmp))
        if not dry_run:
            shutil.move(str(xmp), str(dst_xmp))

    return moves


def filter_directory(
    input_dir,
    junk_dir_name: str = 'junk',
    model=None,
    model_path: str = DEFAULT_YOLO_MODEL,
    device: Optional[str] = None,
    dry_run: bool = False,
    write_csv: bool = True,
    preview_map: Optional[dict] = None,
    progress_cb=None,
    **detect_kwargs,
) -> dict:
    """
    Scan input_dir for images, move junk frames into input_dir/<junk_dir_name>/.

    For RAW files, previews are extracted to a temp dir for YOLO, then the
    ORIGINAL RAW (plus any .xmp sidecar) is moved into the junk folder.

    Args:
        input_dir: directory to scan (non-recursive).
        junk_dir_name: subfolder name under input_dir for rejected frames.
        model: pre-loaded YOLO instance (optional).
        model_path, device, detect_kwargs: passed through to detect_junk.
        dry_run: show what would move without moving.
        write_csv: write junk_filter.csv with per-frame results.
        preview_map: optional dict {raw_path: preview_path} from caller that
            already extracted previews (avoids redoing the work).

    Returns dict with keys: 'kept', 'junked', 'by_reason', 'results'.
    """
    input_dir = Path(input_dir)
    junk_dir = input_dir / junk_dir_name

    jpg_files, raw_files = find_images(input_dir)

    if not jpg_files and not raw_files:
        print(f"No images found in {input_dir}")
        return {'kept': [], 'junked': [], 'by_reason': {}, 'results': []}

    print(f"Found {len(jpg_files)} JPG/PNG, {len(raw_files)} RAW files")

    # For junk detection we need readable JPEGs. Extract RAW previews on demand.
    # We map: scan_path -> original_path (same for JPEGs, preview -> RAW for RAWs)
    scan_paths: list[Path] = list(jpg_files)
    scan_to_original: dict[Path, Path] = {p: p for p in jpg_files}

    tmp_context = None
    if raw_files:
        if preview_map is None:
            tmp_context = tempfile.TemporaryDirectory()
            tmp_dir = tmp_context.name
            preview_map = extract_raw_previews(raw_files, tmp_dir, workers=default_workers())
        for raw_path, prev_path in preview_map.items():
            scan_paths.append(prev_path)
            scan_to_original[prev_path] = raw_path

    try:
        print(f"\nRunning YOLO junk filter on {len(scan_paths)} images...")
        results = detect_junk(
            scan_paths,
            model=model,
            model_path=model_path,
            device=device,
            progress_cb=progress_cb,
            **detect_kwargs,
        )
    finally:
        if tmp_context is not None:
            tmp_context.cleanup()

    # Remap results from scan_paths back to original files
    for r in results:
        r.path = scan_to_original.get(r.path, r.path)

    kept = [r for r in results if not r.is_junk]
    junked = [r for r in results if r.is_junk]

    by_reason: dict[str, int] = {}
    for r in results:
        by_reason[r.reason] = by_reason.get(r.reason, 0) + 1

    print(f"\nJunk filter: {len(junked)} / {len(results)} flagged as junk")
    for reason, count in sorted(by_reason.items()):
        print(f"  {reason}: {count}")

    # Move junk files
    if junked:
        action = "Would move" if dry_run else "Moving"
        print(f"\n{action} {len(junked)} junk files → {junk_dir}")
        for r in junked:
            _move_with_sidecar(r.path, junk_dir, dry_run=dry_run)

    # CSV log
    if write_csv and results:
        csv_path = input_dir / 'junk_filter.csv'
        if not dry_run:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'filename', 'path', 'is_junk', 'reason',
                    'n_detections', 'n_usable', 'img_w', 'img_h',
                ])
                for r in results:
                    writer.writerow([
                        r.path.name,
                        str(r.path),
                        r.is_junk,
                        r.reason,
                        len(r.detections),
                        len(r.usable_detections),
                        r.img_width,
                        r.img_height,
                    ])
            print(f"Wrote {csv_path}")

    return {
        'kept': [r.path for r in kept],
        'junked': [r.path for r in junked],
        'by_reason': by_reason,
        'results': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Move frames with no usable vehicle into a junk subfolder.",
    )
    parser.add_argument("input_dir", help="Directory of images to filter")
    parser.add_argument(
        "--junk_dir_name",
        default="junk",
        help="Subfolder name for junk frames (default: junk)",
    )
    parser.add_argument(
        "--yolo_model",
        default=DEFAULT_YOLO_MODEL,
        help=f"YOLO weights file (default: {DEFAULT_YOLO_MODEL}, auto-downloads)",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.3,
        help="YOLO detection confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--edge_margin_px",
        type=float,
        default=5.0,
        help="Pixels within image edge to count as 'touching' (default: 5)",
    )
    parser.add_argument(
        "--min_visible_frac",
        type=float,
        default=0.1,
        help="Edge-touching vehicle is 'chopped' if visible dim < this fraction "
             "of image dim along that axis (default: 0.1)",
    )
    parser.add_argument(
        "--min_area_frac",
        type=float,
        default=0.002,
        help="Ignore vehicles smaller than this fraction of image area "
             "(default: 0.002, filters distant traffic)",
    )
    parser.add_argument(
        "--edge_min_area_frac",
        type=float,
        default=0.05,
        help="Edge-touching vehicles must be at least this fraction of image "
             "area to count as 'usable' (default: 0.05). Catches slivers that "
             "pass the width/height gate but are clearly chopped fragments.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="YOLO batch size (default: 32)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device: cpu / mps / cuda (default: auto-detect)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Report what would be moved without actually moving files",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Skip writing junk_filter.csv",
    )

    args = parser.parse_args()

    result = filter_directory(
        args.input_dir,
        junk_dir_name=args.junk_dir_name,
        model_path=args.yolo_model,
        device=args.device,
        dry_run=args.dry_run,
        write_csv=not args.no_csv,
        min_conf=args.min_conf,
        edge_margin_px=args.edge_margin_px,
        min_visible_frac=args.min_visible_frac,
        min_area_frac=args.min_area_frac,
        edge_min_area_frac=args.edge_min_area_frac,
        batch_size=args.batch_size,
    )

    total = len(result['kept']) + len(result['junked'])
    if total:
        pct = 100 * len(result['junked']) / total
        print(f"\nDone. {len(result['junked'])}/{total} ({pct:.1f}%) moved to junk.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

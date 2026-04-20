#!/usr/bin/env python3
"""
classify.py

End-to-end image classification pipeline:
1. Run inference on all images (JPG, PNG, or RAW with preview extraction)
2. Burst dedup: pick best frame per burst
3. Write tiered keywords (embed for JPG, XMP for RAW)

Usage:
    python classify.py /path/to/images
    python classify.py /path/to/images --output_dir ./output
    python classify.py /path/to/images --nef_dir /path/to/nefs  # XMP sidecars for RAWs

For RAW files (NEF, CR2, ARW, etc.), previews are extracted via exiftool.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

from image_utils import (
    RAW_EXTENSIONS,
    JPG_EXTENSIONS,
    ExiftoolProcess,
    extract_raw_previews,
    find_images,
    check_exiftool,
    default_workers,
)


# =============================================================================
# Classifier
# =============================================================================

class ImageClassifier:
    def __init__(self, model_path, device=None):
        """Load trained model."""
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()

        self.model = model
        self.class_names = checkpoint['class_names']
        self.class_to_idx = checkpoint['class_to_idx']

        print(f"Classes: {self.class_names}")
        print(f"Using device: {self.device}")

    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


class ImageDataset(Dataset):
    """Dataset for batch inference."""

    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, idx, True
        except Exception:
            return torch.zeros(3, 224, 224), idx, False


# =============================================================================
# Inference
# =============================================================================

def run_inference(classifier, image_files, batch_size=32, num_workers=4, original_paths=None, progress_cb=None):
    """
    Run batched inference on image files.

    Args:
        classifier: ImageClassifier instance
        image_files: list of paths to classify (may be previews for RAW)
        batch_size: batch size for DataLoader
        num_workers: number of workers for DataLoader
        original_paths: optional dict mapping image_file -> original path
                       (used when classifying extracted previews)

    Returns:
        list of result dicts with original paths
    """
    results = []

    if not image_files:
        return results

    print(f"\nRunning inference on {len(image_files)} images (batch_size={batch_size})...")

    transform = classifier.get_transform()
    dataset = ImageDataset(image_files, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    processed = 0
    with torch.no_grad():
        for batch_tensors, batch_indices, batch_valid in loader:
            batch_tensors = batch_tensors.to(classifier.device)
            logits = classifier.model(batch_tensors)
            probs = F.softmax(logits, dim=1)
            pred_indices = logits.argmax(dim=1)

            for i in range(len(batch_indices)):
                idx = batch_indices[i].item()
                valid = batch_valid[i].item()
                image_path = image_files[idx]

                if not valid:
                    continue

                # Use original path if provided (for RAW files)
                if original_paths and image_path in original_paths:
                    actual_path = original_paths[image_path]
                else:
                    actual_path = image_path

                pred_idx = pred_indices[i].item()
                pred_class = classifier.class_names[pred_idx]
                confidence = probs[i, pred_idx].item()

                result = {
                    'filename': actual_path.name,
                    'path': str(actual_path),
                    'classification': pred_class,
                    'confidence': confidence,
                    'confidence_reject': probs[i, classifier.class_to_idx['reject']].item(),
                    'confidence_select': probs[i, classifier.class_to_idx['select']].item()
                }
                results.append(result)

            processed += len(batch_indices)
            print(f"\r  Processed {processed}/{len(image_files)} images...", end="", flush=True)
            if progress_cb is not None:
                progress_cb(processed, len(image_files))

    print()
    return results


# =============================================================================
# Burst Deduplication
# =============================================================================

def parse_burst_base(filename):
    """
    Extract base frame identifier from filename.
    Only strips SHORT trailing number suffixes (1-2 digits) which indicate burst variants.
    Examples:
        2025-04-25-Z9_BLW0124.jpg -> 2025-04-25-Z9_BLW0124
        2025-04-25-Z9_BLW0124-3.jpg -> 2025-04-25-Z9_BLW0124
        2025-04-25-Z9_BLW0124-3-2.jpg -> 2025-04-25-Z9_BLW0124
        2026-01-31-PCA-Sebring-023573.jpg -> 2026-01-31-PCA-Sebring-023573 (keeps long numbers)
    """
    stem = Path(filename).stem
    # Remove trailing -N suffixes where N is 1-2 digits (burst variants like -2, -3)
    # Don't strip longer numbers (like -023573) which are likely sequence numbers
    match = re.match(r'^(.+?)(-\d{1,2})*$', stem)
    if match:
        return match.group(1)
    return stem


def get_capture_times(file_paths):
    """
    Extract capture time and shutter speed from image files using exiftool.
    Returns dict mapping path -> {'timestamp': float, 'shutter': float}
    """
    if not file_paths:
        return {}

    # Use exiftool JSON output for batch extraction
    paths_str = [str(p) for p in file_paths]
    try:
        result = subprocess.run(
            ['exiftool', '-json', '-DateTimeOriginal', '-SubSecTimeOriginal', '-ExposureTime'] + paths_str,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            return {}

        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError):
        return {}

    capture_times = {}
    for item in data:
        path = Path(item.get('SourceFile', ''))

        # Parse timestamp
        dt_str = item.get('DateTimeOriginal', '')
        subsec = item.get('SubSecTimeOriginal', '0')

        if not dt_str:
            continue

        try:
            # Parse "2026:01:31 11:12:34" format
            dt = datetime.strptime(dt_str, '%Y:%m:%d %H:%M:%S')
            # Add subseconds
            subsec_float = float(f'0.{subsec}') if subsec else 0.0
            timestamp = dt.timestamp() + subsec_float
        except (ValueError, TypeError):
            continue

        # Parse shutter speed (might be "1/1000" or "0.001" or similar)
        shutter = 0.0
        exp_str = item.get('ExposureTime', '')
        if exp_str:
            try:
                if '/' in str(exp_str):
                    num, denom = str(exp_str).split('/')
                    shutter = float(num) / float(denom)
                else:
                    shutter = float(exp_str)
            except (ValueError, TypeError):
                pass

        capture_times[path] = {'timestamp': timestamp, 'shutter': shutter}

    return capture_times


def burst_dedup_by_time(results, capture_times, threshold=0.5):
    """
    Group results into bursts based on capture time proximity.
    Two frames are in the same burst if the gap between them is less than
    max(threshold, shutter_speed + 0.05).

    Args:
        results: list of result dicts with 'path' keys
        capture_times: dict from get_capture_times()
        threshold: base time threshold in seconds (default 0.5s)

    Returns:
        winners: list of winner results (best frame per burst, select only)
        bursts: dict mapping burst_id to list of all frames in that burst
    """
    # Sort results by capture time
    def get_time(r):
        path = Path(r['path'])
        ct = capture_times.get(path, {})
        return ct.get('timestamp', 0)

    sorted_results = sorted(results, key=get_time)

    # Group into bursts
    bursts = defaultdict(list)
    burst_id = 0

    for i, r in enumerate(sorted_results):
        path = Path(r['path'])
        ct = capture_times.get(path, {})
        timestamp = ct.get('timestamp', 0)
        shutter = ct.get('shutter', 0)

        if i == 0:
            bursts[burst_id].append(r)
            continue

        # Check gap to previous frame
        prev_r = sorted_results[i - 1]
        prev_path = Path(prev_r['path'])
        prev_ct = capture_times.get(prev_path, {})
        prev_timestamp = prev_ct.get('timestamp', 0)
        prev_shutter = prev_ct.get('shutter', 0)

        gap = timestamp - prev_timestamp

        # Adaptive threshold: max of base threshold or shutter + buffer
        adaptive_threshold = max(threshold, max(shutter, prev_shutter) + 0.1)

        if gap > adaptive_threshold:
            # New burst
            burst_id += 1

        bursts[burst_id].append(r)

    # Pick best from each burst, only if classified as "select"
    winners = []
    for bid, frames in bursts.items():
        best = max(frames, key=lambda x: x['confidence_select'])
        if best['classification'] == 'select':
            winners.append(best)

    # Convert burst keys to strings for consistency
    bursts_dict = {f'burst_{k}': v for k, v in bursts.items()}

    return winners, bursts_dict


def burst_dedup(results, capture_times=None, time_threshold=None):
    """
    Group results by burst, pick the one with highest confidence_select.
    Only keeps winners that are classified as "select".

    If time_threshold is provided and capture_times available, uses time-based
    grouping. Otherwise falls back to filename-based grouping.

    Returns:
        winners: list of winner results (best frame per burst, select only)
        bursts: dict mapping burst base to list of all frames in that burst
    """
    # Use time-based grouping if threshold specified and we have capture times
    if time_threshold is not None and time_threshold > 0 and capture_times:
        return burst_dedup_by_time(results, capture_times, time_threshold)

    # Fall back to filename-based grouping
    bursts = defaultdict(list)
    for r in results:
        base = parse_burst_base(r['filename'])
        bursts[base].append(r)

    # Pick best from each burst, only if classified as "select"
    winners = []
    for base, frames in bursts.items():
        best = max(frames, key=lambda x: x['confidence_select'])
        if best['classification'] == 'select':
            winners.append(best)

    return winners, dict(bursts)


# =============================================================================
# Keyword Writing
# =============================================================================

def get_tier_keyword(confidence):
    """Return tier keyword based on confidence, or None if below threshold."""
    # Tiers from 90-99 (each 1% increment)
    for threshold in range(99, 89, -1):  # 99, 98, 97, ... 90
        if confidence >= threshold / 100.0:
            return f"robo_{threshold}"
    return None


def write_xmp_sidecar(target_path, keyword):
    """
    Write keyword to XMP sidecar for RAW file.
    Uses exiftool to create/update sidecar, preserving existing metadata.
    """
    target_path = Path(target_path)
    xmp_path = target_path.with_suffix('.xmp')

    try:
        # If XMP exists, update it; otherwise create from RAW metadata
        if xmp_path.exists():
            # Update existing XMP sidecar
            result = subprocess.run(
                [
                    'exiftool',
                    '-overwrite_original',
                    f'-Keywords+={keyword}',
                    f'-Subject+={keyword}',
                    f'-HierarchicalSubject+=robo|{keyword}',
                    str(xmp_path)
                ],
                capture_output=True,
                text=True
            )
        else:
            # Create new XMP sidecar from RAW, then add keyword
            result = subprocess.run(
                [
                    'exiftool',
                    '-tagsfromfile', str(target_path),
                    f'-Keywords+={keyword}',
                    f'-Subject+={keyword}',
                    f'-HierarchicalSubject+=robo|{keyword}',
                    '-o', str(xmp_path)
                ],
                capture_output=True,
                text=True
            )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def embed_keyword_in_jpeg(jpeg_path, keyword):
    """Embed keyword directly into JPEG using exiftool."""
    # Determine hierarchy based on keyword type
    if keyword.startswith('robo_'):
        hierarchy = f"AI keywords|robo|{keyword}"
    else:  # select
        hierarchy = f"AI keywords|{keyword}"

    try:
        result = subprocess.run(
            [
                'exiftool',
                '-overwrite_original',
                f'-Keywords+={keyword}',
                f'-Subject+={keyword}',
                f'-HierarchicalSubject+={hierarchy}',
                str(jpeg_path)
            ],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("ERROR: exiftool not found. Install with: brew install exiftool")
        return False


def write_keyword_to_file(target_path, keyword, nef_dir=None):
    """
    Write a single keyword to a file.
    Returns True on success, False on error.
    """
    source_path = Path(target_path)
    stem = source_path.stem

    if nef_dir:
        # Write XMP sidecar next to NEF
        nef_dir_path = Path(nef_dir)
        target = nef_dir_path / f"{stem}.NEF"
        if not target.exists():
            target = nef_dir_path / f"{stem}.nef"
        if not target.exists():
            return False
        return write_xmp_sidecar(target, keyword)
    else:
        # Write to source file
        target = source_path
        if target.suffix.lower() in RAW_EXTENSIONS:
            return write_xmp_sidecar(target, keyword)
        else:
            return embed_keyword_in_jpeg(target, keyword)


def clear_robo_keywords(target_path, nef_dir=None):
    """
    Remove any previously-written 'select' + 'robo_9x' tags from a file
    (or its XMP sidecar). Called before re-writing keywords so tier changes
    after threshold tuning don't leave stale tags behind.

    Handles both the old (`robo|robo_9x`) and new (`AI keywords|robo|robo_9x`)
    hierarchical subject formats.
    """
    source_path = Path(target_path)
    stem = source_path.stem

    if nef_dir:
        nef_dir_path = Path(nef_dir)
        target = nef_dir_path / f"{stem}.NEF"
        if not target.exists():
            target = nef_dir_path / f"{stem}.nef"
        if not target.exists():
            return False
        target_file = target.with_suffix('.xmp')
        if not target_file.exists():
            return True  # nothing to clear
    elif source_path.suffix.lower() in RAW_EXTENSIONS:
        target_file = source_path.with_suffix('.xmp')
        if not target_file.exists():
            return True
    else:
        target_file = source_path

    tags = ['select'] + [f'robo_{i}' for i in range(90, 100)]
    args = ['exiftool', '-overwrite_original']
    for t in tags:
        args += [f'-Keywords-={t}', f'-Subject-={t}']
    for t in tags:
        if t == 'select':
            args += [
                '-HierarchicalSubject-=AI keywords|select',
                '-HierarchicalSubject-=robo|select',
            ]
        else:
            args += [
                f'-HierarchicalSubject-=AI keywords|robo|{t}',
                f'-HierarchicalSubject-=robo|{t}',
            ]
    args.append(str(target_file))

    try:
        result = subprocess.run(args, capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def write_keywords(winners, bursts, nef_dir=None):
    """
    Write tiered keywords to winner images and 'select' to all burst siblings.
    - Winners >= 0.90 get robo_90 through robo_99 keyword (1% increments)
    - All frames in bursts with a winner >= 0.90 get 'select' keyword

    Args:
        winners: list of winner results (best frame per burst)
        bursts: dict mapping burst key to list of all frames
        nef_dir: optional directory for NEF files (writes XMP sidecars there)
    """
    tier_counts = {f"robo_{i}": 0 for i in range(90, 100)}
    tier_counts["below_threshold"] = 0
    winner_written = 0
    select_written = 0
    errors = 0

    # Build reverse lookup: path -> burst_key
    path_to_burst = {}
    for burst_key, frames in bursts.items():
        for frame in frames:
            path_to_burst[frame['path']] = burst_key

    # Track which bursts have winners above threshold
    qualifying_bursts = set()

    # First pass: write robo_9x to winners above threshold
    for row in winners:
        confidence = row['confidence_select']
        keyword = get_tier_keyword(confidence)

        if keyword is None:
            tier_counts["below_threshold"] += 1
            continue

        tier_counts[keyword] += 1
        burst_key = path_to_burst.get(row['path'])
        if burst_key:
            qualifying_bursts.add(burst_key)

        if write_keyword_to_file(row['path'], keyword, nef_dir):
            winner_written += 1
        else:
            errors += 1

    # Second pass: write 'select' to all frames in qualifying bursts
    for burst_key in qualifying_bursts:
        for frame in bursts[burst_key]:
            if write_keyword_to_file(frame['path'], 'select', nef_dir):
                select_written += 1
            else:
                errors += 1

    return tier_counts, winner_written, select_written, errors


# =============================================================================
# Main Pipeline
# =============================================================================

MODELS_DIR = Path(__file__).parent / "models"


def resolve_model(args_model, args_profile):
    """
    Figure out which model .pt to load and return its path.

    Resolution order:
      1. --model <explicit path> wins if provided and exists.
      2. --profile <name> → models/<name>.pt (prints sidecar JSON if present).
      3. Default model.pt in CWD (legacy).
    """
    if args_profile:
        pt = MODELS_DIR / f"{args_profile}.pt"
        if not pt.exists():
            available = sorted(p.stem for p in MODELS_DIR.glob("*.pt")) if MODELS_DIR.exists() else []
            raise SystemExit(
                f"Profile '{args_profile}' not found at {pt}. "
                f"Available: {available or '(none — put models in models/)'}"
            )
        sidecar = pt.with_suffix(".json")
        if sidecar.exists():
            try:
                meta = json.loads(sidecar.read_text())
                print(f"Profile '{args_profile}': {meta.get('description', '(no description)')}")
            except Exception:
                pass
        return pt

    return Path(args_model)


def _noop_progress(event):
    pass


def run_pipeline(
    input_dir,
    model="model.pt",
    profile=None,
    output_dir=None,
    nef_dir=None,
    batch_size=32,
    num_workers=4,
    no_keywords=False,
    dry_run=False,
    burst_threshold=None,
    skip_junk_filter=False,
    junk_dir_name="junk",
    yolo_model="yolov8n.pt",
    junk_min_conf=0.3,
    junk_min_visible_frac=0.1,
    junk_min_area_frac=0.002,
    junk_edge_min_area_frac=0.05,
    preview_workers=None,
    pregen_thumbs=False,
    progress_cb=None,
):
    """
    Run the full classification pipeline as a library call.

    progress_cb(event) receives dicts like:
      {"type": "stage", "stage": "junk"|"inference"|"dedup"|"keywords",
       "message": str}
      {"type": "progress", "stage": str, "current": int, "total": int}
      {"type": "done", "summary": {...}}

    Returns a summary dict with counts, winners, bursts, etc.
    Raises on fatal errors (missing exiftool, missing model, no images).
    """
    if progress_cb is None:
        progress_cb = _noop_progress

    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir
    pw = preview_workers if preview_workers is not None else default_workers()

    if not check_exiftool():
        raise RuntimeError("exiftool not found. Install with: brew install exiftool")

    model_path = resolve_model(model, profile)

    # Find images up-front so we can extract previews once and share across stages.
    jpg_files, raw_files = find_images(input_dir)
    print(f"Found {len(jpg_files)} JPG/PNG and {len(raw_files)} RAW files")
    if not jpg_files and not raw_files:
        raise RuntimeError("No images found.")

    # Single tempdir spans both junk filter and classification so we never
    # extract RAW previews twice.
    with tempfile.TemporaryDirectory() as preview_dir:
        preview_map: dict = {}
        if raw_files:
            progress_cb({"type": "stage", "stage": "previews",
                         "message": f"Extracting RAW previews ({pw} workers)"})
            preview_map = extract_raw_previews(
                raw_files, preview_dir, workers=pw,
                progress_cb=lambda c, t: progress_cb(
                    {"type": "progress", "stage": "previews", "current": c, "total": t}
                ),
            )

        # Step 0: Junk filter (reuses the previews we just extracted)
        junk_summary = None
        if not skip_junk_filter:
            print(f"\n=== Step 0: Junk filter (YOLO vehicle detection) ===")
            progress_cb({"type": "stage", "stage": "junk", "message": "Junk filter"})
            from junk_filter import filter_directory
            junk_summary = filter_directory(
                input_dir,
                junk_dir_name=junk_dir_name,
                model_path=yolo_model,
                dry_run=dry_run,
                min_conf=junk_min_conf,
                min_visible_frac=junk_min_visible_frac,
                min_area_frac=junk_min_area_frac,
                edge_min_area_frac=junk_edge_min_area_frac,
                preview_map=preview_map if preview_map else None,
                progress_cb=lambda c, t: progress_cb(
                    {"type": "progress", "stage": "junk", "current": c, "total": t}
                ),
            )
            # Filter preview_map + file lists to survivors (junked files moved).
            if not dry_run:
                preview_map = {r: p for r, p in preview_map.items() if r.exists()}
                jpg_files = [f for f in jpg_files if f.exists()]
                raw_files = [f for f in raw_files if f.exists()]

        total_files = len(jpg_files) + len(raw_files)
        if total_files == 0:
            raise RuntimeError("No images found after junk filter.")

        # Step 2: Inference
        print(f"\n=== Step 2: Running inference ===")
        progress_cb({"type": "stage", "stage": "inference", "message": "Classifying"})
        classifier = ImageClassifier(model_path)

        results = []
        n_total_for_progress = total_files
        n_done = 0

        def _inference_progress_adapter(current, total):
            progress_cb({
                "type": "progress",
                "stage": "inference",
                "current": n_done + current,
                "total": n_total_for_progress,
            })

        if jpg_files:
            jpg_results = run_inference(
                classifier, jpg_files, batch_size, num_workers,
                progress_cb=_inference_progress_adapter,
            )
            results.extend(jpg_results)
            n_done += len(jpg_files)

        if raw_files and preview_map:
            preview_to_original = {v: k for k, v in preview_map.items()}
            preview_files = list(preview_map.values())
            raw_results = run_inference(
                classifier, preview_files, batch_size, num_workers,
                original_paths=preview_to_original,
                progress_cb=_inference_progress_adapter,
            )
            results.extend(raw_results)

        # Step 2b: Pregenerate UI thumbnails while previews are still on disk.
        # (Runs inside the preview_dir context so preview files are still available.)
        if pregen_thumbs and not dry_run and results:
            print(f"\n=== Step 2b: Pregenerating thumbnails ({pw} workers) ===")
            progress_cb({"type": "stage", "stage": "thumbs",
                         "message": f"Pregenerating thumbs ({pw} workers)"})
            from ui.thumbs import pregenerate_thumbs
            sources = [Path(r['path']) for r in results]
            pregenerate_thumbs(
                input_dir, sources,
                preview_map=preview_map,
                workers=pw,
                progress_cb=lambda c, t: progress_cb(
                    {"type": "progress", "stage": "thumbs", "current": c, "total": t}
                ),
            )

    if not results:
        raise RuntimeError("No images could be processed.")

    # Write results CSV
    results_csv = output_dir / "results.csv"
    if not dry_run:
        print(f"\nWriting full results to {results_csv}")
        with open(results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Step 3: Burst dedup
    print(f"\n=== Step 3: Burst deduplication ===")
    progress_cb({"type": "stage", "stage": "dedup", "message": "Deduping bursts"})

    capture_times = {}
    if burst_threshold is not None and burst_threshold > 0:
        all_paths = [Path(r['path']) for r in results]
        capture_times = get_capture_times(all_paths)

    winners, bursts = burst_dedup(results, capture_times, burst_threshold)
    print(f"Reduced {len(results)} images to {len(winners)} burst winners")

    winners_csv = output_dir / "winners.csv"
    if not dry_run and winners:
        with open(winners_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=winners[0].keys())
            writer.writeheader()
            writer.writerows(winners)

    # Step 4: Write keywords
    tier_counts = {f"robo_{i}": 0 for i in range(90, 100)}
    tier_counts["below_threshold"] = 0
    select_count = 0
    if not no_keywords and winners:
        print(f"\n=== Step 4: Writing tiered keywords ===")
        progress_cb({"type": "stage", "stage": "keywords", "message": "Writing keywords"})
        if dry_run:
            path_to_burst = {
                frame['path']: burst_key
                for burst_key, frames in bursts.items()
                for frame in frames
            }
            qualifying_bursts = set()
            for w in winners:
                kw = get_tier_keyword(w['confidence_select'])
                if kw:
                    tier_counts[kw] += 1
                    bk = path_to_burst.get(w['path'])
                    if bk:
                        qualifying_bursts.add(bk)
                else:
                    tier_counts["below_threshold"] += 1
            select_count = sum(len(bursts[b]) for b in qualifying_bursts)
        else:
            tier_counts, _winner_written, select_written, _errors = write_keywords(
                winners, bursts, nef_dir
            )
            select_count = select_written

    summary = {
        "input_dir": str(input_dir),
        "total_images": len(results),
        "selects": sum(1 for r in results if r['classification'] == 'select'),
        "winners": len(winners),
        "tier_counts": tier_counts,
        "select_siblings": select_count,
        # Keep only the JSON-safe scalar counts — the full junk_summary contains
        # Path/JunkResult objects that can't be serialized onto the SSE stream.
        "junk": (
            {
                "by_reason": junk_summary.get("by_reason", {}),
                "n_junk": len(junk_summary.get("junked", [])),
                "n_kept": len(junk_summary.get("kept", [])),
            }
            if junk_summary else None
        ),
        "results_csv": str(results_csv),
        "winners_csv": str(winners_csv) if winners else None,
    }
    progress_cb({"type": "done", "summary": summary})
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Classify images, dedupe bursts, write tiered keywords"
    )
    parser.add_argument(
        "input_dir",
        help="Directory containing images to classify"
    )
    parser.add_argument(
        "--model",
        default="model.pt",
        help="Path to trained model (default: model.pt)"
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Model profile name — resolves to models/<name>.pt. "
             "Takes precedence over --model when specified."
    )
    parser.add_argument(
        "--output_dir",
        help="Directory to write CSV results (default: input_dir)"
    )
    parser.add_argument(
        "--nef_dir",
        help="Directory containing NEF files (writes XMP sidecars there instead of embedding in JPEGs)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)"
    )
    parser.add_argument(
        "--no_keywords",
        action="store_true",
        help="Skip writing keywords (just output CSVs)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't write anything, just show what would happen"
    )
    parser.add_argument(
        "--burst_threshold",
        type=float,
        default=None,
        help="Time-based burst grouping threshold in seconds (e.g., 0.5). "
             "Frames captured within this time window are grouped together. "
             "Adapts to shutter speed (slow shutters get larger thresholds). "
             "Default: disabled (uses filename-based grouping)"
    )

    # Junk filter options
    parser.add_argument(
        "--skip_junk_filter",
        action="store_true",
        help="Skip the YOLO-based junk filter step (no-car / chopped-car detection)"
    )
    parser.add_argument(
        "--junk_dir_name",
        default="junk",
        help="Subfolder for junk frames (default: junk)"
    )
    parser.add_argument(
        "--yolo_model",
        default="yolov8n.pt",
        help="YOLO weights (default: yolov8n.pt — auto-downloads on first use)"
    )
    parser.add_argument(
        "--junk_min_conf",
        type=float,
        default=0.3,
        help="YOLO detection confidence threshold (default: 0.3)"
    )
    parser.add_argument(
        "--junk_min_visible_frac",
        type=float,
        default=0.1,
        help="Edge-touching vehicle 'chopped' if visible dim < this fraction "
             "of image dim (default: 0.1)"
    )
    parser.add_argument(
        "--junk_min_area_frac",
        type=float,
        default=0.002,
        help="Ignore vehicles smaller than this fraction of image area "
             "(default: 0.002)"
    )
    parser.add_argument(
        "--junk_edge_min_area_frac",
        type=float,
        default=0.05,
        help="Edge-touching vehicles must be at least this fraction of image "
             "area to be considered usable (default: 0.05)"
    )

    args = parser.parse_args()

    try:
        summary = run_pipeline(
            input_dir=args.input_dir,
            model=args.model,
            profile=args.profile,
            output_dir=args.output_dir,
            nef_dir=args.nef_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            no_keywords=args.no_keywords,
            dry_run=args.dry_run,
            burst_threshold=args.burst_threshold,
            skip_junk_filter=args.skip_junk_filter,
            junk_dir_name=args.junk_dir_name,
            yolo_model=args.yolo_model,
            junk_min_conf=args.junk_min_conf,
            junk_min_visible_frac=args.junk_min_visible_frac,
            junk_min_area_frac=args.junk_min_area_frac,
            junk_edge_min_area_frac=args.junk_edge_min_area_frac,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    total = summary['total_images']
    selects = summary['selects']
    print(f"\n{'=' * 50}\nSUMMARY\n{'=' * 50}")
    print(f"Total images:      {total}")
    if total:
        print(f"Classified select: {selects} ({100*selects/total:.1f}%)")
        print(f"Classified reject: {total - selects} ({100*(total-selects)/total:.1f}%)")
    print(f"Burst winners:     {summary['winners']}")

    tc = summary['tier_counts']
    if summary['winners'] and not args.no_keywords:
        print(f"\nKeyword tiers:")
        for threshold in range(99, 89, -1):
            key = f"robo_{threshold}"
            if tc.get(key, 0):
                print(f"  {key} (>=0.{threshold}): {tc[key]}")
        print(f"  Below threshold: {tc['below_threshold']}")
        print(f"\nBurst siblings tagged 'select': {summary['select_siblings']}")

    print(f"\nOutput files:")
    print(f"  {summary['results_csv']}")
    if summary['winners_csv']:
        print(f"  {summary['winners_csv']}")

    return 0


if __name__ == "__main__":
    exit(main())

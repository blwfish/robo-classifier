#!/usr/bin/env python3
"""
classify.py

End-to-end image classification pipeline:
1. Run inference on all images
2. Burst dedup: pick best frame per burst
3. Write tiered keywords (embed for JPG, XMP for RAW)

Usage:
    python classify.py /path/to/images
    python classify.py /path/to/images --output_dir ./output
    python classify.py /path/to/images --nef_dir /path/to/nefs  # XMP sidecars for RAWs
"""

import argparse
import csv
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image


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

def run_inference(classifier, image_files, batch_size=32, num_workers=4):
    """Run batched inference on image files."""
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

                pred_idx = pred_indices[i].item()
                pred_class = classifier.class_names[pred_idx]
                confidence = probs[i, pred_idx].item()

                result = {
                    'filename': image_path.name,
                    'path': str(image_path),
                    'classification': pred_class,
                    'confidence': confidence,
                    'confidence_reject': probs[i, classifier.class_to_idx['reject']].item(),
                    'confidence_select': probs[i, classifier.class_to_idx['select']].item()
                }
                results.append(result)

            processed += len(batch_indices)
            print(f"\r  Processed {processed}/{len(image_files)} images...", end="", flush=True)

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


def burst_dedup(results):
    """
    Group results by burst base, pick the one with highest confidence_select.
    Only keeps winners that are classified as "select".

    Returns:
        winners: list of winner results (best frame per burst, select only)
        bursts: dict mapping burst base to list of all frames in that burst
    """
    # Group by burst base
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

    return winners, bursts


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
    """Write XMP sidecar with tiered keyword (for RAW files)."""
    xmp_path = target_path.with_suffix(".xmp")

    # Determine hierarchy based on keyword type
    if keyword.startswith('robo_'):
        hierarchy = f"AI keywords|robo|{keyword}"
    else:  # select
        hierarchy = f"AI keywords|{keyword}"

    xmp_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:lr="http://ns.adobe.com/lightroom/1.0/">
  <rdf:RDF>
    <rdf:Description rdf:about="">
      <dc:subject>
        <rdf:Bag>
          <rdf:li>{keyword}</rdf:li>
        </rdf:Bag>
      </dc:subject>
      <lr:hierarchicalSubject>
        <rdf:Bag>
          <rdf:li>{hierarchy}</rdf:li>
        </rdf:Bag>
      </lr:hierarchicalSubject>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""

    with open(xmp_path, 'w') as f:
        f.write(xmp_content)
    return True


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
        is_raw = target.suffix.lower() in ['.nef', '.cr2', '.cr3', '.arw', '.orf', '.raf', '.dng']
        if is_raw:
            return write_xmp_sidecar(target, keyword)
        else:
            return embed_keyword_in_jpeg(target, keyword)


def write_keywords(winners, bursts, nef_dir=None):
    """
    Write tiered keywords to winner images and 'select' to all burst siblings.
    - Winners >= 0.90 get robo_90 through robo_99 keyword (1% increments)
    - All frames in bursts with a winner >= 0.90 get 'select' keyword

    Args:
        winners: list of winner results (best frame per burst)
        bursts: dict mapping burst base to list of all frames
        nef_dir: optional directory for NEF files (writes XMP sidecars there)
    """
    tier_counts = {f"robo_{i}": 0 for i in range(90, 100)}
    tier_counts["below_threshold"] = 0
    winner_written = 0
    select_written = 0
    errors = 0

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
        base = parse_burst_base(row['filename'])
        qualifying_bursts.add(base)

        if write_keyword_to_file(row['path'], keyword, nef_dir):
            winner_written += 1
        else:
            errors += 1

    # Second pass: write 'select' to all frames in qualifying bursts
    for base in qualifying_bursts:
        for frame in bursts[base]:
            if write_keyword_to_file(frame['path'], 'select', nef_dir):
                select_written += 1
            else:
                errors += 1

    return tier_counts, winner_written, select_written, errors


# =============================================================================
# Main Pipeline
# =============================================================================

def find_images(input_dir):
    """Find all image files in directory and subdirectories."""
    input_dir = Path(input_dir)
    image_files = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        # Use rglob to recursively search subdirectories
        image_files.extend(input_dir.rglob(ext))
    # Remove duplicates (Windows glob is case-insensitive)
    return sorted(list(set(image_files)))


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

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    # Check for exiftool if we'll need it
    if not args.no_keywords and not args.nef_dir and not args.dry_run:
        try:
            subprocess.run(['exiftool', '-ver'], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("ERROR: exiftool not found. Install with: brew install exiftool")
            print("       Or use --nef_dir to write XMP sidecars instead.")
            return 1

    # Step 1: Find images
    print(f"\n=== Step 1: Finding images in {input_dir} ===")
    image_files = find_images(input_dir)
    print(f"Found {len(image_files)} images")

    if not image_files:
        print("No images found. Exiting.")
        return 1

    # Step 2: Run inference
    print(f"\n=== Step 2: Running inference ===")
    classifier = ImageClassifier(args.model)
    results = run_inference(classifier, image_files, args.batch_size, args.num_workers)

    # Write full results CSV
    results_csv = output_dir / "results.csv"
    if not args.dry_run:
        print(f"\nWriting full results to {results_csv}")
        with open(results_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # Step 3: Burst dedup
    print(f"\n=== Step 3: Burst deduplication ===")
    winners, bursts = burst_dedup(results)
    print(f"Reduced {len(results)} images to {len(winners)} burst winners (select only)")

    # Write winners CSV
    winners_csv = output_dir / "winners.csv"
    if not args.dry_run:
        print(f"Writing winners to {winners_csv}")
        with open(winners_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=winners[0].keys())
            writer.writeheader()
            writer.writerows(winners)

    # Step 4: Write keywords
    if not args.no_keywords:
        print(f"\n=== Step 4: Writing tiered keywords ===")
        if args.dry_run:
            tier_counts = {f"robo_{i}": 0 for i in range(90, 100)}
            tier_counts["below_threshold"] = 0
            qualifying_bursts = set()
            for w in winners:
                kw = get_tier_keyword(w['confidence_select'])
                if kw:
                    tier_counts[kw] += 1
                    qualifying_bursts.add(parse_burst_base(w['filename']))
                else:
                    tier_counts["below_threshold"] += 1
            select_count = sum(len(bursts[b]) for b in qualifying_bursts)
            winner_written, select_written, errors = 0, 0, 0
            print("(dry run - no keywords written)")
        else:
            tier_counts, winner_written, select_written, errors = write_keywords(winners, bursts, args.nef_dir)
            select_count = select_written
            print(f"Wrote {winner_written} robo_9x keywords, {select_written} select keywords ({errors} errors)")

    # Summary
    print(f"\n{'=' * 50}")
    print("SUMMARY")
    print('=' * 50)
    total = len(results)
    selects = sum(1 for r in results if r['classification'] == 'select')
    print(f"Total images:      {total}")
    print(f"Classified select: {selects} ({100*selects/total:.1f}%)")
    print(f"Classified reject: {total - selects} ({100*(total-selects)/total:.1f}%)")
    print(f"Burst winners:     {len(winners)}")

    if not args.no_keywords:
        print(f"\nKeyword tiers (from {len(winners)} winners):")
        for threshold in range(99, 89, -1):  # 99 down to 90
            key = f"robo_{threshold}"
            count = tier_counts.get(key, 0)
            if count > 0:  # Only show tiers with results
                print(f"  {key} (>=0.{threshold}): {count}")
        print(f"  Below threshold: {tier_counts['below_threshold']}")
        print(f"\nBurst siblings tagged 'select': {select_count}")

    print(f"\nOutput files:")
    print(f"  {results_csv}")
    print(f"  {winners_csv}")

    return 0


if __name__ == "__main__":
    exit(main())

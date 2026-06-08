#!/usr/bin/env python3
"""
write_tiered_keywords.py

Reads a winners CSV and writes tiered keywords based on confidence.
- For RAW files: writes XMP sidecars via exiftool (preserves existing metadata)
- For JPEGs: embeds keywords directly using exiftool

Tiers: robo_90 (>=0.90) through robo_99 (>=0.99), matching classify.py.

Usage:
    python write_tiered_keywords.py --winners results.csv
    python write_tiered_keywords.py --winners results.csv --nef_dir /path/to/nefs
"""

import argparse
import csv
import subprocess
from pathlib import Path

from image_utils import RAW_EXTENSIONS
from classify import get_tier_keyword, write_xmp_sidecar, embed_keyword_in_jpeg


def main():
    parser = argparse.ArgumentParser(
        description="Write tiered keywords from winners CSV"
    )
    parser.add_argument(
        "--winners",
        required=True,
        help="Path to winners CSV (from burst deduplication)"
    )
    parser.add_argument(
        "--nef_dir",
        help="Directory containing NEF files (writes XMP sidecars there)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be written without writing"
    )

    args = parser.parse_args()

    # Check for exiftool
    try:
        subprocess.run(['exiftool', '-ver'], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: exiftool not found. Install with: brew install exiftool")
        return

    # Read winners
    with open(args.winners) as f:
        winners = list(csv.DictReader(f))

    print(f"Read {len(winners)} winners from {args.winners}")

    tier_counts = {f"robo_{i}": 0 for i in range(90, 100)}
    tier_counts["below_threshold"] = 0
    written = 0
    errors = 0

    for i, row in enumerate(winners, 1):
        confidence = float(row['confidence_select'])
        keyword = get_tier_keyword(confidence)

        if keyword is None:
            tier_counts["below_threshold"] += 1
            continue

        tier_counts[keyword] += 1

        source_path = Path(row['path'])
        stem = source_path.stem

        if args.nef_dir:
            nef_dir = Path(args.nef_dir)
            target = None
            for ext in RAW_EXTENSIONS:
                candidate = nef_dir / f"{stem}{ext}"
                if candidate.exists():
                    target = candidate
                    break
            if target is None:
                print(f"  WARNING: No RAW file found for {stem}")
                errors += 1
                continue
            use_xmp = True
        else:
            target = source_path
            use_xmp = target.suffix.lower() in RAW_EXTENSIONS

        if args.dry_run:
            method = "XMP sidecar" if use_xmp else "embed in JPEG"
            print(f"  [{i}/{len(winners)}] Would write {keyword} to {target.name} ({method})")
        else:
            if use_xmp:
                success = write_xmp_sidecar(target, keyword)
            else:
                success = embed_keyword_in_jpeg(target, keyword)

            if success:
                written += 1
            else:
                errors += 1

            if i % 100 == 0:
                print(f"  Processed {i}/{len(winners)}...")

    print(f"\n=== SUMMARY ===")
    for threshold in range(99, 89, -1):
        key = f"robo_{threshold}"
        if tier_counts[key]:
            print(f"{key} (>={threshold/100:.2f}): {tier_counts[key]}")
    print(f"Below threshold:  {tier_counts['below_threshold']}")

    if not args.dry_run:
        print(f"\nWrote {written} keywords")
        if errors:
            print(f"Errors: {errors}")
    else:
        print(f"\n(dry run - no files written)")


if __name__ == "__main__":
    main()

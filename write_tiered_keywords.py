#!/usr/bin/env python3
"""
write_tiered_keywords.py

Reads a winners CSV and writes tiered keywords based on confidence.
- For RAW files: writes XMP sidecars via exiftool (preserves existing metadata)
- For JPEGs: embeds keywords directly using exiftool

Tiers: robo_90 (>=0.90) through robo_99 (>=0.99), matching classify.py.

If --model is given, the model's JSON sidecar is read for accept_keyword and
reject_keyword.  accept_keyword is written alongside the tier for each winner.
If --results_csv is also given, reject_keyword is written to all non-winning
images in that CSV.

Usage:
    python write_tiered_keywords.py --winners winners.csv
    python write_tiered_keywords.py --winners winners.csv --nef_dir /path/to/nefs
    python write_tiered_keywords.py --winners winners.csv --model pca \\
        --results_csv results.csv
"""

import argparse
import csv
import subprocess
from pathlib import Path

from image_utils import RAW_EXTENSIONS
from classify import (
    get_tier_keyword, load_model_keywords,
    write_xmp_sidecar, embed_keyword_in_jpeg, write_keyword_to_file,
    MODELS_DIR,
)


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
        "--model",
        help="Model name (looks up <name>.pt in model library) or path to .pt file. "
             "Loads accept_keyword / reject_keyword from its JSON sidecar."
    )
    parser.add_argument(
        "--results_csv",
        help="Full results CSV (results.csv). If --model has a reject_keyword, "
             "it is written to all images not in the winners set."
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

    # Load model keywords from sidecar
    model_kws = {"accept_keyword": None, "reject_keyword": None}
    if args.model:
        model_arg = Path(args.model)
        if model_arg.suffix == ".pt" and model_arg.exists():
            pt_path = model_arg
        else:
            pt_path = MODELS_DIR / f"{args.model}.pt"
        model_kws = load_model_keywords(pt_path)
        if model_kws["accept_keyword"]:
            print(f"Model accept keyword: {model_kws['accept_keyword']}")
        if model_kws["reject_keyword"]:
            print(f"Model reject keyword: {model_kws['reject_keyword']}")

    accept_kw = model_kws["accept_keyword"]
    reject_kw = model_kws["reject_keyword"]

    # Read winners
    with open(args.winners) as f:
        winners = list(csv.DictReader(f))

    print(f"Read {len(winners)} winners from {args.winners}")
    winner_paths = {row['path'] for row in winners}

    tier_counts = {f"robo_{i}": 0 for i in range(90, 100)}
    tier_counts["below_threshold"] = 0
    written = 0
    reject_written = 0
    errors = 0

    for i, row in enumerate(winners, 1):
        confidence = float(row['confidence_select'])
        keyword = get_tier_keyword(confidence)

        if keyword is None:
            tier_counts["below_threshold"] += 1
            continue

        tier_counts[keyword] += 1
        source_path = Path(row['path'])

        if args.dry_run:
            method = "XMP sidecar" if source_path.suffix.lower() in RAW_EXTENSIONS else "embed"
            extra = f" + {accept_kw}" if accept_kw else ""
            print(f"  [{i}/{len(winners)}] Would write {keyword}{extra} to {source_path.name} ({method})")
        else:
            if write_keyword_to_file(row['path'], keyword, args.nef_dir):
                written += 1
            else:
                errors += 1
            if accept_kw:
                if not write_keyword_to_file(row['path'], accept_kw, args.nef_dir):
                    errors += 1

        if i % 100 == 0 and not args.dry_run:
            print(f"  Processed {i}/{len(winners)}...")

    # Write reject keyword to non-winners if model defines one and results provided
    if reject_kw and args.results_csv:
        with open(args.results_csv) as f:
            all_results = list(csv.DictReader(f))
        losers = [
            r for r in all_results
            if r['path'] not in winner_paths
            and r.get('classification') != 'decode_failed'
        ]
        print(f"\nWriting reject keyword '{reject_kw}' to {len(losers)} non-winners...")
        for i, row in enumerate(losers, 1):
            if args.dry_run:
                print(f"  Would write {reject_kw} to {Path(row['path']).name}")
            else:
                if write_keyword_to_file(row['path'], reject_kw, args.nef_dir):
                    reject_written += 1
                else:
                    errors += 1
            if i % 100 == 0 and not args.dry_run:
                print(f"  Processed {i}/{len(losers)} rejects...")

    print(f"\n=== SUMMARY ===")
    for threshold in range(99, 89, -1):
        key = f"robo_{threshold}"
        if tier_counts[key]:
            print(f"{key} (>={threshold/100:.2f}): {tier_counts[key]}")
    print(f"Below threshold:  {tier_counts['below_threshold']}")

    if not args.dry_run:
        print(f"\nWrote {written} tier keywords")
        if accept_kw:
            print(f"Wrote accept keyword ({accept_kw}) to qualifying winners")
        if reject_kw and args.results_csv:
            print(f"Wrote reject keyword ({reject_kw}) to {reject_written} non-winners")
        if errors:
            print(f"Errors: {errors}")
    else:
        print(f"\n(dry run - no files written)")


if __name__ == "__main__":
    main()

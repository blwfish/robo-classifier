#!/usr/bin/env python3
"""
write_tiered_keywords.py

Reads a winners CSV and writes tiered keywords based on confidence.
- For RAW files (NEF): writes XMP sidecars
- For JPEGs: embeds keywords directly using exiftool

Tiers:
  - robo_99: confidence >= 0.99
  - robo_98: confidence >= 0.98
  - robo_97: confidence >= 0.97

Usage:
    python write_tiered_keywords.py --winners results.csv
    python write_tiered_keywords.py --winners results.csv --nef_dir /path/to/nefs
"""

import argparse
import csv
import subprocess
from pathlib import Path


def get_tier_keyword(confidence):
    """Return tier keyword based on confidence, or None if below threshold."""
    if confidence >= 0.99:
        return "robo_99"
    elif confidence >= 0.98:
        return "robo_98"
    elif confidence >= 0.97:
        return "robo_97"
    return None


def write_xmp_sidecar(target_path, keyword):
    """
    Write XMP sidecar with tiered keyword (for RAW files).
    """
    xmp_path = target_path.with_suffix(".xmp")

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
          <rdf:li>robo|{keyword}</rdf:li>
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
    """
    Embed keyword directly into JPEG using exiftool.
    Uses += to add without overwriting existing keywords.
    """
    try:
        result = subprocess.run(
            [
                'exiftool',
                '-overwrite_original',
                f'-Keywords+={keyword}',
                f'-Subject+={keyword}',
                f'-HierarchicalSubject+=robo|{keyword}',
                str(jpeg_path)
            ],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("ERROR: exiftool not found. Install with: brew install exiftool")
        return False


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

    # Check for exiftool if we might need it
    if not args.nef_dir:
        try:
            subprocess.run(['exiftool', '-ver'], capture_output=True)
        except FileNotFoundError:
            print("ERROR: exiftool not found. Install with: brew install exiftool")
            print("       Or use --nef_dir to write XMP sidecars for RAW files instead.")
            return

    # Read winners
    with open(args.winners) as f:
        winners = list(csv.DictReader(f))

    print(f"Read {len(winners)} winners from {args.winners}")

    # Count tiers
    tier_counts = {"robo_99": 0, "robo_98": 0, "robo_97": 0, "below_threshold": 0}
    written = 0
    errors = 0

    for i, row in enumerate(winners, 1):
        confidence = float(row['confidence_select'])
        keyword = get_tier_keyword(confidence)

        if keyword is None:
            tier_counts["below_threshold"] += 1
            continue

        tier_counts[keyword] += 1

        # Determine target path and method
        source_path = Path(row['path'])
        stem = source_path.stem

        if args.nef_dir:
            # Look for matching NEF in specified directory
            nef_dir = Path(args.nef_dir)
            target = nef_dir / f"{stem}.NEF"
            if not target.exists():
                target = nef_dir / f"{stem}.nef"
            if not target.exists():
                print(f"  WARNING: No NEF found for {stem}")
                errors += 1
                continue
            use_xmp = True
        else:
            # Write directly to source file
            target = source_path
            # Use XMP for RAW files, embed for JPEGs
            use_xmp = target.suffix.lower() in ['.nef', '.cr2', '.cr3', '.arw', '.orf', '.raf', '.dng']

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

            # Progress indicator
            if i % 100 == 0:
                print(f"  Processed {i}/{len(winners)}...")

    print(f"\n=== SUMMARY ===")
    print(f"robo_99 (>=0.99): {tier_counts['robo_99']}")
    print(f"robo_98 (>=0.98): {tier_counts['robo_98']}")
    print(f"robo_97 (>=0.97): {tier_counts['robo_97']}")
    print(f"Below threshold:  {tier_counts['below_threshold']}")

    if not args.dry_run:
        print(f"\nWrote {written} keywords")
        if errors:
            print(f"Errors: {errors}")
    else:
        print(f"\n(dry run - no files written)")


if __name__ == "__main__":
    main()

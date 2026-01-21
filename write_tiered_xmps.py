#!/usr/bin/env python3
"""
write_tiered_xmps.py

Reads a winners CSV and writes XMP sidecars with tiered keywords based on confidence.

Tiers:
  - robo_99: confidence >= 0.99
  - robo_98: confidence >= 0.98
  - robo_97: confidence >= 0.97

Usage:
    python write_tiered_xmps.py --winners lime-rock-winners.csv --nef_dir /path/to/nefs
    python write_tiered_xmps.py --winners lime-rock-winners.csv  # writes next to source JPGs
"""

import argparse
import csv
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


def write_xmp_sidecar(target_path, keyword, confidence):
    """
    Write XMP sidecar with tiered keyword.
    target_path: the NEF or image file to write sidecar for
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


def main():
    parser = argparse.ArgumentParser(
        description="Write tiered XMP sidecars from winners CSV"
    )
    parser.add_argument(
        "--winners",
        required=True,
        help="Path to winners CSV (from burst deduplication)"
    )
    parser.add_argument(
        "--nef_dir",
        help="Directory containing NEF files (if different from JPG source)"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print what would be written without writing"
    )

    args = parser.parse_args()

    # Read winners
    with open(args.winners) as f:
        winners = list(csv.DictReader(f))

    print(f"Read {len(winners)} winners from {args.winners}")

    # Count tiers
    tier_counts = {"robo_99": 0, "robo_98": 0, "robo_97": 0, "below_threshold": 0}
    written = 0

    for row in winners:
        confidence = float(row['confidence_select'])
        keyword = get_tier_keyword(confidence)

        if keyword is None:
            tier_counts["below_threshold"] += 1
            continue

        tier_counts[keyword] += 1

        # Determine target path
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
                continue
        else:
            # Write sidecar next to source file
            target = source_path

        if args.dry_run:
            print(f"  Would write {target}.xmp with {keyword}")
        else:
            write_xmp_sidecar(target, keyword, confidence)
            written += 1

    print(f"\n=== SUMMARY ===")
    print(f"robo_99 (>=0.99): {tier_counts['robo_99']}")
    print(f"robo_98 (>=0.98): {tier_counts['robo_98']}")
    print(f"robo_97 (>=0.97): {tier_counts['robo_97']}")
    print(f"Below threshold:  {tier_counts['below_threshold']}")

    if not args.dry_run:
        print(f"\nWrote {written} XMP sidecars")
    else:
        print(f"\n(dry run - no files written)")


if __name__ == "__main__":
    main()

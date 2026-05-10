#!/usr/bin/env python3
"""
prepare_training_data.py

Organizes exported JPGs into training and test sets.
Handles class imbalance by computing weights for weighted loss.

Usage:
    python prepare_training_data.py --select_dir /path/to/select --reject_dir /path/to/reject --output_dir /path/to/dataset
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Callable, Optional

from sklearn.model_selection import train_test_split


def _find_images(directory: Path) -> list[Path]:
    return [f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg"}]


def prepare_dataset(
    select_dir: str | Path,
    reject_dir: str | Path,
    output_dir: str | Path,
    test_size: float = 0.2,
    random_seed: int = 42,
    progress_cb: Optional[Callable[[dict], None]] = None,
) -> dict:
    """
    Organize select/reject images into train/test splits.

    Output structure:
        output_dir/
            train/select/  train/reject/
            test/select/   test/reject/
            dataset_stats.json

    Progress events emitted via progress_cb:
        {type: "scan",   select: N, reject: M}
        {type: "split",  train_select: N, train_reject: M, test_select: N, test_reject: M}
        {type: "copy",   done: N, total: T}
        {type: "done",   stats: {...}}
    """

    def emit(event: dict):
        if progress_cb:
            progress_cb(event)

    select_dir = Path(select_dir)
    reject_dir = Path(reject_dir)
    output_dir = Path(output_dir)

    # Validate sources
    if not select_dir.is_dir():
        raise ValueError(f"select_dir not found: {select_dir}")
    if not reject_dir.is_dir():
        raise ValueError(f"reject_dir not found: {reject_dir}")

    # Create output dirs
    for subdir in ["train/select", "train/reject", "test/select", "test/reject"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    # Scan
    select_images = _find_images(select_dir)
    reject_images = _find_images(reject_dir)
    print(f"Found {len(select_images)} select, {len(reject_images)} reject")
    emit({"type": "scan", "select": len(select_images), "reject": len(reject_images)})

    if not select_images:
        raise ValueError(f"No images found in select_dir: {select_dir}")
    if not reject_images:
        raise ValueError(f"No images found in reject_dir: {reject_dir}")

    # Split
    select_train, select_test = train_test_split(
        select_images, test_size=test_size, random_state=random_seed
    )
    reject_train, reject_test = train_test_split(
        reject_images, test_size=test_size, random_state=random_seed
    )
    print(f"Train: {len(select_train)} select, {len(reject_train)} reject")
    print(f"Test:  {len(select_test)} select,  {len(reject_test)} reject")
    emit({
        "type": "split",
        "train_select": len(select_train), "train_reject": len(reject_train),
        "test_select":  len(select_test),  "test_reject":  len(reject_test),
    })

    # Copy
    copies = [
        (select_train, output_dir / "train" / "select"),
        (select_test,  output_dir / "test"  / "select"),
        (reject_train, output_dir / "train" / "reject"),
        (reject_test,  output_dir / "test"  / "reject"),
    ]
    total = sum(len(c[0]) for c in copies)
    done = 0
    for images, dest in copies:
        for img in images:
            shutil.copy2(img, dest / img.name)
            done += 1
            if done % 50 == 0 or done == total:
                emit({"type": "copy", "done": done, "total": total})
    emit({"type": "copy", "done": total, "total": total})

    # Class weights for weighted loss
    n_select = len(select_train)
    n_reject = len(reject_train)
    n_total  = n_select + n_reject
    w_select = n_total / (2 * n_select)
    w_reject = n_total / (2 * n_reject)

    stats = {
        "train":  {"select": n_select,  "reject": n_reject},
        "test":   {"select": len(select_test), "reject": len(reject_test)},
        "class_weights": {"select": w_select, "reject": w_reject},
        "select_ratio": round(n_select / n_total * 100, 2),
    }
    (output_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"select ratio: {stats['select_ratio']}%  weights: select={w_select:.3f} reject={w_reject:.3f}")
    emit({"type": "done", "stats": stats})
    return stats


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--select_dir", required=True, help="Directory of select images")
    parser.add_argument("--reject_dir", required=True, help="Directory of reject images")
    parser.add_argument("--output_dir", default="./dataset", help="Output directory (default: ./dataset)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction (default: 0.2)")
    # Legacy alias kept for backward compatibility
    parser.add_argument("--data_dir", help="(legacy) parent dir containing select/ and reject/ subdirs")
    args = parser.parse_args()

    if args.data_dir and not (args.select_dir or args.reject_dir):
        data_dir = Path(args.data_dir)
        select_dir = data_dir / "select"
        reject_dir = data_dir / "reject"
    else:
        select_dir = args.select_dir
        reject_dir = args.reject_dir

    prepare_dataset(select_dir, reject_dir, args.output_dir, test_size=args.test_size)


if __name__ == "__main__":
    main()

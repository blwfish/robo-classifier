#!/usr/bin/env python3
"""
prepare_training_data.py

Organizes exported JPGs from Lightroom into training and test sets.
Handles class imbalance by computing weights for weighted loss.

Usage:
    python prepare_training_data.py --data_dir /path/to/exported/frames --output_dir ./dataset
"""

import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
import json

def count_images(directory):
    """Count JPG files in directory."""
    return len(list(Path(directory).glob("*.jpg"))) + len(list(Path(directory).glob("*.JPG")))

def prepare_dataset(data_dir, output_dir, test_size=0.2, random_seed=42):
    """
    Organize images into train/test splits.
    
    Expected input structure:
        data_dir/
            interesting/
                *.jpg
            boring/
                *.jpg
    
    Output structure:
        output_dir/
            train/
                interesting/
                boring/
            test/
                interesting/
                boring/
            dataset_stats.json
    """
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    train_select = output_dir / "train" / "select"
    train_reject = output_dir / "train" / "reject"
    test_select = output_dir / "test" / "select"
    test_reject = output_dir / "test" / "reject"

    for d in [train_select, train_reject, test_select, test_reject]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect all images from each class
    select_dir = data_dir / "select"
    reject_dir = data_dir / "reject"

    select_images = list(select_dir.glob("*.jpg")) + list(select_dir.glob("*.JPG"))
    reject_images = list(reject_dir.glob("*.jpg")) + list(reject_dir.glob("*.JPG"))
    
    print(f"Found {len(select_images)} select images")
    print(f"Found {len(reject_images)} reject images")

    # Split each class
    select_train, select_test = train_test_split(
        select_images,
        test_size=test_size,
        random_state=random_seed
    )

    reject_train, reject_test = train_test_split(
        reject_images,
        test_size=test_size,
        random_state=random_seed
    )

    print(f"\nTrain set: {len(select_train)} select, {len(reject_train)} reject")
    print(f"Test set: {len(select_test)} select, {len(reject_test)} reject")

    # Copy files
    for img in select_train:
        shutil.copy(img, train_select / img.name)

    for img in select_test:
        shutil.copy(img, test_select / img.name)

    for img in reject_train:
        shutil.copy(img, train_reject / img.name)

    for img in reject_test:
        shutil.copy(img, test_reject / img.name)

    # Compute class weights for weighted loss (to handle imbalance)
    total_select = len(select_train)
    total_reject = len(reject_train)
    total = total_select + total_reject

    weight_select = total / (2 * total_select)
    weight_reject = total / (2 * total_reject)

    stats = {
        "train": {
            "select": len(select_train),
            "reject": len(reject_train)
        },
        "test": {
            "select": len(select_test),
            "reject": len(reject_test)
        },
        "class_weights": {
            "select": weight_select,
            "reject": weight_reject
        },
        "select_ratio": round(total_select / total * 100, 2)
    }

    # Save stats
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nClass weights (for loss function):")
    print(f"  select: {weight_select:.4f}")
    print(f"  reject: {weight_reject:.4f}")
    print(f"  select ratio: {stats['select_ratio']}%")
    print(f"\nDataset stats saved to {stats_file}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from Lightroom exports"
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Path to directory containing 'interesting/' and 'boring/' subdirectories"
    )
    parser.add_argument(
        "--output_dir",
        default="./dataset",
        help="Output directory for train/test split (default: ./dataset)"
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    prepare_dataset(args.data_dir, args.output_dir, test_size=args.test_size)

if __name__ == "__main__":
    main()

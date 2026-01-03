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
    train_interesting = output_dir / "train" / "interesting"
    train_boring = output_dir / "train" / "boring"
    test_interesting = output_dir / "test" / "interesting"
    test_boring = output_dir / "test" / "boring"
    
    for d in [train_interesting, train_boring, test_interesting, test_boring]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Collect all images from each class
    interesting_dir = data_dir / "interesting"
    boring_dir = data_dir / "boring"
    
    interesting_images = list(interesting_dir.glob("*.jpg")) + list(interesting_dir.glob("*.JPG"))
    boring_images = list(boring_dir.glob("*.jpg")) + list(boring_dir.glob("*.JPG"))
    
    print(f"Found {len(interesting_images)} interesting images")
    print(f"Found {len(boring_images)} boring images")
    
    # Split each class
    interesting_train, interesting_test = train_test_split(
        interesting_images, 
        test_size=test_size, 
        random_state=random_seed
    )
    
    boring_train, boring_test = train_test_split(
        boring_images,
        test_size=test_size,
        random_state=random_seed
    )
    
    print(f"\nTrain set: {len(interesting_train)} interesting, {len(boring_train)} boring")
    print(f"Test set: {len(interesting_test)} boring, {len(boring_test)} boring")
    
    # Copy files
    for img in interesting_train:
        shutil.copy(img, train_interesting / img.name)
    
    for img in interesting_test:
        shutil.copy(img, test_interesting / img.name)
    
    for img in boring_train:
        shutil.copy(img, train_boring / img.name)
    
    for img in boring_test:
        shutil.copy(img, test_boring / img.name)
    
    # Compute class weights for weighted loss (to handle imbalance)
    total_interesting = len(interesting_train)
    total_boring = len(boring_train)
    total = total_interesting + total_boring
    
    weight_interesting = total / (2 * total_interesting)
    weight_boring = total / (2 * total_boring)
    
    stats = {
        "train": {
            "interesting": len(interesting_train),
            "boring": len(boring_train)
        },
        "test": {
            "interesting": len(interesting_test),
            "boring": len(boring_test)
        },
        "class_weights": {
            "interesting": weight_interesting,
            "boring": weight_boring
        },
        "interesting_ratio": round(total_interesting / total * 100, 2)
    }
    
    # Save stats
    stats_file = output_dir / "dataset_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nClass weights (for loss function):")
    print(f"  interesting: {weight_interesting:.4f}")
    print(f"  boring: {weight_boring:.4f}")
    print(f"  interesting ratio: {stats['interesting_ratio']}%")
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

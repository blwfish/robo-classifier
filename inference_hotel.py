#!/usr/bin/env python3
"""
inference_hotel.py

Classifies images and writes XMP sidecars for Lightroom smart collections.

Supports:
  - JPG, PNG input
  - NEF input (extracts embedded JPG preview via ImageMagick)
  - Batch processing with CSV output
  - XMP sidecar generation for Lightroom

Usage:
    python inference_hotel.py --model model.pt --input_dir ./frames --output_csv results.csv

    For NEF files, ensure you have ImageMagick installed:
        brew install imagemagick  (on macOS)
"""

import os
import subprocess
import tempfile

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import csv
from pathlib import Path
from PIL import Image

from classify import ImageClassifier, ImageDataset


def _classify_single(classifier, image_path):
    """Classify one image. Returns (class_name, confidence, predictions_dict)."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"  ERROR: Could not load {image_path}: {e}")
        return None, None, None

    transform = classifier.get_transform()
    img_tensor = transform(img).unsqueeze(0).to(classifier.device)

    with torch.no_grad():
        logits = classifier.model(img_tensor)
        probs = F.softmax(logits, dim=1)

    pred_idx = logits.argmax(dim=1).item()
    pred_class = classifier.class_names[pred_idx]
    confidence = probs[0, pred_idx].item()

    predictions = {
        classifier.class_names[i]: float(probs[0, i].item())
        for i in range(len(classifier.class_names))
    }

    return pred_class, confidence, predictions


def extract_nef_preview(nef_path, temp_jpg=None):
    """
    Extract embedded JPG preview from NEF using ImageMagick.
    Returns path to temporary JPG file, or None on failure.
    Caller is responsible for deleting the file when done.
    """
    if temp_jpg is None:
        fd, temp_jpg = tempfile.mkstemp(suffix='.jpg')
        os.close(fd)

    try:
        subprocess.run(
            ['convert', f"{nef_path}[0]", '-quality', '90', temp_jpg],
            check=True,
            capture_output=True
        )
        return temp_jpg
    except subprocess.CalledProcessError:
        print(f"  ERROR: Could not extract preview from {nef_path}")
        os.unlink(temp_jpg)
        return None
    except FileNotFoundError:
        print("  ERROR: ImageMagick not found. Install with: brew install imagemagick")
        os.unlink(temp_jpg)
        return None


def write_xmp_sidecar(image_path, classification, confidence):
    """
    Write classification keyword to XMP sidecar via exiftool.

    Uses dc:subject / lr:hierarchicalSubject so Lightroom keyword smart
    collections work. Previously wrote to Iptc4xmpCore:CiKeywords (creator
    contact field), which Lightroom ignores for keyword indexing.

    The sidecar is placed at <stem>.xmp alongside the image file, not at
    <filename>.xmp (which was the previous broken behaviour for RAW files).
    """
    import subprocess
    image_path = Path(image_path)
    xmp_path = image_path.with_suffix('.xmp')

    if xmp_path.exists():
        cmd = [
            'exiftool', '-overwrite_original',
            f'-Keywords+={classification}',
            f'-Subject+={classification}',
            f'-HierarchicalSubject+=robo|{classification}',
            str(xmp_path),
        ]
    else:
        cmd = [
            'exiftool',
            '-tagsfromfile', str(image_path),
            f'-Keywords+={classification}',
            f'-Subject+={classification}',
            f'-HierarchicalSubject+=robo|{classification}',
            '-o', str(xmp_path),
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        print("  ERROR: exiftool not found. Install with: brew install exiftool")
        return False


def process_directory(classifier, input_dir, output_csv=None, write_xmp=True,
                      confidence_threshold=0.5, batch_size=32, num_workers=4):
    """
    Process all images in a directory using batched inference.

    Args:
        classifier: ImageClassifier instance
        input_dir: directory containing images
        output_csv: optional path to write CSV results
        write_xmp: if True, write XMP sidecars
        confidence_threshold: only write XMP if confidence >= this (use 0 to write all)
        batch_size: batch size for inference
        num_workers: number of data loader workers
    """

    input_dir = Path(input_dir)

    image_files = []
    nef_files = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        image_files.extend(input_dir.glob(ext))
    for ext in ['*.nef', '*.NEF']:
        nef_files.extend(input_dir.glob(ext))

    image_files = sorted(image_files)
    nef_files = sorted(nef_files)

    print(f"Found {len(image_files)} images and {len(nef_files)} NEF files to process")

    results = []

    if image_files:
        print(f"\nProcessing {len(image_files)} images in batches of {batch_size}...")

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

                    if write_xmp and confidence >= confidence_threshold:
                        write_xmp_sidecar(image_path, pred_class, confidence)

                processed += len(batch_indices)
                print(f"\r  Processed {processed}/{len(image_files)} images...", end="", flush=True)

        print()

    if nef_files:
        print(f"\nProcessing {len(nef_files)} NEF files...")
        for i, image_path in enumerate(nef_files, 1):
            print(f"[{i:4d}/{len(nef_files)}] Processing {image_path.name}...", end=" ")

            jpg_temp = extract_nef_preview(image_path)
            if jpg_temp is None:
                print("SKIP (extract failed)")
                continue

            try:
                pred_class, confidence, predictions = _classify_single(classifier, jpg_temp)
            finally:
                os.unlink(jpg_temp)

            if pred_class is None:
                print("SKIP (load failed)")
                continue

            print(f"{pred_class} ({confidence:.4f})")

            result = {
                'filename': image_path.name,
                'path': str(image_path),
                'classification': pred_class,
                'confidence': confidence,
                'confidence_reject': predictions.get('reject', 0.0),
                'confidence_select': predictions.get('select', 0.0)
            }
            results.append(result)

            if write_xmp and confidence >= confidence_threshold:
                write_xmp_sidecar(image_path, pred_class, confidence)

    if output_csv and results:
        output_csv = Path(output_csv)
        print(f"\nWriting results to {output_csv}")
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    if results:
        select_count = sum(1 for r in results if r['classification'] == 'select')
        print(f"\n=== SUMMARY ===")
        print(f"Processed: {len(results)} images")
        print(f"Select: {select_count} ({100*select_count/len(results):.2f}%)")
        print(f"Reject: {len(results) - select_count} ({100*(len(results)-select_count)/len(results):.2f}%)")
    else:
        print("\nNo images processed.")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Classify images and write XMP sidecars for Lightroom"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model (from train_classifier.py)"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input directory containing images (JPG, NEF, etc.)"
    )
    parser.add_argument(
        "--output_csv",
        help="Optional: write CSV results to this file"
    )
    parser.add_argument(
        "--no_xmp",
        action="store_true",
        help="Don't write XMP sidecars (just CSV)"
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Only write XMP if confidence > this threshold (default: 0.5)"
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

    args = parser.parse_args()

    classifier = ImageClassifier(args.model)
    process_directory(
        classifier,
        args.input_dir,
        output_csv=args.output_csv,
        write_xmp=not args.no_xmp,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()

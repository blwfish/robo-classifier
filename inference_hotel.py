#!/usr/bin/env python3
"""
inference_hotel.py

Classifies images and writes XMP sidecars for Lightroom smart collections.

Supports:
  - JPG, PNG input
  - NEF input (extracts embedded JPG preview)
  - Batch processing with CSV output
  - XMP sidecar generation for Lightroom

Usage:
    python inference_hotel.py --model model.pt --input_dir ./frames --output_csv results.csv
    
    For NEF files, ensure you have ImageMagick installed:
        brew install imagemagick  (on macOS)
"""

import torch
import torch.nn.functional as F
from torchvision import models, transforms
import argparse
import csv
import json
from pathlib import Path
from PIL import Image
import subprocess
import tempfile
import sys

class InterestingClassifier:
    def __init__(self, model_path, device=None):
        """Load trained model."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Load model
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
        print(f"Using device: {self.device}\n")
    
    def get_transform(self):
        """Return inference transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def classify_image(self, image_path):
        """
        Classify a single image.
        Returns: (class_name, confidence_score, predictions_dict)
        """
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"  ERROR: Could not load {image_path}: {e}")
            return None, None, None
        
        transform = self.get_transform()
        img_tensor = transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=1)
        
        pred_idx = logits.argmax(dim=1).item()
        pred_class = self.class_names[pred_idx]
        confidence = probs[0, pred_idx].item()
        
        # Full prediction dict for potential threshold tuning later
        predictions = {
            self.class_names[i]: float(probs[0, i].item())
            for i in range(len(self.class_names))
        }
        
        return pred_class, confidence, predictions

def extract_nef_preview(nef_path, temp_jpg=None):
    """
    Extract embedded JPG preview from NEF using ImageMagick.
    Returns path to temporary JPG file.
    """
    if temp_jpg is None:
        temp_jpg = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
    
    try:
        # ImageMagick: extract first image (the preview) from NEF
        subprocess.run(
            ['convert', f"{nef_path}[0]", '-quality', '90', temp_jpg],
            check=True,
            capture_output=True
        )
        return temp_jpg
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Could not extract preview from {nef_path}")
        return None
    except FileNotFoundError:
        print("  ERROR: ImageMagick not found. Install with: brew install imagemagick")
        return None

def write_xmp_sidecar(image_path, classification, confidence):
    """
    Write XMP sidecar with classification results.
    Lightroom can read these and create smart collections.
    """
    xmp_path = Path(str(image_path) + ".xmp")
    
    # Simple XMP format with custom namespace
    xmp_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:Iptc4xmpCore="http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/">
  <rdf:RDF>
    <rdf:Description rdf:about="uuid:faf5bdd5-ba3d-11da-ad31-d33d75182f1b" xmlns:Iptc4xmpCore="http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/">
      <Iptc4xmpCore:CreatorContactInfo>
        <rdf:Seq>
          <rdf:li>
            <rdf:Description>
              <Iptc4xmpCore:CiKeywords>
                <rdf:Bag>
                  <rdf:li>x5-{classification}</rdf:li>
                </rdf:Bag>
              </Iptc4xmpCore:CiKeywords>
            </rdf:Description>
          </rdf:li>
        </rdf:Seq>
      </Iptc4xmpCore:CreatorContactInfo>
    </rdf:Description>
    <rdf:Description rdf:about="uuid:faf5bdd5-ba3d-11da-ad31-d33d75182f1b" xmlns:x5="http://example.com/x5/1.0/">
      <x5:classification>{classification}</x5:classification>
      <x5:confidence>{confidence:.4f}</x5:confidence>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
"""
    
    try:
        with open(xmp_path, 'w') as f:
            f.write(xmp_content)
        return True
    except Exception as e:
        print(f"  ERROR writing XMP sidecar: {e}")
        return False

def process_directory(classifier, input_dir, output_csv=None, write_xmp=True, 
                     confidence_threshold=0.5):
    """
    Process all images in a directory.
    
    Args:
        classifier: InterestingClassifier instance
        input_dir: directory containing images
        output_csv: optional path to write CSV results
        write_xmp: if True, write XMP sidecars
        confidence_threshold: only write XMP if confidence > this (use 0 to write all)
    """
    
    input_dir = Path(input_dir)
    results = []
    
    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG', '*.nef', '*.NEF']:
        image_files.extend(input_dir.glob(ext))
    
    image_files = sorted(image_files)
    
    print(f"Found {len(image_files)} images to process\n")
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i:4d}/{len(image_files)}] Processing {image_path.name}...", end=" ")
        
        # Handle NEF files
        if image_path.suffix.lower() == '.nef':
            jpg_temp = extract_nef_preview(image_path)
            if jpg_temp is None:
                print("SKIP (extract failed)")
                continue
            classify_path = jpg_temp
        else:
            classify_path = image_path
        
        # Classify
        pred_class, confidence, predictions = classifier.classify_image(classify_path)
        
        if pred_class is None:
            print("SKIP (load failed)")
            continue
        
        print(f"{pred_class} ({confidence:.4f})")
        
        # Store result
        result = {
            'filename': image_path.name,
            'path': str(image_path),
            'classification': pred_class,
            'confidence': confidence,
            'confidence_boring': predictions['boring'],
            'confidence_interesting': predictions['interesting']
        }
        results.append(result)
        
        # Write XMP if confidence exceeds threshold
        if write_xmp and confidence >= confidence_threshold:
            write_xmp_sidecar(image_path, pred_class, confidence)
    
    # Write CSV
    if output_csv:
        output_csv = Path(output_csv)
        print(f"\nWriting results to {output_csv}")
        
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    # Summary
    interesting_count = sum(1 for r in results if r['classification'] == 'interesting')
    print(f"\n=== SUMMARY ===")
    print(f"Processed: {len(results)} images")
    print(f"Interesting: {interesting_count} ({100*interesting_count/len(results):.2f}%)")
    print(f"Boring: {len(results) - interesting_count} ({100*(len(results)-interesting_count)/len(results):.2f}%)")
    
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
    
    args = parser.parse_args()
    
    classifier = InterestingClassifier(args.model)
    process_directory(
        classifier,
        args.input_dir,
        output_csv=args.output_csv,
        write_xmp=not args.no_xmp,
        confidence_threshold=args.confidence_threshold
    )

if __name__ == "__main__":
    main()

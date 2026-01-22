# Robo-Classifier

Classification system for identifying "interesting" racing footage from high-volume auto-capture streams. Uses ResNet50 fine-tuned on manually curated examples.

## What's Interesting?

The classifier learns to detect:
- Cars with wheels off the ground
- Tire smoke (hard braking into corners)
- Crashes and impacts
- Unusual car orientations (near-misses)
- Passes and overtaking moves

## Quick Start

After pulling images from camera cards:

```bash
source .venv/bin/activate
python classify.py /path/to/images
```

This will:
1. Run inference on all images
2. Deduplicate bursts (pick best frame per burst)
3. Write tiered keywords to images above 0.97 confidence
4. Tag all frames in qualifying bursts with `select`

## Project Structure

```
robo-classifier/
├── classify.py               # Main pipeline (inference + dedup + keywords)
├── prepare_training_data.py  # Split Lightroom exports into train/test
├── train_classifier.py       # Fine-tune ResNet50 (weighted loss for imbalance)
├── inference_hotel.py        # Low-level inference (used by classify.py)
├── write_tiered_keywords.py  # Standalone keyword writer
├── select/                   # Sample "interesting" images for training
├── reject/                   # Sample "boring" images for training
├── model.pt                  # Trained model (generated)
└── .venv/                    # Python virtual environment
```

## Workflow

### Training (one-time setup)

1. **Data Prep**: Export JPGs from Lightroom to `select/` and `reject/`
2. **Split**: `python prepare_training_data.py --data_dir . --output_dir ./dataset`
3. **Train**: `python train_classifier.py --dataset_dir ./dataset --model_output model.pt`

### Production (after each shoot)

```bash
python classify.py /path/to/card/images
```

Output:
- `results.csv` - all images with classifications
- `winners.csv` - best frame per burst (select only)
- Keywords embedded in JPEGs (or XMP sidecars for RAW)

### Lightroom Integration

Create smart collections:
- `robo_99` - highest confidence shots (≥0.99)
- `robo_98` - high confidence (≥0.98)
- `robo_97` - moderate confidence (≥0.97)
- `select` - all frames from interesting bursts (browse for alternates)

Untagged images remain in the catalog as a safety net.

## Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch torchvision scikit-learn pillow

# For JPEG keyword embedding
brew install exiftool

# For NEF preview extraction (optional)
brew install imagemagick
```

## Command Reference

### classify.py (main pipeline)

```bash
python classify.py /path/to/images [options]

Options:
  --model PATH        Path to trained model (default: model.pt)
  --output_dir PATH   Where to write CSVs (default: input_dir)
  --nef_dir PATH      Write XMP sidecars here instead of embedding in JPEGs
  --batch_size N      Batch size for inference (default: 32)
  --no_keywords       Skip writing keywords (just output CSVs)
  --dry_run           Show what would happen without writing
```

### Training scripts

```bash
# Prepare dataset
python prepare_training_data.py --data_dir . --output_dir ./dataset

# Train model
python train_classifier.py --dataset_dir ./dataset --model_output model.pt --epochs 15
```

## Key Details

- **Model**: ResNet50 pretrained on ImageNet, fine-tuned for binary classification
- **Input size**: 224x224
- **Device**: Auto-detects MPS (Apple Silicon), CUDA, or CPU
- **Class imbalance**: Handled via weighted CrossEntropyLoss
- **Burst detection**: Groups by filename stem (e.g., `BLW0124-3.jpg` → `BLW0124`)
- **Keywords**: Embedded directly in JPEGs via exiftool; XMP sidecars for RAW files

## Performance

On M4 Max with MPS:
- ~42,000 images: ~2 minutes inference
- Keyword writing: ~1-2 minutes for ~15,000 files

## Two Workflows Supported

1. **RAW+JPG** (your workflow): XMP sidecars work with NEFs
2. **JPG-only** (Vic's workflow): Keywords embedded directly in JPEGs

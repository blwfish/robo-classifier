# Robo-Classifier

Image classifier for identifying "interesting" racing footage from high-volume auto-capture streams. Uses ResNet50 fine-tuned on manually curated examples.

## Purpose

Classifies racing photos to find notable moments:
- Cars with wheels off ground
- Tire smoke (hard braking)
- Crashes and impacts
- Unusual car orientations
- Passes and overtaking

## Project Structure

```
robo-classifier/
├── classify.py               # Main pipeline: inference + dedup + keywords
├── prepare_training_data.py  # Split Lightroom exports into train/test
├── train_classifier.py       # Fine-tune ResNet50 (weighted loss for imbalance)
├── inference_hotel.py        # Batch classify (low-level, used by classify.py)
├── write_tiered_keywords.py  # Standalone keyword writer
├── select/                   # Sample "interesting" images
├── reject/                   # Sample "boring" images
├── model.pt                  # Trained model (generated)
└── .venv/                    # Python virtual environment
```

## Workflow

### Production (after each shoot)

```bash
source .venv/bin/activate
python classify.py /path/to/images
```

### Training (one-time)

1. **Data Prep**: Export JPGs from Lightroom to `select/` and `reject/`
2. **Split**: `python prepare_training_data.py --data_dir . --output_dir ./dataset`
3. **Train**: `python train_classifier.py --dataset_dir ./dataset --model_output model.pt`

## Dependencies

```bash
# In .venv
pip install torch torchvision scikit-learn pillow
brew install exiftool      # for JPEG keyword embedding
brew install imagemagick   # for NEF preview extraction (optional)
```

## Key Details

- **Model**: ResNet50 pretrained on ImageNet, fine-tuned for binary classification
- **Input size**: 224x224
- **Device**: Auto-detects MPS (Apple Silicon), CUDA, or CPU
- **Class imbalance**: ~3% interesting, handled via weighted CrossEntropyLoss
- **Burst detection**: Groups by filename stem (e.g., `BLW0124-3.jpg` → `BLW0124`)
- **Keywords**:
  - `robo_99`, `robo_98`, `robo_97` for tiered confidence winners
  - `select` for all frames in qualifying bursts
  - Embedded in JPEGs via exiftool; XMP sidecars for RAW files

## Output

- `results.csv` - all images with classifications and confidence scores
- `winners.csv` - best frame per burst (select only)
- Keywords embedded in files for Lightroom smart collections

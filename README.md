# X5 Interesting Frame Classifier

Classification system for identifying "interesting" racing footage from high-volume auto-capture streams. Uses ResNet50 fine-tuned on manually curated examples.

## What's Interesting?

The classifier learns to detect:
- Cars with wheels off the ground (multiple wheel lift is especially notable)
- Tire smoke (hard braking into corners)
- Crashes and impacts
- Unusual car orientations (near-misses)
- Passes and overtaking moves

## Project Structure

```
x5-work/
├── prepare_training_data.py    # Organize exported frames into train/test sets
├── train_classifier.py          # Fine-tune ResNet50
├── inference_hotel.py           # Batch classify and write XMP sidecars
├── README.md                    # This file
├── STATUS.md                    # Current progress
├── TODO.md                      # Next steps
├── requirements.txt             # Python dependencies
└── model.pt                     # Trained model (generated)
```

## Workflow

### 1. Data Preparation

Export your manually curated frames from Lightroom as JPGs:
- Interesting frames to: `data/interesting/`
- Boring frames to: `data/boring/`

Then organize into train/test splits:

```bash
python prepare_training_data.py \
    --data_dir ./data \
    --output_dir ./dataset
```

This creates:
- `dataset/train/{interesting,boring}/` — training set
- `dataset/test/{interesting,boring}/` — evaluation set
- `dataset/dataset_stats.json` — class weights for loss function

### 2. Training

Fine-tune ResNet50:

```bash
python train_classifier.py \
    --dataset_dir ./dataset \
    --model_output model.pt \
    --epochs 15
```

The script uses weighted loss to handle class imbalance (typically ~3% interesting, ~97% boring).

### 3. Hotel Inference

After shooting, classify your NEF files and write XMP sidecars:

```bash
python inference_hotel.py \
    --model model.pt \
    --input_dir /Volumes/RaceFootage/Sebring2025/NEFs \
    --output_csv results.csv
```

This creates:
- `results.csv` — classification results with confidence scores
- `*.nef.xmp` sidecars — Lightroom-readable metadata

In Lightroom, create a smart collection: filter by keywords `x5-interesting` to instantly see classified frames.

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For NEF processing (optional but recommended)
brew install imagemagick
```

## Dependencies

- PyTorch
- torchvision
- scikit-learn
- Pillow
- ImageMagick (for NEF preview extraction)

## Parameters

### Training

- **Model:** ResNet50 (pretrained on ImageNet)
- **Input size:** 224×224
- **Batch size:** 32
- **Learning rate:** 1e-4 (fine-tuning, goes slow)
- **Epochs:** 15 (adjust if converging early/late)
- **Loss:** WeightedCrossEntropy (accounts for class imbalance)

### Inference

- **Confidence threshold:** 0.5 (only write XMP if confidence > this)
- **Output:** CSV + XMP sidecars
- **Supported formats:** JPG, PNG, NEF (with ImageMagick)

## Notes

- Class imbalance is expected (~0.67% interesting at Lime Rock, ~2.75% at Sebring). The loss function weights accordingly.
- Precision matters more than recall—better to miss 10% of interesting frames than flag 1000 boring ones.
- The confidence threshold can be tuned post-training if needed.
- XMP sidecars are human-readable XML; Lightroom can read them immediately after import.

## Performance Notes

- Training on 275 interesting + ~3000 boring frames should converge in ~10-15 epochs
- Inference on 10,000 frames takes ~30-60 seconds on modern hardware
- NEF preview extraction (ImageMagick) is the bottleneck for large batches; consider preprocessing to JPG if needed

## Future Work

- Real-time Z9 attachment over USB/network
- Confidence threshold tuning dashboard
- Hardened XMP sidecar format (current is simple but functional)
- Performance optimization (model quantization, mobile-friendly inference)

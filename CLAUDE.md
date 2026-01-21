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
├── prepare_training_data.py  # Split Lightroom exports into train/test
├── train_classifier.py       # Fine-tune ResNet50 (weighted loss for imbalance)
├── inference_hotel.py        # Batch classify + write XMP sidecars
├── select/                   # Sample "interesting" images
├── reject/                   # Sample "boring" images
└── model.pt                  # Trained model (generated)
```

## Workflow

1. **Data Prep**: Export JPGs from Lightroom to `select/` and `reject/`
2. **Split**: `python prepare_training_data.py --data_dir . --output_dir ./dataset`
3. **Train**: `python train_classifier.py --dataset_dir ./dataset --model_output model.pt`
4. **Infer**: `python inference_hotel.py --model model.pt --input_dir ./NEFs --output_csv results.csv`

## Dependencies

```bash
pip install torch torchvision scikit-learn Pillow
brew install imagemagick  # for NEF preview extraction
```

## Key Details

- **Model**: ResNet50 pretrained on ImageNet, fine-tuned for binary classification
- **Input size**: 224x224
- **Class imbalance**: ~3% interesting, handled via weighted CrossEntropyLoss
- **XMP sidecars**: Written for Lightroom smart collections (keyword: `select`)
- **NEF support**: Uses ImageMagick to extract embedded preview

## Environment

- Python 3.11+
- PyTorch with MPS (Apple Silicon) or CUDA support
- ImageMagick for NEF processing

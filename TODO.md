# TODO

## Recent Updates (Feb 2026)

### ✅ Merged Features
- [x] RAW file support (NEF, CR2, ARW, etc.) with automatic preview extraction
- [x] Time-based burst grouping (optional `--burst_threshold` flag)
- [x] Hierarchical keywords (`AI keywords|robo|{keyword}`) for better Lightroom organization
- [x] Expanded confidence tiers (robo_90 through robo_99 instead of just 97-99)
- [x] Recursive subdirectory search
- [x] Optimized RAW preview extraction using exiftool stay_open mode

## PRIORITY 1: Before Sebring (Target: 2-3 weeks)

### Week 1: Setup & Data Preparation

- [ ] **Hardware Setup**
  - [ ] Unbox Snapdragon 8 RDK X5
  - [ ] Boot and verify OS
  - [ ] Test camera module (any sample images)
  - [ ] Verify network connectivity

- [ ] **Development Environment**
  - [ ] Install Python 3.11+ (check: `python3 --version`)
  - [ ] Create venv: `python3 -m venv venv && source venv/bin/activate`
  - [ ] Install PyTorch (with MPS for Apple Silicon: `pip install torch torchvision`)
  - [ ] Install dependencies: `pip install scikit-learn pillow`
  - [ ] Install exiftool: `brew install exiftool` (required for RAW support)
  - [ ] Verify: `python3 -c "import torch; print(torch.__version__)"`

- [ ] **Data Preparation**
  - [ ] In Lightroom, select all 275 Lime Rock "interesting" frames
  - [ ] Export as JPG to: `/Volumes/Files/claude/x5-work/data/interesting/`
  - [ ] In Lightroom, select ~2500 random "boring" frames from anywhere
  - [ ] Export as JPG to: `/Volumes/Files/claude/x5-work/data/boring/`
  - [ ] Verify directory structure:
    ```
    data/
    ├── interesting/
    │   └── *.jpg (275 files)
    └── boring/
        └── *.jpg (~2500 files)
    ```

- [ ] **Dataset Split**
  - [ ] Run: `python prepare_training_data.py --data_dir ./data --output_dir ./dataset`
  - [ ] Verify output:
    - `dataset/train/interesting/` (220 files approx)
    - `dataset/train/boring/` (2000 files approx)
    - `dataset/test/interesting/` (55 files approx)
    - `dataset/test/boring/` (500 files approx)
    - `dataset/dataset_stats.json`
  - [ ] Review `dataset_stats.json` — note class weights for weighted loss

### Week 2: Training

- [ ] **Initial Training Run**
  - [ ] Run: `python train_classifier.py --dataset_dir ./dataset --model_output model.pt --epochs 15`
  - [ ] Monitor: each epoch should show decreasing train loss and increasing accuracy
  - [ ] Expected: best test accuracy should be 85-95% (imbalanced data helps boring classification)

- [ ] **Evaluate Results**
  - [ ] Review final metrics printed to console
  - [ ] Check if test accuracy > 90%
  - [ ] If accuracy is low (<80%), investigate:
    - [ ] Data quality — spot-check boring_frames, are they actually boring?
    - [ ] Class imbalance — is the weighting helping?
    - [ ] Try longer training (increase `--epochs`)

- [ ] **Save Model & Checkpoint**
  - [ ] Trained model saved to: `model.pt`
  - [ ] Make a backup: `cp model.pt model_v1.pt`
  - [ ] Commit to git with notes on training results

### Week 3: Pre-Sebring Testing

- [ ] **Hotel Workflow Test**
  - [ ] Create test directory with sample NEFs from your archive
  - [ ] Run: `python inference_hotel.py --model model.pt --input_dir ./test_frames --output_csv test_results.csv`
  - [ ] Verify:
    - [ ] CSV output created with classifications
    - [ ] XMP sidecars written (check for `*.nef.xmp` files)
    - [ ] No crashes on various input formats

- [ ] **Lightroom Integration Test**
  - [ ] Import a few test NEFs into Lightroom
  - [ ] Also import their XMP sidecars (should happen automatically)
  - [ ] Create smart collection: Keywords > x5-interesting
  - [ ] Verify: interesting frames appear in the collection

- [ ] **Confidence Threshold Tuning**
  - [ ] Run inference on sample with different thresholds:
    - [ ] `--confidence_threshold 0.5` (default)
    - [ ] `--confidence_threshold 0.7` (stricter)
    - [ ] `--confidence_threshold 0.3` (looser)
  - [ ] Decide which setting you prefer (balance false positives vs. false negatives)
  - [ ] Document chosen threshold in README

- [ ] **Documentation**
  - [ ] Update STATUS.md with actual results
  - [ ] Update README with real training metrics
  - [ ] Create quick-start guide for Sebring workflow

---

## PRIORITY 2: At Sebring (During Event)

- [ ] **Batch Processing**
  - [ ] After each day of shooting, transfer NEFs from Z9 + Z6III to working disk
  - [ ] Run inference: `python inference_hotel.py --model model.pt --input_dir ./NEFs --output_csv day1_results.csv`
  - [ ] Monitor results: % interesting, any anomalies?

- [ ] **Hotel Review Workflow**
  - [ ] Import NEFs + sidecars into Lightroom
  - [ ] Smart collection shows interesting frames
  - [ ] Flag for later review/processing

- [ ] **Quick Iteration** (if needed)
  - [ ] If accuracy is way off, hand-curate a few more examples
  - [ ] Re-train lightweight (~5 epochs) with Sebring-specific data
  - [ ] Re-run inference

---

## PRIORITY 3: After Sebring (Optional, Future)

- [ ] **Real-time Z9 Attachment**
  - [ ] Research Z9 USB/network interface
  - [ ] Design streaming pipeline
  - [ ] Test on-camera inference vs. remote classification

- [ ] **Model Optimization**
  - [ ] Quantize model for edge deployment (TorchScript, ONNX)
  - [ ] Profile inference speed
  - [ ] Optimize for real-time use case

- [ ] **Railroad Mode**
  - [ ] Plan rolling stock detection/OCR system
  - [ ] Design layout-based location tracking
  - [ ] Prototype with test footage

---

## Backlog (Not Scheduled)

- [x] ~~More sophisticated XMP sidecar format~~ - Now uses exiftool (preserves metadata)
- [x] ~~RAW file support~~ - Completed with automatic preview extraction
- [ ] Confidence score histogram/analysis tool
- [ ] Training visualization (loss curves, confusion matrix)
- [ ] Batch processing GUI (if needed for non-technical team members)
- [ ] Performance benchmarking (inference speed on various image sizes)
- [ ] Multi-GPU training (not needed for current dataset size, but future-proofing)

---

## Notes

- **Git workflow**: Commit after each major milestone (data prep, training complete, integration tested)
- **Model versioning**: Keep numbered backups (model_v1.pt, model_v2.pt, etc.) with corresponding notes
- **Data backup**: Ensure Lightroom catalog is backed up before running exports
- **Performance**: On M4 Max, expect:
  - Data prep: <1 minute
  - Training (15 epochs): ~30-45 minutes
  - Inference (10,000 frames): ~30-60 seconds (depending on NEF processing)

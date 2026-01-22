# Status

## Current Phase: Production Ready

### ✅ Completed

- [x] Model trained on Lime Rock data (select/reject classes)
- [x] Batched inference with MPS (Apple Silicon) support
- [x] Burst deduplication (picks best frame per burst)
- [x] Tiered keyword output (robo_97, robo_98, robo_99)
- [x] JPEG keyword embedding via exiftool
- [x] XMP sidecar support for RAW files
- [x] Consolidated `classify.py` pipeline
- [x] Tested on 42k image dataset (Lime Rock)

### Results (Lime Rock Test)

```
Total images:      41,798
Classified select: 35,746 (85.5%)
Classified reject:  6,052 (14.5%)
Burst winners:     10,043 (select only)

Keyword tiers:
  robo_99 (≥0.99):    329
  robo_98 (≥0.98):  1,367
  robo_97 (≥0.97):  1,755
  Below threshold:  6,592

Burst siblings tagged 'select': 13,663
```

### Workflow

1. Copy images from camera cards to working directory
2. Run: `source .venv/bin/activate && python classify.py /path/to/images`
3. Import into Lightroom
4. Use smart collections (robo_99, robo_98, robo_97, select) to review

### Known Limitations

- Lightroom ignores XMP sidecars for JPEGs (must embed keywords directly)
- NEF processing requires ImageMagick for preview extraction
- Model trained on racing footage; may not generalize to other domains

### Future Work

- [ ] Real-time Z9 attachment over USB/network
- [ ] Model retraining with Sebring data
- [ ] Confidence threshold tuning based on field experience

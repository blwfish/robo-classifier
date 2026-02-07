# Status

## Current Phase: Production Ready

### ✅ Completed

- [x] Model trained on Lime Rock data (select/reject classes)
- [x] Batched inference with MPS (Apple Silicon) support
- [x] Burst deduplication (picks best frame per burst)
- [x] Tiered keyword output (robo_90 through robo_99 - 10 tiers)
- [x] Hierarchical keyword structure (`AI keywords|robo|{keyword}`)
- [x] JPEG keyword embedding via exiftool
- [x] XMP sidecar support for RAW files (NEF, CR2, ARW, etc.)
- [x] Automatic RAW preview extraction with exiftool stay_open mode
- [x] Time-based burst grouping (optional, adaptive to shutter speed)
- [x] Recursive subdirectory search
- [x] Consolidated `classify.py` pipeline
- [x] Tested on 42k image dataset (Lime Rock)

### Results (Lime Rock Test)

```
Total images:      41,798
Classified select: 35,746 (85.5%)
Classified reject:  6,052 (14.5%)
Burst winners:     10,043 (select only)

Keyword tiers (expanded to robo_90-99):
  robo_99 (≥0.99):    329
  robo_98 (≥0.98):  1,367
  robo_97 (≥0.97):  1,755
  robo_90-96:       (would capture more if re-run with new tiers)
  Below 0.90:       6,592

Burst siblings tagged 'select': 13,663
```

### Workflow

1. Copy images from camera cards to working directory (JPG, RAW, or mixed)
2. Run: `source .venv/bin/activate && python classify.py /path/to/images`
   - Optional: Use `--burst_threshold 0.5` for time-based burst grouping
3. Import into Lightroom (automatically picks up XMP sidecars for RAW)
4. Use smart collections with hierarchical keywords:
   - `AI keywords > robo > robo_99` (and 98, 97, ... down to 90)
   - `AI keywords > select` (all frames from interesting bursts)

### Known Limitations

- Lightroom ignores XMP sidecars for JPEGs (must embed keywords directly - now handled automatically)
- Model trained on racing footage; may not generalize to other domains
- Time-based burst grouping requires valid EXIF timestamps (falls back to filename-based if missing)

### Future Work

- [ ] Real-time Z9 attachment over USB/network
- [ ] Model retraining with Sebring data
- [ ] Confidence threshold tuning based on field experience

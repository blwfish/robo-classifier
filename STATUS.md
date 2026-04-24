# Status

## Current Phase: Review App (UI-driven), post-MVP optimization

### ✅ Completed this phase

- [x] **Junk filter**: YOLOv8-based pre-pass detects frames with no vehicle
      or only edge-clipped vehicles, moves them into `junk/`. Tunable edge-area
      rule (default 0.05) — user explicitly wanted aggressive filtering.
- [x] **FastAPI review UI** at `localhost:8765`:
  - Folder browser (`/api/ls`, persists last-used dir)
  - Pipeline runner with SSE progress + per-stage timestamps + live elapsed counter
  - Threshold tuner screen: histogram, dual sliders, tier chips, filtered thumb grid,
    Write-keywords / Dry-run buttons with `clear_first` safety for re-tuning
  - Grid screen: filter (winners/all/selects), sort by confidence
  - Detail screen: arrow-key nav, `1-5` color labels, `C` crop, `P` preview-crop, `X` junk
  - All review state persisted as Lightroom-compatible XMP (`crs:Crop*`, `xmp:Label`)
- [x] **Model profiles**: `models/<name>.pt` + `--profile` CLI / UI dropdown
- [x] **Hardware tuning presets**: `presets/<name>.toml` + `--preset` CLI / UI dropdown
      Shipped: `m1-max`, `nvidia-desktop`, `cpu-only`; see `presets/CUSTOM.md`.
- [x] **Performance**:
  - rawpy replaces exiftool for RAW previews (27% faster; no subprocess/pipes)
  - 512px preview downsample at extraction (cuts YOLO decode ~13×)
  - Parallel YOLO image decode (ThreadPoolExecutor feeds predict with ndarrays)
  - Thumb pregeneration during pipeline so grid opens instantly
- [x] Regression tests kept passing throughout (62 pytest cases)

### Benchmarks (Road Atlanta Turn 3, 17k NEFs, M1 Max via USB 3.2)

Per-stage budget with current defaults:

| Stage | Rate | 17k total |
|---|---|---|
| Preview extract | ~30 f/s (I/O-bound on USB 3.2, ~450 MB/s) | ~9 min |
| YOLO junk filter | 225 f/s (post parallel-decode) | ~75s |
| ResNet classifier (~11k survivors) | ~100 f/s | ~2 min |
| Thumb pregen | parallel with classify | free |
| **Total pipeline** | | **~12 min** |

(~15-min baseline before preview downsample + parallel decode.)

Extract is I/O-bound on the 450 MB/s USB 3.2 bus — further CPU-side optimizations
won't move that bar. Internal NVMe or Thunderbolt external would give ~3×.

### Tunable via preset

- `extract.workers`, `extract.max_preview_edge`
- `junk_filter.batch_size`, `imgsz`, `min_conf`, `min_visible_frac`,
  `min_area_frac`, `edge_min_area_frac`
- `classifier.batch_size`, `num_workers`

### Deferred / not doing

- **In-memory preview pipeline refactor** (numpy arrays instead of tempdir JPEGs):
  analyzed and would save ~3 min on a 17k shoot, but bench showed we're
  I/O-bound on the source drive, not CPU-bound. Refactor cost not justified
  for Brian's rig; revisit if someone's bottleneck is CPU-side.
- Real-time Z9 attachment over USB/network.
- Model retraining with post-junk-filter survivors.

### Known constraints

- Lightroom ignores XMP sidecars for JPEGs (handled: embed directly).
- Detail view re-extracts RAW preview via exiftool on first open per image
  (slow once per RAW, cached thereafter). Low priority — review is a one-shot.
- Time-based burst grouping requires valid EXIF timestamps (falls back to filename).

### Next steps (when resumed)

1. First real-world test at Lime Rock (~41k NEFs) — timestamps will show
   exactly where time goes at that scale.
2. Vic (Windows + nVidia, 2-3 robo cams) pulls latest + runs with the
   `nvidia-desktop` preset, then tunes his own `vic-rig.toml`.
3. Deferred UI work (see `UI_PLAN.md`): training a new model from the UI
   (wrap `prepare_training_data.py` + `train_classifier.py`).

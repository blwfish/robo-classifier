# Custom tuning presets

Tuning presets are TOML files that bundle hardware-specific pipeline settings
(worker counts, batch sizes, preview sizes, junk-filter thresholds) so you can
swap them in by name instead of memorizing a dozen CLI flags.

## Creating your own

1. Copy one of the shipped presets as a starting point:
   ```
   cp presets/m1-max.toml presets/my-rig.toml
   ```
2. Edit the values to match your setup. See the other files for what each
   section controls.
3. Use it from the CLI or the UI dropdown:
   ```
   python classify.py /path/to/shoot --preset my-rig
   ```

## When to tune what

| Bottleneck | What to adjust |
|---|---|
| Source drive I/O | `extract.workers` (more doesn't help once the bus saturates); consider staging files to a faster disk |
| YOLO / GPU | `junk_filter.batch_size`, `junk_filter.imgsz` |
| CPU-bound image decode | `extract.max_preview_edge` — smaller = faster decode downstream |
| Classifier throughput | `classifier.batch_size`, `classifier.num_workers` |
| False-positive edge chops | `junk_filter.edge_min_area_frac` (lower = keep more edge cars) |

## Precedence

Explicit CLI flags override the preset. The preset overrides code defaults.
So a preset sets your baseline, and you can override any single knob for a
one-off run without editing the file.

## Not tuned here (on purpose)

- `--model` / `--profile` — that's series-specific (PCA vs. IMSA etc.), not
  hardware, so it's a separate flag.
- `--burst_threshold` — shooting-style, not hardware.
- `--input_dir` — obvious, per-run.

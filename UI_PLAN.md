# UI Plan — Running Notes

Captures UI-related wishes mentioned in conversation so they don't get lost.
Not a spec; a checklist to refine before building the UI.

## Purpose
First-order screening for PCA (and other) racing shoots, before anything
enters Lightroom. Wraps the CLI pipeline so the user doesn't have to
remember Python invocations.

## Core flows

### 1. Run the pipeline
- Pick input directory (JPEG or RAW).
- **Tickbox: run junk filter** (default on). Junk goes to a `junk/`
  subfolder of the input dir.
- **Pick a model profile** from a dropdown (sourced from `models/`).
- Kick off the run; show progress.

### 2. Browse results
- Crude image browser over the survivors.
- Per-image actions: select / color-label / crop.
- Optionally also browse the `junk/` folder to sanity-check the filter
  (expected to be rarely used once trust is established).

### 3. Train a new model
- UI wrapper around `prepare_training_data.py` + `train_classifier.py`.
- Pick `select/` and `reject/` dirs, name the profile, go.
- Deferred — add after the core run + browse flows work.

## Deferred decisions

- Junk-filter sensitivity (e.g. flagging tiny background-only cars) — punt
  to a UI toggle later rather than baking thresholds in now.
- Color label semantics (what each color means) — TBD when we build the
  browser.
- Crop tool behavior (aspect ratio, non-destructive, export format) — TBD.

## Out of scope (for now)
- Lightroom integration beyond the existing keyword-writing.
- Multi-user / cloud.

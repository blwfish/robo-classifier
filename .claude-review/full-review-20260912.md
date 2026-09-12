# Full Review — robo-classifier core pipeline
Scope: app_config.py, classify.py, image_utils.py, inference_hotel.py, ingest.py, junk_filter.py, perf.py, prepare_training_data.py, presets_loader.py, review.py, train_classifier.py, write_tiered_keywords.py, ui/*.py, tests/. Excludes air3_ingest/ (separate tool, reviewed independently 2026-07-02).

Pass ID: `robo-classifier-20260912-a1f3`

## Summary
| Pass        | Critical | High | Med | Low | Status |
|-------------|----------|------|-----|-----|--------|
| Code review | 1 | 11 | 18 | 11 | ✓ ran (9 angles) |
| Interface   | 1 | 3 | 3 | 6 | ✓ ran |
| Inventory   | 0 | 0 | 20 | 36 | ⚠ PARTIAL (9/15 modules — user-selected scope, disclosed in advance; skipped: app_config.py, presets_loader.py, perf.py, train_classifier.py, prepare_training_data.py, ui/app.py) |
| Test review | 0 | 4 | 3 | 2 | ✓ ran |
| Coverage    | — | — | 9 | — | ✓ ran (floor-check only) |
| **Total**   | **2** | **18** | **53** | **55** | **128** |

(Note: 1 of Phase 1's 2 Critical findings and 1 of Phase 1.5's Critical are distinct — see Critical section. Several High/Medium findings across phases corroborate the same underlying defect from different angles; each is listed once at its highest-informative phase, with cross-references noted.)

## Critical

- [code] review.py:429-466 — Flask `/api/write_keywords`'s "dry run" flag only changes a print prefix; `write_keywords()` has no `dry_run` parameter, so a preview request writes real keywords into production photo metadata with no way to undo. Traced to the UI's first commit; never fixed despite later passes touching this function. [robo-classifier-20260912-a1f3#01]
- [code] ingest.py:284-288,405-417 — `_known_hash` dedup-skip never verifies the recorded `dest_path` still exists on disk; if an already-ingested file is later moved/deleted, re-ingesting the same card silently and permanently skips it — irrecoverable data loss in the tool's core anti-duplicate path. [robo-classifier-20260912-a1f3#02]
- [interface] ui/static/app.js:1277-1284 vs ui/app.py `WriteKeywordsRequest.burst_threshold` — the "Write Keywords" UI action never re-sends `burst_threshold`; every write-keywords call silently falls back to filename-based burst grouping even when the run used time-based grouping, mis-tagging which frames are "select" vs siblings. 100% reproducible for any time-based-grouped run. [robo-classifier-20260912-a1f3#03]

## High

- [code] classify.py:1108-1114 / image_utils.py:178-179 — `--max_preview_edge 0` (documented as "disable downsampling") triggers `Image.thumbnail((0,0))` → `ZeroDivisionError`, silently swallowed, breaking ALL RAW preview extraction for the run. [#04]
- [code] review.py:206-215 — `score = overlap + (1000 if name_match else 0)`, recovered via `% 1000`; corrupts the small-JPG-proxy-dir acceptance gate once overlap >= 1000, plausible at this tool's 5000+-file scale. [#05]
- [code] ui/app.py:336 + `_load_winners_csv` — absolute vs. possibly-relative path string comparison; `is_winner` silently `False` for every image when winners.csv was produced by a CLI run with a relative input_dir. Independently rediscovered by both Phase 1.5 passes. [#06]
- [code] classify.py:250-260 `get_capture_times` — one bad exiftool chunk (of up to 500 paths) discards ALL previously-successful chunks' data, reverting burst-grouping for the whole run. [#07]
- [code] classify.py:349-369 `burst_dedup_by_time` — a frame with no EXIF timestamp forces the *next* real-timestamped frame into a new burst too, silently splitting a contiguous burst and producing duplicate "winners." [#08]
- [code] classify.py:732 `resolve_model` / ui/pipeline_runner.py:85 — `resolve_model` raises `SystemExit` (a `BaseException`), uncatchable by the job runner's `except Exception`; the UI shows a job stuck at "running" forever with no error surfaced. [#09]
- [code]/[interface] classify.py:719-745 `resolve_model` / ui/app.py:461-464 `write_keywords_endpoint` / ui/train_runner.py — pipeline/keyword-sidecar lookups only ever check the hardcoded `MODELS_DIR`, never `app_config.model_library`; once a user trains a model via the UI's configured library, the pipeline can never find it. Two call sites share this root cause. [#10]
- [code] ui/review.py:33-34 `get_roll_angle` — asserts an unverified RollAngle/CropAngle sign-convention as fact with no test/citation; writes crop-correction metadata into every processed image's real XMP on every run. This project's own history includes a prior incident of an entire mechanism built on an unverified Lightroom-metadata assumption. [#11]
- [code] classify.py / inference_hotel.py / ui/review.py — three duplicated RAW/JPEG XMP-target-routing implementations; inference_hotel's copy has already diverged to flat select/reject tagging, permanently missing the tiered-keyword scheme, with no parity test to catch further drift. [#12]
- [code] review.py vs ui/thumbs.py — divergent parallel thumbnail implementations: review.py has no mtime cache invalidation and no `exif_transpose` (stale/wrongly-rotated thumbnails possible) while ui/thumbs.py does both correctly; undocumented, no parity test. [#13]
- [code] classify.py `run_pipeline` / ui/app.py `write_keywords_endpoint` — dry-run logic reimplemented 3x from scratch instead of sharing real code; this exact drift pattern already produced the review.py Critical (#01) — proof the risk isn't theoretical. [#14]
- [interface] ui/app.py:461-464 `write_keywords_endpoint` — same hardcoded-MODELS_DIR bug as #10, recurring for the accept/reject-keyword JSON sidecar lookup. [#15]
- [interface] ui/static/app.js (`state.profile` set only in `onRun()`, never restored by `tryRestoreLastSession()`/`openSession()`) — reopening a results directory in a fresh page load silently drops model identity; configured accept/reject keywords for that model are silently skipped on write. [#16]
- [interface] ingest.py:373,470 vs ui/app.py:727-740 vs ui/ingest_runner.py:79-81 — `ingest()` emits its own `__end__` event before the outer runner finalizes `job.summary`/`job.status`; a narrow race can show "Error: unknown" in the UI for a successful ingest. [#17]
- [test] review.py:89 `compute_bursts_and_winners` — zero test coverage for the entire 307-line file; no parity test against classify.py's `burst_dedup`, whose logic it duplicates with at least one confirmed omission (the classification=='select' filter). [#18]
- [test] inference_hotel.py:89 vs classify.py:443 `_keyword_hierarchy` — hand-duplicated with cross-referencing docstrings instead of a shared import; no parity test. [#19]
- [test] image_utils.py (whole file) — zero test coverage, including the unpinned `max_preview_edge` threshold whose `0` value crashes (Critical-adjacent finding #04). [#20]
- [test] ui/pipeline_runner.py / ui/ingest_runner.py / ui/train_runner.py — three parallel Job/JobManager implementations, zero test coverage on any of the three, no parity check. [#21]

## Medium (53 total — condensed; full detail in phase working files)

Notable items: TOML backslash-escaping bug in app_config.py risking silent config loss for the project's Windows collaborator; inference_hotel.py's RAW support hardcoded to .nef only; junk_filter.py's `max_conf` CSV column mixing float/empty-string types; write_tiered_keywords.py's `.get()` default-substitution bug silently treating rows with missing `classification` as valid rejects; ui/thumbs.py's inconsistent error handling between its on-demand and batch thumbnail paths; 9 files at literal 0% test coverage (prepare_training_data.py, review.py, train_classifier.py, ui/app.py, ui/ingest_runner.py, ui/pipeline_runner.py, ui/thumbs.py, ui/train_runner.py, write_tiered_keywords.py); the Job/JobManager triplication (Q1, strongest reuse-angle convergence at 4 independent hits); per-file exiftool subprocess spawns instead of reusing the persistent `ExiftoolProcess`; three duplicate SSE progress generators in ui/app.py; several duplicated classification tables (RAW/JPG extension sets, PHOTO_EXTENSIONS) at risk of silent drift.

Full itemized list: see `/private/tmp/claude-501/-Volumes-Files-claude-robo-classifier/3719c266-4e21-40fe-9603-71479d095271/scratchpad/phase1_deduped_findings.md`, `phase1_5_interface_findings.md`, `phase2_deduped_findings.md`, and the Phase 3 agent transcript for exact file:line citations and severities.

## Low (55 total — condensed)

Mostly docstring/comment inaccuracies (stale diagnostic messages, contradictory `>=`/`>` documentation already counted at Medium above where it's the same underlying defect), dead code (unused `raw_exts` variable, unused `_touches_edge` function), minor duplication (tier-counting scaffolds, train/eval loop near-duplication), and low-exposure gaps in inference_hotel.py (a standalone tool not wired into the production pipeline — confirmed via grep, only its own unit tests reference it).

---
Note: interface check independently rediscovered 3 of Phase 1's correctness findings (the MODELS_DIR/model_library mismatch, the SystemExit bug, and the is_winner path-resolution bug) — corroboration across passes, not double-counted in the totals above. Test review's parity-test gaps corroborate Phase 1's reuse-angle (Q1, Q2, Q5) findings, confirming those duplications aren't just style nits.

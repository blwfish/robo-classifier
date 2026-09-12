# Phase 1 deduped findings — robo-classifier core pipeline
Merged from 9 angle passes (1=general, 2=shallow, 3=git-blame, 4=prior-fix-precedent, 5=CLAUDE.md-compliance, 6=reuse, 7=simplification, 8=efficiency, 9=altitude).
"Hit by" lists which angles independently found it.

## Correctness findings (from angles 1-5, cross-corroborated by 6-9 where noted)

1. **review.py:429-466 `/api/write_keywords`** — the Flask "dry run" flag only changes a print prefix; `write_keywords()` has no `dry_run` param at all, so a dry-run request writes real keywords. Traced back to the UI's first commit (bbf066e); the FastAPI `ui/app.py:494` and `write_tiered_keywords.py` implementations both correctly gate on dry_run — this is the one that was never fixed. Hit by: 3 (angle).

2. **classify.py:1108-1114 / image_utils.py:178-179** — `--max_preview_edge 0` is documented as "disable downsampling" but `Image.thumbnail((0,0))` raises `ZeroDivisionError` inside Pillow (repro'd). Swallowed by a broad `except Exception` in `_extract_one_rawpy`, so it silently breaks ALL RAW preview extraction instead of erroring visibly. Hit by: 2.

3. **ingest.py:407-417 `_known_hash`** — dedup-skip never verifies the previously-recorded `dest_path` still exists on disk. If a user deletes/moves an already-ingested file, re-ingesting the same card silently skips it forever with no copy anywhere — silent permanent data loss. Hit by: 2.

4. **ingest.py:378-393** — only `PHOTO_EXTENSIONS` get EXIF lookups; `VIDEO_EXTENSIONS` files never do, so every ingested video falls back to `ts="00000000000000", alias="Unknown"` — wrong names/sort order for an entire file class, not an edge case. Hit by: 1 (angle 2), but structural/clear.

5. **review.py:206-215 `_find_small_jpg_dir`** — `score = overlap + (1000 if name_match else 0)`, recovered via `% 1000`. Breaks silently once real overlap >= 1000 (plausible at this tool's stated 5000+-file scale) — corrupts the 10%-match acceptance gate. Hit by: 2 (angles 1, 2).

6. **prepare_training_data.py:131-145** — `--select_dir`/`--reject_dir` are `required=True` in argparse, making the documented `--data_dir` fallback path unreachable. Repro'd: the exact command in this project's own CLAUDE.md ("Training" section) fails with an argparse error. Hit by: 1.

7. **ui/app.py:336 + `_load_winners_csv`** — `is_winner = str(f) in winners` compares an absolute resolved path against whatever (possibly relative) path string is in winners.csv. If winners.csv was produced by a CLI run with a relative input_dir, every image silently shows `is_winner: False` in the UI even though real winners exist. Hit by: 1.

8. **inference_hotel.py:148/213/253 vs :303** — docstring/code use `>=` for the confidence threshold; CLI `--help` text says `>`. Threshold-boundary contract contradiction (project's own Threshold-Boundary Testing Rule calls this out explicitly). Hit by: 3 (angles 1, 2, 5) — strong convergence.

9. **classify.py:250-260 `get_capture_times`** — exiftool calls are chunked at 500 paths (ARG_MAX safety), but if ANY chunk returns nonzero exit code, the ENTIRE result is discarded (`return None`) even though earlier chunks parsed fine. One bad file among thousands silently reverts burst-grouping for the whole run to filename-based. Hit by: 2 (angles 1, 5).

10. **ingest.py:196-224 `_read_exif_batch`** — unlike classify.py's chunked equivalent, sends the ENTIRE photo-path list to exiftool in one subprocess call with no ARG_MAX chunking. Risk of uncaught `OSError` (E2BIG) on a large card — not in the caught exception set (`FileNotFoundError`, `TimeoutExpired`, `JSONDecodeError`), so it would propagate and abort the whole ingest. Hit by: 2 (angles 1, 5) — parallel-implementation-rule violation against classify.py's own safeguard.

11. **ingest.py:407 vs :435** — `_known_hash`'s DB read is only wrapped in `except OSError`, while the adjacent `_record()` call is wrapped in `except sqlite3.OperationalError` for the same lock-contention failure mode. `sqlite3.OperationalError` is not an `OSError` subclass — a locked DB during `_known_hash` escapes the per-file try/except, aborts `ingest()` for all remaining files, and skips `conn.close()` (connection leak). Hit by: 1.

12. **classify.py:118 `ImageDataset.__getitem__`** (shared with inference_hotel.py via import) — catches only `(OSError, SyntaxError, ValueError)`; Pillow's `DecompressionBombError` is none of these, so a single such file crashes the DataLoader worker and aborts the whole batch instead of being recorded as `decode_failed`. Hit by: 1.

13. **inference_hotel.py:195-196** — decode failures in the batched-JPEG loop are dropped with `continue` and no CSV row/counter/log — unlike classify.py's `run_inference()`, which records every failure. Silent data loss in the count. Hit by: 1.

14. **write_tiered_keywords.py** (whole file) — never calls `clear_robo_keywords()` before writing (doesn't even import it), unlike `classify.write_keywords()` which explicitly does this "so tier changes after threshold tuning don't leave stale tags behind." Re-running this standalone script after a threshold change leaves stale/duplicate `robo_9x` tags. Hit by: 1 (angle 1), corroborated by angle 5's related finding (#16 below) that write_tiered_keywords.py's reject-scope also silently differs from classify.py's.

15. **app_config.py:100-123** — `_save`/`_parse_toml` escape `"` but not `\`; a config value ending in a literal backslash (e.g. a Windows path `D:\models\`) corrupts on round-trip and is silently dropped on next load (regex fails to match). Repro'd. Directly relevant: this repo has a documented Windows collaborator (Vic) per project memory. Hit by: 1 (angle 4), plus angle 9 independently flagged the same TOML parser as architecturally wrong (see quality section) and angle 5 flagged its comment as inaccurate.

16. **write_tiered_keywords.py vs classify.py `write_keywords()`** — reject-keyword scope silently differs: write_tiered_keywords.py tags every non-winner row as reject; classify.py's version restricts that to non-winners within *qualifying bursts* only. Same nominal feature, two undocumented-divergent implementations (Parallel Implementation Rule violation). Hit by: 1.

17. **classify.py:349-369 `burst_dedup_by_time`** — a frame with no EXIF timestamp forces the *next* real-timestamped frame into a new burst too (not just the timestamp-less frame itself), silently splitting an otherwise-contiguous burst and producing duplicate "winners." The known/tested fix (86a15b8) covers only the missing-timestamp frame itself, not this side effect on its neighbor. Hit by: 1 (git-blame angle, high-confidence historical trace).

18. **classify.py:271-291** — `SubSecTimeOriginal` is read once and applied regardless of which date field ultimately succeeds (`DateTimeOriginal` vs `CreateDate` fallback). If `DateTimeOriginal` fails but `CreateDate` succeeds, a subsec value that belongs to `DateTimeOriginal` gets misapplied to `CreateDate`'s timestamp — up to 1s of burst-timing error. Interaction between two independently-correct historical fixes (4db4c7f + 6d1466e) that never revisited each other. Hit by: 1 (git-blame).

19. **classify.py:732 `resolve_model` / ui/pipeline_runner.py:85`** — `resolve_model()` raises `SystemExit` (not `RuntimeError`, contradicting `run_pipeline`'s own docstring "Raises on fatal errors"). The background job runner only catches `except Exception`; `SystemExit` is a `BaseException` and escapes uncaught. Job status is never set to "error," but `finally` still emits `__end__` — UI shows the job stuck at `status="running"` forever with no error surfaced. Hit by: 1, but a concrete, reproducible UI-hang bug.

20. **classify.py:719-745 (`resolve_model`) / ui/app.py / ui/train_runner.py** — newly-trained models save into `app_config.model_library` (user-configured dir), but `resolve_model`/pipeline lookups only ever check the hardcoded repo-local `MODELS_DIR`. Once a user configures a non-default `model_library` and trains via the UI, the pipeline can never find that model — an interface-contract gap between the training and inference sides. Hit by: 1, but a clean end-to-end trace.

21. **classify.py:976-978** — stale diagnostic message still says "(all files are missing DateTimeOriginal)" after 6d1466e added a 3-field fallback chain (DateTimeOriginal→CreateDate→FileModifyDate); misleads debugging. Hit by: 1.

22. **classify.py:1271-1274** — dry-run summary print is unconditional and prints the literal string `None` for results_csv/winners_csv (introduced as a side effect of 6d1466e's None-initialization fix for the dry-run UnboundLocalError). Hit by: 1.

23. **image_utils.py:244-248** — `extract_raw_previews()`'s docstring says it returns a single dict; it actually returns `(preview_map, failed)` except on the empty-input early-return path, which returns a bare `{}` — inconsistent with the rest of the function's own contract. Currently unreachable given both call sites guard with `if raw_files:` first, but a live trap for any future caller. Hit by: 2 (angles 1, 5 — docstring emphasis).

24. **junk_filter.py:171** — docstring is self-contradictory ("one per input path" + "skipped silently") and factually wrong: decode failures ARE appended with `reason='decode_failed'` and logged with warnings, not skipped. Hit by: 1.

25. **ui/review.py:33-34 `get_roll_angle`** — docstring asserts a RollAngle/CropAngle sign-convention as fact with no verification, test, or citation — unverified external-system assumption (this project's own Spec Review Rule calls this the highest-severity category, citing this exact project's prior Lightroom-metadata incident). Compounds with #26 below (Z9-specific portrait-angle special case, same function). Hit by: 1.

26. **ui/review.py:24-26,48-49 `_PORTRAIT_THRESHOLD`** — a hardcoded ±90°/10° Z9-specific RollAngle convention with no camera-model check, in a codebase (ingest.py) that already documents many other camera bodies with potentially different RollAngle conventions. No self-verification against `CameraModelName`. Hit by: 1.

27. **ingest.py:33** — `"NIKON Z 6_3": "Z6III"` camera-alias entry is explicitly commented `# placeholder — verify against real file`, i.e. an admittedly-unverified assumption shipped into a table that determines permanent renamed filenames. Hit by: 2 (angles 5, 9).

28. **perf.py:139-145 `_storage_class_macos`** — the function's own docstring promises `"usb3"`/`"usb2"` as distinct return values, but the implementation always falls through to `return "usb3"` regardless of the match — `"usb2"` can never actually be returned. Also a fragile `"10" in line` substring match with no anchoring. Hit by: 1.

29. **Duplicated classification tables** (Syntactic-Semantic Seam Rule violations — same semantic distinction, multiple hand-maintained copies that can silently drift):
    - `ingest.py:96-99 PHOTO_EXTENSIONS` vs `image_utils.py:18-20 RAW_EXTENSIONS/JPG_EXTENSIONS` — already divergent (adds .jpg/.jpeg redundantly, omits .png). Hit by 2 (angles 5, 9).
    - `inference_hotel.py:157-163` hardcodes RAW support to only `.nef`/`.NEF` vs the full 8-extension `RAW_EXTENSIONS` set everyone else uses. Hit by 2 (angles 5, per-angle).
    - `ui/app.py:592 train_data_stats` — hardcoded, case-sensitive `{".jpg",".jpeg",".JPG",".JPEG"}` instead of the codebase's `.lower()`-normalized pattern; undercounts vs what training will actually process. Hit by 3 (angles 1, 5, 9) — strong convergence.
    - `review.py:122` — a computed-but-unused `raw_exts` (upper+lower union) sits next to the real, correctly-used `RAW_EXTENSIONS` check. Dead duplicate. Hit by 2 (angles 5, 9/7).

## Data-capture modules flagged for Phase 2 inventory (union across all 9 angles)
ingest.py, classify.py, image_utils.py, junk_filter.py, inference_hotel.py, review.py, ui/review.py, ui/thumbs.py, perf.py, app_config.py, presets_loader.py, write_tiered_keywords.py, train_classifier.py/prepare_training_data.py, ui/app.py.

---

## Quality findings (angles 6-9: reuse / simplification / efficiency / altitude)

Q1. **Job/JobManager triplication** — `ui/pipeline_runner.py`, `ui/train_runner.py`, `ui/ingest_runner.py` each independently define a near-identical dataclass+manager (id/events queue/status/summary/error/_thread, `_jobs` dict + lock, create/get/start, identical try/except/finally + `__end__` sentinel protocol). `ingest_runner.py`'s own docstring admits "Mirrors the pattern of pipeline_runner.py." Already-visible drift: only train_runner's status enum includes "preparing"/"training" instead of just "running." Hit by 4 angles (6,7,9, and independently noted again) — the single strongest convergence in the whole review; a shared `BaseJob`/generic `JobManager` would remove ~25-30 duplicated lines x3 and stop future protocol drift.

Q2. **Three duplicated RAW/JPEG XMP-target-routing implementations** — `classify.py` (write_xmp_sidecar/embed_keyword_in_jpeg), `inference_hotel.py` (hand-copied from classify.py, already diverged: inference_hotel's copy never gained the tiered-keyword scheme, writes flat select/reject only — a real behavioral regression, not just style), and `ui/review.py`'s `_target_for_write`/`_run_exiftool` (third independent copy). Hit by 3 angles (6, 9, and cross-referenced by correctness finding on inference_hotel's write_xmp_sidecar).

Q3. **inference_hotel.py duplicates classify.py wholesale**: `_keyword_hierarchy` (line-for-line copy, docstring literally says "matches classify._keyword_hierarchy" instead of importing it, despite already doing `from classify import ImageClassifier, ImageDataset`), `write_xmp_sidecar` (near-verbatim copy, already diverged error-handling: prints vs silently returns False), RAW preview extraction (ImageMagick `convert` instead of `image_utils.extract_raw_previews`), and image discovery (hand-rolled glob instead of `image_utils.find_images()`, only catches NEF). Hit by 3 angles (6, 9, corroborated by correctness angle 1's inference_hotel findings). Given how much of this file is a superseded/legacy duplicate with real behavioral drift, worth asking whether inference_hotel.py should be deprecated rather than patched.

Q4. **Per-file exiftool subprocess spawns instead of reusing `ExiftoolProcess`** (the `-stay_open` persistent-process class `image_utils.py` was built specifically to avoid this cost): classify.py's `write_xmp_sidecar`/`embed_keyword_in_jpeg`/`clear_robo_keywords`/`write_keywords()` hot loops; `write_tiered_keywords.py`'s same pattern; `review.py`'s `preextract_nef_previews`/`get_or_create_thumb` (serial, one exiftool spawn per file — `ui/thumbs.py` in the same repo does this correctly via `ExiftoolProcess`); `ui/thumbs.py`'s `get_thumb` itself instantiates+tears down a fresh `ExiftoolProcess` per HTTP request instead of sharing one. Hit by 2 angles (6, 8) with multiple call sites each — real perf impact at the tool's stated 5000+-file scale.

Q5. **review.py vs ui/thumbs.py — divergent parallel thumbnail implementations**, not just duplicated code: review.py's thumbnails have no mtime-based cache invalidation (can go stale) and no `ImageOps.exif_transpose` (can be rotated wrong), while ui/thumbs.py's does both correctly. Undocumented as an intentional simplification. Hit by 2 angles (6, 9) — this is a correctness-adjacent quality finding per the severity calibration (divergent parallel implementation with no test enforcing parity).

Q6. **app_config.py's hand-rolled regex TOML parser** exists specifically to "avoid a toml dependency" (per its own comment) while `presets_loader.py` in the same codebase already uses stdlib `tomllib`. The comment is also inaccurate (angle 5): the real constraint is that `tomllib` is read-only and this module needs to write, not "avoiding a dependency." This is also the parser with the correctness bug (finding #15 above, backslash-escaping). Hit by 3 angles (4-correctness, 5-doc-accuracy, 9-architecture) — same code hit from three different directions.

Q7. **junk_filter.py: `_touches_edge()` is dead code, `_is_edge_chopped()` re-derives the identical edge-touch logic inline** instead of calling it. Tests exercise the unused function directly, so production logic and tested logic can silently diverge. Hit by 2 angles (1, 6/7).

Q8. **Three near-identical SSE `event_stream()` generators in `ui/app.py`** (progress/train-progress/ingest-progress routes) — byte-for-byte identical poll/heartbeat/drain/emit logic; also, all three call the blocking `queue.Queue.get(timeout=0.25)` directly inside `async def` with no executor offload, which can stall the event loop under concurrent SSE clients. Hit by 2 angles (1-correctness/efficiency framing, 7-simplification framing).

Q9. **Tier-counting/summary scaffolding hand-copied 3-4x** across classify.py, write_tiered_keywords.py, ui/app.py (`tier_counts = {f"robo_{i}": 0 for i in range(90,100)}` + the tiers-99-to-90 print loop). Hit by 1 (angle 7).

Q10. **dry-run logic reimplemented 3x**: classify.py's `run_pipeline` dry-run branch and ui/app.py's `write_keywords_endpoint` dry-run branch both independently reimplement what `write_keywords()` does for real (build reverse lookup, walk winners, assign tier, accumulate qualifying bursts) purely to simulate a preview — risk of drift from real behavior, and relates directly to correctness finding #1 (review.py's dry-run being a no-op) as the pattern's most severe instance. Hit by 1 (angle 7).

Q11. Misc smaller duplication/simplification (each hit by 1 angle, Low/Medium): `write_keyword_to_file`/`clear_robo_keywords` duplicate an 8-line nef_dir-resolution block (classify.py); `train_classifier.py`'s `train_epoch`/`evaluate` are near-identical loops differing only in train/eval mode and backward pass; `ingest.py detect_cards` Darwin/Windows branches duplicate dict construction; `app_config.py`'s three properties (`model_library`/`dataset_scratch`/`default_profile`) repeat the same get/strip/empty-check pattern; `classify.py burst_dedup_by_time`'s double `burst_id += 1` is confusing but intentional (readability nit, not a bug).

Q12. Efficiency-only findings not otherwise listed: `ingest.py` hashes then separately copies each file (2 full reads of possibly slow USB/card media instead of streaming hash-while-copy); the same source directory gets recursively walked 3 independent times across `detect_cards`→`_count_files`, `/api/ingest/scan`, and `ingest()` itself; `inference_hotel.py`'s NEF path is fully serial (subprocess extraction + single-image forward pass) unlike the batched JPG path a few lines above — itself another symptom of Q3's "this file is a superseded legacy duplicate"; `classify.py:178-183` re-decodes an already-failed image a second time solely to recover the exception class name.

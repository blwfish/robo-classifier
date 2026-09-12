# Phase 2 deduped inventory findings — 9 high-risk data-capture modules
3 haiku passes per module, deduped. "Hit by" = convergence count across the 3 passes for that module.

## ingest.py

P2-1. **ingest.py:204-232** — exiftool requested only 4 tags (DateTimeOriginal, CreateDate, CameraModelName, Model); no enumeration/documentation of what other EXIF/maker-note/XMP/GPS fields exist and are dropped. Hit by 3/3.

P2-2. **ingest.py:220-224** — batch EXIF timeout or JSON parse error returns an empty dict with no per-file error tracking; downstream code cannot distinguish "batch failed" from "file genuinely has no EXIF data" — every affected file then silently gets fallback name/alias with zero indication in output. Hit by 2/3 (also ties to Phase 1 finding on the fallback itself).

P2-3. **ingest.py:228** — `rec.get("SourceFile", "")` — if exiftool ever omits SourceFile from a record, multiple files silently collide at key `""` in the exif_map dict, silently losing all but one file's EXIF data. Hit by 1/3, but a real correctness bug if triggered.

P2-4. **ingest.py:435 vs adjacent code** — only `sqlite3.OperationalError` caught; `DatabaseError`/`ProgrammingError`/`IntegrityError` would crash the batch unhandled. (Corroborates Phase 1 finding #11's asymmetric-catch issue from the opposite angle.) Hit by 1/3.

P2-5. **ingest.py:186** — `f.stat().st_size` after rglob with no try/except; a file deleted between enumeration and stat crashes `scan_source()` with an uncaught OSError, and the caller doesn't wrap the call either. Hit by 1/3.

P2-6. **ingest.py:227** — no type check after `json.loads()`; malformed (non-list) exiftool output raises an uncaught TypeError on the `for rec in records` loop. Hit by 1/3.

P2-7. **ingest.py:265-269 schema** — `dest_path`, `orig_filename`, `source_label`, `camera_alias` are SQLite `TEXT` columns for genuinely unbounded external strings — not a hard violation in SQLite (TEXT is already unbounded there) but worth normalizing to match the project's LONGTEXT convention/intent. Hit by 1/3, low-stakes.

P2-8. **ingest.py:244** — `dt_str = dt_str[:19]` hard-truncates EXIF datetime strings with no overflow flag; ISO 8601 with timezone offset can exceed 19 chars. Hit by 1/3.

P2-9. **ingest.py:150-154** — no fallback to filesystem mtime when EXIF datetime is missing/unreliable (matters for card-clock-drift detection). Hit by 1/3, design gap not a bug.

## classify.py

P2-10. **classify.py:250-262 `get_capture_times`** — chunk-failure discards all previously-successful chunks' data (same root issue as Phase 1 finding #9, now confirmed independently by all 3 inventory passes). Hit by 3/3.

P2-11. **classify.py:299-302** — files with no parseable timestamp are silently skipped (printed WARNING only), with no error counter surfaced in the function's return value — caller can't tell how many files fell back to filename-based grouping. Hit by 2/3.

P2-12. **classify.py:314-315** (per one pass's line reference) — malformed `ExposureTime` silently defaults shutter speed to 0.0 with no logging; caller can't identify which files had invalid exposure data. Hit by 1/3 — new finding, not previously surfaced.

P2-13. **classify.py:47-48 RESULTS_FIELDS** — `classification`/`error_class` are unbounded strings with no length cap or overflow flag; low real risk since values come from a small closed set today, but nothing enforces that. Hit by 1/3.

P2-14. **classify.py:634-635, 690-707** — `accept_kw`/`reject_kw` loaded from a model's JSON sidecar with no length/content validation before being embedded directly into exiftool subprocess command arguments. Hit by 2/3 — worth flagging as a hardening item (subprocess arg injection surface, though the model JSON is locally-authored, not attacker-controlled in normal use).

P2-15. **classify.py:463-488, 498-510, 681, 686-707** — `write_xmp_sidecar()`/`embed_keyword_in_jpeg()`/`clear_robo_keywords()` return False/failure-count on error but never surface exiftool's actual stderr; failures are counted but not diagnosable. Hit by 2/3.

P2-16. **classify.py:950-954** — `csv.DictWriter(..., extrasaction='ignore')` silently drops any result-dict field not in RESULTS_FIELDS with no verification step confirming all expected fields are present before writing. Hit by 1/3.

## image_utils.py

P2-17. **image_utils.py:189-192 (rawpy path)** — bare `except Exception` conflates "no preview," "format unsupported," "corrupt file," and "I/O error" into one identical `(raw_path, None)` return — caller cannot distinguish any of these. Hit by 3/3 — strongest convergence in this module.

P2-18. **image_utils.py:106-127, 204-209 (exiftool path)** — empty bytes returned identically whether the file genuinely has no preview, exiftool timed out (30s timeout kills process silently), or exiftool errored — all three collapse to the same "failed" outcome with no distinguishing reason. Hit by 3/3 — strongest convergence.

P2-19. **image_utils.py:119-127** — early EOF on exiftool's stdout breaks the read loop and returns partial data; if exiftool crashes mid-transfer, incomplete binary preview data is written to disk and misclassified as a successful extraction (not caught as a failure at all). Hit by 1/3 — real correctness bug: a corrupted/truncated preview could silently pass through as valid.

P2-20. **image_utils.py:164-167** — non-JPEG preview format (e.g. BMP) is dropped with `(raw_path, None)` and no logging of what format was actually found. Hit by 2/3.

P2-21. **image_utils.py:205-206** — `open().write()` in the exiftool worker is unguarded; a disk I/O error (permission, disk full) propagates uncaught inside a ThreadPoolExecutor worker. Hit by 2/3.

P2-22. **image_utils.py:264-268, 303** — the `failed` list returned by `extract_raw_previews()` is a bare list of paths with zero error-reason classification; and the progress counter at line 264 increments on both success AND failure, so "Extracted 5000/5000" doesn't mean 5000 succeeded. Hit by 2/3 — the progress-counter half is a distinct, real misleading-output bug.

## junk_filter.py

P2-23. **junk_filter.py:454** — `max_conf = max((d.conf for d in usable), default='')` — mixed-type CSV column: float when detections exist, empty string `''` when none. Breaks any downstream numeric parsing of that column. Hit by 3/3 — strongest convergence, corroborates the correctness angle's independent finding on the same line.

P2-24. **junk_filter.py:237-239** — per-image YOLO inference time is available (`speed['inference']`) but only accumulated into a summary total, never stored per-row in the CSV. Hit by 1/3 (observability gap, not a bug).

P2-25. **junk_filter.py:254,256** — bounding-box coordinates and computed detection area are used internally for filtering but never written to the CSV — no way to reconstruct/audit why a detection was filtered as "tiny" after the fact. Hit by 1/3.

P2-26. **junk_filter.py:250** — `pred.names[cls_idx]` has no bounds check; a malformed YOLO response with an out-of-range class index would crash here uncaught. Hit by 1/3.

P2-27. **junk_filter.py:251,260** — non-vehicle-class detections and area-filtered-out detections are both dropped via `continue` with no counter — `len(r.detections)` only reflects survivors, so "no vehicle found" and "vehicle found but filtered" are indistinguishable after the fact. Hit by 2/3.

P2-28. **junk_filter.py:228** — `model.predict()` call has no try/except; a YOLO inference failure propagates uncaught and crashes the whole batch. Hit by 1/3.

P2-29. **junk_filter.py:443-468** — CSV file write has no try/except around the file-open/write; an unwritable path or full disk propagates uncaught. Hit by 1/3.

## inference_hotel.py

P2-30. **inference_hotel.py:209,249** — `error_class` field is initialized to empty string and NEVER populated on any failure path, unlike classify.py's equivalent which always sets it. Hit by 3/3 — strongest convergence in this module, corroborates Phase 1's correctness-angle finding.

P2-31. **inference_hotel.py:247-248** — `predictions.get('reject'/'select', 0.0)` silently substitutes 0.0 if the model's class names don't include 'reject'/'select', with no validation the model actually has those classes (unlike classify.py, which validates and raises at load time). Hit by 2/3.

P2-32. **inference_hotel.py:190-196, 227-238** — NEF-extraction failures and batch decode failures are printed ("SKIP") but never recorded as a result row or counted — they simply vanish from the output entirely, unlike classify.py's `decode_failed` rows. Hit by 3/3.

P2-33. **inference_hotel.py:213-214, 253-254** — `write_xmp_sidecar()`'s boolean return value is discarded at both call sites; XMP write failures are completely silent. Hit by 2/3.

P2-34. **inference_hotel.py:75-76, 131-132** — subprocess stdout/stderr are captured (`capture_output=True`) but never read; ImageMagick/exiftool's actual error diagnostics are discarded, replaced with generic hardcoded messages. Hit by 2/3.

P2-35. **inference_hotel.py:207-208, 247-248** — confidence values from the model forward pass are used without validating for NaN/None before being written to CSV. Hit by 1/3, low-likelihood trigger.

P2-36. **inference_hotel.py:260** — CSV column order comes from `results[0].keys()` rather than a fixed canonical field list — corroborates Phase 1.5's finding (I9) that this duplicates rather than imports classify.py's RESULTS_FIELDS. Hit by 2/3.

## review.py

P2-37. **review.py:81-82** — `float(row["confidence_select"])`/`float(row.get("confidence_reject", 0.0))` have no try/except; a malformed or missing value in results.csv crashes `load_results()` entirely (not per-row — the whole load fails). Hit by 3/3 — strongest convergence in this module.

P2-38. **review.py:249-266, 295-316, 336** — bare `except Exception` (often `: pass` or `: return None`) across `preextract_nef_previews`/`get_or_create_thumb`/`_make_thumb`, with no error counter anywhere; the `done` counter in the preview-prefetch loop only tracks attempts, not successes vs. failures. Hit by 3/3 — corroborates Phase 1's efficiency/reuse findings about this file's exiftool handling, now from the error-visibility angle.

P2-39. **review.py:89-105 `compute_bursts_and_winners`** — does not verify `classification == 'select'` before naming a frame the burst "winner" — confirms Phase 1.5's interface finding (I7) from the data side. Hit by 1/3 (but corroborated cross-phase).

P2-40. **review.py:80-85** — `confidence`, `confidence_reject`, `error_class`, `filename` are all read from CSV but never used anywhere in the file — undocumented as intentionally dropped. Hit by 2/3, cosmetic/waste not a bug.

P2-41. **review.py:171-196** — `PermissionError` silently continues (twice, in sibling and children directory enumeration) inside `_find_small_jpg_dir` with no counter — can't tell how many candidate directories were inaccessible vs. genuinely absent. Hit by 1/3.

## ui/review.py

P2-42. **ui/review.py (get_roll_angle/read_state)** — EXIF `Orientation` (tag 274) is never read alongside `RollAngle`; the two are orthogonal (rotation vs. tilt-correction) and a portrait shot could have `Orientation=6, RollAngle=0`, leaving the tool's portrait-detection incomplete. Hit by 3/3 — strongest convergence in this module.

P2-43. **ui/review.py:178** — `Label` value from XMP is accepted and returned with no validation against the `LABEL_COLORS` closed set the write path (`set_label`) enforces — an asymmetric contract: write-side validates, read-side doesn't. Hit by 2/3.

P2-44. **ui/review.py:182-186** — crop bounds (`CropLeft/Top/Right/Bottom/Angle`) are read back via `.get(field, default)` with no re-validation of the `[0,1]` range / `left<right`/`top<bottom` invariants that `set_crop()` enforces on write — a corrupted XMP sidecar with out-of-range or inverted values is silently accepted. Hit by 2/3.

P2-45. **ui/review.py:180** — `if item.get('HasCrop'):` treats the field as a bare truthy check with no type validation — a string value like `"false"` from corrupted metadata would be truthy and falsely report a crop exists. Hit by 1/3 — a real, if narrow-trigger, bug.

P2-46. **ui/review.py read_state's except clause** — omits `IndexError` (only catches `FileNotFoundError, json.JSONDecodeError`), unlike the sibling `get_roll_angle()` which does catch it; `data[0]` on an empty-but-valid JSON array (`[]`) would raise uncaught. Hit by 1/3.

## ui/thumbs.py

P2-47. **ui/thumbs.py:99 `_write_thumb_from_input`** — bare `except Exception: return None`/similar with zero error categorization (decode vs. permission vs. disk-full all identical). Hit by 3/3.

P2-48. **ui/thumbs.py:150, 152-157** — `_write_thumb_from_input()`'s return value is discarded by the batch caller; the progress counter increments on every attempt regardless of success, so a "processed N/N" count conflates successes and failures. Hit by 3/3 — strongest, matches Phase 1's efficiency-angle note on this same file from a different direction.

P2-49. **ui/thumbs.py:160** — `list(as_completed([...]))` collects futures but never calls `.result()` on them — any exception raised inside a worker thread is silently discarded, never surfaced. Hit by 3/3.

P2-50. **ui/thumbs.py:59 vs :93** — the identical `Image.open(io.BytesIO(preview))` operation is unguarded (can raise uncaught) in `get_thumb()` at line 59, but guarded (`except: return None`) at line 93 for a parallel call site — inconsistent error handling for the same operation depending on which code path reaches it. Hit by 1/3 — a real correctness inconsistency, not just missing logging.

P2-51. **ui/thumbs.py:132-133** — a RAW source with no entry in `preview_map` is silently skipped via `continue` with no counter. Hit by 2/3.

## write_tiered_keywords.py

P2-52. **write_tiered_keywords.py:105** — `float(row['confidence_select'])` has no try/except; a malformed or missing field crashes the entire run on the first bad row (worse than review.py's equivalent, which at least fails per-load rather than per-row but still fails wholesale). Hit by 3/3 — strongest convergence.

P2-53. **write_tiered_keywords.py:92-93, 133-134** — `open(args.winners)`/`open(args.results_csv)` are unguarded; a missing or unreadable file crashes with an unhandled exception (inconsistent with the exiftool-version check at line 68, which IS wrapped). Hit by 3/3.

P2-54. **write_tiered_keywords.py:138** — `r.get('classification') != 'decode_failed'` — if `classification` is ever missing from a row, `.get()` returns `None`, and `None != 'decode_failed'` evaluates `True`, so the row is silently treated as a normal (non-decode-failed) reject candidate instead of being excluded or flagged. Hit by 2/3 — a real, subtle logic bug via the `dict.get(key, default)`-substitution pattern this project's own CLAUDE.md calls out by name.

P2-55. **write_tiered_keywords.py** — `filename`, `confidence`, `confidence_reject`, `error_class` fields are read from the CSV schema but never used/labeled as intentionally dropped. Hit by 3/3, cosmetic.

P2-56. **write_tiered_keywords.py:96** — `row['path']` added to `winner_paths` set with no check that it's non-empty/valid before use. Hit by 1/3.

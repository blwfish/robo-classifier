# Phase 1.5 interface findings — new (not already in Phase 1 list)

Note: the following were independently rediscovered by BOTH Phase 1.5 passes and ALSO already exist in the Phase 1 correctness list (do not re-score, just note as corroborated by interface check in the final report):
- resolve_model hardcoded MODELS_DIR vs app_config.model_library (= Phase1 finding #20)
- resolve_model raises SystemExit, not caught by except Exception (= Phase1 finding #19)
- classify.py input_dir not resolved vs ui/app.py's .resolve() → is_winner always False (= Phase1 finding #7)

## New interface-only findings

I2. **ui/app.py:462-463 `write_keywords_endpoint`** — same hardcoded-MODELS_DIR bug as finding #20, recurs here for loading a model's accept/reject-keyword JSON sidecar, instead of using app_config.model_library like /api/library, /api/profiles, /api/train/start do. No mechanism.

I4. **ui/app.py:253-261 (`/api/run`) vs ui/pipeline_runner.py:68 (`PerfRecorder(preset=options.get("preset",""))`)** — the run handler merges a preset's resolved kwargs into `options` but never stores the preset's own name under `options["preset"]`. Every UI-triggered run logs an empty preset to perf_log.jsonl; the perf table always shows "—" even when a preset was used. Found independently by both interface passes. No mechanism.

I5. **ingest.py:373,470 vs ui/app.py:727-740 vs ui/ingest_runner.py:79-81** — only `ingest()` (of the 3 job types) emits its own mid-flight `{"type":"__end__"}` before `ui/ingest_runner.py` finalizes `job.summary`/`job.status` and pushes its own `__end__`. Narrow (two-statement) race window where the SSE handler can drain `job.summary == None` on a successful run, showing "Error: unknown" in the UI for a run that actually succeeded. No mechanism, no test.

I6. **classify.py:1052 / train_classifier.py:188 / prepare_training_data.py:125 (emit `{"type":"done",...}`) vs ui/static/app.js streamProgress/streamTrainProgress** — neither frontend handler has a case for event type "done"; these events are silently dropped (UI waits for a separate synthesized "status" event instead). No enum/shared vocabulary ties producer event types to consumer switch cases — this class of drift (new/renamed event type) isn't caught by anything.

I7. **review.py:89-105 `compute_bursts_and_winners` vs classify.py:398-427 `burst_dedup`** — review.py reimplements grouping/winner-selection from scratch rather than calling classify.burst_dedup, and its version omits the `classification == 'select'` filter classify.burst_dedup applies before naming a frame the burst "winner." A reject-classified frame could theoretically be named burst winner in the review UI if its confidence_select clears the UI's threshold slider. No parity test; any future change to classify.py's dedup contract won't propagate here.

I8. **ui/app.py:460-476 + ui/static/app.js (state.profile only set in onRun(), never restored by tryRestoreLastSession()/openSession())** — reopening an existing results directory in a fresh page load silently loses the model identity (`req.model_name` ends up null); any model-specific accept/reject keywords configured for that run are silently skipped on write with no warning to the user.

I9. **inference_hotel.py:202-210,260 vs classify.py:47-49 `RESULTS_FIELDS`/`WINNERS_FIELDS`** — inference_hotel.py hand-duplicates the same 7 field names instead of importing the canonical constant, and derives column order from `results[0].keys()` rather than a fixed list. A future change to RESULTS_FIELDS won't propagate here. (Corroborates the broader Phase 1 Q3 finding that inference_hotel.py is a legacy duplicate of classify.py.)

I11. **ui/static/app.js:1277-1284 `runWriteKeywords` vs ui/app.py `WriteKeywordsRequest.burst_threshold`** — the pipeline-run request sends `burst_threshold` (used for time-based burst grouping when writing winners.csv), but the later "Write Keywords" call from the Threshold screen never re-sends it (no `burstThreshold` field in frontend `state` at all). `req.burst_threshold` is always None server-side, so `/api/write_keywords` silently falls back to filename-based burst grouping (classify.burst_dedup) even when the original run used time-based grouping — can tag the wrong set of burst siblings as "select" vs. what actually produced winners.csv.

I12. **ui/app.py:470-476 `write_keywords_endpoint` vs ui/app.py:344 `session()` vs review.py:81 `load_results`** — `write_keywords_endpoint` does `r['confidence_select']` directly (only catches ValueError/TypeError, not KeyError), while `session()` guards with `.get('confidence_select')` first. A results.csv missing that column (older schema, hand-edited) crashes write_keywords_endpoint with an unhandled 500 instead of degrading like session() does. Same unguarded direct-index pattern also in review.py:81.

I13. **perf.py:275-282 `PerfRecorder._handle` scan_done branch vs ingest.py:360-368** — ingest() emits one scan_done event per source with that source's own file count; PerfRecorder does `self._cur_files = max(self._cur_files, n)`, correct only if count were cumulative. With 2+ ingest sources (documented multi-card workflow), the recorded scan-stage file count becomes the largest single source's count instead of the sum, silently understating files_per_s in perf_log.jsonl.

I14. **ui/pipeline_runner.py:74 `rec.set_stage_bytes("extract", nbytes)` vs classify.py's actual stage names ("previews"/"junk"/"inference"/"dedup"/"keywords"/"thumbs", never "extract")** — pipeline_runner pre-measures input bytes specifically so the RAW-preview-extraction stage can report MB/s, but stores them under a stage key the progress events never use. perf.py's `self._stage_bytes.get(self._cur_stage, 0)` lookup can never match, so mb_per_s for that stage is silently never computed. classify.py's own docstring enumerating stage names is also stale (omits "previews" and "thumbs") — likely how this drifted.

I15. **classify.py:443-447 `_keyword_hierarchy` vs inference_hotel.py:89-93** — two independently-maintained copies of the same keyword→HierarchicalSubject mapping; inference_hotel.py already imports from classify.py but re-implements this one function instead, with only a docstring comment (not an enforced mechanism) asserting parity. (Same underlying issue as Phase 1 Q3.)

I16. **image_utils.py:217-248 vs :303 `extract_raw_previews`** — docstring says returns a dict; empty-input early-return honors that (`return {}`), but every other path returns a 2-tuple. Both current callers unpack as a 2-tuple and only avoid the empty-dict branch by independently guarding `if raw_files:` first — not enforced by the function itself. (Same as Phase 1 finding #23, interface framing.)

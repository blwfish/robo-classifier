"""
Tests for ui/ingest_runner.py — had zero test coverage before this file
(robo-classifier-20260912-a1f3#20 / full-review Phase 3.5). Also covers the
double-__end__ race regression (robo-classifier-20260912-a1f3#17): ingest()
emits its own __end__-with-summary event, and job.summary must already be
set by the time it's queued.
"""

import time
from unittest.mock import patch

from ui.ingest_runner import MANAGER, IngestJob


def _wait_done(job, timeout=2.0):
    deadline = time.time() + timeout
    while job._thread is not None and job._thread.is_alive() and time.time() < deadline:
        time.sleep(0.01)


def _drain(job):
    events = []
    while not job.events.empty():
        events.append(job.events.get_nowait())
    return events


class TestIngestJobManager:
    def test_create_stores_sources_and_dest(self):
        sources = [{"path": "/x", "label": "card1"}]
        job = MANAGER.create(sources, "/dest")
        assert isinstance(job, IngestJob)
        assert job.sources == sources
        assert job.dest_dir == "/dest"

    def test_start_with_ingest_emitting_own_end_sets_summary_atomically(self):
        job = MANAGER.create([{"path": "/x", "label": "card1"}], "/dest")

        def fake_ingest(sources, dest_dir, progress_cb):
            progress_cb({"type": "progress", "done": 1, "total": 1})
            summary = {"copied": 1, "skipped": 0, "errors": 0, "total": 1, "dest": dest_dir}
            progress_cb({"type": "__end__", "summary": summary})
            return summary

        # PerfRecorder.cb forwards events to downstream_cb (the runner's own
        # emit()) unchanged in the real implementation — a passthrough mock
        # reproduces that without needing perf.py's own machinery.
        class _PassthroughRec:
            def __init__(self, downstream_cb, **kw):
                self.cb = downstream_cb

        with patch("ingest.ingest", side_effect=fake_ingest), \
             patch("perf.PerfRecorder", side_effect=_PassthroughRec):
            MANAGER.start(job)
            _wait_done(job)

        assert job.status == "done"
        assert job.summary == {"copied": 1, "skipped": 0, "errors": 0, "total": 1, "dest": "/dest"}

        events = _drain(job)
        types = [e["type"] for e in events]
        # Only one __end__ — ingest()'s own, not a second one from the runner.
        assert types.count("__end__") == 1

    def test_start_sets_error_status_on_exception(self):
        job = MANAGER.create([{"path": "/x", "label": "card1"}], "/dest")

        class _PassthroughRec:
            def __init__(self, downstream_cb, **kw):
                self.cb = downstream_cb

        with patch("ingest.ingest", side_effect=RuntimeError("disk full")), \
             patch("perf.PerfRecorder", side_effect=_PassthroughRec):
            MANAGER.start(job)
            _wait_done(job)

        assert job.status == "error"
        assert job.error == "disk full"
        events = _drain(job)
        assert [e["type"] for e in events].count("__end__") == 1

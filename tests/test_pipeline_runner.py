"""
Tests for ui/pipeline_runner.py — had zero test coverage before this file
(robo-classifier-20260912-a1f3#20 / full-review Phase 3.5). ui/job_runner.py
(the shared generic manager this delegates to) is tested separately in
tests/test_job_runner.py; this covers pipeline_runner's own thin wiring.
"""

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from ui.pipeline_runner import MANAGER, Job


def _wait_done(job, timeout=2.0):
    deadline = time.time() + timeout
    while job._thread is not None and job._thread.is_alive() and time.time() < deadline:
        time.sleep(0.01)


class TestPipelineJobManager:
    def test_create_converts_input_dir_to_path(self, tmp_path):
        job = MANAGER.create(str(tmp_path), {"model": "x.pt"})
        assert isinstance(job, Job)
        assert job.input_dir == tmp_path
        assert job.options == {"model": "x.pt"}

    def test_get_returns_created_job(self, tmp_path):
        job = MANAGER.create(str(tmp_path), {})
        assert MANAGER.get(job.id) is job

    def test_start_runs_pipeline_and_sets_summary(self, tmp_path):
        job = MANAGER.create(str(tmp_path), {"model": "x.pt"})

        fake_summary = {"winners": 3}
        with patch("classify.run_pipeline", return_value=fake_summary) as mock_run, \
             patch("perf.measure_input_bytes", return_value=0), \
             patch("perf.PerfRecorder") as MockRec:
            MockRec.return_value.cb = MagicMock()
            MANAGER.start(job)
            _wait_done(job)

        assert job.status == "done"
        assert job.summary == fake_summary
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["input_dir"] == tmp_path

    def test_start_sets_error_status_on_exception(self, tmp_path):
        job = MANAGER.create(str(tmp_path), {})

        with patch("classify.run_pipeline", side_effect=RuntimeError("boom")), \
             patch("perf.measure_input_bytes", return_value=0), \
             patch("perf.PerfRecorder") as MockRec:
            MockRec.return_value.cb = MagicMock()
            MANAGER.start(job)
            _wait_done(job)

        assert job.status == "error"
        assert job.error == "boom"

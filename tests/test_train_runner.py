"""
Tests for ui/train_runner.py — had zero test coverage before this file
(robo-classifier-20260912-a1f3#20 / full-review Phase 3.5).
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

from ui.train_runner import MANAGER, TrainJob


def _wait_done(job, timeout=2.0):
    deadline = time.time() + timeout
    while job._thread is not None and job._thread.is_alive() and time.time() < deadline:
        time.sleep(0.01)


def _make_job(tmp_path, **overrides):
    kwargs = dict(
        select_dir="sel", reject_dir="rej", dataset_dir=str(tmp_path / "ds"),
        test_size=0.2, model_output=str(tmp_path / "m.pt"), model_name="m",
        description="test model", epochs=3, learning_rate=0.001, batch_size=8,
    )
    kwargs.update(overrides)
    return MANAGER.create(**kwargs)


class TestTrainJobManager:
    def test_create_stores_all_fields(self, tmp_path):
        job = _make_job(tmp_path)
        assert isinstance(job, TrainJob)
        assert job.epochs == 3
        assert job.model_name == "m"

    def test_start_runs_both_phases_and_writes_sidecar(self, tmp_path):
        job = _make_job(tmp_path, accept_keyword="robo_ok")

        with patch("prepare_training_data.prepare_dataset") as mock_prepare, \
             patch("train_classifier.train_classifier",
                   return_value={"best_acc": 0.97}) as mock_train:
            MANAGER.start(job)
            _wait_done(job)

        assert job.status == "done"
        assert job.summary == {"best_acc": 0.97}
        mock_prepare.assert_called_once()
        mock_train.assert_called_once()

        sidecar = Path(job.model_output).with_suffix(".json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert data["best_acc"] == 0.97
        assert data["accept_keyword"] == "robo_ok"
        assert "reject_keyword" not in data  # not set on this job

    def test_status_transitions_through_preparing_and_training(self, tmp_path):
        job = _make_job(tmp_path)
        seen = []

        def fake_prepare(**kw):
            seen.append(job.status)

        def fake_train(**kw):
            seen.append(job.status)
            return {"best_acc": 0.5}

        with patch("prepare_training_data.prepare_dataset", side_effect=fake_prepare), \
             patch("train_classifier.train_classifier", side_effect=fake_train):
            MANAGER.start(job)
            _wait_done(job)

        assert seen == ["preparing", "training"]
        assert job.status == "done"  # finalized by the shared JobManager wrapper

    def test_start_sets_error_status_on_exception(self, tmp_path):
        job = _make_job(tmp_path)

        with patch("prepare_training_data.prepare_dataset",
                   side_effect=RuntimeError("bad dataset")):
            MANAGER.start(job)
            _wait_done(job)

        assert job.status == "error"
        assert job.error == "bad dataset"

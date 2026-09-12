"""
Threaded training job runner for the UI.

Two-phase: prepare dataset, then train. Both phases stream events via the
same queue so the frontend gets a single SSE stream for the whole job.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ui.job_runner import BaseJob, JobManager


@dataclass(kw_only=True)
class TrainJob(BaseJob):
    # Phase 1: prepare
    select_dir: str
    reject_dir: str
    dataset_dir: str
    test_size: float
    # Phase 2: train
    model_output: str     # full path to .pt file in model_library
    model_name: str       # stem name only (for sidecar + display)
    description: str
    epochs: int
    learning_rate: float
    batch_size: int
    accept_keyword: str = ""  # written to XMP alongside robo_9x tier for accepts
    reject_keyword: str = ""  # written to XMP for non-winning burst frames


def _write_sidecar(job: TrainJob, best_acc: float):
    """Write <model_name>.json sidecar next to the .pt file."""
    import json
    from datetime import datetime
    sidecar = Path(job.model_output).with_suffix(".json")
    data = {
        "description":  job.description or job.model_name,
        "trained_at":   datetime.now().isoformat(timespec="seconds"),
        "best_acc":     best_acc,
        "epochs":       job.epochs,
        "learning_rate": job.learning_rate,
        "batch_size":   job.batch_size,
        "select_dir":   job.select_dir,
        "reject_dir":   job.reject_dir,
        "test_size":    job.test_size,
    }
    if job.accept_keyword:
        data["accept_keyword"] = job.accept_keyword
    if job.reject_keyword:
        data["reject_keyword"] = job.reject_keyword
    sidecar.write_text(json.dumps(data, indent=2))


def _run(job: TrainJob, emit) -> dict:
    # ---- Phase 1: prepare dataset ----
    job.status = "preparing"
    emit({"type": "phase", "phase": "prepare", "message": "Preparing dataset split…"})
    from prepare_training_data import prepare_dataset
    prepare_dataset(
        select_dir=job.select_dir,
        reject_dir=job.reject_dir,
        output_dir=job.dataset_dir,
        test_size=job.test_size,
        progress_cb=emit,
    )

    # ---- Phase 2: train ----
    job.status = "training"
    emit({"type": "phase", "phase": "train", "message": f"Training {job.epochs} epochs…"})
    from train_classifier import train_classifier
    result = train_classifier(
        dataset_dir=job.dataset_dir,
        model_output=job.model_output,
        epochs=job.epochs,
        learning_rate=job.learning_rate,
        batch_size=job.batch_size,
        progress_cb=emit,
    )

    # Write JSON sidecar into model_library alongside the .pt
    _write_sidecar(job, result["best_acc"])

    return result


class _TrainJobManager:
    def __init__(self):
        self._manager = JobManager(job_cls=TrainJob)

    def create(self, **kwargs) -> TrainJob:
        return self._manager.create(**kwargs)

    def get(self, job_id: str):
        return self._manager.get(job_id)

    def start(self, job: TrainJob):
        self._manager.start(job, _run)


MANAGER = _TrainJobManager()

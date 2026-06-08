"""
Threaded training job runner for the UI.

Two-phase: prepare dataset, then train. Both phases stream events via the
same queue so the frontend gets a single SSE stream for the whole job.
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainJob:
    id: str
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

    events: queue.Queue = field(default_factory=queue.Queue)
    status: str = "pending"   # pending | preparing | training | done | error
    summary: Optional[dict] = None
    error: Optional[str] = None
    _thread: Optional[threading.Thread] = None


class TrainJobManager:
    def __init__(self):
        self._jobs: dict[str, TrainJob] = {}
        self._lock = threading.Lock()

    def create(self, **kwargs) -> TrainJob:
        job = TrainJob(id=uuid.uuid4().hex[:12], **kwargs)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[TrainJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job: TrainJob):
        def _cb(event):
            job.events.put(event)

        def _run():
            job.events.put({"type": "started"})
            try:
                # ---- Phase 1: prepare dataset ----
                job.status = "preparing"
                job.events.put({"type": "phase", "phase": "prepare",
                                "message": "Preparing dataset split…"})
                from prepare_training_data import prepare_dataset
                prepare_dataset(
                    select_dir=job.select_dir,
                    reject_dir=job.reject_dir,
                    output_dir=job.dataset_dir,
                    test_size=job.test_size,
                    progress_cb=_cb,
                )

                # ---- Phase 2: train ----
                job.status = "training"
                job.events.put({"type": "phase", "phase": "train",
                                "message": f"Training {job.epochs} epochs…"})
                from train_classifier import train_classifier
                result = train_classifier(
                    dataset_dir=job.dataset_dir,
                    model_output=job.model_output,
                    epochs=job.epochs,
                    learning_rate=job.learning_rate,
                    batch_size=job.batch_size,
                    progress_cb=_cb,
                )

                # Write JSON sidecar into model_library alongside the .pt
                _write_sidecar(job, result["best_acc"])

                job.summary = result
                job.status = "done"

            except Exception as e:
                tb = traceback.format_exc()
                print(tb, file=sys.stderr)
                job.error = str(e)
                job.status = "error"
                job.events.put({"type": "error", "message": str(e)})
            finally:
                job.events.put({"type": "__end__"})

        job._thread = threading.Thread(target=_run, daemon=True)
        job._thread.start()


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


MANAGER = TrainJobManager()

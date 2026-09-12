"""
Threaded pipeline job runner for the UI.

Runs classify.run_pipeline() in a background thread and exposes its progress
events as an SSE-friendly queue. Jobs are identified by a random id so the
frontend can reconnect/resume the event stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ui.job_runner import BaseJob, JobManager


@dataclass(kw_only=True)
class Job(BaseJob):
    input_dir: Path
    options: dict


def _run(job: Job, emit) -> dict:
    # Import here so the UI server starts fast (torch import is slow).
    from classify import run_pipeline
    from perf import PerfRecorder, measure_input_bytes
    from image_utils import IMAGE_EXTENSIONS

    options = dict(job.options)
    options.setdefault("pregen_thumbs", True)

    rec = PerfRecorder(
        downstream_cb=emit,
        run_type="pipeline",
        input_path=str(job.input_dir),
        preset=options.get("preset", ""),
        profile=options.get("profile", "") or "",
    )
    # Pre-measure input bytes so extract stage can report MB/s.
    try:
        nbytes = measure_input_bytes(job.input_dir, IMAGE_EXTENSIONS)
        rec.set_stage_bytes("extract", nbytes)
    except Exception:
        pass

    return run_pipeline(
        input_dir=job.input_dir,
        progress_cb=rec.cb,
        **options,
    )


class _PipelineJobManager:
    def __init__(self):
        self._manager = JobManager(job_cls=Job)

    def create(self, input_dir: str, options: dict) -> Job:
        return self._manager.create(input_dir=Path(input_dir), options=options)

    def get(self, job_id: str):
        return self._manager.get(job_id)

    def start(self, job: Job):
        self._manager.start(job, _run)


MANAGER = _PipelineJobManager()

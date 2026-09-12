"""
Threaded ingest job runner for the UI.

Runs ingest() in a background thread and exposes progress events as an
SSE-friendly queue.
"""

from __future__ import annotations

from dataclasses import dataclass

from ui.job_runner import BaseJob, JobManager


@dataclass(kw_only=True)
class IngestJob(BaseJob):
    sources: list[dict]   # [{path, label, force}]
    dest_dir: str


def _run(job: IngestJob, emit):
    from ingest import ingest as run_ingest
    from perf import PerfRecorder

    # Use first source path as the input_path for storage classification.
    input_path = job.sources[0]["path"] if job.sources else job.dest_dir

    rec = PerfRecorder(
        downstream_cb=emit,
        run_type="ingest",
        input_path=input_path,
        extra={"dest_path": job.dest_dir, "source_count": len(job.sources)},
    )
    # ingest() emits its own {"type": "__end__", "summary": ...} event before
    # returning (documented in its own docstring); JobManager.start()'s emit()
    # wrapper sets job.summary/job.status from that event atomically, so this
    # function's return value is never used in practice.
    run_ingest(
        sources=job.sources,
        dest_dir=job.dest_dir,
        progress_cb=rec.cb,
    )


class _IngestJobManager:
    def __init__(self):
        self._manager = JobManager(job_cls=IngestJob)

    def create(self, sources: list[dict], dest_dir: str) -> IngestJob:
        return self._manager.create(sources=sources, dest_dir=dest_dir)

    def get(self, job_id: str):
        return self._manager.get(job_id)

    def start(self, job: IngestJob):
        self._manager.start(job, _run)


MANAGER = _IngestJobManager()

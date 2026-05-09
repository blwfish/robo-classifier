"""
Threaded ingest job runner for the UI.

Mirrors the pattern of pipeline_runner.py: runs ingest() in a background
thread and exposes progress events as an SSE-friendly queue.
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class IngestJob:
    id: str
    sources: list[dict]   # [{path, label, force}]
    dest_dir: str
    events: queue.Queue = field(default_factory=queue.Queue)
    status: str = "pending"   # pending | running | done | error
    summary: Optional[dict] = None
    error: Optional[str] = None
    _thread: Optional[threading.Thread] = None


class IngestJobManager:
    def __init__(self):
        self._jobs: dict[str, IngestJob] = {}
        self._lock = threading.Lock()

    def create(self, sources: list[dict], dest_dir: str) -> IngestJob:
        job = IngestJob(id=uuid.uuid4().hex[:12], sources=sources, dest_dir=dest_dir)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[IngestJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job: IngestJob):
        def _progress(event):
            job.events.put(event)

        def _run():
            job.status = "running"
            job.events.put({"type": "started"})
            try:
                from ingest import ingest as run_ingest
                summary = run_ingest(
                    sources=job.sources,
                    dest_dir=job.dest_dir,
                    progress_cb=_progress,
                )
                job.summary = summary
                job.status = "done"
            except Exception as e:
                tb = traceback.format_exc()
                print(tb, file=sys.stderr)
                job.error = str(e)
                job.status = "error"
                job.events.put({"type": "error", "message": str(e)})
            finally:
                # Sentinel so SSE consumers close cleanly.
                job.events.put({"type": "__end__"})

        job._thread = threading.Thread(target=_run, daemon=True)
        job._thread.start()


MANAGER = IngestJobManager()

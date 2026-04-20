"""
Threaded pipeline job runner for the UI.

Runs classify.run_pipeline() in a background thread and exposes its progress
events as an SSE-friendly queue. Jobs are identified by a random id so the
frontend can reconnect/resume the event stream.
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
class Job:
    id: str
    input_dir: Path
    options: dict
    events: queue.Queue = field(default_factory=queue.Queue)
    status: str = "pending"  # pending | running | done | error
    summary: Optional[dict] = None
    error: Optional[str] = None
    _thread: Optional[threading.Thread] = None


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, input_dir: str, options: dict) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], input_dir=Path(input_dir), options=options)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job: Job):
        def _progress(event):
            job.events.put(event)

        def _run():
            job.status = "running"
            job.events.put({"type": "started"})
            try:
                # Import here so the UI server starts fast (torch import is slow).
                from classify import run_pipeline

                # UI runs always pregenerate thumbs so the grid opens instantly.
                options = dict(job.options)
                options.setdefault("pregen_thumbs", True)
                summary = run_pipeline(
                    input_dir=job.input_dir,
                    progress_cb=_progress,
                    **options,
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
                # Sentinel so SSE consumers can close cleanly.
                job.events.put({"type": "__end__"})

        job._thread = threading.Thread(target=_run, daemon=True)
        job._thread.start()


MANAGER = JobManager()

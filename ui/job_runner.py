"""
Shared threaded-job-runner infrastructure for the UI's background jobs.

pipeline_runner.py, ingest_runner.py, and train_runner.py used to each
independently hand-copy the same Job dataclass skeleton (id/events/status/
summary/error/_thread) and JobManager pattern (create/get/start with a
try/except/finally wrapping a background thread), with no shared base and
no test coverage on any of the three — they had already drifted slightly
out of sync (only train_runner's status values included "preparing"/
"training" instead of just "running"). See
robo-classifier-20260912-a1f3#21 (Q1 in the full-review Phase 1 report).
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(kw_only=True)
class BaseJob:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    events: queue.Queue = field(default_factory=queue.Queue)
    status: str = "pending"   # pending | running | done | error (subclasses
                              # may set additional in-progress values, e.g.
                              # train_runner's "preparing"/"training")
    summary: Optional[dict] = None
    error: Optional[str] = None
    _thread: Optional[threading.Thread] = None


# A run_body: (job, emit) -> summary dict on success (raises on failure).
RunBody = Callable[[BaseJob, Callable[[dict], None]], Optional[dict]]


class JobManager:
    """Generic threaded job manager. `job_cls` must be a BaseJob subclass."""

    def __init__(self, job_cls: type = BaseJob):
        self._job_cls = job_cls
        self._jobs: dict[str, BaseJob] = {}
        self._lock = threading.Lock()

    def create(self, **kwargs) -> BaseJob:
        job = self._job_cls(**kwargs)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Optional[BaseJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, job: BaseJob, run_body: RunBody):
        """
        Run `run_body(job, emit)` in a background thread.

        run_body performs the actual work, calling `emit(event)` for
        progress events, and returns a summary dict on success (or raises
        on failure). If run_body's underlying work function has its own
        documented "__end__ with summary" event contract (only ingest()
        does, today), `emit()` sets job.summary/job.status atomically
        *before* that event reaches the queue — this is what avoids the
        race where an SSE consumer could observe the __end__ event before
        job.summary was populated (robo-classifier-20260912-a1f3#17). In
        that case run_body's own return value is ignored and no second
        __end__ is emitted.

        run_body is free to set job.status to job-specific intermediate
        values during its own execution — this wrapper only sets
        job.status itself at start ("running") and at the end
        ("done"/"error").
        """
        ended = {"flag": False}

        def emit(event: dict):
            if event.get("type") == "__end__":
                ended["flag"] = True
                if "summary" in event:
                    job.summary = event["summary"]
                job.status = "done"
            job.events.put(event)

        def _run():
            job.status = "running"
            job.events.put({"type": "started"})
            try:
                result = run_body(job, emit)
                if not ended["flag"]:
                    job.summary = result
                    job.status = "done"
            except Exception as e:
                tb = traceback.format_exc()
                print(tb, file=sys.stderr)
                job.error = str(e)
                job.status = "error"
                job.events.put({"type": "error", "message": str(e)})
            finally:
                if not ended["flag"]:
                    job.events.put({"type": "__end__"})

        job._thread = threading.Thread(target=_run, daemon=True)
        job._thread.start()

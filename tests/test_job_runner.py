"""
Tests for ui/job_runner.py — the shared threaded-job-runner infrastructure
extracted from pipeline_runner.py/ingest_runner.py/train_runner.py's three
independently-hand-copied Job/JobManager implementations (see
robo-classifier-20260912-a1f3#21). These had zero test coverage combined;
testing the shared mechanism once here covers all three runners' common
behavior, including the double-__end__ race regression
(robo-classifier-20260912-a1f3#17).
"""

import time
from dataclasses import dataclass

import pytest

from ui.job_runner import BaseJob, JobManager


@dataclass(kw_only=True)
class _PlainJob(BaseJob):
    payload: str = ""


def _wait_done(job, timeout=2.0):
    deadline = time.time() + timeout
    while job._thread.is_alive() and time.time() < deadline:
        time.sleep(0.01)


def _drain(job):
    events = []
    while not job.events.empty():
        events.append(job.events.get_nowait())
    return events


class TestJobManagerCreateGet:
    def test_create_assigns_unique_ids(self):
        mgr = JobManager(job_cls=_PlainJob)
        j1 = mgr.create(payload="a")
        j2 = mgr.create(payload="b")
        assert j1.id != j2.id
        assert j1.status == "pending"

    def test_get_returns_created_job(self):
        mgr = JobManager(job_cls=_PlainJob)
        job = mgr.create(payload="x")
        assert mgr.get(job.id) is job

    def test_get_unknown_id_returns_none(self):
        mgr = JobManager(job_cls=_PlainJob)
        assert mgr.get("nonexistent") is None


class TestJobManagerStartSuccess:
    def test_successful_run_body_sets_summary_and_done(self):
        mgr = JobManager(job_cls=_PlainJob)
        job = mgr.create(payload="x")

        def run_body(job, emit):
            emit({"type": "progress", "n": 1})
            return {"result": "ok"}

        mgr.start(job, run_body)
        _wait_done(job)

        assert job.status == "done"
        assert job.summary == {"result": "ok"}
        assert job.error is None

        events = _drain(job)
        types = [e["type"] for e in events]
        assert types == ["started", "progress", "__end__"]

    def test_run_body_can_set_intermediate_status_values(self):
        # train_runner's "preparing"/"training" pattern: run_body sets
        # job-specific status values mid-run; the wrapper must not clobber
        # them until the very end.
        mgr = JobManager(job_cls=_PlainJob)
        job = mgr.create(payload="x")
        seen_statuses = []

        def run_body(job, emit):
            job.status = "phase_one"
            seen_statuses.append(job.status)
            job.status = "phase_two"
            seen_statuses.append(job.status)
            return {"ok": True}

        mgr.start(job, run_body)
        _wait_done(job)

        assert seen_statuses == ["phase_one", "phase_two"]
        assert job.status == "done"  # wrapper still finalizes at the end


class TestJobManagerStartFailure:
    def test_exception_sets_error_status_and_message(self):
        mgr = JobManager(job_cls=_PlainJob)
        job = mgr.create(payload="x")

        def run_body(job, emit):
            raise RuntimeError("boom")

        mgr.start(job, run_body)
        _wait_done(job)

        assert job.status == "error"
        assert job.error == "boom"
        assert job.summary is None

        events = _drain(job)
        types = [e["type"] for e in events]
        assert types == ["started", "error", "__end__"]


class TestJobManagerOwnEndEvent:
    def test_run_body_emitting_own_end_is_not_duplicated(self):
        # Regression test for robo-classifier-20260912-a1f3#17: a work
        # function like ingest() that emits its own {"type": "__end__",
        # "summary": ...} event must have job.summary/job.status set
        # atomically from that event (not race against the wrapper), and
        # the wrapper must not emit a second __end__ afterward.
        mgr = JobManager(job_cls=_PlainJob)
        job = mgr.create(payload="x")

        def run_body(job, emit):
            emit({"type": "progress", "n": 1})
            emit({"type": "__end__", "summary": {"copied": 3}})
            # Return value is irrelevant once __end__ was already emitted.
            return {"ignored": True}

        mgr.start(job, run_body)
        _wait_done(job)

        assert job.status == "done"
        assert job.summary == {"copied": 3}

        events = _drain(job)
        types = [e["type"] for e in events]
        assert types == ["started", "progress", "__end__"]
        assert types.count("__end__") == 1

    def test_end_event_summary_set_before_event_is_queued(self):
        # The exact ordering guarantee the fix depends on: by the time a
        # consumer can observe the __end__ event on the queue, job.summary
        # is already populated — never a null summary racing a "done" event.
        mgr = JobManager(job_cls=_PlainJob)
        job = mgr.create(payload="x")
        observed_summary_at_end = {}

        def run_body(job, emit):
            def emit_and_check(event):
                emit(event)
                if event.get("type") == "__end__":
                    # By the time emit() returns, job.summary must already
                    # be set — a consumer draining the queue right now
                    # would see a populated summary, not None.
                    observed_summary_at_end["value"] = job.summary
            emit_and_check({"type": "__end__", "summary": {"done": True}})
            return None

        mgr.start(job, run_body)
        _wait_done(job)

        assert observed_summary_at_end["value"] == {"done": True}

"""Tests for app.py -- FastAPI route-handler contract logic.

Route handlers are plain functions (FastAPI only wires up the ASGI/HTTP
transport around them), so these call them directly rather than pulling in
an HTTP test client -- faster, and exercises exactly the validation/routing
logic under test without a real ffmpeg/ffprobe dependency. The ffmpeg
merge pipeline itself was verified separately against real Air3 footage.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app
import merge


@pytest.fixture(autouse=True)
def isolated_prefs(tmp_path, monkeypatch):
    # Never touch the real air3_ingest/.prefs.json from tests.
    monkeypatch.setattr(app, "PREFS_PATH", tmp_path / ".prefs.json")


def fake_group(names):
    return merge.ClipGroup(clips=[SimpleNamespace(mp4_path=Path(n)) for n in names])


class TestGapThresholdValidation:
    def test_scan_rejects_zero(self, tmp_path):
        with pytest.raises(HTTPException) as exc_info:
            app.scan(app.ScanRequest(source_dir=str(tmp_path), gap_threshold_s=0))
        assert exc_info.value.status_code == 400

    def test_scan_rejects_negative(self, tmp_path):
        with pytest.raises(HTTPException) as exc_info:
            app.scan(app.ScanRequest(source_dir=str(tmp_path), gap_threshold_s=-5))
        assert exc_info.value.status_code == 400

    def test_process_rejects_zero(self, tmp_path):
        with pytest.raises(HTTPException) as exc_info:
            app.process(app.ProcessRequest(
                source_dir=str(tmp_path), destination_dir=str(tmp_path), gap_threshold_s=0,
            ))
        assert exc_info.value.status_code == 400

    def test_scan_accepts_small_positive_value(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ([], [], []))
        result = app.scan(app.ScanRequest(source_dir=str(tmp_path), gap_threshold_s=0.001))
        assert result == {"groups": [], "warnings": []}


class TestSourceDestValidation:
    def test_scan_rejects_nonexistent_source_dir(self, tmp_path):
        with pytest.raises(HTTPException) as exc_info:
            app.scan(app.ScanRequest(source_dir=str(tmp_path / "nope"), gap_threshold_s=300))
        assert exc_info.value.status_code == 400

    def test_process_requires_destination_dir(self, tmp_path):
        with pytest.raises(HTTPException) as exc_info:
            app.process(app.ProcessRequest(
                source_dir=str(tmp_path), destination_dir="", gap_threshold_s=300,
            ))
        assert exc_info.value.status_code == 400


class TestChooseDirValidation:
    def test_rejects_invalid_which(self):
        with pytest.raises(HTTPException) as exc_info:
            app.choose_dir(app.ChooseDirRequest(which="bogus"))
        assert exc_info.value.status_code == 400


class TestProcessGroupSelection:
    def test_default_processes_every_discovered_group(self, tmp_path, monkeypatch):
        groups = [fake_group(["a.mp4"]), fake_group(["b.mp4", "c.mp4"])]
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: (groups, [], []))
        calls = []
        monkeypatch.setattr(merge, "merge_group", lambda g, d: calls.append(g) or merge.MergeResult(
            ok=True, output_path=Path("/fake/out.mp4"), source_files=[c.mp4_path.name for c in g.clips],
        ))
        result = app.process(app.ProcessRequest(
            source_dir=str(tmp_path), destination_dir=str(tmp_path), gap_threshold_s=300,
            selected_groups=None,
        ))
        assert len(calls) == 2
        assert all(r["ok"] for r in result["results"])

    def test_duplicate_selected_groups_processed_only_once(self, tmp_path, monkeypatch):
        group = fake_group(["a.mp4", "b.mp4"])
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ([group], [], []))
        calls = []
        monkeypatch.setattr(merge, "merge_group", lambda g, d: calls.append(g) or merge.MergeResult(
            ok=True, output_path=Path("/fake/out.mp4"), source_files=["a.mp4", "b.mp4"],
        ))
        result = app.process(app.ProcessRequest(
            source_dir=str(tmp_path), destination_dir=str(tmp_path), gap_threshold_s=300,
            selected_groups=[["a.mp4", "b.mp4"], ["a.mp4", "b.mp4"]],
        ))
        assert len(calls) == 1
        assert len(result["results"]) == 1

    def test_stale_selection_reports_error_without_crashing(self, tmp_path, monkeypatch):
        # Simulates the source folder changing between an earlier /api/scan
        # and this /api/process call: the requested clip-name set no longer
        # matches any freshly-discovered group.
        current_group = fake_group(["a.mp4", "b.mp4"])
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ([current_group], [], []))
        calls = []
        monkeypatch.setattr(merge, "merge_group", lambda g, d: calls.append(g))
        result = app.process(app.ProcessRequest(
            source_dir=str(tmp_path), destination_dir=str(tmp_path), gap_threshold_s=300,
            selected_groups=[["a.mp4", "b.mp4", "c.mp4"]],  # stale: c.mp4 wasn't in the original group
        ))
        assert calls == []  # merge_group must never run against a mismatched selection
        assert len(result["results"]) == 1
        assert result["results"][0]["ok"] is False
        assert "re-scan" in result["results"][0]["error"]

    def test_merge_group_exception_reported_per_group_not_raised(self, tmp_path, monkeypatch):
        group = fake_group(["a.mp4"])
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ([group], [], []))

        def boom(g, d):
            raise RuntimeError("ffmpeg exploded")
        monkeypatch.setattr(merge, "merge_group", boom)

        result = app.process(app.ProcessRequest(
            source_dir=str(tmp_path), destination_dir=str(tmp_path), gap_threshold_s=300,
        ))
        assert result["results"][0]["ok"] is False
        assert "ffmpeg exploded" in result["results"][0]["error"]

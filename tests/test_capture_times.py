"""Tests for get_capture_times() and find_images() in classify.py.

get_capture_times runs exiftool under the hood; we mock subprocess.run so
no external process is spawned. The interesting logic is JSON parsing and
the fractional-shutter-speed handling.
"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import classify


def _fake_run_factory(stdout, returncode=0):
    """Return a callable compatible with subprocess.run."""
    def run(*args, **kwargs):
        m = MagicMock()
        m.returncode = returncode
        m.stdout = stdout
        return m
    return run


# =============================================================================
# get_capture_times — exiftool JSON parsing
# =============================================================================

class TestGetCaptureTimes:
    def test_empty_input_returns_empty(self):
        assert classify.get_capture_times([]) == {}

    def test_parses_timestamp_and_shutter_fraction(self, monkeypatch):
        data = [{
            "SourceFile":        "/x/a.jpg",
            "DateTimeOriginal":  "2026:04:20 14:30:00",
            "SubSecTimeOriginal": "123",
            "ExposureTime":      "1/1000",
        }]
        monkeypatch.setattr(classify.subprocess, "run",
                            _fake_run_factory(json.dumps(data)))
        result = classify.get_capture_times([Path("/x/a.jpg")])
        assert Path("/x/a.jpg") in result
        entry = result[Path("/x/a.jpg")]
        assert entry["shutter"] == pytest.approx(0.001)
        # Timestamp should include sub-seconds (epoch-based, so just check type + positive)
        assert entry["timestamp"] > 0

    def test_decimal_shutter(self, monkeypatch):
        data = [{
            "SourceFile":        "/x/a.jpg",
            "DateTimeOriginal":  "2026:04:20 14:30:00",
            "ExposureTime":      "0.5",
        }]
        monkeypatch.setattr(classify.subprocess, "run",
                            _fake_run_factory(json.dumps(data)))
        result = classify.get_capture_times([Path("/x/a.jpg")])
        assert result[Path("/x/a.jpg")]["shutter"] == 0.5

    def test_missing_subseconds_ok(self, monkeypatch):
        data = [{
            "SourceFile":       "/x/a.jpg",
            "DateTimeOriginal": "2026:04:20 14:30:00",
            "ExposureTime":     "1/500",
        }]
        monkeypatch.setattr(classify.subprocess, "run",
                            _fake_run_factory(json.dumps(data)))
        result = classify.get_capture_times([Path("/x/a.jpg")])
        assert Path("/x/a.jpg") in result

    def test_missing_timestamp_skipped(self, monkeypatch):
        data = [{"SourceFile": "/x/a.jpg", "ExposureTime": "1/1000"}]
        monkeypatch.setattr(classify.subprocess, "run",
                            _fake_run_factory(json.dumps(data)))
        assert classify.get_capture_times([Path("/x/a.jpg")]) == {}

    def test_malformed_timestamp_skipped(self, monkeypatch):
        data = [{
            "SourceFile":       "/x/a.jpg",
            "DateTimeOriginal": "not-a-date",
        }]
        monkeypatch.setattr(classify.subprocess, "run",
                            _fake_run_factory(json.dumps(data)))
        assert classify.get_capture_times([Path("/x/a.jpg")]) == {}

    def test_invalid_shutter_defaults_zero(self, monkeypatch):
        data = [{
            "SourceFile":       "/x/a.jpg",
            "DateTimeOriginal": "2026:04:20 14:30:00",
            "ExposureTime":     "weird",
        }]
        monkeypatch.setattr(classify.subprocess, "run",
                            _fake_run_factory(json.dumps(data)))
        result = classify.get_capture_times([Path("/x/a.jpg")])
        # Entry present but shutter falls back to 0.0
        assert result[Path("/x/a.jpg")]["shutter"] == 0.0

    def test_exiftool_missing_returns_empty(self, monkeypatch):
        def raise_nf(*a, **kw): raise FileNotFoundError()
        monkeypatch.setattr(classify.subprocess, "run", raise_nf)
        assert classify.get_capture_times([Path("/x/a.jpg")]) == {}

    def test_exiftool_nonzero_returns_empty(self, monkeypatch):
        monkeypatch.setattr(classify.subprocess, "run",
                            _fake_run_factory("", returncode=1))
        assert classify.get_capture_times([Path("/x/a.jpg")]) == {}

    def test_malformed_json_returns_empty(self, monkeypatch):
        monkeypatch.setattr(classify.subprocess, "run",
                            _fake_run_factory("not json"))
        assert classify.get_capture_times([Path("/x/a.jpg")]) == {}


# =============================================================================
# find_images
# =============================================================================

class TestFindImages:
    def test_separates_jpg_and_raw(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"")
        (tmp_path / "b.JPEG").write_bytes(b"")
        (tmp_path / "c.png").write_bytes(b"")
        (tmp_path / "d.nef").write_bytes(b"")
        (tmp_path / "e.CR3").write_bytes(b"")
        (tmp_path / "notes.txt").write_bytes(b"")

        jpgs, raws = classify.find_images(tmp_path)
        jpg_names = {p.name for p in jpgs}
        raw_names = {p.name for p in raws}
        assert jpg_names == {"a.jpg", "b.JPEG", "c.png"}
        assert raw_names == {"d.nef", "e.CR3"}

    def test_sorted(self, tmp_path):
        for name in ("z.jpg", "a.jpg", "m.jpg"):
            (tmp_path / name).write_bytes(b"")
        jpgs, _ = classify.find_images(tmp_path)
        assert [p.name for p in jpgs] == ["a.jpg", "m.jpg", "z.jpg"]

    def test_subdirectories_ignored(self, tmp_path):
        # find_images doesn't recurse — it scans one level only
        (tmp_path / "a.jpg").write_bytes(b"")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.jpg").write_bytes(b"")
        jpgs, _ = classify.find_images(tmp_path)
        assert [p.name for p in jpgs] == ["a.jpg"]

    def test_empty_dir(self, tmp_path):
        jpgs, raws = classify.find_images(tmp_path)
        assert jpgs == [] and raws == []

    def test_unknown_extensions_ignored(self, tmp_path):
        (tmp_path / "a.tiff").write_bytes(b"")
        (tmp_path / "b.heic").write_bytes(b"")
        jpgs, raws = classify.find_images(tmp_path)
        assert jpgs == [] and raws == []

    def test_all_raw_extensions_recognized(self, tmp_path):
        for ext in classify.RAW_EXTENSIONS:
            (tmp_path / f"x{ext}").write_bytes(b"")
        _, raws = classify.find_images(tmp_path)
        assert len(raws) == len(classify.RAW_EXTENSIONS)

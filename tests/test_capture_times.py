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

    def test_exiftool_missing_returns_empty_dict(self, monkeypatch):
        # exiftool not found at all: nothing can be extracted, but this must
        # not be conflated with "some files have no capture time" via a
        # separate sentinel — an empty dict is the uniform "nothing usable"
        # result (robo-classifier-20260912-a1f3#07).
        def raise_nf(*a, **kw): raise FileNotFoundError()
        monkeypatch.setattr(classify.subprocess, "run", raise_nf)
        assert classify.get_capture_times([Path("/x/a.jpg")]) == {}

    def test_exiftool_nonzero_on_one_batch_does_not_discard_others(self, monkeypatch):
        # A batch that fails no longer discards other, already-successful
        # batches' data — only the failed batch's own files end up missing
        # from the result (previously this returned None unconditionally,
        # discarding everything).
        calls = {"n": 0}
        good_data = json.dumps([{"SourceFile": "/x/a.jpg", "DateTimeOriginal": "2026:01:01 00:00:00"}])

        def fake_run(cmd, **kw):
            calls["n"] += 1
            from unittest.mock import MagicMock
            resp = MagicMock()
            if calls["n"] == 1:
                resp.returncode = 1
                resp.stdout = ""
            else:
                resp.returncode = 0
                resp.stdout = good_data
            return resp

        monkeypatch.setattr(classify.subprocess, "run", fake_run)
        paths = [Path(f"/x/{i}.jpg") for i in range(501)]  # forces 2 batches (>500)
        result = classify.get_capture_times(paths)
        # Only the second (successful) batch's file made it through — the
        # first batch's failure didn't wipe it out, and didn't itself
        # contribute anything either.
        assert list(result.keys()) == [Path("/x/a.jpg")]

    def test_malformed_json_on_one_batch_does_not_discard_others(self, monkeypatch):
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

    def test_recursive_includes_subdirectories(self, tmp_path):
        (tmp_path / "root.jpg").write_bytes(b"")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.jpg").write_bytes(b"")
        jpgs, _ = classify.find_images(tmp_path, recursive=True)
        names = {p.name for p in jpgs}
        assert names == {"root.jpg", "nested.jpg"}

    def test_recursive_skips_dot_directories(self, tmp_path):
        (tmp_path / "visible.jpg").write_bytes(b"")
        dot_dir = tmp_path / ".hidden"
        dot_dir.mkdir()
        (dot_dir / "shadow.jpg").write_bytes(b"")
        # macOS Spotlight dir
        spotlight = tmp_path / ".Spotlight-V100"
        spotlight.mkdir()
        (spotlight / "index.jpg").write_bytes(b"")
        jpgs, _ = classify.find_images(tmp_path, recursive=True)
        assert [p.name for p in jpgs] == ["visible.jpg"]

    def test_non_recursive_does_not_include_subdirectories(self, tmp_path):
        (tmp_path / "root.jpg").write_bytes(b"")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.jpg").write_bytes(b"")
        jpgs, _ = classify.find_images(tmp_path, recursive=False)
        assert [p.name for p in jpgs] == ["root.jpg"]

    def test_all_raw_extensions_recognized(self, tmp_path):
        for ext in classify.RAW_EXTENSIONS:
            (tmp_path / f"x{ext}").write_bytes(b"")
        _, raws = classify.find_images(tmp_path)
        assert len(raws) == len(classify.RAW_EXTENSIONS)

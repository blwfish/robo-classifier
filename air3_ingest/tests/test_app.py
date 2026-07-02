"""Tests for app.py -- FastAPI route-handler contract logic.

Route handlers are plain functions (FastAPI only wires up the ASGI/HTTP
transport around them), so these call them directly rather than pulling in
an HTTP test client -- faster, and exercises exactly the validation/routing
logic under test without a real ffmpeg/ffprobe dependency. The ffmpeg
merge pipeline itself was verified separately against real Air3 footage.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app
import audio_merge
import merge


@pytest.fixture(autouse=True)
def isolated_prefs(tmp_path, monkeypatch):
    # Never touch the real air3_ingest/.prefs.json from tests.
    monkeypatch.setattr(app, "PREFS_PATH", tmp_path / ".prefs.json")


def fake_group(names):
    return merge.ClipGroup(clips=[SimpleNamespace(mp4_path=Path(n)) for n in names])


def fake_audio_group(names):
    return audio_merge.AudioClipGroup(clips=[SimpleNamespace(wav_path=Path(n)) for n in names])


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
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ("video", [], [], [], {}))
        result = app.scan(app.ScanRequest(source_dir=str(tmp_path), gap_threshold_s=0.001))
        assert result == {"kind": "video", "groups": [], "warnings": []}


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


class TestLoadPrefs:
    def test_missing_file_returns_defaults(self):
        assert app.load_prefs() == app.DEFAULT_PREFS

    def test_valid_prefs_merged_over_defaults(self):
        app.PREFS_PATH.write_text(json.dumps({"source_dir": "/custom"}))
        prefs = app.load_prefs()
        assert prefs["source_dir"] == "/custom"
        assert prefs["destination_dir"] == app.DEFAULT_PREFS["destination_dir"]

    def test_malformed_json_falls_back_to_defaults(self):
        app.PREFS_PATH.write_text("{not valid json")
        assert app.load_prefs() == app.DEFAULT_PREFS

    def test_unreadable_file_falls_back_to_defaults(self):
        app.PREFS_PATH.mkdir()  # a directory, not a file -- read_text() raises OSError
        assert app.load_prefs() == app.DEFAULT_PREFS

    @pytest.mark.parametrize("bad_top_level", [[1, 2, 3], None, 42, "just a string"])
    def test_non_dict_json_falls_back_to_defaults(self, bad_top_level):
        # Regression: {**DEFAULT_PREFS, **json.loads(...)} used to raise an
        # uncaught TypeError when the file was syntactically valid JSON
        # but not an object -- crashing every endpoint that calls
        # load_prefs() until the file was manually fixed.
        app.PREFS_PATH.write_text(json.dumps(bad_top_level))
        assert app.load_prefs() == app.DEFAULT_PREFS

    def test_null_valued_known_key_falls_back_to_default(self):
        # Regression: a dict-shaped prefs file with a null value for a
        # known key merged "successfully" and silently overwrote the
        # numeric default with None, which then raised deep inside
        # _validated_gap_threshold instead of a clean 400.
        app.PREFS_PATH.write_text(json.dumps({"gap_threshold_s": None}))
        prefs = app.load_prefs()
        assert prefs["gap_threshold_s"] == app.DEFAULT_PREFS["gap_threshold_s"]

    def test_wrong_typed_known_key_falls_back_others_preserved(self):
        app.PREFS_PATH.write_text(json.dumps({
            "gap_threshold_s": "not-a-number",
            "source_dir": "/custom",
        }))
        prefs = app.load_prefs()
        assert prefs["gap_threshold_s"] == app.DEFAULT_PREFS["gap_threshold_s"]
        assert prefs["source_dir"] == "/custom"

    def test_int_accepted_for_float_default(self):
        # JSON doesn't distinguish 300 from 300.0 -- an int must be
        # accepted for a float-typed default, not rejected as wrong-type.
        app.PREFS_PATH.write_text(json.dumps({"gap_threshold_s": 120}))
        assert app.load_prefs()["gap_threshold_s"] == 120

    def test_unknown_extra_key_is_ignored_not_an_error(self):
        app.PREFS_PATH.write_text(json.dumps({"source_dir": "/custom", "made_up_key": "x"}))
        prefs = app.load_prefs()
        assert prefs["source_dir"] == "/custom"
        assert "made_up_key" not in prefs


class TestChooseFolderDialog:
    """Exercises choose_folder_dialog() directly by mocking subprocess.run
    -- it previously had zero test coverage under any condition (the one
    test touching /api/choose_dir fails validation before reaching it)."""

    def test_success_returns_path_no_error(self, monkeypatch):
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="/Volumes/Air3_mSD/DCIM\n", stderr="",
        ))
        chosen, error = app.choose_folder_dialog("prompt", None)
        assert chosen == "/Volumes/Air3_mSD/DCIM"
        assert error is None

    def test_user_cancel_returns_none_none(self, monkeypatch):
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="", stderr="execution error: User canceled. (-128)",
        ))
        chosen, error = app.choose_folder_dialog("prompt", None)
        assert chosen is None
        assert error is None

    def test_genuine_applescript_failure_returns_error(self, monkeypatch):
        # Regression: a real failure (e.g. an Automation permission
        # denial) used to be indistinguishable from a user cancel -- both
        # collapsed to None with stderr discarded entirely.
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=1, stdout="",
            stderr="execution error: Not authorized to send Apple events (-1743)",
        ))
        chosen, error = app.choose_folder_dialog("prompt", None)
        assert chosen is None
        assert error is not None
        assert "1743" in error

    def test_osascript_missing_returns_error_not_crash(self, monkeypatch):
        def boom(*a, **k):
            raise FileNotFoundError("osascript")
        monkeypatch.setattr(app.subprocess, "run", boom)
        chosen, error = app.choose_folder_dialog("prompt", None)
        assert chosen is None
        assert error is not None

    def test_unexpected_non_path_output_returns_error(self, monkeypatch):
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="not a path", stderr="",
        ))
        chosen, error = app.choose_folder_dialog("prompt", None)
        assert chosen is None
        assert error is not None

    def test_empty_success_output_returns_error_not_silently_accepted(self, monkeypatch):
        # An exit-0 with empty stdout used to become "" -- falsy but not
        # None, so it would have slipped past `if chosen is None` and
        # silently persisted an empty path into prefs.
        monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="", stderr="",
        ))
        chosen, error = app.choose_folder_dialog("prompt", None)
        assert chosen is None
        assert error is not None


class TestChooseDirEndpoint:
    def test_genuine_failure_raises_502(self, monkeypatch):
        monkeypatch.setattr(app, "choose_folder_dialog", lambda *a, **k: (None, "boom"))
        with pytest.raises(HTTPException) as exc_info:
            app.choose_dir(app.ChooseDirRequest(which="source"))
        assert exc_info.value.status_code == 502

    def test_cancel_returns_cancelled_true(self, monkeypatch):
        monkeypatch.setattr(app, "choose_folder_dialog", lambda *a, **k: (None, None))
        result = app.choose_dir(app.ChooseDirRequest(which="source"))
        assert result["cancelled"] is True

    def test_success_saves_prefs_and_returns_path(self, monkeypatch):
        monkeypatch.setattr(app, "choose_folder_dialog", lambda *a, **k: ("/chosen/path", None))
        result = app.choose_dir(app.ChooseDirRequest(which="source"))
        assert result == {"cancelled": False, "path": "/chosen/path"}
        assert app.load_prefs()["source_dir"] == "/chosen/path"


class TestProcessGroupSelection:
    def test_default_processes_every_discovered_group(self, tmp_path, monkeypatch):
        groups = [fake_group(["a.mp4"]), fake_group(["b.mp4", "c.mp4"])]
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ("video", groups, [], [], {}))
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
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ("video", [group], [], [], {}))
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
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ("video", [current_group], [], [], {}))
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
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ("video", [group], [], [], {}))

        def boom(g, d):
            raise RuntimeError("ffmpeg exploded")
        monkeypatch.setattr(merge, "merge_group", boom)

        result = app.process(app.ProcessRequest(
            source_dir=str(tmp_path), destination_dir=str(tmp_path), gap_threshold_s=300,
        ))
        assert result["results"][0]["ok"] is False
        assert "ffmpeg exploded" in result["results"][0]["error"]

    def test_same_filename_in_different_subfolders_not_aliased(self, tmp_path, monkeypatch):
        # Regression: group identity used to be filename-only, so two
        # clips named identically in different *MEDIA subfolders (a real
        # DJI SD-card pagination pattern -- 100MEDIA, 101MEDIA, ... with
        # numbering restarting per folder) collapsed to the same dict key,
        # silently aliasing a "process" request onto the wrong group.
        group_a = fake_group(["100MEDIA/DJI_0001.MP4"])
        group_b = fake_group(["101MEDIA/DJI_0001.MP4"])
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ("video", [group_a, group_b], [], [], {}))
        calls = []
        monkeypatch.setattr(merge, "merge_group", lambda g, d: calls.append(g) or merge.MergeResult(
            ok=True, output_path=Path("/fake/out.mp4"),
            source_files=[c.mp4_path.name for c in g.clips],
        ))
        result = app.process(app.ProcessRequest(
            source_dir=str(tmp_path), destination_dir=str(tmp_path), gap_threshold_s=300,
            selected_groups=[["101MEDIA/DJI_0001.MP4"]],
        ))
        assert len(calls) == 1
        assert calls[0] is group_b
        assert len(result["results"]) == 1

    def test_failed_count_reflects_number_of_failures(self, tmp_path, monkeypatch):
        ok_group = fake_group(["a.mp4"])
        fail_group = fake_group(["b.mp4"])
        monkeypatch.setattr(app, "_discover_and_group", lambda *a, **k: ("video", [ok_group, fail_group], [], [], {}))

        def fake_merge(g, d):
            name = g.clips[0].mp4_path.name
            if name == "a.mp4":
                return merge.MergeResult(ok=True, output_path=Path("/fake/out.mp4"), source_files=[name])
            return merge.MergeResult(ok=False, output_path=None, source_files=[name], error="boom")
        monkeypatch.setattr(merge, "merge_group", fake_merge)

        result = app.process(app.ProcessRequest(
            source_dir=str(tmp_path), destination_dir=str(tmp_path), gap_threshold_s=300,
        ))
        assert result["failed_count"] == 1


class TestDetectSourceKind:
    def test_only_mp4_is_video(self, tmp_path):
        (tmp_path / "a.MP4").write_bytes(b"")
        assert app._detect_source_kind(tmp_path) == "video"

    def test_only_wav_is_audio(self, tmp_path):
        (tmp_path / "a.wav").write_bytes(b"")
        assert app._detect_source_kind(tmp_path) == "audio"

    def test_neither_is_empty(self, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"")
        assert app._detect_source_kind(tmp_path) == "empty"

    def test_truly_empty_dir_is_empty(self, tmp_path):
        assert app._detect_source_kind(tmp_path) == "empty"

    def test_both_present_prefers_video(self, tmp_path):
        # A folder holding both is not a real-world case (a drone card and
        # an audio recorder's card are never the same folder), but the
        # priority must still be deterministic rather than depend on
        # filesystem iteration order.
        (tmp_path / "a.MP4").write_bytes(b"")
        (tmp_path / "b.wav").write_bytes(b"")
        assert app._detect_source_kind(tmp_path) == "video"

    def test_case_insensitive_extensions(self, tmp_path):
        (tmp_path / "a.wAv").write_bytes(b"")
        assert app._detect_source_kind(tmp_path) == "audio"


class TestAudioProcessDispatch:
    def test_process_dispatches_to_audio_merge_for_audio_kind(self, tmp_path, monkeypatch):
        group = fake_audio_group(["a.wav", "b.wav"])
        monkeypatch.setattr(app, "_discover_and_group",
                             lambda *a, **k: ("audio", [group], [], [], {"trims": {}}))
        calls = []

        def fake_merge_audio(g, trims, d):
            calls.append((g, trims, d))
            return audio_merge.AudioMergeResult(
                ok=True, output_path=Path("/fake/out.wav"),
                source_files=[c.wav_path.name for c in g.clips],
            )
        monkeypatch.setattr(audio_merge, "merge_audio_group", fake_merge_audio)

        result = app.process(app.ProcessRequest(
            source_dir=str(tmp_path), destination_dir=str(tmp_path), gap_threshold_s=300,
        ))
        assert len(calls) == 1
        assert calls[0][0] is group
        assert calls[0][1] == {}
        assert result["results"][0]["ok"] is True
        assert result["results"][0]["source_files"] == ["a.wav", "b.wav"]

    def test_scan_reports_audio_kind_and_device_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app, "_detect_source_kind", lambda p: "audio")
        monkeypatch.setattr(audio_merge, "discover_audio_clips", lambda p: ([], []))
        monkeypatch.setattr(audio_merge, "group_audio_clips", lambda clips: ([], {}, []))
        result = app.scan(app.ScanRequest(source_dir=str(tmp_path), gap_threshold_s=300))
        assert result["kind"] == "audio"
        assert result["groups"] == []

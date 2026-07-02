"""Tests for merge.py -- grouping, path-safety, and telemetry-lookup logic.

These are pure-logic tests: no ffmpeg/ffprobe subprocess calls (that's
covered by manual verification against real Air3 footage, see the project
history). Constructs Clip/ClipProbe/SrtCue directly so grouping/threshold
behavior can be pinned precisely and cheaply.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import merge
from srt_parser import SrtCue

BASE = datetime(2026, 6, 29, 10, 0, 0)


def make_probe(**overrides):
    fields = dict(
        duration_s=60.0, codec_name="hevc", width=3840, height=2160,
        r_frame_rate="60000/1001", creation_time="2026-06-29T14:00:00.000000Z",
    )
    fields.update(overrides)
    return merge.ClipProbe(**fields)


def make_cue(wall_clock, has_gps=True, **overrides):
    fields = dict(
        framecnt=1, difftime_ms=33, cue_start_s=0.0, cue_end_s=0.033,
        wall_clock=wall_clock, iso=100, shutter="1/2000.0", fnum=1.7, ev=0.0,
        color_md="default", focal_len=24.0,
        latitude=41.5 if has_gps else None,
        longitude=-75.5 if has_gps else None,
        rel_alt=10.0 if has_gps else None,
        abs_alt=200.0 if has_gps else None,
        ct=5500,
    )
    fields.update(overrides)
    return SrtCue(**fields)


def make_clip(mp4_path, start_dt, end_dt, cues=None, probe=None, srt_error=None):
    cues = cues or []
    return merge.Clip(
        mp4_path=Path(mp4_path),
        srt_path=None,
        cues=cues,
        srt_error=srt_error,
        probe=probe or make_probe(),
        start_dt=start_dt,
        end_dt=end_dt,
        start_is_estimated=not bool(cues),
    )


class TestGroupClipsThreshold:
    """Pins the gap < gap_threshold_s contract (strict less-than): a gap
    exactly equal to the threshold must NOT merge."""

    def test_empty_list(self):
        assert merge.group_clips([], gap_threshold_s=300) == []

    def test_single_clip(self):
        clip = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        groups = merge.group_clips([clip], gap_threshold_s=300)
        assert len(groups) == 1
        assert groups[0].clips == [clip]
        assert groups[0].gap_to_next_s is None

    def test_gap_just_below_threshold_merges(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        b_start = a.end_dt + timedelta(seconds=299.999)
        b = make_clip("/fake/b.mp4", b_start, b_start + timedelta(seconds=60))
        groups = merge.group_clips([a, b], gap_threshold_s=300)
        assert len(groups) == 1
        assert groups[0].clips == [a, b]

    def test_gap_exactly_at_threshold_does_not_merge(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        b_start = a.end_dt + timedelta(seconds=300.0)
        b = make_clip("/fake/b.mp4", b_start, b_start + timedelta(seconds=60))
        groups = merge.group_clips([a, b], gap_threshold_s=300)
        assert len(groups) == 2

    def test_gap_just_above_threshold_splits(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        b_start = a.end_dt + timedelta(seconds=300.001)
        b = make_clip("/fake/b.mp4", b_start, b_start + timedelta(seconds=60))
        groups = merge.group_clips([a, b], gap_threshold_s=300)
        assert len(groups) == 2

    def test_negative_gap_merges(self):
        # Overlapping/out-of-order timestamps (e.g. clock jitter) should
        # still merge -- a negative gap is trivially < any positive threshold.
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        b_start = a.end_dt - timedelta(seconds=5)
        b = make_clip("/fake/b.mp4", b_start, b_start + timedelta(seconds=60))
        groups = merge.group_clips([a, b], gap_threshold_s=300)
        assert len(groups) == 1

    def test_three_clips_two_groups_with_correct_gap_to_next(self):
        a = make_clip("/fake/a.mp4", BASE, BASE + timedelta(seconds=60))
        b_start = a.end_dt + timedelta(seconds=1)  # chunk-split-style tiny gap
        b = make_clip("/fake/b.mp4", b_start, b_start + timedelta(seconds=60))
        c_start = b.end_dt + timedelta(seconds=3600)  # different flight
        c = make_clip("/fake/c.mp4", c_start, c_start + timedelta(seconds=60))
        groups = merge.group_clips([a, b, c], gap_threshold_s=300)
        assert len(groups) == 2
        assert groups[0].clips == [a, b]
        assert groups[1].clips == [c]
        assert groups[0].gap_to_next_s == pytest.approx(3600.0)
        assert groups[1].gap_to_next_s is None


class TestIso6709:
    def test_positive_lat_negative_lon(self):
        # No zero-padding on the integer part -- matches real Apple/QuickTime
        # ISO6709 location tags (e.g. "+37.3349-122.0090+075.000/").
        assert merge._iso6709(41.620801, -75.778972, 220.994) == "+41.6208-75.7790+220.994/"

    def test_precision_truncated_to_fixed_decimals(self):
        s = merge._iso6709(1.23456789, -2.3456789, 3.456789)
        assert s == "+1.2346-2.3457+3.457/"


class TestNextAvailablePath:
    def test_no_collision_returns_original(self, tmp_path):
        target = tmp_path / "AIR3_20260629_100000.mp4"
        assert merge._next_available_path(target) == target

    def test_one_collision_returns_v2(self, tmp_path):
        target = tmp_path / "AIR3_20260629_100000.mp4"
        target.write_bytes(b"")
        result = merge._next_available_path(target)
        assert result == tmp_path / "AIR3_20260629_100000_v2.mp4"

    def test_two_collisions_returns_v3(self, tmp_path):
        target = tmp_path / "AIR3_20260629_100000.mp4"
        target.write_bytes(b"")
        (tmp_path / "AIR3_20260629_100000_v2.mp4").write_bytes(b"")
        result = merge._next_available_path(target)
        assert result == tmp_path / "AIR3_20260629_100000_v3.mp4"


class TestCheckUniformStream:
    def test_matching_streams_returns_none(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe())
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe())
        assert merge._check_uniform_stream([a, b]) is None

    def test_mismatched_resolution_returns_message(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe(width=3840, height=2160))
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe(width=1920, height=1080))
        msg = merge._check_uniform_stream([a, b])
        assert msg is not None
        assert "a.mp4" in msg and "b.mp4" in msg

    def test_mismatched_codec_returns_message(self):
        a = make_clip("/fake/a.mp4", BASE, BASE, probe=make_probe(codec_name="hevc"))
        b = make_clip("/fake/b.mp4", BASE, BASE, probe=make_probe(codec_name="h264"))
        assert merge._check_uniform_stream([a, b]) is not None


class TestGpsCueLookup:
    def test_first_gps_cue_skips_clip_with_no_cues(self):
        no_srt = make_clip("/fake/a.mp4", BASE, BASE, cues=[])
        with_gps = make_clip("/fake/b.mp4", BASE, BASE, cues=[make_cue(BASE, has_gps=True)])
        found = merge._first_gps_cue([no_srt, with_gps])
        assert found is not None
        assert found.latitude == 41.5

    def test_first_gps_cue_skips_cues_without_gps(self):
        clip = make_clip("/fake/a.mp4", BASE, BASE, cues=[
            make_cue(BASE, has_gps=False),
            make_cue(BASE, has_gps=True, latitude=1.0, longitude=2.0),
        ])
        found = merge._first_gps_cue([clip])
        assert found.latitude == 1.0

    def test_last_gps_cue_finds_last_valid_before_tail_dropout(self):
        # Mirrors the real observed pattern: last N cues in the last clip
        # lost GPS lock right before touchdown.
        clip = make_clip("/fake/a.mp4", BASE, BASE, cues=[
            make_cue(BASE, has_gps=True, latitude=9.0, longitude=9.0),
            make_cue(BASE, has_gps=False),
            make_cue(BASE, has_gps=False),
        ])
        found = merge._last_gps_cue([clip])
        assert found is not None
        assert found.latitude == 9.0

    def test_no_gps_anywhere_returns_none(self):
        clip = make_clip("/fake/a.mp4", BASE, BASE, cues=[make_cue(BASE, has_gps=False)])
        assert merge._first_gps_cue([clip]) is None
        assert merge._last_gps_cue([clip]) is None


class TestGroupSummary:
    def _real_file(self, tmp_path, name, size=1024):
        p = tmp_path / name
        p.write_bytes(b"x" * size)
        return p

    def test_location_none_when_no_gps_in_group(self, tmp_path):
        clip = make_clip(self._real_file(tmp_path, "a.mp4"), BASE, BASE + timedelta(seconds=60),
                          cues=[make_cue(BASE, has_gps=False)])
        group = merge.ClipGroup(clips=[clip])
        summary = merge.group_summary(group)
        assert summary["start_location"] is None
        assert summary["end_location"] is None

    def test_missing_srt_listed_by_filename(self, tmp_path):
        clip = make_clip(self._real_file(tmp_path, "a.mp4"), BASE, BASE + timedelta(seconds=60),
                          cues=[], srt_error="no .SRT sidecar found")
        group = merge.ClipGroup(clips=[clip])
        summary = merge.group_summary(group)
        assert summary["missing_srt"] == ["a.mp4"]

    def test_total_size_and_duration_summed_across_clips(self, tmp_path):
        a = make_clip(self._real_file(tmp_path, "a.mp4", size=100), BASE, BASE + timedelta(seconds=10),
                       probe=make_probe(duration_s=10.0))
        b = make_clip(self._real_file(tmp_path, "b.mp4", size=200), BASE, BASE + timedelta(seconds=20),
                       probe=make_probe(duration_s=20.0))
        summary = merge.group_summary(merge.ClipGroup(clips=[a, b]))
        assert summary["total_size_bytes"] == 300
        assert summary["total_duration_s"] == 30.0

"""Tests for audio_merge.py -- per-device continuity detection, grouping,
and path-safety helpers for Zoom F3 / Sound Devices MixPre-6 / Zoom H3-VR.

Pure-logic tests: no ffmpeg/ffprobe/exiftool/bwfmetaedit subprocess calls
(that pipeline was verified separately against real sample recordings from
all three devices -- see project history). Constructs AudioClip/AudioProbe
directly so each device's continuity rule can be pinned precisely, including
the exact boundary values confirmed against real files.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audio_merge
from audio_probe import DEVICE_F3, DEVICE_H3VR, DEVICE_MIXPRE6, DEVICE_UNKNOWN, AudioProbe

BASE = datetime(2026, 6, 5, 14, 0, 0)


def make_probe(**overrides):
    fields = dict(
        duration_s=100.0, duration_samples=4_800_000, sample_rate=48000,
        channels=4, bits_per_sample=24, codec_name="pcm_s24le",
        device=DEVICE_H3VR, encoded_by="ZOOM H3-VR Handy Recorder",
        start_dt=BASE, time_reference_samples=0,
        scene="SoBo-0819", take="006", note="Infield turn 2",
    )
    fields.update(overrides)
    return AudioProbe(**fields)


def make_clip(wav_name, start_dt=BASE, probe=None, start_is_estimated=False):
    return audio_merge.AudioClip(
        wav_path=Path(f"/fake/{wav_name}"),
        probe=probe or make_probe(),
        start_dt=start_dt,
        start_is_estimated=start_is_estimated,
    )


class TestNaturalSortKey:
    def test_zero_padded_sorts_numerically(self):
        assert audio_merge._natural_sort_key(Path("Take_009.WAV")) < audio_merge._natural_sort_key(Path("Take_010.WAV"))

    def test_unpadded_sorts_numerically_not_lexically(self):
        # A plain string sort would put "10" before "9" -- this must not.
        assert audio_merge._natural_sort_key(Path("Take_9.WAV")) < audio_merge._natural_sort_key(Path("Take_10.WAV"))

    def test_no_trailing_digits_falls_back_to_stem(self):
        key = audio_merge._natural_sort_key(Path("no_number_here.WAV"))
        assert key == ("no_number_here", -1)


class TestF3Continuity:
    """Confirmed real split: '...Swoope_001.WAV' -> '...Swoope_002.WAV',
    exact 0-sample gap (BASE.md/conversation record)."""

    def _pair(self, idx_prev=1, idx_cur=2, prefix_prev="20251003-611 Swoope", prefix_cur=None,
              date_cur=None, tr_prev=0, dur_prev=134_784_000, tr_cur=134_784_000, sr=96000):
        prefix_cur = prefix_cur if prefix_cur is not None else prefix_prev
        prev = make_clip(
            f"{prefix_prev}_{idx_prev:03d}.WAV",
            probe=make_probe(device=DEVICE_F3, time_reference_samples=tr_prev,
                              duration_samples=dur_prev, sample_rate=sr),
        )
        cur = make_clip(
            f"{prefix_cur}_{idx_cur:03d}.WAV",
            start_dt=date_cur or BASE,
            probe=make_probe(device=DEVICE_F3, time_reference_samples=tr_cur, sample_rate=sr),
        )
        return prev, cur

    def test_exact_zero_gap_is_continuous(self):
        prev, cur = self._pair()
        warnings = []
        assert audio_merge._f3_continuous(prev, cur, warnings) == 0
        assert warnings == []

    def test_small_overlap_within_bound_is_continuous_and_trimmed(self):
        # cur starts 100 samples before prev's computed end -> a real (if
        # tiny) overlap to trim, well under the 2s plausibility bound.
        prev, cur = self._pair(tr_cur=134_784_000 - 100)
        assert audio_merge._f3_continuous(prev, cur, []) == 100

    def test_discrepancy_just_over_bound_refuses(self):
        sr = 96000
        over_samples = int(audio_merge.MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S * sr) + 1
        prev, cur = self._pair(tr_cur=134_784_000 + over_samples)  # a gap, not an overlap
        warnings = []
        assert audio_merge._f3_continuous(prev, cur, warnings) is None
        assert any("off by" in w for w in warnings)

    def test_discrepancy_just_under_bound_accepted(self):
        sr = 96000
        under_samples = int(audio_merge.MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S * sr) - 1
        prev, cur = self._pair(tr_cur=134_784_000 + under_samples)
        assert audio_merge._f3_continuous(prev, cur, []) == 0  # gap, not overlap -> no trim, but still continuous

    def test_discrepancy_exactly_at_bound_is_accepted(self):
        # Pins the strict `>` (not `>=`) contract: a discrepancy exactly
        # equal to the bound must land on the same side as "just under",
        # not "just over" -- the check only refuses when *strictly* past it.
        sr = 96000
        exact_samples = int(audio_merge.MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S * sr)
        prev, cur = self._pair(tr_cur=134_784_000 + exact_samples)
        assert audio_merge._f3_continuous(prev, cur, []) == 0

    def test_non_consecutive_index_is_not_continuous(self):
        prev, cur = self._pair(idx_prev=1, idx_cur=3)
        assert audio_merge._f3_continuous(prev, cur, []) is None

    def test_mismatched_prefix_is_not_continuous(self):
        prev, cur = self._pair(prefix_cur="20251003-611 Craigsville")
        assert audio_merge._f3_continuous(prev, cur, []) is None

    def test_different_calendar_date_is_not_continuous(self):
        prev, cur = self._pair(date_cur=BASE + timedelta(days=1))
        assert audio_merge._f3_continuous(prev, cur, []) is None

    def test_missing_time_reference_is_not_continuous(self):
        prev = make_clip("20251003-611 Swoope_001.WAV",
                          probe=make_probe(device=DEVICE_F3, time_reference_samples=None))
        cur = make_clip("20251003-611 Swoope_002.WAV",
                         probe=make_probe(device=DEVICE_F3, time_reference_samples=0))
        assert audio_merge._f3_continuous(prev, cur, []) is None


class TestMixPre6Continuity:
    """Confirmed real split: MixPre-275.WAV -> MixPre-276.WAV, exact
    0-sample gap, matching BwfxmlNote='Tehachapi Loop' (Take increments:
    275/276, so it's deliberately NOT used as a signal here)."""

    def _pair(self, idx_prev=275, idx_cur=276, note_prev="Tehachapi Loop", note_cur="Tehachapi Loop",
              take_prev="275", take_cur="276", tr_cur=100_000_000, dur_prev=100_000_000, sr=96000):
        prev = make_clip(
            f"MixPre-{idx_prev}.WAV",
            probe=make_probe(device=DEVICE_MIXPRE6, note=note_prev, take=take_prev,
                              time_reference_samples=0, duration_samples=dur_prev, sample_rate=sr),
        )
        cur = make_clip(
            f"MixPre-{idx_cur}.WAV",
            probe=make_probe(device=DEVICE_MIXPRE6, note=note_cur, take=take_cur,
                              time_reference_samples=tr_cur, sample_rate=sr),
        )
        return prev, cur

    def test_matching_note_exact_gap_is_continuous(self):
        assert audio_merge._mixpre6_continuous(*self._pair(), []) == 0

    def test_take_number_is_not_a_gating_field_even_when_wildly_different(self):
        # Explicitly confirms Take is NOT a gating field on this device --
        # uses a large, non-sequential jump (not just +1) so this is
        # genuinely distinct from the default pair, which already has
        # take_prev="275"/take_cur="276".
        prev, cur = self._pair(take_prev="003", take_cur="899")
        assert audio_merge._mixpre6_continuous(prev, cur, []) == 0

    def test_discrepancy_just_over_bound_refuses(self):
        sr = 96000
        over_samples = int(audio_merge.MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S * sr) + 1
        prev, cur = self._pair(tr_cur=100_000_000 + over_samples)
        warnings = []
        assert audio_merge._mixpre6_continuous(prev, cur, warnings) is None
        assert any("off by" in w for w in warnings)

    def test_discrepancy_just_under_bound_accepted(self):
        sr = 96000
        under_samples = int(audio_merge.MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S * sr) - 1
        prev, cur = self._pair(tr_cur=100_000_000 + under_samples)
        assert audio_merge._mixpre6_continuous(prev, cur, []) == 0

    def test_discrepancy_exactly_at_bound_is_accepted(self):
        sr = 96000
        exact_samples = int(audio_merge.MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S * sr)
        prev, cur = self._pair(tr_cur=100_000_000 + exact_samples)
        assert audio_merge._mixpre6_continuous(prev, cur, []) == 0

    def test_mismatched_note_is_not_continuous(self):
        prev, cur = self._pair(note_cur="Caliente downtown")
        assert audio_merge._mixpre6_continuous(prev, cur, []) is None

    def test_empty_note_on_both_sides_is_not_continuous(self):
        # Deliberately conservative: filename+timing alone, with zero
        # corroborating scene/note text, is one fewer signal than every
        # confirmed real case -- refuse rather than guess.
        prev, cur = self._pair(note_prev="", note_cur="")
        assert audio_merge._mixpre6_continuous(prev, cur, []) is None

    def test_non_consecutive_index_is_not_continuous(self):
        prev, cur = self._pair(idx_prev=275, idx_cur=277)
        assert audio_merge._mixpre6_continuous(prev, cur, []) is None

    def test_non_mixpre_filename_pattern_is_not_continuous(self):
        prev = make_clip("Take-275.WAV", probe=make_probe(device=DEVICE_MIXPRE6, note="x"))
        cur = make_clip("Take-276.WAV", probe=make_probe(device=DEVICE_MIXPRE6, note="x"))
        assert audio_merge._mixpre6_continuous(prev, cur, []) is None


class TestH3VRContinuity:
    """Confirmed real split: SoBo-0819_006/007/008.WAV, take number CONSTANT
    ("006") across all three, each boundary trimming exactly 17,280 samples
    (-0.36s) at 48kHz. Confirmed NOT diagnostic: the same -17,280-sample
    signature also appears between two separate, take-number-incrementing
    takes (StV02->StV03) -- so continuity here is decided by take number,
    never by the sample-timing alone."""

    def _pair(self, idx_prev=6, idx_cur=7, take_prev="006", take_cur="006",
              tr_prev=0, dur_prev=178_913_280, tr_cur=178_913_280 - 17_280, sr=48000,
              prefix="SoBo-0819"):
        prev = make_clip(
            f"{prefix}_{idx_prev:03d}.WAV",
            probe=make_probe(device=DEVICE_H3VR, take=take_prev, time_reference_samples=tr_prev,
                              duration_samples=dur_prev, sample_rate=sr),
        )
        cur = make_clip(
            f"{prefix}_{idx_cur:03d}.WAV",
            probe=make_probe(device=DEVICE_H3VR, take=take_cur, time_reference_samples=tr_cur, sample_rate=sr),
        )
        return prev, cur

    def test_confirmed_real_overlap_value_is_continuous_and_trimmed(self):
        prev, cur = self._pair()
        assert audio_merge._h3vr_continuous(prev, cur, []) == 17_280

    def test_constant_take_number_required(self):
        prev, cur = self._pair(take_prev="006", take_cur="006")
        assert audio_merge._h3vr_continuous(prev, cur, []) == 17_280

    def test_incrementing_take_number_is_not_continuous_even_with_same_overlap_signature(self):
        # The exact real-world trap: identical -17,280-sample timing
        # signature, but Take differs -> these are two separate takes.
        prev, cur = self._pair(take_prev="010", take_cur="011")
        assert audio_merge._h3vr_continuous(prev, cur, []) is None

    def test_empty_take_on_both_sides_is_not_continuous(self):
        prev, cur = self._pair(take_prev="", take_cur="")
        assert audio_merge._h3vr_continuous(prev, cur, []) is None

    def test_missing_time_reference_falls_back_to_untrimmed_append_with_warning(self):
        prev = make_clip("SoBo-0819_006.WAV",
                          probe=make_probe(device=DEVICE_H3VR, take="006", time_reference_samples=None))
        cur = make_clip("SoBo-0819_007.WAV",
                         probe=make_probe(device=DEVICE_H3VR, take="006", time_reference_samples=None))
        warnings = []
        assert audio_merge._h3vr_continuous(prev, cur, warnings) == 0
        assert any("no TimeReference" in w for w in warnings)

    def test_implausibly_large_computed_trim_falls_back_to_untrimmed_append_with_warning(self):
        # Take number already confirms continuity here -- an outsized
        # computed trim should degrade to a plain append, not refuse the
        # merge outright (unlike F3/MixPre, where an outsized discrepancy
        # means continuity itself is in doubt).
        sr = 48000
        over_samples = int(audio_merge.MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S * sr) + 1
        prev, cur = self._pair(tr_cur=0 - over_samples, dur_prev=0)  # huge overlap
        warnings = []
        assert audio_merge._h3vr_continuous(prev, cur, warnings) == 0
        assert any("implausibly large" in w for w in warnings)

    def test_discrepancy_exactly_at_bound_is_trimmed_not_degraded(self):
        # At exactly the bound, take number already confirmed continuity,
        # and the discrepancy check's strict `>` means this still counts
        # as a "plausible" (if maximal) computed trim, not a degrade case.
        sr = 48000
        exact_samples = int(audio_merge.MAX_PLAUSIBLE_BOUNDARY_DISCREPANCY_S * sr)
        prev, cur = self._pair(tr_cur=0 - exact_samples, dur_prev=0)
        warnings = []
        assert audio_merge._h3vr_continuous(prev, cur, warnings) == exact_samples
        assert warnings == []

    def test_non_consecutive_index_is_not_continuous(self):
        prev, cur = self._pair(idx_prev=6, idx_cur=8)
        assert audio_merge._h3vr_continuous(prev, cur, []) is None


class TestCheckContinuityDispatch:
    def test_different_devices_never_continuous(self):
        prev = make_clip("a_001.WAV", probe=make_probe(device=DEVICE_F3))
        cur = make_clip("a_002.WAV", probe=make_probe(device=DEVICE_H3VR))
        assert audio_merge._check_continuity(prev, cur, []) is None

    def test_unknown_device_never_auto_merged(self):
        prev = make_clip("a_001.WAV", probe=make_probe(device=DEVICE_UNKNOWN))
        cur = make_clip("a_002.WAV", probe=make_probe(device=DEVICE_UNKNOWN))
        assert audio_merge._check_continuity(prev, cur, []) is None


class TestGroupAudioClips:
    def test_empty_list(self):
        groups, trims, warnings = audio_merge.group_audio_clips([])
        assert groups == [] and trims == {} and warnings == []

    def test_single_clip(self):
        clip = make_clip("SoBo-0819_006.WAV")
        groups, trims, _ = audio_merge.group_audio_clips([clip])
        assert len(groups) == 1
        assert groups[0].clips == [clip]
        assert trims == {}

    def test_confirmed_h3vr_three_way_split_groups_together_with_correct_trims(self):
        c6 = make_clip("SoBo-0819_006.WAV",
                        probe=make_probe(device=DEVICE_H3VR, take="006",
                                          time_reference_samples=0, duration_samples=178_913_280))
        c7 = make_clip("SoBo-0819_007.WAV",
                        probe=make_probe(device=DEVICE_H3VR, take="006",
                                          time_reference_samples=178_913_280 - 17_280, duration_samples=178_913_280))
        c8 = make_clip("SoBo-0819_008.WAV",
                        probe=make_probe(device=DEVICE_H3VR, take="006",
                                          time_reference_samples=2 * 178_913_280 - 2 * 17_280))
        groups, trims, _ = audio_merge.group_audio_clips([c6, c7, c8])
        assert len(groups) == 1
        assert [c.wav_path.name for c in groups[0].clips] == ["SoBo-0819_006.WAV", "SoBo-0819_007.WAV", "SoBo-0819_008.WAV"]
        assert trims[(c6.wav_path, c7.wav_path)] == 17_280
        assert trims[(c7.wav_path, c8.wav_path)] == 17_280

    def test_groups_sorted_by_start_time_not_discovery_order(self):
        early = make_clip("a_001.WAV", start_dt=BASE, probe=make_probe(device=DEVICE_UNKNOWN))
        late = make_clip("b_001.WAV", start_dt=BASE + timedelta(hours=1), probe=make_probe(device=DEVICE_UNKNOWN))
        groups, _, _ = audio_merge.group_audio_clips([late, early])
        assert [g.clips[0].wav_path.name for g in groups] == ["a_001.WAV", "b_001.WAV"]


class TestCheckUniformStream:
    def test_matching_returns_none(self):
        a = make_clip("a.WAV", probe=make_probe())
        b = make_clip("b.WAV", probe=make_probe())
        assert audio_merge._check_uniform_stream([a, b]) is None

    def test_mismatched_sample_rate_returns_message(self):
        a = make_clip("a.WAV", probe=make_probe(sample_rate=48000))
        b = make_clip("b.WAV", probe=make_probe(sample_rate=96000))
        msg = audio_merge._check_uniform_stream([a, b])
        assert msg is not None and "a.WAV" in msg and "b.WAV" in msg

    def test_mismatched_channels_returns_message(self):
        a = make_clip("a.WAV", probe=make_probe(channels=2))
        b = make_clip("b.WAV", probe=make_probe(channels=4))
        assert audio_merge._check_uniform_stream([a, b]) is not None


class TestNextAvailablePathAndSanitize:
    def test_sanitize_strips_unsafe_characters(self):
        assert audio_merge._sanitize_filename_part('a/b:c*d?e"f<g>h|i\\j') == "a_b_c_d_e_f_g_h_i_j"

    def test_sanitize_empty_falls_back_to_untitled(self):
        assert audio_merge._sanitize_filename_part("   ") == "untitled"

    def test_sanitize_preserves_safe_text(self):
        assert audio_merge._sanitize_filename_part("Tehachapi Loop") == "Tehachapi Loop"


class TestAudioGroupSummary:
    def test_shape_and_device_scene_note_surfaced(self, tmp_path):
        p1 = tmp_path / "a.WAV"
        p1.write_bytes(b"x" * 100)
        clip = audio_merge.AudioClip(
            wav_path=p1, probe=make_probe(device=DEVICE_MIXPRE6, scene="MixPre", note="Tehachapi Loop"),
            start_dt=BASE, start_is_estimated=False,
        )
        summary = audio_merge.audio_group_summary(audio_merge.AudioClipGroup(clips=[clip]))
        assert summary["device"] == DEVICE_MIXPRE6
        assert summary["scene"] == "MixPre"
        assert summary["note"] == "Tehachapi Loop"
        assert summary["start_location"] is None and summary["end_location"] is None
        assert summary["total_size_bytes"] == 100
        assert summary["clip_paths"] == [str(p1)]

    def test_end_dt_is_start_plus_duration_not_last_clip_start(self, tmp_path):
        # Regression: end_dt previously used last.start_dt (the last clip's
        # START time), silently mislabeling it as the group's end -- pin
        # the correct value (matches merge.py's group_summary convention).
        p1 = tmp_path / "a.WAV"
        p1.write_bytes(b"x")
        last_start = BASE + timedelta(hours=1)
        clip = audio_merge.AudioClip(
            wav_path=p1, probe=make_probe(duration_s=42.5), start_dt=last_start, start_is_estimated=False,
        )
        summary = audio_merge.audio_group_summary(audio_merge.AudioClipGroup(clips=[clip]))
        assert summary["end_dt"] == (last_start + timedelta(seconds=42.5)).isoformat()

"""Tests for srt_parser.py -- DJI Air3 .SRT telemetry parsing.

This is the sole path by which flight telemetry (GPS, altitude, camera
settings) survives into the merged output. A parser that silently drops or
mis-parses fields loses that data permanently (the source .SRT is never
touched, but nothing downstream re-derives it). Fixtures below are drawn
from real Air3 footage, including a real GPS-lock-dropout tail observed on
an actual card (see srt_parser.py's FIELDS_RE comment).
"""

import re
import sys
from datetime import datetime
from pathlib import Path

import pytest

from srt_parser import SrtCue, SrtParseError, format_cue_block, parse_srt

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cue_block(
    index=1,
    start_tc="00:00:00,000",
    end_tc="00:00:00,033",
    framecnt=1,
    difftime_ms=33,
    dt="2026-06-29 10:27:07.249",
    gps='[latitude: 41.620801] [longitude: -75.778972] [rel_alt: 36.400 abs_alt: 220.994] ',
    ct=5579,
):
    return (
        f"{index}\n"
        f"{start_tc} --> {end_tc}\n"
        f'<font size="28">FrameCnt: {framecnt}, DiffTime: {difftime_ms}ms\n'
        f"{dt}\n"
        f"[iso: 210] [shutter: 1/5000.0] [fnum: 1.7] [ev: 0] [color_md : default] "
        f"[focal_len: 24.00] {gps}[ct: {ct}] </font>\n"
    )


def write_srt(tmp_path, text, name="clip.SRT"):
    p = tmp_path / name
    p.write_text(text)
    return p


class TestParseBasicCue:
    def test_all_fields_extracted_with_gps(self, tmp_path):
        srt = write_srt(tmp_path, cue_block())
        cues = parse_srt(srt)
        assert len(cues) == 1
        c = cues[0]
        assert c.framecnt == 1
        assert c.difftime_ms == 33
        assert c.cue_start_s == 0.0
        assert c.cue_end_s == pytest.approx(0.033)
        assert c.wall_clock == datetime(2026, 6, 29, 10, 27, 7, 249000)
        assert c.iso == 210
        assert c.shutter == "1/5000.0"  # preserved as raw string, not parsed to a float
        assert c.fnum == 1.7
        assert c.ev == 0.0
        assert c.color_md == "default"
        assert c.focal_len == 24.00
        assert c.latitude == 41.620801
        assert c.longitude == -75.778972
        assert c.rel_alt == 36.400
        assert c.abs_alt == 220.994
        assert c.ct == 5579
        assert c.has_gps is True

    def test_multiple_cues_parsed_in_order(self, tmp_path):
        text = cue_block(index=1, framecnt=1) + "\n" + cue_block(
            index=2, framecnt=2, start_tc="00:00:00,033", end_tc="00:00:00,066",
        )
        cues = parse_srt(write_srt(tmp_path, text))
        assert [c.framecnt for c in cues] == [1, 2]

    def test_negative_longitude_and_ev_parsed(self, tmp_path):
        text = cue_block(
            gps='[latitude: 41.627580] [longitude: -75.776184] [rel_alt: 83.700 abs_alt: 268.294] ',
        )
        c = parse_srt(write_srt(tmp_path, text))[0]
        assert c.longitude == -75.776184


class TestGpsDropout:
    """The GPS block is captured as one atomic optional group (see
    srt_parser.py FIELDS_RE comment): DJI drops all four GPS fields together
    during a lock dropout, never a partial subset. This is the exact
    real-world pattern found on an actual card's last ~12 frames before
    touchdown."""

    def test_gps_entirely_absent_is_valid_and_sets_none(self, tmp_path):
        text = cue_block(gps="")
        c = parse_srt(write_srt(tmp_path, text))[0]
        assert c.has_gps is False
        assert c.latitude is None
        assert c.longitude is None
        assert c.rel_alt is None
        assert c.abs_alt is None
        # non-GPS fields on the same cue are unaffected
        assert c.iso == 210
        assert c.ct == 5579

    def test_gps_partially_present_is_a_parse_error(self, tmp_path):
        # Only latitude/longitude present, rel_alt/abs_alt missing: this is
        # not a recognized shape (real dropouts drop all four together), so
        # it must raise rather than silently accept a malformed line.
        text = cue_block(gps="[latitude: 41.620801] [longitude: -75.778972] ")
        with pytest.raises(SrtParseError):
            parse_srt(write_srt(tmp_path, text))

    def test_gps_missing_only_altitude_is_a_parse_error(self, tmp_path):
        text = cue_block(
            gps="[latitude: 41.620801] [longitude: -75.778972] [rel_alt: 36.400] ",
        )
        with pytest.raises(SrtParseError):
            parse_srt(write_srt(tmp_path, text))


class TestMalformedBlocks:
    def test_too_few_lines_raises(self, tmp_path):
        text = "1\n00:00:00,000 --> 00:00:00,033\nshort block\n"
        with pytest.raises(SrtParseError):
            parse_srt(write_srt(tmp_path, text))

    def test_non_numeric_cue_index_raises(self, tmp_path):
        text = cue_block().replace("1\n00:00:00,000", "not_a_number\n00:00:00,000", 1)
        with pytest.raises(SrtParseError):
            parse_srt(write_srt(tmp_path, text))

    def test_missing_arrow_in_timecode_line_raises(self, tmp_path):
        text = cue_block().replace("00:00:00,000 --> 00:00:00,033", "00:00:00,000 to 00:00:00,033")
        with pytest.raises(SrtParseError):
            parse_srt(write_srt(tmp_path, text))

    def test_malformed_timecode_raises(self, tmp_path):
        text = cue_block().replace("00:00:00,000 -->", "0:0:0.000 -->")
        with pytest.raises(SrtParseError):
            parse_srt(write_srt(tmp_path, text))

    def test_inverted_timecode_raises(self, tmp_path):
        # Regression: an end time before the start time used to parse
        # cleanly into a negative-duration cue, which then flowed straight
        # into merge.py's cumulative_offset arithmetic and corrupted every
        # subsequent cue's timestamp in the merged subtitle track.
        text = cue_block(start_tc="00:00:05,000", end_tc="00:00:00,000")
        with pytest.raises(SrtParseError):
            parse_srt(write_srt(tmp_path, text))

    def test_missing_framecnt_header_raises(self, tmp_path):
        text = cue_block().replace("FrameCnt: 1, DiffTime: 33ms", "garbage header")
        with pytest.raises(SrtParseError):
            parse_srt(write_srt(tmp_path, text))

    def test_unparseable_datetime_raises(self, tmp_path):
        text = cue_block(dt="not-a-date")
        with pytest.raises(SrtParseError):
            parse_srt(write_srt(tmp_path, text))

    def test_missing_required_field_raises(self, tmp_path):
        # `ct` is required (not part of the optional GPS group) -- dropping
        # it must fail loudly rather than silently defaulting.
        text = cue_block().replace("[ct: 5579]", "")
        with pytest.raises(SrtParseError):
            parse_srt(write_srt(tmp_path, text))

    def test_error_message_includes_offending_block(self, tmp_path):
        text = cue_block(dt="not-a-date")
        with pytest.raises(SrtParseError) as exc_info:
            parse_srt(write_srt(tmp_path, text))
        assert "not-a-date" in str(exc_info.value)


class TestFormatCueBlockRoundtrip:
    def _reparse(self, tmp_path, formatted_block):
        # format_cue_block emits one cue's worth of text (index, timecode,
        # header, datetime, fields) -- exactly parse_srt's expected block shape.
        return parse_srt(write_srt(tmp_path, formatted_block))[0]

    def test_roundtrip_with_gps(self, tmp_path):
        original = parse_srt(write_srt(tmp_path, cue_block(), name="orig.SRT"))[0]
        formatted = format_cue_block(7, 12.5, 12.7, original)
        reparsed = self._reparse(tmp_path, formatted)
        assert reparsed.framecnt == 7
        assert reparsed.latitude == pytest.approx(original.latitude)
        assert reparsed.longitude == pytest.approx(original.longitude)
        assert reparsed.rel_alt == pytest.approx(original.rel_alt)
        assert reparsed.abs_alt == pytest.approx(original.abs_alt)
        assert reparsed.shutter == original.shutter
        assert reparsed.has_gps is True

    def test_roundtrip_without_gps_omits_bracket_entirely(self, tmp_path):
        original = parse_srt(write_srt(tmp_path, cue_block(gps=""), name="orig.SRT"))[0]
        formatted = format_cue_block(1, 0.0, 0.033, original)
        assert "latitude" not in formatted
        assert "None" not in formatted  # the classic silent-substitution bug
        reparsed = self._reparse(tmp_path, formatted)
        assert reparsed.has_gps is False
        assert reparsed.latitude is None

    def test_offset_time_applied_to_output_timecode(self, tmp_path):
        original = parse_srt(write_srt(tmp_path, cue_block(), name="orig.SRT"))[0]
        formatted = format_cue_block(1, 3661.0, 3661.5, original)  # 1h 1m 1s
        assert "01:01:01,000 --> 01:01:01,500" in formatted

    def test_millisecond_rollover_boundary(self, tmp_path):
        # fmt_tc rounds ms and rolls over into the next second at exactly
        # 1000ms -- pin the boundary rather than trust the rounding math.
        original = parse_srt(write_srt(tmp_path, cue_block(), name="orig.SRT"))[0]
        formatted = format_cue_block(1, 1.9996, 2.0, original)  # rounds to 2.000s, not 1.1000
        first_line_after_index = formatted.splitlines()[1]
        assert first_line_after_index.startswith("00:00:02,000")


class TestTimecodeBoundaries:
    def test_zero(self, tmp_path):
        c = parse_srt(write_srt(tmp_path, cue_block(start_tc="00:00:00,000", end_tc="00:00:00,001")))[0]
        assert c.cue_start_s == 0.0

    def test_one_hour_boundary(self, tmp_path):
        c = parse_srt(write_srt(tmp_path, cue_block(start_tc="01:00:00,000", end_tc="01:00:00,033")))[0]
        assert c.cue_start_s == 3600.0

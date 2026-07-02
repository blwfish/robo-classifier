"""Tests for audio_probe.py's classify_device() -- the single canonical
device classifier every continuity check in audio_merge.py dispatches off
of. Real encoded_by/Originator strings observed on actual hardware are used
verbatim (including Zoom H3-VR's trailing whitespace, confirmed present via
both ffprobe and exiftool on real files)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio_probe import DEVICE_F3, DEVICE_H3VR, DEVICE_MIXPRE6, DEVICE_UNKNOWN, classify_device


class TestClassifyDevice:
    def test_zoom_f3(self):
        assert classify_device("ZOOM F3") == DEVICE_F3

    def test_zoom_h3vr_with_real_trailing_whitespace(self):
        # Confirmed on real hardware: both ffprobe's encoded_by tag and
        # exiftool's Originator carry trailing spaces on this device.
        assert classify_device("ZOOM H3-VR Handy Recorder       ") == DEVICE_H3VR

    def test_mixpre6_with_serial_suffix(self):
        assert classify_device("SoundDev: MixPre-6 QC0417289027") == DEVICE_MIXPRE6

    def test_mixpre6_prefix_alone(self):
        assert classify_device("SoundDev: MixPre-6") == DEVICE_MIXPRE6

    def test_none_is_unknown(self):
        assert classify_device(None) == DEVICE_UNKNOWN

    def test_empty_string_is_unknown(self):
        assert classify_device("") == DEVICE_UNKNOWN

    def test_unrelated_device_is_unknown(self):
        assert classify_device("TASCAM DR-40") == DEVICE_UNKNOWN

    def test_different_mixpre_model_is_not_assumed_to_be_mixpre6(self):
        # Only MixPre-6 behavior has been confirmed against real split data
        # (see audio_merge.py docstring) -- a MixPre-3/MixPre-10/etc. must
        # not silently inherit MixPre-6's continuity rule.
        assert classify_device("SoundDev: MixPre-10T QC9999999") == DEVICE_UNKNOWN

    def test_leading_whitespace_still_classified(self):
        assert classify_device("  ZOOM F3") == DEVICE_F3

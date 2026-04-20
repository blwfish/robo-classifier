"""Tests for burst-detection logic in classify.py.

Burst detection is the heart of classify.py: it decides which frames in a
sequence count as the "same moment" so only the best example ends up in the
winners list (and receives a robo_9x keyword). Regressions here either miss
interesting frames or flood the Lightroom catalog with near-duplicates.
"""

import pytest

import classify


# =============================================================================
# parse_burst_base — filename-based burst identification
# =============================================================================

class TestParseBurstBase:
    def test_no_burst_suffix(self):
        assert classify.parse_burst_base("2025-04-25-Z9_BLW0124.jpg") == \
               "2025-04-25-Z9_BLW0124"

    def test_single_burst_suffix_stripped(self):
        assert classify.parse_burst_base("2025-04-25-Z9_BLW0124-3.jpg") == \
               "2025-04-25-Z9_BLW0124"

    def test_multiple_burst_suffixes_stripped(self):
        # -3-2 means "third burst variant, second edit" or similar — strip both
        assert classify.parse_burst_base("2025-04-25-Z9_BLW0124-3-2.jpg") == \
               "2025-04-25-Z9_BLW0124"

    def test_long_numeric_suffix_preserved(self):
        # A 6-digit trailing number is a sequence number, not a burst variant —
        # stripping it would collapse unrelated frames into one "burst"
        assert classify.parse_burst_base("2026-01-31-PCA-Sebring-023573.jpg") == \
               "2026-01-31-PCA-Sebring-023573"

    def test_extension_stripped(self):
        assert classify.parse_burst_base("foo.JPG") == "foo"
        assert classify.parse_burst_base("foo.jpeg") == "foo"

    def test_no_dashes(self):
        assert classify.parse_burst_base("IMG_1234.jpg") == "IMG_1234"

    def test_two_digit_suffix_stripped(self):
        # 1-2 digit suffixes are burst variants
        assert classify.parse_burst_base("BLW0124-12.jpg") == "BLW0124"

    def test_three_digit_suffix_preserved(self):
        # 3+ digit trailing number is probably a sequence number, not a variant
        assert classify.parse_burst_base("BLW0124-123.jpg") == "BLW0124-123"


# =============================================================================
# get_tier_keyword — confidence → keyword mapping
# =============================================================================

class TestTierKeyword:
    def test_below_threshold_returns_none(self):
        assert classify.get_tier_keyword(0.0) is None
        assert classify.get_tier_keyword(0.89) is None
        assert classify.get_tier_keyword(0.899) is None

    @pytest.mark.parametrize("conf,expected", [
        (0.90, "robo_90"),
        (0.91, "robo_91"),
        (0.95, "robo_95"),
        (0.99, "robo_99"),
        (1.0,  "robo_99"),
    ])
    def test_exact_boundaries(self, conf, expected):
        assert classify.get_tier_keyword(conf) == expected

    def test_monotonic(self):
        # Higher confidence → same-or-higher-tier keyword
        prev_tier = -1
        for conf in [0.90 + i * 0.01 for i in range(10)]:
            kw = classify.get_tier_keyword(conf)
            assert kw is not None
            n = int(kw.removeprefix("robo_"))
            assert n >= prev_tier
            prev_tier = n

    def test_floating_point_just_below_boundary(self):
        # 0.9499999 rounds naturally into the 94 bucket, not 95
        assert classify.get_tier_keyword(0.9499) == "robo_94"

    def test_above_1_clamped(self):
        # Not a realistic scenario, but shouldn't crash
        assert classify.get_tier_keyword(1.5) == "robo_99"


# =============================================================================
# burst_dedup — filename-based grouping
# =============================================================================

def _result(filename, conf, path=None, cls=None):
    """Build a synthetic inference result."""
    p = path or f"/images/{filename}"
    return {
        "filename": filename,
        "path": p,
        "confidence_select": conf,
        "classification": cls if cls is not None
                          else ("select" if conf >= 0.5 else "reject"),
    }


class TestBurstDedupByFilename:
    def test_single_frame_burst(self):
        results = [_result("BLW0001.jpg", 0.95)]
        winners, bursts = classify.burst_dedup(results)
        assert len(winners) == 1
        assert winners[0]["filename"] == "BLW0001.jpg"
        assert list(bursts.keys()) == ["BLW0001"]

    def test_burst_picks_highest_confidence(self):
        results = [
            _result("BLW0001-1.jpg", 0.70),
            _result("BLW0001-2.jpg", 0.95),   # winner
            _result("BLW0001-3.jpg", 0.85),
        ]
        winners, bursts = classify.burst_dedup(results)
        assert len(winners) == 1
        assert winners[0]["filename"] == "BLW0001-2.jpg"
        # All three frames are grouped together
        assert len(bursts["BLW0001"]) == 3

    def test_reject_burst_produces_no_winner(self):
        # Best-in-burst is classified 'reject' → no winner even if it's the max
        results = [
            _result("BLW0001-1.jpg", 0.2, cls="reject"),
            _result("BLW0001-2.jpg", 0.4, cls="reject"),
        ]
        winners, bursts = classify.burst_dedup(results)
        assert winners == []
        assert len(bursts["BLW0001"]) == 2

    def test_mixed_burst_winner_must_be_select(self):
        # Best confidence is on a 'reject' frame — no winner, even though some
        # frames in the burst are 'select'. This matches the code: only the
        # top confidence_select frame is considered, then its classification
        # is checked.
        results = [
            _result("BLW0001-1.jpg", 0.95, cls="reject"),   # top conf
            _result("BLW0001-2.jpg", 0.80, cls="select"),
        ]
        winners, bursts = classify.burst_dedup(results)
        assert winners == []

    def test_multiple_bursts_independent(self):
        results = [
            _result("BLW0001-1.jpg", 0.95),
            _result("BLW0001-2.jpg", 0.80),
            _result("BLW0002-1.jpg", 0.92),
        ]
        winners, bursts = classify.burst_dedup(results)
        assert len(winners) == 2
        assert set(bursts.keys()) == {"BLW0001", "BLW0002"}


# =============================================================================
# burst_dedup_by_time — time-based grouping with adaptive threshold
# =============================================================================

class TestBurstDedupByTime:
    def _make(self, filenames, times, shutters=None):
        """Build results + capture_times for (filename, time_offset) pairs."""
        results = [_result(fn, 0.95, path=f"/x/{fn}") for fn in filenames]
        base = 1_700_000_000.0
        ct = {}
        for fn, t, sh in zip(filenames, times, shutters or [0] * len(filenames)):
            from pathlib import Path
            ct[Path(f"/x/{fn}")] = {"timestamp": base + t, "shutter": sh}
        return results, ct

    def test_close_frames_are_one_burst(self):
        results, ct = self._make(
            ["a.jpg", "b.jpg", "c.jpg"],
            times=[0.0, 0.1, 0.2],
        )
        winners, bursts = classify.burst_dedup_by_time(results, ct, threshold=0.5)
        assert len(bursts) == 1
        assert len(list(bursts.values())[0]) == 3

    def test_gap_over_threshold_splits_burst(self):
        results, ct = self._make(
            ["a.jpg", "b.jpg", "c.jpg"],
            times=[0.0, 0.1, 2.0],   # 1.9s gap > 0.5 threshold
        )
        winners, bursts = classify.burst_dedup_by_time(results, ct, threshold=0.5)
        assert len(bursts) == 2

    def test_adaptive_threshold_for_slow_shutter(self):
        """A 1s exposure followed by another 0.8s later should NOT split,
        because the adaptive threshold becomes max(0.5, 1.0+0.1) = 1.1s."""
        results, ct = self._make(
            ["a.jpg", "b.jpg"],
            times=[0.0, 0.8],
            shutters=[1.0, 1.0],
        )
        winners, bursts = classify.burst_dedup_by_time(results, ct, threshold=0.5)
        assert len(bursts) == 1

    def test_sorted_by_timestamp_not_input_order(self):
        # Input order shouldn't affect grouping — timestamps do
        results, ct = self._make(
            ["c.jpg", "a.jpg", "b.jpg"],
            times=[2.0, 0.0, 0.1],
        )
        winners, bursts = classify.burst_dedup_by_time(results, ct, threshold=0.5)
        # a + b cluster, c alone
        assert len(bursts) == 2

    def test_missing_capture_time_treated_as_zero(self):
        # Frames without metadata all get timestamp=0, clustering them together
        results = [_result("a.jpg", 0.9), _result("b.jpg", 0.95)]
        winners, bursts = classify.burst_dedup_by_time(results, {}, threshold=0.5)
        assert len(bursts) == 1

    def test_routing_falls_back_to_filename_without_threshold(self):
        """burst_dedup() without time_threshold should use filename grouping."""
        results = [
            _result("BLW0001-1.jpg", 0.9),
            _result("BLW0001-2.jpg", 0.95),
        ]
        winners, bursts = classify.burst_dedup(results)
        # Filename-based: single burst keyed by "BLW0001"
        assert "BLW0001" in bursts

    def test_routing_uses_time_when_threshold_provided(self):
        from pathlib import Path
        results = [
            _result("a.jpg", 0.9, path="/x/a.jpg"),
            _result("b.jpg", 0.95, path="/x/b.jpg"),
        ]
        ct = {
            Path("/x/a.jpg"): {"timestamp": 100.0, "shutter": 0.001},
            Path("/x/b.jpg"): {"timestamp": 105.0, "shutter": 0.001},
        }
        winners, bursts = classify.burst_dedup(
            results, capture_times=ct, time_threshold=0.5
        )
        # Time-based: 5s gap splits into 2 bursts, keys look like "burst_0"
        assert any(k.startswith("burst_") for k in bursts.keys())
        assert len(bursts) == 2

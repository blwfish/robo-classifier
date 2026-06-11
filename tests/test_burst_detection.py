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

    def test_lightroom_collision_suffix_preserved(self):
        # A Lightroom collision rename (-3) must NOT be stripped. These are
        # unrelated shots that happen to share a sequence number; stripping
        # the suffix would merge them into one burst and lose a keeper.
        assert classify.parse_burst_base("2025-04-25-Z9_BLW0124-3.jpg") == \
               "2025-04-25-Z9_BLW0124-3"

    def test_multiple_collision_suffixes_preserved(self):
        assert classify.parse_burst_base("2025-04-25-Z9_BLW0124-3-2.jpg") == \
               "2025-04-25-Z9_BLW0124-3-2"

    def test_long_numeric_suffix_preserved(self):
        assert classify.parse_burst_base("2026-01-31-PCA-Sebring-023573.jpg") == \
               "2026-01-31-PCA-Sebring-023573"

    def test_extension_stripped(self):
        assert classify.parse_burst_base("foo.JPG") == "foo"
        assert classify.parse_burst_base("foo.jpeg") == "foo"

    def test_no_dashes(self):
        assert classify.parse_burst_base("IMG_1234.jpg") == "IMG_1234"

    def test_short_numeric_suffix_preserved(self):
        # Lightroom collision renames with short suffixes must not be stripped
        assert classify.parse_burst_base("BLW0124-12.jpg") == "BLW0124-12"

    def test_long_numeric_suffix_also_preserved(self):
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

    @pytest.mark.parametrize("conf,expected", [
        # Just below 0.91 → robo_90
        (0.9099, "robo_90"),
        # Just above 0.91 → robo_91
        (0.9101, "robo_91"),
        # Just below 0.95 → robo_94
        (0.9499, "robo_94"),
        # Just above 0.95 → robo_95
        (0.9501, "robo_95"),
        # Just below 0.98 → robo_97
        (0.9799, "robo_97"),
        # Just above 0.98 → robo_98
        (0.9801, "robo_98"),
    ])
    def test_internal_boundary_just_below_and_above(self, conf, expected):
        """Pin the < vs >= contract at three internal tier boundaries."""
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

    def test_each_unique_filename_is_own_burst(self):
        # With no suffix stripping, each file is its own singleton burst.
        # All three select frames produce winners — no deduplication.
        results = [
            _result("BLW0001.jpg", 0.70),
            _result("BLW0002.jpg", 0.95),
            _result("BLW0003.jpg", 0.85),
        ]
        winners, bursts = classify.burst_dedup(results)
        assert len(winners) == 3
        assert set(bursts.keys()) == {"BLW0001", "BLW0002", "BLW0003"}

    def test_reject_produces_no_winner(self):
        results = [_result("BLW0001.jpg", 0.2, cls="reject")]
        winners, bursts = classify.burst_dedup(results)
        assert winners == []
        assert "BLW0001" in bursts

    def test_select_produces_winner(self):
        results = [_result("BLW0001.jpg", 0.95, cls="select")]
        winners, bursts = classify.burst_dedup(results)
        assert len(winners) == 1
        assert winners[0]["filename"] == "BLW0001.jpg"

    def test_lightroom_collision_rename_is_independent_burst(self):
        # BLW0001-2.jpg is a Lightroom collision rename of a different file —
        # it must NOT be grouped with BLW0001.jpg; both are separate winners.
        results = [
            _result("BLW0001.jpg",   0.95),
            _result("BLW0001-2.jpg", 0.80),
        ]
        winners, bursts = classify.burst_dedup(results)
        assert len(winners) == 2
        assert "BLW0001"   in bursts
        assert "BLW0001-2" in bursts


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
        """A 1s exposure followed by another 1.15s later should NOT split.
        Empirically measured Z9/Z6III overhead at 1s shutter: 0.13-0.15s.
        Buffer is +0.20s → adaptive_threshold = max(0.5, 1.0+0.20) = 1.20s.
        Gap of 1.15s < 1.20s → stays in same burst."""
        results, ct = self._make(
            ["a.jpg", "b.jpg"],
            times=[0.0, 1.15],
            shutters=[1.0, 1.0],
        )
        winners, bursts = classify.burst_dedup_by_time(results, ct, threshold=0.5)
        assert len(bursts) == 1

    def test_adaptive_threshold_splits_intentional_pause(self):
        """A 1s exposure with a >1.20s pause should split into a new burst.
        Gap of 2.0s > max(0.5, 1.0+0.20)=1.20s → new burst."""
        results, ct = self._make(
            ["a.jpg", "b.jpg"],
            times=[0.0, 2.0],
            shutters=[1.0, 1.0],
        )
        winners, bursts = classify.burst_dedup_by_time(results, ct, threshold=0.5)
        assert len(bursts) == 2

    def test_sorted_by_timestamp_not_input_order(self):
        # Input order shouldn't affect grouping — timestamps do
        results, ct = self._make(
            ["c.jpg", "a.jpg", "b.jpg"],
            times=[2.0, 0.0, 0.1],
        )
        winners, bursts = classify.burst_dedup_by_time(results, ct, threshold=0.5)
        # a + b cluster, c alone
        assert len(bursts) == 2

    def test_missing_capture_time_becomes_singleton(self):
        # Frames absent from capture_times get their own singleton burst so
        # they don't accidentally cluster with real-time neighbours.
        results = [_result("a.jpg", 0.9), _result("b.jpg", 0.95)]
        winners, bursts = classify.burst_dedup_by_time(results, {}, threshold=0.5)
        assert len(bursts) == 2

    def test_routing_falls_back_to_filename_without_threshold(self):
        """burst_dedup() without time_threshold should use filename grouping."""
        results = [
            _result("BLW0001.jpg", 0.9),
            _result("BLW0002.jpg", 0.95),
        ]
        winners, bursts = classify.burst_dedup(results)
        # Filename-based: each unique stem is its own burst
        assert "BLW0001" in bursts
        assert "BLW0002" in bursts
        assert len(bursts) == 2

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

    # --- medium-severity: threshold boundary and empty input ---

    def test_gap_exactly_at_threshold_not_split(self):
        # Condition is strict `>`, so gap == adaptive_threshold stays in one burst.
        # shutter=0 → adaptive_threshold = max(0.5, 0+0.20) = 0.5
        results, ct = self._make(["a.jpg", "b.jpg"], times=[0.0, 0.5])
        winners, bursts = classify.burst_dedup_by_time(results, ct, threshold=0.5)
        assert len(bursts) == 1

    def test_gap_just_above_threshold_splits(self):
        # gap = 0.5001 > 0.5 → new burst
        results, ct = self._make(["a.jpg", "b.jpg"], times=[0.0, 0.5001])
        winners, bursts = classify.burst_dedup_by_time(results, ct, threshold=0.5)
        assert len(bursts) == 2

    def test_adaptive_threshold_gap_below_boundary_not_split(self):
        # shutter=1.0 → adaptive_threshold = max(0.5, 1.0+0.20) = 1.20
        # gap=1.199 < 1.20 → stays in one burst.
        # (gap=1.20 exactly is untestable here: large base timestamps cause
        # floating-point cancellation that makes the computed gap > 1.20.)
        results, ct = self._make(["a.jpg", "b.jpg"], times=[0.0, 1.199], shutters=[1.0, 1.0])
        winners, bursts = classify.burst_dedup_by_time(results, ct, threshold=0.5)
        assert len(bursts) == 1

    def test_empty_input_returns_empty(self):
        winners, bursts = classify.burst_dedup_by_time([], {}, threshold=0.5)
        assert winners == []
        assert bursts == {}

    def test_burst_dedup_filename_empty_input(self):
        winners, bursts = classify.burst_dedup([])
        assert winners == []
        assert bursts == {}

    def test_burst_dedup_time_empty_input_via_router(self):
        from pathlib import Path
        winners, bursts = classify.burst_dedup([], capture_times={}, time_threshold=0.5)
        assert winners == []
        assert bursts == {}

    # --- Fix 5: time_threshold=0 and time_threshold=None both mean "disabled" ---

    def test_time_threshold_zero_routes_to_filename_mode(self):
        """time_threshold=0 must NOT enter time-based grouping (0 is the disabled sentinel).
        Filename-based grouping should be used instead."""
        from pathlib import Path
        results = [
            _result("BLW0001.jpg", 0.9, path="/x/BLW0001.jpg"),
            _result("BLW0002.jpg", 0.95, path="/x/BLW0002.jpg"),
        ]
        ct = {
            Path("/x/BLW0001.jpg"): {"timestamp": 100.0, "shutter": 0.001},
            Path("/x/BLW0002.jpg"): {"timestamp": 100.1, "shutter": 0.001},
        }
        winners, bursts = classify.burst_dedup(results, capture_times=ct, time_threshold=0)
        # Filename-based: stems are the keys, NOT "burst_N" keys
        assert "BLW0001" in bursts
        assert "BLW0002" in bursts
        assert not any(k.startswith("burst_") for k in bursts.keys())

    def test_time_threshold_none_with_capture_times_routes_to_filename_mode(self):
        """time_threshold=None with a populated capture_times dict must still use
        filename-based grouping — None means 'disabled', not 'zero tolerance'."""
        from pathlib import Path
        results = [
            _result("A.jpg", 0.9, path="/x/A.jpg"),
            _result("B.jpg", 0.95, path="/x/B.jpg"),
        ]
        ct = {
            Path("/x/A.jpg"): {"timestamp": 200.0, "shutter": 0.001},
            Path("/x/B.jpg"): {"timestamp": 200.05, "shutter": 0.001},
        }
        winners, bursts = classify.burst_dedup(results, capture_times=ct, time_threshold=None)
        # Must be filename-based
        assert "A" in bursts
        assert "B" in bursts
        assert not any(k.startswith("burst_") for k in bursts.keys())

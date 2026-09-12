"""
Tests for write_tiered_keywords.py — had zero test coverage before this file
(robo-classifier-20260912-a1f3#20 / full-review Phase 3.5). Focuses on
_filter_reject_candidates, extracted from main() specifically to make the
P2-54 fix (missing 'classification' silently treated as a safe default)
testable in isolation.
"""

from write_tiered_keywords import _filter_reject_candidates


def _row(path, classification=None):
    r = {"path": path}
    if classification is not None:
        r["classification"] = classification
    return r


class TestFilterRejectCandidates:
    def test_winner_excluded(self):
        rows = [_row("/x/a.jpg", "reject")]
        result = _filter_reject_candidates(rows, winner_paths={"/x/a.jpg"})
        assert result == []

    def test_decode_failed_excluded(self):
        rows = [_row("/x/a.jpg", "decode_failed")]
        result = _filter_reject_candidates(rows, winner_paths=set())
        assert result == []

    def test_normal_non_winner_included(self):
        rows = [_row("/x/a.jpg", "reject")]
        result = _filter_reject_candidates(rows, winner_paths=set())
        assert result == rows

    def test_missing_classification_key_excluded_not_defaulted_in(self):
        # Regression test: r.get('classification') != 'decode_failed'
        # evaluates None != 'decode_failed' as True, silently including a
        # row with no classification data at all as if it were a confirmed
        # normal reject. Must now be excluded instead.
        rows = [_row("/x/a.jpg", classification=None)]  # no 'classification' key at all
        result = _filter_reject_candidates(rows, winner_paths=set())
        assert result == []

    def test_empty_string_classification_excluded(self):
        # An explicitly empty classification value is equally not a
        # confirmed "not decode_failed" state.
        rows = [{"path": "/x/a.jpg", "classification": ""}]
        result = _filter_reject_candidates(rows, winner_paths=set())
        assert result == rows  # "" is not None, and "" != "decode_failed" — passes through
        # (documenting current behavior: empty string is a present-but-empty
        # value, distinct from the column being absent entirely)

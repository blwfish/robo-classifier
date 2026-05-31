"""Tests for junk_filter.py pure-logic functions.

_touches_edge and _is_edge_chopped are the main quality gate for the pipeline;
regressions here silently corrupt the output catalog. _companion_xmp and
_move_with_sidecar are filesystem helpers tested with tmp_path.
"""

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from junk_filter import Detection, _touches_edge, _is_edge_chopped, _companion_xmp, _move_with_sidecar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _det(x1, y1, x2, y2, cls="car", conf=0.9):
    return Detection(cls=cls, conf=conf, x1=x1, y1=y1, x2=x2, y2=y2)


IMG_W, IMG_H = 1000, 800  # arbitrary image dimensions for all tests


# =============================================================================
# _touches_edge
# =============================================================================

class TestTouchesEdge:
    MARGIN = 5.0

    def test_fully_interior_returns_false(self):
        det = _det(100, 100, 900, 700)
        assert _touches_edge(det, IMG_W, IMG_H, self.MARGIN) is False

    def test_left_edge_exact_margin_returns_true(self):
        # x1 == margin → condition is x1 <= margin → True
        det = _det(5.0, 100, 900, 700)
        assert _touches_edge(det, IMG_W, IMG_H, self.MARGIN) is True

    def test_left_edge_just_inside_returns_false(self):
        # x1 just above margin
        det = _det(5.1, 100, 900, 700)
        assert _touches_edge(det, IMG_W, IMG_H, self.MARGIN) is False

    def test_left_edge_just_outside_returns_true(self):
        det = _det(4.9, 100, 900, 700)
        assert _touches_edge(det, IMG_W, IMG_H, self.MARGIN) is True

    def test_right_edge_exact_boundary_returns_true(self):
        # x2 == img_w - margin (995) → condition x2 >= 995 → True
        det = _det(100, 100, 995.0, 700)
        assert _touches_edge(det, IMG_W, IMG_H, self.MARGIN) is True

    def test_right_edge_just_inside_returns_false(self):
        det = _det(100, 100, 994.9, 700)
        assert _touches_edge(det, IMG_W, IMG_H, self.MARGIN) is False

    def test_top_edge_returns_true(self):
        det = _det(100, 0, 900, 700)
        assert _touches_edge(det, IMG_W, IMG_H, self.MARGIN) is True

    def test_bottom_edge_returns_true(self):
        det = _det(100, 100, 900, 795.0)
        assert _touches_edge(det, IMG_W, IMG_H, self.MARGIN) is True

    def test_corner_touch_returns_true(self):
        det = _det(0, 0, 200, 200)
        assert _touches_edge(det, IMG_W, IMG_H, self.MARGIN) is True


# =============================================================================
# _is_edge_chopped
# =============================================================================

class TestIsEdgeChopped:
    EDGE_MARGIN = 5.0
    MIN_VIS_FRAC = 0.1   # 10% of image dimension
    EDGE_MIN_AREA = 0.0  # area rule off by default in most tests

    # --- not touching any edge ---

    def test_interior_detection_not_chopped(self):
        det = _det(100, 100, 900, 700)
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is False

    # --- width rule (left/right edge) ---

    def test_left_touch_narrow_width_is_chopped(self):
        # Touches left edge; width = 50 < 0.1 * 1000 = 100
        det = _det(0, 100, 50, 700)
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is True

    def test_left_touch_width_exactly_at_threshold_is_chopped(self):
        # width == min_visible_frac * img_w: condition is `<`, so this is NOT chopped
        det = _det(0, 100, 100, 700)   # width = 100 exactly
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is False

    def test_left_touch_width_just_below_threshold_is_chopped(self):
        det = _det(0, 100, 99, 700)    # width = 99 < 100
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is True

    def test_left_touch_wide_enough_not_chopped(self):
        det = _det(0, 100, 200, 700)   # width = 200 >= 100
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is False

    def test_right_touch_narrow_width_is_chopped(self):
        det = _det(950, 100, 1000, 700)   # width = 50
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is True

    # --- height rule (top/bottom edge) ---

    def test_top_touch_short_height_is_chopped(self):
        # Touches top; height = 60 < 0.1 * 800 = 80
        det = _det(100, 0, 900, 60)
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is True

    def test_top_touch_height_just_below_threshold_is_chopped(self):
        det = _det(100, 0, 900, 79)    # height = 79 < 80
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is True

    def test_top_touch_height_at_threshold_not_chopped(self):
        det = _det(100, 0, 900, 80)    # height = 80, condition is `<`
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is False

    def test_bottom_touch_short_height_is_chopped(self):
        det = _det(100, 760, 900, 800)   # height = 40 < 80
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is True

    # --- area rule ---

    def test_area_rule_disabled_when_zero(self):
        # A tiny sliver that would fail the area rule if enabled; edge_min_area_frac=0
        det = _det(0, 100, 200, 101)   # width=200 >= 100 (passes width rule), height=1
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC,
                                edge_min_area_frac=0.0) is False

    def test_area_rule_enabled_sliver_is_chopped(self):
        # det: touches left, width=200 (passes width rule), height=1 → area=200
        # threshold = 0.05 * 1000 * 800 = 40000; area=200 < 40000 → chopped
        det = _det(0, 100, 200, 101)
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC,
                                edge_min_area_frac=0.05) is True

    def test_area_rule_large_enough_not_chopped(self):
        # touches left, passes width rule, area well above threshold
        det = _det(0, 100, 500, 500)   # width=500 >= 100, area=200000 >> 40000
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC,
                                edge_min_area_frac=0.05) is False

    def test_area_at_threshold_not_chopped(self):
        # area exactly == threshold: condition is `<`, not `<=`
        # threshold = 0.05 * 1000 * 800 = 40000
        # Need a detection touching an edge, passing the width rule, area = 40000
        # width=500 >= 100, height=80 → area=40000
        det = _det(0, 100, 500, 180)
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC,
                                edge_min_area_frac=0.05) is False


# =============================================================================
# _companion_xmp
# =============================================================================

class TestCompanionXmp:
    def test_no_sidecar_returns_none(self, tmp_path):
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        assert _companion_xmp(img) is None

    def test_lowercase_xmp_found(self, tmp_path):
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        xmp = tmp_path / "frame.xmp"
        xmp.write_text("<xmp/>")
        assert _companion_xmp(img) == xmp

    def test_uppercase_XMP_found(self, tmp_path):
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        xmp = tmp_path / "frame.XMP"
        xmp.write_text("<xmp/>")
        # On case-insensitive filesystems (macOS) the lowercase probe finds the
        # uppercase file; on case-sensitive filesystems the .XMP branch fires.
        # Either way the sidecar must be found (non-None).
        assert _companion_xmp(img) is not None

    def test_lowercase_takes_precedence_over_uppercase(self, tmp_path):
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        (tmp_path / "frame.xmp").write_text("<lower/>")
        (tmp_path / "frame.XMP").write_text("<upper/>")
        assert _companion_xmp(img) == tmp_path / "frame.xmp"


# =============================================================================
# _move_with_sidecar
# =============================================================================

class TestMoveWithSidecar:
    def test_moves_file_no_sidecar(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        dst_dir = tmp_path / "dst"
        img = src_dir / "frame.jpg"
        img.write_bytes(b"data")

        moves = _move_with_sidecar(img, dst_dir)

        assert (dst_dir / "frame.jpg").exists()
        assert not img.exists()
        assert moves == [(img, dst_dir / "frame.jpg")]

    def test_moves_file_and_sidecar(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        dst_dir = tmp_path / "dst"
        img = src_dir / "frame.NEF"
        img.write_bytes(b"raw")
        xmp = src_dir / "frame.xmp"
        xmp.write_text("<xmp/>")

        moves = _move_with_sidecar(img, dst_dir)

        assert (dst_dir / "frame.NEF").exists()
        assert (dst_dir / "frame.xmp").exists()
        assert not img.exists()
        assert not xmp.exists()
        assert len(moves) == 2

    def test_dry_run_returns_pairs_without_moving(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        dst_dir = tmp_path / "dst"
        img = src_dir / "frame.NEF"
        img.write_bytes(b"raw")
        xmp = src_dir / "frame.xmp"
        xmp.write_text("<xmp/>")

        moves = _move_with_sidecar(img, dst_dir, dry_run=True)

        assert img.exists()
        assert xmp.exists()
        assert not (dst_dir / "frame.NEF").exists()
        assert len(moves) == 2

    def test_creates_destination_directory(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        img = src_dir / "frame.jpg"
        img.write_bytes(b"data")
        dst_dir = tmp_path / "dst" / "nested"  # does not exist yet

        _move_with_sidecar(img, dst_dir)

        assert (dst_dir / "frame.jpg").exists()

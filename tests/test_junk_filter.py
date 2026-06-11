"""Tests for junk_filter.py pure-logic functions.

_touches_edge and _is_edge_chopped are the main quality gate for the pipeline;
regressions here silently corrupt the output catalog. _companion_xmp and
_move_with_sidecar are filesystem helpers tested with tmp_path.
"""

import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from junk_filter import (
    Detection,
    _touches_edge,
    _is_edge_chopped,
    _companion_xmp,
    _move_with_sidecar,
    detect_junk,
)


# ---------------------------------------------------------------------------
# Mock YOLO helpers
# ---------------------------------------------------------------------------

def _make_yolo_box(cls_idx: int, conf: float, x1: float, y1: float, x2: float, y2: float):
    """Return a mock YOLO box (pred.boxes element)."""
    box = MagicMock()
    box.cls.item.return_value = cls_idx
    box.conf.item.return_value = conf
    # xyxy[0] must have a .tolist() method (the code calls b.xyxy[0].tolist())
    xyxy_row = MagicMock()
    xyxy_row.tolist.return_value = [x1, y1, x2, y2]
    box.xyxy = [xyxy_row]
    return box


def _make_yolo_pred(
    img_w: int,
    img_h: int,
    boxes,   # list of mock boxes
    names: dict = None,
):
    """Return a mock YOLO Results object for a single image."""
    pred = MagicMock()
    pred.orig_shape = (img_h, img_w)
    pred.names = names if names is not None else {0: 'car', 1: 'truck', 2: 'person'}
    pred.boxes = boxes
    pred.speed = {'preprocess': 1.0, 'inference': 5.0, 'postprocess': 0.5}
    return pred


def _make_yolo_model(preds_per_batch):
    """
    Return a mock YOLO model whose predict() cycles through preds_per_batch
    (a list of lists — one inner list of preds per predict() call).

    For tests that only call predict() once, pass [[pred1, pred2, ...]] or
    just [[pred]] for a single image.
    """
    model = MagicMock()
    call_count = [0]

    def predict(arrays, **kwargs):
        result = preds_per_batch[call_count[0]]
        call_count[0] += 1
        return result

    model.predict.side_effect = predict
    return model


def _make_test_jpeg(path: Path, width: int = 100, height: int = 80) -> Path:
    """Write a tiny solid-colour JPEG and return the path."""
    from PIL import Image as PILImage
    img = PILImage.new('RGB', (width, height), color=(128, 64, 32))
    img.save(str(path), format='JPEG')
    return path


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

    def test_left_touch_width_at_threshold_not_chopped(self):
        # width == min_visible_frac * img_w: condition is strict `<`, so NOT chopped
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

    def test_corner_touch_height_fails_while_width_passes(self):
        # Touches left AND top. Width=200 ≥ 100 (passes width rule),
        # height=40 < 80 (fails height rule) → chopped.
        det = _det(0, 0, 200, 40)
        assert _is_edge_chopped(det, IMG_W, IMG_H,
                                self.EDGE_MARGIN, self.MIN_VIS_FRAC) is True

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


# =============================================================================
# detect_junk — min_area_frac threshold (Fix 5 & Fix 6)
# =============================================================================

# All tests use a 1000×800 image so min_area_frac * img_w * img_h = area_threshold.
# Default min_area_frac = 0.002  →  threshold = 0.002 * 1000 * 800 = 1600 px².

_DETECT_IMG_W = 1000
_DETECT_IMG_H = 800
_MIN_AREA_FRAC = 0.002
_AREA_THRESHOLD = _MIN_AREA_FRAC * _DETECT_IMG_W * _DETECT_IMG_H  # 1600.0


def _make_car_box(x1, y1, x2, y2, conf=0.9):
    """Shorthand: a 'car' box (cls_idx=0) for the standard names dict."""
    return _make_yolo_box(0, conf, x1, y1, x2, y2)


class TestDetectJunkMinAreaFrac:
    """
    Pin the `min_area_frac` pre-filter that silently skips tiny detections
    before they reach _is_edge_chopped.

    Image: 1000×800.  area threshold = 0.002 * 1000 * 800 = 1600 px².
    """

    def _run_one(self, tmp_path, box, min_area_frac=_MIN_AREA_FRAC,
                 edge_min_area_frac=0.0, min_visible_frac=0.1):
        """Helper: run detect_junk on a single test JPEG with a single mock box."""
        img_path = _make_test_jpeg(tmp_path / "frame.jpg",
                                   width=_DETECT_IMG_W, height=_DETECT_IMG_H)
        pred = _make_yolo_pred(_DETECT_IMG_W, _DETECT_IMG_H, [box])
        model = _make_yolo_model([[pred]])
        results = detect_junk(
            [img_path],
            model=model,
            device='cpu',
            min_area_frac=min_area_frac,
            edge_min_area_frac=edge_min_area_frac,
            min_visible_frac=min_visible_frac,
        )
        assert len(results) == 1
        return results[0]

    # Fix 5, case 1: area just below threshold → detection skipped → no_vehicle
    def test_area_just_below_threshold_detection_skipped(self, tmp_path):
        # area = 1599 < 1600  → filtered out → treated as no_vehicle
        # 1599 = 41 * 39
        box = _make_car_box(x1=100, y1=100, x2=141, y2=139, conf=0.9)
        # actual area: 41 * 39 = 1599
        result = self._run_one(tmp_path, box)
        assert result.reason == 'no_vehicle', (
            "Detection below min_area_frac threshold should be skipped (no_vehicle)"
        )
        assert result.is_junk is True
        assert result.detections == []  # was filtered, never appended
        assert result.usable_detections == []

    # Fix 5, case 2: area exactly at threshold → NOT skipped (condition is strict `<`)
    def test_area_exactly_at_threshold_detection_not_skipped(self, tmp_path):
        # area = 1600 exactly  → 1600 < 1600 is False → detection kept
        # 1600 = 40 * 40; place well away from edges so it's usable
        box = _make_car_box(x1=100, y1=100, x2=140, y2=140, conf=0.9)
        result = self._run_one(tmp_path, box)
        assert result.reason == 'keep', (
            "Detection at exactly min_area_frac threshold should NOT be skipped"
        )
        assert result.is_junk is False
        assert len(result.detections) == 1

    # Fix 5, case 3: area just above threshold → processed normally → keep
    def test_area_just_above_threshold_detection_kept(self, tmp_path):
        # area = 1601 = 41 * 39 + small adjustment; use 41*40=1640
        box = _make_car_box(x1=100, y1=100, x2=141, y2=140, conf=0.9)
        result = self._run_one(tmp_path, box)
        assert result.reason == 'keep'
        assert result.is_junk is False
        assert len(result.detections) == 1


class TestDetectJunkMinAreaFracEdgeChopComposition:
    """
    Fix 6: Verify that min_area_frac pre-filter and _is_edge_chopped compose
    correctly.

    (a) area < min_area_frac  →  vehicle ignored entirely (not edge-chopped, not counted).
    (b) area >= min_area_frac AND edge-chopped  →  counted in 'all_edge_chopped'.
    """

    def _run_one(self, tmp_path, box, min_area_frac=_MIN_AREA_FRAC,
                 edge_min_area_frac=0.0, min_visible_frac=0.1, edge_margin_px=5.0):
        img_path = _make_test_jpeg(tmp_path / "frame.jpg",
                                   width=_DETECT_IMG_W, height=_DETECT_IMG_H)
        pred = _make_yolo_pred(_DETECT_IMG_W, _DETECT_IMG_H, [box])
        model = _make_yolo_model([[pred]])
        results = detect_junk(
            [img_path],
            model=model,
            device='cpu',
            min_area_frac=min_area_frac,
            edge_min_area_frac=edge_min_area_frac,
            min_visible_frac=min_visible_frac,
            edge_margin_px=edge_margin_px,
        )
        assert len(results) == 1
        return results[0]

    def test_sub_threshold_edge_vehicle_is_ignored_entirely(self, tmp_path):
        """
        A tiny vehicle touching the left edge with area below min_area_frac:
        should be ignored entirely — reason is 'no_vehicle', not 'all_edge_chopped'.

        This pins that the area filter fires BEFORE edge-chop logic, so the
        vehicle isn't even counted in detections.
        """
        # Place a tiny box touching left edge, area = 39*39 = 1521 < 1600
        box = _make_car_box(x1=0, y1=100, x2=39, y2=139, conf=0.9)
        result = self._run_one(tmp_path, box)
        assert result.reason == 'no_vehicle', (
            "Sub-threshold vehicle touching edge must be ignored (not edge-chopped)"
        )
        assert result.detections == []
        assert result.usable_detections == []

    def test_at_threshold_edge_vehicle_triggers_all_edge_chopped(self, tmp_path):
        """
        A vehicle touching the left edge with area >= min_area_frac AND
        visible width below min_visible_frac: should land in 'all_edge_chopped'.

        area = 40*40 = 1600 >= 1600 (not filtered by area pre-filter).
        visible width = 40 < 0.1 * 1000 = 100  → edge-chopped.
        """
        box = _make_car_box(x1=0, y1=100, x2=40, y2=140, conf=0.9)
        result = self._run_one(tmp_path, box)
        assert result.reason == 'all_edge_chopped', (
            "Vehicle at min_area_frac threshold, edge-chopped, must count as all_edge_chopped"
        )
        assert result.is_junk is True
        assert len(result.detections) == 1
        assert len(result.usable_detections) == 0

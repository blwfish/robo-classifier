"""Tests for ui/review.py — XMP color labels and crop rectangle I/O.

ui/review.py is imported via importlib to avoid the naming conflict with the
root-level review.py CLI script. All tests that actually write XMP metadata
require exiftool on PATH and are skipped automatically when it is absent.
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # for image_utils dependency

spec = importlib.util.spec_from_file_location("ui_review", ROOT / "ui" / "review.py")
ui_review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ui_review)

# Pull the public API into module scope for convenience.
set_label  = ui_review.set_label
set_crop   = ui_review.set_crop
clear_crop = ui_review.clear_crop
read_state = ui_review.read_state
get_roll_angle = ui_review.get_roll_angle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXIFTOOL_PRESENT = shutil.which("exiftool") is not None
requires_exiftool = pytest.mark.skipif(
    not _EXIFTOOL_PRESENT, reason="exiftool not found"
)

# Optional: real Z9 NEF for get_roll_angle integration test.
SAMPLE_NEF = Path(
    "/Volumes/archive3/images/2026/2026-02-28 Brunswick/20260228104700-Z9_BLW4747.NEF"
)
requires_nef = pytest.mark.skipif(
    not SAMPLE_NEF.exists(), reason="archive3 not mounted"
)


def make_jpeg(path: Path) -> Path:
    """Write a minimal valid JPEG to *path* and return it."""
    from PIL import Image
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(str(path), "JPEG")
    return path


# =============================================================================
# Module import sanity
# =============================================================================

class TestModuleImport:
    def test_public_functions_callable(self):
        # Basic smoke-test: the importlib dance must expose all four public APIs.
        for fn in (set_label, set_crop, clear_crop, read_state, get_roll_angle):
            assert callable(fn)

    def test_label_colors_constant_exists(self):
        # LABEL_COLORS is part of the documented API surface.
        assert hasattr(ui_review, "LABEL_COLORS")
        assert "" in ui_review.LABEL_COLORS   # empty string clears the label


# =============================================================================
# set_label + read_state
# =============================================================================

class TestSetLabel:
    @requires_exiftool
    def test_set_green_and_read_back(self, tmp_path):
        # The canonical use-case: mark a JPEG green in the review UI.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        ok, _ = set_label(jpeg, "Green")
        assert ok
        state = read_state(jpeg)
        assert state["label"] == "Green"

    @requires_exiftool
    def test_clear_label_with_empty_string(self, tmp_path):
        # Setting color="" must remove a previously set label.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        set_label(jpeg, "Red")
        ok, _ = set_label(jpeg, "")
        assert ok
        state = read_state(jpeg)
        assert state["label"] is None

    @requires_exiftool
    def test_overwrite_label(self, tmp_path):
        # A second set_label call must replace the previous value.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        set_label(jpeg, "Red")
        set_label(jpeg, "Blue")
        state = read_state(jpeg)
        assert state["label"] == "Blue"

    def test_invalid_color_returns_false(self, tmp_path):
        # An unrecognised color string must be rejected without touching the file.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        ok, msg = set_label(jpeg, "Magenta")
        assert ok is False
        assert "invalid" in msg.lower() or "magenta" in msg.lower()

    @requires_exiftool
    def test_all_valid_colors_accepted(self, tmp_path):
        # Every color in LABEL_COLORS must be writable without error.
        for color in ui_review.LABEL_COLORS:
            jpeg = make_jpeg(tmp_path / f"photo_{color or 'clear'}.jpg")
            ok, msg = set_label(jpeg, color)
            assert ok, f"set_label({color!r}) failed: {msg}"


# =============================================================================
# set_crop + read_state
# =============================================================================

class TestSetCrop:
    @requires_exiftool
    def test_crop_roundtrip(self, tmp_path):
        # A crop rect written via set_crop must survive through read_state.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        ok, _ = set_crop(jpeg, 0.1, 0.2, 0.9, 0.8)
        assert ok
        state = read_state(jpeg)
        crop = state["crop"]
        assert crop is not None
        assert abs(crop["left"]   - 0.1) < 1e-4
        assert abs(crop["top"]    - 0.2) < 1e-4
        assert abs(crop["right"]  - 0.9) < 1e-4
        assert abs(crop["bottom"] - 0.8) < 1e-4

    @requires_exiftool
    def test_angle_stored_correctly(self, tmp_path):
        # Horizon correction angle must survive the XMP round-trip.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        set_crop(jpeg, 0.05, 0.05, 0.95, 0.95, angle=1.75)
        state = read_state(jpeg)
        assert abs(state["crop"]["angle"] - 1.75) < 1e-2

    @requires_exiftool
    def test_zero_angle_default(self, tmp_path):
        # Omitting the angle argument must store 0.0 (no rotation).
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        set_crop(jpeg, 0.1, 0.1, 0.9, 0.9)
        state = read_state(jpeg)
        assert abs(state["crop"]["angle"]) < 1e-4

    def test_left_ge_right_rejected(self, tmp_path):
        # A degenerate crop where left ≥ right must be rejected.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        ok, msg = set_crop(jpeg, 0.8, 0.1, 0.5, 0.9)
        assert ok is False

    def test_top_ge_bottom_rejected(self, tmp_path):
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        ok, msg = set_crop(jpeg, 0.1, 0.9, 0.9, 0.5)
        assert ok is False

    def test_left_out_of_range_rejected(self, tmp_path):
        # Values outside [0, 1] are not valid normalised coordinates.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        ok, msg = set_crop(jpeg, -0.1, 0.0, 0.9, 1.0)
        assert ok is False

    def test_right_out_of_range_rejected(self, tmp_path):
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        ok, msg = set_crop(jpeg, 0.0, 0.0, 1.1, 1.0)
        assert ok is False

    def test_top_out_of_range_rejected(self, tmp_path):
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        ok, msg = set_crop(jpeg, 0.0, -0.5, 1.0, 1.0)
        assert ok is False

    def test_bottom_out_of_range_rejected(self, tmp_path):
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        ok, msg = set_crop(jpeg, 0.0, 0.0, 1.0, 1.5)
        assert ok is False


# =============================================================================
# clear_crop
# =============================================================================

class TestClearCrop:
    @requires_exiftool
    def test_clear_removes_crop(self, tmp_path):
        # After clear_crop, read_state must report crop=None.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        set_crop(jpeg, 0.1, 0.1, 0.9, 0.9)
        ok, _ = clear_crop(jpeg)
        assert ok
        state = read_state(jpeg)
        assert state["crop"] is None

    @requires_exiftool
    def test_clear_preserves_label(self, tmp_path):
        # Clearing the crop must not disturb an existing color label.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        set_label(jpeg, "Yellow")
        set_crop(jpeg, 0.1, 0.1, 0.9, 0.9)
        clear_crop(jpeg)
        state = read_state(jpeg)
        assert state["label"] == "Yellow"


# =============================================================================
# read_state — fresh file with no XMP
# =============================================================================

class TestReadStateFreshFile:
    @requires_exiftool
    def test_fresh_jpeg_returns_none_label_and_crop(self, tmp_path):
        # A newly written JPEG with no XMP must return neutral state.
        jpeg = make_jpeg(tmp_path / "fresh.jpg")
        state = read_state(jpeg)
        assert state["label"] is None
        assert state["crop"] is None

    @requires_exiftool
    def test_read_state_returns_dict(self, tmp_path):
        # Return type must always be a dict with the expected keys.
        jpeg = make_jpeg(tmp_path / "fresh.jpg")
        state = read_state(jpeg)
        assert isinstance(state, dict)
        assert "label" in state
        assert "crop" in state


# =============================================================================
# get_roll_angle
# =============================================================================

class TestGetRollAngle:
    @requires_exiftool
    def test_no_roll_angle_tag_returns_zero(self, tmp_path):
        # A plain JPEG has no RollAngle EXIF tag — must default to 0.0 safely.
        jpeg = make_jpeg(tmp_path / "photo.jpg")
        angle = get_roll_angle(jpeg)
        assert isinstance(angle, float)
        assert angle == 0.0

    @requires_exiftool
    @requires_nef
    def test_real_nef_returns_float(self):
        # A real Z9 NEF must return a float (may be 0.0 if portrait or level).
        angle = get_roll_angle(SAMPLE_NEF)
        assert isinstance(angle, float)

    @requires_exiftool
    @requires_nef
    def test_portrait_orientation_returns_zero(self):
        # Portrait shots (RollAngle ≈ ±90°) must return 0.0 — let Lightroom handle them.
        # This test is informational: if the sample NEF is landscape it will also pass.
        angle = get_roll_angle(SAMPLE_NEF)
        # Whatever it is, the abs value must not be near 90 (if the function is
        # suppressing portrait angles correctly, we won't see ~90 returned).
        assert not (80.0 < abs(angle) < 100.0), \
            "get_roll_angle should suppress near-90° roll values (portrait)"

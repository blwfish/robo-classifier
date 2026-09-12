"""
Tests for image_utils.py — had zero coverage before this file (see
robo-classifier-20260912-a1f3#20 / full-review Phase 3.5). Focuses on
find_images, check_exiftool, and _extract_one_rawpy's max_preview_edge
handling — including a regression test for the max_edge=0 ZeroDivisionError
fixed in the same review (robo-classifier-20260912-a1f3#04).
"""

import io
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import image_utils
from image_utils import (
    default_workers, find_images, check_exiftool, _extract_one_rawpy,
    RAW_EXTENSIONS, JPG_EXTENSIONS,
)


# =============================================================================
# default_workers
# =============================================================================

def test_default_workers_at_least_two():
    assert default_workers() >= 2


# =============================================================================
# find_images
# =============================================================================

class TestFindImages:
    def test_non_recursive_finds_top_level_only(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        (tmp_path / "b.nef").write_bytes(b"x")
        (tmp_path / "c.txt").write_bytes(b"x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.jpg").write_bytes(b"x")

        jpgs, raws = find_images(tmp_path, recursive=False)
        assert [p.name for p in jpgs] == ["a.jpg"]
        assert [p.name for p in raws] == ["b.nef"]

    def test_recursive_includes_subdirectories(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "d.jpg").write_bytes(b"x")

        jpgs, _ = find_images(tmp_path, recursive=True)
        assert {p.name for p in jpgs} == {"a.jpg", "d.jpg"}

    def test_recursive_skips_dot_directories(self, tmp_path):
        (tmp_path / "a.jpg").write_bytes(b"x")
        hidden = tmp_path / ".trash"
        hidden.mkdir()
        (hidden / "junk.jpg").write_bytes(b"x")

        jpgs, _ = find_images(tmp_path, recursive=True)
        assert [p.name for p in jpgs] == ["a.jpg"]

    def test_extension_matching_is_case_insensitive(self, tmp_path):
        (tmp_path / "a.JPG").write_bytes(b"x")
        (tmp_path / "b.NEF").write_bytes(b"x")
        jpgs, raws = find_images(tmp_path)
        assert len(jpgs) == 1
        assert len(raws) == 1

    def test_results_sorted(self, tmp_path):
        (tmp_path / "b.jpg").write_bytes(b"x")
        (tmp_path / "a.jpg").write_bytes(b"x")
        jpgs, _ = find_images(tmp_path)
        assert [p.name for p in jpgs] == ["a.jpg", "b.jpg"]

    def test_empty_directory_returns_empty_lists(self, tmp_path):
        jpgs, raws = find_images(tmp_path)
        assert jpgs == []
        assert raws == []


# =============================================================================
# check_exiftool
# =============================================================================

class TestCheckExiftool:
    def test_returns_true_when_available(self, monkeypatch):
        monkeypatch.setattr(
            image_utils.subprocess, "run",
            lambda *a, **kw: MagicMock(returncode=0),
        )
        assert check_exiftool() is True

    def test_returns_false_when_not_found(self, monkeypatch):
        def raise_nf(*a, **kw):
            raise FileNotFoundError()
        monkeypatch.setattr(image_utils.subprocess, "run", raise_nf)
        assert check_exiftool() is False

    def test_returns_false_on_nonzero_exit(self, monkeypatch):
        def raise_cpe(*a, **kw):
            raise subprocess.CalledProcessError(1, ["exiftool"])
        monkeypatch.setattr(image_utils.subprocess, "run", raise_cpe)
        assert check_exiftool() is False


# =============================================================================
# _extract_one_rawpy — max_preview_edge handling
# =============================================================================

def _fake_jpeg_bytes(size=(800, 600)):
    img = Image.new("RGB", size, color=(120, 40, 200))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    return buf.getvalue()


class _FakeThumb:
    def __init__(self, data, fmt):
        self.data = data
        self.format = fmt


class _FakeRawpyContext:
    def __init__(self, thumb):
        self._thumb = thumb

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_thumb(self):
        return self._thumb


class TestExtractOneRawpyMaxEdge:
    """
    Regression coverage for robo-classifier-20260912-a1f3#04: passing
    max_edge=0 (CLI-documented as "disable downsampling") used to call
    Image.thumbnail((0, 0)), which raises ZeroDivisionError inside Pillow.
    The fix treats 0 the same as None: skip downsampling entirely.
    """

    def _run_with_mocked_rawpy(self, tmp_path, max_edge, image_size=(800, 600)):
        import rawpy as real_rawpy
        thumb = _FakeThumb(_fake_jpeg_bytes(image_size), real_rawpy.ThumbFormat.JPEG)
        raw_path = tmp_path / "IMG_0001.NEF"
        raw_path.write_bytes(b"not a real raw file")

        with patch("rawpy.imread", return_value=_FakeRawpyContext(thumb)):
            return _extract_one_rawpy(raw_path, tmp_path, max_edge=max_edge)

    def test_max_edge_zero_disables_downsampling_without_crashing(self, tmp_path):
        raw_path, preview_path = self._run_with_mocked_rawpy(tmp_path, max_edge=0)
        assert preview_path is not None
        with Image.open(preview_path) as img:
            assert img.size == (800, 600)  # unchanged — 0 means "disabled"

    def test_max_edge_none_disables_downsampling(self, tmp_path):
        raw_path, preview_path = self._run_with_mocked_rawpy(tmp_path, max_edge=None)
        assert preview_path is not None
        with Image.open(preview_path) as img:
            assert img.size == (800, 600)

    def test_max_edge_below_image_size_downsamples(self, tmp_path):
        raw_path, preview_path = self._run_with_mocked_rawpy(tmp_path, max_edge=400)
        assert preview_path is not None
        with Image.open(preview_path) as img:
            assert max(img.size) <= 400

    def test_max_edge_above_image_size_leaves_unchanged(self, tmp_path):
        # max(img.size) > max_edge is False when max_edge is larger than the
        # image — must not upscale.
        raw_path, preview_path = self._run_with_mocked_rawpy(
            tmp_path, max_edge=2048, image_size=(800, 600)
        )
        assert preview_path is not None
        with Image.open(preview_path) as img:
            assert img.size == (800, 600)

    def test_max_edge_exactly_at_image_size_not_downsampled(self, tmp_path):
        # Boundary: max(img.size) > max_edge is strict, so equal must not trigger.
        raw_path, preview_path = self._run_with_mocked_rawpy(
            tmp_path, max_edge=800, image_size=(800, 600)
        )
        assert preview_path is not None
        with Image.open(preview_path) as img:
            assert img.size == (800, 600)

    def test_non_jpeg_thumb_format_returns_none(self, tmp_path):
        import rawpy as real_rawpy
        thumb = _FakeThumb(b"fake bmp data", real_rawpy.ThumbFormat.BITMAP)
        raw_path = tmp_path / "IMG_0002.NEF"
        raw_path.write_bytes(b"not a real raw file")
        with patch("rawpy.imread", return_value=_FakeRawpyContext(thumb)):
            result_path, preview_path = _extract_one_rawpy(raw_path, tmp_path, max_edge=512)
        assert preview_path is None

    def test_rawpy_exception_returns_none_not_raises(self, tmp_path):
        raw_path = tmp_path / "IMG_0003.NEF"
        raw_path.write_bytes(b"not a real raw file")
        with patch("rawpy.imread", side_effect=OSError("corrupt file")):
            result_path, preview_path = _extract_one_rawpy(raw_path, tmp_path, max_edge=512)
        assert preview_path is None
        assert result_path == raw_path

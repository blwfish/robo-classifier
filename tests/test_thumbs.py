"""
Tests for ui/thumbs.py — had zero test coverage before this file
(robo-classifier-20260912-a1f3#20 / full-review Phase 3.5). Focuses on
get_thumb's error-handling consistency fix and pregenerate_thumbs' failure
visibility fix (P2-50 / P2-48).
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from ui.thumbs import get_thumb, pregenerate_thumbs, _cache_path


def _make_jpeg(path: Path, size=(200, 150)):
    Image.new("RGB", size, color=(10, 20, 30)).save(path, "JPEG")
    return path


class TestGetThumb:
    def test_missing_source_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            get_thumb(tmp_path, tmp_path / "nope.jpg")

    def test_valid_jpeg_produces_thumb(self, tmp_path):
        src = _make_jpeg(tmp_path / "a.jpg")
        out = get_thumb(tmp_path, src, size=64)
        assert out.exists()
        with Image.open(out) as img:
            assert max(img.size) <= 64

    def test_corrupt_source_raises_runtime_error_not_arbitrary_exception(self, tmp_path):
        # Regression test for robo-classifier-20260912-a1f3, P2-50: the
        # decode step used to be unguarded, so a corrupt file raised
        # whatever PIL exception happened to occur (e.g.
        # UnidentifiedImageError) instead of a consistent, diagnosable
        # RuntimeError matching the "no embedded preview" error style.
        src = tmp_path / "corrupt.jpg"
        src.write_bytes(b"not a real jpeg")
        with pytest.raises(RuntimeError, match="Could not create thumbnail"):
            get_thumb(tmp_path, src)

    def test_cached_thumb_reused_when_fresh(self, tmp_path):
        src = _make_jpeg(tmp_path / "a.jpg")
        first = get_thumb(tmp_path, src, size=64)
        first_mtime = first.stat().st_mtime
        second = get_thumb(tmp_path, src, size=64)
        assert second == first
        assert second.stat().st_mtime == first_mtime


class TestPregenerateThumbsFailureVisibility:
    def test_mixed_success_and_failure_all_get_processed(self, tmp_path, capsys):
        # Regression test for robo-classifier-20260912-a1f3, P2-48: the
        # batch progress counter used to increment on every attempt
        # regardless of success, so "Pregenerated N/N" didn't distinguish
        # successes from failures. This test pins that every source is at
        # least attempted and the good one produces a real thumbnail; a
        # failure must not crash or block the batch.
        good = _make_jpeg(tmp_path / "good.jpg")
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not a real jpeg")

        pregenerate_thumbs(tmp_path, [good, bad], workers=1, size=64)

        good_out = _cache_path(tmp_path, good, 64)
        bad_out = _cache_path(tmp_path, bad, 64)
        assert good_out.exists()
        assert not bad_out.exists()  # failed silently but didn't abort the batch

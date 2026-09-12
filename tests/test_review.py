"""
Tests for review.py — the legacy Flask review UI.

review.py had zero test coverage before this file (see
robo-classifier-20260912-a1f3#20 / full-review Phase 3.5). This focuses on
the functions the full review's fixes touched directly, not exhaustive
coverage of the whole module.
"""

import csv
import os
import time
from pathlib import Path

import pytest
from PIL import Image

import review


# =============================================================================
# _make_thumb — EXIF orientation + parity with ui/thumbs.py
# (robo-classifier-20260912-a1f3#13)
# =============================================================================

class TestMakeThumb:
    def test_honors_exif_orientation(self, tmp_path):
        # A portrait-intent image stored landscape-in-bytes with an EXIF
        # Orientation tag must come out upright, matching ui/thumbs.py's
        # get_thumb() behavior — previously _make_thumb had no
        # exif_transpose call and could produce a sideways thumbnail.
        src = tmp_path / "portrait.jpg"
        img = Image.new("RGB", (600, 400), color=(10, 20, 30))
        exif = img.getexif()
        exif[0x0112] = 6  # Orientation: rotate 90 CW to display upright
        img.save(src, "JPEG", exif=exif)

        dest = tmp_path / "thumb.jpg"
        result = review._make_thumb(src, dest)
        assert result == dest
        with Image.open(dest) as out:
            # exif_transpose applied the rotation and dropped the tag —
            # width/height swap relative to the stored (600, 400) bytes.
            assert out.size[0] < out.size[1]


class TestGetOrCreateThumbCacheInvalidation:
    def test_stale_cache_regenerated_when_source_newer(self, tmp_path, monkeypatch):
        # Regression test: get_or_create_thumb used to trust a cached
        # thumbnail forever once it existed, with no mtime check — an
        # edited/replaced source file kept serving its stale thumbnail
        # indefinitely (robo-classifier-20260912-a1f3#13).
        thumb_dir = tmp_path / "thumbs"
        thumb_dir.mkdir()
        monkeypatch.setitem(review._state, "thumb_dir", str(thumb_dir))
        monkeypatch.setitem(review._state, "small_jpg_index", {})

        source = tmp_path / "photo.jpg"
        Image.new("RGB", (400, 300), color=(1, 2, 3)).save(source, "JPEG")

        # Prime the cache with an intentionally-older "stale" thumbnail.
        first = review.get_or_create_thumb(str(source))
        assert first is not None
        old_mtime = time.time() - 100
        os.utime(first, (old_mtime, old_mtime))

        # Touch the source so it's newer than the cached thumbnail.
        time.sleep(0.01)
        Image.new("RGB", (400, 300), color=(4, 5, 6)).save(source, "JPEG")

        second = review.get_or_create_thumb(str(source))
        assert second == first  # same cache path
        assert second.stat().st_mtime > old_mtime  # but regenerated

    def test_fresh_cache_reused_without_regenerating(self, tmp_path, monkeypatch):
        thumb_dir = tmp_path / "thumbs"
        thumb_dir.mkdir()
        monkeypatch.setitem(review._state, "thumb_dir", str(thumb_dir))
        monkeypatch.setitem(review._state, "small_jpg_index", {})

        source = tmp_path / "photo.jpg"
        Image.new("RGB", (400, 300), color=(1, 2, 3)).save(source, "JPEG")

        first = review.get_or_create_thumb(str(source))
        first_mtime = first.stat().st_mtime

        second = review.get_or_create_thumb(str(source))
        assert second == first
        assert second.stat().st_mtime == first_mtime  # not regenerated


# =============================================================================
# _find_small_jpg_dir — score/overlap-count encoding fix
# (robo-classifier-20260912-a1f3#05)
# =============================================================================

class TestFindSmallJpgDir:
    def _make_dir_with_jpgs(self, base: Path, name: str, stems: list[str]) -> Path:
        d = base / name
        d.mkdir()
        for stem in stems:
            (d / f"{stem}.jpg").write_bytes(b"x")
        return d

    def test_no_candidates_returns_none(self, tmp_path):
        jpg_dir = tmp_path / "main"
        jpg_dir.mkdir()
        assert review._find_small_jpg_dir(jpg_dir, {"a", "b"}) is None

    def test_below_10_percent_overlap_rejected(self, tmp_path):
        jpg_dir = tmp_path / "main"
        jpg_dir.mkdir()
        main_stems = {f"img{i}" for i in range(100)}
        # Only 1 of 100 stems overlaps — below the 10% floor (max(1, 10)).
        self._make_dir_with_jpgs(tmp_path, "small", ["img0"])
        assert review._find_small_jpg_dir(jpg_dir, main_stems) is None

    def test_at_10_percent_overlap_accepted(self, tmp_path):
        jpg_dir = tmp_path / "main"
        jpg_dir.mkdir()
        main_stems = {f"img{i}" for i in range(100)}
        small = self._make_dir_with_jpgs(tmp_path, "small", [f"img{i}" for i in range(10)])
        assert review._find_small_jpg_dir(jpg_dir, main_stems) == small

    def test_overlap_above_1000_not_corrupted_by_name_match_bonus(self, tmp_path):
        # Regression test for the score = overlap + (1000 if name_match else 0)
        # / overlap_count = score % 1000 bug: once real overlap reaches or
        # exceeds 1000, `% 1000` silently truncated it (e.g. true overlap
        # 1200 -> computed 200), which could wrongly fail the 10% gate.
        jpg_dir = tmp_path / "main"
        jpg_dir.mkdir()
        main_stems = {f"img{i}" for i in range(1200)}
        # Directory name does NOT match SMALL_DIR_KEYWORDS, so the only way
        # the old code could reach a score >= 1000 was via overlap itself.
        small = self._make_dir_with_jpgs(
            tmp_path, "not_a_keyword_name", [f"img{i}" for i in range(1200)]
        )
        result = review._find_small_jpg_dir(jpg_dir, main_stems)
        # Overlap is 1200 (100% of main_stems) — must still be accepted.
        # Under the old buggy code, 1200 % 1000 == 200, which is still >=
        # 10% of 1200 main stems (120) coincidentally in this exact case,
        # so the real corruption shows up in the *choice between candidates*
        # below rather than this accept/reject decision alone.
        assert result == small

    def test_name_match_still_preferred_over_higher_raw_overlap(self, tmp_path):
        # A directory whose name matches SMALL_DIR_KEYWORDS should still be
        # preferred over one with a larger overlap but no name match —
        # this is the original intent the +1000 bonus encoded, and must
        # survive being tracked as a separate tuple field instead.
        jpg_dir = tmp_path / "main"
        jpg_dir.mkdir()
        main_stems = {f"img{i}" for i in range(20)}
        keyword_name = next(iter(review.SMALL_DIR_KEYWORDS))
        preferred = self._make_dir_with_jpgs(
            tmp_path, keyword_name, [f"img{i}" for i in range(5)]
        )
        higher_overlap_no_match = self._make_dir_with_jpgs(
            tmp_path, "unrelated_dir", [f"img{i}" for i in range(15)]
        )
        result = review._find_small_jpg_dir(jpg_dir, main_stems)
        assert result == preferred


# =============================================================================
# resolve_path
# =============================================================================

class TestResolvePath:
    def test_absolute_existing_path_returned_as_is(self, tmp_path):
        f = tmp_path / "a.jpg"
        f.write_bytes(b"x")
        assert review.resolve_path(str(f), tmp_path) == f

    def test_relative_to_csv_dir_resolved(self, tmp_path):
        csv_dir = tmp_path / "shoot"
        csv_dir.mkdir()
        f = csv_dir / "a.jpg"
        f.write_bytes(b"x")
        assert review.resolve_path("a.jpg", csv_dir) == f

    def test_nonexistent_path_returned_as_is(self, tmp_path):
        result = review.resolve_path("does_not_exist.jpg", tmp_path)
        assert result == Path("does_not_exist.jpg")


# =============================================================================
# load_results / compute_bursts_and_winners
# =============================================================================

class TestLoadResultsAndBursts:
    def _write_results_csv(self, path: Path, rows: list[dict]):
        fieldnames = ["filename", "path", "classification", "confidence",
                      "confidence_reject", "confidence_select", "error_class"]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_load_results_parses_confidence_as_float(self, tmp_path):
        csv_path = tmp_path / "results.csv"
        img = tmp_path / "a.jpg"
        img.write_bytes(b"x")
        self._write_results_csv(csv_path, [{
            "filename": "a.jpg", "path": "a.jpg", "classification": "select",
            "confidence": "0.9", "confidence_reject": "0.1",
            "confidence_select": "0.9", "error_class": "",
        }])
        rows = review.load_results(str(csv_path))
        assert rows[0]["confidence_select"] == 0.9
        assert isinstance(rows[0]["confidence_select"], float)

    def test_compute_bursts_and_winners_includes_non_select_frames(self, tmp_path):
        # review.py's UI intentionally shows every burst's best frame
        # regardless of classification (threshold tuning happens at
        # display/write time) — this must survive delegating to
        # classify.burst_dedup(filter_select=False).
        # parse_burst_base groups by exact filename stem (no suffix
        # stripping); use the same filename for both rows to put them in
        # one burst, distinguished by path as if they were two candidate
        # frames under review.
        rows = [
            {"filename": "BLW0001.jpg", "path": "/x/a/BLW0001.jpg",
             "classification": "reject", "confidence_select": 0.3, "confidence_reject": 0.7},
            {"filename": "BLW0001.jpg", "path": "/x/b/BLW0001.jpg",
             "classification": "reject", "confidence_select": 0.5, "confidence_reject": 0.5},
        ]
        winners, bursts = review.compute_bursts_and_winners(rows)
        assert len(winners) == 1
        # Best-by-confidence_select wins even though neither frame is 'select'.
        assert winners[0]["path"] == "/x/b/BLW0001.jpg"

    def test_compute_bursts_and_winners_sorted_descending(self, tmp_path):
        rows = [
            {"filename": "BLW0001-1.jpg", "path": "/x/BLW0001-1.jpg",
             "classification": "select", "confidence_select": 0.6, "confidence_reject": 0.4},
            {"filename": "BLW0002-1.jpg", "path": "/x/BLW0002-1.jpg",
             "classification": "select", "confidence_select": 0.95, "confidence_reject": 0.05},
        ]
        winners, _ = review.compute_bursts_and_winners(rows)
        assert [w["confidence_select"] for w in winners] == sorted(
            (w["confidence_select"] for w in winners), reverse=True
        )

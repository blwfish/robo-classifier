"""Tests for ingest.py — camera model aliases, EXIF parsing, filename building,
file hashing, directory scanning, and the SQLite manifest.

The end-to-end ingest() function requires exiftool + real files and is not
tested here. These tests cover every pure function and the SQLite helpers,
which together determine correctness of the rename pipeline and dedup logic.
"""

import hashlib
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ingest import (
    model_alias,
    _parse_dt,
    build_dest_name,
    _file_hash,
    scan_source,
    _open_manifest,
    _record,
    _known_hash,
)


# =============================================================================
# model_alias — camera model → filename slug
# =============================================================================

class TestModelAlias:
    def test_nikon_z9(self):
        # The Z9 is the primary shooting body — its alias must be exactly "Z9".
        assert model_alias("NIKON Z 9") == "Z9"

    def test_dji_air3_fc8284(self):
        # DJI bodies use FC part numbers in EXIF, not marketing names.
        assert model_alias("FC8284") == "Air3"

    def test_canon_r5(self):
        assert model_alias("CANON EOS R5") == "R5"

    def test_case_insensitive_lookup(self):
        # EXIF model strings from different cameras vary in capitalisation.
        assert model_alias("nikon z 9") == "Z9"
        assert model_alias("Nikon Z 9") == "Z9"

    def test_unknown_model_falls_back_to_slug(self):
        # An unrecognised model must produce a usable filename, not raise.
        alias = model_alias("FUJIFILM X-T5")
        assert alias  # non-empty
        assert " " not in alias  # spaces are unsafe in filenames

    def test_empty_string_returns_unknown(self):
        # An empty EXIF model field (common for video files) must not crash.
        assert model_alias("") == "Unknown"

    def test_slug_strips_spaces(self):
        # Spaces in the fallback slug would break filenames.
        alias = model_alias("Some Unknown Camera")
        assert " " not in alias

    def test_slug_strips_unsafe_characters(self):
        # Characters like / : * ? are forbidden in most filesystems.
        alias = model_alias("Brand/Model:X*1")
        for ch in ' /\\:*?"<>|':
            assert ch not in alias


# =============================================================================
# _parse_dt — EXIF datetime string → compact timestamp
# =============================================================================

class TestParseDt:
    def test_standard_exif_format(self):
        # exiftool returns "YYYY:MM:DD HH:MM:SS" — must become 14-digit string.
        assert _parse_dt("2026:02:28 10:37:58") == "20260228103758"

    def test_strips_positive_timezone_exif_format(self):
        # DJI and some mirrorless bodies append "+00:00" to EXIF-format strings.
        assert _parse_dt("2026:02:28 10:37:58+00:00") == "20260228103758"

    def test_strips_negative_timezone_exif_format(self):
        # Negative UTC offset on an EXIF-format string.
        assert _parse_dt("2026:02:28 10:37:58-05:00") == "20260228103758"

    def test_iso_format_with_positive_timezone(self):
        # DJI Air 3 uses ISO 8601 with dashes + "T" separator and a "+" offset.
        # The old split("+")[0].split("-")[0] logic ate the date dashes, returning None.
        assert _parse_dt("2026-02-28T10:37:58+05:30") == "20260228103758"

    def test_iso_format_with_negative_timezone(self):
        assert _parse_dt("2026-02-28T10:37:58-05:00") == "20260228103758"

    def test_iso_format_no_timezone(self):
        assert _parse_dt("2026-02-28T10:37:58") == "20260228103758"

    def test_empty_string_returns_none(self):
        # exiftool returns "" when no DateTimeOriginal tag exists.
        assert _parse_dt("") is None

    def test_none_returns_none(self):
        assert _parse_dt(None) is None

    def test_garbage_string_returns_none(self):
        # Reject strings that don't contain enough digits to form a timestamp.
        assert _parse_dt("not-a-date") is None

    def test_fewer_than_14_digits_returns_none(self):
        # A truncated timestamp (e.g. date only) must not produce a partial result.
        assert _parse_dt("2026:02:28") is None

    def test_result_is_exactly_14_chars(self):
        result = _parse_dt("2026:02:28 10:37:58")
        assert len(result) == 14

    def test_result_is_all_digits(self):
        result = _parse_dt("2026:02:28 10:37:58")
        assert result.isdigit()


# =============================================================================
# build_dest_name — assemble renamed filename
# =============================================================================

class TestBuildDestName:
    def test_standard_rename(self):
        # Normal case: all three fields present and valid.
        name = build_dest_name("20260228103758", "NIKON Z 9", "BLW4730.NEF")
        assert name == "20260228103758-Z9_BLW4730.NEF"

    def test_none_datetime_gives_zero_prefix(self):
        # Files with no EXIF date must still get a valid (if sorted-to-front) name.
        name = build_dest_name(None, "NIKON Z 9", "BLW4730.NEF")
        assert name.startswith("00000000000000-")

    def test_empty_model_gives_unknown_alias(self):
        # Video files often have no CameraModelName — must degrade gracefully.
        name = build_dest_name("20260228103758", "", "clip.mp4")
        assert "Unknown" in name

    def test_original_filename_preserved(self):
        # The original filename must appear verbatim after the alias.
        name = build_dest_name("20260228103758", "NIKON Z 9", "IMG_0001.JPG")
        assert name.endswith("_IMG_0001.JPG")

    def test_format_structure(self):
        # The separator between timestamp, alias, and original name must be exact
        # so that downstream tools can reliably split the parts.
        name = build_dest_name("20260228103758", "NIKON Z 9", "BLW4730.NEF")
        ts, rest = name.split("-", 1)
        alias, orig = rest.split("_", 1)
        assert ts == "20260228103758"
        assert alias == "Z9"
        assert orig == "BLW4730.NEF"


# =============================================================================
# _file_hash — SHA-256 content fingerprint
# =============================================================================

class TestFileHash:
    def test_produces_64_char_hex(self, tmp_path):
        # SHA-256 hex digest is always 64 characters.
        f = tmp_path / "test.bin"
        f.write_bytes(b"hello world")
        result = _file_hash(f)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_content_same_hash(self, tmp_path):
        # Two files with identical bytes must produce the same hash (dedup relies on this).
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"identical content")
        b.write_bytes(b"identical content")
        assert _file_hash(a) == _file_hash(b)

    def test_different_content_different_hash(self, tmp_path):
        # Files with different content must not collide (correctness of dedup).
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"content A")
        b.write_bytes(b"content B")
        assert _file_hash(a) != _file_hash(b)

    def test_matches_hashlib_directly(self, tmp_path):
        # Cross-check against hashlib to ensure the algorithm isn't accidentally changed.
        f = tmp_path / "check.bin"
        data = b"verification payload"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _file_hash(f) == expected


# =============================================================================
# scan_source — directory tree enumeration
# =============================================================================

class TestScanSource:
    def _make_file(self, path: Path, content: bytes = b"x") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_flat_dir_returns_media_files(self, tmp_path):
        # A plain local folder with no DCIM/ must return all supported extensions.
        self._make_file(tmp_path / "a.jpg")
        self._make_file(tmp_path / "b.nef")
        results = scan_source(tmp_path)
        exts = {r["ext"] for r in results}
        assert ".jpg" in exts
        assert ".nef" in exts

    def test_dcim_subdir_scoped(self, tmp_path):
        # A card with DCIM/ must only index files under DCIM/ — not firmware dirs.
        self._make_file(tmp_path / "DCIM" / "100NCD90" / "shot.nef")
        self._make_file(tmp_path / "MISC" / "noise.dat")
        results = scan_source(tmp_path)
        paths = [r["path"] for r in results]
        assert any("DCIM" in p for p in paths)
        assert not any("MISC" in p for p in paths)

    def test_hidden_dirs_skipped(self, tmp_path):
        # macOS cards contain .Spotlight-V100 etc. — those must be invisible.
        self._make_file(tmp_path / ".hidden_dir" / "secret.jpg")
        self._make_file(tmp_path / "visible.jpg")
        results = scan_source(tmp_path)
        paths = [r["path"] for r in results]
        assert not any(".hidden_dir" in p for p in paths)
        assert any("visible.jpg" in p for p in paths)

    def test_non_media_extensions_ignored(self, tmp_path):
        # Non-photo/video files (thumbnails, logs) must be excluded.
        self._make_file(tmp_path / "thumb.thm")
        self._make_file(tmp_path / "data.xml")
        self._make_file(tmp_path / "real.jpg")
        results = scan_source(tmp_path)
        exts = {r["ext"] for r in results}
        assert ".thm" not in exts
        assert ".xml" not in exts
        assert ".jpg" in exts

    def test_result_includes_path_ext_size(self, tmp_path):
        # Each result dict must carry the three fields ingest() reads.
        self._make_file(tmp_path / "photo.jpg", content=b"data" * 100)
        results = scan_source(tmp_path)
        assert len(results) == 1
        r = results[0]
        assert "path" in r
        assert "ext" in r
        assert "size" in r
        assert r["ext"] == ".jpg"
        assert r["size"] > 0

    def test_sorted_by_path(self, tmp_path):
        # Stable sort ensures deterministic processing order across runs.
        self._make_file(tmp_path / "zzz.jpg")
        self._make_file(tmp_path / "aaa.jpg")
        results = scan_source(tmp_path)
        paths = [r["path"] for r in results]
        assert paths == sorted(paths)

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert scan_source(tmp_path) == []

    def test_video_extensions_included(self, tmp_path):
        # Video files from drone passes must be ingested alongside stills.
        self._make_file(tmp_path / "clip.mp4")
        self._make_file(tmp_path / "clip.mov")
        results = scan_source(tmp_path)
        exts = {r["ext"] for r in results}
        assert ".mp4" in exts
        assert ".mov" in exts


# =============================================================================
# _open_manifest / _record / _known_hash — SQLite dedup manifest
# =============================================================================

class TestManifest:
    def test_open_creates_table(self, tmp_path):
        # _open_manifest must initialise the schema so subsequent calls don't fail.
        db_path = tmp_path / "ingest.db"
        conn = _open_manifest(db_path)
        # Verify the table exists by querying it.
        conn.execute("SELECT count(*) FROM ingested")
        conn.close()

    def test_unknown_hash_returns_none(self, tmp_path):
        # A hash that was never recorded must return None, not raise.
        conn = _open_manifest(tmp_path / "ingest.db")
        result = _known_hash(conn, "deadbeef" * 8)
        conn.close()
        assert result is None

    def test_record_and_lookup(self, tmp_path):
        # After recording a hash, _known_hash must return the associated dest_path.
        conn = _open_manifest(tmp_path / "ingest.db")
        h = "a" * 64
        _record(conn, h, "/dest/file.nef", "Z9", "BLW0001.NEF", "CardA")
        result = _known_hash(conn, h)
        conn.close()
        assert result == "/dest/file.nef"

    def test_duplicate_record_does_not_raise(self, tmp_path):
        # INSERT OR REPLACE semantics: re-recording a known hash must not crash.
        conn = _open_manifest(tmp_path / "ingest.db")
        h = "b" * 64
        _record(conn, h, "/dest/a.nef", "Z9", "a.NEF", "CardA")
        _record(conn, h, "/dest/b.nef", "Z9", "a.NEF", "CardA")  # must not raise
        conn.close()

    def test_duplicate_record_updates_dest(self, tmp_path):
        # After a re-record the lookup should return the newer dest_path.
        conn = _open_manifest(tmp_path / "ingest.db")
        h = "c" * 64
        _record(conn, h, "/dest/old.nef", "Z9", "old.NEF", "CardA")
        _record(conn, h, "/dest/new.nef", "Z9", "old.NEF", "CardA")
        result = _known_hash(conn, h)
        conn.close()
        assert result == "/dest/new.nef"

    def test_separate_hashes_independent(self, tmp_path):
        # Two distinct files must not share manifest entries.
        conn = _open_manifest(tmp_path / "ingest.db")
        _record(conn, "1" * 64, "/dest/one.nef", "Z9", "one.NEF", "CardA")
        _record(conn, "2" * 64, "/dest/two.nef", "Z9", "two.NEF", "CardA")
        assert _known_hash(conn, "1" * 64) == "/dest/one.nef"
        assert _known_hash(conn, "2" * 64) == "/dest/two.nef"
        conn.close()

    def test_open_manifest_creates_parent_dirs(self, tmp_path):
        # The manifest directory might not exist yet on first run.
        nested = tmp_path / "a" / "b" / "c" / "ingest.db"
        conn = _open_manifest(nested)
        conn.close()
        assert nested.exists()

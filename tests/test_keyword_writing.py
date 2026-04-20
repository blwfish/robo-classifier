"""Tests for keyword writing in classify.py.

write_keywords() is the orchestration function that decides which files get
what keyword. Bugs here either:
- Miss tagging qualifying winners (robo_9x not written → no smart collection hit)
- Over-tag with 'select' (floods Lightroom with false positives)

We mock write_keyword_to_file so no exiftool is actually invoked.
"""

import subprocess
from unittest.mock import patch, MagicMock

import pytest

import classify


def _row(path, confidence, classification="select"):
    return {
        "path": path,
        "filename": path.split("/")[-1],
        "confidence_select": confidence,
        "classification": classification,
    }


# =============================================================================
# write_keywords — orchestration logic
# =============================================================================

class TestWriteKeywords:
    def test_winner_above_threshold_gets_robo_keyword(self):
        winners = [_row("/x/a.jpg", 0.95)]
        bursts  = {"b0": [_row("/x/a.jpg", 0.95)]}
        writes = []
        with patch.object(classify, "write_keyword_to_file",
                          side_effect=lambda p, kw, nef_dir=None: writes.append((p, kw)) or True):
            tiers, winner_written, select_written, errors = \
                classify.write_keywords(winners, bursts)
        # /x/a.jpg tagged with robo_95 (pass 1) and 'select' (pass 2)
        assert ("/x/a.jpg", "robo_95") in writes
        assert ("/x/a.jpg", "select") in writes
        assert winner_written == 1
        assert select_written == 1
        assert errors == 0
        assert tiers["robo_95"] == 1

    def test_winner_below_threshold_no_write(self):
        winners = [_row("/x/a.jpg", 0.85)]
        bursts  = {"b0": [_row("/x/a.jpg", 0.85)]}
        writes = []
        with patch.object(classify, "write_keyword_to_file",
                          side_effect=lambda p, kw, nef_dir=None: writes.append((p, kw)) or True):
            tiers, winner_written, select_written, errors = \
                classify.write_keywords(winners, bursts)
        assert writes == []
        assert winner_written == 0
        assert select_written == 0
        assert tiers["below_threshold"] == 1

    def test_non_winning_burst_members_get_select_only_if_winner_qualifies(self):
        """A burst with ONE qualifying winner should tag its SIBLINGS with
        'select', not the winner itself (which gets robo_9x). A burst without
        a qualifying winner gets no 'select' tags on any frame."""
        # burst A: winner 0.95 (qualifies), sibling 0.80
        # burst B: best 0.85 (below threshold), sibling 0.70
        winners = [
            _row("/x/a_best.jpg", 0.95),
            _row("/x/b_best.jpg", 0.85),
        ]
        bursts = {
            "A": [_row("/x/a_best.jpg", 0.95), _row("/x/a_sib.jpg", 0.80)],
            "B": [_row("/x/b_best.jpg", 0.85), _row("/x/b_sib.jpg", 0.70)],
        }
        writes = []
        with patch.object(classify, "write_keyword_to_file",
                          side_effect=lambda p, kw, nef_dir=None: writes.append((p, kw)) or True):
            classify.write_keywords(winners, bursts)

        # Burst A: winner gets robo_95, both frames (incl. winner) get select
        assert ("/x/a_best.jpg", "robo_95") in writes
        assert ("/x/a_best.jpg", "select")  in writes
        assert ("/x/a_sib.jpg",  "select")  in writes
        # Burst B: nothing written
        assert not any(p.startswith("/x/b_") for p, _ in writes)

    def test_exiftool_failure_counted_as_error(self):
        winners = [_row("/x/a.jpg", 0.95)]
        bursts  = {"b0": [_row("/x/a.jpg", 0.95)]}
        with patch.object(classify, "write_keyword_to_file", return_value=False):
            tiers, ww, sw, errors = classify.write_keywords(winners, bursts)
        # One error for the winner + one for the select sweep
        assert errors == 2
        assert ww == 0
        assert sw == 0

    def test_mixed_tier_counts(self):
        winners = [
            _row("/x/a.jpg", 0.91),   # robo_91
            _row("/x/b.jpg", 0.97),   # robo_97
            _row("/x/c.jpg", 0.95),   # robo_95
            _row("/x/d.jpg", 0.80),   # below threshold
        ]
        bursts = {fn.strip("/x/").split(".")[0]: [_row(fn, c)]
                  for fn, c in [("/x/a.jpg", 0.91), ("/x/b.jpg", 0.97),
                                ("/x/c.jpg", 0.95), ("/x/d.jpg", 0.80)]}
        with patch.object(classify, "write_keyword_to_file", return_value=True):
            tiers, _, _, _ = classify.write_keywords(winners, bursts)
        assert tiers["robo_91"] == 1
        assert tiers["robo_95"] == 1
        assert tiers["robo_97"] == 1
        assert tiers["below_threshold"] == 1
        # Unoccupied tiers stay at 0
        assert tiers["robo_99"] == 0

    def test_nef_dir_propagated(self):
        """When nef_dir is provided, it must flow through to
        write_keyword_to_file so XMP sidecars land in the NEF folder."""
        winners = [_row("/jpg/a.jpg", 0.95)]
        bursts = {"b0": [_row("/jpg/a.jpg", 0.95)]}
        captured = []
        with patch.object(classify, "write_keyword_to_file",
                          side_effect=lambda p, kw, nef_dir=None:
                              captured.append(nef_dir) or True):
            classify.write_keywords(winners, bursts, nef_dir="/raw/")
        assert all(d == "/raw/" for d in captured)


# =============================================================================
# embed_keyword_in_jpeg — exiftool command construction
# =============================================================================

class TestEmbedKeywordInJpeg:
    def test_robo_keyword_uses_hierarchical_AI_keywords_path(self, monkeypatch):
        captured = []
        def fake_run(args, **kw):
            captured.append(args)
            m = MagicMock()
            m.returncode = 0
            return m
        monkeypatch.setattr(classify.subprocess, "run", fake_run)
        assert classify.embed_keyword_in_jpeg("/x/a.jpg", "robo_95") is True
        args = captured[0]
        # Hierarchy: "AI keywords|robo|robo_95"
        assert any("AI keywords|robo|robo_95" in a for a in args)
        assert "-overwrite_original" in args
        assert str("/x/a.jpg") in args

    def test_select_keyword_uses_shorter_hierarchy(self, monkeypatch):
        captured = []
        monkeypatch.setattr(classify.subprocess, "run",
                            lambda args, **kw: captured.append(args) or
                            MagicMock(returncode=0))
        classify.embed_keyword_in_jpeg("/x/a.jpg", "select")
        # "AI keywords|select" — no robo segment in between
        assert any("AI keywords|select" in a for a in captured[0])
        assert not any("robo" in a for a in captured[0])

    def test_exiftool_missing_returns_false(self, monkeypatch, capsys):
        def raise_notfound(*a, **kw):
            raise FileNotFoundError("exiftool")
        monkeypatch.setattr(classify.subprocess, "run", raise_notfound)
        assert classify.embed_keyword_in_jpeg("/x/a.jpg", "select") is False
        # Should print an install hint so the user knows the fix
        out = capsys.readouterr().out
        assert "exiftool" in out

    def test_nonzero_exit_returns_false(self, monkeypatch):
        monkeypatch.setattr(classify.subprocess, "run",
                            lambda *a, **kw: MagicMock(returncode=1))
        assert classify.embed_keyword_in_jpeg("/x/a.jpg", "select") is False


# =============================================================================
# write_xmp_sidecar — exiftool command construction
# =============================================================================

class TestWriteXmpSidecar:
    def test_new_sidecar_created_from_raw(self, tmp_path, monkeypatch):
        # No existing XMP → should build command with -tagsfromfile + -o
        raw = tmp_path / "x.NEF"
        raw.write_bytes(b"")   # existence only; contents don't matter here
        captured = []
        monkeypatch.setattr(classify.subprocess, "run",
                            lambda args, **kw: captured.append(args) or
                            MagicMock(returncode=0))
        assert classify.write_xmp_sidecar(raw, "robo_95") is True
        args = captured[0]
        assert "-tagsfromfile" in args
        assert str(raw) in args
        # Output path points at the sidecar
        assert any(str(raw.with_suffix(".xmp")) in a for a in args)
        # Hierarchical subject uses robo| prefix
        assert any("-HierarchicalSubject+=robo|robo_95" in a for a in args)

    def test_existing_sidecar_updated_not_recreated(self, tmp_path, monkeypatch):
        raw = tmp_path / "x.NEF"
        raw.write_bytes(b"")
        sidecar = raw.with_suffix(".xmp")
        sidecar.write_text("<xmp/>")
        captured = []
        monkeypatch.setattr(classify.subprocess, "run",
                            lambda args, **kw: captured.append(args) or
                            MagicMock(returncode=0))
        classify.write_xmp_sidecar(raw, "robo_95")
        # Should NOT use -tagsfromfile when sidecar already exists
        assert "-tagsfromfile" not in captured[0]
        # Should use -overwrite_original on the sidecar directly
        assert "-overwrite_original" in captured[0]

    def test_exiftool_missing_returns_false(self, tmp_path, monkeypatch):
        raw = tmp_path / "x.NEF"
        raw.write_bytes(b"")
        def raise_nf(*a, **kw): raise FileNotFoundError("exiftool")
        monkeypatch.setattr(classify.subprocess, "run", raise_nf)
        assert classify.write_xmp_sidecar(raw, "select") is False


# =============================================================================
# write_keyword_to_file — routing between JPEG and RAW paths
# =============================================================================

class TestWriteKeywordRouting:
    def test_jpeg_routes_to_embed(self, monkeypatch):
        called = []
        monkeypatch.setattr(classify, "embed_keyword_in_jpeg",
                            lambda p, kw: called.append(("jpeg", p, kw)) or True)
        monkeypatch.setattr(classify, "write_xmp_sidecar",
                            lambda p, kw: called.append(("xmp", p, kw)) or True)
        assert classify.write_keyword_to_file("/x/a.jpg", "select") is True
        assert called[0][0] == "jpeg"

    def test_raw_routes_to_xmp(self, monkeypatch):
        called = []
        monkeypatch.setattr(classify, "embed_keyword_in_jpeg",
                            lambda p, kw: called.append(("jpeg",)) or True)
        monkeypatch.setattr(classify, "write_xmp_sidecar",
                            lambda p, kw: called.append(("xmp",)) or True)
        classify.write_keyword_to_file("/x/a.NEF", "robo_95")
        assert called[0][0] == "xmp"

    def test_nef_dir_sidecar_placement(self, tmp_path, monkeypatch):
        """When nef_dir is given, XMP goes alongside the NEF in nef_dir, not
        next to the source JPEG."""
        nef_dir = tmp_path / "raws"
        nef_dir.mkdir()
        nef_file = nef_dir / "foo.NEF"
        nef_file.write_bytes(b"")

        captured = []
        monkeypatch.setattr(classify, "write_xmp_sidecar",
                            lambda p, kw: captured.append(p) or True)
        classify.write_keyword_to_file("/preview/foo.jpg", "robo_95",
                                       nef_dir=str(nef_dir))
        assert captured == [nef_file]

    def test_missing_nef_returns_false(self, tmp_path, monkeypatch):
        # nef_dir provided but the NEF doesn't exist — return False
        empty_dir = tmp_path / "raws"
        empty_dir.mkdir()
        assert classify.write_keyword_to_file("/preview/ghost.jpg", "select",
                                              nef_dir=str(empty_dir)) is False

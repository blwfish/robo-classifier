"""Tests for inference_hotel.write_xmp_sidecar.

Verifies that the sidecar uses Lightroom-readable keyword namespaces
(dc:subject / lr:hierarchicalSubject via exiftool) and that the sidecar
path is <stem>.xmp, not <filename>.xmp.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inference_hotel


class TestWriteXmpSidecar:
    def test_new_sidecar_calls_tagsfromfile(self, tmp_path, monkeypatch):
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        captured = []
        monkeypatch.setattr(inference_hotel.subprocess, "run",
                            lambda args, **kw: captured.append(args) or
                            MagicMock(returncode=0))
        result = inference_hotel.write_xmp_sidecar(img, "select", 0.95)
        assert result is True
        args = captured[0]
        assert "-tagsfromfile" in args

    def test_existing_sidecar_updated_without_tagsfromfile(self, tmp_path, monkeypatch):
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        xmp = tmp_path / "frame.xmp"
        xmp.write_text("<xmp/>")
        captured = []
        monkeypatch.setattr(inference_hotel.subprocess, "run",
                            lambda args, **kw: captured.append(args) or
                            MagicMock(returncode=0))
        inference_hotel.write_xmp_sidecar(img, "select", 0.95)
        assert "-tagsfromfile" not in captured[0]
        assert "-overwrite_original" in captured[0]

    def test_sidecar_path_replaces_suffix_not_appends(self, tmp_path, monkeypatch):
        # Regression: old code wrote frame.NEF.xmp; correct is frame.xmp
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        captured = []
        monkeypatch.setattr(inference_hotel.subprocess, "run",
                            lambda args, **kw: captured.append(args) or
                            MagicMock(returncode=0))
        inference_hotel.write_xmp_sidecar(img, "select", 0.95)
        args = captured[0]
        flat = " ".join(str(a) for a in args)
        assert "frame.NEF.xmp" not in flat
        assert str(tmp_path / "frame.xmp") in flat

    def test_keyword_written_to_dc_subject(self, tmp_path, monkeypatch):
        # Lightroom reads Keywords (dc:subject), not Iptc4xmpCore:CiKeywords.
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        captured = []
        monkeypatch.setattr(inference_hotel.subprocess, "run",
                            lambda args, **kw: captured.append(args) or
                            MagicMock(returncode=0))
        inference_hotel.write_xmp_sidecar(img, "select", 0.95)
        args = captured[0]
        assert any("-Keywords+=select" in a for a in args)
        assert any("-Subject+=select" in a for a in args)

    def test_hierarchical_subject_uses_robo_prefix(self, tmp_path, monkeypatch):
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        captured = []
        monkeypatch.setattr(inference_hotel.subprocess, "run",
                            lambda args, **kw: captured.append(args) or
                            MagicMock(returncode=0))
        inference_hotel.write_xmp_sidecar(img, "select", 0.95)
        assert any("-HierarchicalSubject+=robo|select" in a for a in captured[0])

    def test_exiftool_missing_returns_false(self, tmp_path, monkeypatch):
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        def raise_nf(*a, **kw): raise FileNotFoundError("exiftool")
        monkeypatch.setattr(inference_hotel.subprocess, "run", raise_nf)
        assert inference_hotel.write_xmp_sidecar(img, "select", 0.95) is False

    def test_nonzero_exit_returns_false(self, tmp_path, monkeypatch):
        img = tmp_path / "frame.NEF"
        img.write_bytes(b"")
        monkeypatch.setattr(inference_hotel.subprocess, "run",
                            lambda *a, **kw: MagicMock(returncode=1))
        assert inference_hotel.write_xmp_sidecar(img, "select", 0.95) is False

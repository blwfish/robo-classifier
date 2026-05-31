"""Tests for presets_loader.py.

load_preset, merge_preset_and_overrides, and list_presets are the interface
between user config files and run_pipeline kwargs. The critical risk is
merge_preset_and_overrides: a None override must NOT silently win over a
valid preset value.
"""

import sys
import tomllib
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import presets_loader


# =============================================================================
# merge_preset_and_overrides
# =============================================================================

class TestMergePresetAndOverrides:
    def test_preset_value_used_when_override_is_none(self):
        preset = {"batch_size": 32, "num_workers": 4}
        explicit = {"batch_size": None}
        result = presets_loader.merge_preset_and_overrides(preset, explicit)
        assert result["batch_size"] == 32

    def test_explicit_value_wins_over_preset(self):
        preset = {"batch_size": 32}
        explicit = {"batch_size": 64}
        result = presets_loader.merge_preset_and_overrides(preset, explicit)
        assert result["batch_size"] == 64

    def test_explicit_zero_wins_over_preset(self):
        # Zero is falsy but is a valid override — must not be treated as None.
        preset = {"junk_min_conf": 0.3}
        explicit = {"junk_min_conf": 0}
        result = presets_loader.merge_preset_and_overrides(preset, explicit)
        assert result["junk_min_conf"] == 0

    def test_explicit_false_wins_over_preset(self):
        preset = {"skip_junk": True}
        explicit = {"skip_junk": False}
        result = presets_loader.merge_preset_and_overrides(preset, explicit)
        assert result["skip_junk"] is False

    def test_key_absent_from_preset_added_if_explicit(self):
        preset = {}
        explicit = {"batch_size": 16}
        result = presets_loader.merge_preset_and_overrides(preset, explicit)
        assert result["batch_size"] == 16

    def test_all_none_overrides_returns_preset_unchanged(self):
        preset = {"batch_size": 32, "num_workers": 4}
        explicit = {"batch_size": None, "num_workers": None}
        result = presets_loader.merge_preset_and_overrides(preset, explicit)
        assert result == preset

    def test_empty_explicit_returns_preset_copy(self):
        preset = {"batch_size": 32}
        result = presets_loader.merge_preset_and_overrides(preset, {})
        assert result == preset
        assert result is not preset  # must be a copy


# =============================================================================
# load_preset
# =============================================================================

class TestLoadPreset:
    def test_loads_real_preset_by_name(self):
        # At least one real preset must exist and load without error.
        presets = presets_loader.list_presets()
        if not presets:
            pytest.skip("no presets in presets/ directory")
        name = presets[0]["name"]
        kwargs = presets_loader.load_preset(name)
        assert isinstance(kwargs, dict)

    def test_unknown_name_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            presets_loader.load_preset("definitely_does_not_exist_xyz")

    def test_error_message_lists_available_presets(self):
        try:
            presets_loader.load_preset("no_such_preset")
        except FileNotFoundError as e:
            # Must tell the user what IS available so they can self-correct.
            assert "Available" in str(e)

    def test_loads_from_explicit_path(self, tmp_path):
        toml_content = b"""
description = "test preset"

[extract]
workers = 8
max_preview_edge = 256

[classifier]
batch_size = 16
"""
        preset_file = tmp_path / "myrig.toml"
        preset_file.write_bytes(toml_content)
        kwargs = presets_loader.load_preset(preset_file)
        assert kwargs["preview_workers"] == 8
        assert kwargs["max_preview_edge"] == 256
        assert kwargs["batch_size"] == 16

    def test_unknown_toml_keys_silently_ignored(self, tmp_path):
        # A key not in PRESET_KEY_MAP must not raise; future-proof for new fields.
        toml_content = b"""
[extract]
workers = 4
future_option = "ignored"
"""
        preset_file = tmp_path / "future.toml"
        preset_file.write_bytes(toml_content)
        kwargs = presets_loader.load_preset(preset_file)
        assert kwargs["preview_workers"] == 4
        assert "future_option" not in kwargs

    def test_missing_section_returns_partial_kwargs(self, tmp_path):
        # A preset with only [classifier] must return only classifier keys.
        toml_content = b"""
[classifier]
batch_size = 64
"""
        preset_file = tmp_path / "partial.toml"
        preset_file.write_bytes(toml_content)
        kwargs = presets_loader.load_preset(preset_file)
        assert kwargs["batch_size"] == 64
        assert "preview_workers" not in kwargs


# =============================================================================
# list_presets
# =============================================================================

class TestListPresets:
    def test_returns_list(self):
        result = presets_loader.list_presets()
        assert isinstance(result, list)

    def test_each_entry_has_name_and_description(self):
        result = presets_loader.list_presets()
        for entry in result:
            assert "name" in entry
            assert "description" in entry

    def test_missing_presets_dir_returns_empty(self, monkeypatch):
        monkeypatch.setattr(presets_loader, "PRESETS_DIR",
                            Path("/tmp/nonexistent_presets_xyz"))
        assert presets_loader.list_presets() == []

    def test_invalid_toml_file_skipped_gracefully(self, tmp_path, monkeypatch):
        monkeypatch.setattr(presets_loader, "PRESETS_DIR", tmp_path)
        (tmp_path / "bad.toml").write_bytes(b"not valid toml [[[")
        (tmp_path / "good.toml").write_bytes(b'description = "ok"')
        result = presets_loader.list_presets()
        names = [e["name"] for e in result]
        assert "good" in names
        # bad.toml should appear but with empty description (parse skipped)
        assert "bad" in names

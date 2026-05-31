"""Tests for app_config.py — minimal TOML parser and AppConfig persistence.

AppConfig is the sole mechanism for persisting user paths (model_library,
dataset_scratch) across sessions. Regressions here break the review UI's
ability to find trained models or write dataset splits. Tests redirect
CONFIG_PATH so they never touch the real ~/.robo-classifier/config.toml.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import app_config
from app_config import _parse_toml, AppConfig


# =============================================================================
# _parse_toml — minimal flat TOML parser
# =============================================================================

class TestParseToml:
    def test_simple_key_value(self):
        # The parser must handle the exact format _save() writes.
        result = _parse_toml('key = "value"')
        assert result == {"key": "value"}

    def test_ignores_comments(self):
        # Comment lines beginning with # must be silently skipped.
        result = _parse_toml('# this is a comment\nkey = "val"')
        assert result == {"key": "val"}
        assert "#" not in result

    def test_ignores_blank_lines(self):
        # Blank lines appear between sections in the saved file.
        result = _parse_toml('\n\nkey = "val"\n\n')
        assert result == {"key": "val"}

    def test_ignores_malformed_lines(self):
        # Lines without the key = "value" pattern must be skipped, not raise.
        result = _parse_toml("not_valid\nkey = \"val\"\nalso bad = no quotes")
        assert result == {"key": "val"}

    def test_handles_extra_whitespace(self):
        # Whitespace around = is allowed by TOML and appears in hand-edited files.
        result = _parse_toml('key   =   "value"')
        assert result == {"key": "value"}

    def test_empty_value(self):
        # An empty quoted string is a valid value representing "unset".
        result = _parse_toml('key = ""')
        assert result == {"key": ""}

    def test_multiple_keys(self):
        text = 'model_library = "/models"\ndataset_scratch = "/scratch"'
        result = _parse_toml(text)
        assert result["model_library"] == "/models"
        assert result["dataset_scratch"] == "/scratch"

    def test_value_with_spaces(self):
        # Paths with spaces must survive the round-trip.
        result = _parse_toml('key = "/Volumes/My Drive/models"')
        assert result == {"key": "/Volumes/My Drive/models"}


# =============================================================================
# AppConfig.get / set — round-trip persistence
# =============================================================================

class TestGetSet:
    def test_get_missing_key_returns_empty_string(self, tmp_path, monkeypatch):
        # get() must return "" for any key not yet configured, not raise.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        assert cfg.get("model_library") == ""

    def test_set_persists_across_reload(self, tmp_path, monkeypatch):
        # A set() followed by a fresh AppConfig() must read back the same value.
        config_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_config, "CONFIG_PATH", config_path)

        cfg = AppConfig()
        cfg.set("model_library", "/Volumes/models")

        cfg2 = AppConfig()
        assert cfg2.get("model_library") == "/Volumes/models"

    def test_set_strips_whitespace(self, tmp_path, monkeypatch):
        # Leading/trailing whitespace in user-supplied paths must be trimmed.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set("model_library", "  /Volumes/models  ")
        assert cfg.get("model_library") == "/Volumes/models"

    def test_set_overwrites_existing(self, tmp_path, monkeypatch):
        # A second set() on the same key must replace the previous value.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set("model_library", "/first")
        cfg.set("model_library", "/second")

        cfg2 = AppConfig()
        assert cfg2.get("model_library") == "/second"


# =============================================================================
# AppConfig.set_many — atomic multi-key update
# =============================================================================

class TestSetMany:
    def test_set_many_persists_all_keys(self, tmp_path, monkeypatch):
        # set_many must save every key in a single write so they all survive reload.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set_many({"model_library": "/m", "dataset_scratch": "/d"})

        cfg2 = AppConfig()
        assert cfg2.get("model_library") == "/m"
        assert cfg2.get("dataset_scratch") == "/d"

    def test_set_many_strips_whitespace(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set_many({"model_library": "  /m  "})
        assert cfg.get("model_library") == "/m"

    def test_set_many_partial_update_preserves_other_keys(self, tmp_path, monkeypatch):
        # Updating one key must not erase another key that was already set.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set("model_library", "/m")
        cfg.set_many({"dataset_scratch": "/d"})

        cfg2 = AppConfig()
        assert cfg2.get("model_library") == "/m"
        assert cfg2.get("dataset_scratch") == "/d"


# =============================================================================
# AppConfig.missing_required
# =============================================================================

class TestMissingRequired:
    def test_all_missing_on_fresh_config(self, tmp_path, monkeypatch):
        # A brand-new config file has no paths set — both required keys missing.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        missing = cfg.missing_required()
        assert "model_library" in missing
        assert "dataset_scratch" in missing

    def test_empty_after_both_set(self, tmp_path, monkeypatch):
        # missing_required must return [] when every required field has a value.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set_many({"model_library": "/m", "dataset_scratch": "/d"})
        assert cfg.missing_required() == []

    def test_partial_set_still_missing_one(self, tmp_path, monkeypatch):
        # Setting only one required key must still report the other as missing.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set("model_library", "/m")
        missing = cfg.missing_required()
        assert "dataset_scratch" in missing
        assert "model_library" not in missing

    def test_whitespace_only_value_counts_as_missing(self, tmp_path, monkeypatch):
        # A value that is all whitespace is semantically unset.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg._data["model_library"] = "   "  # bypass set() to avoid strip
        assert "model_library" in cfg.missing_required()


# =============================================================================
# AppConfig.as_dict — structured representation for the UI
# =============================================================================

class TestAsDict:
    def test_includes_all_fields(self, tmp_path, monkeypatch):
        # as_dict must expose every key in _FIELDS so the UI can render them all.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        d = cfg.as_dict()
        assert "model_library" in d["fields"]
        assert "dataset_scratch" in d["fields"]

    def test_field_has_value_label_required(self, tmp_path, monkeypatch):
        # Each field entry must contain value, label, and required for the UI form.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        field = cfg.as_dict()["fields"]["model_library"]
        assert "value" in field
        assert "label" in field
        assert "required" in field

    def test_missing_required_in_as_dict(self, tmp_path, monkeypatch):
        # as_dict must include missing_required so the UI knows what to highlight.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        d = cfg.as_dict()
        assert "missing_required" in d
        assert isinstance(d["missing_required"], list)

    def test_missing_required_empty_when_all_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set_many({"model_library": "/m", "dataset_scratch": "/d"})
        d = cfg.as_dict()
        assert d["missing_required"] == []

    def test_value_reflects_current_setting(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set("model_library", "/Volumes/models")
        d = cfg.as_dict()
        assert d["fields"]["model_library"]["value"] == "/Volumes/models"


# =============================================================================
# AppConfig.model_library / dataset_scratch properties
# =============================================================================

class TestPathProperties:
    def test_model_library_none_when_unset(self, tmp_path, monkeypatch):
        # Unset path properties must return None, not an empty Path.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        assert cfg.model_library is None

    def test_dataset_scratch_none_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        assert cfg.dataset_scratch is None

    def test_model_library_returns_path(self, tmp_path, monkeypatch):
        # A configured model_library must come back as a Path, not a str.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set("model_library", "/Volumes/models")
        assert isinstance(cfg.model_library, Path)
        assert str(cfg.model_library) == "/Volumes/models"

    def test_dataset_scratch_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set("dataset_scratch", "/tmp/scratch")
        assert isinstance(cfg.dataset_scratch, Path)

    def test_tilde_is_expanded(self, tmp_path, monkeypatch):
        # Paths written with ~ must be resolved to the real home directory.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set("model_library", "~/models")
        assert "~" not in str(cfg.model_library)
        assert str(cfg.model_library).startswith("/")


# =============================================================================
# Unknown-key preservation
# =============================================================================

class TestUnknownKeyPreservation:
    def test_unknown_key_survives_save(self, tmp_path, monkeypatch):
        # Keys not in _FIELDS (e.g. future or third-party keys) must not be
        # silently dropped when any _FIELDS key is updated.
        config_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_config, "CONFIG_PATH", config_path)

        cfg = AppConfig()
        cfg._data["future_option"] = "yes"
        cfg.set("model_library", "/m")  # triggers _save()

        cfg2 = AppConfig()
        assert cfg2.get("future_option") == "yes"


# =============================================================================
# Missing file — graceful load
# =============================================================================

# =============================================================================
# AppConfig.default_profile — optional, not in missing_required
# =============================================================================

class TestDefaultProfile:
    def test_default_profile_none_when_unset(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        assert cfg.default_profile is None

    def test_default_profile_returns_value_when_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set("default_profile", "pca")
        assert cfg.default_profile == "pca"

    def test_default_profile_whitespace_only_returns_none(self, tmp_path, monkeypatch):
        # "   " is meaningless as a profile name — treat as unset.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg._data["default_profile"] = "   "
        assert cfg.default_profile is None

    def test_default_profile_not_in_missing_required(self, tmp_path, monkeypatch):
        # default_profile is optional; leaving it unset must not block the UI.
        monkeypatch.setattr(app_config, "CONFIG_PATH", tmp_path / "config.toml")
        cfg = AppConfig()
        cfg.set_many({"model_library": "/m", "dataset_scratch": "/d"})
        assert "default_profile" not in cfg.missing_required()
        assert cfg.missing_required() == []

    def test_default_profile_survives_roundtrip(self, tmp_path, monkeypatch):
        config_path = tmp_path / "config.toml"
        monkeypatch.setattr(app_config, "CONFIG_PATH", config_path)
        cfg = AppConfig()
        cfg.set("default_profile", "imsa")
        cfg2 = AppConfig()
        assert cfg2.default_profile == "imsa"


# =============================================================================

class TestMissingFile:
    def test_load_missing_file_does_not_raise(self, tmp_path, monkeypatch):
        # A user who has never run the app has no config file — must start clean.
        nonexistent = tmp_path / "nonexistent" / "config.toml"
        monkeypatch.setattr(app_config, "CONFIG_PATH", nonexistent)
        cfg = AppConfig()  # must not raise
        assert cfg.get("model_library") == ""

    def test_load_missing_file_gives_empty_config(self, tmp_path, monkeypatch):
        nonexistent = tmp_path / "nonexistent" / "config.toml"
        monkeypatch.setattr(app_config, "CONFIG_PATH", nonexistent)
        cfg = AppConfig()
        assert cfg.missing_required() != []  # nothing set

"""
app_config.py — user configuration for robo-classifier.

Config file: ~/.robo-classifier/config.toml
Written on first save; never committed to git.

Required paths (must be set before training or model use):
    model_library   — directory of <name>.pt + <name>.json model pairs
    dataset_scratch — writable scratch space for train/test splits

Usage:
    from app_config import config
    lib = config.model_library        # Path or None if unset
    config.set("model_library", "/Volumes/scratch/models")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


CONFIG_PATH = Path.home() / ".robo-classifier" / "config.toml"

# Keys the app cares about, with human labels and whether they're required.
_FIELDS = {
    "model_library":    {"label": "Model library",    "required": True},
    "dataset_scratch":  {"label": "Dataset scratch",  "required": True},
    "default_profile":  {"label": "Default profile",  "required": False},
}


class AppConfig:
    def __init__(self):
        self._data: dict[str, str] = {}
        self._load()

    # ---- public API ----

    @property
    def model_library(self) -> Optional[Path]:
        v = self._data.get("model_library", "").strip()
        return Path(v).expanduser() if v else None

    @property
    def dataset_scratch(self) -> Optional[Path]:
        v = self._data.get("dataset_scratch", "").strip()
        return Path(v).expanduser() if v else None

    @property
    def default_profile(self) -> Optional[str]:
        v = self._data.get("default_profile", "").strip()
        return v or None

    def get(self, key: str) -> str:
        return self._data.get(key, "")

    def set(self, key: str, value: str) -> None:
        self._data[key] = value.strip()
        self._save()

    def set_many(self, updates: dict[str, str]) -> None:
        for k, v in updates.items():
            self._data[k] = v.strip()
        self._save()

    def missing_required(self) -> list[str]:
        """Return list of required keys that are unset or empty."""
        return [k for k, meta in _FIELDS.items() if meta["required"] and not self._data.get(k, "").strip()]

    def as_dict(self) -> dict:
        return {
            "fields": {
                k: {
                    "value":    self._data.get(k, ""),
                    "label":    meta["label"],
                    "required": meta["required"],
                }
                for k, meta in _FIELDS.items()
            },
            "missing_required": self.missing_required(),
        }

    # ---- I/O ----

    def _load(self):
        if not CONFIG_PATH.exists():
            self._data = {}
            return
        try:
            self._data = _parse_toml(CONFIG_PATH.read_text())
        except Exception:
            self._data = {}

    def _save(self):
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# robo-classifier configuration\n"]
        for k, meta in _FIELDS.items():
            v = self._data.get(k, "").replace('"', '\\"')
            lines.append(f"# {meta['label']}\n")
            lines.append(f'{k} = "{v}"\n\n')
        # Preserve any extra keys not in _FIELDS
        for k, v in self._data.items():
            if k not in _FIELDS:
                escaped = v.replace('"', '\\"')
                lines.append(f'{k} = "{escaped}"\n')
        CONFIG_PATH.write_text("".join(lines))


def _parse_toml(text: str) -> dict[str, str]:
    """
    Minimal TOML parser — handles only flat string assignments.
    Avoids a toml dependency; the config is intentionally simple.
    """
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Allow escaped quotes (\") inside the value; use non-greedy match up
        # to the final unescaped closing quote.
        m = re.match(r'^(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"$', line)
        if m:
            result[m.group(1)] = m.group(2).replace('\\"', '"')
    return result


# Module-level singleton
config = AppConfig()

"""
Hardware-tuning preset loader.

Presets live in presets/<name>.toml and bundle the per-rig knobs that shift
around between machines (worker counts, batch sizes, preview sizes, detection
thresholds). Load one up front, then any key not overridden explicitly on the
command line uses the preset's value.

See presets/CUSTOM.md for the expected schema and authoring guidance.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


PRESETS_DIR = Path(__file__).resolve().parent / "presets"


# Flat mapping from preset "dotted.key" -> canonical Python kwarg name used
# throughout classify.run_pipeline. Keeping it in one place avoids spreading
# the TOML shape across the codebase.
PRESET_KEY_MAP = {
    "extract.workers":             "preview_workers",
    "extract.max_preview_edge":    "max_preview_edge",
    "junk_filter.batch_size":      "junk_batch_size",
    "junk_filter.imgsz":           "junk_imgsz",
    "junk_filter.min_conf":        "junk_min_conf",
    "junk_filter.min_visible_frac": "junk_min_visible_frac",
    "junk_filter.min_area_frac":   "junk_min_area_frac",
    "junk_filter.edge_min_area_frac": "junk_edge_min_area_frac",
    "classifier.batch_size":       "batch_size",
    "classifier.num_workers":      "num_workers",
}


def list_presets() -> list[dict]:
    """Return [{name, description}] for every preset in presets/."""
    if not PRESETS_DIR.exists():
        return []
    out = []
    for p in sorted(PRESETS_DIR.glob("*.toml")):
        desc = ""
        try:
            with open(p, "rb") as f:
                data = tomllib.load(f)
            desc = data.get("description", "")
        except Exception:
            pass
        out.append({"name": p.stem, "description": desc, "path": str(p)})
    return out


def load_preset(name_or_path) -> dict:
    """
    Load a preset by name (presets/<name>.toml) or absolute path, and return
    a flat dict of kwargs suitable for splatting into run_pipeline.

    Raises FileNotFoundError if the preset doesn't exist.
    """
    p = Path(name_or_path)
    if not p.exists():
        p = PRESETS_DIR / f"{name_or_path}.toml"
    if not p.exists():
        available = [x["name"] for x in list_presets()]
        raise FileNotFoundError(
            f"Preset '{name_or_path}' not found. Available: {available or '(none)'}"
        )

    with open(p, "rb") as f:
        data = tomllib.load(f)

    # Flatten section.key into canonical kwarg names.
    kwargs: dict = {}
    for dotted, kwarg in PRESET_KEY_MAP.items():
        section, _, key = dotted.partition(".")
        section_dict = data.get(section, {})
        if key in section_dict:
            kwargs[kwarg] = section_dict[key]
    return kwargs


def merge_preset_and_overrides(preset_kwargs: dict, explicit: dict) -> dict:
    """
    Merge a preset's kwargs with explicit overrides. `explicit` values win
    when they're not None. A missing key falls back to the preset.
    """
    out = dict(preset_kwargs)
    for k, v in explicit.items():
        if v is not None:
            out[k] = v
    return out

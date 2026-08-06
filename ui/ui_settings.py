"""Persistence for presentation-layer preferences.

Mirrors the shape of `core.runtime_settings`: JSON under APPDATA_DIR, every
read merged against defaults so a partial or corrupt file degrades instead
of raising. Kept separate from runtime_settings because these keys are
purely about how the app looks, not how it behaves.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from core.config import APPDATA_DIR


UI_SETTINGS_PATH = APPDATA_DIR / "ui_settings.json"

THEME_PRESET_AUTO = "auto"
ANIMATION_QUALITIES = ("auto", "high", "balanced", "minimal")

DEFAULT_UI_SETTINGS: Dict[str, Any] = {
    # "auto" follows the persona (JARVIS -> gold, FRIDAY -> cyan); any other
    # value pins a specific preset and persona no longer drives color.
    "theme_preset": THEME_PRESET_AUTO,
    # Empty string uses the preset accent. Otherwise "#rrggbb".
    "accent_override": "",
    "animation_quality": "auto",
    # Saved as hex of QWidget.saveGeometry(); empty means "center on screen".
    "window_geometry": "",
    "window_maximized": False,
    "frameless": True,
}

_HEX_COLOR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
_HEX_BLOB = re.compile(r"^[0-9a-fA-F]*$")


def _merge_defaults(data: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(DEFAULT_UI_SETTINGS)
    if not isinstance(data, dict):
        return merged

    preset = str(data.get("theme_preset", merged["theme_preset"]) or "").strip().lower()
    if preset:
        merged["theme_preset"] = preset

    accent = str(data.get("accent_override", "") or "").strip()
    merged["accent_override"] = accent if _HEX_COLOR.match(accent) else ""

    quality = str(data.get("animation_quality", merged["animation_quality"]) or "").strip().lower()
    merged["animation_quality"] = quality if quality in ANIMATION_QUALITIES else "auto"

    geometry = str(data.get("window_geometry", "") or "").strip()
    # Reject anything that is not a plausible saveGeometry() blob so a bad
    # value cannot make the window unrestorable.
    merged["window_geometry"] = geometry if _HEX_BLOB.match(geometry) and len(geometry) <= 4096 else ""

    if isinstance(data.get("window_maximized"), bool):
        merged["window_maximized"] = data["window_maximized"]

    if isinstance(data.get("frameless"), bool):
        merged["frameless"] = data["frameless"]

    return merged


def load_ui_settings() -> Dict[str, Any]:
    if UI_SETTINGS_PATH.exists():
        try:
            with UI_SETTINGS_PATH.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return _merge_defaults(payload)
        except Exception:
            pass

    defaults = dict(DEFAULT_UI_SETTINGS)
    save_ui_settings(defaults)
    return defaults


def save_ui_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    merged = _merge_defaults(settings)
    APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    with UI_SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return merged


def update_ui_settings(**changes: Any) -> Dict[str, Any]:
    """Read, apply `changes`, write back, return the merged result."""
    current = load_ui_settings()
    current.update(changes)
    return save_ui_settings(current)

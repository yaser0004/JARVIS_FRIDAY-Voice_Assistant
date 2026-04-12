from __future__ import annotations

import json
from typing import Any, Dict

from core.compute_runtime import normalize_compute_mode
from core.config import APPDATA_DIR


RUNTIME_SETTINGS_PATH = APPDATA_DIR / "runtime_settings.json"

DEFAULT_RUNTIME_SETTINGS: Dict[str, Any] = {
    "compute_mode": "auto",
    "tts_enabled": True,
    "tts_profile": "female",
    "response_verbosity": "normal",
}


def _merge_defaults(data: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(DEFAULT_RUNTIME_SETTINGS)
    if not isinstance(data, dict):
        return merged

    merged["compute_mode"] = normalize_compute_mode(data.get("compute_mode"))

    if isinstance(data.get("tts_enabled"), bool):
        merged["tts_enabled"] = data["tts_enabled"]

    profile = str(data.get("tts_profile", merged["tts_profile"]) or "").strip().lower()
    merged["tts_profile"] = "male" if profile == "male" else "female"

    verbosity = str(data.get("response_verbosity", merged["response_verbosity"]) or "").strip().lower()
    if verbosity not in {"brief", "normal", "detailed"}:
        verbosity = "normal"
    merged["response_verbosity"] = verbosity
    return merged


def load_runtime_settings() -> Dict[str, Any]:
    if RUNTIME_SETTINGS_PATH.exists():
        try:
            with RUNTIME_SETTINGS_PATH.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return _merge_defaults(payload)
        except Exception:
            pass

    defaults = dict(DEFAULT_RUNTIME_SETTINGS)
    save_runtime_settings(defaults)
    return defaults


def save_runtime_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    merged = _merge_defaults(settings)
    APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    with RUNTIME_SETTINGS_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return merged

from __future__ import annotations

import json
from typing import Any, Dict, List

from core.config import APPDATA_DIR


WAKEWORD_CONFIG_PATH = APPDATA_DIR / "wakeword.json"

DEFAULT_WAKEWORD_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "sensitivity": 0.35,
    "activation_phrases": ["jarvis", "hey jarvis"],
    "strict_phrase_prefix": False,
    "auto_restart_after_response": True,
    "follow_up_after_response": True,
    "follow_up_timeout": 8,
    "max_followup_turns": 1,
}


def _merge_defaults(data: Dict[str, Any] | None) -> Dict[str, Any]:
    merged = dict(DEFAULT_WAKEWORD_CONFIG)
    if not isinstance(data, dict):
        return merged

    if isinstance(data.get("enabled"), bool):
        merged["enabled"] = data["enabled"]

    try:
        sensitivity = float(data.get("sensitivity", merged["sensitivity"]))
    except Exception:
        sensitivity = float(merged["sensitivity"])
    merged["sensitivity"] = max(0.1, min(0.95, sensitivity))

    raw_phrases = data.get("activation_phrases")
    if isinstance(raw_phrases, list):
        phrases = []
        for item in raw_phrases:
            text = str(item).strip().lower()
            if text:
                phrases.append(text)
        if phrases:
            merged["activation_phrases"] = phrases[:8]

    if isinstance(data.get("strict_phrase_prefix"), bool):
        merged["strict_phrase_prefix"] = data["strict_phrase_prefix"]

    if isinstance(data.get("auto_restart_after_response"), bool):
        merged["auto_restart_after_response"] = data["auto_restart_after_response"]

    if isinstance(data.get("follow_up_after_response"), bool):
        merged["follow_up_after_response"] = data["follow_up_after_response"]

    try:
        follow_up_timeout = int(data.get("follow_up_timeout", merged["follow_up_timeout"]))
    except Exception:
        follow_up_timeout = int(merged["follow_up_timeout"])
    merged["follow_up_timeout"] = max(3, min(20, follow_up_timeout))

    try:
        max_followup_turns = int(data.get("max_followup_turns", merged["max_followup_turns"]))
    except Exception:
        max_followup_turns = int(merged["max_followup_turns"])
    merged["max_followup_turns"] = max(0, min(3, max_followup_turns))

    return merged


def load_wakeword_config() -> Dict[str, Any]:
    if WAKEWORD_CONFIG_PATH.exists():
        try:
            with WAKEWORD_CONFIG_PATH.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return _merge_defaults(payload)
        except Exception:
            pass

    config = dict(DEFAULT_WAKEWORD_CONFIG)
    save_wakeword_config(config)
    return config


def save_wakeword_config(config: Dict[str, Any]) -> Dict[str, Any]:
    merged = _merge_defaults(config)
    APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    with WAKEWORD_CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return merged


def parse_phrases_csv(value: str) -> List[str]:
    items = [item.strip().lower() for item in str(value).split(",")]
    return [item for item in items if item]


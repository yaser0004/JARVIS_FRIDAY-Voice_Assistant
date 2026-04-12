from __future__ import annotations

import tkinter as tk
from typing import Any, Dict


_ROOT = None


def _get_root() -> tk.Tk:
    global _ROOT
    if _ROOT is None:
        _ROOT = tk.Tk()
        _ROOT.withdraw()
    return _ROOT


def _response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


def handle(entities: Dict[str, Any]) -> Dict[str, Any]:
    action = str(entities.get("clipboard_action", "read")).lower()
    payload = str(entities.get("clipboard_text", ""))

    try:
        root = _get_root()

        if action == "read":
            text = root.clipboard_get()
            return _response(True, "Clipboard content retrieved.", {"text": text})

        if action in {"copy", "write"}:
            root.clipboard_clear()
            root.clipboard_append(payload)
            root.update()
            return _response(True, "Clipboard updated.", {"text": payload})

        if action == "paste":
            text = root.clipboard_get()
            return _response(True, "Clipboard content ready to paste.", {"text": text})

        return _response(False, f"Unsupported clipboard action: {action}")
    except Exception as exc:
        return _response(False, f"I could not handle clipboard action: {exc}", {"error": str(exc)})

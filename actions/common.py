"""Helpers shared by the action modules.

Every action handler returns the same response dict shape, and several
modules fetch small JSON/text payloads over HTTP with the same headers,
timeout handling, and swallow-and-return-None error policy. Both live here
so the behavior stays identical across modules.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from urllib.request import Request, urlopen

_USER_AGENT = "Mozilla/5.0"
DEFAULT_TIMEOUT_SECONDS = 5


def response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


def fetch_text(url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> str | None:
    request = Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception:
        return None


def fetch_json(url: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Dict[str, Any] | None:
    raw = fetch_text(url, timeout=timeout)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None

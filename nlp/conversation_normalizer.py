from __future__ import annotations

import re


_WS_RE = re.compile(r"\s+")
_FILLER_RE = re.compile(r"\b(?:um+|uh+|hmm+|er+|ah+)\b", re.IGNORECASE)
_ASSISTANT_PREFIX_RE = re.compile(
    r"^\s*(?:hey|hi|hello|ok|okay|yo)?\s*(?:jarvis|friday|assistant|aria)\b[\s,!:;-]*",
    re.IGNORECASE,
)

_LEADING_PATTERNS = [
    re.compile(r"^\s*do\s+me\s+a\s+favor(?:\s+and)?\s+", re.IGNORECASE),
    re.compile(r"^\s*(?:can|could|would|will|shall)\s+(?:you|u)\s+", re.IGNORECASE),
    re.compile(r"^\s*would\s+you\s+mind\s+", re.IGNORECASE),
    re.compile(r"^\s*i\s+(?:need|want)\s+you\s+to\s+", re.IGNORECASE),
    re.compile(r"^\s*i\s+(?:would|\'d)?\s*like\s+you\s+to\s+", re.IGNORECASE),
    re.compile(r"^\s*help\s+me\s+(?:to\s+)?", re.IGNORECASE),
    re.compile(r"^\s*(?:please|kindly|just)\s+", re.IGNORECASE),
]

_TRAILING_PATTERNS = [
    re.compile(r"\s+(?:please|thanks|thank\s+you)\s*$", re.IGNORECASE),
    re.compile(r"\s+(?:for\s+me|if\s+you\s+can|if\s+possible|right\s+now)\s*$", re.IGNORECASE),
]


def _collapse_ws(text: str) -> str:
    return _WS_RE.sub(" ", str(text or "")).strip()


def _strip_leading_polite_phrases(text: str) -> str:
    value = str(text or "")
    for _ in range(8):
        changed = False
        for pattern in _LEADING_PATTERNS:
            updated = pattern.sub("", value, count=1)
            if updated != value:
                value = updated
                changed = True
                break
        if not changed:
            break
    return _collapse_ws(value)


def _strip_trailing_polite_phrases(text: str) -> str:
    value = str(text or "")
    for _ in range(6):
        changed = False
        for pattern in _TRAILING_PATTERNS:
            updated = pattern.sub("", value, count=1)
            if updated != value:
                value = updated
                changed = True
                break
        if not changed:
            break
    return _collapse_ws(value)


def normalize_command_text(text: str) -> str:
    """
    Normalize natural conversational command phrasing into a compact command core.
    """
    value = _collapse_ws(str(text or "").lower())
    if not value:
        return ""

    value = _collapse_ws(_FILLER_RE.sub(" ", value))
    value = _collapse_ws(_ASSISTANT_PREFIX_RE.sub("", value, count=1))
    value = _strip_leading_polite_phrases(value)
    value = _strip_trailing_polite_phrases(value)
    return _collapse_ws(value)

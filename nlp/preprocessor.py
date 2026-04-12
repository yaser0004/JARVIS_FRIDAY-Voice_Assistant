from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s\-\./:%]")


def clean(text: str) -> str:
    normalized = text.strip().lower()
    normalized = normalize_percent_words(normalized)
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def normalize_percent_words(text: str) -> str:
    replacements = {
        "maximum": "max",
        "minimum": "min",
        "percent": "%",
    }
    out = text
    for source, target in replacements.items():
        out = out.replace(source, target)
    return out

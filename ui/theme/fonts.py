"""Font resolution.

The repository ships zero-byte placeholders for Orbitron, Share Tech Mono
and Rajdhani, so the previous `QFont("Orbitron")` calls silently fell through
to the system default and the entire app rendered in one undifferentiated
face. Resolution here is a chain: the first family actually installed wins,
and any real font files that do get dropped into assets/fonts are picked up
automatically.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

from PyQt6.QtGui import QFont, QFontDatabase

from core.session_logging import trace_event
from ui.theme.tokens import Typography


_ASSET_FONTS = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts"

_loaded_bundled: list[str] = []


def load_bundled_fonts() -> list[str]:
    """Register any non-empty font files under assets/fonts.

    Returns the family names that were added. Empty placeholder files are
    skipped rather than reported as failures.
    """
    global _loaded_bundled
    if _loaded_bundled:
        return list(_loaded_bundled)

    families: list[str] = []
    if not _ASSET_FONTS.is_dir():
        return families

    for path in sorted(_ASSET_FONTS.glob("*.tt[fc]")) + sorted(_ASSET_FONTS.glob("*.otf")):
        try:
            if path.stat().st_size <= 0:
                continue
            font_id = QFontDatabase.addApplicationFont(str(path))
        except OSError:
            continue
        if font_id < 0:
            trace_event("ui.theme", "bundled_font_rejected", path=str(path))
            continue
        families.extend(QFontDatabase.applicationFontFamilies(font_id))

    _loaded_bundled = families
    if families:
        trace_event("ui.theme", "bundled_fonts_loaded", families=families)
    return list(families)


def _first_installed(stack: Sequence[str], installed: set[str]) -> str | None:
    for family in stack:
        if family in installed:
            return family
    return None


def resolve_typography(base: Typography | None = None) -> Typography:
    """Pick a concrete family for each role from what this machine has."""
    base = base or Typography()
    load_bundled_fonts()

    try:
        installed = set(QFontDatabase.families())
    except Exception:
        installed = set()

    display = _first_installed(base.display_stack, installed)
    body = _first_installed(base.body_stack, installed)
    mono = _first_installed(base.mono_stack, installed)

    # A missing family is survivable: Qt substitutes. Falling back to the
    # resolved body face keeps the type system coherent instead of dropping
    # to whatever the platform default happens to be.
    body = body or base.body
    display = display or body
    mono = mono or body

    resolved = replace(base, display=display, body=body, mono=mono)
    trace_event(
        "ui.theme",
        "typography_resolved",
        display=display,
        body=body,
        mono=mono,
        installed_count=len(installed),
    )
    return resolved


def font(
    family: str,
    size: int,
    *,
    weight: QFont.Weight = QFont.Weight.Normal,
    letter_spacing: float = 0.0,
    caps: bool = False,
) -> QFont:
    """Build a QFont, with optional tracking and small-caps styling."""
    f = QFont(family, size)
    f.setWeight(weight)
    if letter_spacing:
        f.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 100.0 + letter_spacing)
    if caps:
        f.setCapitalization(QFont.Capitalization.AllUppercase)
    f.setHintingPreference(QFont.HintingPreference.PreferFullHinting)
    return f


def families_report(names: Iterable[str]) -> dict[str, bool]:
    """Diagnostic helper: which of `names` are actually available."""
    try:
        installed = set(QFontDatabase.families())
    except Exception:
        installed = set()
    return {name: name in installed for name in names}

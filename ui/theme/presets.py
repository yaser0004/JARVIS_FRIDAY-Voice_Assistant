"""Built-in themes.

JARVIS and FRIDAY are the persona themes. Monochrome and Daylight are
persona-neutral alternatives for users who do not want the film palette.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor

from ui.theme.tokens import Metrics, OrbProfile, Palette, Theme, with_alpha


def _c(value: str, alpha: int = 255) -> QColor:
    color = QColor(value)
    color.setAlpha(alpha)
    return color


# ---------------------------------------------------------------------------
# JARVIS
#
# Warm gold and amber with a hot white core. Denser weave, slower and
# statelier motion than FRIDAY.
# ---------------------------------------------------------------------------

_JARVIS_ACCENT = _c("#ffb43c")

JARVIS = Theme(
    name="JARVIS",
    palette=Palette(
        bg=_c("#05070a"),
        surface=_c("#0d0f13", 236),
        surface_alt=_c("#14161b", 242),
        surface_hi=with_alpha(_JARVIS_ACCENT, 26),
        border=with_alpha(_JARVIS_ACCENT, 44),
        border_strong=with_alpha(_JARVIS_ACCENT, 116),
        text=_c("#ede6d8"),
        text_dim=_c("#9c9384"),
        text_faint=_c("#5f5a50"),
        accent=_JARVIS_ACCENT,
        accent_dim=_c("#c98b24"),
        accent_hot=_c("#ffe9b8"),
        glow=_c("#ff8a18"),
        ok=_c("#48d18a"),
        warn=_c("#ffc24a"),
        danger=_c("#ff4d5e"),
        is_dark=True,
    ),
    orb=OrbProfile(
        core=_c("#fff4d6"),
        filament_a=_c("#ffc24a"),
        filament_b=_c("#ff8a18"),
        bloom=_c("#ff6a00"),
        ring=_c("#ffd98a"),
        filaments=26,
        points=72,
        rings=3,
        ticks_per_ring=52,
        spin_rate=13.0,
        tilt_rate=8.0,
        ring_rate=24.0,
        tape_rate=0.05,
    ),
)


# ---------------------------------------------------------------------------
# FRIDAY
#
# Sibling of the JARVIS sphere: identical architecture so they read as one
# system, differentiated by a cool palette and a tighter, faster profile.
# ---------------------------------------------------------------------------

_FRIDAY_ACCENT = _c("#38cfff")

FRIDAY = Theme(
    name="FRIDAY",
    palette=Palette(
        bg=_c("#04070b"),
        surface=_c("#0b1016", 236),
        surface_alt=_c("#10171f", 242),
        surface_hi=with_alpha(_FRIDAY_ACCENT, 26),
        border=with_alpha(_FRIDAY_ACCENT, 44),
        border_strong=with_alpha(_FRIDAY_ACCENT, 116),
        text=_c("#dceef7"),
        text_dim=_c("#7f98a8"),
        text_faint=_c("#4c5f6c"),
        accent=_FRIDAY_ACCENT,
        accent_dim=_c("#1e93b8"),
        accent_hot=_c("#d6f6ff"),
        glow=_c("#0a84ff"),
        ok=_c("#3adfa0"),
        warn=_c("#ffc24a"),
        danger=_c("#ff5470"),
        is_dark=True,
    ),
    orb=OrbProfile(
        core=_c("#eaf9ff"),
        filament_a=_c("#4fd8ff"),
        filament_b=_c("#0a84ff"),
        bloom=_c("#00a6ff"),
        ring=_c("#bfefff"),
        filaments=20,
        points=72,
        rings=3,
        ticks_per_ring=44,
        spin_rate=17.0,
        tilt_rate=11.0,
        ring_rate=33.0,
        tape_rate=0.08,
    ),
)


# ---------------------------------------------------------------------------
# Monochrome
# ---------------------------------------------------------------------------

_MONO_ACCENT = _c("#c8d2dc")

MONOCHROME = Theme(
    name="Monochrome",
    palette=Palette(
        bg=_c("#08090b"),
        surface=_c("#101216", 236),
        surface_alt=_c("#171a1f", 242),
        surface_hi=with_alpha(_MONO_ACCENT, 22),
        border=with_alpha(_MONO_ACCENT, 40),
        border_strong=with_alpha(_MONO_ACCENT, 104),
        text=_c("#e4e8ec"),
        text_dim=_c("#8d959d"),
        text_faint=_c("#565c63"),
        accent=_MONO_ACCENT,
        accent_dim=_c("#8c949c"),
        accent_hot=_c("#ffffff"),
        glow=_c("#9fb0bf"),
        ok=_c("#6fd39b"),
        warn=_c("#e3c169"),
        danger=_c("#e8697a"),
        is_dark=True,
    ),
    orb=OrbProfile(
        core=_c("#ffffff"),
        filament_a=_c("#d5dde5"),
        filament_b=_c("#8b98a5"),
        bloom=_c("#9fb0bf"),
        ring=_c("#eef2f6"),
        filaments=23,
        points=72,
        rings=3,
        ticks_per_ring=48,
        spin_rate=15.0,
        tilt_rate=9.5,
        ring_rate=28.0,
        tape_rate=0.06,
    ),
)


# ---------------------------------------------------------------------------
# Daylight
#
# The orb stays luminous here; OrbWidget paints a dark viewport disc behind
# it when `palette.is_dark` is False, which is why these orb colors are not
# darkened for the light surface.
# ---------------------------------------------------------------------------

_LIGHT_ACCENT = _c("#0a6ec4")

DAYLIGHT = Theme(
    name="Daylight",
    palette=Palette(
        bg=_c("#eef1f5"),
        surface=_c("#ffffff", 250),
        surface_alt=_c("#e7ecf2", 250),
        surface_hi=with_alpha(_LIGHT_ACCENT, 26),
        border=with_alpha(_c("#0d1b2a"), 40),
        border_strong=with_alpha(_LIGHT_ACCENT, 130),
        text=_c("#101418"),
        text_dim=_c("#525d69"),
        text_faint=_c("#8a949e"),
        accent=_LIGHT_ACCENT,
        accent_dim=_c("#0d5698"),
        accent_hot=_c("#3ea1ee"),
        glow=_c("#0a6ec4"),
        ok=_c("#1a8f5f"),
        warn=_c("#a5730a"),
        danger=_c("#c62b42"),
        is_dark=False,
    ),
    orb=OrbProfile(
        core=_c("#eaf9ff"),
        filament_a=_c("#4fd8ff"),
        filament_b=_c("#0a84ff"),
        bloom=_c("#00a6ff"),
        ring=_c("#bfefff"),
        filaments=22,
        points=72,
        rings=3,
        ticks_per_ring=46,
        spin_rate=15.0,
        tilt_rate=9.5,
        ring_rate=28.0,
        tape_rate=0.06,
    ),
)


PRESETS: dict[str, Theme] = {
    "jarvis": JARVIS,
    "friday": FRIDAY,
    "monochrome": MONOCHROME,
    "daylight": DAYLIGHT,
}

# Human-readable labels for the Appearance settings page.
PRESET_LABELS: dict[str, str] = {
    "jarvis": "JARVIS (gold)",
    "friday": "FRIDAY (cyan)",
    "monochrome": "Monochrome",
    "daylight": "Daylight",
}

# Which preset a persona selects when "theme follows persona" is enabled.
PERSONA_PRESETS: dict[str, str] = {
    "JARVIS": "jarvis",
    "FRIDAY": "friday",
}

DEFAULT_PRESET = "friday"


def preset(key: str | None) -> Theme:
    """Look up a preset by key, falling back to the default."""
    return PRESETS.get(str(key or "").strip().lower(), PRESETS[DEFAULT_PRESET])


def preset_for_persona(assistant_name: str | None) -> str:
    """Preset key a persona maps to."""
    return PERSONA_PRESETS.get(str(assistant_name or "").strip().upper(), DEFAULT_PRESET)


DEFAULT_METRICS = Metrics()

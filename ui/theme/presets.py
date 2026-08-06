"""Built-in themes.

JARVIS and FRIDAY are the persona themes. Monochrome and Light are
persona-neutral alternatives for users who do not want the film palette.

Three rules hold across every preset, and they are what separates this from
the previous palette:

1. Borders are NEUTRAL. They used to be `accent@44`, which put a faintly
   glowing gold or cyan edge on every panel in the app. A tinted hairline on
   every surface is the single most recognisable sci-fi-HUD signature, and it
   is the reason the interface read as a prop rather than a product.
2. Elevation is NEUTRAL. `surface`/`surface_alt`/`surface_hi` are white (or
   black, on light) at low alpha, so panels lift by luminance rather than by
   hue. This is also what lets them sit over the acrylic backdrop without
   fighting the wallpaper behind it.
3. Accent is spent on ONE thing per screen -- focus, the send button, an
   active toggle, the state dot. Everywhere else the accent's only job is to
   tell JARVIS from FRIDAY.

A consequence of (1)-(3) worth stating: the presets now differ almost solely
by accent. That is deliberate. The persona should change the app's signature
colour, not repaint every surface in the window.

`glow` survives on the Palette because OrbProfile consumers read it; nothing
in the stylesheet touches it.
"""

from __future__ import annotations

from PyQt6.QtGui import QColor

from ui.theme.tokens import Metrics, OrbProfile, Palette, Theme, with_alpha


def _c(value: str, alpha: int = 255) -> QColor:
    color = QColor(value)
    color.setAlpha(alpha)
    return color


def _w(alpha: int) -> QColor:
    """White at `alpha` -- neutral elevation for the dark themes."""
    return QColor(255, 255, 255, alpha)


def _k(alpha: int) -> QColor:
    """Black at `alpha` -- neutral elevation for the light theme."""
    return QColor(0, 0, 0, alpha)


def _dark_palette(accent: QColor, accent_dim: QColor, accent_hot: QColor,
                  glow: QColor) -> Palette:
    """The shared neutral dark ground. Only the accent family varies."""
    return Palette(
        bg=_c("#101014"),          # was #04070b: pure-black + neon was the tell
        surface=_w(12),            # panels lift by luminance, not by hue
        surface_alt=_w(18),        # fields, cards, the user's turn
        surface_hi=_w(28),         # hover
        border=_w(22),             # neutral hairline, never accent-tinted
        border_strong=_w(40),
        text=_c("#f4f4f6"),
        text_dim=_c("#a6a6b0"),
        text_faint=_c("#6f6f7a"),
        accent=accent,
        accent_dim=accent_dim,
        accent_hot=accent_hot,
        glow=glow,
        ok=_c("#6ccb5f"),
        warn=_c("#f7c948"),
        danger=_c("#e8505f"),
        is_dark=True,
    )


# ---------------------------------------------------------------------------
# JARVIS
#
# Warm gold and amber with a hot white core. Denser weave, slower and
# statelier motion than FRIDAY.
# ---------------------------------------------------------------------------

_JARVIS_ACCENT = _c("#f0a531")   # warmer and less neon than the old #ffb43c

JARVIS = Theme(
    name="JARVIS",
    palette=_dark_palette(
        accent=_JARVIS_ACCENT,
        accent_dim=_c("#c98b24"),
        accent_hot=_c("#ffdb9b"),
        glow=_c("#ff8a18"),
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

_FRIDAY_ACCENT = _c("#3aa0ff")   # Fluent blue, not the old neon cyan #38cfff

FRIDAY = Theme(
    name="FRIDAY",
    palette=_dark_palette(
        accent=_FRIDAY_ACCENT,
        accent_dim=_c("#1f6fbe"),
        accent_hot=_c("#cfe6ff"),
        glow=_c("#0a84ff"),
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

_MONO_ACCENT = _c("#c9ccd4")

MONOCHROME = Theme(
    name="Monochrome",
    palette=_dark_palette(
        accent=_MONO_ACCENT,
        accent_dim=_c("#8c949c"),
        accent_hot=_c("#ffffff"),
        glow=_c("#9fb0bf"),
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
# Light
#
# Was "Daylight", and was effectively a reskin of FRIDAY -- it reused FRIDAY's
# luminous cyan orb wholesale, which is what made it read as an afterthought.
# The palette is now a real Fluent light ground with its own accent and its
# own semantic colours, all darkened enough to hold contrast on white.
#
# The orb profile is still the inherited luminous one: OrbWidget paints a dark
# viewport disc behind the orb when `palette.is_dark` is False. Giving light a
# properly dark orb profile means reconciling that disc first, which is a
# separate change.
# ---------------------------------------------------------------------------

_LIGHT_ACCENT = _c("#0f6cbd")

LIGHT = Theme(
    name="Light",
    palette=Palette(
        bg=_c("#f3f3f3"),
        surface=_c("#ffffff", 180),
        surface_alt=_c("#ffffff", 230),
        surface_hi=_k(10),
        border=_k(24),
        border_strong=_k(46),
        text=_c("#1b1b1f"),
        text_dim=_c("#5c5c66"),
        text_faint=_c("#8b8b95"),
        accent=_LIGHT_ACCENT,
        accent_dim=_c("#0b4f8c"),
        accent_hot=_c("#2b88d8"),
        glow=_c("#0f6cbd"),
        ok=_c("#0f7b34"),
        warn=_c("#9d5d00"),
        danger=_c("#c4314b"),
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


# "daylight" is the old key for LIGHT. It stays resolvable because it may
# already be sitting in a user's ui_settings.json, but it is absent from
# PRESET_LABELS so the Appearance page offers only the current name.
PRESETS: dict[str, Theme] = {
    "jarvis": JARVIS,
    "friday": FRIDAY,
    "monochrome": MONOCHROME,
    "light": LIGHT,
    "daylight": LIGHT,
}

# Human-readable labels for the Appearance settings page. Order is the order
# the page presents them in.
PRESET_LABELS: dict[str, str] = {
    "jarvis": "JARVIS (gold)",
    "friday": "FRIDAY (blue)",
    "monochrome": "Monochrome",
    "light": "Light",
}

# Backwards-compatible alias: DAYLIGHT was the public name of this theme.
DAYLIGHT = LIGHT

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

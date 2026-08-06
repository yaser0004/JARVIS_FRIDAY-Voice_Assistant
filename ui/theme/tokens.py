"""Design tokens for the JARVIS/FRIDAY interface.

Everything visual resolves from a single `Theme` value: the generated
stylesheet, the custom-painted widgets, and the icon factory all read the
same tokens, so a theme switch is one assignment plus a repaint.

Colors are stored as `QColor` because the painting code vastly outnumbers
the stylesheet code. Use `css()` when a token needs to reach QSS.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

from PyQt6.QtGui import QColor


# ---------------------------------------------------------------------------
# color helpers
# ---------------------------------------------------------------------------


def with_alpha(color: QColor, alpha: int) -> QColor:
    """Copy of `color` with an explicit 0-255 alpha."""
    out = QColor(color)
    out.setAlpha(max(0, min(255, int(alpha))))
    return out


def fade(color: QColor, factor: float) -> QColor:
    """Copy of `color` scaled to `factor` (0.0 - 1.0) of full opacity."""
    return with_alpha(color, int(round(max(0.0, min(1.0, factor)) * 255)))


def mix(a: QColor, b: QColor, t: float) -> QColor:
    """Linear blend from `a` (t=0) to `b` (t=1), alpha included."""
    t = max(0.0, min(1.0, t))
    return QColor(
        int(round(a.red() + (b.red() - a.red()) * t)),
        int(round(a.green() + (b.green() - a.green()) * t)),
        int(round(a.blue() + (b.blue() - a.blue()) * t)),
        int(round(a.alpha() + (b.alpha() - a.alpha()) * t)),
    )


def shift(color: QColor, *, hue: float = 0.0, sat: float = 1.0, val: float = 1.0) -> QColor:
    """Rotate hue by degrees and scale saturation/value, clamped."""
    h, s, v, a = color.getHsvF()
    if h < 0:
        h = 0.0
    h = (h + hue / 360.0) % 1.0
    s = max(0.0, min(1.0, s * sat))
    v = max(0.0, min(1.0, v * val))
    out = QColor()
    out.setHsvF(h, s, v, a)
    return out


def css(color: QColor, alpha: int | None = None) -> str:
    """Qt stylesheet `rgba(...)` literal for a token."""
    a = color.alpha() if alpha is None else max(0, min(255, int(alpha)))
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {a})"


def hexs(color: QColor) -> str:
    """Opaque `#rrggbb` literal, for stylesheet properties that reject rgba."""
    return f"#{color.red():02x}{color.green():02x}{color.blue():02x}"


# ---------------------------------------------------------------------------
# token groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    """Surface, text and accent colors."""

    bg: QColor           # window backdrop
    surface: QColor      # panel fill
    surface_alt: QColor  # nested / inset fill
    surface_hi: QColor   # hover fill
    border: QColor
    border_strong: QColor

    text: QColor
    text_dim: QColor
    text_faint: QColor

    accent: QColor
    accent_dim: QColor
    accent_hot: QColor
    glow: QColor

    ok: QColor
    warn: QColor
    danger: QColor

    is_dark: bool = True

    def with_accent(self, accent: QColor) -> "Palette":
        """Rederive the whole accent family from a single user-chosen hue."""
        return replace(
            self,
            accent=QColor(accent),
            accent_dim=shift(accent, sat=0.86, val=0.72),
            accent_hot=shift(accent, sat=0.30, val=1.0),
            glow=shift(accent, sat=1.0, val=0.85),
        )


@dataclass(frozen=True)
class OrbProfile:
    """Identity of the orb for one persona.

    Structure follows the Age of Ultron JARVIS hologram: an internal network
    of connections inside a spherical center, circumscribed by audio-reactive
    exterior rings of data.
    """

    core: QColor
    filament_a: QColor
    filament_b: QColor
    bloom: QColor
    ring: QColor

    filaments: int = 26
    points: int = 72       # samples per filament
    rings: int = 3
    ticks_per_ring: int = 48

    spin_rate: float = 14.0   # yaw, degrees per second
    tilt_rate: float = 9.0    # pitch, degrees per second
    ring_rate: float = 26.0   # ring rotation, degrees per second
    tape_rate: float = 0.06   # reel-to-reel tick scroll, revolutions per second


@dataclass(frozen=True)
class Metrics:
    radius_sm: int = 6
    radius_md: int = 10
    radius_lg: int = 14
    radius_pill: int = 999

    space_xs: int = 4
    space_sm: int = 8
    space_md: int = 12
    space_lg: int = 18
    space_xl: int = 26

    border_w: int = 1
    control_h: int = 34
    icon: int = 18


@dataclass(frozen=True)
class Typography:
    """Resolved font families. Populated at startup from what is installed."""

    display: str = "Bahnschrift"
    body: str = "Segoe UI"
    mono: str = "Cascadia Mono"

    size_xs: int = 9
    size_sm: int = 10
    size_md: int = 11
    size_lg: int = 13
    size_xl: int = 17

    # Preference order, first installed family wins. Resolved by ui.theme.fonts.
    display_stack: Sequence[str] = field(
        default_factory=lambda: ("Bahnschrift", "Segoe UI Semibold", "Segoe UI", "Arial")
    )
    body_stack: Sequence[str] = field(
        default_factory=lambda: ("Segoe UI", "Bahnschrift", "Arial")
    )
    mono_stack: Sequence[str] = field(
        default_factory=lambda: ("Cascadia Mono", "Consolas", "Courier New")
    )


@dataclass(frozen=True)
class Theme:
    name: str
    palette: Palette
    orb: OrbProfile
    metrics: Metrics = field(default_factory=Metrics)
    type: Typography = field(default_factory=Typography)

    # convenience passthroughs, the painting code reads these constantly
    @property
    def c(self) -> Palette:
        return self.palette

    @property
    def m(self) -> Metrics:
        return self.metrics

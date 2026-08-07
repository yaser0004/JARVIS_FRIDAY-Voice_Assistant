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

    A glowing core inside a ring of radial bars whose lengths are the
    waveform. The persona is carried entirely by colour -- the geometry is
    identical for all of them.

    This used to describe a projected filament sphere and carried
    `filaments`, `points`, `rings` and `ticks_per_ring`. Those were read only
    inside a branch that could never execute, because the orb dropped to a
    degenerate fallback below 72px and was only ever instantiated at 28px.
    """

    core: QColor       # the centre
    wave_a: QColor     # bar gradient, one end
    wave_b: QColor     # bar gradient, the other end
    bloom: QColor      # outer glow
    ring: QColor       # the quiet baseline the bars stand on

    bars: int = 48
    spin_rate: float = 26.0   # degrees per second of ring drift


@dataclass(frozen=True)
class Metrics:
    """One geometry scale, held everywhere.

    The old values were 6/10/14 plus a one-off 20 on the composer, which is
    what made the surfaces look assembled rather than designed.
    """

    radius_sm: int = 4
    radius_md: int = 8
    radius_lg: int = 12
    radius_xl: int = 16      # the composer, the only genuinely large surface
    radius_pill: int = 999

    space_xs: int = 4
    space_sm: int = 8
    space_md: int = 12
    space_lg: int = 16
    space_xl: int = 24
    space_2xl: int = 32

    border_w: int = 1
    control_h: int = 32
    icon: int = 16

    # -- window backdrop ----------------------------------------------------
    # `acrylic_alpha` is the tint alpha to use once the DWM acrylic backdrop
    # is confirmed live. `backdrop_alpha` is the value actually painted, and
    # it stays fully opaque until the native chrome proves blur is active --
    # on Linux, when the window is merely translucent with nothing blurring
    # behind it, a 70%-opaque window over a sharp desktop looks broken.
    # Read it from the chrome's signal, never straight off this dataclass.
    acrylic_alpha: int = 178
    backdrop_alpha: int = 255

    # 0 on Windows 10, where the corner-rounding API does not exist and the
    # acrylic backdrop ignores SetWindowRgn. Raised when attr 33 succeeds.
    window_radius: int = 0


@dataclass(frozen=True)
class Typography:
    """Resolved font families. Populated at startup from what is installed.

    Display and body are deliberately the *same* family. The previous split
    (condensed Bahnschrift over Segoe UI) is what made section titles read as
    an instrument panel; one family at two weights reads as a designed system.
    IBM Plex ships in assets/fonts, so this is bundled rather than hoping the
    machine happens to have the face installed.
    """

    display: str = "IBM Plex Sans"
    body: str = "IBM Plex Sans"
    mono: str = "IBM Plex Mono"

    size_xs: int = 10
    size_sm: int = 11
    size_md: int = 12
    size_lg: int = 14
    size_xl: int = 20

    # Preference order, first installed family wins. Resolved by ui.theme.fonts.
    display_stack: Sequence[str] = field(
        default_factory=lambda: ("IBM Plex Sans", "Segoe UI Semibold", "Segoe UI", "Arial")
    )
    body_stack: Sequence[str] = field(
        default_factory=lambda: ("IBM Plex Sans", "Segoe UI", "Arial")
    )
    mono_stack: Sequence[str] = field(
        default_factory=lambda: ("IBM Plex Mono", "Cascadia Mono", "Consolas", "Courier New")
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

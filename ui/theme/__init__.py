"""Live theme access.

One `ThemeManager` owns the active `Theme`. Everything that draws reads it
through `theme()` at paint time rather than caching module constants, which
is what makes a runtime theme switch actually repaint the custom-painted
widgets (the orb, the waves, the chat bubbles, the window shell) and not
only the stylesheet-driven ones.

Listeners connect to `manager().changed`.
"""

from __future__ import annotations

from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import QApplication

from core.session_logging import trace_event
from ui.theme.fonts import font as _build_font
from ui.theme.fonts import load_bundled_fonts, resolve_typography
from ui.theme.presets import (
    DEFAULT_PRESET,
    PRESET_LABELS,
    PRESETS,
    preset,
    preset_for_persona,
)
from ui.theme.qss import build_qss
from ui.theme.tokens import (
    Metrics,
    OrbProfile,
    Palette,
    Theme,
    Typography,
    css,
    fade,
    hexs,
    mix,
    shift,
    with_alpha,
)
from ui.ui_settings import THEME_PRESET_AUTO, load_ui_settings, update_ui_settings


__all__ = [
    "Theme",
    "Palette",
    "OrbProfile",
    "Metrics",
    "Typography",
    "ThemeManager",
    "manager",
    "theme",
    "palette",
    "orb_profile",
    "metrics",
    "init_theme",
    "set_preset",
    "set_accent",
    "set_persona",
    "reapply",
    "display_font",
    "body_font",
    "mono_font",
    "css",
    "fade",
    "hexs",
    "mix",
    "shift",
    "with_alpha",
    "PRESETS",
    "PRESET_LABELS",
    "load_fonts",
]


class ThemeManager(QObject):
    """Holds the active theme and notifies the UI when it changes."""

    changed = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self._theme: Theme = preset(DEFAULT_PRESET)
        self._preset_key: str = THEME_PRESET_AUTO
        self._accent_override: str = ""
        self._persona: str = "FRIDAY"
        self._typography: Typography | None = None
        self._asset_urls: dict[str, str] = {}
        self._initialized = False

    # -- state ------------------------------------------------------------

    @property
    def theme(self) -> Theme:
        return self._theme

    @property
    def preset_key(self) -> str:
        return self._preset_key

    @property
    def accent_override(self) -> str:
        return self._accent_override

    @property
    def persona(self) -> str:
        return self._persona

    def resolved_preset_key(self) -> str:
        """The preset actually in effect, resolving "auto" through persona."""
        if self._preset_key == THEME_PRESET_AUTO:
            return preset_for_persona(self._persona)
        return self._preset_key

    # -- lifecycle --------------------------------------------------------

    def initialize(self, persona: str | None = None) -> Theme:
        """Load persisted preferences and resolve fonts. Needs a QApplication.

        Safe to call more than once: main.py initializes before building the
        window and the window bootstraps itself for the QA harness, which
        constructs it directly. Repeat calls must not discard in-session
        changes by re-reading the file.
        """
        if self._initialized:
            if persona:
                self.set_persona(persona)
            return self._theme

        load_bundled_fonts()
        self._typography = resolve_typography()

        settings = load_ui_settings()
        self._preset_key = str(settings.get("theme_preset") or THEME_PRESET_AUTO)
        self._accent_override = str(settings.get("accent_override") or "")
        if persona:
            self._persona = str(persona).strip().upper() or self._persona

        self._initialized = True
        self._rebuild(notify=False)
        trace_event(
            "ui.theme",
            "initialized",
            preset=self._preset_key,
            resolved=self.resolved_preset_key(),
            accent_override=self._accent_override,
            persona=self._persona,
        )
        return self._theme

    def set_indicator_assets(self, urls: dict[str, str]) -> None:
        """Register file URLs for combo/spin/check indicators, then restyle."""
        self._asset_urls = dict(urls or {})
        if self._initialized:
            self.apply_stylesheet()

    # -- mutation ---------------------------------------------------------

    def set_preset(self, key: str | None, *, persist: bool = True) -> None:
        new_key = str(key or THEME_PRESET_AUTO).strip().lower()
        if new_key != THEME_PRESET_AUTO and new_key not in PRESETS:
            return
        if new_key == self._preset_key:
            return
        self._preset_key = new_key
        if persist:
            update_ui_settings(theme_preset=new_key)
        self._rebuild()

    def set_accent(self, color: QColor | str | None, *, persist: bool = True) -> None:
        if color is None or color == "":
            value = ""
        else:
            qc = QColor(color)
            value = hexs(qc) if qc.isValid() else ""
        if value == self._accent_override:
            return
        self._accent_override = value
        if persist:
            update_ui_settings(accent_override=value)
        self._rebuild()

    def set_persona(self, assistant_name: str | None) -> None:
        """Persona only repaints when the preset is following it."""
        name = str(assistant_name or "").strip().upper()
        if not name or name == self._persona:
            return
        self._persona = name
        if self._preset_key == THEME_PRESET_AUTO:
            self._rebuild()

    # -- application ------------------------------------------------------

    def _rebuild(self, *, notify: bool = True) -> None:
        base = preset(self.resolved_preset_key())

        pal = base.palette
        if self._accent_override:
            override = QColor(self._accent_override)
            if override.isValid():
                pal = pal.with_accent(override)

        typography = self._typography or base.type
        self._theme = Theme(
            name=base.name,
            palette=pal,
            orb=base.orb,
            metrics=base.metrics,
            type=typography,
        )

        self._refresh_icons()
        self.apply_stylesheet()
        if notify:
            trace_event("ui.theme", "changed", resolved=self.resolved_preset_key())
            self.changed.emit()

    def _refresh_icons(self) -> None:
        """Recolor icons for the new palette and re-export QSS indicators.

        Imported here rather than at module scope: ui.icons resolves its
        default color through this module.
        """
        from ui import icons as icon_factory

        icon_factory.invalidate_cache()
        pal = self._theme.palette
        try:
            self._asset_urls = icon_factory.export_indicator_assets(
                {
                    "default": pal.text,
                    "chevron_down": pal.text_dim,
                    "chevron_down_dim": pal.text_faint,
                    "caret_up": pal.text_dim,
                    "caret_down": pal.text_dim,
                    # The checked indicator sits on an accent-filled box.
                    "check": pal.bg if pal.is_dark else pal.surface,
                }
            )
        except Exception as exc:
            trace_event("ui.theme", "indicator_export_failed", error=str(exc))
            self._asset_urls = {}

    def apply_stylesheet(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        app.setStyleSheet(build_qss(self._theme, self._asset_urls))


_manager: ThemeManager | None = None


def manager() -> ThemeManager:
    global _manager
    if _manager is None:
        _manager = ThemeManager()
    return _manager


def theme() -> Theme:
    """The active theme. Call this at paint time, do not cache it."""
    return manager().theme


def palette() -> Palette:
    return manager().theme.palette


def orb_profile() -> OrbProfile:
    return manager().theme.orb


def metrics() -> Metrics:
    return manager().theme.metrics


def init_theme(persona: str | None = None) -> Theme:
    return manager().initialize(persona)


def set_preset(key: str | None) -> None:
    manager().set_preset(key)


def set_accent(color: QColor | str | None) -> None:
    manager().set_accent(color)


def set_persona(assistant_name: str | None) -> None:
    manager().set_persona(assistant_name)


def reapply() -> None:
    manager().apply_stylesheet()


def on_theme_changed(slot: Callable[[], None]) -> None:
    manager().changed.connect(slot)


# ---------------------------------------------------------------------------
# font accessors
#
# These are functions, not module constants: the resolved families are not
# known until a QApplication exists, and they change when the theme does.
# ---------------------------------------------------------------------------


def display_font(size: int | None = None, *, weight: QFont.Weight = QFont.Weight.DemiBold,
                 tracking: float = 2.0, caps: bool = False) -> QFont:
    t = manager().theme.type
    return _build_font(t.display, size or t.size_md, weight=weight,
                       letter_spacing=tracking, caps=caps)


def body_font(size: int | None = None, *, weight: QFont.Weight = QFont.Weight.Normal) -> QFont:
    t = manager().theme.type
    return _build_font(t.body, size or t.size_md, weight=weight)


def mono_font(size: int | None = None, *, weight: QFont.Weight = QFont.Weight.Normal,
              tracking: float = 0.0) -> QFont:
    t = manager().theme.type
    return _build_font(t.mono, size or t.size_sm, weight=weight, letter_spacing=tracking)


def load_fonts() -> None:
    """Backward-compatible alias for the old ui.theme.load_fonts()."""
    load_bundled_fonts()

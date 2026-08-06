"""The empty state: what fills the window before there is a conversation.

Previously this was OrbStage, which owned a large hero orb and animated it
down into a compact one when the first message arrived. That morph is gone.
The orb is now a small live mark in the app bar, present at every moment of
the app's life rather than growing and shrinking, so the main column no
longer has to reflow around it and the layout is simply stable.

What remains here is a greeting and a quiet waveform, shown on an empty
conversation and hidden once there is a transcript to read.
"""

from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme import body_font, display_font, manager as theme_manager
from ui.widgets.waveform_widget import WaveformWidget


WAVE_HEIGHT = 54
WAVE_WIDTH = 320


class EmptyState(QWidget):
    """Greeting plus waveform, shown while the conversation is empty."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        self.greeting = QLabel("", self)
        self.greeting.setObjectName("greeting")
        self.greeting.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.greeting.setFont(display_font(26, weight=QFont.Weight.Medium))
        root.addWidget(self.greeting)

        # A greeting alone on an empty column reads as an unfinished screen.
        # One quiet line telling you what to do makes it read as a start.
        self.hint = QLabel("", self)
        self.hint.setObjectName("caption")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint.setFont(body_font(12))
        root.addWidget(self.hint)

        # The waves sit at a deliberately narrow measure. Run them full width
        # and they stop reading as part of the assistant and start reading as
        # a divider across the window.
        self.waves = WaveformWidget(self)
        self.waves.setMinimumHeight(0)
        self.waves.setFixedHeight(WAVE_HEIGHT)
        self.waves.setMaximumWidth(WAVE_WIDTH)
        wave_row = QHBoxLayout()
        wave_row.setContentsMargins(0, 0, 0, 0)
        wave_row.addStretch(1)
        wave_row.addWidget(self.waves)
        wave_row.addStretch(1)
        root.addLayout(wave_row)

        # Only the label is faded. A QGraphicsOpacityEffect renders its source
        # into an offscreen surface that a custom painter never reaches, so
        # putting one on WaveformWidget -- which paints itself -- would make
        # it render as nothing at all. The waves are hidden outright instead.
        self._greeting_effect = QGraphicsOpacityEffect(self.greeting)
        self.greeting.setGraphicsEffect(self._greeting_effect)
        self._greeting_effect.setOpacity(1.0)

        self._fade = QPropertyAnimation(self._greeting_effect, b"opacity", self)
        self._fade.setDuration(260)
        self._fade.setEasingCurve(QEasingCurve.Type.InOutCubic)

        theme_manager().changed.connect(self._apply_fonts)

    # -- state ------------------------------------------------------------

    def set_greeting(self, text: str, hint: str = "") -> None:
        self.greeting.setText(text)
        self.hint.setText(hint)
        self.hint.setVisible(bool(hint))

    def set_state(self, state: str) -> None:
        # State is spelled out once, in the app bar. Repeating it here was
        # redundant chrome.
        self.waves.set_state(str(state or "IDLE").strip().upper())

    def set_active(self, active: bool, animate: bool = True) -> None:
        """Show or hide the empty state as the conversation comes and goes."""
        active = bool(active)
        self._fade.stop()
        if not active:
            self.setVisible(False)
            self._greeting_effect.setOpacity(0.0)
            return

        self.setVisible(True)
        if not animate:
            self._greeting_effect.setOpacity(1.0)
            return
        self._fade.setStartValue(self._greeting_effect.opacity())
        self._fade.setEndValue(1.0)
        self._fade.start()

    def _apply_fonts(self) -> None:
        self.greeting.setFont(display_font(26, weight=QFont.Weight.Medium))

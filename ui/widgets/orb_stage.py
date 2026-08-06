"""The orb's home in the window.

Two presentations of the same element:

  hero      empty state. The orb is large and centred with a greeting under
            it, the way a new conversation looks in a modern assistant.
  compact   a conversation exists. The orb scales down and stays visible as
            a live presence rather than vanishing into a header mark.

The transition is animated on a single factor, so the orb, the waves and the
greeting all move together instead of snapping between two layouts.
"""

from __future__ import annotations

from PyQt6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    pyqtProperty,
)
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme import display_font, manager as theme_manager
from ui.widgets.orb import OrbWidget
from ui.widgets.waveform_widget import WaveformWidget


HERO_HEIGHT = 384
COMPACT_HEIGHT = 150
HERO_WAVE = 60
COMPACT_WAVE = 30
HERO_WAVE_WIDTH = 340
COMPACT_WAVE_WIDTH = 200


class OrbStage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._factor = 1.0  # 1.0 = hero, 0.0 = compact
        self._compact = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        self.orb = OrbWidget(self)
        # The stage owns the sizing; the orb scales to whatever it is given.
        self.orb.setMinimumSize(0, 0)
        self.orb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self.orb, 1)

        # The waves sit under the orb at a deliberately narrow measure. Run
        # full width and they stop reading as part of the orb and start
        # reading as a divider across the window.
        self.waves = WaveformWidget(self)
        self.waves.setMinimumHeight(0)
        self.waves.setFixedHeight(HERO_WAVE)
        self.waves.setMaximumWidth(HERO_WAVE_WIDTH)
        wave_row = QHBoxLayout()
        wave_row.setContentsMargins(0, 0, 0, 0)
        wave_row.addStretch(1)
        wave_row.addWidget(self.waves)
        wave_row.addStretch(1)
        root.addLayout(wave_row)

        self.greeting = QLabel("", self)
        self.greeting.setObjectName("greeting")
        self.greeting.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.greeting.setFont(display_font(22, weight=QFont.Weight.Medium))
        root.addWidget(self.greeting)

        self._greeting_effect = QGraphicsOpacityEffect(self.greeting)
        self.greeting.setGraphicsEffect(self._greeting_effect)
        self._greeting_effect.setOpacity(1.0)

        self._anim = QPropertyAnimation(self, b"stageFactor", self)
        self._anim.setDuration(420)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        self.setFixedHeight(HERO_HEIGHT)
        theme_manager().changed.connect(self._apply_fonts)

    # -- animated factor --------------------------------------------------

    @pyqtProperty(float)
    def stageFactor(self) -> float:
        return self._factor

    @stageFactor.setter
    def stageFactor(self, value: float) -> None:
        self._factor = max(0.0, min(1.0, float(value)))
        height = COMPACT_HEIGHT + (HERO_HEIGHT - COMPACT_HEIGHT) * self._factor
        wave = COMPACT_WAVE + (HERO_WAVE - COMPACT_WAVE) * self._factor
        wave_w = COMPACT_WAVE_WIDTH + (HERO_WAVE_WIDTH - COMPACT_WAVE_WIDTH) * self._factor
        self.setFixedHeight(int(round(height)))
        self.waves.setFixedHeight(int(round(wave)))
        self.waves.setMaximumWidth(int(round(wave_w)))
        self._greeting_effect.setOpacity(self._factor)
        # Collapsing the label rather than only fading it keeps the compact
        # layout from carrying an invisible band of empty space.
        self.greeting.setVisible(self._factor > 0.02)

    # -- state ------------------------------------------------------------

    def set_compact(self, compact: bool, animate: bool = True) -> None:
        compact = bool(compact)
        if compact == self._compact:
            return
        self._compact = compact
        target = 0.0 if compact else 1.0

        self._anim.stop()
        if not animate:
            self.stageFactor = target
            return
        self._anim.setStartValue(self._factor)
        self._anim.setEndValue(target)
        self._anim.start()

    @property
    def compact(self) -> bool:
        return self._compact

    def set_greeting(self, text: str) -> None:
        self.greeting.setText(text)

    def set_state(self, state: str) -> None:
        # State is spelled out once, in the app bar. Repeating it under the
        # orb was redundant chrome.
        normalized = str(state or "IDLE").strip().upper()
        self.orb.set_state(normalized)
        self.waves.set_state(normalized)

    def _apply_fonts(self) -> None:
        self.greeting.setFont(display_font(22, weight=QFont.Weight.Medium))

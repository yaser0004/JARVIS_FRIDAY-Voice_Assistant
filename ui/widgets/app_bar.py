"""The top bar.

Carries identity on the left and the few controls that are not part of the
conversation on the right. Everything else that used to live in a permanent
sidebar is now behind the diagnostics toggle.

In stage 5 this also becomes the window's drag handle, so the layout leaves
room for window buttons on the right.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QPainter
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QToolButton, QWidget

from ui.icons import icon, pixmap
from ui.theme import display_font, manager as theme_manager, mono_font, theme, with_alpha


class StateDot(QWidget):
    """A small live indicator that carries state without spelling it out."""

    COLORS = {
        "IDLE": "text_faint",
        "LISTENING": "accent",
        "PROCESSING": "warn",
        "SPEAKING": "ok",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self._state = "IDLE"

    def set_state(self, state: str) -> None:
        self._state = str(state or "IDLE").strip().upper()
        self.update()

    def paintEvent(self, _event) -> None:
        palette = theme().palette
        color = getattr(palette, self.COLORS.get(self._state, "text_faint"))
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(with_alpha(color, 60))
        painter.drawEllipse(self.rect())
        painter.setBrush(color)
        painter.drawEllipse(self.rect().adjusted(3, 3, -3, -3))


class AppBar(QWidget):
    settings_requested = pyqtSignal()
    logs_requested = pyqtSignal()
    diagnostics_toggled = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("appBar")
        self.setFixedHeight(46)

        row = QHBoxLayout(self)
        row.setContentsMargins(14, 0, 10, 0)
        row.setSpacing(10)

        self.mark = QLabel(self)
        self.mark.setFixedSize(20, 20)
        row.addWidget(self.mark)

        self.brand = QLabel("JARVIS", self)
        self.brand.setObjectName("brand")
        self.brand.setFont(display_font(12, tracking=13.0))
        row.addWidget(self.brand)

        self.dot = StateDot(self)
        row.addWidget(self.dot)

        self.state_label = QLabel("idle", self)
        self.state_label.setObjectName("caption")
        self.state_label.setFont(mono_font(9, tracking=6.0))
        row.addWidget(self.state_label)

        row.addStretch(1)

        self.diagnostics_btn = QToolButton(self)
        self.diagnostics_btn.setObjectName("barButton")
        self.diagnostics_btn.setToolTip("Diagnostics")
        self.diagnostics_btn.setCheckable(True)
        self.diagnostics_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.diagnostics_btn.setFixedSize(32, 32)
        self.diagnostics_btn.toggled.connect(self.diagnostics_toggled.emit)
        row.addWidget(self.diagnostics_btn)

        self.logs_btn = QToolButton(self)
        self.logs_btn.setObjectName("barButton")
        self.logs_btn.setToolTip("Logs")
        self.logs_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.logs_btn.setFixedSize(32, 32)
        self.logs_btn.clicked.connect(self.logs_requested.emit)
        row.addWidget(self.logs_btn)

        self.settings_btn = QToolButton(self)
        self.settings_btn.setObjectName("barButton")
        self.settings_btn.setToolTip("Settings")
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.setFixedSize(32, 32)
        self.settings_btn.clicked.connect(self.settings_requested.emit)
        row.addWidget(self.settings_btn)

        # Reserved for the window buttons added in the frameless stage.
        self.window_controls = QWidget(self)
        controls = QHBoxLayout(self.window_controls)
        controls.setContentsMargins(6, 0, 0, 0)
        controls.setSpacing(2)
        row.addWidget(self.window_controls)

        self._apply_theme()
        theme_manager().changed.connect(self._apply_theme)

    def _apply_theme(self) -> None:
        active = theme()
        self.mark.setPixmap(pixmap("brand", active.palette.accent, 20,
                                   self.devicePixelRatioF()))
        self.brand.setFont(display_font(12, tracking=13.0))
        self.diagnostics_btn.setIcon(icon("search"))
        self.logs_btn.setIcon(icon("logs"))
        self.settings_btn.setIcon(icon("settings"))
        for button in (self.diagnostics_btn, self.logs_btn, self.settings_btn):
            button.setIconSize(QSize(17, 17))
        self.dot.update()

    def set_brand_name(self, name: str) -> None:
        self.brand.setText(str(name or "JARVIS").strip().upper() or "JARVIS")

    def set_state(self, state: str) -> None:
        normalized = str(state or "IDLE").strip().upper()
        self.dot.set_state(normalized)
        self.state_label.setText(normalized.lower())

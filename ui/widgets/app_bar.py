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
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QToolButton, QWidget

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


#: Windows convention is a wide, short window button with no gaps between them.
WINDOW_BUTTON_W = 44
WINDOW_BUTTON_H = 32
BAR_HEIGHT = 44
MARK_SIZE = 28


class AppBar(QWidget):
    settings_requested = pyqtSignal()
    logs_requested = pyqtSignal()
    diagnostics_toggled = pyqtSignal(bool)
    minimize_requested = pyqtSignal()
    maximize_requested = pyqtSignal()
    close_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("appBar")
        self.setFixedHeight(BAR_HEIGHT)

        row = QHBoxLayout(self)
        # No right margin: the close button must reach the window edge, which
        # is what makes the corner a real Fitts's-law target.
        row.setContentsMargins(14, 0, 0, 0)
        row.setSpacing(10)

        # Identity only. The live orb moved out to sit above the composer,
        # where it is large enough to actually read -- at 28px in here it was
        # a coloured smudge nobody could find.
        self.mark = QLabel(self)
        self.mark.setFixedSize(MARK_SIZE, MARK_SIZE)
        row.addWidget(self.mark)

        self.brand = QLabel("JARVIS", self)
        self.brand.setObjectName("brand")
        self.brand.setFont(display_font(13, tracking=7.0))
        row.addWidget(self.brand)

        self.dot = StateDot(self)
        row.addWidget(self.dot)

        self.state_label = QLabel("idle", self)
        self.state_label.setObjectName("caption")
        self.state_label.setFont(mono_font(10))
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

        # Window buttons. Hidden until the native chrome actually installs --
        # with an OS titlebar still present these would be a second, redundant
        # set of controls.
        self.window_controls = QWidget(self)
        self.window_controls.setProperty("windowDragExcluded", True)
        controls = QHBoxLayout(self.window_controls)
        controls.setContentsMargins(8, 0, 0, 0)
        controls.setSpacing(0)

        self.minimize_btn = QToolButton(self.window_controls)
        self.minimize_btn.setObjectName("windowButton")
        self.minimize_btn.setToolTip("Minimize")
        self.minimize_btn.setFixedSize(WINDOW_BUTTON_W, WINDOW_BUTTON_H)
        self.minimize_btn.clicked.connect(self.minimize_requested.emit)
        controls.addWidget(self.minimize_btn)

        self.maximize_btn = QToolButton(self.window_controls)
        self.maximize_btn.setObjectName("windowButton")
        self.maximize_btn.setToolTip("Maximize")
        self.maximize_btn.setFixedSize(WINDOW_BUTTON_W, WINDOW_BUTTON_H)
        self.maximize_btn.clicked.connect(self.maximize_requested.emit)
        controls.addWidget(self.maximize_btn)

        self.close_btn = QToolButton(self.window_controls)
        self.close_btn.setObjectName("windowClose")
        self.close_btn.setToolTip("Close")
        self.close_btn.setFixedSize(WINDOW_BUTTON_W, WINDOW_BUTTON_H)
        self.close_btn.clicked.connect(self.close_requested.emit)
        controls.addWidget(self.close_btn)

        self.window_controls.setVisible(False)
        row.addWidget(self.window_controls)

        # The bar is the window's drag handle, so anything clickable on it has
        # to opt out or it swallows the drag. Property rather than isinstance:
        # the bar will keep gaining widgets and draggable is the right default.
        for button in (self.diagnostics_btn, self.logs_btn, self.settings_btn):
            button.setProperty("windowDragExcluded", True)

        self._maximized = False
        self._apply_theme()
        theme_manager().changed.connect(self._apply_theme)

    # -- window controls ---------------------------------------------------

    def set_window_controls_visible(self, visible: bool) -> None:
        """Shown only when the native frameless chrome actually installed."""
        visible = bool(visible)
        self.window_controls.setVisible(visible)
        # The close button has to reach the window edge to be a real corner
        # target, but with the controls hidden the settings button would end
        # up flush against it instead.
        layout = self.layout()
        if layout is not None:
            layout.setContentsMargins(14, 0, 0 if visible else 10, 0)

    def set_maximized(self, maximized: bool) -> None:
        maximized = bool(maximized)
        if maximized == self._maximized:
            return
        self._maximized = maximized
        self.maximize_btn.setToolTip("Restore" if maximized else "Maximize")
        self.maximize_btn.setIcon(icon("restore" if maximized else "maximize"))

    def set_state(self, state: str) -> None:
        normalized = str(state or "IDLE").strip().upper()
        self.dot.set_state(normalized)
        self.state_label.setText(normalized.lower())

    def _apply_theme(self) -> None:
        active = theme()
        self.mark.setPixmap(pixmap("brand", active.palette.accent, MARK_SIZE,
                                   self.devicePixelRatioF()))
        self.brand.setFont(display_font(13, tracking=7.0))
        self.diagnostics_btn.setIcon(icon("search"))
        self.logs_btn.setIcon(icon("logs"))
        self.settings_btn.setIcon(icon("settings"))
        for button in (self.diagnostics_btn, self.logs_btn, self.settings_btn):
            button.setIconSize(QSize(17, 17))

        self.minimize_btn.setIcon(icon("minimize"))
        self.maximize_btn.setIcon(icon("restore" if self._maximized else "maximize"))
        self.close_btn.setIcon(icon("close"))
        for button in (self.minimize_btn, self.maximize_btn, self.close_btn):
            button.setIconSize(QSize(11, 11))
        self.dot.update()

    def set_brand_name(self, name: str) -> None:
        self.brand.setText(str(name or "JARVIS").strip().upper() or "JARVIS")

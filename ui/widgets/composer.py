"""The input composer.

One rounded surface holding the text field and its controls, rather than six
separate widgets sitting in a row. The buttons are flat and quiet inside it
so the field reads as a single object.

It owns the controls but exposes them by name, because the QA harness drives
`window.input_box`, `window.send_btn`, `window.mic_btn` and `window.plus_btn`
directly and those names must survive.
"""

from __future__ import annotations

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ui.icons import icon
from ui.theme import body_font, manager as theme_manager


class Composer(QFrame):
    submitted = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("composer")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(4)

        # Attachment chip slot. Populated in stage 7; kept here so the
        # attachment sits inside the composer surface rather than floating
        # above it.
        self.attachment_slot = QWidget(self)
        slot_layout = QHBoxLayout(self.attachment_slot)
        slot_layout.setContentsMargins(0, 0, 0, 0)
        slot_layout.setSpacing(6)
        self.attachment_slot.setVisible(False)
        root.addWidget(self.attachment_slot)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        self.plus_btn = QToolButton(self)
        self.plus_btn.setObjectName("composerButton")
        self.plus_btn.setToolTip("Attach an image")
        self.plus_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.plus_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.plus_btn.setFixedSize(34, 34)
        self.plus_btn.setEnabled(False)

        self.web_toggle_btn = QToolButton(self)
        self.web_toggle_btn.setObjectName("composerToggle")
        self.web_toggle_btn.setToolTip("Web overview: OFF")
        self.web_toggle_btn.setCheckable(True)
        self.web_toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.web_toggle_btn.setFixedSize(34, 34)
        self.web_toggle_btn.setEnabled(False)

        self.input_box = QLineEdit(self)
        self.input_box.setObjectName("composerInput")
        self.input_box.setPlaceholderText("Ask JARVIS anything")
        self.input_box.setFont(body_font(12))
        self.input_box.setEnabled(False)
        self.input_box.returnPressed.connect(self.submitted.emit)

        self.mic_btn = QPushButton("", self)
        self.mic_btn.setObjectName("composerButton")
        self.mic_btn.setToolTip("Hold to talk")
        self.mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.mic_btn.setFixedSize(34, 34)
        self.mic_btn.setEnabled(False)

        self.send_btn = QPushButton("", self)
        self.send_btn.setObjectName("composerSend")
        self.send_btn.setToolTip("Send")
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.send_btn.setFixedSize(34, 34)
        self.send_btn.setEnabled(False)

        row.addWidget(self.plus_btn)
        row.addWidget(self.web_toggle_btn)
        row.addWidget(self.input_box, 1)
        row.addWidget(self.mic_btn)
        row.addWidget(self.send_btn)
        root.addLayout(row)

        self._apply_icons()
        theme_manager().changed.connect(self._apply_icons)

    def _apply_icons(self) -> None:
        self.plus_btn.setIcon(icon("attach"))
        self.web_toggle_btn.setIcon(icon("globe"))
        for button in (self.plus_btn, self.web_toggle_btn, self.mic_btn, self.send_btn):
            button.setIconSize(QSize(18, 18))

    def set_attachment_widget(self, widget: QWidget | None) -> None:
        layout = self.attachment_slot.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget() is not None:
                item.widget().setParent(None)
        if widget is not None:
            layout.addWidget(widget)
        self.attachment_slot.setVisible(widget is not None)

    def set_placeholder(self, text: str) -> None:
        self.input_box.setPlaceholderText(text)

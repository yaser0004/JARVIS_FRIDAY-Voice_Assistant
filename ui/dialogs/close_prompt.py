"""What to do when the window is closed.

Was a raw QMessageBox with three custom buttons, which is the least polished
thing the app put on screen: a system icon, a bold primary line, a smaller
informative line, and buttons in whatever order the platform felt like.

The return contract is load-bearing -- JarvisMainWindow.closeEvent branches on
these three exact strings, and anything else makes the app either unclosable
or able to exit without shutting the pipeline down. Dismissing the dialog by
any route must mean "cancel", never "close".
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ui.native import apply_dark_titlebar
from ui.theme import body_font, display_font, theme


BACKGROUND = "background"
CLOSE = "close"
CANCEL = "cancel"


class ClosePrompt(QDialog):
    def __init__(self, assistant_name: str = "JARVIS", parent=None) -> None:
        super().__init__(parent)
        self._choice = CANCEL          # any dismissal means cancel
        self._name = str(assistant_name or "JARVIS")

        self.setWindowTitle(f"Close {self._name}")
        self.setModal(True)
        self.setMinimumWidth(420)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 22, 24, 18)
        root.setSpacing(8)

        heading = QLabel(f"Close {self._name}?", self)
        heading.setObjectName("heading")
        heading.setFont(display_font(16, weight=QFont.Weight.DemiBold))
        root.addWidget(heading)

        body = QLabel(
            f"{self._name} can keep listening from the system tray, or shut "
            f"down completely and release the microphone and model.",
            self,
        )
        body.setObjectName("caption")
        body.setWordWrap(True)
        body.setFont(body_font(12))
        root.addWidget(body)

        root.addSpacing(12)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)

        cancel_btn = QPushButton("Cancel", self)
        cancel_btn.setObjectName("ghost")
        cancel_btn.clicked.connect(lambda: self._finish(CANCEL))
        buttons.addWidget(cancel_btn)

        buttons.addStretch(1)

        quit_btn = QPushButton("Close completely", self)
        quit_btn.setObjectName("danger")
        quit_btn.clicked.connect(lambda: self._finish(CLOSE))
        buttons.addWidget(quit_btn)

        background_btn = QPushButton("Run in background", self)
        background_btn.setObjectName("primary")
        background_btn.setDefault(True)
        background_btn.setAutoDefault(True)
        background_btn.clicked.connect(lambda: self._finish(BACKGROUND))
        buttons.addWidget(background_btn)

        root.addLayout(buttons)
        background_btn.setFocus()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        apply_dark_titlebar(self, theme().palette.is_dark)

    def _finish(self, choice: str) -> None:
        self._choice = choice
        if choice == CANCEL:
            self.reject()
        else:
            self.accept()

    def choice(self) -> str:
        return self._choice

    @classmethod
    def ask(cls, assistant_name: str, parent=None) -> str:
        """Run the prompt and return one of background / close / cancel."""
        dialog = cls(assistant_name, parent)
        dialog.exec()
        return dialog.choice()

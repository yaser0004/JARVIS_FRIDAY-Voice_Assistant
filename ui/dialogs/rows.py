"""Building blocks for the settings pages.

One row type and one card type, used for every setting on every page. That
uniformity is what makes the window read as a single system rather than as a
pile of ad-hoc widgets, and it is the difference between the old flat column
and something that looks designed.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme import body_font, display_font


class SettingRow(QFrame):
    """A title, an optional description, and a control on the right.

    `stacked=True` puts the control on its own line underneath, for controls
    that need the full width (a phrase list, a long readout).
    """

    def __init__(self, title: str, control: QWidget | None = None,
                 description: str = "", stacked: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("settingRow")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 11, 14, 11)
        outer.setSpacing(8)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(16)

        text_column = QVBoxLayout()
        text_column.setContentsMargins(0, 0, 0, 0)
        text_column.setSpacing(2)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("rowTitle")
        self.title_label.setFont(body_font(12))
        self.title_label.setWordWrap(True)
        text_column.addWidget(self.title_label)

        self.description_label = QLabel(description, self)
        self.description_label.setObjectName("rowDescription")
        self.description_label.setFont(body_font(11))
        self.description_label.setWordWrap(True)
        self.description_label.setVisible(bool(description))
        text_column.addWidget(self.description_label)

        top.addLayout(text_column, 1)

        self.control = control
        if control is not None and not stacked:
            control.setSizePolicy(QSizePolicy.Policy.Preferred,
                                  QSizePolicy.Policy.Preferred)
            top.addWidget(control, 0, Qt.AlignmentFlag.AlignVCenter)

        outer.addLayout(top)

        if control is not None and stacked:
            outer.addWidget(control)

    def set_description(self, text: str) -> None:
        self.description_label.setText(text)
        self.description_label.setVisible(bool(text))


class SettingsCard(QFrame):
    """A group of rows on one surface, hairline-separated."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("settingsCard")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._rows = 0

    def add(self, widget: QWidget) -> QWidget:
        if self._rows:
            divider = QFrame(self)
            divider.setObjectName("rowDivider")
            divider.setFixedHeight(1)
            self._layout.addWidget(divider)
        self._layout.addWidget(widget)
        self._rows += 1
        return widget

    def add_row(self, title: str, control: QWidget | None = None,
                description: str = "", stacked: bool = False) -> SettingRow:
        return self.add(SettingRow(title, control, description, stacked, self))


class SettingsPage(QWidget):
    """A titled column of cards."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(28, 24, 28, 24)
        self._layout.setSpacing(16)

        self.title_label = QLabel(title, self)
        self.title_label.setObjectName("pageTitle")
        self.title_label.setFont(display_font(19, weight=QFont.Weight.DemiBold))
        self._layout.addWidget(self.title_label)

    def add_card(self) -> SettingsCard:
        card = SettingsCard(self)
        self._layout.addWidget(card)
        return card

    def add_section(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setObjectName("sectionTitle")
        self._layout.addWidget(label)
        return label

    def add_widget(self, widget: QWidget) -> QWidget:
        self._layout.addWidget(widget)
        return widget

    def finish(self) -> None:
        self._layout.addStretch(1)

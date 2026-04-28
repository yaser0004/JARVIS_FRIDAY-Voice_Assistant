from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPropertyAnimation, QTimer, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QResizeEvent
from PyQt6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme import FONT_BODY, FONT_MONO


class ChatBubble(QWidget):
    def __init__(
        self,
        role: str,
        text: str,
        intent: Optional[str] = None,
        assistant_name: str = "JARVIS",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.role = role
        self.intent = intent

        normalized_assistant = str(assistant_name or "JARVIS").strip().upper() or "JARVIS"
        role_text = normalized_assistant if str(role).strip().lower() == "assistant" else role.upper()
        self.role_label = QLabel(role_text)
        self.role_label.setFont(FONT_MONO)
        self.role_label.setStyleSheet("color: rgba(144, 184, 194, 160);")

        self.message_label = QLabel(text)
        self.message_label.setWordWrap(True)
        self.message_label.setFont(FONT_BODY)
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.message_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.message_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

        self.intent_label = QLabel(intent or "")
        self.intent_label.setFont(FONT_MONO)
        self.intent_label.setStyleSheet("color: rgba(144, 184, 194, 130);")
        self.intent_label.setVisible(bool(intent))

        body = QVBoxLayout(self)
        body.setContentsMargins(12, 8, 12, 8)
        body.setSpacing(4)
        body.addWidget(self.role_label)
        body.addWidget(self.message_label)
        body.addWidget(self.intent_label)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._target_width = 0
        self.set_target_width(220)

        self.opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(0.0)

    def set_assistant_name(self, assistant_name: str) -> None:
        if str(self.role).strip().lower() != "assistant":
            return
        normalized_assistant = str(assistant_name or "JARVIS").strip().upper() or "JARVIS"
        self.role_label.setText(normalized_assistant)

    def set_target_width(self, width: int) -> None:
        target = max(160, int(width))
        if target == self._target_width:
            return
        self._target_width = target
        self.setFixedWidth(target)
        self.message_label.setFixedWidth(max(80, target - 24))
        self.updateGeometry()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect().adjusted(1, 1, -1, -1)
        if self.role == "user":
            fill = QColor(6, 48, 58, 210)
            border = QColor(0, 245, 255, 50)
        else:
            fill = QColor(6, 15, 24, 220)
            border = QColor(0, 245, 255, 100)

        painter.setBrush(fill)
        painter.setPen(QPen(border, 1))
        painter.drawRoundedRect(rect, 12, 12)
        super().paintEvent(event)


class ChatWidget(QScrollArea):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._assistant_name = "JARVIS"

        self.container = QWidget(self)
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(8)
        self.layout.addStretch(1)

        self.setWidget(self.container)
        self.max_messages = 50
        self._items: list[QWidget] = []
        self._stick_to_bottom = True
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bar = self.verticalScrollBar()
        bar.valueChanged.connect(self._on_scroll_value_changed)
        bar.rangeChanged.connect(self._on_scroll_range_changed)

    @staticmethod
    def _extract_bubble(row: QWidget) -> Optional[ChatBubble]:
        layout = row.layout()
        if layout is None:
            return None
        for idx in range(layout.count()):
            widget = layout.itemAt(idx).widget()
            if isinstance(widget, ChatBubble):
                return widget
        return None

    def _update_bubble_widths(self) -> None:
        viewport_width = max(220, self.viewport().width())
        usable = max(180, viewport_width - 18)
        target = min(480, int(usable * 0.84))
        for row in self._items:
            bubble = self._extract_bubble(row)
            if bubble is not None:
                bubble.set_target_width(target)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_bubble_widths()
        if self._stick_to_bottom:
            self._schedule_scroll_to_bottom()

    def add_message(self, role: str, text: str, intent: Optional[str] = None) -> None:
        bubble = ChatBubble(role, text, intent=intent, assistant_name=self._assistant_name)
        row = QWidget(self.container)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        if role == "user":
            row_layout.addStretch(1)
            row_layout.addWidget(bubble)
        else:
            row_layout.addWidget(bubble)
            row_layout.addStretch(1)

        self.layout.insertWidget(self.layout.count() - 1, row)
        self._items.append(row)
        self._update_bubble_widths()

        while len(self._items) > self.max_messages:
            old = self._items.pop(0)
            old.setParent(None)
            old.deleteLater()

        self.container.adjustSize()
        self._animate_row(row, role)
        self._schedule_scroll_to_bottom()

    def set_assistant_name(self, assistant_name: str) -> None:
        normalized_assistant = str(assistant_name or "JARVIS").strip().upper() or "JARVIS"
        if normalized_assistant == self._assistant_name:
            return
        self._assistant_name = normalized_assistant
        for row in self._items:
            bubble = self._extract_bubble(row)
            if bubble is not None:
                bubble.set_assistant_name(normalized_assistant)

    def _scroll_to_bottom(self) -> None:
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _on_scroll_value_changed(self, value: int) -> None:
        bar = self.verticalScrollBar()
        self._stick_to_bottom = value >= max(0, bar.maximum() - 12)

    def _on_scroll_range_changed(self, _minimum: int, _maximum: int) -> None:
        if self._stick_to_bottom:
            self._scroll_to_bottom()

    def _schedule_scroll_to_bottom(self) -> None:
        # Delay a few times so layout/animation changes cannot hide the final message.
        for delay_ms in (0, 40, 100, 200, 320):
            QTimer.singleShot(delay_ms, self._scroll_to_bottom)

    def _animate_row(self, row: QWidget, role: str) -> None:
        bubble = self._extract_bubble(row)
        if bubble is None:
            return
        opacity = bubble.graphicsEffect()
        fade = QPropertyAnimation(opacity, b"opacity", self)
        fade.setDuration(280)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.finished.connect(self._scroll_to_bottom)
        fade.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


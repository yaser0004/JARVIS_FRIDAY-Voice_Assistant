"""Conversation view.

Asymmetric by role: your turn sits in a soft rounded container, the
assistant's reply is plain text on the background under a small label. That
is what makes a long answer read as a document rather than as a chat log,
and it is why the previous symmetric bordered bubbles felt dated.

`ChatWidget.layout` stays a real `QVBoxLayout` attribute (shadowing
`QWidget.layout()`), because the QA harness counts rows with
`window.chat.layout.count()`.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPropertyAnimation, QSize, Qt, QTimer
from PyQt6.QtGui import QFont, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.theme import body_font, manager as theme_manager, mono_font


MAX_CONTENT_WIDTH = 720
USER_MAX_FRACTION = 0.74


class MessageTurn(QFrame):
    """One turn, styled by role through the stylesheet.

    Deliberately no custom paintEvent. A widget that both carries a
    QGraphicsOpacityEffect and paints itself renders as nothing, because the
    effect redirects painting to an offscreen surface that the custom painter
    never reaches. That combination silently swallowed every user turn.
    """

    def __init__(
        self,
        role: str,
        text: str,
        intent: Optional[str] = None,
        assistant_name: str = "JARVIS",
        image: Optional[QPixmap] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.role = str(role or "assistant").strip().lower()
        self.intent = intent
        self.is_user = self.role == "user"
        self.setObjectName("userTurn" if self.is_user else "assistantTurn")

        body = QVBoxLayout(self)
        body.setContentsMargins(*( (14, 10, 14, 11) if self.is_user else (2, 0, 2, 0) ))
        body.setSpacing(6)

        # The assistant is named; the user is not labelled at all, which is
        # what keeps the transcript from looking like a chat app.
        self.role_label: QLabel | None = None
        if not self.is_user:
            self.role_label = QLabel(str(assistant_name or "JARVIS").strip().upper())
            self.role_label.setObjectName("turnRole")
            self.role_label.setFont(body_font(10, weight=QFont.Weight.DemiBold))
            body.addWidget(self.role_label)

        # Attached image, shown in the turn that actually carried it.
        self.image_label: QLabel | None = None
        if image is not None and not image.isNull():
            self.image_label = QLabel(self)
            self.image_label.setObjectName("turnImage")
            self._source_pixmap = image
            self.image_label.setPixmap(
                image.scaled(
                    QSize(260, 200),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            body.addWidget(self.image_label,
                           alignment=Qt.AlignmentFlag.AlignRight if self.is_user
                           else Qt.AlignmentFlag.AlignLeft)

        self.message_label = QLabel(text)
        self.message_label.setWordWrap(True)
        self.message_label.setFont(body_font(12))
        self.message_label.setObjectName("turnText")
        self.message_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.message_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.message_label.setSizePolicy(QSizePolicy.Policy.Preferred,
                                         QSizePolicy.Policy.Preferred)
        body.addWidget(self.message_label)

        self.intent_label = QLabel(intent or "")
        self.intent_label.setObjectName("turnIntent")
        self.intent_label.setFont(mono_font(9))
        self.intent_label.setVisible(bool(intent) and not self.is_user)
        body.addWidget(self.intent_label)

        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._target_width = 0
        self.set_target_width(320)

        self.opacity = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity)
        self.opacity.setOpacity(0.0)

    def set_assistant_name(self, assistant_name: str) -> None:
        if self.role_label is None:
            return
        self.role_label.setText(str(assistant_name or "JARVIS").strip().upper())

    def set_target_width(self, width: int) -> None:
        target = max(180, int(width))
        if target == self._target_width:
            return
        self._target_width = target
        self.setFixedWidth(target)
        inset = 28 if self.is_user else 4
        self.message_label.setFixedWidth(max(120, target - inset))
        self.updateGeometry()



class ChatWidget(QScrollArea):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self._assistant_name = "JARVIS"

        self.container = QWidget(self)
        self.container.setObjectName("chatContainer")
        # Named `layout` deliberately: the QA harness reads
        # window.chat.layout.count() and would break on a rename.
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(0, 8, 0, 16)
        self.layout.setSpacing(20)
        self.layout.addStretch(1)

        self.setWidget(self.container)
        self.max_messages = 60
        self._items: list[QWidget] = []
        self._stick_to_bottom = True
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        bar = self.verticalScrollBar()
        bar.valueChanged.connect(self._on_scroll_value_changed)
        bar.rangeChanged.connect(self._on_scroll_range_changed)

        theme_manager().changed.connect(self._repaint_turns)

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _extract_turn(row: QWidget) -> Optional[MessageTurn]:
        layout = row.layout()
        if layout is None:
            return None
        for idx in range(layout.count()):
            widget = layout.itemAt(idx).widget()
            if isinstance(widget, MessageTurn):
                return widget
        return None

    def _repaint_turns(self) -> None:
        for row in self._items:
            row.update()
            turn = self._extract_turn(row)
            if turn is not None:
                turn.update()

    def _content_width(self) -> int:
        viewport = max(240, self.viewport().width())
        return min(MAX_CONTENT_WIDTH, viewport - 24)

    def _update_widths(self) -> None:
        content = self._content_width()
        for row in self._items:
            turn = self._extract_turn(row)
            if turn is None:
                continue
            turn.set_target_width(
                int(content * USER_MAX_FRACTION) if turn.is_user else content
            )

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_widths()
        if self._stick_to_bottom:
            self._schedule_scroll_to_bottom()

    # -- content ----------------------------------------------------------

    def add_message(self, role: str, text: str, intent: Optional[str] = None,
                    image: Optional[QPixmap] = None) -> None:
        turn = MessageTurn(role, text, intent=intent,
                           assistant_name=self._assistant_name, image=image)
        row = QWidget(self.container)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(0)

        # Content is centred to a comfortable measure and the user's turn is
        # pushed right inside it, rather than pinned to the window edge.
        pad = max(0, (max(240, self.viewport().width()) - self._content_width()) // 2)
        row_layout.addSpacing(pad)
        if turn.is_user:
            row_layout.addStretch(1)
            row_layout.addWidget(turn)
        else:
            row_layout.addWidget(turn)
            row_layout.addStretch(1)
        row_layout.addSpacing(pad)

        self.layout.insertWidget(self.layout.count() - 1, row)
        self._items.append(row)
        self._update_widths()

        while len(self._items) > self.max_messages:
            old = self._items.pop(0)
            old.setParent(None)
            old.deleteLater()

        self.container.adjustSize()
        self._animate_row(row)
        self._schedule_scroll_to_bottom()

    def clear_messages(self) -> None:
        for row in self._items:
            row.setParent(None)
            row.deleteLater()
        self._items = []

    def is_empty(self) -> bool:
        return not self._items

    def set_assistant_name(self, assistant_name: str) -> None:
        normalized = str(assistant_name or "JARVIS").strip().upper() or "JARVIS"
        if normalized == self._assistant_name:
            return
        self._assistant_name = normalized
        for row in self._items:
            turn = self._extract_turn(row)
            if turn is not None:
                turn.set_assistant_name(normalized)

    # -- scrolling --------------------------------------------------------

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

    def _animate_row(self, row: QWidget) -> None:
        turn = self._extract_turn(row)
        if turn is None:
            return
        effect = turn.graphicsEffect()
        fade = QPropertyAnimation(effect, b"opacity", self)
        fade.setDuration(260)
        fade.setStartValue(0.0)
        fade.setEndValue(1.0)
        fade.finished.connect(self._scroll_to_bottom)
        fade.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)

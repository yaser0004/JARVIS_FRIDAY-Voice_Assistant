from __future__ import annotations

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation
from PyQt6.QtWidgets import QWidget


def create_fade_animation(widget: QWidget, duration_ms: int = 320) -> QPropertyAnimation:
    animation = QPropertyAnimation(widget, b"windowOpacity")
    animation.setDuration(duration_ms)
    animation.setStartValue(0.0)
    animation.setEndValue(1.0)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    return animation


def create_slide_animation(
    widget: QWidget,
    start: QPoint,
    end: QPoint,
    duration_ms: int = 320,
) -> QPropertyAnimation:
    animation = QPropertyAnimation(widget, b"pos")
    animation.setDuration(duration_ms)
    animation.setStartValue(start)
    animation.setEndValue(end)
    animation.setEasingCurve(QEasingCurve.Type.OutCubic)
    return animation

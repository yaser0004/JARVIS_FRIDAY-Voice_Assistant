from __future__ import annotations

import math
import random
from dataclasses import dataclass

from PyQt6.QtCore import QElapsedTimer, QPropertyAnimation, QRectF, Qt, QTimer, pyqtProperty
from PyQt6.QtGui import QColor, QConicalGradient, QPainter, QPainterPath, QPen, QRadialGradient
from PyQt6.QtWidgets import QWidget

from ui.theme import CYAN, CYAN_DIM, GREEN, ORANGE


@dataclass
class Particle:
    radius: float
    angle: float
    speed: float
    size: float


class OrbWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self._state = "IDLE"
        self._scale = 1.0
        self.rotation = 0.0
        self.phase = 0.0
        self.audio_level = 0.0

        self.particles = [
            Particle(
                radius=random.uniform(68, 120),
                angle=random.uniform(0, math.tau),
                speed=random.uniform(0.6, 1.5),
                size=random.uniform(2, 5),
            )
            for _ in range(6)
        ]

        self._clock = QElapsedTimer()
        self._clock.start()
        self._last_tick_ms = self._clock.elapsed()

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.setTimerType(Qt.TimerType.CoarseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self.scale_anim = QPropertyAnimation(self, b"orbScale", self)
        self.scale_anim.setDuration(250)

    def _tick(self) -> None:
        if not self.isVisible():
            return

        now = self._clock.elapsed()
        dt = max(0.001, min((now - self._last_tick_ms) / 1000.0, 0.05))
        self._last_tick_ms = now

        self.rotation = (self.rotation + 52.0 * dt) % 360.0
        self.phase += 4.2 * dt
        for particle in self.particles:
            particle.angle = (particle.angle + particle.speed * dt) % math.tau
        self.update()

    def set_state(self, state: str) -> None:
        if state == self._state:
            return

        target = 1.1 if state == "LISTENING" else 1.0
        self.scale_anim.stop()
        self.scale_anim.setStartValue(self._scale)
        self.scale_anim.setEndValue(target)
        self.scale_anim.start()
        self._state = state

    def set_audio_level(self, level: float) -> None:
        self.audio_level = max(0.0, min(1.0, level))

    @pyqtProperty(float)
    def orbScale(self) -> float:
        return self._scale

    @orbScale.setter
    def orbScale(self, value: float) -> None:
        self._scale = value
        self.update()

    def _state_color(self) -> QColor:
        if self._state == "PROCESSING":
            return ORANGE
        if self._state == "SPEAKING":
            return GREEN
        if self._state == "LISTENING":
            return CYAN
        return CYAN_DIM

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)

        cx = self.width() / 2
        cy = self.height() / 2
        base_radius = min(self.width(), self.height()) * 0.22 * self._scale
        pulse = 1.0 + 0.03 * math.sin(self.phase)
        radius = base_radius * pulse
        color = self._state_color()

        glow = QRadialGradient(cx, cy, radius * 2.0)
        glow.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 100))
        glow.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 0))
        painter.setBrush(glow)
        painter.drawEllipse(QRectF(cx - radius * 2.0, cy - radius * 2.0, radius * 4.0, radius * 4.0))

        orb_gradient = QRadialGradient(cx, cy, radius)
        orb_gradient.setColorAt(0.0, QColor(color.red(), color.green(), color.blue(), 180))
        orb_gradient.setColorAt(1.0, QColor(color.red(), color.green(), color.blue(), 60))
        painter.setBrush(orb_gradient)
        painter.drawEllipse(QRectF(cx - radius, cy - radius, radius * 2, radius * 2))

        painter.save()
        painter.translate(cx, cy)
        painter.rotate(self.rotation)
        pen = QPen(QColor(color.red(), color.green(), color.blue(), 150), 2)
        pen.setDashPattern([4, 6])
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(-radius * 1.3, -radius * 1.3, radius * 2.6, radius * 2.6))
        painter.restore()

        painter.setBrush(QColor(color.red(), color.green(), color.blue(), 170))
        for particle in self.particles:
            px = cx + particle.radius * math.cos(particle.angle)
            py = cy + particle.radius * math.sin(particle.angle)
            painter.drawEllipse(QRectF(px, py, particle.size, particle.size))

        if self._state == "LISTENING":
            for i in range(3):
                growth = (self.phase * 20 + i * 40) % 120
                alpha = max(0, 120 - int(growth))
                pen = QPen(QColor(color.red(), color.green(), color.blue(), alpha), 2)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                r = radius + growth
                painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 255, 255, 160))
            bars = 10
            for i in range(bars):
                bar_h = (0.2 + self.audio_level + 0.35 * math.sin(self.phase * 2 + i)) * 15
                bw = 6
                gap = 4
                x = cx - ((bars * (bw + gap)) / 2) + i * (bw + gap)
                y = cy - bar_h / 2
                painter.drawRoundedRect(QRectF(x, y, bw, bar_h), 2, 2)

        if self._state == "PROCESSING":
            painter.save()
            painter.translate(cx, cy)
            painter.rotate(-self.rotation * 2)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            conic = QConicalGradient(0, 0, self.rotation)
            conic.setColorAt(0.0, QColor(255, 160, 100, 250))
            conic.setColorAt(0.5, QColor(255, 107, 43, 50))
            conic.setColorAt(1.0, QColor(255, 160, 100, 250))
            painter.setPen(QPen(conic, 5))
            painter.drawArc(QRectF(-radius * 1.55, -radius * 1.55, radius * 3.1, radius * 3.1), 30 * 16, 140 * 16)
            painter.drawArc(QRectF(-radius * 1.72, -radius * 1.72, radius * 3.44, radius * 3.44), 220 * 16, 100 * 16)
            painter.restore()

        if self._state == "SPEAKING":
            path = QPainterPath()
            points = 56
            for i in range(points + 1):
                theta = (i / points) * math.tau
                modulation = 1.0 + 0.06 * math.sin(theta * 8 + self.phase * 4) * (0.5 + self.audio_level)
                r = radius * 1.12 * modulation
                x = cx + r * math.cos(theta)
                y = cy + r * math.sin(theta)
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(0, 255, 136, 170), 3))
            painter.drawPath(path)

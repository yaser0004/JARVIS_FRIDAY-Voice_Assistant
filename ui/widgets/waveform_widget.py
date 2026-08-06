"""Status waveform.

Sits under the orb and carries the assistant's state as motion. The previous
version drew 40 flat lines and animated on a phase counter in three of its
four states, so the wave looked the same whether the assistant was thinking
or talking. Each state now has its own legible behaviour, and the widget
reports the live amplitude that drives the orb's ring ticks.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QLinearGradient, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import QWidget

from core.session_logging import trace_event
from ui import anim_clock
from ui.theme import manager as theme_manager, mix, theme, with_alpha


BARS = 48
FFT_HZ = 12.0  # analysis rate, deliberately decoupled from the frame rate


class WaveformWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(80)
        # Same reason as the orb: an opaque widget over the acrylic backdrop
        # punches a solid rectangle through the blur, so it composites now.
        # The shell repaints underneath it every frame as a result, which is
        # affordable because RoundedContainer's fill is a single rect.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

        self.bars = BARS
        self.heights = np.zeros(self.bars, dtype=np.float32)
        self.targets = np.zeros(self.bars, dtype=np.float32)
        self.state = "IDLE"
        self.phase = 0.0
        self.tts_level = 0.0
        self._level = 0.0
        self._bar_index = np.arange(self.bars, dtype=np.float32)
        self._fft_accum = 0.0
        self._sweep = 0.0

        self.sample_rate = 16000
        self.stream = None
        self.ring = deque(maxlen=2048)

        theme_manager().changed.connect(self.update)

    # -- lifecycle --------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        anim_clock.subscribe(self._tick)

    def hideEvent(self, event) -> None:
        anim_clock.unsubscribe(self._tick)
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        anim_clock.unsubscribe(self._tick)
        self._stop_mic_stream()
        super().closeEvent(event)

    # -- inputs -----------------------------------------------------------

    def set_state(self, state: str) -> None:
        self.state = str(state or "IDLE").strip().upper()
        if self.state == "LISTENING":
            self._start_mic_stream()
        else:
            self._stop_mic_stream()

    def set_tts_level(self, level: float) -> None:
        try:
            self.tts_level = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            self.tts_level = 0.0

    def level(self) -> float:
        """Current overall amplitude, 0-1. Feeds the orb's ring ticks."""
        return self._level

    # -- audio ------------------------------------------------------------

    def _start_mic_stream(self) -> None:
        if self.stream is not None:
            return

        def callback(indata, frames, _time, status):
            if status:
                return
            samples = np.squeeze(indata).astype(np.float32)
            self.ring.extend(samples)

        # This stream only feeds the amplitude animation; it must never be able
        # to crash a caller's state transition (e.g. hold-to-talk), which is
        # exactly what happened when the device was busy/unavailable and this
        # was unguarded — the caller's own capture never started, and its
        # button/flags were left stuck with nothing left to reset them.
        try:
            self.stream = sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=1024,
                callback=callback,
            )
            self.stream.start()
        except Exception as exc:
            self.stream = None
            trace_event("ui.waveform", "mic_stream_start_failed", error=str(exc))

    def _stop_mic_stream(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
        except Exception as exc:
            trace_event("ui.waveform", "mic_stream_stop_failed", error=str(exc))
        finally:
            self.stream = None

    def _fft_targets(self) -> np.ndarray:
        if len(self.ring) < self.bars * 2:
            return np.zeros(self.bars, dtype=np.float32)

        arr = np.fromiter(self.ring, dtype=np.float32)
        if arr.size > 1024:
            arr = arr[-1024:]
        spectrum = np.abs(np.fft.rfft(arr))
        if spectrum.size <= 1:
            return np.zeros(self.bars, dtype=np.float32)
        bins = np.array_split(spectrum[1:], self.bars)
        values = np.array([float(np.mean(chunk)) for chunk in bins], dtype=np.float32)
        values /= max(np.max(values), 1e-6)
        return values

    # -- animation --------------------------------------------------------

    def _tick(self, dt: float) -> None:
        if not self.isVisible():
            return

        self.phase += dt * 2.4
        self._sweep = (self._sweep + dt * 0.55) % 1.0

        if self.state == "LISTENING":
            # Analysis rate is held at FFT_HZ by accumulating dt. Tying it to
            # a frame counter silently doubled the FFT rate when the frame
            # interval went from 33 ms to 16 ms.
            self._fft_accum += dt
            if self._fft_accum >= 1.0 / FFT_HZ:
                self._fft_accum = 0.0
                self.targets = self._fft_targets()

        elif self.state == "SPEAKING":
            vals = (0.18 + self.tts_level * 0.85
                    + 0.3 * np.sin(self.phase * 2.6 + self._bar_index * 0.28))
            self.targets = np.clip(vals, 0.05, 1.0).astype(np.float32, copy=False)

        elif self.state == "PROCESSING":
            # A band of activity travelling across the field: reads clearly as
            # work in progress rather than as sound.
            centre = self._sweep * (self.bars + 16) - 8
            distance = (self._bar_index - centre) / 7.0
            vals = 0.06 + 0.85 * np.exp(-distance * distance)
            self.targets = vals.astype(np.float32, copy=False)

        else:
            vals = 0.05 + 0.03 * np.sin(self.phase * 0.7 + self._bar_index * 0.22)
            self.targets = vals.astype(np.float32, copy=False)

        # Frame-rate independent smoothing, so the settle time is the same
        # whether the clock is delivering 60 or 30 FPS.
        alpha = 1.0 - math.exp(-dt / 0.055)
        self.heights += (self.targets - self.heights) * alpha
        self._level = float(np.clip(self.heights.mean() * 1.8, 0.0, 1.0))
        self.update()

    # -- painting ---------------------------------------------------------

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        palette = theme().palette

        width, height = self.width(), self.height()
        # No background fill: an opaque rectangle here would punch a hole
        # through the acrylic backdrop. The parent's backdrop shows through.
        if width <= 2 or height <= 2:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        centre_y = height / 2.0
        max_height = height * 0.42
        bar_width = max(2.0, width / (self.bars * 1.9))
        spacing = bar_width * 0.55
        total = self.bars * bar_width + (self.bars - 1) * spacing
        start_x = (width - total) / 2.0

        accent = palette.accent
        hot = palette.accent_hot if self.state != "IDLE" else palette.accent_dim

        # Horizontal falloff so the field reads as one object with a centre,
        # rather than as a row of unrelated bars.
        gradient = QLinearGradient(start_x, 0, start_x + total, 0)
        gradient.setColorAt(0.0, with_alpha(accent, 40))
        gradient.setColorAt(0.5, with_alpha(mix(accent, hot, 0.5), 235))
        gradient.setColorAt(1.0, with_alpha(accent, 40))

        heights = self.heights.tolist()

        # Additive underglow: one soft mirrored silhouette behind the bars.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        glow = QPolygonF()
        for idx, amp in enumerate(heights):
            x = start_x + idx * (bar_width + spacing) + bar_width / 2.0
            glow.append(QPointF(x, centre_y - amp * max_height * 1.25))
        for idx in range(len(heights) - 1, -1, -1):
            x = start_x + idx * (bar_width + spacing) + bar_width / 2.0
            glow.append(QPointF(x, centre_y + heights[idx] * max_height * 1.25))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(with_alpha(palette.glow, 26))
        painter.drawPolygon(glow)

        # The bars themselves, mirrored around the centre line.
        pen = QPen(gradient, bar_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for idx, amp in enumerate(heights):
            x = start_x + idx * (bar_width + spacing) + bar_width / 2.0
            extent = max(0.6, amp * max_height)
            painter.drawLine(QPointF(x, centre_y - extent), QPointF(x, centre_y + extent))

        # A hairline baseline keeps the field anchored when everything is quiet.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setPen(QPen(with_alpha(palette.border, 90), 1))
        painter.drawLine(QPointF(start_x, centre_y), QPointF(start_x + total, centre_y))
        painter.end()

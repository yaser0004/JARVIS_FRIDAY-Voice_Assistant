from __future__ import annotations
from collections import deque

import numpy as np
import sounddevice as sd
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QWidget

from ui.theme import CYAN


class WaveformWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(80)
        self.bars = 40
        self.heights = np.zeros(self.bars, dtype=np.float32)
        self.targets = np.zeros(self.bars, dtype=np.float32)
        self.state = "IDLE"
        self.phase = 0.0
        self.tts_level = 0.0
        self._bar_index = np.arange(self.bars, dtype=np.float32)
        self._fft_frame_skip = 0
        self._idle_frame_skip = 0

        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.setTimerType(Qt.TimerType.CoarseTimer)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

        self.sample_rate = 16000
        self.stream = None
        self.ring = deque(maxlen=2048)

    def set_state(self, state: str) -> None:
        self.state = state
        if state == "LISTENING":
            self._start_mic_stream()
        else:
            self._stop_mic_stream()

    def set_tts_level(self, level: float) -> None:
        self.tts_level = max(0.0, min(1.0, level))

    def _start_mic_stream(self) -> None:
        if self.stream is not None:
            return

        def callback(indata, frames, _time, status):
            if status:
                return
            samples = np.squeeze(indata).astype(np.float32)
            self.ring.extend(samples)

        self.stream = sd.InputStream(
            channels=1,
            samplerate=self.sample_rate,
            blocksize=1024,
            callback=callback,
        )
        self.stream.start()

    def _stop_mic_stream(self) -> None:
        if self.stream is None:
            return
        try:
            self.stream.stop()
            self.stream.close()
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
        values = np.array([float(np.mean(bin_chunk)) for bin_chunk in bins], dtype=np.float32)
        values /= max(np.max(values), 1e-6)
        return values

    def _tick(self) -> None:
        self.phase += 0.08
        if self.state == "LISTENING":
            self._fft_frame_skip = (self._fft_frame_skip + 1) % 3
            if self._fft_frame_skip == 0:
                self.targets = self._fft_targets()
        elif self.state == "SPEAKING":
            vals = 0.2 + self.tts_level * 0.9 + 0.35 * np.sin(self.phase * 2.5 + self._bar_index * 0.3)
            self.targets = np.clip(vals, 0.05, 1.0).astype(np.float32, copy=False)
        else:
            self._idle_frame_skip = (self._idle_frame_skip + 1) % 2
            if self._idle_frame_skip != 0:
                return
            vals = 0.04 + 0.02 * np.sin(self.phase + self._bar_index * 0.18)
            self.targets = vals.astype(np.float32, copy=False)

        self.heights = (0.78 * self.heights) + (0.22 * self.targets)
        if self.isVisible():
            self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        center_y = h / 2
        bar_width = max(2.0, w / (self.bars * 1.8))
        spacing = bar_width * 0.45

        gradient = QLinearGradient(0, 0, w, 0)
        gradient.setColorAt(0.0, QColor(CYAN.red(), CYAN.green(), CYAN.blue(), 60))
        gradient.setColorAt(0.5, QColor(CYAN.red(), CYAN.green(), CYAN.blue(), 220))
        gradient.setColorAt(1.0, QColor(CYAN.red(), CYAN.green(), CYAN.blue(), 60))
        painter.setPen(QPen(gradient, bar_width, cap=Qt.PenCapStyle.RoundCap))

        total = self.bars * bar_width + (self.bars - 1) * spacing
        start_x = (w - total) / 2
        max_height = h * 0.45

        for idx, amp in enumerate(self.heights):
            x = start_x + idx * (bar_width + spacing) + bar_width / 2
            bh = float(amp * max_height)
            painter.drawLine(int(x), int(center_y - bh), int(x), int(center_y + bh))

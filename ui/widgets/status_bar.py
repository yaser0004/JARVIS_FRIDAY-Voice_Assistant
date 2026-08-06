"""System load meters.

Previously three raw mono labels reading "CPU: 16% GPU: 0. RAM: 69" in the
main window, which clipped at the sidebar width. Now labelled meters living
inside the diagnostics panel, where technical readouts belong.
"""

from __future__ import annotations

import threading

import psutil
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from ui.theme import mono_font


class MetricsStatusBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        self.cpu_label, self.cpu_bar = self._meter(layout, "CPU")
        self.gpu_label, self.gpu_bar = self._meter(layout, "GPU")
        self.ram_label, self.ram_bar = self._meter(layout, "RAM")

        self._gpu_load_cached = 0.0
        self._gpu_lock = threading.Lock()
        self._gpu_stop = threading.Event()
        self._gpu_probe_thread = threading.Thread(target=self._gpu_probe_loop, daemon=True)
        self._gpu_probe_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(800)
        self.refresh()

    def _meter(self, parent_layout: QVBoxLayout, name: str) -> tuple[QLabel, QProgressBar]:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        caption = QLabel(name)
        caption.setObjectName("readout")
        caption.setFont(mono_font(9, tracking=8.0))
        caption.setFixedWidth(34)

        bar = QProgressBar()
        bar.setObjectName("subtle")
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedHeight(5)

        value = QLabel("--%")
        value.setObjectName("readout")
        value.setFont(mono_font(9))
        value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        value.setFixedWidth(44)

        row.addWidget(caption)
        row.addWidget(bar, 1)
        row.addWidget(value)
        parent_layout.addLayout(row)
        return value, bar

    def _gpu_probe_loop(self) -> None:
        while not self._gpu_stop.is_set():
            value = self._gpu_load_probe()
            with self._gpu_lock:
                self._gpu_load_cached = value
            self._gpu_stop.wait(2.0)

    def _gpu_load_probe(self) -> float:
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            if not gpus:
                return 0.0
            peak_load = max(float(getattr(gpu, "load", 0.0)) for gpu in gpus)
            return peak_load * 100.0
        except Exception:
            return 0.0

    def _gpu_load(self) -> float:
        with self._gpu_lock:
            return float(self._gpu_load_cached)

    def refresh(self) -> None:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        gpu = self._gpu_load()

        for label, bar, value in (
            (self.cpu_label, self.cpu_bar, cpu),
            (self.gpu_label, self.gpu_bar, gpu),
            (self.ram_label, self.ram_bar, ram),
        ):
            label.setText(f"{value:.0f}%")
            bar.setValue(int(max(0.0, min(100.0, value))))

    def closeEvent(self, event) -> None:
        self._gpu_stop.set()
        super().closeEvent(event)

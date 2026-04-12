from __future__ import annotations

import threading
import time

import psutil
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QWidget

from ui.theme import FONT_MONO


class MetricsStatusBar(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.cpu_label = QLabel("CPU: --%")
        self.gpu_label = QLabel("GPU: --%")
        self.ram_label = QLabel("RAM: --%")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)

        for label in [self.cpu_label, self.gpu_label, self.ram_label]:
            label.setFont(FONT_MONO)
            layout.addWidget(label)

        self._gpu_load_cached = 0.0
        self._gpu_lock = threading.Lock()
        self._gpu_stop = threading.Event()
        self._gpu_probe_thread = threading.Thread(target=self._gpu_probe_loop, daemon=True)
        self._gpu_probe_thread.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(800)
        self.refresh()

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

        self.cpu_label.setText(f"CPU: {cpu:.0f}%")
        self.gpu_label.setText(f"GPU: {gpu:.1f}%")
        self.ram_label.setText(f"RAM: {ram:.0f}%")

    def closeEvent(self, event) -> None:
        self._gpu_stop.set()
        super().closeEvent(event)

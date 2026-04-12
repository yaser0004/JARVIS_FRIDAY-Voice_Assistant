from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtWidgets import QMenu, QStyle, QSystemTrayIcon

from core.session_logging import trace_event


class JarvisSystemTray(QSystemTrayIcon):
    def __init__(self, window, icon_path: Path, parent=None) -> None:
        icon_exists = bool(icon_path.exists() and icon_path.stat().st_size > 0)
        if icon_path.exists() and icon_path.stat().st_size > 0:
            icon = QIcon(str(icon_path))
        else:
            icon = window.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
        super().__init__(icon, parent)
        self.window = window
        self.wake_enabled = True

        menu = QMenu()
        self.show_action = QAction("Show Window", self)
        self.toggle_wake_action = QAction("Toggle Wake Word", self)
        self.quit_action = QAction("Quit", self)

        menu.addAction(self.show_action)
        menu.addAction(self.toggle_wake_action)
        menu.addSeparator()
        menu.addAction(self.quit_action)

        self.setContextMenu(menu)
        self.show_action.triggered.connect(self.restore_window)
        self.toggle_wake_action.triggered.connect(self.toggle_wake_word)
        self.quit_action.triggered.connect(self.window.force_quit)

        self.activated.connect(self.on_activated)
        self.setToolTip("JARVIS - Active")
        trace_event("ui.tray", "initialized", icon_path=str(icon_path), custom_icon=icon_exists)

    def on_activated(self, reason) -> None:
        trace_event("ui.tray", "activated", reason=str(reason))
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.restore_window()

    def restore_window(self) -> None:
        trace_event("ui.tray", "restore_window")
        self.window.showNormal()
        self.window.activateWindow()
        self.window.raise_()

    def toggle_wake_word(self) -> None:
        trace_event("ui.tray", "toggle_wake_word_requested", wake_enabled=bool(self.wake_enabled))
        if self.wake_enabled:
            if self.window.pipeline is not None:
                self.window.pipeline.update_wakeword_settings(enabled=False)
            self.window.disable_wake_word()
        else:
            if self.window.pipeline is not None:
                self.window.pipeline.update_wakeword_settings(enabled=True)
            self.window.enable_wake_word()

        wake_status = (
            self.window.pipeline.get_wakeword_status()
            if self.window.pipeline is not None and hasattr(self.window.pipeline, "get_wakeword_status")
            else {"enabled": False, "available": False, "initializing": False}
        )
        wake_enabled = bool(wake_status.get("enabled", False))
        wake_ready = bool(wake_status.get("available", False))
        wake_initializing = bool(wake_status.get("initializing", False))

        self.wake_enabled = wake_enabled and (wake_ready or wake_initializing)

        if wake_enabled and wake_ready:
            self.setToolTip("JARVIS - Wake word listening")
        elif wake_enabled and wake_initializing:
            self.setToolTip("JARVIS - Wake word starting")
        else:
            self.setToolTip("JARVIS - Active")

        trace_event(
            "ui.tray",
            "toggle_wake_word_applied",
            wake_enabled=wake_enabled,
            wake_ready=wake_ready,
            wake_initializing=wake_initializing,
            tray_wake_enabled=bool(self.wake_enabled),
        )



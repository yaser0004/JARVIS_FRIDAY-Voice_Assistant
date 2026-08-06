"""No-op window chrome for non-Windows platforms.

Same surface as `win_chrome.WindowChrome`, every capability off. The window
keeps its ordinary native frame and paints a fully opaque backdrop, which is
the correct result: without a compositor blurring anything behind it, a
translucent window over a sharp desktop just looks broken.
"""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QWidget


def apply_dark_titlebar(widget, dark: bool = True) -> bool:
    """No-op. Non-Windows window managers own their own decoration."""
    return False


class WindowChrome(QObject):
    backdrop_alpha_changed = pyqtSignal(int)

    def __init__(self, window: QWidget, drag_widget: QWidget | None = None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent or window)
        self._window = window
        self._drag_widget = drag_widget

        self.active = False
        self.acrylic_active = False
        self.supports_rounded_corners = False
        self.backdrop_alpha = 255

    # -- lifecycle ---------------------------------------------------------

    def install(self) -> bool:
        return False

    def uninstall(self) -> None:
        return None

    def handle_native_event(self, event_type, message):  # noqa: ANN001
        """Never ours. The caller must still return (False, 0), not None."""
        return None

    # -- actions -----------------------------------------------------------

    def toggle_maximize(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def resize_border_px(self) -> int:
        return 0

    def on_theme_changed(self) -> None:
        return None

    def debug_frame_bounds(self):
        return None

    def stats(self) -> dict:
        return {"active": False, "platform": "null"}

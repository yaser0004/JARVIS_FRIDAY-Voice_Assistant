"""Frameless window chrome and acrylic backdrop for Windows.

Everything ctypes lives here. The window calls `install()` once, forwards
`nativeEvent` to `handle_native_event()`, and otherwise stays platform-neutral.

Three findings shape this module, all measured on the target machine
(Windows 10 22H2, build 19045, PyQt6 6.7.0 / Qt 6.7.1) rather than recalled:

1. `super().nativeEvent()` returns `(False, None)` in PyQt6 6.7, and a `None`
   second element crashes the process with an access violation. sip maps the
   C++ `qintptr *result` out-parameter and writes through it without a None
   check. Every exit path in this module returns an `int`. See `_MSG_RESULT`.

2. `Qt.WindowType.FramelessWindowHint` is NOT used and must not be. Qt's
   default window style is already `WS_THICKFRAME | WS_CAPTION | WS_SYSMENU |
   WS_MIN/MAXIMIZEBOX` (measured: GWL_STYLE 0x96CF0000). That is exactly the
   style this recipe wants -- it is what gives us Aero Snap, Win+arrow, native
   edge resize and the DWM drop shadow for free. FramelessWindowHint strips it
   to 0x96000000 and destroys all four. Framelessness comes from eating
   WM_NCCALCSIZE alone, and no SetWindowLongPtrW call is needed.

3. The Win11 APIs fail with an HRESULT rather than silently no-op'ing on
   19045: DWMWA_WINDOW_CORNER_PREFERENCE (33) and DWMWA_SYSTEMBACKDROP_TYPE
   (38) both return E_INVALIDARG. So capability is detected by `hr == 0`, never
   by build number.

`install()` is a transaction. Only WM_NCCALCSIZE is load-bearing; every other
step is optional and failure degrades rather than breaks. If the load-bearing
part does not take, `uninstall()` runs and the window keeps an ordinary native
frame -- never an undraggable brick.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from typing import Optional

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QWidget

from core.session_logging import trace_event, trace_exception


# ---------------------------------------------------------------------------
# win32 constants
# ---------------------------------------------------------------------------

WM_NCCALCSIZE = 0x0083
WM_NCHITTEST = 0x0084
WM_SETTINGCHANGE = 0x001A
WM_ENTERSIZEMOVE = 0x0231
WM_EXITSIZEMOVE = 0x0232
WM_DWMCOMPOSITIONCHANGED = 0x031E

HTCLIENT = 1
HTCAPTION = 2
HTLEFT, HTRIGHT, HTTOP, HTTOPLEFT, HTTOPRIGHT = 10, 11, 12, 13, 14
HTBOTTOM, HTBOTTOMLEFT, HTBOTTOMRIGHT = 15, 16, 17

SM_CXFRAME = 32
SM_CYFRAME = 33
SM_CXPADDEDBORDER = 92

# DWM window attributes. 20 is correct on 19045; 19 is only valid on builds
# 17763-18362 and returns E_INVALIDARG here, so do not implement the common
# "try 20 then fall back to 19" pattern in the other order.
DWMWA_EXTENDED_FRAME_BOUNDS = 9
DWMWA_USE_IMMERSIVE_DARK_MODE = 20
DWMWA_WINDOW_CORNER_PREFERENCE = 33
DWMWCP_ROUND = 2

# Undocumented, reverse-engineered, and deliberately treated as best-effort.
WCA_ACCENT_POLICY = 19
ACCENT_DISABLED = 0
ACCENT_ENABLE_BLURBEHIND = 3
ACCENT_ENABLE_ACRYLICBLURBEHIND = 4

# The second element of a nativeEvent return. Named so the reason it is never
# None is greppable.
_MSG_RESULT = int


class MSG(ctypes.Structure):
    _fields_ = [
        ("hWnd", wintypes.HWND),
        ("message", ctypes.c_uint),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", wintypes.POINT),
    ]


class NCCALCSIZE_PARAMS(ctypes.Structure):
    _fields_ = [("rgrc", wintypes.RECT * 3), ("lppos", ctypes.c_void_p)]


class ACCENT_POLICY(ctypes.Structure):
    _fields_ = [
        ("AccentState", ctypes.c_int),
        ("AccentFlags", ctypes.c_int),
        ("GradientColor", ctypes.c_uint),   # 0xAABBGGRR -- ABGR, not ARGB
        ("AnimationId", ctypes.c_int),
    ]


class WINDOWCOMPOSITIONATTRIBDATA(ctypes.Structure):
    _fields_ = [
        ("Attribute", ctypes.c_int),
        ("Data", ctypes.POINTER(ACCENT_POLICY)),
        ("SizeOfData", ctypes.c_size_t),
    ]


class MARGINS(ctypes.Structure):
    _fields_ = [("cxLeftWidth", ctypes.c_int), ("cxRightWidth", ctypes.c_int),
                ("cyTopHeight", ctypes.c_int), ("cyBottomHeight", ctypes.c_int)]


# ---------------------------------------------------------------------------
# library loading -- every call site tolerates these being None
# ---------------------------------------------------------------------------

try:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
except OSError:                                             # pragma: no cover
    _user32 = None
try:
    _dwmapi = ctypes.WinDLL("dwmapi", use_last_error=True)
except OSError:                                             # pragma: no cover
    _dwmapi = None


def _has(lib, name: str) -> bool:
    return lib is not None and hasattr(lib, name)


def _dwm_set_attr(hwnd: int, attr: int, value: int) -> bool:
    """Set a DWORD DWM attribute. Returns True only on S_OK.

    This is the capability probe for the whole module: on Windows 10 the Win11
    attributes return E_INVALIDARG, so a plain `hr == 0` is a more honest and
    more portable switch than comparing build numbers.
    """
    if not _has(_dwmapi, "DwmSetWindowAttribute"):
        return False
    try:
        val = ctypes.c_int(int(value))
        hr = _dwmapi.DwmSetWindowAttribute(
            wintypes.HWND(hwnd), ctypes.c_uint(attr),
            ctypes.byref(val), ctypes.sizeof(val),
        )
        return hr == 0
    except Exception:
        return False


def _transparency_enabled() -> bool:
    """Read Settings > Personalization > Colors > Transparency effects.

    This is the failure mode with no return code: SetWindowCompositionAttribute
    reports success while transparency is off (and under battery saver) and you
    simply get no blur.
    """
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
        )
        try:
            value, _ = winreg.QueryValueEx(key, "EnableTransparency")
            return bool(value)
        finally:
            winreg.CloseKey(key)
    except Exception:
        # Absent key means the default, which is on.
        return True


class WindowChrome(QObject):
    """Frameless chrome + acrylic backdrop for one top-level window."""

    backdrop_alpha_changed = pyqtSignal(int)

    #: consecutive handler faults tolerated before the module self-disables
    MAX_FAULTS = 8

    def __init__(self, window: QWidget, drag_widget: QWidget | None = None,
                 parent: QObject | None = None) -> None:
        super().__init__(parent or window)
        self._window = window
        self._drag_widget = drag_widget
        self._hwnd: int = 0
        self._faults = 0
        self._nccalcsize_seen = 0
        self._messages_seen = 0
        self._moving = False

        self.active = False
        self.acrylic_active = False
        self.supports_rounded_corners = False
        self.backdrop_alpha = 255

        self._tint_alpha = 178
        self._tint_rgb = (16, 16, 20)

    # -- lifecycle ---------------------------------------------------------

    def install(self) -> bool:
        """Attach to the window. Must be called AFTER show().

        Returns False if the window should keep its ordinary native frame.
        """
        if _user32 is None:
            trace_event("ui.chrome", "unavailable", reason="user32 did not load")
            return False

        hwnd = int(self._window.winId())
        if not hwnd:
            trace_event("ui.chrome", "unavailable", reason="winId() is 0; call after show()")
            return False
        self._hwnd = hwnd

        # Deliberately no SetWindowLongPtrW: Qt's default style is already the
        # one we want, and every style change is a SWP_FRAMECHANGED we would
        # have to get right. See the module docstring.
        self.active = True
        self._apply_frame_changed()

        # Watchdog: if WM_NCCALCSIZE never arrives the frameless transform did
        # not take, and a half-installed chrome is worse than none.
        QTimer.singleShot(1500, self._verify_installed)

        trace_event("ui.chrome", "installed", hwnd=hwnd)
        return True

    def uninstall(self) -> None:
        if not self.active:
            return
        self.active = False
        self.set_acrylic(False)
        self._set_backdrop_alpha(255)
        self._apply_frame_changed()
        trace_event("ui.chrome", "uninstalled", faults=self._faults)

    def _verify_installed(self) -> None:
        if not self.active:
            return
        if self._nccalcsize_seen == 0:
            trace_event(
                "ui.chrome", "watchdog_tripped",
                messages_seen=self._messages_seen,
                detail="WM_NCCALCSIZE never arrived; reverting to the native frame",
            )
            self.uninstall()

    def _apply_frame_changed(self) -> None:
        """Force DWM to recompute the frame after the handler set changes."""
        if not (self._hwnd and _has(_user32, "SetWindowPos")):
            return
        SWP = 0x0002 | 0x0001 | 0x0004 | 0x0020  # NOMOVE|NOSIZE|NOZORDER|FRAMECHANGED
        try:
            _user32.SetWindowPos(wintypes.HWND(self._hwnd), None, 0, 0, 0, 0, SWP)
        except Exception:
            pass

    # -- native event dispatch --------------------------------------------

    def handle_native_event(self, event_type, message) -> Optional[tuple[bool, _MSG_RESULT]]:
        """Return (handled, result) for messages we own, else None.

        The caller MUST convert a None into `(False, 0)` itself -- returning
        None, or anything whose second element is None, crashes PyQt6 6.7.
        """
        if not self.active:
            return None
        try:
            msg = ctypes.cast(int(message), ctypes.POINTER(MSG)).contents
            self._messages_seen += 1
            return self._dispatch(msg)
        except Exception as exc:                             # pragma: no cover
            self._faults += 1
            trace_exception("ui.chrome", exc, faults=self._faults)
            if self._faults >= self.MAX_FAULTS:
                # An exception here does not propagate usefully: Qt treats the
                # message as unhandled, which for WM_NCCALCSIZE means the frame
                # flickers back forever. Self-disable instead.
                self.uninstall()
            return False, 0

    def _dispatch(self, msg: MSG) -> Optional[tuple[bool, _MSG_RESULT]]:
        """Stage 3 is observe-only; the handlers land in the next stages."""
        if msg.message == WM_NCCALCSIZE:
            self._nccalcsize_seen += 1
        return None

    # -- acrylic -----------------------------------------------------------

    def set_acrylic(self, enabled: bool) -> bool:
        return False

    def _set_backdrop_alpha(self, alpha: int) -> None:
        alpha = max(0, min(255, int(alpha)))
        if alpha == self.backdrop_alpha:
            return
        self.backdrop_alpha = alpha
        self.backdrop_alpha_changed.emit(alpha)

    # -- actions -----------------------------------------------------------

    def toggle_maximize(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()

    def resize_border_px(self) -> int:
        """Physical pixels. Never hardcode 8 -- this scales with DPI."""
        if not _has(_user32, "GetSystemMetricsForDpi"):
            if _has(_user32, "GetSystemMetrics"):
                return int(_user32.GetSystemMetrics(SM_CXFRAME)) + 4
            return 8
        dpi = self._dpi()
        return (int(_user32.GetSystemMetricsForDpi(SM_CXFRAME, dpi))
                + int(_user32.GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)))

    def _dpi(self) -> int:
        if _has(_user32, "GetDpiForWindow") and self._hwnd:
            try:
                return int(_user32.GetDpiForWindow(wintypes.HWND(self._hwnd))) or 96
            except Exception:
                return 96
        return 96

    def on_theme_changed(self) -> None:
        return None

    def debug_frame_bounds(self):
        """DWMWA_EXTENDED_FRAME_BOUNDS -- the only reliable view of the frame."""
        if not (self._hwnd and _has(_dwmapi, "DwmGetWindowAttribute")):
            return None
        rect = wintypes.RECT()
        try:
            hr = _dwmapi.DwmGetWindowAttribute(
                wintypes.HWND(self._hwnd), ctypes.c_uint(DWMWA_EXTENDED_FRAME_BOUNDS),
                ctypes.byref(rect), ctypes.sizeof(rect),
            )
        except Exception:
            return None
        if hr != 0:
            return None
        return rect.left, rect.top, rect.right, rect.bottom

    def stats(self) -> dict:
        return {
            "active": self.active,
            "platform": "win32",
            "acrylic": self.acrylic_active,
            "rounded": self.supports_rounded_corners,
            "backdrop_alpha": self.backdrop_alpha,
            "messages_seen": self._messages_seen,
            "nccalcsize_seen": self._nccalcsize_seen,
            "faults": self._faults,
            "transparency_enabled": _transparency_enabled(),
        }

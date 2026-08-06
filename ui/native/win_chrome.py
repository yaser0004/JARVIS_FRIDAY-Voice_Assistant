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

from PyQt6.QtCore import QObject, QPoint, QTimer, pyqtSignal
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

# Shell appbar, for the auto-hiding taskbar case.
ABM_GETSTATE = 0x00000004
ABM_GETTASKBARPOS = 0x00000005
ABS_AUTOHIDE = 0x00000001
ABE_LEFT, ABE_TOP, ABE_RIGHT, ABE_BOTTOM = 0, 1, 2, 3

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


class APPBARDATA(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("hWnd", wintypes.HWND),
        ("uCallbackMessage", ctypes.c_uint),
        ("uEdge", ctypes.c_uint),
        ("rc", wintypes.RECT),
        ("lParam", wintypes.LPARAM),
    ]


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
try:
    _shell32 = ctypes.WinDLL("shell32", use_last_error=True)
except OSError:                                             # pragma: no cover
    _shell32 = None


def _has(lib, name: str) -> bool:
    return lib is not None and hasattr(lib, name)


def _declare() -> None:
    """Pin argument and return types.

    ctypes defaults every return to c_int, which silently truncates the
    pointer-width returns on x64 -- SHAppBarMessage returns UINT_PTR and
    SetWindowLongPtr-family calls return LONG_PTR.
    """
    if _user32 is not None:
        if hasattr(_user32, "IsZoomed"):
            _user32.IsZoomed.argtypes = [wintypes.HWND]
            _user32.IsZoomed.restype = wintypes.BOOL
        if hasattr(_user32, "GetWindowRect"):
            _user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
            _user32.GetWindowRect.restype = wintypes.BOOL
        if hasattr(_user32, "GetDpiForWindow"):
            _user32.GetDpiForWindow.argtypes = [wintypes.HWND]
            _user32.GetDpiForWindow.restype = wintypes.UINT
        if hasattr(_user32, "GetSystemMetricsForDpi"):
            _user32.GetSystemMetricsForDpi.argtypes = [ctypes.c_int, wintypes.UINT]
            _user32.GetSystemMetricsForDpi.restype = ctypes.c_int
        if hasattr(_user32, "SetWindowCompositionAttribute"):
            _user32.SetWindowCompositionAttribute.argtypes = [
                wintypes.HWND, ctypes.POINTER(WINDOWCOMPOSITIONATTRIBDATA)]
            _user32.SetWindowCompositionAttribute.restype = wintypes.BOOL
    if _shell32 is not None and hasattr(_shell32, "SHAppBarMessage"):
        _shell32.SHAppBarMessage.argtypes = [wintypes.DWORD, ctypes.POINTER(APPBARDATA)]
        _shell32.SHAppBarMessage.restype = ctypes.c_size_t   # UINT_PTR


_declare()


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


def apply_dark_titlebar(widget, dark: bool = True) -> bool:
    """Darken a native-framed window's OS title bar.

    Dialogs keep their system frame -- they are not worth the frameless
    machinery, and an opaque QDialog under an acrylic parent would defeat the
    blur anyway. Without this they get a light grey Windows caption sitting on
    top of a dark window, which is the single most dated thing on screen.

    Attribute 20 is the correct one on this build; 19 is only valid on
    17763-18362 and returns E_INVALIDARG here.
    """
    try:
        hwnd = int(widget.winId())
    except Exception:
        return False
    if not hwnd:
        return False
    ok = _dwm_set_attr(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1 if dark else 0)
    if ok and _has(_user32, "SetWindowPos"):
        # The caption is not repainted until the frame is invalidated, so
        # without this the window stays light until it is moved.
        try:
            SWP = 0x0002 | 0x0001 | 0x0004 | 0x0020
            _user32.SetWindowPos(wintypes.HWND(hwnd), None, 0, 0, 0, 0, SWP)
        except Exception:
            pass
    return ok


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

        # Cosmetic here (the caption is never drawn once WM_NCCALCSIZE is
        # eaten) but it still colours the thin DWM frame and transient states
        # like the snap preview and Alt-Tab. 20 is correct on this build; 19
        # is only valid on 17763-18362.
        _dwm_set_attr(hwnd, DWMWA_USE_IMMERSIVE_DARK_MODE, 1)

        # Win11 only. E_INVALIDARG on Windows 10, which is exactly how we
        # learn that the silhouette has to stay square: the acrylic backdrop
        # ignores SetWindowRgn, so there is no other way to round it.
        self.supports_rounded_corners = _dwm_set_attr(
            hwnd, DWMWA_WINDOW_CORNER_PREFERENCE, DWMWCP_ROUND)

        # Deepens the shadow that WS_THICKFRAME already gives us. Documented,
        # and safe to skip on failure.
        if _has(_dwmapi, "DwmExtendFrameIntoClientArea"):
            try:
                margins = MARGINS(1, 1, 1, 1)
                _dwmapi.DwmExtendFrameIntoClientArea(
                    wintypes.HWND(hwnd), ctypes.byref(margins))
            except Exception:
                pass

        self._apply_frame_changed()
        self.set_acrylic(True)

        # Watchdog: if WM_NCCALCSIZE never arrives the frameless transform did
        # not take, and a half-installed chrome is worse than none.
        QTimer.singleShot(1500, self._verify_installed)

        trace_event(
            "ui.chrome", "installed", hwnd=hwnd,
            acrylic=self.acrylic_active,
            rounded=self.supports_rounded_corners,
            drag_mode=self._drag_mode(),
        )
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
        if msg.message == WM_NCCALCSIZE:
            self._nccalcsize_seen += 1
            return self._on_nccalcsize(msg)
        if msg.message == WM_NCHITTEST:
            return self._on_nchittest(msg)

        # Windows 10's acrylic makes window dragging stutter badly. It is a
        # bug in Windows, not in the caller -- the upstream Qt frameless
        # library closed the same report as "not fixable from my side". Going
        # fully opaque for the duration of the drag is the one mitigation that
        # is definitely fast; swapping to ACCENT_ENABLE_BLURBEHIND is not,
        # since state 3 has its own lag reports. Both messages are passed on
        # (False) so Qt still processes them normally.
        if msg.message == WM_ENTERSIZEMOVE:
            self._moving = True
            if self.acrylic_active and self._drag_mode() == "solid":
                self._set_accent_state(ACCENT_DISABLED)
                self._set_backdrop_alpha(255)
            return False, 0

        if msg.message == WM_EXITSIZEMOVE:
            self._moving = False
            if self.acrylic_active and self._drag_mode() == "solid":
                self._set_accent_state(ACCENT_ENABLE_ACRYLICBLURBEHIND)
                self._set_backdrop_alpha(self._tint_alpha)
            return False, 0

        # Transparency can be switched off in Settings, or by battery saver,
        # at any time. Re-evaluate rather than keeping a dead translucent
        # window.
        if msg.message in (WM_SETTINGCHANGE, WM_DWMCOMPOSITIONCHANGED):
            QTimer.singleShot(0, self._reevaluate_acrylic)
            return False, 0

        return None

    @staticmethod
    def _drag_mode() -> str:
        """`solid` (default) drops the blur while moving; `keep` leaves it on."""
        return (os.environ.get("JARVIS_ACRYLIC_DRAG_MODE") or "solid").strip().lower()

    def _reevaluate_acrylic(self) -> None:
        if not self.active:
            return
        wanted = _transparency_enabled()
        if wanted and not self.acrylic_active:
            self.set_acrylic(True)
        elif not wanted and self.acrylic_active:
            trace_event("ui.chrome", "acrylic_lost", reason="transparency turned off")
            self.set_acrylic(False)

    # -- WM_NCHITTEST ------------------------------------------------------

    def _on_nchittest(self, msg: MSG) -> Optional[tuple[bool, _MSG_RESULT]]:
        """Resize zones around the edges, caption over the app bar.

        Returning HTCAPTION and letting DefWindowProc take it from there is
        what buys drag-to-edge snap, Win+arrow AND double-click-to-maximize
        without writing a single mouse handler.

        Two traps live here:

        * lParam is in PHYSICAL pixels, while every Qt geometry call is in
          LOGICAL pixels. At 96 DPI they are identical, so a bug here passes
          every test on a 100%-scaling machine and breaks at 125% or 150%.
        * All eight resize zones must be OFF while maximized, or the top 8px
          of a maximized window becomes a resize strip lying over the app bar
          and swallowing clicks on the brand and the window buttons.
        """
        # Signed 16-bit halves: a window on a second monitor to the left has
        # negative screen coordinates, and masking without sign-extending
        # sends the hit test to the wrong edge.
        x = ctypes.c_short(msg.lParam & 0xFFFF).value
        y = ctypes.c_short((msg.lParam >> 16) & 0xFFFF).value

        rect = wintypes.RECT()
        if not (_has(_user32, "GetWindowRect")
                and _user32.GetWindowRect(wintypes.HWND(msg.hWnd), ctypes.byref(rect))):
            return None

        zoomed = bool(_has(_user32, "IsZoomed")
                      and _user32.IsZoomed(wintypes.HWND(msg.hWnd)))

        if not zoomed:
            border = self.resize_border_px()
            left = x < rect.left + border
            right = x >= rect.right - border
            top = y < rect.top + border
            bottom = y >= rect.bottom - border
            code = {
                (True, False, True, False): HTTOPLEFT,
                (True, False, False, True): HTBOTTOMLEFT,
                (False, True, True, False): HTTOPRIGHT,
                (False, True, False, True): HTBOTTOMRIGHT,
                (True, False, False, False): HTLEFT,
                (False, True, False, False): HTRIGHT,
                (False, False, True, False): HTTOP,
                (False, False, False, True): HTBOTTOM,
            }.get((left, right, top, bottom))
            if code is not None:
                return True, code

        if self._is_drag_area(x, y):
            return True, HTCAPTION
        return None

    def _is_drag_area(self, x: int, y: int) -> bool:
        """True when the physical point falls on draggable app-bar chrome."""
        bar = self._drag_widget
        if bar is None or not bar.isVisible():
            return False

        # Physical -> logical before handing anything to Qt. See the note in
        # _on_nchittest; this is the line that breaks at 125% scaling if the
        # division is dropped.
        ratio = self._window.devicePixelRatioF() or 1.0
        local = bar.mapFromGlobal(QPoint(int(round(x / ratio)), int(round(y / ratio))))
        if not bar.rect().contains(local):
            return False

        # Walk up from the deepest child. A dynamic property beats an
        # isinstance check because the bar will keep gaining widgets, and
        # "draggable" is the right default for decoration.
        child = bar.childAt(local)
        while child is not None and child is not bar:
            if child.property("windowDragExcluded"):
                return False
            child = child.parentWidget()
        return True

    def set_drag_widget(self, widget: QWidget | None) -> None:
        self._drag_widget = widget

    # -- WM_NCCALCSIZE -----------------------------------------------------

    def _on_nccalcsize(self, msg: MSG) -> tuple[bool, _MSG_RESULT]:
        """Eat the non-client area so the client fills the whole window.

        This one message is what makes the window frameless. Because
        WS_THICKFRAME and WS_CAPTION are still set, DefWindowProc keeps giving
        us snap, edge resize, the shadow and double-click-to-maximize.

        The maximized case is the trap. Measured on this machine, a maximized
        window's rect is (-8, -8, 1928, 1048) against a work area of
        (0, 0, 1920, 1040): it overflows by exactly the frame width on all four
        edges, which clips content and covers the taskbar. AdjustWindowRectExForDpi
        is the wrong tool here -- it yields 7 and leaves a 2px overhang. The
        frame + padded-border metrics land on rcWork exactly.
        """
        if not msg.wParam:
            return True, 0

        params = ctypes.cast(msg.lParam, ctypes.POINTER(NCCALCSIZE_PARAMS)).contents

        if _has(_user32, "IsZoomed") and _user32.IsZoomed(wintypes.HWND(msg.hWnd)):
            dpi = self._dpi()
            if _has(_user32, "GetSystemMetricsForDpi"):
                cx = (int(_user32.GetSystemMetricsForDpi(SM_CXFRAME, dpi))
                      + int(_user32.GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)))
                cy = (int(_user32.GetSystemMetricsForDpi(SM_CYFRAME, dpi))
                      + int(_user32.GetSystemMetricsForDpi(SM_CXPADDEDBORDER, dpi)))
            else:
                cx = int(_user32.GetSystemMetrics(SM_CXFRAME)) + 4
                cy = int(_user32.GetSystemMetrics(SM_CYFRAME)) + 4

            params.rgrc[0].left += cx
            params.rgrc[0].top += cy
            params.rgrc[0].right -= cx
            params.rgrc[0].bottom -= cy

            # With auto-hide on, rcWork equals rcMonitor, and a maximized
            # window pinned flush to the edge stops the taskbar being summoned.
            # Leaving one pixel keeps it reachable.
            edge = self._autohide_taskbar_edge()
            if edge == ABE_BOTTOM:
                params.rgrc[0].bottom -= 1
            elif edge == ABE_TOP:
                params.rgrc[0].top += 1
            elif edge == ABE_LEFT:
                params.rgrc[0].left += 1
            elif edge == ABE_RIGHT:
                params.rgrc[0].right -= 1

        return True, 0

    def _autohide_taskbar_edge(self) -> int:
        """Which edge an auto-hiding taskbar is on, or -1.

        Untested on this machine -- ABM_GETSTATE reports 0 here, so the
        auto-hide branch above has never actually run.
        """
        if not _has(_shell32, "SHAppBarMessage"):
            return -1
        try:
            data = APPBARDATA()
            data.cbSize = ctypes.sizeof(APPBARDATA)
            state = _shell32.SHAppBarMessage(ABM_GETSTATE, ctypes.byref(data))
            if not (int(state) & ABS_AUTOHIDE):
                return -1

            probe = APPBARDATA()
            probe.cbSize = ctypes.sizeof(APPBARDATA)
            if not _shell32.SHAppBarMessage(ABM_GETTASKBARPOS, ctypes.byref(probe)):
                return ABE_BOTTOM
            return int(probe.uEdge)
        except Exception:
            return -1

    # -- acrylic -----------------------------------------------------------

    def _accent_abgr(self) -> int:
        """Tint as 0xAABBGGRR.

        GradientColor is ABGR, not ARGB -- verified by measurement, not by
        reading a doc: passing 0xCC0000FF produced red-dominant pixels.
        """
        r, g, b = self._tint_rgb
        return ((self._tint_alpha & 0xFF) << 24 | (b & 0xFF) << 16
                | (g & 0xFF) << 8 | (r & 0xFF))

    def _set_accent_state(self, state: int) -> bool:
        if not (self._hwnd and _has(_user32, "SetWindowCompositionAttribute")):
            return False
        try:
            policy = ACCENT_POLICY()
            policy.AccentState = int(state)
            # Measured 0 against 2 and 0x1E0: all three produced byte-identical
            # interior and edge pixels for accent state 4 on this build. The
            # folklore value of 2 is not load-bearing.
            policy.AccentFlags = 0
            policy.GradientColor = (
                self._accent_abgr() if state != ACCENT_DISABLED else 0
            )
            policy.AnimationId = 0

            data = WINDOWCOMPOSITIONATTRIBDATA()
            data.Attribute = WCA_ACCENT_POLICY
            data.Data = ctypes.pointer(policy)
            data.SizeOfData = ctypes.sizeof(policy)
            return bool(_user32.SetWindowCompositionAttribute(
                wintypes.HWND(self._hwnd), ctypes.byref(data)))
        except Exception:
            return False

    def set_acrylic(self, enabled: bool) -> bool:
        """Turn the DWM acrylic backdrop on or off.

        The whole SetWindowCompositionAttribute surface is undocumented and
        reverse-engineered, and Microsoft has changed its behaviour between
        feature updates, so this is best-effort by construction: any failure
        leaves an opaque window rather than a broken one.
        """
        if not enabled:
            ok = self._set_accent_state(ACCENT_DISABLED)
            self.acrylic_active = False
            self._set_backdrop_alpha(255)
            return ok

        if not self.active:
            return False
        # Reports success while Settings > Personalization > Colors >
        # Transparency is off, and simply does not blur. There is no return
        # code for this, so it has to be read directly.
        if not _transparency_enabled():
            trace_event("ui.chrome", "acrylic_skipped", reason="EnableTransparency is 0")
            self.acrylic_active = False
            self._set_backdrop_alpha(255)
            return False

        if not self._set_accent_state(ACCENT_ENABLE_ACRYLICBLURBEHIND):
            trace_event("ui.chrome", "acrylic_failed")
            self.acrylic_active = False
            self._set_backdrop_alpha(255)
            return False

        self.acrylic_active = True
        self._set_backdrop_alpha(self._tint_alpha)
        trace_event("ui.chrome", "acrylic_enabled", alpha=self._tint_alpha)
        return True

    def set_tint(self, rgb: tuple[int, int, int], alpha: int) -> None:
        self._tint_rgb = (int(rgb[0]) & 0xFF, int(rgb[1]) & 0xFF, int(rgb[2]) & 0xFF)
        self._tint_alpha = max(0, min(255, int(alpha)))
        if self.acrylic_active:
            self._set_accent_state(ACCENT_ENABLE_ACRYLICBLURBEHIND)
            self._set_backdrop_alpha(self._tint_alpha)

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

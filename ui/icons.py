"""Vector icon factory.

The app previously used `QStyle.standardIcon`, which pulls native Windows
artwork: full-color, inconsistent in weight, and in several cases plainly
wrong (Send was a checkmark, the microphone was a speaker, Settings was a
list, Attach was a folder). Everything here is drawn as `QPainterPath` on a
24 unit grid with a single stroke weight, so icons recolor with the theme,
stay crisp at any device pixel ratio, and carry no licensing.

Usage:
    button.setIcon(icon("send"))
    button.setIcon(icon("mic", color=palette.danger))

Icons are cached; `invalidate_cache()` is called for you on a theme change.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Iterable

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap, QTransform

from core.config import APPDATA_DIR


GRID = 24.0
STROKE = 1.75

# (stroke paths, fill paths)
Drawing = tuple[list[QPainterPath], list[QPainterPath]]
Drawer = Callable[[], Drawing]


# ---------------------------------------------------------------------------
# path helpers
# ---------------------------------------------------------------------------


def _poly(points: Iterable[tuple[float, float]], close: bool = False) -> QPainterPath:
    path = QPainterPath()
    pts = list(points)
    if not pts:
        return path
    path.moveTo(*pts[0])
    for x, y in pts[1:]:
        path.lineTo(x, y)
    if close:
        path.closeSubpath()
    return path


def _rect(x: float, y: float, w: float, h: float, r: float = 0.0) -> QPainterPath:
    path = QPainterPath()
    if r > 0:
        path.addRoundedRect(QRectF(x, y, w, h), r, r)
    else:
        path.addRect(QRectF(x, y, w, h))
    return path


def _ellipse(cx: float, cy: float, rx: float, ry: float | None = None) -> QPainterPath:
    ry = rx if ry is None else ry
    path = QPainterPath()
    path.addEllipse(QPointF(cx, cy), rx, ry)
    return path


def _arc(cx: float, cy: float, r: float, start_deg: float, span_deg: float) -> QPainterPath:
    path = QPainterPath()
    box = QRectF(cx - r, cy - r, r * 2, r * 2)
    path.arcMoveTo(box, start_deg)
    path.arcTo(box, start_deg, span_deg)
    return path


def _gear(cx: float, cy: float, r_out: float, r_in: float, teeth: int = 8,
          tooth_frac: float = 0.42) -> QPainterPath:
    """A real gear outline rather than a circle with bumps drawn on it."""
    path = QPainterPath()
    step = math.tau / teeth
    half = step * tooth_frac / 2.0
    first = True
    for i in range(teeth):
        mid = i * step
        # tooth face, then the valley leading to the next tooth
        for angle, radius in (
            (mid - half, r_out),
            (mid + half, r_out),
            (mid + half + (step / 2 - half) * 0.55, r_in),
            (mid + step - half - (step / 2 - half) * 0.55, r_in),
        ):
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            if first:
                path.moveTo(x, y)
                first = False
            else:
                path.lineTo(x, y)
    path.closeSubpath()
    return path


# ---------------------------------------------------------------------------
# the icon set
#
# Each entry returns (paths to stroke, paths to fill).
# ---------------------------------------------------------------------------


def _send() -> Drawing:
    # Paper plane, with the fold line that makes it read as a plane and not
    # an arbitrary quadrilateral.
    body = _poly([(21, 3), (3, 10.6), (10.2, 13.6), (13.2, 21)], close=True)
    fold = _poly([(10.2, 13.6), (21, 3)])
    return [body, fold], []


def _mic() -> Drawing:
    capsule = _rect(9, 2.5, 6, 11.5, 3)
    cradle = _arc(12, 12.2, 5.4, 200, 140)
    stem = _poly([(12, 17.8), (12, 21)])
    base = _poly([(8.6, 21), (15.4, 21)])
    return [capsule, cradle, stem, base], []


def _mic_active() -> Drawing:
    strokes, fills = _mic()
    strokes.append(_arc(12, 12.2, 8.2, 205, 130))
    return strokes, fills


def _stop() -> Drawing:
    return [], [_rect(6.8, 6.8, 10.4, 10.4, 2.2)]


def _attach() -> Drawing:
    return [_poly([(12, 4.6), (12, 19.4)]), _poly([(4.6, 12), (19.4, 12)])], []


def _image() -> Drawing:
    frame = _rect(3, 5, 18, 14, 2.4)
    ridge = _poly([(3.6, 16.4), (9.2, 10.8), (12.4, 14), (16.2, 10.2), (20.4, 14.4)])
    return [frame, ridge], [_ellipse(8.4, 9.4, 1.5)]


def _screenshot() -> Drawing:
    # Capture brackets around a screen region.
    tl = _poly([(3, 8), (3, 4.4), (6.6, 4.4)])
    tr = _poly([(17.4, 4.4), (21, 4.4), (21, 8)])
    br = _poly([(21, 16), (21, 19.6), (17.4, 19.6)])
    bl = _poly([(6.6, 19.6), (3, 19.6), (3, 16)])
    inner = _rect(7.6, 8.6, 8.8, 6.8, 1.4)
    return [tl, tr, br, bl, inner], []


def _camera() -> Drawing:
    body = _rect(2.6, 6.6, 18.8, 13.2, 2.6)
    hump = _poly([(8.4, 6.6), (9.8, 4.2), (14.2, 4.2), (15.6, 6.6)])
    lens = _ellipse(12, 13.2, 3.9)
    return [body, hump, lens], []


def _globe() -> Drawing:
    outline = _ellipse(12, 12, 9)
    meridian = _ellipse(12, 12, 3.9, 9)
    lat_top = _poly([(3.6, 9), (20.4, 9)])
    lat_bottom = _poly([(3.6, 15), (20.4, 15)])
    return [outline, meridian, lat_top, lat_bottom], []


def _settings() -> Drawing:
    return [_gear(12, 12, 9.4, 6.9), _ellipse(12, 12, 3.1)], []


def _logs() -> Drawing:
    sheet = _poly([(5.4, 2.8), (14.2, 2.8), (18.6, 7.2), (18.6, 21.2), (5.4, 21.2)], close=True)
    corner = _poly([(14.2, 2.8), (14.2, 7.2), (18.6, 7.2)])
    lines = [
        _poly([(8.4, 11.4), (15.6, 11.4)]),
        _poly([(8.4, 14.6), (15.6, 14.6)]),
        _poly([(8.4, 17.8), (13.2, 17.8)]),
    ]
    return [sheet, corner, *lines], []


def _close() -> Drawing:
    return [_poly([(6, 6), (18, 18)]), _poly([(18, 6), (6, 18)])], []


def _minimize() -> Drawing:
    return [_poly([(5.5, 12), (18.5, 12)])], []


def _maximize() -> Drawing:
    return [_rect(5.5, 5.5, 13, 13, 1.6)], []


def _restore() -> Drawing:
    back = _poly([(8.4, 5.6), (18.4, 5.6), (18.4, 15.6)])
    front = _rect(5.6, 8.4, 10, 10, 1.4)
    return [back, front], []


def _trash() -> Drawing:
    lid = _poly([(4.2, 6.4), (19.8, 6.4)])
    handle = _poly([(9.4, 6.4), (9.4, 4.2), (14.6, 4.2), (14.6, 6.4)])
    body = _poly([(6.2, 6.4), (7.2, 20.6), (16.8, 20.6), (17.8, 6.4)])
    rib_a = _poly([(10.4, 10.2), (10.4, 17)])
    rib_b = _poly([(13.6, 10.2), (13.6, 17)])
    return [lid, handle, body, rib_a, rib_b], []


def _chevron_right() -> Drawing:
    return [_poly([(9.6, 5.4), (16.2, 12), (9.6, 18.6)])], []


def _chevron_down() -> Drawing:
    return [_poly([(5.4, 9.6), (12, 16.2), (18.6, 9.6)])], []


def _chevron_left() -> Drawing:
    return [_poly([(14.4, 5.4), (7.8, 12), (14.4, 18.6)])], []


def _caret_up() -> Drawing:
    return [], [_poly([(6.5, 14.5), (12, 9), (17.5, 14.5)], close=True)]


def _caret_down() -> Drawing:
    return [], [_poly([(6.5, 9.5), (12, 15), (17.5, 9.5)], close=True)]


def _eye() -> Drawing:
    lid = QPainterPath()
    lid.moveTo(2.4, 12)
    lid.cubicTo(6.4, 5.6, 17.6, 5.6, 21.6, 12)
    lid.cubicTo(17.6, 18.4, 6.4, 18.4, 2.4, 12)
    return [lid, _ellipse(12, 12, 3.2)], []


def _warning() -> Drawing:
    tri = _poly([(12, 3.4), (22, 20.4), (2, 20.4)], close=True)
    bar = _poly([(12, 10), (12, 15)])
    return [tri, bar], [_ellipse(12, 17.8, 0.95)]


def _check() -> Drawing:
    return [_poly([(4.8, 12.6), (9.8, 17.6), (19.2, 6.6)])], []


def _search() -> Drawing:
    return [_ellipse(10.6, 10.6, 6.6), _poly([(15.4, 15.4), (20.6, 20.6)])], []


def _refresh() -> Drawing:
    ring = _arc(12, 12, 8, 60, 260)
    head = _poly([(16.4, 1.8), (16.4, 6.6), (11.6, 6.6)], close=True)
    return [ring], [head]


def _brand() -> Drawing:
    """The orb mark: a bright core inside circumscribing data rings.

    The rings are tilted deliberately. Axis-aligned, the mark collapses into
    a passable eye at 16px, which is the size the tray uses most.
    """
    outer = _ellipse(12, 12, 9.6)
    ring_a = QPainterPath(_ellipse(12, 12, 9.6, 3.6))
    ring_b = QPainterPath(_ellipse(12, 12, 9.6, 3.6))

    tilt_a = QTransform().translate(12, 12).rotate(-26).translate(-12, -12)
    tilt_b = QTransform().translate(12, 12).rotate(30).translate(-12, -12)
    return [outer, tilt_a.map(ring_a), tilt_b.map(ring_b)], [_ellipse(12, 12, 2.9)]


_REGISTRY: dict[str, Drawer] = {
    "send": _send,
    "mic": _mic,
    "mic_active": _mic_active,
    "stop": _stop,
    "attach": _attach,
    "image": _image,
    "screenshot": _screenshot,
    "camera": _camera,
    "globe": _globe,
    "settings": _settings,
    "logs": _logs,
    "close": _close,
    "minimize": _minimize,
    "maximize": _maximize,
    "restore": _restore,
    "trash": _trash,
    "chevron_right": _chevron_right,
    "chevron_down": _chevron_down,
    "chevron_left": _chevron_left,
    "caret_up": _caret_up,
    "caret_down": _caret_down,
    "eye": _eye,
    "warning": _warning,
    "check": _check,
    "search": _search,
    "refresh": _refresh,
    "brand": _brand,
}


def available() -> list[str]:
    return sorted(_REGISTRY)


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


_pixmap_cache: dict[tuple, QPixmap] = {}
_icon_cache: dict[tuple, QIcon] = {}


def invalidate_cache() -> None:
    """Drop rendered icons. Called when the theme changes."""
    _pixmap_cache.clear()
    _icon_cache.clear()


def _resolve_color(color: QColor | str | None) -> QColor:
    if color is None:
        from ui.theme import theme  # local import avoids an import cycle

        return QColor(theme().palette.text)
    return QColor(color)


def pixmap(name: str, color: QColor | str | None = None, size: int = 24,
           dpr: float = 1.0, stroke: float = STROKE) -> QPixmap:
    """Render one icon. `size` is in logical pixels."""
    resolved = _resolve_color(color)
    key = (name, resolved.rgba(), int(size), round(float(dpr), 3), round(float(stroke), 3))
    hit = _pixmap_cache.get(key)
    if hit is not None:
        return hit

    drawer = _REGISTRY.get(name)
    px = QPixmap(max(1, int(round(size * dpr))), max(1, int(round(size * dpr))))
    px.setDevicePixelRatio(dpr)
    px.fill(Qt.GlobalColor.transparent)

    if drawer is not None:
        strokes, fills = drawer()
        painter = QPainter(px)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # QPainter already applies the pixmap's devicePixelRatio, so it works
        # in logical pixels here. Scaling by size*dpr/GRID would apply the
        # ratio twice and render the glyph off the edge of its own box.
        scale = size / GRID
        painter.scale(scale, scale)

        if fills:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(resolved)
            for path in fills:
                painter.drawPath(path)

        if strokes:
            pen = QPen(resolved, stroke)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for path in strokes:
                painter.drawPath(path)

        painter.end()

    _pixmap_cache[key] = px
    return px


def icon(name: str, color: QColor | str | None = None, size: int = 24,
         *, disabled_color: QColor | str | None = None) -> QIcon:
    """A themed QIcon with normal and disabled variants.

    Both 1x and 2x pixmaps are attached so the icon stays sharp when the
    window moves to a scaled display.
    """
    resolved = _resolve_color(color)
    if disabled_color is None:
        from ui.theme import theme

        disabled_color = theme().palette.text_faint
    disabled = QColor(disabled_color)

    key = (name, resolved.rgba(), disabled.rgba(), int(size))
    hit = _icon_cache.get(key)
    if hit is not None:
        return hit

    out = QIcon()
    for dpr in (1.0, 2.0):
        out.addPixmap(pixmap(name, resolved, size, dpr), QIcon.Mode.Normal, QIcon.State.Off)
        out.addPixmap(pixmap(name, disabled, size, dpr), QIcon.Mode.Disabled, QIcon.State.Off)
    _icon_cache[key] = out
    return out


def app_icon(color: QColor | str | None = None) -> QIcon:
    """Window and tray icon, built at the sizes Windows actually asks for."""
    resolved = _resolve_color(color)
    out = QIcon()
    for size in (16, 20, 24, 32, 48, 64, 128, 256):
        # Thin strokes vanish at 16px, so scale the weight with the size.
        weight = max(1.2, STROKE * (1.0 if size >= 48 else 1.45))
        out.addPixmap(pixmap("brand", resolved, size, 1.0, stroke=weight))
    return out


# ---------------------------------------------------------------------------
# stylesheet indicator assets
#
# QSS sub-controls (combo arrows, spin arrows, check marks) can only take an
# image via url(), so those few icons are written to disk. The color is baked
# into the filename: Qt caches url() images by path, and a stale cache would
# leave last theme's arrows behind.
# ---------------------------------------------------------------------------


ICON_CACHE_DIR = APPDATA_DIR / "icon_cache"

_INDICATORS = {
    "chevron_down": ("chevron_down", 12),
    "caret_up": ("caret_up", 10),
    "caret_down": ("caret_down", 10),
    "check": ("check", 13),
}


def export_indicator_assets(colors: dict[str, QColor]) -> dict[str, str]:
    """Write indicator PNGs and return `{key: file url}` for the stylesheet.

    `colors` maps an indicator key to its color; missing keys use "default".
    """
    try:
        ICON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        return {}

    default = colors.get("default") or _resolve_color(None)
    urls: dict[str, str] = {}

    wanted = dict(_INDICATORS)
    wanted["chevron_down_dim"] = ("chevron_down", 12)

    for key, (name, size) in wanted.items():
        color = QColor(colors.get(key, default))
        path = ICON_CACHE_DIR / f"{key}_{color.rgba():08x}_{size}.png"
        if not path.exists():
            # 2x so the indicator stays sharp on scaled displays; QSS sizes it
            # down via the explicit width/height in the rule.
            if not pixmap(name, color, size, 2.0).save(str(path), "PNG"):
                continue
        urls[key] = Path(path).as_posix()

    return urls

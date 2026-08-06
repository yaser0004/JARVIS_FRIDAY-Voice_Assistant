"""The orb.

Replaces the previous soft radial-gradient sphere with a dashed ring and six
floating dots. That shape is the generic assistant orb; this one follows the
Age of Ultron J.A.R.V.I.S. hologram, which was art-directed as a "modernist,
man-made construction": precise line work, a dense internal network of
connections inside a spherical center, a small very bright core, and
audio-reactive exterior rings of data evoking hard drive platters and
reel-to-reel tape.

Persona changes the palette, the density and the motion profile, never the
architecture, so JARVIS and FRIDAY read as the same machine configured two
ways. State is carried by motion and ring behavior rather than by recoloring
the sphere, because color-coding a blob per state is what makes assistant
UIs look interchangeable.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from PyQt6.QtCore import QElapsedTimer, QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
)
from PyQt6.QtWidgets import QWidget

from core.session_logging import trace_event
from ui import anim_clock
from ui.theme import manager as theme_manager, mix, theme, with_alpha
from ui.widgets.orb.sphere import (
    build_filaments,
    build_ring_bases,
    depth_bands,
    project,
    ring_points,
    rotation_matrix,
    runs,
)


DEPTH_BANDS = 4

# Per-state animation targets. Everything is eased toward these rather than
# snapped, so a state change reconfigures the orb instead of cutting to it.
#   radius   sphere scale
#   spin     multiplier on the persona spin/tilt rate
#   bright   filament opacity multiplier
#   ring     multiplier on the persona ring rate
#   tick     ring tick length multiplier
#   pulse    filament pulse speed multiplier
_STATE_TARGETS: dict[str, dict[str, float]] = {
    "IDLE": {"radius": 1.00, "spin": 1.00, "bright": 0.60, "ring": 1.00, "tick": 1.00, "pulse": 0.55},
    "LISTENING": {"radius": 1.06, "spin": 0.70, "bright": 0.96, "ring": 0.80, "tick": 1.65, "pulse": 0.9},
    "PROCESSING": {"radius": 0.94, "spin": 2.30, "bright": 1.00, "ring": 2.50, "tick": 1.15, "pulse": 2.6},
    "SPEAKING": {"radius": 1.02, "spin": 1.20, "bright": 0.92, "ring": 1.35, "tick": 1.70, "pulse": 1.3},
}

_QUALITY_STEPS = (1.0, 0.78, 0.58, 0.42)


class Ripple:
    __slots__ = ("radius", "life", "outbound")

    def __init__(self, radius: float, outbound: bool) -> None:
        self.radius = radius
        self.life = 1.0
        self.outbound = outbound


class OrbWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        # The orb used to paint an opaque backdrop and declare
        # WA_OpaquePaintEvent, so its 60 repaints a second would not force the
        # parent shell to redraw. That is no longer possible: an opaque widget
        # over the acrylic backdrop punches a solid rectangle through the blur.
        # It now composites, and the bloom pixmap is baked on transparency
        # instead of on palette.bg.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

        self._state = "IDLE"
        self._audio_level = 0.0
        self._audio_smooth = 0.0

        # Eased animation parameters, initialized to the idle targets.
        self._params = dict(_STATE_TARGETS["IDLE"])
        self._targets = dict(_STATE_TARGETS["IDLE"])

        self._yaw = 0.0
        self._pitch = 0.0
        self._breath = 0.0
        self._ring_phases = np.zeros(3, dtype=np.float64)
        self._tape_phase = 0.0
        self._sweep = 0.0

        self._ripples: list[Ripple] = []
        self._ripple_cooldown = 0.0

        self._ring_rolls = np.zeros(3, dtype=np.float64)
        self._identity = np.eye(3)

        self._quality_index = 0
        self._geometry_key: tuple | None = None
        self._filaments = None
        self._ring_angles = np.zeros(0)
        self._pulse_fil = np.zeros(0, dtype=np.int32)
        self._pulse_pos = np.zeros(0)
        self._pulse_speed = np.zeros(0)

        self._bloom_cache: dict[tuple, QPixmap] = {}

        self._paint_times: deque[float] = deque(maxlen=600)
        self._paint_clock = QElapsedTimer()
        self._profile = anim_clock.profiling_enabled()
        self._frames_since_check = 0

        theme_manager().changed.connect(self._on_theme_changed)

    # -- lifecycle --------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        anim_clock.subscribe(self._advance)

    def hideEvent(self, event) -> None:
        anim_clock.unsubscribe(self._advance)
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        anim_clock.unsubscribe(self._advance)
        super().closeEvent(event)

    def _on_theme_changed(self) -> None:
        self._bloom_cache.clear()
        self._geometry_key = None
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._bloom_cache.clear()

    # -- external inputs --------------------------------------------------

    def set_state(self, state: str) -> None:
        normalized = str(state or "IDLE").strip().upper()
        if normalized == self._state:
            return
        self._state = normalized
        self._targets = dict(_STATE_TARGETS.get(normalized, _STATE_TARGETS["IDLE"]))
        # A state change is the one moment a ripple is always justified.
        self._spawn_ripple(force=True)

    def set_audio_level(self, level: float) -> None:
        """Live amplitude, 0-1. Drives the ring ticks and the core."""
        try:
            self._audio_level = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            self._audio_level = 0.0

    # -- geometry ---------------------------------------------------------

    def _ensure_geometry(self) -> None:
        profile = theme().orb
        quality = _QUALITY_STEPS[self._quality_index]
        filament_count = max(6, int(round(profile.filaments * quality)))
        samples = max(20, int(round(profile.points * quality)))
        tick_count = max(12, int(round(profile.ticks_per_ring * quality)))
        key = (filament_count, samples, profile.rings, tick_count)

        if key == self._geometry_key:
            return

        self._filaments = build_filaments(filament_count, samples)
        self._ring_angles = np.linspace(0.0, 2.0 * math.pi, tick_count, endpoint=False)

        # Phases and rolls must stay the same length as the ring count: the
        # per-ring direction array in _advance is built from their size, and
        # a mismatch would broadcast wrongly rather than raise.
        rings = int(profile.rings)
        if self._ring_phases.size != rings or self._ring_rolls.size != rings:
            self._ring_phases = np.zeros(rings, dtype=np.float64)
            # Stagger the starting precession so the rings never align.
            self._ring_rolls = np.linspace(0.0, math.pi, rings, endpoint=False)

        pulses = max(3, filament_count // 3)
        rng = np.random.default_rng(29)
        self._pulse_fil = rng.integers(0, filament_count, size=pulses).astype(np.int32)
        self._pulse_pos = rng.random(pulses)
        self._pulse_speed = rng.uniform(0.16, 0.42, size=pulses)

        self._geometry_key = key
        trace_event(
            "ui.orb",
            "geometry_rebuilt",
            filaments=filament_count,
            samples=samples,
            rings=int(profile.rings),
            ticks=tick_count,
            quality=quality,
        )

    # -- animation --------------------------------------------------------

    def _advance(self, dt: float) -> None:
        if not self.isVisible():
            return

        profile = theme().orb

        # Ease parameters toward the current state's targets. The time
        # constant is deliberately slow enough to be legible as a transition.
        blend = 1.0 - math.exp(-dt / 0.22)
        for key, target in self._targets.items():
            self._params[key] += (target - self._params[key]) * blend

        self._audio_smooth += (self._audio_level - self._audio_smooth) * (1.0 - math.exp(-dt / 0.08))

        self._yaw = (self._yaw + profile.spin_rate * self._params["spin"] * dt) % 360.0
        self._pitch = (self._pitch + profile.tilt_rate * self._params["spin"] * dt) % 360.0
        self._breath += dt * 1.15
        self._tape_phase += profile.tape_rate * self._params["ring"] * dt * math.tau
        self._sweep = (self._sweep + dt * 1.25 * self._params["ring"]) % 1.0

        if self._ring_phases.size:
            # Alternating direction keeps the rings from reading as one solid
            # rotating cage.
            directions = np.array([1.0 if i % 2 == 0 else -0.72
                                   for i in range(self._ring_phases.size)])
            rate = math.radians(profile.ring_rate) * self._params["ring"]
            self._ring_phases = (self._ring_phases + directions * rate * dt) % math.tau
            # Precession is much slower than the platter spin: the rings drift
            # in screen space while the data on them travels.
            self._ring_rolls = (self._ring_rolls
                                + directions * rate * 0.16 * dt) % math.tau

        if self._pulse_pos.size:
            self._pulse_pos = (self._pulse_pos
                               + self._pulse_speed * self._params["pulse"] * dt) % 1.0

        self._advance_ripples(dt)
        self.update()

    def _advance_ripples(self, dt: float) -> None:
        for ripple in self._ripples:
            direction = 1.0 if ripple.outbound else -1.0
            ripple.radius += direction * dt * 0.85
            ripple.life -= dt * 0.75
        self._ripples = [r for r in self._ripples if r.life > 0.0 and 0.1 < r.radius < 2.2]

        self._ripple_cooldown -= dt
        if self._ripple_cooldown <= 0.0 and self._state in {"LISTENING", "SPEAKING"}:
            self._spawn_ripple()

    def _spawn_ripple(self, force: bool = False) -> None:
        if self._state not in {"LISTENING", "SPEAKING"} and not force:
            return
        outbound = self._state != "LISTENING"
        self._ripples.append(Ripple(radius=0.42 if outbound else 1.75, outbound=outbound))
        self._ripple_cooldown = 0.85 if self._state == "LISTENING" else 1.1
        if len(self._ripples) > 5:
            self._ripples.pop(0)

    # -- painting ---------------------------------------------------------

    def _backdrop(self, width: int, height: int, bloom_size: int, palette,
                  profile, dpr: float) -> QPixmap:
        """The outer bloom, baked onto transparency.

        The bloom covers essentially the whole widget, so blitting it
        additively every frame was the single most expensive operation in the
        orb. Baking it into one pixmap turns that into a plain copy. The size
        is bucketed so the breathing animation still hits the cache.

        Filled with transparency rather than palette.bg: an opaque fill would
        show as a solid rectangle punched through the acrylic backdrop.
        palette.bg is consequently no longer part of the cache key.
        """
        key = (width, height, bloom_size, profile.bloom.rgba(), round(dpr, 2))
        hit = self._bloom_cache.get(key)
        if hit is not None:
            return hit

        # Via QImage, not QPixmap. A bare QPixmap is created in the screen
        # format -- RGB32 with no alpha channel on Windows -- so filling it
        # with Qt.transparent produces solid BLACK, which showed up as a black
        # rectangle punched through the acrylic. An explicitly premultiplied
        # ARGB image is the only way to guarantee the alpha survives.
        image = QImage(max(1, int(width * dpr)), max(1, int(height * dpr)),
                       QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(dpr)
        image.fill(Qt.GlobalColor.transparent)

        if bloom_size > 4:
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
            cx, cy = width / 2.0, height / 2.0
            half = bloom_size / 2.0
            gradient = QRadialGradient(cx, cy, half)
            gradient.setColorAt(0.0, with_alpha(profile.bloom, 74))
            gradient.setColorAt(0.35, with_alpha(profile.bloom, 30))
            gradient.setColorAt(0.72, with_alpha(profile.bloom, 9))
            gradient.setColorAt(1.0, with_alpha(profile.bloom, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(QRectF(cx - half, cy - half, bloom_size, bloom_size))
            painter.end()

        px = QPixmap.fromImage(image)
        px.setDevicePixelRatio(dpr)

        # A handful of buckets is enough; unbounded growth would leak on
        # every resize.
        if len(self._bloom_cache) > 12:
            self._bloom_cache.clear()
        self._bloom_cache[key] = px
        return px

    def paintEvent(self, _event) -> None:
        if self._profile:
            self._paint_clock.restart()

        self._ensure_geometry()
        active = theme()
        profile = active.orb
        palette = active.palette

        painter = QPainter(self)

        width, height = self.width(), self.height()
        cx, cy = width / 2.0, height / 2.0
        short = min(width, height)

        breathe = 1.0 + 0.018 * math.sin(self._breath)
        radius = short * 0.215 * self._params["radius"] * breathe
        audio = self._audio_smooth

        # Opaque backdrop plus baked bloom, in one blit. Bucketed to 12 px so
        # the breathing animation does not miss the cache every frame.
        bloom_size = int(round(radius * 5.0 / 12.0)) * 12
        painter.drawPixmap(0, 0, self._backdrop(width, height, bloom_size,
                                                palette, profile,
                                                self.devicePixelRatioF()))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # On a light surface the luminous orb has nothing to glow against, so
        # it gets its own dark viewport to sit in.
        if not palette.is_dark:
            viewport = QRadialGradient(cx, cy, short * 0.46)
            viewport.setColorAt(0.0, QColor(6, 10, 16, 255))
            viewport.setColorAt(0.82, QColor(6, 10, 16, 235))
            viewport.setColorAt(1.0, QColor(6, 10, 16, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(viewport)
            painter.drawEllipse(QRectF(cx - short * 0.46, cy - short * 0.46,
                                       short * 0.92, short * 0.92))

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)

        rot = rotation_matrix(yaw_deg=self._yaw, pitch_deg=self._pitch)

        ring_geometry = self._ring_geometry(cx, cy, radius * 1.24, audio)
        self._draw_rings(painter, ring_geometry, profile.ring, front=False)
        self._draw_filaments(painter, rot, cx, cy, radius, profile)
        self._draw_core(painter, cx, cy, radius, profile, audio)
        self._draw_ripples(painter, cx, cy, radius, profile)
        self._draw_rings(painter, ring_geometry, profile.ring, front=True)

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.end()

        if self._profile:
            self._record_paint(self._paint_clock.nsecsElapsed() / 1_000_000.0)

    # -- filaments --------------------------------------------------------

    def _draw_filaments(self, painter: QPainter, rot, cx, cy, radius, profile) -> None:
        filaments = self._filaments
        if filaments is None:
            return

        xy, depth = project(filaments.points, rot, cx, cy, radius)
        bands = depth_bands(depth, DEPTH_BANDS)
        brightness = self._params["bright"]

        # One pen per band, reused across every filament run in that band.
        pens: list[QPen] = []
        for band in range(DEPTH_BANDS):
            t = band / max(1, DEPTH_BANDS - 1)
            color = mix(profile.filament_b, profile.filament_a, t)
            alpha = int(round((26 + 150 * t) * brightness))
            pen = QPen(with_alpha(color, max(0, min(255, alpha))))
            pen.setWidthF(0.7 + 0.85 * t)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            pens.append(pen)

        painter.setBrush(Qt.BrushStyle.NoBrush)
        for f in range(filaments.count):
            # One tolist() per filament. Iterating the numpy array directly
            # boxes every scalar and measured 2.7x slower for identical
            # output, which was the single largest cost in the whole orb.
            row_xy = xy[f].tolist()
            for start, stop, band in runs(bands[f]):
                # Extend one sample so consecutive bands visually join.
                stop = min(stop + 1, len(row_xy))
                if stop - start < 2:
                    continue
                painter.setPen(pens[band])
                painter.drawPolyline(QPolygonF(
                    [QPointF(x, y) for x, y in row_xy[start:stop]]
                ))

        self._draw_pulses(painter, xy, depth, profile, brightness)

    def _draw_pulses(self, painter: QPainter, xy, depth, profile, brightness) -> None:
        """Energy running through the network, brighter at the intersections."""
        if not self._pulse_pos.size:
            return
        samples = xy.shape[1]
        painter.setPen(Qt.PenStyle.NoPen)

        for idx, raw_pos in enumerate(self._pulse_pos.tolist()):
            fil = int(self._pulse_fil[idx])
            if fil >= xy.shape[0]:
                continue
            pos = int(raw_pos * samples) % samples
            z = float(depth[fil, pos])
            if z < -0.35:
                continue  # far side, would read as a floating speck
            near = (z + 1.0) * 0.5
            alpha = int(round(210 * near * brightness))
            size = 1.1 + 1.5 * near
            painter.setBrush(with_alpha(profile.core, max(0, min(255, alpha))))
            px, py = float(xy[fil, pos, 0]), float(xy[fil, pos, 1])
            painter.drawEllipse(QRectF(px - size, py - size, size * 2, size * 2))

    # -- rings ------------------------------------------------------------

    def _ring_geometry(self, cx, cy, radius, audio):
        if not self._ring_angles.size or not self._ring_rolls.size:
            return None

        ring_u, ring_v = build_ring_bases(self._ring_rolls)
        points = ring_points(ring_u, ring_v, self._ring_angles, self._ring_phases)

        # Tick length: a travelling envelope (the reel-to-reel scroll) times
        # the live amplitude. This is the audio-reactive part of the film
        # reference and the main consumer of set_audio_level().
        envelope = 0.55 + 0.45 * np.sin(
            self._ring_angles[None, :] * 3.0 + self._tape_phase
        )
        length = 0.055 * self._params["tick"] * (0.55 + 0.75 * audio) * envelope

        if self._state == "PROCESSING":
            # A graduated arc sweeping the rings.
            delta = np.abs(((self._ring_angles[None, :] / math.tau) - self._sweep + 0.5) % 1.0 - 0.5)
            length = length * (1.0 + 1.4 * np.exp(-(delta * 14.0) ** 2))

        # Ticks sit outboard of the ring band, so a long tick can never spear
        # through the sphere.
        length = np.clip(length, 0.012, 0.16)

        # Identity rotation: the ring bases already carry their orientation,
        # and the rings deliberately do not share the sphere's tumble.
        inner, inner_depth = project(points, self._identity, cx, cy, radius)
        outer, _ = project(points * (1.0 + length)[..., None],
                           self._identity, cx, cy, radius)
        return inner, outer, inner_depth

    def _draw_rings(self, painter: QPainter, geometry, color: QColor, front: bool) -> None:
        """Draw the ring band and its data ticks for one depth hemisphere.

        The band matters: ticks alone read as scattered hairs, and the film
        reference is explicitly rings of data circumscribing the hologram.
        """
        if geometry is None:
            return
        inner, outer, depth = geometry

        select = depth >= 0.0 if front else depth < 0.0
        # Behind the sphere the ring is dimmed rather than culled, which is
        # what sells it as circumscribing a solid volume.
        base_alpha = 205 if front else 54

        for ring in range(inner.shape[0]):
            mask = select[ring]
            if not mask.any():
                continue

            inner_row = inner[ring].tolist()
            outer_row = outer[ring].tolist()

            # The continuous band, as runs of consecutive visible samples.
            band_pen = QPen(with_alpha(color, int(base_alpha * 0.5)))
            band_pen.setWidthF(1.3 if front else 0.9)
            band_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(band_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for start, stop, visible in runs(mask.astype(np.int32)):
                if not visible or stop - start < 2:
                    continue
                painter.drawPolyline(QPolygonF(
                    [QPointF(x, y) for x, y in inner_row[start:stop]]
                ))

            # The ticks, grouped into a few brightness levels so one pen
            # covers many of them.
            near = (depth[ring] + 1.0) * 0.5
            indices = np.flatnonzero(mask)
            levels = np.clip((near[indices] * 4).astype(np.int32), 0, 3)
            tick_pen = QPen(color)
            tick_pen.setWidthF(1.5 if front else 1.0)
            tick_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            for level in range(4):
                subset = indices[levels == level].tolist()
                if not subset:
                    continue
                tick_pen.setColor(
                    with_alpha(color, int(base_alpha * (0.45 + 0.55 * (level / 3.0))))
                )
                painter.setPen(tick_pen)
                for i in subset:
                    painter.drawLine(
                        QPointF(inner_row[i][0], inner_row[i][1]),
                        QPointF(outer_row[i][0], outer_row[i][1]),
                    )

    # -- core and ripples -------------------------------------------------

    def _draw_core(self, painter: QPainter, cx, cy, radius, profile, audio) -> None:
        """A small, very bright core with tight bloom.

        Deliberately tight. A wide falloff washes the filament network out of
        the middle of the sphere, which is the one place the weave is densest
        and most worth seeing.
        """
        core_r = radius * (0.13 + 0.035 * audio) * (1.0 + 0.04 * math.sin(self._breath * 1.7))
        extent = core_r * 2.6
        gradient = QRadialGradient(cx, cy, extent)
        gradient.setColorAt(0.0, with_alpha(profile.core, 235))
        gradient.setColorAt(0.30, with_alpha(profile.core, 120))
        gradient.setColorAt(0.62, with_alpha(profile.filament_a, 34))
        gradient.setColorAt(1.0, with_alpha(profile.filament_b, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(QRectF(cx - extent, cy - extent, extent * 2, extent * 2))

    def _draw_ripples(self, painter: QPainter, cx, cy, radius, profile) -> None:
        if not self._ripples:
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for ripple in self._ripples:
            alpha = int(max(0.0, min(1.0, ripple.life)) * 90)
            if alpha <= 0:
                continue
            pen = QPen(with_alpha(profile.ring, alpha))
            pen.setWidthF(1.2)
            painter.setPen(pen)
            r = radius * ripple.radius
            painter.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

    # -- adaptive quality -------------------------------------------------

    def _record_paint(self, ms: float) -> None:
        self._paint_times.append(ms)
        self._frames_since_check += 1
        if self._frames_since_check < 180 or len(self._paint_times) < 120:
            return
        self._frames_since_check = 0

        ordered = sorted(self._paint_times)
        p95 = ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]

        # Budget is 8 ms at the default size. Step detail down rather than
        # dropping frames; step back up if there is comfortable headroom.
        if p95 > 8.0 and self._quality_index < len(_QUALITY_STEPS) - 1:
            self._quality_index += 1
            self._geometry_key = None
            trace_event("ui.orb", "quality_reduced", p95_ms=round(p95, 2),
                        quality=_QUALITY_STEPS[self._quality_index])
            self._paint_times.clear()
        elif p95 < 4.0 and self._quality_index > 0:
            self._quality_index -= 1
            self._geometry_key = None
            trace_event("ui.orb", "quality_restored", p95_ms=round(p95, 2),
                        quality=_QUALITY_STEPS[self._quality_index])
            self._paint_times.clear()

    def paint_stats(self) -> dict:
        """p50/p95 paint time in ms. Populated when JARVIS_UI_PROFILE is set."""
        if not self._paint_times:
            return {"samples": 0}
        ordered = sorted(self._paint_times)
        n = len(ordered)
        return {
            "samples": n,
            "p50_ms": round(ordered[n // 2], 3),
            "p95_ms": round(ordered[min(n - 1, int(0.95 * (n - 1)))], 3),
            "max_ms": round(ordered[-1], 3),
            "quality": _QUALITY_STEPS[self._quality_index],
        }

    def set_profiling(self, enabled: bool) -> None:
        self._profile = bool(enabled)
        self._paint_times.clear()

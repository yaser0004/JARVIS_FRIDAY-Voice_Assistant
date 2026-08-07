"""The assistant's presence: a circular waveform ring.

A glowing core inside a ring of radial bars whose lengths are the waveform.
Circular because it should read as one object at a glance, and bars because
the assistant's state is fundamentally about sound.

This replaces a projected 3D filament sphere. That renderer was real -- it
rotated an (N, P, 3) point cloud per frame and depth-sorted it -- but it never
actually ran: it dropped to a degenerate 8-filament fallback below 72px and
the only instance in the app was pinned at 28px, so every pixel the user ever
saw came from the fallback. What is drawn now is what the code says.

The four states are distinguishable without reading the label:

    IDLE        bars barely move, the ring drifts slowly
    LISTENING   bars are the live microphone spectrum
    PROCESSING  a bright band sweeps around the ring
    SPEAKING    bars pulse on a cadence

SPEAKING is synthetic on purpose. The backend sets SPEAKING and returns to
IDLE microseconds later while audio plays for seconds, and there is no TTS
amplitude to sample, so anything implying real reactivity would be a lie.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from PyQt6.QtCore import QElapsedTimer, QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QRadialGradient
from PyQt6.QtWidgets import QWidget

from ui import anim_clock
from ui.theme import manager as theme_manager, mix, theme, with_alpha
from ui.widgets import audio_source


#: Arc length each bar gets, in logical pixels. The bar count follows from
#: the ring's circumference rather than a size threshold: 48 bars around a
#: 64px orb is a bar every 1.9px, which renders as a fuzzy halo instead of as
#: bars. This keeps them crisp at every size.
ARC_PER_BAR = 3.6
MIN_BARS = 12

#: Radii as a fraction of the orb's half-size. Two constraints fight here:
#: the core's gradient has to finish well inside the baseline ring or its
#: wash covers the feet of the bars, and the bars need enough travel that a
#: loud one is obviously longer than a quiet one. A ring set too far out
#: leaves no room to grow and every state renders as the same dotted circle.
CORE_R = 0.15
BASE_R = 0.38
REACH = 0.56

#: Amplitude easing time constant, seconds.
SMOOTHING = 0.055

_STATES = ("IDLE", "LISTENING", "PROCESSING", "SPEAKING")


class OrbWidget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        # Small, because this is now placed by its container rather than
        # demanding a hero-sized box the way the sphere did.
        self.setMinimumSize(24, 24)
        # An opaque widget over the acrylic backdrop punches a solid rectangle
        # through the blur, so the orb composites.
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAutoFillBackground(False)

        self._state = "IDLE"
        self._bars = 0
        self._heights = np.zeros(0, dtype=np.float32)
        self._targets = np.zeros(0, dtype=np.float32)
        self._index = np.zeros(0, dtype=np.float32)

        self._phase = 0.0
        self._angle = 0.0
        self._sweep = 0.0
        self._breath = 0.0

        self._manual_level = 0.0
        self._audio_held = False
        self._static = False

        self._bloom_cache: dict[tuple, QPixmap] = {}
        self._paint_times: deque[float] = deque(maxlen=600)
        self._paint_clock = QElapsedTimer()
        self._profile = anim_clock.profiling_enabled()

        self._set_bar_count(audio_source.BARS)
        theme_manager().changed.connect(self._on_theme_changed)

    # -- lifecycle --------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._static:
            anim_clock.subscribe(self._advance)
            # hideEvent released the microphone; re-take it if we were still
            # listening when the window went away.
            if self._state == "LISTENING":
                self._acquire_audio()

    def hideEvent(self, event) -> None:
        # Both hideEvent and closeEvent must release: the shared clock only
        # stops its 16ms timer once the subscriber list empties, so a leaked
        # subscription keeps it running for the life of the process.
        anim_clock.unsubscribe(self._advance)
        self._release_audio()
        super().hideEvent(event)

    def closeEvent(self, event) -> None:
        anim_clock.unsubscribe(self._advance)
        self._release_audio()
        super().closeEvent(event)

    def _on_theme_changed(self) -> None:
        self._bloom_cache.clear()
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._bloom_cache.clear()
        self._set_bar_count(self._bars_for_size(min(self.width(), self.height())))

    @staticmethod
    def _bars_for_size(size: int) -> int:
        circumference = 2.0 * math.pi * (max(1, size) * 0.5 * BASE_R)
        return int(max(MIN_BARS, min(audio_source.BARS, circumference / ARC_PER_BAR)))

    def _set_bar_count(self, count: int) -> None:
        count = max(8, int(count))
        if count == self._bars:
            return
        self._bars = count
        self._heights = np.zeros(count, dtype=np.float32)
        self._targets = np.zeros(count, dtype=np.float32)
        self._index = np.arange(count, dtype=np.float32)

    @property
    def compact(self) -> bool:
        return self._bars < audio_source.BARS

    # -- inputs -----------------------------------------------------------

    def set_state(self, state: str) -> None:
        normalized = str(state or "IDLE").strip().upper()
        if normalized not in _STATES:
            normalized = "IDLE"
        if normalized == self._state:
            return
        self._state = normalized
        # The microphone is only open while something is actually listening.
        if normalized == "LISTENING":
            self._acquire_audio()
        else:
            self._release_audio()

    def set_audio_level(self, level: float) -> None:
        """Manual amplitude override, 0 disables it.

        Wins over the microphone. This is what lets the frame bench drive a
        realistic load on a machine with no audio input.
        """
        try:
            self._manual_level = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            self._manual_level = 0.0

    def set_static(self, static: bool) -> None:
        """Draw the idle ring without animating it.

        For animation_quality="minimal". A still ring is perfectly readable,
        unlike a frozen spinning sphere, which just looks like a hung app.
        """
        static = bool(static)
        if static == self._static:
            return
        self._static = static
        if static:
            anim_clock.unsubscribe(self._advance)
            self._release_audio()
        elif self.isVisible():
            anim_clock.subscribe(self._advance)
        self.update()

    def _acquire_audio(self) -> None:
        if not self._audio_held:
            audio_source.source().acquire()
            self._audio_held = True

    def _release_audio(self) -> None:
        if self._audio_held:
            audio_source.source().release()
            self._audio_held = False

    # -- animation --------------------------------------------------------

    def _advance(self, dt: float) -> None:
        if not self.isVisible():
            return

        profile = theme().orb
        self._phase += dt
        self._breath += dt * 1.15
        self._angle = (self._angle + profile.spin_rate * dt * 0.18) % 360.0
        self._sweep = (self._sweep + dt * 0.55) % 1.0

        if self._state == "LISTENING":
            audio_source.source().poll(dt)

        self._targets = self._state_targets()

        # Ease rather than snap, so a state change reconfigures the ring
        # instead of cutting to it.
        alpha = 1.0 - math.exp(-dt / SMOOTHING)
        self._heights += (self._targets - self._heights) * alpha
        self.update()

    def _state_targets(self) -> np.ndarray:
        n = self._bars
        i = self._index

        if self._state == "LISTENING":
            if self._manual_level > 0.001:
                # Override active: a plausible spectrum shape at that level.
                wobble = 0.5 + 0.5 * np.sin(self._phase * 6.0 + i * 0.55)
                return np.clip(self._manual_level * (0.35 + 0.65 * wobble), 0.04, 1.0)
            bins = audio_source.source().bins(n)
            if bins.size == n:
                return np.clip(bins, 0.04, 1.0)
            return np.full(n, 0.06, dtype=np.float32)

        if self._state == "PROCESSING":
            # A bright band travelling around the ring. Distance is measured
            # the short way round so the band crosses the seam smoothly.
            centre = self._sweep * n
            d = np.abs(i - centre)
            d = np.minimum(d, n - d)
            width = max(2.0, n / 9.0)
            return np.clip(0.08 + 0.85 * np.exp(-((d / width) ** 2)), 0.0, 1.0)

        if self._state == "SPEAKING":
            level = self._manual_level if self._manual_level > 0.001 else 0.55
            cadence = 0.5 + 0.5 * np.sin(self._phase * 2.6 + i * 0.26)
            return np.clip(0.16 + level * 0.75 * cadence, 0.05, 1.0)

        # IDLE. Not near-zero: the bars have to stay legible as bars when
        # nothing is happening, or the orb reads as a plain dot at rest.
        return np.clip(0.15 + 0.07 * np.sin(self._phase * 0.7 + i * 0.28), 0.0, 1.0)

    # -- painting ---------------------------------------------------------

    def _bloom(self, size: int, bloom_size: int, colour: QColor, dpr: float) -> QPixmap:
        """Soft outer glow, baked once per bucket.

        Built through an ARGB32_Premultiplied QImage rather than a QPixmap: a
        bare QPixmap is created in the screen format, which has no alpha
        channel on Windows, so filling it with Qt.transparent yields solid
        BLACK and punches a rectangle through the acrylic.
        """
        key = (size, bloom_size, colour.rgba(), round(dpr, 2))
        hit = self._bloom_cache.get(key)
        if hit is not None:
            return hit

        px_size = max(1, int(size * dpr))
        image = QImage(px_size, px_size, QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(dpr)
        image.fill(Qt.GlobalColor.transparent)

        if bloom_size > 4:
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            centre = size / 2.0
            half = bloom_size / 2.0
            gradient = QRadialGradient(centre, centre, half)
            gradient.setColorAt(0.0, with_alpha(colour, 66))
            gradient.setColorAt(0.42, with_alpha(colour, 26))
            gradient.setColorAt(0.75, with_alpha(colour, 8))
            gradient.setColorAt(1.0, with_alpha(colour, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(gradient)
            painter.drawEllipse(QRectF(centre - half, centre - half,
                                       bloom_size, bloom_size))
            painter.end()

        pixmap = QPixmap.fromImage(image)
        pixmap.setDevicePixelRatio(dpr)
        if len(self._bloom_cache) > 8:
            self._bloom_cache.clear()
        self._bloom_cache[key] = pixmap
        return pixmap

    def paintEvent(self, _event) -> None:
        if self._profile:
            self._paint_clock.restart()

        width, height = self.width(), self.height()
        short = min(width, height)
        if short <= 4:
            return

        active = theme()
        profile = active.orb
        palette = active.palette

        painter = QPainter(self)
        cx, cy = width / 2.0, height / 2.0
        radius = short * 0.5

        breathe = 1.0 + 0.02 * math.sin(self._breath)
        core_r = radius * CORE_R * breathe
        base_r = radius * BASE_R
        reach = radius * REACH

        # Glow first, underneath everything.
        bloom_size = int(round(radius * 1.9 / 8.0)) * 8
        painter.drawPixmap(0, 0, self._bloom(short, bloom_size, profile.bloom,
                                             self.devicePixelRatioF()))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # Additive only works over a dark ground. On a light surface every
        # stroke would brighten what is already near-white and the orb would
        # disappear, so light themes composite normally with dark ink.
        painter.setCompositionMode(
            QPainter.CompositionMode.CompositionMode_Plus if palette.is_dark
            else QPainter.CompositionMode.CompositionMode_SourceOver
        )

        self._draw_ring(painter, cx, cy, base_r, profile)
        self._draw_bars(painter, cx, cy, base_r, reach, radius, profile)
        self._draw_core(painter, cx, cy, core_r, profile)

        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.end()

        if self._profile:
            self._paint_times.append(self._paint_clock.nsecsElapsed() / 1_000_000.0)

    def _draw_ring(self, painter, cx, cy, base_r, profile) -> None:
        """The quiet baseline the bars stand on, so the circle reads when silent."""
        pen = QPen(with_alpha(profile.ring, 55))
        pen.setWidthF(max(0.8, base_r * 0.035))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(cx - base_r, cy - base_r, base_r * 2, base_r * 2))

    def _draw_bars(self, painter, cx, cy, base_r, reach, radius, profile) -> None:
        n = self._bars
        if n <= 0:
            return

        # Bars are as wide as the gap between them at the baseline, which keeps
        # the ring reading as a ring rather than as a dotted outline.
        width = max(1.0, (2.0 * math.pi * base_r) / n * 0.5)
        step = 360.0 / n
        rotation = self._angle

        for idx in range(n):
            amp = float(self._heights[idx])
            angle = math.radians(idx * step + rotation)
            sin_a, cos_a = math.sin(angle), math.cos(angle)

            inner = base_r
            # A floor on the length, so a silent bar is still a tick rather
            # than nothing at all.
            outer = base_r + reach * max(0.07, amp)
            # A triangular ramp rather than a linear one: linear leaves a hard
            # colour seam where the last bar meets the first.
            t = 2.0 * idx / n
            if t > 1.0:
                t = 2.0 - t
            colour = mix(profile.wave_a, profile.wave_b, t)

            pen = QPen(with_alpha(colour, int(150 + 105 * min(1.0, amp * 1.6))))
            pen.setWidthF(width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(
                QPointF(cx + cos_a * inner, cy + sin_a * inner),
                QPointF(cx + cos_a * outer, cy + sin_a * outer),
            )

    def _draw_core(self, painter, cx, cy, core_r, profile) -> None:
        painter.setPen(Qt.PenStyle.NoPen)

        # The falloff first, then a crisp disc inside it. The gradient alone
        # reads as a smudge rather than a centre -- most obviously on the
        # light theme, where a soft dark glow on a white page just looks like
        # a stain.
        extent = core_r * 2.1
        gradient = QRadialGradient(cx, cy, extent)
        gradient.setColorAt(0.0, with_alpha(profile.core, 200))
        gradient.setColorAt(0.34, with_alpha(profile.core, 105))
        gradient.setColorAt(0.68, with_alpha(profile.wave_a, 40))
        gradient.setColorAt(1.0, with_alpha(profile.wave_b, 0))
        painter.setBrush(gradient)
        painter.drawEllipse(QRectF(cx - extent, cy - extent, extent * 2, extent * 2))

        solid = core_r * 0.62
        painter.setBrush(with_alpha(profile.core, 245))
        painter.drawEllipse(QRectF(cx - solid, cy - solid, solid * 2, solid * 2))

    # -- profiling --------------------------------------------------------

    def paint_stats(self) -> dict:
        if not self._paint_times:
            return {"samples": 0}
        ordered = sorted(self._paint_times)

        def pct(p: float) -> float:
            idx = min(len(ordered) - 1, max(0, int(round(p * (len(ordered) - 1)))))
            return round(ordered[idx], 3)

        return {
            "samples": len(ordered),
            "p50_ms": pct(0.50),
            "p95_ms": pct(0.95),
            "max_ms": round(ordered[-1], 3),
            # Reported for the bench's table. The old adaptive ladder that used
            # to vary this only ever ran under profiling and had no effect on
            # what was painted, so it is gone rather than left to mislead.
            "quality": 1.0,
        }

    def set_profiling(self, enabled: bool) -> None:
        self._profile = bool(enabled)
        self._paint_times.clear()

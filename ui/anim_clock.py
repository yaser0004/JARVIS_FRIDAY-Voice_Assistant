"""One animation clock for the whole interface.

The orb and the waveform each used to own a 33 ms `CoarseTimer`. That caps
the UI at 30 FPS, doubles the number of wake-ups, and lets the two widgets
drift out of phase with each other. Everything animated now subscribes to a
single 16 ms `PreciseTimer` that emits the real elapsed delta.

The clock also measures what it actually delivered. A configured 16 ms
interval proves nothing: per-widget paint time can be well inside budget
while the composed window still lands at 35 FPS. `stats()` reports the
delivered interval, which is the number that means 60 FPS.
"""

from __future__ import annotations

import os
from collections import deque
from typing import Callable

from PyQt6.QtCore import QElapsedTimer, QObject, Qt, QTimer, pyqtSignal

from core.session_logging import trace_event


TARGET_INTERVAL_MS = 16
# dt is clamped so a stall (a modal dialog, a blocking call on the UI thread)
# cannot teleport every animation forward by a full second.
MAX_DT = 0.05


class AnimationClock(QObject):
    """Emits `tick(dt_seconds)` while at least one subscriber is listening."""

    tick = pyqtSignal(float)

    def __init__(self, interval_ms: int = TARGET_INTERVAL_MS) -> None:
        super().__init__()
        self._interval_ms = int(interval_ms)
        self._subscribers: list[Callable[[float], None]] = []

        self._elapsed = QElapsedTimer()
        self._elapsed.start()
        self._last_ns = self._elapsed.nsecsElapsed()

        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer.setInterval(self._interval_ms)
        self._timer.timeout.connect(self._on_timeout)

        self._intervals: deque[float] = deque(maxlen=2000)
        self._frames = 0

        # Phase realignment, see _maybe_realign().
        self._recent: deque[float] = deque(maxlen=40)
        self._since_check = 0
        self._realigns = 0
        self._consecutive_failures = 0
        # Consecutive failed attempts, not a lifetime cap: re-phasing is
        # triggered by ordinary events (restore from tray, monitor change),
        # so a session-long budget would run out and pin the UI at half rate.
        self._realign_budget = 4

    # -- subscription -----------------------------------------------------

    def subscribe(self, slot: Callable[[float], None]) -> None:
        if slot in self._subscribers:
            return
        self._subscribers.append(slot)
        self.tick.connect(slot)
        if not self._timer.isActive():
            # Reset the delta so the first frame after a pause is not a jump.
            self._last_ns = self._elapsed.nsecsElapsed()
            # Resuming re-phases the timer against the compositor, so the
            # detector has to judge the new regime on new samples only.
            self._recent.clear()
            self._since_check = 0
            self._consecutive_failures = 0
            self._timer.start()

    def unsubscribe(self, slot: Callable[[float], None]) -> None:
        if slot not in self._subscribers:
            return
        self._subscribers.remove(slot)
        try:
            self.tick.disconnect(slot)
        except (TypeError, RuntimeError):
            pass
        if not self._subscribers:
            self._timer.stop()

    @property
    def running(self) -> bool:
        return self._timer.isActive()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    # -- pacing -----------------------------------------------------------

    def set_interval(self, interval_ms: int) -> None:
        self._interval_ms = max(4, int(interval_ms))
        self._timer.setInterval(self._interval_ms)

    def _on_timeout(self) -> None:
        now = self._elapsed.nsecsElapsed()
        raw = (now - self._last_ns) / 1_000_000_000.0
        self._last_ns = now
        self._frames += 1
        self._intervals.append(raw)
        self._recent.append(raw)
        self.tick.emit(max(0.0005, min(raw, MAX_DT)))
        self._maybe_realign()

    def _maybe_realign(self) -> None:
        """Nudge the timer's phase when every frame is missing its deadline.

        A free-running timer has an arbitrary phase relative to the display's
        refresh. If it happens to fire just before a refresh, the frame's
        work lands after that boundary and waits for the next one, so the
        delivered interval doubles and stays doubled: a steady 31 ms at a
        16 ms target, which reads as 32 FPS with the CPU mostly idle.

        Restarting the timer re-phases it against the work that just
        finished. The budget applies to *consecutive* failed attempts, so a
        machine that genuinely cannot keep up stops being poked, while a
        session that re-phases many times over hours (restore from tray,
        moving to a monitor with a different refresh rate, a composition
        reset) gets corrected every time.
        """
        # A 20 frame window detects the doubling inside a second, which keeps
        # the transient off the start of the first animation the user sees.
        self._since_check += 1
        if self._since_check < 20 or len(self._recent) < 20:
            return
        self._since_check = 0

        ordered = sorted(self._recent)
        median_ms = ordered[len(ordered) // 2] * 1000.0

        if median_ms <= self._interval_ms * 1.5:
            # A good run proves the machine can hit the target, so whatever
            # failed before was phase and not capacity. Clear the budget.
            self._consecutive_failures = 0
            return

        if self._consecutive_failures >= self._realign_budget:
            return

        self._realigns += 1
        self._consecutive_failures += 1
        self._recent.clear()
        self._timer.stop()
        self._timer.start()
        self._last_ns = self._elapsed.nsecsElapsed()
        trace_event(
            "ui.anim_clock",
            "phase_realigned",
            median_ms=round(median_ms, 2),
            target_ms=self._interval_ms,
            total=self._realigns,
            consecutive=self._consecutive_failures,
        )

    # -- measurement ------------------------------------------------------

    def reset_stats(self) -> None:
        self._intervals.clear()
        self._frames = 0

    def stats(self) -> dict:
        """Delivered frame interval in milliseconds."""
        if not self._intervals:
            return {"samples": 0}
        ordered = sorted(self._intervals)
        n = len(ordered)

        def pct(p: float) -> float:
            idx = min(n - 1, max(0, int(round(p * (n - 1)))))
            return ordered[idx] * 1000.0

        mean = sum(ordered) / n
        return {
            "samples": n,
            "frames": self._frames,
            "p50_ms": round(pct(0.50), 3),
            "p95_ms": round(pct(0.95), 3),
            "p99_ms": round(pct(0.99), 3),
            "max_ms": round(ordered[-1] * 1000.0, 3),
            "mean_fps": round(1.0 / mean, 1) if mean > 0 else 0.0,
            "interval_ms": self._interval_ms,
        }

    def log_stats(self, label: str = "") -> dict:
        payload = self.stats()
        trace_event("ui.anim_clock", "frame_stats", label=label, **payload)
        return payload


_clock: AnimationClock | None = None


def clock() -> AnimationClock:
    global _clock
    if _clock is None:
        _clock = AnimationClock()
    return _clock


def subscribe(slot: Callable[[float], None]) -> None:
    try:
        clock().subscribe(slot)
    except RuntimeError:
        # Interpreter teardown can destroy the clock before the widgets that
        # use it. A widget losing its animation during shutdown is not worth
        # an exception on the way out.
        pass


def unsubscribe(slot: Callable[[float], None]) -> None:
    try:
        clock().unsubscribe(slot)
    except RuntimeError:
        pass


def profiling_enabled() -> bool:
    """Whether widgets should collect and report their own paint timings."""
    return os.getenv("JARVIS_UI_PROFILE", "").strip().lower() in {"1", "true", "yes"}

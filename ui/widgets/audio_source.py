"""Shared microphone capture and spectrum, for anything that visualises voice.

There is exactly one `sd.InputStream` in the UI, owned here and reference
counted, because two widgets opening their own streams on the same device is
how you get one of them silently failing.

Extracted from WaveformWidget when the orb became the app's audio visualiser.
Before that the widget captured audio, computed a level, and nothing ever read
it -- `WaveformWidget.level()` had zero call sites, so the orb's "audio
reactive" ring ticks sat pinned at their floor in every real run.

The stream is only open while something is actually listening: `acquire()` on
entering LISTENING, `release()` on leaving it.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from core.session_logging import trace_event


#: Spectrum resolution. Consumers that draw fewer bars should resample.
BARS = 48

#: Analysis rate, deliberately decoupled from the frame rate -- an FFT per
#: frame is wasted work when the ear cannot follow it that fast anyway.
FFT_HZ = 12.0

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
RING_SAMPLES = 2048


class AudioSource:
    """Reference-counted microphone spectrum.

    Not a QObject: it holds no Qt state and is polled from the animation
    clock rather than emitting signals, which keeps the audio callback free of
    any cross-thread Qt machinery.
    """

    def __init__(self) -> None:
        self._refs = 0
        self._stream = None
        self._ring: deque[float] = deque(maxlen=RING_SAMPLES)
        self._bins = np.zeros(BARS, dtype=np.float32)
        self._level = 0.0
        self._accum = 0.0

    # -- lifecycle ---------------------------------------------------------

    def acquire(self) -> None:
        """Open the stream on the first caller. Safe to call repeatedly."""
        self._refs += 1
        if self._refs > 1 or self._stream is not None:
            return
        self._start()

    def release(self) -> None:
        """Close the stream once the last caller lets go."""
        self._refs = max(0, self._refs - 1)
        if self._refs == 0:
            self._stop()

    @property
    def active(self) -> bool:
        return self._stream is not None

    def _start(self) -> None:
        try:
            import sounddevice as sd
        except Exception as exc:                            # pragma: no cover
            trace_event("ui.audio", "sounddevice_unavailable", error=str(exc))
            return

        def callback(indata, frames, _time, status):
            if status:
                return
            self._ring.extend(np.squeeze(indata).astype(np.float32))

        # This stream only feeds an animation; it must never be able to crash a
        # caller's state transition. When the device was busy and this was
        # unguarded, hold-to-talk's own capture never started and its button
        # and flags were left stuck with nothing to reset them.
        try:
            self._stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            trace_event("ui.audio", "mic_stream_start_failed", error=str(exc))

    def _stop(self) -> None:
        if self._stream is None:
            self._decay()
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exc:
            trace_event("ui.audio", "mic_stream_stop_failed", error=str(exc))
        finally:
            self._stream = None
            self._ring.clear()
            self._decay()

    def _decay(self) -> None:
        self._bins = np.zeros(BARS, dtype=np.float32)
        self._level = 0.0
        self._accum = 0.0

    # -- sampling ----------------------------------------------------------

    def poll(self, dt: float) -> bool:
        """Advance the analysis clock. True when `bins()` changed.

        Called from the animation clock by whichever widget is drawing, so the
        FFT runs at FFT_HZ no matter how many consumers there are.
        """
        if self._stream is None:
            return False
        self._accum += max(0.0, float(dt))
        if self._accum < (1.0 / FFT_HZ):
            return False
        self._accum = 0.0
        self._bins = self._spectrum()
        self._level = float(np.clip(self._bins.mean() * 1.8, 0.0, 1.0))
        return True

    def _spectrum(self) -> np.ndarray:
        if len(self._ring) < BARS * 2:
            return np.zeros(BARS, dtype=np.float32)

        arr = np.fromiter(self._ring, dtype=np.float32)
        if arr.size > BLOCK_SIZE:
            arr = arr[-BLOCK_SIZE:]
        spectrum = np.abs(np.fft.rfft(arr))
        if spectrum.size <= 1:
            return np.zeros(BARS, dtype=np.float32)
        # Drop the DC bin, then mean-pool into BARS buckets.
        chunks = np.array_split(spectrum[1:], BARS)
        values = np.array([float(np.mean(c)) for c in chunks], dtype=np.float32)
        values /= max(float(np.max(values)), 1e-6)
        return values

    # -- readers -----------------------------------------------------------

    def bins(self, count: int = BARS) -> np.ndarray:
        """Normalised spectrum, 0-1. Resampled when `count` differs."""
        if count == BARS or count <= 0:
            return self._bins
        idx = np.linspace(0, BARS - 1, num=count)
        return np.interp(idx, np.arange(BARS), self._bins).astype(np.float32)

    def level(self) -> float:
        return self._level


_source: AudioSource | None = None


def source() -> AudioSource:
    global _source
    if _source is None:
        _source = AudioSource()
    return _source

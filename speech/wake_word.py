from __future__ import annotations

import os
import queue
import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from PyQt6.QtCore import QObject, pyqtSignal

from core.config import WAKE_CHIME_PATH, log_performance
from speech.stt import SpeechToText


class WakeWordDetector(QObject):
    wake_word_detected = pyqtSignal()
    state_changed = pyqtSignal(str)
    transcript_ready = pyqtSignal(str)
    wake_acknowledged = pyqtSignal(str)

    def __init__(
        self,
        stt: Optional[SpeechToText] = None,
        sensitivity: float = 0.5,
        activation_phrases: Optional[list[str]] = None,
        strict_phrase_prefix: bool = False,
        follow_up_timeout: int = 10,
    ) -> None:
        super().__init__()
        self.stt = stt
        self.sensitivity = max(0.1, min(0.95, float(sensitivity)))
        self.activation_phrases = [str(p).strip().lower() for p in (activation_phrases or []) if str(p).strip()]
        self.strict_phrase_prefix = bool(strict_phrase_prefix)
        self.follow_up_timeout = max(3, min(20, int(follow_up_timeout)))
        self.sample_rate = 16000
        self.chunk_size = 1280
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)
        self._disabled_reason = ""
        self._phrase_fallback = self.stt is not None and bool(self.activation_phrases)
        self._prefer_phrase_mode = self._phrase_fallback
        self._detection_paused = False
        self._speech_energy_threshold = 0.01

        self.model = None

        if os.getenv("JARVIS_ENABLE_WAKEWORD", "0") != "1":
            self._disabled_reason = (
                "Wake-word backend is disabled by default due native runtime instability. "
                "Set JARVIS_ENABLE_WAKEWORD=1 to opt in."
            )
            return

        if self.stt is None:
            self.stt = SpeechToText()

        try:
            from openwakeword.model import Model as OWWModel

            self.model = OWWModel(inference_framework="onnx")
            if self._prefer_phrase_mode:
                self._disabled_reason = "Using phrase-based wake mode from configured activation phrases."
        except Exception:
            try:
                from openwakeword.model import Model as OWWModel

                # Last fallback: use the library default backend if available.
                self.model = OWWModel()
                if self._prefer_phrase_mode:
                    self._disabled_reason = "Using phrase-based wake mode from configured activation phrases."
            except Exception:
                self.model = None
                self._phrase_fallback = self.stt is not None and bool(self.activation_phrases)
                if self._phrase_fallback:
                    self._disabled_reason = "openWakeWord unavailable; using phrase fallback mode."
                else:
                    self._disabled_reason = "openWakeWord initialization failed."

    def _play_chime(self) -> None:
        if not WAKE_CHIME_PATH.exists():
            return
        try:
            samples, sr = sf.read(str(WAKE_CHIME_PATH), always_2d=False)
            sd.play(samples, sr, blocking=False)
        except Exception as exc:
            log_performance("wake_chime_error", 0.0, str(exc))

    def _audio_callback(self, indata, frames, _time, status) -> None:
        if status:
            return
        audio = np.squeeze(indata).astype(np.float32)
        try:
            self._queue.put_nowait(audio)
        except queue.Full:
            pass

    def _predict_score(self, audio: np.ndarray) -> float:
        try:
            samples = np.asarray(audio, dtype=np.float32).flatten()
            if samples.size == 0:
                return 0.0
            samples = np.clip(samples, -1.0, 1.0)
            pcm16 = (samples * 32767.0).astype(np.int16)

            output = self.model.predict(pcm16)
            if isinstance(output, dict) and output:
                scores: list[float] = []
                for value in output.values():
                    if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
                        scores.append(float(value[-1]))
                    elif isinstance(value, (int, float, np.floating)):
                        scores.append(float(value))
                if scores:
                    return float(max(scores))
        except Exception:
            return 0.0
        return 0.0

    def _clear_audio_queue(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def set_detection_paused(self, paused: bool) -> None:
        self._detection_paused = bool(paused)
        if self._detection_paused:
            self._clear_audio_queue()

    def _capture_transcript_from_queue(self, timeout: float) -> str:
        if self.stt is None:
            return ""

        deadline = time.perf_counter() + max(1.0, float(timeout))
        chunk_seconds = self.chunk_size / self.sample_rate
        min_blocks = max(2, int(0.25 / chunk_seconds))
        silence_limit = max(2, int(0.8 / chunk_seconds))

        speech_started = False
        silence_blocks = 0
        recorded: list[np.ndarray] = []

        while self._running and time.perf_counter() < deadline:
            try:
                chunk = self._queue.get(timeout=0.2)
            except queue.Empty:
                if speech_started and recorded:
                    silence_blocks += 1
                    if silence_blocks >= silence_limit:
                        break
                continue

            if chunk.size == 0:
                continue

            energy = float(np.sqrt(np.mean(np.square(chunk))))

            if not speech_started:
                if energy >= self._speech_energy_threshold:
                    speech_started = True
                    recorded.append(chunk)
                continue

            recorded.append(chunk)
            if energy < self._speech_energy_threshold:
                silence_blocks += 1
                if silence_blocks >= silence_limit and len(recorded) >= min_blocks:
                    break
            else:
                silence_blocks = 0

        if not recorded:
            return ""

        audio = np.concatenate(recorded, axis=0)
        transcript = self.stt.transcribe(audio, sample_rate=self.sample_rate)
        return str(transcript.get("text", "")).strip()

    def _capture_follow_up_from_stt(self) -> None:
        if self.stt is None:
            return

        transcript = self.stt.listen_once(timeout=6)
        first_text = str(transcript.get("text", "")).strip()
        if first_text:
            filtered = self._strip_activation_phrase(first_text)
            if filtered and not self._looks_like_wake_only(filtered):
                self.transcript_ready.emit(filtered)
                return

        self.wake_acknowledged.emit("Yes? Listening.")
        time.sleep(0.6)
        follow = self.stt.listen_once(timeout=self.follow_up_timeout)
        follow_text = str(follow.get("text", "")).strip()
        if not follow_text:
            return
        follow_filtered = self._strip_activation_phrase(follow_text)
        if follow_filtered:
            self.transcript_ready.emit(follow_filtered)

    def _listen_loop(self) -> None:
        while self._running:
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype="float32",
                    blocksize=self.chunk_size,
                    callback=self._audio_callback,
                ):
                    while self._running:
                        if self._detection_paused:
                            self._clear_audio_queue()
                            time.sleep(0.08)
                            continue

                        try:
                            chunk = self._queue.get(timeout=0.2)
                        except queue.Empty:
                            continue

                        start = time.perf_counter()
                        score = self._predict_score(chunk)
                        if score >= self.sensitivity:
                            latency = (time.perf_counter() - start) * 1000
                            log_performance("wake_word_trigger", latency)
                            self.wake_word_detected.emit()
                            self.state_changed.emit("LISTENING")
                            self._play_chime()
                            try:
                                self._capture_follow_up()
                            finally:
                                self._clear_audio_queue()
                            self.state_changed.emit("IDLE")
            except Exception as exc:
                if self._running:
                    log_performance("wake_stream_error", 0.0, str(exc))
                    self.state_changed.emit("IDLE")

            if not self._running:
                break

            log_performance("wake_stream_recover", 0.0, "reopening_input_stream")
            time.sleep(0.12)

    def _listen_loop_phrase_fallback(self) -> None:
        while self._running:
            if self._detection_paused:
                time.sleep(0.2)
                continue

            if self.stt is None:
                time.sleep(0.2)
                continue

            transcript = self.stt.listen_once(timeout=4)
            text = str(transcript.get("text", "")).strip()
            if not text:
                continue
            if not self._contains_activation_phrase(text):
                continue

            log_performance("wake_word_trigger", 0.0, "mode=phrase_fallback")
            self.wake_word_detected.emit()
            self.state_changed.emit("LISTENING")
            self._play_chime()

            filtered = self._strip_activation_phrase(text)
            if filtered and not self._looks_like_wake_only(filtered):
                self.transcript_ready.emit(filtered)
            else:
                self._capture_follow_up()

            self.state_changed.emit("IDLE")

    @staticmethod
    def _normalize_phrase(text: str) -> str:
        chars = [ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(text or "")]
        return " ".join("".join(chars).split())

    def _contains_activation_phrase(self, text: str) -> bool:
        normalized = self._normalize_phrase(text)
        if not normalized:
            return False
        for phrase in self.activation_phrases:
            p = self._normalize_phrase(phrase)
            if not p:
                continue
            if normalized.startswith(p) or f" {p} " in f" {normalized} ":
                return True
        return False

    def _looks_like_wake_only(self, text: str) -> bool:
        normalized = self._normalize_phrase(text)
        if not normalized:
            return True
        for phrase in self.activation_phrases:
            if normalized == self._normalize_phrase(phrase):
                return True
        return False

    def _capture_follow_up(self) -> None:
        if self.stt is None:
            return

        # Phrase fallback mode does not run a continuous callback stream.
        if self.model is None:
            self._capture_follow_up_from_stt()
            return

        first_text = self._capture_transcript_from_queue(timeout=6)
        if first_text:
            filtered = self._strip_activation_phrase(first_text)
            if filtered and not self._looks_like_wake_only(filtered):
                self.transcript_ready.emit(filtered)
                return

        self.wake_acknowledged.emit("Yes? Listening.")
        time.sleep(0.6)
        follow_text = self._capture_transcript_from_queue(timeout=self.follow_up_timeout)
        if not follow_text:
            return
        follow_filtered = self._strip_activation_phrase(follow_text)
        if follow_filtered:
            self.transcript_ready.emit(follow_filtered)

    def _strip_activation_phrase(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""

        lowered = cleaned.lower()
        if not self.activation_phrases:
            return cleaned

        for phrase in self.activation_phrases:
            if lowered.startswith(phrase):
                tail = cleaned[len(phrase) :].strip(" ,.!?;:")
                return tail or cleaned

        if self.strict_phrase_prefix:
            return ""
        return cleaned

    def start(self) -> None:
        if self._running and self._thread is not None and self._thread.is_alive():
            return
        if self._running and (self._thread is None or not self._thread.is_alive()):
            # If the worker thread exited unexpectedly, allow a clean restart.
            self._running = False

        target = None
        if self._prefer_phrase_mode and self._phrase_fallback:
            target = self._listen_loop_phrase_fallback
        elif self.model is not None:
            target = self._listen_loop
        elif self._phrase_fallback:
            target = self._listen_loop_phrase_fallback

        if target is None:
            return

        self._detection_paused = False
        self._running = True
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def is_available(self) -> bool:
        return self.model is not None or (self._phrase_fallback and self.stt is not None)

    def unavailable_reason(self) -> str:
        return str(self._disabled_reason or "")

    def stop(self) -> None:
        self._running = False
        self._detection_paused = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._clear_audio_queue()


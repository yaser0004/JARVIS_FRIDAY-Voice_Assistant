from __future__ import annotations

import ctypes
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import sounddevice as sd

from core.compute_runtime import normalize_compute_mode
from core.config import log_performance


@dataclass
class AudioChunk:
    data: np.ndarray
    sample_rate: int


class SpeechToText:
    _WINDOWS_CUDA_DLLS = (
        "cudnn_ops64_9.dll",
        "cudnn_cnn64_9.dll",
        "cudnn64_9.dll",
        "cublas64_12.dll",
        "cublasLt64_12.dll",
        "cudart64_12.dll",
    )

    def __init__(self, compute_mode: str = "auto") -> None:
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"
        self.model_name = "small.en"
        self._unavailable_reason = ""
        self._init_attempted = False
        self._init_in_progress = False
        self._compute_mode = normalize_compute_mode(compute_mode)

        # Avoid fatal OpenMP duplicate-runtime aborts on some Windows stacks.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")

    @staticmethod
    def _env_flag(name: str) -> bool:
        return os.getenv(name, "0").strip().lower() in {"1", "true", "yes"}

    def _cuda_runtime_ready(self) -> tuple[bool, str]:
        if self._env_flag("JARVIS_STT_FORCE_CUDA"):
            return True, ""

        try:
            import ctranslate2 as ct2

            if int(ct2.get_cuda_device_count()) < 1:
                return False, "No CUDA device detected by ctranslate2."
        except Exception as exc:
            return False, f"ctranslate2 CUDA probe failed: {exc}"

        if os.name != "nt" or self._env_flag("JARVIS_STT_ALLOW_UNVERIFIED_CUDA"):
            return True, ""

        missing: List[str] = []
        for dll_name in self._WINDOWS_CUDA_DLLS:
            try:
                ctypes.WinDLL(dll_name)
            except Exception:
                missing.append(dll_name)

        if missing:
            return False, f"Missing CUDA/cuDNN DLLs: {', '.join(missing)}"

        return True, ""

    def _reset_runtime(self) -> None:
        self.model = None
        self._init_attempted = False
        self._init_in_progress = False
        self._unavailable_reason = ""

    def warmup(self) -> None:
        self._ensure_model()

    def get_status(self) -> Dict[str, object]:
        if self.model is not None:
            return {
                "available": True,
                "initialized": True,
                "initializing": False,
                "reason": "",
                "device": self.device,
                "compute_type": self.compute_type,
            }

        if self._init_in_progress:
            return {
                "available": True,
                "initialized": False,
                "initializing": True,
                "reason": "Speech-to-text model is initializing.",
                "device": self.device,
                "compute_type": self.compute_type,
            }

        if self._init_attempted:
            return {
                "available": not bool(self._unavailable_reason),
                "initialized": True,
                "initializing": False,
                "reason": self._unavailable_reason or "Speech-to-text model is initializing.",
                "device": self.device,
                "compute_type": self.compute_type,
            }

        try:
            import faster_whisper  # noqa: F401

            return {
                "available": True,
                "initialized": False,
                "initializing": False,
                "reason": "Model will initialize on first voice request.",
                "device": self.device,
                "compute_type": self.compute_type,
            }
        except Exception as exc:
            return {
                "available": False,
                "initialized": False,
                "initializing": False,
                "reason": f"faster-whisper import failed: {exc}",
                "device": self.device,
                "compute_type": self.compute_type,
            }

    def set_compute_mode(self, mode: str) -> None:
        selected = normalize_compute_mode(mode)
        if selected == self._compute_mode:
            return
        self._compute_mode = selected
        self._reset_runtime()

    def _ensure_model(self) -> None:
        if self.model is not None or self._init_in_progress:
            return
        if self._init_attempted and self._unavailable_reason:
            return
        if self._init_attempted and self.model is None and not self._unavailable_reason:
            return

        self._init_attempted = True
        self._init_in_progress = True
        try:
            try:
                from faster_whisper import WhisperModel
            except Exception as exc:
                self._unavailable_reason = f"faster-whisper import failed: {exc}"
                return

            use_cuda_env = self._env_flag("JARVIS_STT_USE_CUDA")
            cuda_ready, cuda_reason = self._cuda_runtime_ready()
            attempts = []

            if self._compute_mode == "gpu":
                if cuda_ready:
                    attempts.append(("medium.en", "cuda", "float16"))
                else:
                    log_performance("stt_cuda_fallback", 0.0, cuda_reason)
                attempts.append(("small.en", "cpu", "int8"))
            elif self._compute_mode == "cpu":
                attempts.append(("small.en", "cpu", "int8"))
            else:
                # In auto mode, prefer stable CPU startup unless CUDA is explicitly requested.
                if use_cuda_env and cuda_ready:
                    attempts.append(("medium.en", "cuda", "float16"))
                elif use_cuda_env and not cuda_ready:
                    log_performance("stt_cuda_fallback", 0.0, cuda_reason)
                attempts.append(("small.en", "cpu", "int8"))

            errors: List[str] = []
            if use_cuda_env and not cuda_ready:
                errors.append(f"cuda_probe:{cuda_reason}")
            for model_name, device, compute_type in attempts:
                try:
                    self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
                    self.device = device
                    self.compute_type = compute_type
                    self.model_name = model_name
                    self._unavailable_reason = ""
                    return
                except Exception as exc:
                    errors.append(f"{device}:{exc}")

            self.model = None
            self._unavailable_reason = f"STT initialization failed ({'; '.join(errors)})"
        finally:
            self._init_in_progress = False

    def transcribe(self, audio_array: np.ndarray, sample_rate: int = 16000) -> Dict[str, float | str]:
        self._ensure_model()
        if self.model is None:
            return {
                "text": "",
                "language": "en",
                "confidence": 0.0,
                "duration_ms": 0.0,
                "error": self._unavailable_reason,
            }

        start = time.perf_counter()
        try:
            segments, info = self.model.transcribe(
                audio_array.astype(np.float32),
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            segment_list = list(segments)
        except Exception as exc:
            cuda_error = str(exc)
            if self.device == "cuda":
                log_performance("stt_cuda_runtime_error", 0.0, cuda_error)
                self._compute_mode = "cpu"
                self._reset_runtime()
                self._ensure_model()

                if self.model is not None:
                    try:
                        segments, info = self.model.transcribe(
                            audio_array.astype(np.float32),
                            beam_size=5,
                            language="en",
                            vad_filter=True,
                        )
                        segment_list = list(segments)
                        log_performance("stt_cuda_recovered_cpu", 0.0, "success=true")
                    except Exception as retry_exc:
                        return {
                            "text": "",
                            "language": "en",
                            "confidence": 0.0,
                            "duration_ms": 0.0,
                            "error": f"STT failed on CUDA and CPU fallback: cuda={cuda_error}; cpu={retry_exc}",
                        }
                else:
                    return {
                        "text": "",
                        "language": "en",
                        "confidence": 0.0,
                        "duration_ms": 0.0,
                        "error": f"STT CUDA failed and CPU fallback could not initialize: {self._unavailable_reason or cuda_error}",
                    }
            else:
                return {
                    "text": "",
                    "language": "en",
                    "confidence": 0.0,
                    "duration_ms": 0.0,
                    "error": f"STT transcription failed: {exc}",
                }

        text = " ".join(seg.text.strip() for seg in segment_list).strip()

        confidences: List[float] = []
        for seg in segment_list:
            value = getattr(seg, "avg_logprob", None)
            if value is not None:
                confidences.append(float(np.exp(value)))

        confidence = float(np.mean(confidences)) if confidences else 0.0
        duration_ms = (time.perf_counter() - start) * 1000
        log_performance("stt_transcribe", duration_ms, f"lang={getattr(info, 'language', 'en')}")

        return {
            "text": text,
            "language": getattr(info, "language", "en"),
            "confidence": confidence,
            "duration_ms": duration_ms,
        }

    def listen_once(self, timeout: int = 10) -> Dict[str, float | str]:
        self._ensure_model()
        if self.model is None:
            return {
                "text": "",
                "language": "en",
                "confidence": 0.0,
                "duration_ms": 0.0,
                "error": self._unavailable_reason,
            }

        sample_rate = 16000
        block_duration = 0.2
        block_samples = int(sample_rate * block_duration)
        max_blocks = int(timeout / block_duration)

        recorded: List[np.ndarray] = []
        silence_blocks = 0
        silence_limit = 6
        threshold = 0.01

        try:
            with sd.InputStream(channels=1, samplerate=sample_rate, dtype="float32") as stream:
                for _ in range(max_blocks):
                    chunk, _ = stream.read(block_samples)
                    chunk = chunk.flatten()
                    recorded.append(chunk)
                    energy = float(np.sqrt(np.mean(np.square(chunk))))

                    if energy < threshold:
                        silence_blocks += 1
                    else:
                        silence_blocks = 0

                    if silence_blocks >= silence_limit and len(recorded) > 4:
                        break
        except Exception as exc:
            return {
                "text": "",
                "language": "en",
                "confidence": 0.0,
                "duration_ms": 0.0,
                "error": f"Microphone read failed: {exc}",
            }

        if not recorded:
            return {"text": "", "language": "en", "confidence": 0.0, "duration_ms": 0.0}

        audio = np.concatenate(recorded, axis=0)
        return self.transcribe(audio, sample_rate)


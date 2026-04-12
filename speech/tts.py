from __future__ import annotations

import asyncio
import io
import os
import subprocess
import threading
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

import numpy as np
import sounddevice as sd

from core.config import log_performance


class TextToSpeech:
    PROFILE_VOICES = {
        "female": {
            "edge": "en-IE-EmilyNeural",
            "kokoro": "af_heart",
            "sapi_hint": "Zira",
            "rate": 0,
        },
        "male": {
            "edge": "en-GB-RyanNeural",
            "kokoro": "am_adam",
            "sapi_hint": "David",
            "rate": -1,
        },
    }

    def __init__(self, voice: str | None = None, speed: float = 1.1, profile: str = "female") -> None:
        self.profile = "female"
        self.voice = "af_heart"
        self._edge_voice = "en-IE-EmilyNeural"
        self._sapi_voice_hint = "Zira"
        self._sapi_rate = 0
        self.speed = speed
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis_tts")
        self._lock = threading.Lock()
        self._kokoro_module: Any | None = None
        self._edge_tts_module: Any | None = None
        self._pipeline: Any | None = None
        self._pipeline_lang = ""
        self._sample_rate = 24000
        self._sapi_proc: Optional[subprocess.Popen[str]] = None
        self._repo_id = os.getenv("JARVIS_TTS_REPO", "hexgrad/Kokoro-82M").strip() or "hexgrad/Kokoro-82M"
        self._backend_mode = os.getenv("JARVIS_TTS_BACKEND", "auto").strip().lower()
        self._allow_kokoro_download = os.getenv("JARVIS_TTS_ALLOW_KOKORO_DOWNLOAD", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        self._skip_kokoro_runtime = False
        normalized_profile = str(profile or "female").strip().lower()
        self.set_profile(normalized_profile if normalized_profile in {"female", "male"} else "female")
        if voice:
            self.voice = str(voice).strip() or self.voice

    @staticmethod
    def _language_code_for_voice(voice: str) -> str:
        code = str(voice or "").strip().lower()[:1]
        return code if code in {"a", "b", "e", "f", "h", "i", "j", "p", "z"} else "a"

    @staticmethod
    def _cache_has_file(repo_id: str, filename: str) -> bool:
        try:
            from huggingface_hub import hf_hub_download

            hf_hub_download(repo_id=repo_id, filename=filename, local_files_only=True)
            return True
        except Exception:
            return False

    def _kokoro_runtime_ready(self) -> tuple[bool, str]:
        if self._allow_kokoro_download:
            return True, ""

        required = ["config.json", "kokoro-v1_0.pth"]
        missing = [name for name in required if not self._cache_has_file(self._repo_id, name)]
        if missing:
            return (
                False,
                "Kokoro model assets are not cached locally. "
                "Set JARVIS_TTS_ALLOW_KOKORO_DOWNLOAD=1 to allow first-run download.",
            )
        return True, ""

    def _load_kokoro(self) -> Any:
        if self._kokoro_module is not None:
            return self._kokoro_module

        warnings.filterwarnings(
            "ignore",
            message="`resume_download` is deprecated",
            category=FutureWarning,
            module="huggingface_hub.file_download",
        )
        import kokoro

        self._kokoro_module = kokoro
        return kokoro

    def _load_edge_tts(self) -> Any:
        if self._edge_tts_module is not None:
            return self._edge_tts_module

        import edge_tts

        self._edge_tts_module = edge_tts
        return edge_tts

    async def _synthesize_edge_tts_async(self, text: str) -> tuple[np.ndarray, int]:
        edge_tts = self._load_edge_tts()
        communicate = edge_tts.Communicate(
            text,
            voice=self._edge_voice,
            rate="+0%",
            pitch="+0Hz",
        )

        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk.get("type") == "audio":
                audio_data.extend(chunk.get("data", b""))

        if not audio_data:
            raise RuntimeError("Edge TTS returned no audio data.")

        import soundfile as sf

        with io.BytesIO(bytes(audio_data)) as audio_buffer:
            samples, sample_rate = sf.read(audio_buffer, dtype="float32")

        arr = np.asarray(samples, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.mean(arr, axis=1, dtype=np.float32)

        peak = float(np.max(np.abs(arr))) if arr.size else 0.0
        if peak > 1e-6:
            arr = np.clip(arr * (0.9 / peak), -1.0, 1.0)

        return arr.astype(np.float32, copy=False), int(sample_rate)

    def _synthesize_edge_tts(self, text: str) -> tuple[np.ndarray, int]:
        return asyncio.run(self._synthesize_edge_tts_async(text))

    def _ensure_pipeline(self) -> Any:
        kokoro = self._load_kokoro()
        pipeline_ctor = getattr(kokoro, "KPipeline", None)
        if not callable(pipeline_ctor):
            raise RuntimeError("Kokoro KPipeline API is unavailable.")

        ready, reason = self._kokoro_runtime_ready()
        if not ready:
            raise RuntimeError(reason)

        lang_code = self._language_code_for_voice(self.voice)
        if self._pipeline is not None and self._pipeline_lang == lang_code:
            return self._pipeline

        requested_device = os.getenv("JARVIS_TTS_DEVICE", "cpu").strip().lower()
        kwargs = {"lang_code": lang_code}
        if requested_device in {"cpu", "cuda"}:
            kwargs["device"] = requested_device

        kwargs["repo_id"] = self._repo_id
        self._pipeline = pipeline_ctor(**kwargs)
        self._pipeline_lang = lang_code
        return self._pipeline

    def _synthesize_kokoro(self, text: str) -> tuple[np.ndarray, int]:
        kokoro = self._load_kokoro()

        # Backward compatibility for older wrappers.
        if hasattr(kokoro, "generate"):
            samples, sample_rate = kokoro.generate(text, voice=self.voice, speed=self.speed)
            return np.asarray(samples, dtype=np.float32), int(sample_rate)

        pipeline = self._ensure_pipeline()
        lang_code = self._language_code_for_voice(self.voice)
        voice_candidates = [self.voice]
        if lang_code == "a":
            if self.profile == "male":
                voice_candidates.extend(["am_adam", "am_michael", "af_heart"])
            else:
                voice_candidates.extend(["af_heart", "af_bella", "am_adam"])

        last_error = ""
        seen = set()
        for voice_name in voice_candidates:
            candidate = str(voice_name or "").strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)

            if not self._allow_kokoro_download and not self._cache_has_file(self._repo_id, f"voices/{candidate}.pt"):
                last_error = f"voice '{candidate}' is not cached locally"
                continue

            try:
                chunks: list[np.ndarray] = []
                for result in pipeline(text, voice=candidate, speed=self.speed, split_pattern=r"\n+"):
                    audio = getattr(result, "audio", None)
                    if audio is None:
                        continue

                    if hasattr(audio, "detach"):
                        arr = audio.detach().cpu().numpy()
                    elif hasattr(audio, "numpy"):
                        arr = audio.numpy()
                    else:
                        arr = np.asarray(audio)

                    arr = np.asarray(arr, dtype=np.float32).flatten()
                    if arr.size:
                        chunks.append(arr)

                if chunks:
                    if candidate != self.voice:
                        self.voice = candidate
                    return np.concatenate(chunks), self._sample_rate

                last_error = f"no audio chunks for voice '{candidate}'"
            except Exception as exc:
                last_error = f"voice '{candidate}' failed: {exc}"

        raise RuntimeError(f"Kokoro synthesis failed ({last_error})")

    def set_profile(self, profile: str) -> None:
        selected = "male" if str(profile or "").strip().lower() == "male" else "female"
        cfg = self.PROFILE_VOICES[selected]
        self.profile = selected
        self._edge_voice = str(cfg["edge"])
        self.voice = str(cfg["kokoro"])
        self._sapi_voice_hint = str(cfg["sapi_hint"])
        self._sapi_rate = int(cfg["rate"])

        # Invalidate pipeline when language group changes to keep voice loading correct.
        self._pipeline = None
        self._pipeline_lang = ""

    def _speak_windows_fallback(self, text: str) -> tuple[bool, str]:
        if os.name != "nt":
            return False, "Windows SAPI fallback is unavailable on this OS."

        voice_hint = str(self._sapi_voice_hint or "").replace("'", "")
        rate_value = int(self._sapi_rate)

        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$s.Volume = 100; "
            f"$hint = '{voice_hint}'; "
            "if (-not [string]::IsNullOrWhiteSpace($hint)) { "
            "  $candidate = $s.GetInstalledVoices() | "
            "    ForEach-Object { $_.VoiceInfo.Name } | "
            "    Where-Object { $_ -like \"*${hint}*\" } | "
            "    Select-Object -First 1; "
            "  if ($candidate) { $s.SelectVoice($candidate) } "
            "}; "
            f"$s.Rate = {rate_value}; "
            "$t = [Console]::In.ReadToEnd(); "
            "if ([string]::IsNullOrWhiteSpace($t)) { exit 2 }; "
            "$s.Speak($t)"
        )

        proc: Optional[subprocess.Popen[str]] = None
        try:
            proc = subprocess.Popen(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            self._sapi_proc = proc
            stdout, stderr = proc.communicate(input=text, timeout=120)
            if proc.returncode == 0:
                return True, ""
            details = (stderr or stdout or "").strip() or f"returncode={proc.returncode}"
            return False, details
        except Exception as exc:
            return False, str(exc)
        finally:
            self._sapi_proc = None
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass

    def speak(self, text: str) -> None:
        text = str(text or "").strip()
        if not text:
            return

        start = time.perf_counter()
        prefer_sapi = self._backend_mode == "sapi"
        attempt_edge = self._backend_mode in {"auto", "edge"}
        attempt_kokoro = self._backend_mode in {"auto", "kokoro"} and not self._skip_kokoro_runtime

        if attempt_edge:
            try:
                samples, sample_rate = self._synthesize_edge_tts(text)
                with self._lock:
                    sd.stop()
                    if os.getenv("JARVIS_TTS_NO_PLAYBACK", "0").strip().lower() in {"1", "true", "yes"}:
                        log_performance(
                            "tts_speak",
                            (time.perf_counter() - start) * 1000,
                            "backend=edge;playback=skipped",
                        )
                        return
                    sd.play(samples, sample_rate, blocking=True)
                log_performance(
                    "tts_speak",
                    (time.perf_counter() - start) * 1000,
                    f"backend=edge;voice={self._edge_voice};sr={sample_rate}",
                )
                return
            except Exception as exc:
                log_performance("tts_error", 0.0, f"edge:{exc}")

        if attempt_kokoro and self._backend_mode == "auto":
            ready, reason = self._kokoro_runtime_ready()
            if not ready:
                self._skip_kokoro_runtime = True
                log_performance("tts_fallback", 0.0, reason)
                attempt_kokoro = False

        if attempt_kokoro:
            try:
                samples, sample_rate = self._synthesize_kokoro(text)
                with self._lock:
                    sd.stop()
                    if os.getenv("JARVIS_TTS_NO_PLAYBACK", "0").strip().lower() in {"1", "true", "yes"}:
                        log_performance(
                            "tts_speak",
                            (time.perf_counter() - start) * 1000,
                            "backend=kokoro;playback=skipped",
                        )
                        return
                    sd.play(samples, sample_rate, blocking=True)
                log_performance("tts_speak", (time.perf_counter() - start) * 1000, f"backend=kokoro;sr={sample_rate}")
                return
            except Exception as exc:
                log_performance("tts_error", 0.0, f"kokoro:{exc}")

        fallback_ok, details = self._speak_windows_fallback(text)
        if fallback_ok:
            log_performance("tts_speak", (time.perf_counter() - start) * 1000, "backend=sapi")
            return

        if details:
            log_performance("tts_error", 0.0, f"sapi:{details}")
        else:
            log_performance("tts_error", 0.0, "sapi:fallback failed")

    def speak_async(self, text: str) -> Future[None]:
        return self._executor.submit(self.speak, text)

    def stop(self) -> None:
        with self._lock:
            sd.stop()
            proc = self._sapi_proc
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass


from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from core.compute_runtime import normalize_compute_mode
from core.config import CONTEXT_WINDOW_SIZE, GEMMA_LITERTLM_PATH
from core.config import log_performance
from core.session_logging import trace_event, trace_exception
from llm.system_prompt import DEFAULT_SYSTEM_PROMPT

SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT


def _resolve_model_path() -> Path:
    explicit_path = os.getenv("JARVIS_LLM_MODEL_PATH", "").strip()
    if explicit_path:
        return Path(explicit_path)
    return Path(GEMMA_LITERTLM_PATH)


class GemmaBridge:
    def __init__(self, compute_mode: str | None = None) -> None:
        self.llm = None
        self._unavailable_reason = ""
        self._worker_proc = None
        self._worker_ready = False
        self._worker_lock = threading.Lock()
        self._status_state = "initializing"
        self._status_message = "LLM bridge created."
        self._last_worker_error = ""
        self._thread_count = int(os.getenv("JARVIS_LLM_THREADS", "4"))
        self._worker_ready_timeout_s = float(os.getenv("JARVIS_LLM_WORKER_READY_TIMEOUT_S", "25"))
        self._worker_response_timeout_s = float(os.getenv("JARVIS_LLM_WORKER_RESPONSE_TIMEOUT_S", "45"))
        self._vision_timeout_s = float(os.getenv("JARVIS_LLM_VISION_TIMEOUT_S", "240"))
        self._cpu_max_tokens = int(os.getenv("JARVIS_LLM_MAX_TOKENS_CPU", "160"))
        self._context_turns = max(0, int(os.getenv("JARVIS_LLM_CONTEXT_TURNS", str(CONTEXT_WINDOW_SIZE))))
        selected_mode = compute_mode if compute_mode is not None else os.getenv("JARVIS_COMPUTE_MODE", "auto")
        self._compute_mode = normalize_compute_mode(selected_mode)
        force_worker_env = os.getenv("JARVIS_LLM_SUBPROCESS", "").strip().lower()
        if force_worker_env in {"1", "true", "yes"}:
            self._force_worker = True
        elif force_worker_env in {"0", "false", "no"}:
            self._force_worker = False
        else:
            # Default to subprocess worker mode for stability unless explicitly overridden.
            self._force_worker = True
        self._active_backend_gpu: bool | None = None
        self._active_backend_name = ""
        self._auto_simple_streak = 0
        self._last_switch_monotonic = 0.0
        self._auto_switch_cooldown_s = float(os.getenv("JARVIS_AUTO_COMPUTE_SWITCH_COOLDOWN", "15"))
        self._last_gpu_attempt_failed_at = 0.0
        self._gpu_retry_cooldown_s = float(os.getenv("JARVIS_GPU_RETRY_COOLDOWN", "60"))
        self._disable_in_process = False
        self._model_path = _resolve_model_path()
        self._supports_vision = False
        self._gpu_backend_supported = self._detect_gpu_backend_supported()

        if not self._model_path.exists():
            self._unavailable_reason = f"Missing model file: {self._model_path}"
            self._set_status("unavailable", self._unavailable_reason)
        else:
            self._set_status("initializing", "LLM runtime is warming up.")

        self._trace(
            "initialized",
            compute_mode=self._compute_mode,
            force_worker=bool(self._force_worker),
            model_path=str(self._model_path),
            model_exists=bool(self._model_path.exists()),
            gpu_backend_supported=bool(self._gpu_backend_supported),
            worker_ready_timeout_s=self._worker_ready_timeout_s,
            worker_response_timeout_s=self._worker_response_timeout_s,
            vision_timeout_s=self._vision_timeout_s,
            cpu_max_tokens=self._cpu_max_tokens,
            context_turns=self._context_turns,
        )

    @staticmethod
    def _detect_gpu_backend_supported() -> bool:
        """Cheap, model-free capability check: is a Vulkan runtime present?

        LiteRT-LM's GPU backend runs on Vulkan (Windows/Linux) or Metal (macOS).
        This does not guarantee the Gemma runtime will successfully offload —
        the worker still falls back to CPU on failure — it just avoids
        attempting GPU on machines that plainly can't do it.
        """
        try:
            import ctypes

            if os.name == "nt":
                ctypes.WinDLL("vulkan-1.dll")
                return True
            if sys.platform == "darwin":
                return True  # Metal is always present on macOS.
            ctypes.CDLL("libvulkan.so.1")
            return True
        except Exception:
            return False

    def _set_status(self, state: str, message: str) -> None:
        self._status_state = str(state or "initializing").strip().lower() or "initializing"
        self._status_message = str(message or "").strip()

    def get_status(self) -> Dict[str, Any]:
        mode = "worker" if self._force_worker else "auto"
        if self.llm is not None:
            mode = "in_process"
        elif self._worker_ready:
            mode = "worker"

        return {
            "state": self._status_state,
            "mode": mode,
            "message": self._status_message,
            "supports_vision": bool(self._supports_vision),
            "ready": bool(self.is_ready()),
            "available": bool(self.is_available()),
            "last_error": self._last_worker_error,
        }

    def _trace(self, event: str, **details: Any) -> None:
        trace_event("backend.llm", event, **details)

    def is_available(self) -> bool:
        return self._model_path.exists()

    def supports_gpu_offload(self) -> bool:
        return bool(self._gpu_backend_supported)

    def supports_vision(self) -> bool:
        return bool(self._supports_vision)

    def set_compute_mode(self, mode: str) -> None:
        selected = normalize_compute_mode(mode)
        if selected == self._compute_mode:
            self._trace("set_compute_mode_skipped", mode=selected)
            return

        previous_mode = self._compute_mode
        self._compute_mode = selected
        self._trace("set_compute_mode", previous_mode=previous_mode, active_mode=self._compute_mode)
        self._auto_simple_streak = 0
        if self._compute_mode == "cpu" and self._active_backend_gpu:
            self.close()
        if self._compute_mode == "gpu" and self._active_backend_gpu is False:
            self.close()

    def _select_gpu_for_request(self, user_message: str, device_hint: Optional[str] = None) -> bool:
        from core.compute_runtime import choose_device_for_query

        hint = str(device_hint or "").strip().lower()
        if hint in {"cpu", "gpu"}:
            return hint == "gpu" and self._gpu_backend_supported

        if self._compute_mode == "cpu":
            return False
        if self._compute_mode == "gpu":
            return self._gpu_backend_supported

        target_gpu = choose_device_for_query(user_message, "auto") == "gpu"
        if target_gpu and not self._gpu_backend_supported:
            return False
        if target_gpu:
            self._auto_simple_streak = 0
            return True

        self._auto_simple_streak += 1
        if self._active_backend_gpu and (time.monotonic() - self._last_switch_monotonic) < self._auto_switch_cooldown_s:
            return True
        if self._active_backend_gpu and self._auto_simple_streak < 3:
            return True
        return False

    def _ensure_runtime_for_request(self, user_message: str, device_hint: Optional[str] = None) -> bool:
        if not self.is_available():
            self._trace("ensure_runtime_unavailable", reason=self._unavailable_reason)
            self._set_status("unavailable", self._unavailable_reason)
            return False

        prefer_gpu = self._select_gpu_for_request(user_message, device_hint)
        self._set_status("initializing", "Starting local LLM runtime.")
        self._trace("ensure_runtime_started", prefer_gpu=bool(prefer_gpu), device_hint=device_hint)
        if prefer_gpu and not self._gpu_backend_supported:
            prefer_gpu = False
        if prefer_gpu and self.is_ready() and self._active_backend_gpu is False:
            if (time.monotonic() - self._last_gpu_attempt_failed_at) < self._gpu_retry_cooldown_s:
                prefer_gpu = False

        if self.is_ready() and self._active_backend_gpu == prefer_gpu:
            self._trace("ensure_runtime_reuse_backend", backend=self._active_backend_name, prefer_gpu=bool(prefer_gpu))
            return True

        if self.is_ready() and self._active_backend_gpu != prefer_gpu:
            self._trace("ensure_runtime_switch_backend", from_backend=self._active_backend_name, prefer_gpu=bool(prefer_gpu))
            self.close()

        selected_backend = self._start_runtime(prefer_gpu, device_hint=device_hint)
        if not selected_backend:
            self._trace("ensure_runtime_failed", prefer_gpu=bool(prefer_gpu), reason=self._unavailable_reason)
            self._set_status("error", self._unavailable_reason or "LLM runtime failed to start.")
            return False

        self._active_backend_name = selected_backend
        self._active_backend_gpu = selected_backend == "gpu"
        self._last_switch_monotonic = time.monotonic()
        if prefer_gpu and selected_backend != "gpu":
            self._last_gpu_attempt_failed_at = self._last_switch_monotonic
        elif selected_backend == "gpu":
            self._last_gpu_attempt_failed_at = 0.0
        log_performance("llm_backend_switch", 0.0, f"backend={selected_backend};mode={self._compute_mode}")
        self._trace("ensure_runtime_ready", backend=selected_backend, active_mode=self._compute_mode)
        backend_text = "GPU" if selected_backend == "gpu" else "CPU"
        self._set_status("ready", f"LLM ready ({backend_text}, {selected_backend}).")
        return True

    def _start_runtime(self, prefer_gpu: bool, device_hint: Optional[str] = None) -> str | None:
        prefer_gpu = bool(prefer_gpu and self._gpu_backend_supported)
        self._trace("start_runtime", prefer_gpu=bool(prefer_gpu), force_worker=bool(self._force_worker))
        selected_backend: str | None = None
        if not self._force_worker and not self._disable_in_process:
            selected_backend = self._init_in_process(prefer_gpu=prefer_gpu)
        elif self._disable_in_process:
            self._trace("start_runtime_skip_in_process", reason="disabled_for_session")
        if selected_backend is None:
            worker_prefer_gpu = prefer_gpu
            explicit_cpu = str(device_hint or "").strip().lower() == "cpu"
            if not worker_prefer_gpu and self._compute_mode == "gpu" and not explicit_cpu:
                worker_prefer_gpu = True
            selected_backend = self._start_worker_runtime(prefer_gpu=worker_prefer_gpu)
        self._trace("start_runtime_result", backend=selected_backend or "none")
        return selected_backend

    def _init_in_process(self, prefer_gpu: bool) -> str | None:
        self.llm = None
        self._supports_vision = False
        self._trace("init_in_process_started", prefer_gpu=bool(prefer_gpu))

        try:
            import litert_lm

            litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        except Exception as exc:
            trace_exception("backend.llm", exc, event="init_in_process_import_failed")
            self._unavailable_reason = f"litert-lm-api import failed: {exc}"
            return None

        attempts = [("cpu", litert_lm.Backend.CPU(thread_count=self._thread_count))]
        if prefer_gpu:
            attempts.insert(0, ("gpu", litert_lm.Backend.GPU()))

        last_error = ""
        for name, backend in attempts:
            self._trace("init_in_process_attempt", backend=name)
            try:
                # vision_backend must be set explicitly at construction time,
                # or image messages fail with "Vision executor should not be null".
                self.llm = litert_lm.Engine(str(self._model_path), backend=backend, vision_backend=backend)
                self._supports_vision = True
                self._unavailable_reason = ""
                self._trace("init_in_process_ready", backend=name, supports_vision=True)
                return name
            except Exception as exc:
                trace_exception("backend.llm", exc, event="init_in_process_attempt_failed", backend=name)
                last_error = f"{name} init failed: {exc}"

        self._unavailable_reason = f"Local LLM initialization failed: {last_error}"
        self._trace("init_in_process_failed", error=self._unavailable_reason)
        return None

    def _start_worker_runtime(self, prefer_gpu: bool) -> str | None:
        worker_path = Path(__file__).resolve().parent / "gemma_worker.py"
        if not worker_path.exists():
            self._trace("start_worker_runtime_missing_worker", worker_path=str(worker_path))
            return None

        self._trace("start_worker_runtime_started", worker_path=str(worker_path), prefer_gpu=bool(prefer_gpu))
        self._set_status("initializing", "Starting LLM worker process.")

        env = os.environ.copy()
        env.setdefault("JARVIS_LLM_MODEL_PATH", str(self._model_path))
        env.setdefault("JARVIS_LLM_THREADS", str(self._thread_count))
        env["JARVIS_ENABLE_LLM_GPU"] = "1" if (prefer_gpu and self._gpu_backend_supported) else "0"

        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(worker_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env=env,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
        except Exception as exc:
            trace_exception("backend.llm", exc, event="start_worker_runtime_spawn_failed")
            return None

        payload, read_error = self._read_worker_json_payload(
            proc,
            timeout_s=self._worker_ready_timeout_s,
            phase="startup",
        )
        if payload is None:
            self._terminate_worker(proc)
            if read_error:
                self._unavailable_reason = f"Local LLM worker failed: {read_error}"
                self._last_worker_error = self._unavailable_reason
            self._trace("start_worker_runtime_no_ready_line", error=read_error)
            self._set_status("error", self._unavailable_reason or "LLM worker startup failed.")
            return None

        if payload.get("ok") and payload.get("event") == "ready":
            self._worker_proc = proc
            self._worker_ready = True
            worker_vision_cap = payload.get("supports_vision")
            if isinstance(worker_vision_cap, bool):
                self._supports_vision = worker_vision_cap
            self._unavailable_reason = ""
            backend = str(payload.get("backend") or ("gpu" if prefer_gpu else "cpu"))
            self._trace(
                "start_worker_runtime_ready",
                backend=backend,
                supports_vision=bool(self._supports_vision),
            )
            self._set_status("ready", f"LLM worker ready ({backend}).")
            return backend

        self._terminate_worker(proc)
        err = str(payload.get("error", "Unknown worker startup error"))
        if err:
            self._unavailable_reason = f"Local LLM worker failed: {err}"
            self._last_worker_error = self._unavailable_reason
        self._trace("start_worker_runtime_failed", error=err)
        self._set_status("error", self._unavailable_reason or "LLM worker failed to start.")
        return None

    @staticmethod
    def _terminate_worker(proc: subprocess.Popen) -> None:
        try:
            proc.terminate()
            proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    @staticmethod
    def _readline_with_timeout(stream: Any, timeout_s: float) -> str | None:
        if stream is None:
            return None

        result: Dict[str, str] = {}
        errors: Dict[str, Exception] = {}

        def _reader() -> None:
            try:
                result["line"] = stream.readline()
            except Exception as exc:  # pragma: no cover - defensive guard for pipe failures
                errors["exc"] = exc

        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        thread.join(timeout=max(0.01, float(timeout_s)))
        if thread.is_alive():
            return None
        if "exc" in errors:
            raise errors["exc"]
        return result.get("line", "")

    def _read_worker_json_payload(
        self,
        proc: subprocess.Popen,
        timeout_s: float,
        phase: str,
    ) -> tuple[Dict[str, Any] | None, str]:
        if proc.stdout is None:
            return None, "worker stdout pipe is unavailable"

        deadline = time.monotonic() + max(0.1, float(timeout_s))
        non_json_lines = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, f"timed out waiting for worker {phase} output"

            line = self._readline_with_timeout(proc.stdout, remaining)
            if line is None:
                return None, f"timed out waiting for worker {phase} output"
            if not line:
                return None, f"worker stopped during {phase}"

            text = line.strip()
            if not text:
                continue

            try:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    return payload, ""
            except Exception:
                non_json_lines += 1
                if non_json_lines <= 3:
                    self._trace(
                        "worker_non_json_output",
                        phase=phase,
                        sample=text[:240],
                    )
                continue

    def is_ready(self) -> bool:
        return self.llm is not None or self._worker_ready

    def close(self) -> None:
        self._trace("close_started", worker_ready=bool(self._worker_ready), in_process_loaded=bool(self.llm is not None))
        proc = self._worker_proc
        if proc is not None:
            with self._worker_lock:
                try:
                    if proc.stdin is not None:
                        proc.stdin.write(json.dumps({"type": "shutdown"}) + "\n")
                        proc.stdin.flush()
                    if proc.stdout is not None:
                        self._readline_with_timeout(proc.stdout, 2.0)
                except Exception:
                    pass
                finally:
                    self._terminate_worker(proc)

        if self.llm is not None:
            try:
                self.llm.close()
            except Exception:
                pass

        self._worker_proc = None
        self._worker_ready = False
        self.llm = None
        self._active_backend_gpu = None
        self._active_backend_name = ""
        self._supports_vision = False
        if self.is_available():
            self._set_status("initializing", "LLM runtime stopped. Will restart on demand.")
        else:
            self._set_status("unavailable", self._unavailable_reason or "LLM model files are unavailable.")
        self._trace("close_completed")

    def cancel_current_generation(self) -> bool:
        """
        Best-effort cancellation for an in-flight generation call.
        In worker mode this force-terminates the worker process so blocked reads unblock quickly.
        """
        self._trace(
            "cancel_generation_requested",
            worker_ready=bool(self._worker_ready),
            in_process_loaded=bool(self.llm is not None),
        )

        cancelled = False
        proc = self._worker_proc
        if proc is not None:
            cancelled = True
            try:
                self._terminate_worker(proc)
            except Exception:
                pass
            self._worker_proc = None
            self._worker_ready = False
            self._active_backend_gpu = None
            self._active_backend_name = ""

        if cancelled:
            self._last_worker_error = "Generation cancelled by user."
            self._set_status("initializing", "Generation cancelled. Runtime will restart on demand.")
            self._trace("cancel_generation_applied", cancelled=True)
            return True

        self._trace("cancel_generation_no_worker")
        return False

    def _generate_with_worker(
        self,
        user_message: str,
        context: Optional[List[Dict[str, str]]],
        image_b64: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        timeout_s: Optional[float] = None,
    ) -> str:
        proc = self._worker_proc
        if not self._worker_ready or proc is None:
            self._trace("generate_worker_unavailable")
            return self._unavailable_reason

        response_timeout_s = float(timeout_s) if isinstance(timeout_s, (int, float)) and float(timeout_s) > 0 else float(
            self._worker_response_timeout_s
        )

        started = time.perf_counter()
        self._trace(
            "generate_worker_started",
            prompt_chars=len(str(user_message or "")),
            context_turns=len(context or []),
            image_mode=bool(image_b64),
            custom_system_prompt=bool(str(system_prompt or "").strip()),
            max_tokens=max_tokens,
        )

        with self._worker_lock:
            try:
                if proc.stdin is None or proc.stdout is None:
                    self._worker_ready = False
                    self._trace("generate_worker_failed", reason="worker_io_unavailable")
                    return "Local LLM worker I/O is unavailable."

                payload = {
                    "type": "generate",
                    "user_message": user_message,
                    "context": context or [],
                }
                if image_b64:
                    payload["image_b64"] = image_b64
                if system_prompt:
                    payload["system_prompt"] = str(system_prompt)
                if isinstance(max_tokens, int) and max_tokens > 0:
                    payload["max_tokens"] = int(max_tokens)

                proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
                proc.stdin.flush()

                result, read_error = self._read_worker_json_payload(
                    proc,
                    timeout_s=response_timeout_s,
                    phase="generate",
                )
                if result is None:
                    self._worker_ready = False
                    self._unavailable_reason = (
                        "Local LLM worker timed out while generating a response. "
                        f"Try a shorter prompt or set JARVIS_LLM_WORKER_RESPONSE_TIMEOUT_S higher (current: {response_timeout_s:.0f}s)."
                    )
                    self._trace("generate_worker_failed", reason=read_error or "worker_timeout")
                    self._terminate_worker(proc)
                    self._worker_proc = None
                    return self._unavailable_reason

                if result.get("ok"):
                    text = str(result.get("text", "")).strip()
                    self._trace(
                        "generate_worker_completed",
                        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                        response_chars=len(text),
                    )
                    return text

                error_message = f"Local LLM worker error: {result.get('error', 'unknown error')}"
                self._trace("generate_worker_failed", reason=error_message)
                return error_message
            except Exception as exc:
                trace_exception("backend.llm", exc, event="generate_worker_exception")
                return f"Local LLM worker error: {exc}"

    def _generate_in_process(
        self,
        user_message: str,
        context: Optional[List[Dict[str, str]]],
        image_bytes: Optional[bytes],
        system_prompt: Optional[str],
        max_tokens: int,
    ) -> str:
        import litert_lm

        max_context_turns = self._context_turns
        turns = (context or [])[-max_context_turns:] if max_context_turns > 0 else []
        messages = []
        for turn in turns:
            content = str(turn.get("text", "")).strip()
            if not content:
                continue
            contents = litert_lm.Contents.of(content)
            role = str(turn.get("role", "user"))
            messages.append(litert_lm.Message.model(contents) if role == "assistant" else litert_lm.Message.user(contents))

        active_system_prompt = str(system_prompt or "").strip() or SYSTEM_PROMPT
        conversation = self.llm.create_conversation(
            messages=messages,
            system_message=active_system_prompt,
            sampler_config=litert_lm.SamplerConfig(temperature=0.7, top_p=0.9),
            max_output_tokens=max_tokens,
        )
        try:
            if image_bytes is not None:
                message: Any = litert_lm.Contents.of(user_message, litert_lm.Content.ImageBytes(image_bytes))
            else:
                message = user_message

            response = conversation.send_message(
                message,
                repetition_penalty_config=litert_lm.RepetitionPenaltyConfig(repetition_penalty=1.1),
                max_output_tokens=max_tokens,
            )
            try:
                content = response.get("content") or []
                parts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("text")]
                return "".join(parts).strip() if parts else str(response).strip()
            except Exception:
                return str(response).strip()
        finally:
            try:
                conversation.close()
            except Exception:
                pass

    @staticmethod
    def _is_incomplete_response(text: str) -> bool:
        value = str(text or "").strip()
        if len(value) < 24:
            return False
        if value.endswith((".", "!", "?", '"', "'", "”", "’")):
            return False
        return True

    @staticmethod
    def _merge_continuation(base: str, continuation: str) -> str:
        left = str(base or "").strip()
        right = str(continuation or "").strip()
        if not left:
            return right
        if not right:
            return left
        if right.lower() in left.lower():
            return left
        if left.endswith("-"):
            return f"{left}{right}"
        return f"{left} {right}".strip()

    def _complete_incomplete_worker_response(
        self,
        *,
        original_prompt: str,
        partial_text: str,
        context: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        max_tokens: int,
    ) -> str:
        if not self._is_incomplete_response(partial_text):
            return partial_text

        continuation_prompt = (
            "Continue from your previous answer without repeating yourself. "
            "Finish the unfinished sentence only."
        )
        continuation_context = list(context or [])
        continuation_context.append({"role": "user", "text": str(original_prompt or "")})
        continuation_context.append({"role": "assistant", "text": str(partial_text or "")})

        follow_max_tokens = min(96, max(32, int(max_tokens) // 2))
        continuation = self._generate_with_worker(
            continuation_prompt,
            continuation_context,
            system_prompt=system_prompt,
            max_tokens=follow_max_tokens,
        )
        continuation_text = str(continuation or "").strip()
        if not continuation_text or continuation_text.lower().startswith("local llm"):
            return partial_text
        return self._merge_continuation(partial_text, continuation_text)

    def _complete_incomplete_inprocess_response(
        self,
        *,
        original_prompt: str,
        partial_text: str,
        context: Optional[List[Dict[str, str]]],
        system_prompt: Optional[str],
        max_tokens: int,
    ) -> str:
        if self.llm is None or not self._is_incomplete_response(partial_text):
            return partial_text

        continuation_prompt = (
            "Continue from your previous answer without repeating yourself. "
            "Finish the unfinished sentence only."
        )
        continuation_context = list(context or [])
        continuation_context.append({"role": "user", "text": str(original_prompt or "")})
        continuation_context.append({"role": "assistant", "text": str(partial_text or "")})
        follow_max_tokens = min(96, max(32, int(max_tokens) // 2))

        try:
            continuation_text = self._generate_in_process(
                continuation_prompt,
                continuation_context,
                image_bytes=None,
                system_prompt=system_prompt,
                max_tokens=follow_max_tokens,
            ).strip()
            if not continuation_text:
                return partial_text
            return self._merge_continuation(partial_text, continuation_text)
        except Exception:
            return partial_text

    @staticmethod
    def _max_tokens_for_request(user_message: str, image_mode: bool = False) -> int:
        token_count = len(str(user_message or "").split())
        if image_mode:
            return 128
        if token_count <= 8:
            return 96
        if token_count <= 18:
            return 144
        if token_count <= 32:
            return 220
        return 320

    def generate(
        self,
        user_message: str,
        context: Optional[List[Dict[str, str]]] = None,
        device_hint: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        system_prompt: Optional[str] = None,
        _vision_cpu_retry: bool = False,
    ) -> str:
        started = time.perf_counter()
        self._set_status("processing", "Generating response.")
        self._trace(
            "generate_started",
            prompt_chars=len(str(user_message or "")),
            context_turns=len(context or []),
            device_hint=device_hint,
            image_mode=bool(image_bytes),
            custom_system_prompt=bool(str(system_prompt or "").strip()),
            retained_context_turns=self._context_turns,
        )
        if not self._ensure_runtime_for_request(user_message, device_hint=device_hint):
            self._trace("generate_runtime_unavailable", reason=self._unavailable_reason)
            self._set_status("error", self._unavailable_reason or "LLM runtime is unavailable.")
            return self._unavailable_reason

        if image_bytes and not self._supports_vision:
            self._trace("generate_vision_unavailable")
            self._set_status("error", "Vision runtime is unavailable.")
            return "Vision is unavailable in the current runtime."

        max_tokens = self._max_tokens_for_request(user_message, image_mode=bool(image_bytes))
        if self._active_backend_gpu is False:
            max_tokens = min(max_tokens, max(32, int(self._cpu_max_tokens)))

        if self.llm is None and self._worker_ready:
            worker_timeout_s = self._worker_response_timeout_s
            if image_bytes:
                worker_timeout_s = max(float(self._worker_response_timeout_s), float(self._vision_timeout_s))
            image_b64 = base64.b64encode(image_bytes).decode("ascii") if image_bytes else None
            text = self._generate_with_worker(
                user_message,
                context,
                image_b64=image_b64,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                timeout_s=worker_timeout_s,
            )
            text_value = str(text or "").strip()
            text_lower = text_value.lower()
            worker_error = text_lower.startswith("local llm worker error") or text_lower.startswith(
                "local llm worker timed out"
            )
            if worker_error:
                self._last_worker_error = text_value
                self._trace(
                    "generate_failed",
                    backend="worker",
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                    reason=text_value[:240],
                    max_tokens=max_tokens,
                )
                self._set_status("error", text_value)
                if image_bytes and (not _vision_cpu_retry) and str(device_hint or "").strip().lower() != "cpu":
                    self._trace("generate_worker_retry_cpu_for_vision")
                    try:
                        self.close()
                    except Exception:
                        pass
                    return self.generate(
                        user_message,
                        context=context,
                        device_hint="cpu",
                        image_bytes=image_bytes,
                        system_prompt=system_prompt,
                        _vision_cpu_retry=True,
                    )
                return text

            if not image_bytes:
                text = self._complete_incomplete_worker_response(
                    original_prompt=user_message,
                    partial_text=text_value,
                    context=context,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )

            self._trace(
                "generate_completed",
                backend="worker",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                response_chars=len(str(text or "")),
                max_tokens=max_tokens,
            )
            self._last_worker_error = ""
            self._set_status("ready", "LLM worker response complete.")
            return text

        if self.llm is None:
            self._trace("generate_failed", reason=self._unavailable_reason)
            self._set_status("error", self._unavailable_reason or "LLM backend is unavailable.")
            return self._unavailable_reason

        try:
            text = self._generate_in_process(
                user_message,
                context,
                image_bytes=image_bytes,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
            )
            if not image_bytes:
                text = self._complete_incomplete_inprocess_response(
                    original_prompt=user_message,
                    partial_text=str(text or ""),
                    context=context,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                )
            self._trace(
                "generate_completed",
                backend="in_process",
                elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
                response_chars=len(str(text or "")),
                max_tokens=max_tokens,
            )
            self._last_worker_error = ""
            self._set_status("ready", "LLM response complete.")
            return text
        except Exception as exc:
            trace_exception("backend.llm", exc, event="generate_exception")
            self._last_worker_error = str(exc)
            self._set_status("error", f"Local LLM error: {exc}")
            return f"Local LLM error: {exc}"

    def prewarm(self, image_mode: bool = False) -> bool:
        """
        Warm up the runtime to reduce first-response latency.
        """
        if not self.is_available():
            self._set_status("unavailable", self._unavailable_reason or "LLM model file is missing.")
            return False

        self._set_status("initializing", "Prewarming local LLM runtime.")
        try:
            if not self._ensure_runtime_for_request("prewarm", device_hint="cpu"):
                self._set_status("error", self._unavailable_reason or "LLM prewarm failed.")
                return False

            if self._worker_ready and self.llm is None:
                self._set_status("ready", "LLM worker prewarm complete.")
            else:
                self._set_status("ready", "LLM prewarm complete.")
            return True
        except Exception as exc:
            self._last_worker_error = str(exc)
            self._set_status("error", f"LLM prewarm failed: {exc}")
            return False

    def generate_stream(
        self,
        user_message: str,
        context: Optional[List[Dict[str, str]]] = None,
        device_hint: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        # No caller currently needs token-level streaming; kept as a thin
        # wrapper around generate() rather than building chunked streaming
        # over the worker's newline-delimited JSON pipe.
        yield self.generate(
            user_message,
            context=context,
            device_hint=device_hint,
            image_bytes=image_bytes,
            system_prompt=system_prompt,
        )

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

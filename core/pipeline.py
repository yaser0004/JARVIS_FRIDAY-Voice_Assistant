from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict

from PyQt6.QtCore import QObject, pyqtSignal

from core.compute_runtime import (
    apply_compute_environment,
    choose_device_for_query,
    detect_compute_capabilities,
    normalize_compute_mode,
)
from core.context_manager import ContextManager
from core.config import log_performance
from core.runtime_settings import load_runtime_settings, save_runtime_settings
from core.session_logging import trace_event, trace_exception
from llm.qwen_bridge import QwenBridge
from memory.sqlite_store import SQLiteStore
from memory.vector_store import VectorStore
from nlp import preprocessor
from nlp.conversation_normalizer import normalize_command_text
from nlp.entity_extractor import EntityExtractor
from nlp.intent_classifier import IntentClassifier
from nlp.router import Router
from speech.stt import SpeechToText
from speech.tts import TextToSpeech
from speech.wake_word import WakeWordDetector
from speech.wakeword_config import load_wakeword_config, save_wakeword_config


def _normalize_verbosity_mode(verbosity: str | None) -> str:
    """Normalize verbosity values to one of: brief, normal, detailed."""
    value = (verbosity or "normal").strip().lower()
    aliases = {
        "short": "brief",
        "concise": "brief",
        "terse": "brief",
        "default": "normal",
        "regular": "normal",
        "long": "detailed",
        "verbose": "detailed",
        "thorough": "detailed",
        "comprehensive": "detailed",
    }
    value = aliases.get(value, value)
    if value not in {"brief", "normal", "detailed"}:
        return "normal"
    return value


def _split_sentences(text: str) -> list[str]:
    chunks = re.split(r"(?<=[.!?])\s+", text)
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def enforce_response_verbosity(response: str, verbosity: str | None) -> str:
    """
    Apply hard response-size limits so local LLM responses return faster.
    """
    if not response:
        return ""

    mode = _normalize_verbosity_mode(verbosity)
    cleaned = re.sub(r"\s+", " ", response).strip()
    if not cleaned:
        return ""

    limits = {
        "brief": {"max_sentences": 2, "max_words": 60, "max_chars": 380},
        "normal": {"max_sentences": 4, "max_words": 180, "max_chars": 1200},
        "detailed": {"max_sentences": 12, "max_words": 520, "max_chars": 4200},
    }
    limit = limits[mode]

    sentences = _split_sentences(cleaned)
    if sentences:
        clipped = " ".join(sentences[: int(limit["max_sentences"])])
    else:
        clipped = cleaned

    words = clipped.split()
    if len(words) > int(limit["max_words"]):
        clipped = " ".join(words[: int(limit["max_words"])]).rstrip(" ,;:-")
        if clipped and clipped[-1] not in ".!?":
            clipped += "."

    if len(clipped) > int(limit["max_chars"]):
        hard_clip = clipped[: int(limit["max_chars"])].rstrip(" ,;:-")
        punct_idx = max(hard_clip.rfind("."), hard_clip.rfind("!"), hard_clip.rfind("?"))
        if punct_idx >= int(limit["max_chars"] * 0.6):
            clipped = hard_clip[: punct_idx + 1].strip()
        else:
            words = hard_clip.split()
            clipped = " ".join(words[:-1]).rstrip(" ,;:-") if len(words) > 1 else hard_clip
            if clipped and clipped[-1] not in ".!?":
                clipped += "."

    return clipped


class JarvisPipeline(QObject):
    pipeline_state_changed = pyqtSignal(str)
    new_message = pyqtSignal(str, str)
    intent_classified = pyqtSignal(str, float)
    intent_diagnostics = pyqtSignal(dict)
    initialization_progress = pyqtSignal(str, int)
    wakeword_availability_changed = pyqtSignal(bool, str)
    ready = pyqtSignal()

    def __init__(self) -> None:
        super().__init__()
        self.current_state = "IDLE"
        self._initialized = False
        self._cancel_requested = False
        self._intent_model_name = os.getenv("JARVIS_INTENT_MODEL", "DistilBERT")
        self._web_verified_mode = False
        self._wakeword_config = load_wakeword_config()
        self._runtime_settings = load_runtime_settings()
        self.tts_enabled = bool(self._runtime_settings.get("tts_enabled", True))
        self._tts_profile = str(self._runtime_settings.get("tts_profile", "female") or "female").strip().lower()
        if self._tts_profile not in {"female", "male"}:
            self._tts_profile = "female"
        self._response_verbosity = _normalize_verbosity_mode(self._runtime_settings.get("response_verbosity"))
        self._compute_mode = normalize_compute_mode(self._runtime_settings.get("compute_mode"))
        apply_compute_environment(self._compute_mode)
        self._compute_capabilities: Dict[str, Any] = {}
        self._compute_caps_lock = threading.Lock()
        self._compute_caps_refresh_running = False
        self._wakeword_initializing = False
        self._wake_session_lock = threading.Lock()
        self._wake_session_active = False
        self._memory_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis_memory")

        self.intent_classifier: IntentClassifier | None = None
        self.entity_extractor: EntityExtractor | None = None
        self.router: Router | None = None
        self.stt: SpeechToText | None = None
        self.tts: TextToSpeech | None = None
        self.wake_word: WakeWordDetector | None = None
        self.llm: QwenBridge | None = None
        self.context: ContextManager | None = None
        self.sqlite: SQLiteStore | None = None
        self.vector_store: VectorStore | None = None

        self._trace(
            "initialized",
            compute_mode=self._compute_mode,
            tts_enabled=bool(self.tts_enabled),
            tts_profile=self._tts_profile,
            response_verbosity=self._response_verbosity,
            wakeword_enabled=bool(self._wakeword_config.get("enabled", False)),
        )

    def _trace(self, event: str, **details: Any) -> None:
        trace_event("backend.pipeline", event, **details)

    def _bootstrap_indexes_async(self) -> None:
        def worker() -> None:
            self._trace("index_bootstrap_started")
            try:
                from actions import app_control, media_control

                app_control.ensure_app_index(force_rescan=not app_control.APP_INDEX_PATH.exists())
                media_control.ensure_media_index(force_rescan=not media_control.MEDIA_INDEX_PATH.exists())
                self._trace("index_bootstrap_completed")
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="index_bootstrap_failed")
                log_performance("index_bootstrap_error", 0.0, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _persist_turn_to_stores(
        self,
        *,
        role: str,
        text: str,
        intent: str,
        confidence: float,
        embed_context_recent: bool = False,
    ) -> None:
        self._trace(
            "persist_turn_to_stores_started",
            role=role,
            text_chars=len(text),
            intent=intent,
            confidence=confidence,
            embed_context_recent=bool(embed_context_recent),
        )
        if self.sqlite is not None:
            try:
                self.sqlite.save_turn(role, text, intent, confidence)
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="sqlite_store_error")
                log_performance("sqlite_store_error", 0.0, str(exc))

        if self.vector_store is not None:
            try:
                self.vector_store.add_memory(text, {"role": role, "intent": intent})
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="vector_store_error")
                log_performance("vector_store_error", 0.0, str(exc))

        if embed_context_recent and self.context is not None and hasattr(self.context, "embed_recent_missing"):
            try:
                self.context.embed_recent_missing(limit=4)
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="context_embed_error")
                log_performance("context_embed_error", 0.0, str(exc))

        self._trace("persist_turn_to_stores_completed", role=role, intent=intent)

    def _persist_turn_async(
        self,
        *,
        role: str,
        text: str,
        intent: str,
        confidence: float,
        embed_context_recent: bool = False,
    ) -> None:
        self._trace(
            "persist_turn_async_queued",
            role=role,
            text_chars=len(text),
            intent=intent,
            confidence=confidence,
            embed_context_recent=bool(embed_context_recent),
        )

        def worker() -> None:
            self._persist_turn_to_stores(
                role=role,
                text=text,
                intent=intent,
                confidence=confidence,
                embed_context_recent=embed_context_recent,
            )

        try:
            self._memory_executor.submit(worker)
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="persist_turn_async_submit_failed")
            log_performance("memory_executor_error", 0.0, str(exc))
            worker()

    def initialize(self) -> None:
        if self._initialized:
            self._trace("initialize_skipped_already_initialized")
            return

        debug_startup = os.getenv("JARVIS_DEV", "").strip().lower() in {"1", "true", "yes"}
        self._trace("initialize_started", debug_startup=debug_startup)
        try:
            steps = [
                (
                    "Loading intent classifier",
                    lambda: IntentClassifier(model_name=self._intent_model_name, compute_mode=self._compute_mode),
                ),
                ("Loading entity extractor", lambda: EntityExtractor()),
                ("Loading speech-to-text", lambda: SpeechToText(compute_mode=self._compute_mode)),
                ("Loading text-to-speech", lambda: TextToSpeech(profile=self._tts_profile)),
                ("Preparing LLM bridge", lambda: QwenBridge(compute_mode=self._compute_mode)),
                ("Loading conversation memory", lambda: ContextManager()),
                ("Loading SQLite store", lambda: SQLiteStore()),
                ("Loading vector store", lambda: VectorStore()),
            ]

            loaded: Dict[str, Any] = {}
            for idx, (label, builder) in enumerate(steps, start=1):
                self._trace("initialize_step_start", step=label, index=idx, total=len(steps))
                if debug_startup:
                    print(f"JARVIS pipeline: init start -> {label}", flush=True)
                try:
                    loaded[label] = builder()
                    status = label
                    self._trace("initialize_step_ok", step=label, index=idx)
                    if debug_startup:
                        print(f"JARVIS pipeline: init ok -> {label}", flush=True)
                except Exception as exc:
                    trace_exception("backend.pipeline", exc, event="initialize_step_failed", step=label)
                    loaded[label] = None
                    status = f"{label} (fallback mode)"
                    log_performance("pipeline_init_error", 0.0, f"{label}: {exc}")
                    if debug_startup:
                        print(f"JARVIS pipeline: init fallback -> {label}: {exc}", flush=True)
                progress = int((idx / len(steps)) * 100)
                self.initialization_progress.emit(status, progress)

            self.intent_classifier = loaded["Loading intent classifier"]
            self.entity_extractor = loaded["Loading entity extractor"]
            self.stt = loaded["Loading speech-to-text"]
            self.tts = loaded["Loading text-to-speech"]
            self.llm = loaded["Preparing LLM bridge"]
            self.context = loaded["Loading conversation memory"]
            self.sqlite = loaded["Loading SQLite store"]
            self.vector_store = loaded["Loading vector store"]
            self._trace(
                "initialize_modules_assigned",
                intent_classifier=bool(self.intent_classifier),
                entity_extractor=bool(self.entity_extractor),
                stt=bool(self.stt),
                tts=bool(self.tts),
                llm=bool(self.llm),
                context=bool(self.context),
                sqlite=bool(self.sqlite),
                vector_store=bool(self.vector_store),
            )

            try:
                self.router = Router(llm=self.llm, cancel_callback=self.cancel_current_action)
                self.router.set_realtime_web_enabled(self._web_verified_mode)
                if hasattr(self.router, "set_response_verbosity"):
                    self.router.set_response_verbosity(self._response_verbosity)
                self._trace("initialize_router_ready")
            except Exception as exc:
                self.router = None
                trace_exception("backend.pipeline", exc, event="initialize_router_failed")
                log_performance("pipeline_init_error", 0.0, f"router:{exc}")

            self._apply_compute_mode_to_modules()
            self._refresh_compute_capabilities_async()
            self._warm_stt_async()
            self._init_wakeword_async(auto_start=bool(self._wakeword_config.get("enabled", False)))
            self._bootstrap_indexes_async()
            self._prewarm_llm_startup()
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="initialize_fatal")
            log_performance("pipeline_init_fatal", 0.0, str(exc))
        finally:
            self._initialized = True
            self._trace("initialize_finished", ready_emitted=True)
            self.ready.emit()

    def initialize_async(self) -> None:
        self._trace("initialize_async_spawned")
        threading.Thread(target=self.initialize, daemon=True).start()

    def start_wake_word(self) -> None:
        with self._wake_session_lock:
            if self._wake_session_active:
                self._trace("start_wake_word_skipped", reason="wake_session_active")
                return
        if self.wake_word is None:
            self._trace("start_wake_word_no_detector", initializing=bool(self._wakeword_initializing))
            if not self._wakeword_initializing:
                self._init_wakeword_async(auto_start=True)
            return
        self._trace("start_wake_word_requested")
        if hasattr(self.wake_word, "set_detection_paused"):
            try:
                self.wake_word.set_detection_paused(False)
            except Exception:
                pass
        self.wake_word.start()

    def stop_wake_word(self) -> None:
        if self.wake_word is not None:
            self._trace("stop_wake_word_requested")
            self.wake_word.stop()

    def shutdown(self) -> None:
        self._trace("shutdown_started")

        try:
            self.stop_wake_word()
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="shutdown_stop_wake_failed")

        if self.tts is not None:
            try:
                self.tts.stop()
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="shutdown_tts_stop_failed")

        if self.llm is not None and hasattr(self.llm, "close"):
            try:
                self.llm.close()
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="shutdown_llm_close_failed")

        try:
            router_llm = getattr(self.router, "llm", None)
            if router_llm is not None and router_llm is not self.llm and hasattr(router_llm, "close"):
                router_llm.close()
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="shutdown_router_llm_close_failed")

        try:
            self._memory_executor.shutdown(wait=False, cancel_futures=False)
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="shutdown_executor_failed")

        self._trace("shutdown_completed")

    def _bind_wakeword_signals(self, detector: WakeWordDetector) -> None:
        detector.transcript_ready.connect(self._handle_wake_transcript)
        detector.state_changed.connect(self._set_state)
        if hasattr(detector, "wake_acknowledged"):
            detector.wake_acknowledged.connect(self._handle_wake_acknowledged)
        self._trace("wakeword_signals_bound")

    def _warm_stt_async(self) -> None:
        if self.stt is None or not hasattr(self.stt, "warmup"):
            self._trace("stt_warmup_skipped")
            return

        self._trace("stt_warmup_started")

        def worker() -> None:
            try:
                self.stt.warmup()
                self._trace("stt_warmup_completed")
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="stt_warmup_failed")
                log_performance("stt_warmup_error", 0.0, str(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _prewarm_llm_startup(self) -> None:
        if self.llm is None or not hasattr(self.llm, "prewarm"):
            self._trace("llm_prewarm_skipped")
            return

        prefer_vision = os.getenv("JARVIS_LLM_PREWARM_VISION", "1").strip().lower() in {"1", "true", "yes"}
        self._trace("llm_prewarm_started", prefer_vision=bool(prefer_vision))

        try:
            ready = bool(self.llm.prewarm(image_mode=prefer_vision))
            self._trace("llm_prewarm_completed", ready=ready, vision_mode=bool(prefer_vision))
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="llm_prewarm_failed")
            log_performance("llm_prewarm_error", 0.0, str(exc))

    def _init_wakeword_async(self, auto_start: bool = False) -> None:
        self._wakeword_initializing = True
        self._trace("wakeword_init_started", auto_start=bool(auto_start))

        if self.wake_word is not None:
            try:
                self.wake_word.stop()
            except Exception:
                pass
        self.wake_word = None

        cfg = dict(self._wakeword_config)
        stt_ref = self.stt

        def worker() -> None:
            message = ""
            available = False
            detector = None
            try:
                os.environ["JARVIS_ENABLE_WAKEWORD"] = "1" if cfg.get("enabled", False) else "0"
                detector = WakeWordDetector(
                    stt=stt_ref,
                    sensitivity=float(cfg.get("sensitivity", 0.5)),
                    activation_phrases=list(cfg.get("activation_phrases", [])),
                    strict_phrase_prefix=bool(cfg.get("strict_phrase_prefix", False)),
                    follow_up_timeout=int(cfg.get("follow_up_timeout", 10)),
                )
                self.wake_word = detector
                self._bind_wakeword_signals(detector)
                if hasattr(detector, "is_available"):
                    try:
                        available = bool(detector.is_available())
                    except Exception:
                        available = bool(getattr(detector, "model", None) is not None)
                else:
                    available = bool(getattr(detector, "model", None) is not None)
                if auto_start and available:
                    detector.start()
                    message = "Wake-word listener is running in the background."
                elif auto_start and not available:
                    reason = ""
                    if hasattr(detector, "unavailable_reason"):
                        reason = str(detector.unavailable_reason() or "")
                    message = reason or "Wake-word listener is unavailable in this runtime."
                elif available:
                    message = "Wake-word listener is available."
                else:
                    reason = ""
                    if hasattr(detector, "unavailable_reason"):
                        reason = str(detector.unavailable_reason() or "")
                    message = reason or "Wake-word listener is unavailable in this runtime."
            except Exception as exc:
                self.wake_word = None
                available = False
                message = f"Wake-word initialization failed: {exc}"
                trace_exception("backend.pipeline", exc, event="wakeword_init_failed")
                log_performance("wakeword_init_error", 0.0, str(exc))
            finally:
                self._wakeword_initializing = False
                self._trace(
                    "wakeword_init_finished",
                    available=bool(available),
                    auto_start=bool(auto_start),
                    message=message,
                )
                self.wakeword_availability_changed.emit(available, message)

        threading.Thread(target=worker, daemon=True).start()

    def set_tts_enabled(self, enabled: bool) -> None:
        self.tts_enabled = bool(enabled)
        self._trace("set_tts_enabled", enabled=self.tts_enabled)
        if not self.tts_enabled and self.tts is not None:
            try:
                self.tts.stop()
            except Exception:
                pass
        self._runtime_settings["tts_enabled"] = self.tts_enabled
        self._runtime_settings = save_runtime_settings(self._runtime_settings)

    def get_tts_settings(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.tts_enabled),
            "profile": self._tts_profile,
            "profiles": ["female", "male"],
        }

    def get_response_settings(self) -> Dict[str, Any]:
        return {
            "verbosity": self._response_verbosity,
            "modes": ["brief", "normal", "detailed"],
        }

    def set_tts_profile(self, profile: str) -> Dict[str, Any]:
        selected = str(profile or "").strip().lower()
        self._tts_profile = "male" if selected == "male" else "female"
        self._trace("set_tts_profile", selected=self._tts_profile)
        self._runtime_settings["tts_profile"] = self._tts_profile
        self._runtime_settings = save_runtime_settings(self._runtime_settings)

        if self.tts is not None and hasattr(self.tts, "set_profile"):
            try:
                self.tts.set_profile(self._tts_profile)
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="set_tts_profile_apply_failed")
                log_performance("tts_profile_apply_error", 0.0, str(exc))

        return self.get_tts_settings()

    def set_response_verbosity(self, verbosity: str) -> Dict[str, Any]:
        self._response_verbosity = _normalize_verbosity_mode(verbosity)
        self._trace("set_response_verbosity", verbosity=self._response_verbosity)
        self._runtime_settings["response_verbosity"] = self._response_verbosity
        self._runtime_settings = save_runtime_settings(self._runtime_settings)
        if self.router is not None and hasattr(self.router, "set_response_verbosity"):
            try:
                self.router.set_response_verbosity(self._response_verbosity)
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="set_response_verbosity_router_failed")
                log_performance("router_verbosity_error", 0.0, str(exc))
        return self.get_response_settings()

    def is_realtime_web_enabled(self) -> bool:
        return bool(self._web_verified_mode)

    def set_realtime_web_enabled(self, enabled: bool) -> None:
        self._web_verified_mode = bool(enabled)
        self._trace("set_realtime_web_enabled", enabled=self._web_verified_mode)
        if self.router is not None:
            self.router.set_realtime_web_enabled(self._web_verified_mode)

    def get_compute_settings(self) -> Dict[str, Any]:
        if not self._compute_capabilities:
            self._refresh_compute_capabilities_async()
        return {
            "mode": self._compute_mode,
            "available_modes": ["auto", "cpu", "gpu"],
            "capabilities": dict(self._compute_capabilities),
        }

    def set_compute_mode(self, mode: str) -> Dict[str, Any]:
        previous_mode = self._compute_mode
        self._compute_mode = normalize_compute_mode(mode)
        self._trace("set_compute_mode", requested_mode=mode, previous_mode=previous_mode, active_mode=self._compute_mode)
        self._runtime_settings["compute_mode"] = self._compute_mode
        self._runtime_settings = save_runtime_settings(self._runtime_settings)
        apply_compute_environment(self._compute_mode)
        self._apply_compute_mode_to_modules()
        self._refresh_compute_capabilities()
        return self.get_compute_settings()

    def _refresh_compute_capabilities(self) -> None:
        self._compute_caps_refresh_running = True
        self._trace("compute_capabilities_refresh_started")
        try:
            caps = detect_compute_capabilities()
            with self._compute_caps_lock:
                self._compute_capabilities = dict(caps)
            self._trace("compute_capabilities_refresh_completed", capabilities=dict(self._compute_capabilities))
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="compute_capabilities_refresh_failed")
            log_performance("compute_caps_error", 0.0, str(exc))
        finally:
            self._compute_caps_refresh_running = False

    def _refresh_compute_capabilities_async(self) -> None:
        if self._compute_caps_refresh_running:
            self._trace("compute_capabilities_refresh_async_skipped")
            return
        self._trace("compute_capabilities_refresh_async_spawned")
        threading.Thread(target=self._refresh_compute_capabilities, daemon=True).start()

    def _apply_compute_mode_to_modules(self) -> None:
        if self.intent_classifier is not None and hasattr(self.intent_classifier, "set_compute_mode"):
            try:
                self.intent_classifier.set_compute_mode(self._compute_mode)
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="apply_compute_mode_intent_failed")
                log_performance("compute_mode_apply_error", 0.0, f"intent_classifier:{exc}")

        if self.stt is not None and hasattr(self.stt, "set_compute_mode"):
            try:
                self.stt.set_compute_mode(self._compute_mode)
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="apply_compute_mode_stt_failed")
                log_performance("compute_mode_apply_error", 0.0, f"stt:{exc}")

        if self.llm is not None and hasattr(self.llm, "set_compute_mode"):
            try:
                self.llm.set_compute_mode(self._compute_mode)
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="apply_compute_mode_llm_failed")
                log_performance("compute_mode_apply_error", 0.0, f"llm:{exc}")

        if self.router is not None:
            self.router.llm = self.llm
        self._trace("apply_compute_mode_completed", active_mode=self._compute_mode)

    def get_wakeword_settings(self) -> Dict[str, Any]:
        return dict(self._wakeword_config)

    def get_wakeword_status(self) -> Dict[str, Any]:
        available = self.is_wakeword_available()
        reason = ""
        if self.wake_word is not None and hasattr(self.wake_word, "unavailable_reason"):
            try:
                reason = str(self.wake_word.unavailable_reason() or "")
            except Exception:
                reason = ""
        return {
            "enabled": bool(self._wakeword_config.get("enabled", False)),
            "available": available,
            "initializing": bool(self._wakeword_initializing),
            "reason": reason,
        }

    def get_stt_status(self) -> Dict[str, Any]:
        if self.stt is None:
            return {
                "available": False,
                "initialized": False,
                "reason": "Speech-to-text module is unavailable.",
                "device": "cpu",
            }

        if hasattr(self.stt, "get_status"):
            try:
                return dict(self.stt.get_status())
            except Exception as exc:
                return {
                    "available": False,
                    "initialized": False,
                    "reason": f"STT status probe failed: {exc}",
                    "device": "cpu",
                }

        return {
            "available": True,
            "initialized": bool(getattr(self.stt, "model", None) is not None),
            "reason": "",
            "device": str(getattr(self.stt, "device", "cpu") or "cpu"),
        }

    def get_llm_status(self) -> Dict[str, Any]:
        bridge = self.llm
        if bridge is None and self.router is not None:
            bridge = getattr(self.router, "llm", None)

        router_vision_status: Dict[str, Any] = {}
        if self.router is not None and hasattr(self.router, "get_vision_runtime_status"):
            try:
                router_vision_status = dict(self.router.get_vision_runtime_status())
            except Exception:
                router_vision_status = {}

        def _apply_router_vision(payload: Dict[str, Any]) -> Dict[str, Any]:
            preferred = str(router_vision_status.get("preferred_backend") or "").strip().lower()
            if preferred:
                payload["vision_backend_preference"] = preferred

            ollama_status = router_vision_status.get("ollama")
            if isinstance(ollama_status, dict):
                ollama_available = bool(ollama_status.get("available", False))
                payload["ollama_vision_available"] = ollama_available
                model_name = str(ollama_status.get("model") or "").strip()
                if model_name:
                    payload["ollama_vision_model"] = model_name
                message = str(ollama_status.get("message") or "").strip()
                if message:
                    payload["ollama_vision_message"] = message
                if preferred == "ollama":
                    payload["supports_vision"] = ollama_available
                    payload["vision_backend"] = "ollama"
                elif preferred == "auto" and ollama_available:
                    payload["supports_vision"] = True
                    payload["vision_backend"] = "ollama"

            if "vision_backend" not in payload:
                payload["vision_backend"] = "qwen2.5-vl"
            return payload

        if bridge is None:
            status = {
                "state": "unavailable",
                "mode": "none",
                "message": "LLM bridge is unavailable.",
                "supports_vision": False,
            }
            return _apply_router_vision(status)

        if hasattr(bridge, "get_status"):
            try:
                status = dict(bridge.get_status())
                if "state" not in status:
                    status["state"] = "ready" if bool(bridge.is_ready()) else "initializing"
                return _apply_router_vision(status)
            except Exception:
                pass

        ready = bool(bridge.is_ready()) if hasattr(bridge, "is_ready") else False
        available = bool(bridge.is_available()) if hasattr(bridge, "is_available") else ready
        supports_vision = bool(bridge.supports_vision()) if hasattr(bridge, "supports_vision") else False
        status = {
            "state": "ready" if ready else ("initializing" if available else "unavailable"),
            "mode": "worker" if ready and getattr(bridge, "llm", None) is None else "in_process",
            "message": "LLM ready." if ready else "LLM is warming up.",
            "supports_vision": supports_vision,
        }
        return _apply_router_vision(status)

    def update_wakeword_settings(
        self,
        *,
        enabled: bool | None = None,
        sensitivity: float | None = None,
        activation_phrases: list[str] | None = None,
        strict_phrase_prefix: bool | None = None,
        auto_restart_after_response: bool | None = None,
        follow_up_after_response: bool | None = None,
        follow_up_timeout: int | None = None,
        max_followup_turns: int | None = None,
    ) -> Dict[str, Any]:
        self._trace(
            "update_wakeword_settings_requested",
            enabled=enabled,
            sensitivity=sensitivity,
            activation_phrases=activation_phrases,
            strict_phrase_prefix=strict_phrase_prefix,
            auto_restart_after_response=auto_restart_after_response,
            follow_up_after_response=follow_up_after_response,
            follow_up_timeout=follow_up_timeout,
            max_followup_turns=max_followup_turns,
        )
        updated = dict(self._wakeword_config)
        if enabled is not None:
            updated["enabled"] = bool(enabled)
        if sensitivity is not None:
            updated["sensitivity"] = float(sensitivity)
        if activation_phrases is not None:
            updated["activation_phrases"] = [str(item).strip().lower() for item in activation_phrases if str(item).strip()]
        if strict_phrase_prefix is not None:
            updated["strict_phrase_prefix"] = bool(strict_phrase_prefix)
        if auto_restart_after_response is not None:
            updated["auto_restart_after_response"] = bool(auto_restart_after_response)
        if follow_up_after_response is not None:
            updated["follow_up_after_response"] = bool(follow_up_after_response)
        if follow_up_timeout is not None:
            updated["follow_up_timeout"] = int(follow_up_timeout)
        if max_followup_turns is not None:
            updated["max_followup_turns"] = int(max_followup_turns)

        self._wakeword_config = save_wakeword_config(updated)
        self._trace("update_wakeword_settings_saved", wakeword_config=dict(self._wakeword_config))
        self._init_wakeword_async(auto_start=bool(self._wakeword_config.get("enabled", False)))

        return dict(self._wakeword_config)

    def set_intent_model(self, model_name: str) -> None:
        selected = (model_name or "").strip()
        if not selected:
            self._trace("set_intent_model_skipped", reason="empty_model_name")
            return

        self._intent_model_name = selected
        self._trace("set_intent_model", model_name=selected, classifier_loaded=bool(self.intent_classifier))
        if self.intent_classifier is None:
            return

        try:
            self.intent_classifier.set_model(selected)
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="set_intent_model_failed", model_name=selected)
            log_performance("intent_model_switch_error", 0.0, str(exc))

    def _set_state(self, state: str) -> None:
        previous = self.current_state
        self.current_state = state
        self._trace("state_changed", previous_state=previous, new_state=state)
        self.pipeline_state_changed.emit(state)

    def _require_initialized(self) -> None:
        if not self._initialized:
            self._trace("require_initialized_failed")
            raise RuntimeError("Pipeline is not initialized yet.")

    @staticmethod
    def _normalize_wake_phrase(text: str) -> str:
        chars = [ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in str(text or "")]
        return " ".join("".join(chars).split())

    def _wake_activation_phrases(self) -> list[str]:
        phrases = self._wakeword_config.get("activation_phrases", [])
        if not isinstance(phrases, list):
            return []
        return [str(item).strip().lower() for item in phrases if str(item).strip()]

    def _strip_activation_phrase(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""

        lowered = cleaned.lower()
        for phrase in self._wake_activation_phrases():
            if lowered.startswith(phrase):
                tail = cleaned[len(phrase) :].strip(" ,.!?;:")
                return tail or cleaned
        return cleaned

    def _looks_like_wake_only(self, text: str) -> bool:
        normalized = self._normalize_wake_phrase(text)
        if not normalized:
            return True
        for phrase in self._wake_activation_phrases():
            if normalized == self._normalize_wake_phrase(phrase):
                return True
        return False

    def _wake_follow_up_timeout(self) -> int:
        try:
            timeout = int(self._wakeword_config.get("follow_up_timeout", 8))
        except Exception:
            timeout = 8
        return max(3, min(20, timeout))

    def _wake_max_followup_turns(self) -> int:
        try:
            turns = int(self._wakeword_config.get("max_followup_turns", 1))
        except Exception:
            turns = 1
        return max(0, min(3, turns))

    def _listen_for_post_response_follow_up(self) -> str:
        if self.stt is None:
            self._trace("wake_followup_skipped", reason="stt_unavailable")
            return ""

        self._trace("wake_followup_listen_start", timeout=self._wake_follow_up_timeout())
        self._set_state("LISTENING")
        transcript = self.stt.listen_once(timeout=self._wake_follow_up_timeout())
        text = str(transcript.get("text", "")).strip()
        if not text:
            stt_error = str(transcript.get("error", "")).strip()
            if stt_error:
                self._trace("wake_followup_listen_error", error=stt_error)
                log_performance("wake_followup_stt_error", 0.0, stt_error)
            self._set_state("IDLE")
            self._trace("wake_followup_listen_empty")
            return ""

        filtered = self._strip_activation_phrase(text)
        if not filtered or self._looks_like_wake_only(filtered):
            self._set_state("IDLE")
            self._trace("wake_followup_filtered_empty", transcript_text=text)
            return ""

        self._trace("wake_followup_received", transcript_text=text, filtered_text=filtered)

        return filtered

    def _wait_for_tts_completion(self, tts_future: Future[Any] | None) -> None:
        if tts_future is None:
            self._trace("tts_wait_skipped", reason="future_none")
            return
        self._trace("tts_wait_started")
        try:
            tts_future.result(timeout=120)
            self._trace("tts_wait_completed")
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="tts_wait_failed")
            log_performance("tts_wait_error", 0.0, str(exc))

    def _resume_wake_listener_after_voice_session(self) -> None:
        if not bool(self._wakeword_config.get("enabled", False)):
            self._trace("wake_resume_skipped", reason="wakeword_disabled")
            return
        if not bool(self._wakeword_config.get("auto_restart_after_response", True)):
            self._trace("wake_resume_skipped", reason="auto_restart_disabled")
            return

        try:
            if self.wake_word is not None and hasattr(self.wake_word, "set_detection_paused"):
                self.wake_word.set_detection_paused(False)
            self.start_wake_word()
            self._trace("wake_resume_requested")
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="wake_resume_failed")
            log_performance("wakeword_resume_error", 0.0, str(exc))

    def _wake_voice_session_worker(self, initial_text: str) -> None:
        pending_text = str(initial_text or "").strip()
        follow_up_enabled = bool(self._wakeword_config.get("follow_up_after_response", True))
        remaining_follow_ups = self._wake_max_followup_turns()
        self._trace(
            "wake_session_started",
            initial_text=pending_text,
            follow_up_enabled=follow_up_enabled,
            remaining_follow_ups=remaining_follow_ups,
        )

        try:
            while pending_text:
                self._trace("wake_session_turn", pending_text=pending_text, remaining_follow_ups=remaining_follow_ups)
                result = self.process_text(pending_text, capture_tts_future=True)
                tts_future = result.get("_tts_future")
                if isinstance(tts_future, Future):
                    self._wait_for_tts_completion(tts_future)

                if self._cancel_requested:
                    break

                if not follow_up_enabled or remaining_follow_ups <= 0:
                    self._trace("wake_session_followup_stopped", reason="followup_disabled_or_exhausted")
                    break

                follow_up_text = self._listen_for_post_response_follow_up()
                if not follow_up_text:
                    self._trace("wake_session_followup_stopped", reason="empty_followup")
                    break

                pending_text = follow_up_text
                remaining_follow_ups -= 1
        except Exception as exc:
            trace_exception("backend.pipeline", exc, event="wake_session_failed")
            log_performance("wake_session_error", 0.0, str(exc))
        finally:
            with self._wake_session_lock:
                self._wake_session_active = False
            self._resume_wake_listener_after_voice_session()
            self._trace("wake_session_finished")

    def _handle_wake_transcript(self, text: str) -> None:
        transcript = str(text or "").strip()
        if not transcript:
            self._trace("wake_transcript_ignored", reason="empty")
            return

        self._trace("wake_transcript_received", transcript=transcript)

        with self._wake_session_lock:
            if self._wake_session_active:
                self._trace("wake_transcript_ignored", reason="session_active")
                return
            self._wake_session_active = True

        paused = False
        if bool(self._wakeword_config.get("auto_restart_after_response", True)):
            try:
                if self.wake_word is not None and hasattr(self.wake_word, "set_detection_paused"):
                    self.wake_word.set_detection_paused(True)
                    paused = True
                    self._trace("wake_detection_paused")
            except Exception:
                paused = False

        if not paused:
            try:
                self.stop_wake_word()
            except Exception:
                pass

        self._trace("wake_session_worker_spawned")
        threading.Thread(target=self._wake_voice_session_worker, args=(transcript,), daemon=True).start()

    def _handle_wake_acknowledged(self, message: str) -> None:
        ack_text = str(message or "Listening.").strip() or "Listening."
        self._trace("wake_acknowledged", message=ack_text)
        self.new_message.emit("assistant", ack_text)
        if self.tts is not None and self.tts_enabled:
            self.tts.speak_async(ack_text)

    def is_wakeword_available(self) -> bool:
        if self._wakeword_initializing:
            return False
        if self.wake_word is None:
            return False
        if hasattr(self.wake_word, "is_available"):
            try:
                return bool(self.wake_word.is_available())
            except Exception:
                return False
        return bool(getattr(self.wake_word, "model", None) is not None)

    def cancel_current_action(self) -> Dict[str, Any]:
        self._trace("cancel_current_action_requested")
        self._cancel_requested = True
        if self.tts:
            try:
                self.tts.stop()
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="cancel_current_action_tts_stop_failed")

        llm_cancel_attempted = False
        llm_cancelled = False
        seen_llm_ids: set[int] = set()
        llm_targets: list[Any] = [self.llm]
        if self.router is not None:
            llm_targets.append(getattr(self.router, "llm", None))

        for llm_target in llm_targets:
            if llm_target is None:
                continue
            target_id = id(llm_target)
            if target_id in seen_llm_ids:
                continue
            seen_llm_ids.add(target_id)
            cancel_fn = getattr(llm_target, "cancel_current_generation", None)
            if not callable(cancel_fn):
                continue

            llm_cancel_attempted = True
            try:
                if bool(cancel_fn()):
                    llm_cancelled = True
            except Exception as exc:
                trace_exception("backend.pipeline", exc, event="cancel_current_action_llm_cancel_failed")

        self._trace(
            "cancel_current_action_applied",
            llm_cancel_attempted=bool(llm_cancel_attempted),
            llm_cancelled=bool(llm_cancelled),
        )
        self._set_state("IDLE")
        return {
            "success": True,
            "response_text": "Cancelled the current action.",
            "data": {
                "cancelled": True,
                "llm_cancel_attempted": bool(llm_cancel_attempted),
                "llm_cancelled": bool(llm_cancelled),
            },
        }

    def _build_intent_diagnostics(self, intent_result: Dict[str, Any]) -> Dict[str, Any]:
        scores = intent_result.get("all_scores") if isinstance(intent_result, dict) else {}
        if not isinstance(scores, dict):
            scores = {}

        ranked = sorted(
            ((str(name), float(value)) for name, value in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        top_candidates = [{"intent": name, "confidence": value} for name, value in ranked[:3]]

        supported_intents: list[str] = []
        if self.intent_classifier is not None and hasattr(self.intent_classifier, "label_encoder"):
            try:
                classes = getattr(self.intent_classifier.label_encoder, "classes_", [])
                supported_intents = [str(item) for item in classes]
            except Exception:
                supported_intents = []

        payload: Dict[str, Any] = {
            "intent": str(intent_result.get("intent", "general_qa")),
            "confidence": float(intent_result.get("confidence", 0.0)),
            "top_candidates": top_candidates,
            "runtime": str(intent_result.get("runtime", "")),
            "provider": str(intent_result.get("provider", "")),
            "latency_ms": float(intent_result.get("latency_ms", 0.0)),
            "supported_intents": supported_intents,
        }
        return payload

    def _emit_direct_intent_diagnostics(self, intent: str, confidence: float, provider: str) -> None:
        intent_result: Dict[str, Any] = {
            "intent": str(intent or "general_qa"),
            "confidence": float(confidence),
            "all_scores": {str(intent or "general_qa"): float(confidence)},
            "runtime": "direct_route",
            "provider": str(provider or ""),
            "latency_ms": 0.0,
        }
        self.intent_classified.emit(intent_result["intent"], float(intent_result["confidence"]))
        self.intent_diagnostics.emit(self._build_intent_diagnostics(intent_result))

    def process_text(self, text: str, *, capture_tts_future: bool = False) -> Dict[str, Any]:
        self._require_initialized()

        self._trace("process_text_started", text_chars=len(str(text or "")), capture_tts_future=bool(capture_tts_future))

        self._cancel_requested = False
        self._set_state("PROCESSING")
        tts_future: Future[Any] | None = None
        request_start = time.perf_counter()

        cleaned_raw = preprocessor.clean(text)
        cleaned = normalize_command_text(cleaned_raw) or cleaned_raw
        compute_hint = choose_device_for_query(cleaned, self._compute_mode)
        self._trace(
            "process_text_preprocessed",
            cleaned_chars=len(cleaned),
            compute_hint=compute_hint,
            normalized=bool(cleaned != cleaned_raw),
        )
        if self.intent_classifier is not None and hasattr(self.intent_classifier, "set_request_device_hint"):
            try:
                self.intent_classifier.set_request_device_hint(compute_hint)
            except Exception:
                pass

        intent_start = time.perf_counter()
        if self.intent_classifier is not None:
            intent_result = self.intent_classifier.predict(cleaned)
        else:
            intent_result = {
                "intent": "general_qa",
                "confidence": 0.0,
                "all_scores": {"general_qa": 1.0},
                "latency_ms": 0.0,
            }
        intent_ms = (time.perf_counter() - intent_start) * 1000
        self._trace(
            "process_text_intent_done",
            intent=str(intent_result.get("intent", "")),
            confidence=float(intent_result.get("confidence", 0.0)),
            intent_ms=round(intent_ms, 2),
        )

        entity_start = time.perf_counter()
        if self.entity_extractor is not None:
            entities = self.entity_extractor.extract_entities(cleaned, intent_result["intent"])
        else:
            entities = {}
        entity_ms = (time.perf_counter() - entity_start) * 1000
        self._trace("process_text_entities_done", entity_count=len(entities), entity_ms=round(entity_ms, 2))

        self.intent_classified.emit(intent_result["intent"], float(intent_result["confidence"]))
        self.intent_diagnostics.emit(self._build_intent_diagnostics(intent_result))
        self.new_message.emit("user", text)

        if self.context is not None:
            self.context.add_turn("user", text, embed=False)

        self._persist_turn_async(
            role="user",
            text=text,
            intent=str(intent_result["intent"]),
            confidence=float(intent_result["confidence"]),
            embed_context_recent=True,
        )

        route_start = time.perf_counter()
        if self.router is not None:
            context_window = self.context.get_window() if self.context is not None else []
            action_result = self.router.route(
                intent_result,
                entities,
                text,
                context_window,
                compute_hint=compute_hint,
            )
        else:
            action_result = {
                "success": False,
                "response_text": "Routing module is unavailable.",
                "data": {},
            }
        route_ms = (time.perf_counter() - route_start) * 1000
        self._trace(
            "process_text_route_done",
            route_ms=round(route_ms, 2),
            action_success=bool(action_result.get("success", False)),
        )

        if self._cancel_requested:
            self._trace("process_text_cancelled")
            self._set_state("IDLE")
            cancelled_result = {
                "text": text,
                "cleaned": cleaned,
                "intent": intent_result,
                "entities": entities,
                "action": {
                    "success": True,
                    "response_text": "Response stopped.",
                    "data": {"cancelled": True},
                },
            }
            if capture_tts_future:
                cancelled_result["_tts_future"] = tts_future
            return cancelled_result

        response_text = str(action_result.get("response_text", "I could not process that request."))
        raw_response_chars = len(response_text)
        response_text = enforce_response_verbosity(response_text, self._response_verbosity)
        self._trace(
            "process_text_response_prepared",
            raw_response_chars=raw_response_chars,
            clipped_response_chars=len(response_text),
            verbosity=self._response_verbosity,
        )

        if self.context is not None:
            self.context.add_turn("assistant", response_text, embed=False)

        self._persist_turn_async(
            role="assistant",
            text=response_text,
            intent=str(intent_result["intent"]),
            confidence=float(intent_result["confidence"]),
            embed_context_recent=True,
        )

        self.new_message.emit("assistant", response_text)

        if not self._cancel_requested and self.tts is not None and self.tts_enabled:
            self._set_state("SPEAKING")
            tts_future = self.tts.speak_async(response_text)
            self._trace("process_text_tts_started", response_chars=len(response_text))

        self._set_state("IDLE")
        total_ms = (time.perf_counter() - request_start) * 1000
        self._trace(
            "process_text_completed",
            total_ms=round(total_ms, 2),
            intent_ms=round(intent_ms, 2),
            entity_ms=round(entity_ms, 2),
            route_ms=round(route_ms, 2),
            intent=str(intent_result.get("intent", "")),
            compute_mode=self._compute_mode,
            device_hint=compute_hint,
        )
        log_performance(
            "pipeline_stage_timing",
            total_ms,
            (
                f"intent_ms={intent_ms:.2f};entity_ms={entity_ms:.2f};route_ms={route_ms:.2f};"
                f"compute_mode={self._compute_mode};device_hint={compute_hint};intent={intent_result.get('intent', '')}"
            ),
        )
        log_performance(
            "pipeline_process_text",
            float(intent_result.get("latency_ms", 0.0)),
            f"compute_mode={self._compute_mode};device_hint={compute_hint}",
        )

        result_payload = {
            "text": text,
            "cleaned": cleaned,
            "intent": intent_result,
            "entities": entities,
            "action": action_result,
        }
        if capture_tts_future:
            result_payload["_tts_future"] = tts_future
        return result_payload

    def process_voice(self) -> Dict[str, Any]:
        self._require_initialized()
        self._trace("process_voice_started")
        if self.stt is None:
            response_text = "Speech input is unavailable in compatibility mode."
            self._trace("process_voice_failed", reason="stt_unavailable")
            self.new_message.emit("assistant", response_text)
            return {
                "success": False,
                "response_text": response_text,
                "transcript": {"text": ""},
            }

        self._set_state("LISTENING")
        transcript = self.stt.listen_once(timeout=10)
        text = str(transcript.get("text", "")).strip()
        self._trace("process_voice_transcript", text_chars=len(text))
        if not text:
            self._set_state("IDLE")
            stt_error = str(transcript.get("error", "")).strip()
            if stt_error:
                response_text = f"I could not use the microphone: {stt_error}"
            else:
                response_text = "I did not catch that. Please try again."
            self.new_message.emit("assistant", response_text)
            self._trace("process_voice_failed", reason="empty_transcript", error=stt_error)
            return {
                "success": False,
                "response_text": response_text,
                "transcript": transcript,
            }

        result = self.process_text(text)
        result["transcript"] = transcript
        self._trace("process_voice_completed", action_success=bool(result.get("action", {}).get("success", False)))
        return result

    def process_recorded_audio(self, audio_array: Any, sample_rate: int = 16000) -> Dict[str, Any]:
        self._require_initialized()
        audio_size = int(getattr(audio_array, "size", 0) or 0)
        self._trace("process_recorded_audio_started", sample_rate=int(sample_rate), audio_size=audio_size)
        if self.stt is None:
            response_text = "Speech input is unavailable in compatibility mode."
            self._trace("process_recorded_audio_failed", reason="stt_unavailable")
            self.new_message.emit("assistant", response_text)
            return {
                "success": False,
                "response_text": response_text,
                "transcript": {"text": ""},
            }

        self._set_state("PROCESSING")
        transcript = self.stt.transcribe(audio_array, sample_rate=sample_rate)
        text = str(transcript.get("text", "")).strip()
        self._trace("process_recorded_audio_transcript", text_chars=len(text))
        if not text:
            self._set_state("IDLE")
            stt_error = str(transcript.get("error", "")).strip()
            if stt_error:
                response_text = f"I could not use the microphone: {stt_error}"
            else:
                response_text = "I did not catch that. Please try again."
            self.new_message.emit("assistant", response_text)
            self._trace("process_recorded_audio_failed", reason="empty_transcript", error=stt_error)
            return {
                "success": False,
                "response_text": response_text,
                "transcript": transcript,
            }

        result = self.process_text(text)
        result["transcript"] = transcript
        self._trace("process_recorded_audio_completed", action_success=bool(result.get("action", {}).get("success", False)))
        return result

    def analyze_image_file(self, file_path: str, prompt: str = "Describe this image in detail.") -> Dict[str, Any]:
        self._require_initialized()
        self._trace("analyze_image_file_started", file_path=file_path, prompt_chars=len(str(prompt or "")))
        if self.router is None:
            self._trace("analyze_image_file_failed", reason="router_unavailable")
            return {"success": False, "response_text": "Routing module is unavailable.", "data": {}}

        self._cancel_requested = False
        self._set_state("PROCESSING")
        self._emit_direct_intent_diagnostics("vision_query", 1.0, "vision_image")
        result = self.router.analyze_image_file(file_path, prompt=prompt)
        if self._cancel_requested:
            self._trace("analyze_image_file_cancelled")
            self._set_state("IDLE")
            return {
                "success": True,
                "response_text": "Response stopped.",
                "data": {"cancelled": True},
            }
        response_text = str(result.get("response_text", "I could not analyze this image."))
        self._trace(
            "analyze_image_file_completed",
            success=bool(result.get("success", False)),
            response_chars=len(response_text),
            mode=str((result.get("data") or {}).get("mode", "")),
        )
        self.new_message.emit("assistant", response_text)
        self._set_state("IDLE")
        return result

    def analyze_camera(self) -> Dict[str, Any]:
        self._require_initialized()
        self._trace("analyze_camera_started")
        if self.router is None:
            self._trace("analyze_camera_failed", reason="router_unavailable")
            return {"success": False, "response_text": "Routing module is unavailable.", "data": {}}

        self._cancel_requested = False
        self._set_state("PROCESSING")
        self._emit_direct_intent_diagnostics("vision_query", 1.0, "vision_camera")
        result = self.router.analyze_camera_capture()
        if self._cancel_requested:
            self._trace("analyze_camera_cancelled")
            self._set_state("IDLE")
            return {
                "success": True,
                "response_text": "Response stopped.",
                "data": {"cancelled": True},
            }
        response_text = str(result.get("response_text", "I could not analyze the camera frame."))
        self._trace(
            "analyze_camera_completed",
            success=bool(result.get("success", False)),
            response_chars=len(response_text),
            mode=str((result.get("data") or {}).get("mode", "")),
        )
        self.new_message.emit("assistant", response_text)
        self._set_state("IDLE")
        return result



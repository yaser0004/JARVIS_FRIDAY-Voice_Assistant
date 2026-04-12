from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

from core.compute_runtime import choose_device_for_query, normalize_compute_mode, windows_cuda_runtime_ready
from core.config import (
    DISTILBERT_ONNX_DIR,
    INTENT_CONFIDENCE_THRESHOLD,
    LABEL_ENCODER_PATH,
    LINEARSVC_PATH,
)
from core.config import log_performance


DEFAULT_INTENTS = [
    "launch_app",
    "close_app",
    "web_search",
    "open_website",
    "play_media",
    "system_volume",
    "system_brightness",
    "power_control",
    "system_settings",
    "general_qa",
    "vision_query",
    "file_control",
    "clipboard_action",
    "stop_cancel",
]

MODEL_ALIASES = {
    "linearsvc": "LinearSVC",
    "linear_svc": "LinearSVC",
    "linear svc": "LinearSVC",
    "bilstm": "BiLSTM",
    "bi_lstm": "BiLSTM",
    "bi lstm": "BiLSTM",
    "distilbert": "DistilBERT",
    "distil_bert": "DistilBERT",
    "distil bert": "DistilBERT",
}

SMALL_TALK_EXACT = {
    "hi",
    "hello",
    "hey",
    "hey there",
    "yo",
    "how are you",
    "how are you doing",
    "how r you",
    "good morning",
    "good afternoon",
    "good evening",
    "what's up",
    "whats up",
}

SMALL_TALK_PREFIXES = (
    "hi ",
    "hello ",
    "hey ",
    "how are you",
    "how r you",
)


class IntentClassifier:
    def __init__(
        self,
        model_dir: Path = DISTILBERT_ONNX_DIR,
        model_name: str = "DistilBERT",
        compute_mode: str = "auto",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.label_encoder = self._load_label_encoder()
        self.tokenizer = None
        self.session = None
        self.input_names: List[str] = []
        self._fallback_model = None
        self._runtime = "linearsvc"
        self._compute_mode = normalize_compute_mode(compute_mode)
        self._request_device_hint: str | None = None
        self._onnx_sessions: Dict[str, Any] = {}
        self._available_onnx_providers: List[str] = []
        self._active_provider = "CPUExecutionProvider"
        self._last_inference_used_onnx = False
        self.model_name = "LinearSVC"

        self.set_model(model_name)

    def set_compute_mode(self, mode: str) -> None:
        self._compute_mode = normalize_compute_mode(mode)

    def set_request_device_hint(self, hint: str | None) -> None:
        normalized = str(hint or "").strip().lower()
        if normalized in {"cpu", "gpu"}:
            self._request_device_hint = normalized
        else:
            self._request_device_hint = None

    def get_runtime_info(self) -> Dict[str, Any]:
        return {
            "runtime": self._runtime,
            "provider": self._active_provider,
            "compute_mode": self._compute_mode,
            "available_providers": list(self._available_onnx_providers),
        }

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        raw = (model_name or "").strip().lower()
        return MODEL_ALIASES.get(raw, "DistilBERT")

    def set_model(self, model_name: str) -> None:
        selected = self._normalize_model_name(model_name)
        self.model_name = selected
        self.session = None
        self.tokenizer = None
        self.input_names = []
        self._fallback_model = None
        self._onnx_sessions = {}
        self._available_onnx_providers = []
        self._active_provider = "CPUExecutionProvider"
        self._last_inference_used_onnx = False

        if selected == "DistilBERT":
            self._init_distilbert_runtime()
            return

        if selected == "BiLSTM":
            # BiLSTM inference runtime is not wired in this lightweight path yet.
            self._runtime = "bilstm_fallback"
            self._init_fallback_runtime()
            log_performance("intent_runtime_fallback", 0.0, "reason=bilstm_runtime_not_available")
            return

        self._runtime = "linearsvc"
        self._init_fallback_runtime()

    def _init_distilbert_runtime(self) -> None:
        onnx_env = os.getenv("JARVIS_ENABLE_ONNX_INTENT", "").strip().lower()
        if onnx_env in {"0", "false", "no"}:
            enable_onnx = False
        elif onnx_env in {"1", "true", "yes"}:
            enable_onnx = True
        else:
            enable_onnx = True

        if enable_onnx and os.getenv("JARVIS_DISABLE_ONNX_INTENT", "0") != "1":
            self._init_onnx_runtime()
        else:
            log_performance("intent_runtime_fallback", 0.0, "reason=onnx_disabled")

        if self.session is None:
            self._runtime = "distilbert_fallback"
            self._init_fallback_runtime()

    def _load_label_encoder(self) -> LabelEncoder:
        if LABEL_ENCODER_PATH.exists():
            try:
                return joblib.load(LABEL_ENCODER_PATH)
            except Exception:
                pass
        encoder = LabelEncoder()
        encoder.fit(DEFAULT_INTENTS)
        return encoder

    def _init_onnx_runtime(self) -> None:
        try:
            import onnxruntime as ort
            from transformers import DistilBertTokenizer

            model_path = self._resolve_model_path(self.model_dir)
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True

            available = set(ort.get_available_providers())
            self._available_onnx_providers = sorted(available)

            cpu_session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self._onnx_sessions = {"cpu": cpu_session}

            can_try_cuda = "CUDAExecutionProvider" in available
            if can_try_cuda and os.name == "nt":
                cuda_ready, cuda_reason = windows_cuda_runtime_ready()
                if not cuda_ready:
                    can_try_cuda = False
                    log_performance("intent_runtime_fallback", 0.0, f"reason=gpu_cuda_preflight_failed:{cuda_reason}")

            if can_try_cuda:
                try:
                    gpu_session = ort.InferenceSession(
                        str(model_path),
                        sess_options=sess_options,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    gpu_providers = [str(item) for item in gpu_session.get_providers()]
                    if "CUDAExecutionProvider" in gpu_providers:
                        self._onnx_sessions["gpu"] = gpu_session
                    else:
                        log_performance("intent_runtime_fallback", 0.0, "reason=gpu_provider_not_active")
                except Exception as exc:
                    log_performance("intent_runtime_fallback", 0.0, f"reason=gpu_provider_init_failed:{exc}")

            self._set_active_session(self._select_onnx_session_key(["startup"]))
            if self.session is None:
                self.session = cpu_session
            self.input_names = [node.name for node in self.session.get_inputs()]
            allow_download = os.getenv("JARVIS_ALLOW_ONLINE_MODEL_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes"}
            if allow_download:
                self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            else:
                self.tokenizer = DistilBertTokenizer.from_pretrained(
                    "distilbert-base-uncased",
                    local_files_only=True,
                )
            self._runtime = "distilbert_onnx_gpu" if self._active_provider == "CUDAExecutionProvider" else "distilbert_onnx_cpu"
        except Exception as exc:
            self.session = None
            self.tokenizer = None
            self.input_names = []
            self._onnx_sessions = {}
            self._available_onnx_providers = []
            self._runtime = "distilbert_fallback"
            log_performance("intent_runtime_fallback", 0.0, f"reason={exc}")

    def _select_onnx_session_key(self, texts: List[str]) -> str | None:
        if not self._onnx_sessions:
            return None

        if self._compute_mode == "cpu":
            return "cpu"

        if self._compute_mode == "gpu":
            return "gpu" if "gpu" in self._onnx_sessions else "cpu"

        if self._request_device_hint == "gpu" and "gpu" in self._onnx_sessions:
            return "gpu"
        if self._request_device_hint == "cpu":
            return "cpu"

        use_gpu = any(choose_device_for_query(text, "auto") == "gpu" for text in texts)
        if use_gpu and "gpu" in self._onnx_sessions:
            return "gpu"
        return "cpu"

    def _set_active_session(self, key: str | None) -> None:
        if key is None:
            self.session = None
            self._active_provider = "CPUExecutionProvider"
            self._runtime = "distilbert_fallback"
            return

        if key == "gpu" and "gpu" in self._onnx_sessions:
            self.session = self._onnx_sessions["gpu"]
        else:
            self.session = self._onnx_sessions.get("cpu")

        providers: List[str] = []
        if self.session is not None:
            try:
                providers = [str(item) for item in self.session.get_providers()]
            except Exception:
                providers = []

        self._active_provider = providers[0] if providers else "CPUExecutionProvider"
        self._runtime = "distilbert_onnx_gpu" if self._active_provider == "CUDAExecutionProvider" else "distilbert_onnx_cpu"

    def _init_fallback_runtime(self) -> None:
        if LINEARSVC_PATH.exists():
            try:
                self._fallback_model = joblib.load(LINEARSVC_PATH)
            except Exception:
                self._fallback_model = None

    @staticmethod
    def _resolve_model_path(model_dir: Path) -> Path:
        direct = model_dir / "model.onnx"
        if direct.exists():
            return direct
        onnx_files = sorted(model_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError(f"No ONNX model found in {model_dir}")
        return onnx_files[0]

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def _resolve_class_index(self, value: object) -> int | None:
        try:
            idx = int(value)
            if 0 <= idx < len(self.label_encoder.classes_):
                return idx
        except Exception:
            pass

        if isinstance(value, str):
            matches = np.where(self.label_encoder.classes_ == value)[0]
            if len(matches) > 0:
                return int(matches[0])
        return None

    def _fallback_logits(self, texts: List[str]) -> np.ndarray:
        num_classes = len(self.label_encoder.classes_)
        logits = np.full((len(texts), num_classes), -8.0, dtype=np.float32)

        if self._fallback_model is None:
            try:
                idx = int(np.where(self.label_encoder.classes_ == "general_qa")[0][0])
            except Exception:
                idx = 0
            logits[:, idx] = 8.0
            return logits

        try:
            if hasattr(self._fallback_model, "decision_function"):
                scores = np.asarray(self._fallback_model.decision_function(texts), dtype=np.float32)
                if scores.ndim == 1:
                    scores = np.expand_dims(scores, axis=1)

                classes = np.asarray(getattr(self._fallback_model, "classes_", np.arange(scores.shape[1])))
                for col, cls in enumerate(classes):
                    cls_idx = self._resolve_class_index(cls)
                    if cls_idx is not None and col < scores.shape[1]:
                        logits[:, cls_idx] = scores[:, col]
                return logits

            preds = np.asarray(self._fallback_model.predict(texts))
            for i, pred in enumerate(preds):
                cls_idx = self._resolve_class_index(pred)
                if cls_idx is not None:
                    logits[i, cls_idx] = 8.0
            return logits
        except Exception:
            return logits

    def _run(self, texts: List[str]) -> np.ndarray:
        if not self._onnx_sessions or self.tokenizer is None:
            self._last_inference_used_onnx = False
            return self._fallback_logits(texts)

        self._set_active_session(self._select_onnx_session_key(texts))
        if self.session is None:
            self._last_inference_used_onnx = False
            return self._fallback_logits(texts)

        encoded = self.tokenizer(
            texts,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        feed = {}
        for name in self.input_names:
            if name in encoded:
                feed[name] = encoded[name].astype(np.int64)
            elif name == "attention_mask":
                feed[name] = encoded["attention_mask"].astype(np.int64)
            elif name == "input_ids":
                feed[name] = encoded["input_ids"].astype(np.int64)

        try:
            outputs = self.session.run(None, feed)
            self._last_inference_used_onnx = True
            return np.asarray(outputs[0])
        except Exception as exc:
            self._last_inference_used_onnx = False
            log_performance("intent_runtime_fallback", 0.0, f"reason=onnx_inference_failed:{exc}")
            return self._fallback_logits(texts)
        finally:
            self._request_device_hint = None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _small_talk_override(self, text: str) -> str | None:
        normalized = self._normalize_text(text)
        if not normalized:
            return None

        if normalized in SMALL_TALK_EXACT:
            return "general_qa"

        if any(normalized.startswith(prefix) for prefix in SMALL_TALK_PREFIXES):
            return "general_qa"

        if len(normalized.split()) <= 2 and normalized in {"thanks", "thank you"}:
            return "general_qa"

        return None

    def _build_override_result(self, intent: str, latency_ms: float) -> Dict[str, object]:
        labels = [str(label) for label in self.label_encoder.classes_]
        score_map = {label: 0.0 for label in labels}
        chosen = intent if intent in score_map else "general_qa"
        if chosen not in score_map and labels:
            chosen = labels[0]
        score_map[chosen] = 1.0

        return {
            "intent": chosen,
            "confidence": 1.0,
            "all_scores": score_map,
            "latency_ms": latency_ms,
            "runtime": self._runtime,
            "provider": self._active_provider,
        }

    def predict(self, text: str) -> Dict[str, object]:
        start = time.perf_counter()
        override = self._small_talk_override(text)
        if override is not None:
            latency_ms = (time.perf_counter() - start) * 1000.0
            result = self._build_override_result(override, latency_ms)
            log_performance("intent_classification", latency_ms, f"intent={override};runtime=rules")
            return result

        logits = self._run([text])
        probs = self._softmax(logits)[0]

        best_idx = int(np.argmax(probs))
        if not self._last_inference_used_onnx and self._fallback_model is not None:
            try:
                pred = self._fallback_model.predict([text])[0]
                resolved = self._resolve_class_index(pred)
                if resolved is not None:
                    best_idx = resolved
            except Exception:
                pass

        confidence = float(probs[best_idx])
        predicted = str(self.label_encoder.inverse_transform([best_idx])[0])

        if self._last_inference_used_onnx and confidence < INTENT_CONFIDENCE_THRESHOLD:
            predicted = "general_qa"

        labels = self.label_encoder.inverse_transform(np.arange(len(probs)))
        score_map = {str(label): float(prob) for label, prob in zip(labels, probs)}
        latency_ms = (time.perf_counter() - start) * 1000.0
        log_performance(
            "intent_classification",
            latency_ms,
            f"intent={predicted};runtime={self._runtime};provider={self._active_provider}",
        )

        return {
            "intent": predicted,
            "confidence": confidence,
            "all_scores": score_map,
            "latency_ms": latency_ms,
            "runtime": self._runtime,
            "provider": self._active_provider,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict[str, object]]:
        if not texts:
            return []

        start = time.perf_counter()
        logits = self._run(texts)
        probs_batch = self._softmax(logits)
        labels = self.label_encoder.inverse_transform(np.arange(probs_batch.shape[1]))
        overrides = [self._small_talk_override(text) for text in texts]

        fallback_preds: List[int | None] = []
        if not self._last_inference_used_onnx and self._fallback_model is not None:
            try:
                raw_preds = self._fallback_model.predict(texts)
                fallback_preds = [self._resolve_class_index(p) for p in raw_preds]
            except Exception:
                fallback_preds = [None] * len(texts)
        else:
            fallback_preds = [None] * len(texts)

        results: List[Dict[str, object]] = []
        for i, probs in enumerate(probs_batch):
            if i < len(overrides) and overrides[i] is not None:
                results.append(self._build_override_result(str(overrides[i]), 0.0))
                continue

            best_idx = int(np.argmax(probs))
            if i < len(fallback_preds) and fallback_preds[i] is not None:
                best_idx = int(fallback_preds[i])
            confidence = float(probs[best_idx])
            predicted = str(self.label_encoder.inverse_transform([best_idx])[0])
            if self._last_inference_used_onnx and confidence < INTENT_CONFIDENCE_THRESHOLD:
                predicted = "general_qa"
            results.append(
                {
                    "intent": predicted,
                    "confidence": confidence,
                    "all_scores": {
                        str(label): float(score) for label, score in zip(labels, probs)
                    },
                    "latency_ms": 0.0,
                    "runtime": self._runtime,
                    "provider": self._active_provider,
                }
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_item = elapsed_ms / max(len(results), 1)
        for item in results:
            item["latency_ms"] = per_item
        log_performance(
            "intent_classification_batch",
            elapsed_ms,
            f"batch={len(texts)};runtime={self._runtime};provider={self._active_provider}",
        )
        return results


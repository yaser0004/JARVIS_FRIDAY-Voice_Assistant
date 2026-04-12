from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "JARVIS"
APPDATA_DIR = Path(os.getenv("APPDATA", str(Path.home()))) / APP_NAME
APPDATA_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = APPDATA_DIR / "performance.log"
DB_FILE = APPDATA_DIR / "jarvis_memory.db"

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
ASSETS_DIR = ROOT_DIR / "assets"
MODELS_DIR = ROOT_DIR / "models"
ML_MODELS_DIR = ROOT_DIR / "ml" / "models"
ML_RESULTS_DIR = ROOT_DIR / "ml" / "results"

DISTILBERT_ONNX_DIR = ML_MODELS_DIR / "distilbert_onnx"
BILSTM_ONNX_PATH = ML_MODELS_DIR / "bilstm.onnx"
LINEARSVC_PATH = ML_MODELS_DIR / "linearsvc.pkl"
LOGREG_PATH = ML_MODELS_DIR / "logreg.pkl"
TOKENIZER_PATH = ML_MODELS_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = ML_MODELS_DIR / "label_encoder.pkl"

GLOVE_PATH = DATA_DIR / "glove" / "glove.6B.100d.txt"
QWEN_VL_GGUF_PATH = MODELS_DIR / "Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
QWEN_VL_MMPROJ_PATH = MODELS_DIR / "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
QWEN_TEXT_FALLBACK_GGUF_PATH = MODELS_DIR / "qwen2.5-3b-instruct-q4_k_m.gguf"
QWEN_GGUF_PATH = QWEN_VL_GGUF_PATH

CNN_VISION_DIR = ML_MODELS_DIR / "cnn_vision"
CNN_VISION_WEIGHTS_PATH = CNN_VISION_DIR / "cnn_scratch.pt"
CNN_VISION_LABELS_PATH = CNN_VISION_DIR / "labels.json"

WAKE_CHIME_PATH = ASSETS_DIR / "sounds" / "wake_chime.wav"
ERROR_SOUND_PATH = ASSETS_DIR / "sounds" / "error.wav"
APP_ICON_PATH = ASSETS_DIR / "icon.ico"

INTENT_CONFIDENCE_THRESHOLD = 0.55
CONTEXT_WINDOW_SIZE = 5


@dataclass(frozen=True)
class PerformanceTargets:
    wake_word_ms: int = 500
    stt_5s_ms: int = 800
    intent_ms: int = 8
    entity_ms: int = 5
    app_launch_ms: int = 300
    system_control_ms: int = 50
    llm_first_token_ms: int = 1000
    tts_start_ms: int = 200


def ensure_runtime_dirs() -> None:
    APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    ML_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ML_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CNN_VISION_DIR.mkdir(parents=True, exist_ok=True)


def log_performance(operation: str, latency_ms: float, details: str = "") -> None:
    ensure_runtime_dirs()
    line = f"{operation}|{latency_ms:.3f}|{details}\n"
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line)

    # Mirror perf metrics into session telemetry when a session trace logger is active.
    try:
        from core.session_logging import trace_event

        trace_event(
            "backend.performance",
            "metric",
            operation=operation,
            latency_ms=round(float(latency_ms), 3),
            details=details,
        )
    except Exception:
        pass


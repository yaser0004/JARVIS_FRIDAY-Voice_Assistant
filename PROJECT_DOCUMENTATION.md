<![CDATA[# JARVIS — Project Documentation

> **Just A Really Very Intelligent System**
>
> A fully offline, multimodal AI desktop assistant for Windows with voice interaction, vision understanding, system automation, and intelligent intent routing — powered entirely by local machine learning models.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [Directory Structure](#3-directory-structure)
4. [Core Pipeline (`core/`)](#4-core-pipeline)
   - 4.1 [JarvisPipeline Orchestrator](#41-jarvispipeline-orchestrator)
   - 4.2 [Configuration & Constants](#42-configuration--constants)
   - 4.3 [Context Manager](#43-context-manager)
   - 4.4 [Compute Runtime](#44-compute-runtime)
   - 4.5 [Runtime Settings](#45-runtime-settings)
   - 4.6 [Session Logging & Telemetry](#46-session-logging--telemetry)
5. [Natural Language Processing (`nlp/`)](#5-natural-language-processing)
   - 5.1 [Intent Classifier](#51-intent-classifier)
   - 5.2 [Entity Extractor](#52-entity-extractor)
   - 5.3 [Router](#53-router)
   - 5.4 [Text Preprocessor](#54-text-preprocessor)
   - 5.5 [Conversation Normalizer](#55-conversation-normalizer)
6. [Large Language Model Integration (`llm/`)](#6-large-language-model-integration)
   - 6.1 [QwenBridge](#61-qwenbridge)
   - 6.2 [Qwen Worker Process](#62-qwen-worker-process)
   - 6.3 [Model Resolution & Auto-Switch](#63-model-resolution--auto-switch)
7. [Speech Processing (`speech/`)](#7-speech-processing)
   - 7.1 [Speech-to-Text (STT)](#71-speech-to-text-stt)
   - 7.2 [Text-to-Speech (TTS)](#72-text-to-speech-tts)
   - 7.3 [Wake Word Detection](#73-wake-word-detection)
   - 7.4 [Wake Word Configuration](#74-wake-word-configuration)
8. [Vision & Multimodal (`vision/`)](#8-vision--multimodal)
   - 8.1 [VisionModel](#81-visionmodel)
   - 8.2 [Screen Capture](#82-screen-capture)
   - 8.3 [Webcam Capture](#83-webcam-capture)
   - 8.4 [CNN Image Classifier](#84-cnn-image-classifier)
   - 8.5 [CNN Architecture (ScratchVisionCNN)](#85-cnn-architecture-scratchvisioncnn)
9. [Action Modules (`actions/`)](#9-action-modules)
   - 9.1 [Application Control](#91-application-control)
   - 9.2 [System Control](#92-system-control)
   - 9.3 [Media Control](#93-media-control)
   - 9.4 [Web Control](#94-web-control)
   - 9.5 [Real-Time Web Intelligence](#95-real-time-web-intelligence)
   - 9.6 [System Information](#96-system-information)
   - 9.7 [Time & World Clock](#97-time--world-clock)
   - 9.8 [File Control](#98-file-control)
   - 9.9 [Clipboard Control](#99-clipboard-control)
10. [Persistent Memory (`memory/`)](#10-persistent-memory)
    - 10.1 [SQLite Store](#101-sqlite-store)
    - 10.2 [Vector Store](#102-vector-store)
11. [User Interface (`ui/`)](#11-user-interface)
    - 11.1 [Main Window](#111-main-window)
    - 11.2 [System Tray](#112-system-tray)
    - 11.3 [Theme & Styling System](#113-theme--styling-system)
    - 11.4 [Animation Utilities](#114-animation-utilities)
    - 11.5 [OrbWidget — Animated Status Orb](#115-orbwidget--animated-status-orb)
    - 11.6 [ChatWidget — Conversation Display](#116-chatwidget--conversation-display)
    - 11.7 [Sidebar — Diagnostics Panel](#117-sidebar--diagnostics-panel)
    - 11.8 [WaveformWidget — Audio Visualizer](#118-waveformwidget--audio-visualizer)
    - 11.9 [MetricsStatusBar — System Metrics](#119-metricstatusbar--system-metrics)
12. [Model Setup & Asset Management](#12-model-setup--asset-management)
13. [Build & Packaging](#13-build--packaging)
14. [Application Entry Point](#14-application-entry-point)
15. [Environment Variables Reference](#15-environment-variables-reference)
16. [Performance Targets & Benchmarks](#16-performance-targets--benchmarks)
17. [Dependency Stack](#17-dependency-stack)
18. [Static Assets](#18-static-assets)
19. [Data Flow Diagrams](#19-data-flow-diagrams)
20. [Security & Privacy Considerations](#20-security--privacy-considerations)
21. [Glossary](#21-glossary)

---

## 1. Executive Summary

**JARVIS** (Just A Really Very Intelligent System) is a production-grade, fully offline AI desktop assistant designed exclusively for the Windows platform. The system provides hands-free voice interaction, multimodal vision understanding, intelligent natural language processing, and deep system automation — all running entirely on local hardware without any cloud dependency.

### Key Differentiators

| Feature | Description |
|---|---|
| **100% Offline** | All ML inference (LLM, STT, TTS, intent classification, vision) runs locally. No data leaves the user's machine. |
| **Hybrid Execution Model** | Deterministic Python scripts handle known commands at millisecond latency; only ambiguous or general queries are routed to the local LLM. |
| **Multimodal Vision** | Qwen2.5-VL GGUF model enables real-time screen analysis, webcam understanding, and arbitrary image Q&A with a custom CNN fallback classifier. |
| **Adaptive Compute** | Automatic CPU/GPU switching based on query complexity scoring, with runtime fallback, cooldown safeguards, and per-module device propagation. |
| **Multi-Voice TTS** | Three-cascading TTS backends (Edge TTS, Kokoro-82M, SAPI) with male/female voice profiles. |
| **Production Packaging** | Nuitka-compiled standalone `.exe` with all assets bundled — zero Python dependency on the end-user machine. |
| **Session Telemetry** | Structured JSONL trace logging with dual-stream stdout/stderr capture, exception hooks, and performance metric instrumentation. |
| **Wake Word System** | Dual-mode wake word detection (neural openWakeWord + phrase fallback) with multi-turn follow-up conversation support. |

### Technology Stack Summary

| Layer | Technology |
|---|---|
| GUI Framework | PyQt6 (QMainWindow + QObject signals) |
| LLM Runtime | llama-cpp-python (GGUF, subprocess-isolated) |
| Speech-to-Text | faster-whisper (CTranslate2 backend) |
| Text-to-Speech | Edge TTS / Kokoro-82M / Windows SAPI |
| Intent Classification | ONNX Runtime (DistilBERT) + LinearSVC fallback |
| Entity Extraction | spaCy + regex + heuristic rules |
| Vision | Qwen2.5-VL via llama-cpp + custom 4-layer CNN |
| NLP | spaCy, GloVe embeddings, regex patterns |
| Memory | SQLite + ChromaDB (vector) + sentence-transformers |
| Build | Nuitka (standalone C compilation) |
| System Metrics | psutil + GPUtil |
| Audio I/O | sounddevice + soundfile + pygame |

### Project Metrics

| Metric | Value |
|---|---|
| Total source files | ~40+ Python modules |
| Core pipeline (`pipeline.py`) | 1,357 lines |
| Main window (`main_window.py`) | 1,663 lines |
| LLM bridge (`qwen_bridge.py`) | 1,089 lines |
| Router (`router.py`) | 817 lines |
| Model setup utility | 791 lines |
| Intent classifier | 533 lines |
| Total project LOC | ~12,000+ |

---

## 2. System Architecture Overview

JARVIS follows a **modular pipeline architecture** where each subsystem operates as an independent module orchestrated by a central `JarvisPipeline` class. The architecture enforces strict separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        JARVIS MAIN PROCESS                         │
│  ┌───────────┐  ┌────────────┐  ┌──────────┐  ┌────────────────┐  │
│  │  PyQt6 UI │  │  Pipeline  │  │  Router  │  │ Action Modules │  │
│  │  (Thread) │→ │  (Core)    │→ │  (NLP)   │→ │  (Deterministic│  │
│  │           │  │            │  │          │  │   Scripts)     │  │
│  └───────────┘  └─────┬──────┘  └────┬─────┘  └────────────────┘  │
│                       │              │                             │
│                       │              ▼                             │
│                       │       ┌──────────────┐                    │
│                       │       │ Intent Class. │                    │
│                       │       │ (ONNX/SVC)    │                    │
│                       │       └──────────────┘                    │
│                       ▼                                            │
│              ┌────────────────┐    ┌───────────────┐              │
│              │  QwenBridge    │    │ Qwen Worker   │              │
│              │  (LLM Control) │───→│ (Subprocess)  │              │
│              └───────┬────────┘    └───────────────┘              │
│                      │                                             │
│              ┌───────┴────────────────────────┐                   │
│              │     Speech Processing          │                   │
│              │  ┌─────┐ ┌─────┐ ┌──────────┐ │                   │
│              │  │ STT │ │ TTS │ │ WakeWord │ │                   │
│              │  └─────┘ └─────┘ └──────────┘ │                   │
│              └────────────────────────────────┘                   │
│              ┌────────────────────────────────┐                   │
│              │     Vision & Memory            │                   │
│              │  ┌────────┐ ┌────────────────┐ │                   │
│              │  │ Vision │ │ SQLite/ChromaDB│ │                   │
│              │  └────────┘ └────────────────┘ │                   │
│              └────────────────────────────────┘                   │
└─────────────────────────────────────────────────────────────────────┘
```

### Execution Flow

```
User Input (Voice/Text)
       │
       ▼
┌──────────────────┐
│ Preprocessor     │  ← Lowercase, strip punctuation, normalize "%"
└──────┬───────────┘
       ▼
┌────────────────────────┐
│ Conversation Normalizer│  ← Strip fillers, polite phrases, name prefix
└──────┬─────────────────┘
       ▼
┌──────────────────┐
│ Intent Classifier │  ← ONNX DistilBERT → softmax → or LinearSVC fallback
└──────┬───────────┘
       ▼
┌──────────────────┐
│ Entity Extractor  │  ← spaCy NER + regex patterns → slot filling
└──────┬───────────┘
       ▼
┌──────────────┐     ┌─────────────────────────┐
│    Router    │────→│ Fast-Path Check          │  (time, sysinfo, media cmds)
│              │     └─────────────────────────┘
│              │     ┌─────────────────────────┐
│              │────→│ Deterministic Action     │  (launch_app, set_volume, ...)
│              │     └─────────────────────────┘
│              │     ┌─────────────────────────┐
│              │────→│ Verified Web Intelligence│  (multi-source synthesis)
│              │     └─────────────────────────┘
│              │     ┌─────────────────────────┐
│              │────→│ LLM (QwenBridge)        │  (generative, ~1-8s)
│              │     └─────────────────────────┘
│              │     ┌─────────────────────────┐
│              │────→│ Small-Talk Fallback      │  (predefined responses)
└──────────────┘     └─────────────────────────┘
       │
       ▼
┌──────────────────┐
│ Verbosity Filter  │  ← Enforce sentence/word/char limits per mode
└──────┬───────────┘
       ▼
┌──────────────┐
│  TTS Output  │  ← Spoken response (Edge TTS / Kokoro / SAPI)
└──────────────┘
```

---

## 3. Directory Structure

```
JARVIS/
├── main.py                    # Application entry point (103 lines)
├── build.py                   # Nuitka build configuration (37 lines)
├── setup_models.py            # Model download & configuration utility (791 lines)
├── requirements.txt           # Python dependency manifest (234 entries)
├── README.md                  # Quick-start guide
├── PROJECT_DOCUMENTATION.md   # This document
│
├── core/                      # Core orchestration and configuration
│   ├── __init__.py
│   ├── pipeline.py            # JarvisPipeline — central orchestrator (1,357 lines)
│   ├── config.py              # Constants, paths, performance targets (86 lines)
│   ├── context_manager.py     # Conversation context with embeddings (125 lines)
│   ├── compute_runtime.py     # CPU/GPU detection & switching (287 lines)
│   ├── runtime_settings.py    # Persistent JSON runtime configuration (60 lines)
│   └── session_logging.py     # Telemetry, tracing, performance logging (286 lines)
│
├── nlp/                       # Natural Language Processing
│   ├── __init__.py
│   ├── intent_classifier.py   # ONNX DistilBERT + LinearSVC intent models (533 lines)
│   ├── entity_extractor.py    # Named entity & parameter extraction (261 lines)
│   ├── router.py              # Intent-to-action routing engine (817 lines)
│   ├── preprocessor.py        # Text normalization (28 lines)
│   └── conversation_normalizer.py  # Filler/polite phrase stripping (76 lines)
│
├── llm/                       # Local LLM integration
│   ├── __init__.py
│   ├── qwen_bridge.py         # Main LLM interface (in-process + worker) (1,089 lines)
│   └── qwen_worker.py         # Isolated subprocess for LLM inference (284 lines)
│
├── speech/                    # Voice I/O
│   ├── __init__.py
│   ├── stt.py                 # Speech-to-Text (faster-whisper) (336 lines)
│   ├── tts.py                 # Text-to-Speech (multi-backend) (377 lines)
│   ├── wake_word.py           # Wake word detection (openWakeWord) (284 lines)
│   └── wakeword_config.py     # Wake word JSON configuration (97 lines)
│
├── vision/                    # Multimodal vision
│   ├── __init__.py
│   ├── vision_model.py        # VisionModel orchestrator (121 lines)
│   ├── screen_capture.py      # Screen capture via pyautogui (32 lines)
│   ├── webcam.py              # Webcam frame capture via OpenCV (32 lines)
│   ├── cnn_classifier.py      # CNN inference wrapper (106 lines)
│   └── cnn_scratch.py         # Custom 4-layer CNN architecture (67 lines)
│
├── actions/                   # Deterministic action scripts
│   ├── __init__.py
│   ├── app_control.py         # Application launch/close/switch (441 lines)
│   ├── system_control.py      # Volume, brightness, power, networking (307 lines)
│   ├── media_control.py       # Music playback (local + streaming) (414 lines)
│   ├── web_control.py         # Browser & website control (180 lines)
│   ├── realtime_web.py        # Web-verified AI overviews (521 lines)
│   ├── system_info.py         # Hardware & connectivity status (137 lines)
│   ├── time_control.py        # Time, date, world clock (448 lines)
│   ├── file_control.py        # File open/read/find (60 lines)
│   └── clipboard_control.py   # Clipboard read/write/paste (46 lines)
│
├── memory/                    # Persistent storage
│   ├── __init__.py
│   ├── sqlite_store.py        # Conversation history & preferences (121 lines)
│   └── vector_store.py        # Embedding-based semantic memory (105 lines)
│
├── ui/                        # User interface
│   ├── __init__.py
│   ├── main_window.py         # JarvisMainWindow — primary GUI (1,663 lines)
│   ├── system_tray.py         # System tray icon & menu (91 lines)
│   ├── theme.py               # Color scheme, fonts, borders (36 lines)
│   ├── animations.py          # Animation utilities (28 lines)
│   └── widgets/               # Reusable UI components
│       ├── __init__.py
│       ├── chat_widget.py     # Conversation thread display (187 lines)
│       ├── orb_widget.py      # Animated status orb (193 lines)
│       ├── sidebar.py         # Navigation & intent diagnostics (210 lines)
│       ├── status_bar.py      # Real-time CPU/GPU/RAM metrics (74 lines)
│       └── waveform_widget.py # Real-time audio waveform visualizer (133 lines)
│
├── ml/                        # ML model assets
│   └── models/
│       ├── distilbert_onnx/   # ONNX intent classification model
│       ├── cnn_vision/        # Custom CNN weights & labels
│       ├── linearsvc.pkl      # LinearSVC fallback classifier
│       ├── logreg.pkl         # Logistic regression model
│       ├── label_encoder.pkl  # Sklearn LabelEncoder
│       └── tokenizer.pkl      # Tokenizer artifact
│
├── models/                    # LLM GGUF model files
│   ├── Qwen2.5-VL-3B-Instruct-Q8_0.gguf
│   ├── mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf
│   └── qwen2.5-3b-instruct-q4_k_m.gguf
│
├── data/                      # Runtime data
│   ├── glove/                 # GloVe word embeddings
│   │   └── glove.6B.100d.txt
│   └── captures/              # Screen & webcam captures
│
└── assets/                    # Static assets
    ├── icon.ico               # Application icon
    ├── style.qss              # Qt stylesheet (696 bytes)
    ├── fonts/
    │   ├── Orbitron-Bold.ttf   # Display headings font
    │   └── ShareTechMono-Regular.ttf  # Monospace font
    └── sounds/
        ├── wake_chime.wav     # Wake word acknowledgment chime
        └── error.wav          # Error notification sound
```

---

## 4. Core Pipeline

### 4.1 JarvisPipeline Orchestrator

**File:** `core/pipeline.py`
**Class:** `JarvisPipeline(QObject)`
**Lines of Code:** 1,357

The `JarvisPipeline` is the central orchestrator of the entire JARVIS system. It extends PyQt6's `QObject` to leverage the Qt signal-slot mechanism for thread-safe GUI updates.

#### Qt Signals

| Signal | Parameters | Description |
|---|---|---|
| `pipeline_state_changed` | `str` | Emitted on state transitions (`IDLE`, `LISTENING`, `PROCESSING`, `SPEAKING`) |
| `new_message` | `str, str` | Emitted when a message is produced (role, text) |
| `intent_classified` | `str, float` | Emitted with classified intent name and confidence score |
| `intent_diagnostics` | `dict` | Emitted with full diagnostics: top candidates, runtime, provider, latency |
| `initialization_progress` | `str, int` | Emitted during startup with step label and percentage |
| `wakeword_availability_changed` | `bool, str` | Emitted when wake word availability changes |
| `ready` | (none) | Emitted when all initialization is complete |

#### Constructor State Initialization

The `__init__` method performs no heavy loading — it only reads configuration files and sets up state:

| State Variable | Type | Default | Source |
|---|---|---|---|
| `current_state` | `str` | `"IDLE"` | Hardcoded |
| `_initialized` | `bool` | `False` | Internal |
| `_cancel_requested` | `bool` | `False` | Internal |
| `_intent_model_name` | `str` | `"DistilBERT"` | `$JARVIS_INTENT_MODEL` |
| `_web_verified_mode` | `bool` | `False` | Internal |
| `_wakeword_config` | `Dict` | (see §7.4) | `wakeword.json` |
| `tts_enabled` | `bool` | `True` | `runtime_settings.json` |
| `_tts_profile` | `str` | `"female"` | `runtime_settings.json` |
| `_response_verbosity` | `str` | `"normal"` | `runtime_settings.json` |
| `_compute_mode` | `str` | `"auto"` | `runtime_settings.json` |
| `_memory_executor` | `ThreadPoolExecutor` | 1 worker | Internal |

#### Initialization Sequence (Exact Order)

The `initialize()` method runs on a background thread via `initialize_async()`. Each step is wrapped in try/except with individual fallback:

```
JarvisPipeline.initialize()
     │
     ├── Step 1: IntentClassifier(model_name, compute_mode)    → 12%
     ├── Step 2: EntityExtractor()                              → 25%
     ├── Step 3: SpeechToText(compute_mode)                     → 37%
     ├── Step 4: TextToSpeech(profile)                          → 50%
     ├── Step 5: QwenBridge(compute_mode)                       → 62%
     ├── Step 6: ContextManager()                               → 75%
     ├── Step 7: SQLiteStore()                                  → 87%
     ├── Step 8: VectorStore()                                  → 100%
     │
     ├── Router(llm, cancel_callback) setup
     ├── Apply compute mode to all modules
     ├── Refresh compute capabilities (async)
     ├── Warm STT engine (async)
     ├── Initialize wake word detector (async)
     ├── Bootstrap app/media indexes (async)
     ├── Prewarm LLM (async)
     │
     └── Emit ready signal
```

If any step fails, it sets that module to `None` and continues with `(fallback mode)` status. The pipeline never crashes on initialization failures.

#### process_text() — Full Request Lifecycle

```python
process_text(text, *, capture_tts_future=False) -> Dict[str, Any]
```

| Phase | Operation | Timing |
|---|---|---|
| 1. Preprocess | `preprocessor.clean(text)` | <1ms |
| 2. Normalize | `normalize_command_text(cleaned)` | <1ms |
| 3. Compute hint | `choose_device_for_query(cleaned, mode)` | <1ms |
| 4. Classify intent | `intent_classifier.predict(cleaned)` → softmax scores | ~10-25ms |
| 5. Extract entities | `entity_extractor.extract_entities(cleaned, intent)` | ~5-15ms |
| 6. Emit diagnostics | `intent_classified` + `intent_diagnostics` signals | <1ms |
| 7. Persist user turn | `_persist_turn_async(role="user", ...)` (background) | async |
| 8. Route | `router.route(intent_result, entities, text, context, compute_hint)` | 10ms-8s |
| 9. Enforce verbosity | `enforce_response_verbosity(response, mode)` | <1ms |
| 10. Persist assistant | `_persist_turn_async(role="assistant", ...)` (background) | async |
| 11. TTS | `tts.speak_async(response)` if enabled | async |
| 12. Log metrics | `log_performance("pipeline_stage_timing", ...)` | <1ms |

#### Response Verbosity Enforcement

The `enforce_response_verbosity()` function applies cascading limits:

| Mode | Max Sentences | Max Words | Max Characters |
|---|---|---|---|
| `brief` | 2 | 60 | 380 |
| `normal` | 4 | 180 | 1,200 |
| `detailed` | 12 | 520 | 4,200 |

Truncation algorithm:
1. Split response into sentences (regex: `(?<=[.!?])\s+`)
2. Take first N sentences per limit
3. Clip to max words, adding period if truncated
4. Clip to max characters, seeking sentence boundary at 60% mark

#### Verbosity Aliases

| Input | Resolves To |
|---|---|
| `short`, `concise`, `terse` | `brief` |
| `default`, `regular` | `normal` |
| `long`, `verbose`, `thorough`, `comprehensive` | `detailed` |

#### Wake Voice Session Worker

When a wake word is detected, a dedicated voice session is spawned:

```
_wake_voice_session_worker(initial_text)
    │
    ├── Loop while pending_text exists:
    │   ├── process_text(pending_text, capture_tts_future=True)
    │   ├── Wait for TTS completion (timeout=120s)
    │   ├── Check cancel_requested
    │   ├── Check follow_up_enabled and remaining_follow_ups
    │   ├── _listen_for_post_response_follow_up()
    │   │   ├── Set state LISTENING
    │   │   ├── stt.listen_once(timeout=follow_up_timeout)
    │   │   ├── Strip activation phrase
    │   │   └── Return filtered text or ""
    │   └── Decrement remaining_follow_ups
    │
    └── Resume wake listener (_resume_wake_listener_after_voice_session)
```

#### Asynchronous Memory Persistence

The `_persist_turn_async` method submits memory writes to a single-threaded `ThreadPoolExecutor`:

```python
_persist_turn_async(role, text, intent, confidence, embed_context_recent)
    → ThreadPoolExecutor.submit(_persist_turn_to_stores)
        ├── SQLiteStore.save_turn(role, text, intent, confidence)
        ├── VectorStore.add_memory(text, {role, intent})
        └── ContextManager.embed_recent_missing(limit=4)
```

If the executor rejects the submission (e.g., shutdown), the persist function runs synchronously as fallback.

#### Cancellation System

```python
cancel_current_action()
    ├── Set _cancel_requested = True
    ├── tts.stop()  — halt speech playback
    ├── For each LLM target (pipeline.llm, router.llm):
    │   └── cancel_current_generation()
    ├── Set state to IDLE
    └── Return {success=True, cancelled=True, llm_cancelled=bool}
```

#### Additional Public Methods

| Method | Description |
|---|---|
| `process_voice()` | Microphone → STT → `process_text()` with transcript metadata |
| `process_recorded_audio(array, sr)` | Pre-recorded audio → STT → `process_text()` |
| `analyze_image_file(path, prompt)` | Image file analysis via `router.analyze_image_file()` |
| `analyze_camera()` | Webcam capture → `router.analyze_camera_capture()` |
| `set_compute_mode(mode)` | Propagate compute mode to all modules |
| `set_tts_enabled(enabled)` | Toggle TTS with immediate stop if disabling |
| `set_tts_profile(profile)` | Switch voice: `"female"` or `"male"` |
| `set_response_verbosity(mode)` | Apply verbosity to pipeline and router |
| `set_realtime_web_enabled(enabled)` | Toggle verified web mode |
| `set_intent_model(model_name)` | Hot-swap intent classifier model |
| `update_wakeword_settings(...)` | Update and persist wake word configuration |
| `get_tts_settings()` | Return `{enabled, profile, profiles}` |
| `get_response_settings()` | Return `{verbosity, modes}` |
| `get_compute_settings()` | Return `{mode, available_modes, capabilities}` |
| `get_wakeword_settings()` | Return full wake word config dict |
| `get_wakeword_status()` | Return `{enabled, available, initializing, reason}` |
| `get_stt_status()` | Return `{available, initialized, reason, device}` |
| `get_llm_status()` | Return `{state, mode, message, supports_vision}` |
| `is_wakeword_available()` | Check if wake word detector is ready |
| `is_realtime_web_enabled()` | Check web mode status |
| `shutdown()` | Graceful cleanup of all subsystems |

---

### 4.2 Configuration & Constants

**File:** `core/config.py`
**Lines of Code:** 86

Centralizes all filesystem paths, model locations, and performance targets.

#### Application Identity

| Constant | Value |
|---|---|
| `APP_NAME` | `"JARVIS"` |
| `APPDATA_DIR` | `%APPDATA%/JARVIS/` |

#### Filesystem Path Constants

| Constant | Resolved Path | Description |
|---|---|---|
| `ROOT_DIR` | Project root | Resolved from `config.py` parent |
| `DATA_DIR` | `ROOT_DIR/data/` | Runtime data |
| `ASSETS_DIR` | `ROOT_DIR/assets/` | Static assets |
| `MODELS_DIR` | `ROOT_DIR/models/` | LLM GGUF files |
| `ML_MODELS_DIR` | `ROOT_DIR/ml/models/` | ML model artifacts |
| `ML_RESULTS_DIR` | `ROOT_DIR/ml/results/` | ML training results |
| `LOG_FILE` | `APPDATA_DIR/performance.log` | Performance log |
| `DB_FILE` | `APPDATA_DIR/jarvis_memory.db` | SQLite database |

#### Model Path Constants

| Constant | Path | Description |
|---|---|---|
| `DISTILBERT_ONNX_DIR` | `ml/models/distilbert_onnx/` | ONNX intent model directory |
| `BILSTM_ONNX_PATH` | `ml/models/bilstm.onnx` | BiLSTM ONNX model |
| `LINEARSVC_PATH` | `ml/models/linearsvc.pkl` | LinearSVC fallback |
| `LOGREG_PATH` | `ml/models/logreg.pkl` | Logistic regression model |
| `TOKENIZER_PATH` | `ml/models/tokenizer.pkl` | Tokenizer artifact |
| `LABEL_ENCODER_PATH` | `ml/models/label_encoder.pkl` | Label encoder |
| `GLOVE_PATH` | `data/glove/glove.6B.100d.txt` | GloVe word vectors |
| `QWEN_VL_GGUF_PATH` | `models/Qwen2.5-VL-3B-Instruct-Q8_0.gguf` | Vision-language model |
| `QWEN_VL_MMPROJ_PATH` | `models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf` | Vision projector |
| `QWEN_TEXT_FALLBACK_GGUF_PATH` | `models/qwen2.5-3b-instruct-q4_k_m.gguf` | Text-only fallback |
| `CNN_VISION_WEIGHTS_PATH` | `ml/models/cnn_vision/cnn_scratch.pt` | CNN weights |
| `CNN_VISION_LABELS_PATH` | `ml/models/cnn_vision/labels.json` | CNN class labels |

#### Asset Path Constants

| Constant | Path |
|---|---|
| `WAKE_CHIME_PATH` | `assets/sounds/wake_chime.wav` |
| `ERROR_SOUND_PATH` | `assets/sounds/error.wav` |
| `APP_ICON_PATH` | `assets/icon.ico` |

#### Threshold Constants

| Constant | Value | Description |
|---|---|---|
| `INTENT_CONFIDENCE_THRESHOLD` | `0.55` | Minimum confidence for non-general_qa classification |
| `CONTEXT_WINDOW_SIZE` | `5` | Max conversation turns in context |

#### Performance Targets Dataclass

```python
@dataclass(frozen=True)
class PerformanceTargets:
    wake_word_ms: int = 500      # Wake word detection
    stt_5s_ms: int = 800         # STT for 5s audio
    intent_ms: int = 8           # Intent classification
    entity_ms: int = 5           # Entity extraction
    app_launch_ms: int = 300     # App launch
    system_control_ms: int = 50  # Volume/brightness
    llm_first_token_ms: int = 1000  # LLM first token
    tts_start_ms: int = 200      # TTS synthesis start
```

#### Performance Logging Function

```python
log_performance(operation, latency_ms, details="")
```

Writes pipe-delimited lines to `APPDATA_DIR/performance.log` and mirrors metrics to the active session trace logger via `trace_event("backend.performance", "metric", ...)`.

---

### 4.3 Context Manager

**File:** `core/context_manager.py`
**Class:** `ContextManager`
**Lines of Code:** 125

Manages a sliding window of conversation turns with optional dense vector embeddings for semantic retrieval.

#### Conversation Turn Data Model

```python
@dataclass
class ConversationTurn:
    role: str                           # "user" or "assistant"
    text: str                           # Raw message text
    timestamp: float                    # Unix epoch timestamp
    embedding: Optional[np.ndarray]     # 384-dimensional vector
```

#### Features

| Feature | Implementation |
|---|---|
| **Sliding Window** | `deque`-based, configurable `window_size` (default: 10) |
| **Dense Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) |
| **Fallback Embeddings** | Deterministic SHA-256–seeded random vectors when sentence-transformers is unavailable |
| **Thread Safety** | All turn access is `threading.Lock`-protected |
| **Semantic Search** | `resolve_reference(query, top_k)` — cosine-similarity search over embedded turns |
| **Lazy Embedding** | `embed_recent_missing(limit)` — backfill embeddings for recent un-embedded turns |
| **Window Retrieval** | `get_window()` — returns list of `{role, text}` dicts for LLM context |
| **Turn Addition** | `add_turn(role, text, embed=False)` — append with optional immediate embedding |

---

### 4.4 Compute Runtime

**File:** `core/compute_runtime.py`
**Lines of Code:** 287

Manages hardware acceleration detection, adaptive CPU/GPU switching, query complexity analysis, and Windows CUDA DLL path management.

#### Compute Modes

| Mode | Behavior |
|---|---|
| `auto` | Dynamically selects CPU or GPU based on query complexity and hardware availability |
| `cpu` | Forces all inference to CPU |
| `gpu` | Forces GPU acceleration (with automatic CPU fallback on errors) |

#### Query Complexity Scorer (`estimate_query_complexity`)

The system uses a weighted scoring formula to determine query complexity:

```
score = (token_count × 0.45) + (keyword_hits × 2.6) + (punctuation_hits × 0.4) + (conjunction_hits × 1.2)
if token_count ≥ 16: score += 2.0
```

**Complexity keywords:** `explain`, `compare`, `difference`, `summarize`, `analysis`, `latest`, `news`, `why`, `how`, `reason`, `steps`, `strategy`, `multi`, `research`

**Conjunction keywords:** `and`, `or`, `while`, `whereas`, `because`, `although`, `however`, `including`, `versus`, `compare`

A query is classified as **complex** when: `score ≥ 10.0 OR token_count ≥ 22 OR keyword_hits ≥ 2`

| Return Field | Type | Description |
|---|---|---|
| `score` | `float` | Weighted complexity score |
| `token_count` | `int` | Number of alphanumeric tokens |
| `keyword_hits` | `int` | Number of complexity keywords found |
| `is_complex` | `bool` | Whether query warrants GPU acceleration |

#### Windows CUDA DLL Management

Required DLLs for CUDA operations:

| DLL Name | Purpose |
|---|---|
| `cudnn_ops64_9.dll` | cuDNN operations |
| `cudnn_cnn64_9.dll` | cuDNN CNN kernels |
| `cudnn64_9.dll` | cuDNN core |
| `cublas64_12.dll` | cuBLAS matrix operations |
| `cublasLt64_12.dll` | cuBLAS light |
| `cudart64_12.dll` | CUDA runtime |

`ensure_windows_cuda_dll_paths()` discovers and registers CUDA directories:
1. Check `CUDA_PATH` and `CUDAToolkit_ROOT` environment variables
2. Scan `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*/bin/`
3. Add discovered directories to `PATH` and register via `os.add_dll_directory()`

#### GPU Capability Detection Flow (`detect_compute_capabilities`)

```
detect_compute_capabilities()
    │
    ├── ONNX Runtime providers (ort.get_available_providers())
    ├── PyTorch CUDA availability (torch.cuda.is_available())
    ├── GPU device name (torch.cuda.get_device_name(0))
    ├── GPUtil GPU count and name
    ├── llama-cpp GPU offload (subprocess probe → llama_supports_gpu_offload())
    ├── ctranslate2 CUDA device count
    └── Windows CUDA DLL preflight check
```

Returns dictionary with keys: `gpu_supported`, `gpu_name`, `onnx_providers`, `onnx_cuda_available`, `torch_cuda_available`, `gpu_count`, `llm_gpu_offload_supported`, `llm_gpu_reason`, `stt_cuda_ready`, `stt_cuda_reason`

#### Environment Variable Application (`apply_compute_environment`)

| Mode | `JARVIS_ENABLE_LLM_GPU` | `JARVIS_STT_USE_CUDA` | `JARVIS_INTENT_PROVIDER` |
|---|---|---|---|
| `cpu` | `0` | `0` | `cpu` |
| `gpu` | `1` | `1` | `gpu` |
| `auto` | `1` | `0` | `auto` |

---

### 4.5 Runtime Settings

**File:** `core/runtime_settings.py`
**Lines of Code:** 60

Manages persistent JSON-based settings stored at `%APPDATA%/JARVIS/runtime_settings.json`.

#### Persisted Settings with Defaults

| Setting | Type | Default | Valid Values | Description |
|---|---|---|---|---|
| `compute_mode` | `str` | `"auto"` | `auto`, `cpu`, `gpu` | Hardware acceleration mode |
| `tts_enabled` | `bool` | `true` | boolean | Enable spoken responses |
| `tts_profile` | `str` | `"female"` | `female`, `male` | Voice gender profile |
| `response_verbosity` | `str` | `"normal"` | `brief`, `normal`, `detailed` | Response length mode |

Settings are merged with defaults on every load (`_merge_defaults`), ensuring forward-compatible schema evolution.

---

### 4.6 Session Logging & Telemetry

**File:** `core/session_logging.py`
**Lines of Code:** 286

Provides structured telemetry, dual-stream output capture, and exception hook installation across all subsystems.

#### SessionTerminalLogger Architecture

| Component | Description |
|---|---|
| **_TeeTextStream** | Custom `io.TextIOBase` subclass that mirrors all writes to both the original stream and a log file |
| **stdout/stderr capture** | `sys.stdout` and `sys.stderr` are replaced with `_TeeTextStream` instances on session start |
| **Exception hooks** | Both `sys.excepthook` and `threading.excepthook` are wrapped to capture unhandled exceptions |
| **Thread safety** | Event sequence numbers and stream writes are protected by `threading.Lock` |
| **atexit handler** | `atexit.register(self._atexit_stop)` ensures logs are finalized on exit |

#### Log File Format

| File | Format | Path Pattern |
|---|---|---|
| **Terminal log** | Plain text (stdout+stderr mirror) | `logs/{start} -- {end}.log` |
| **Trace log** | JSONL (one JSON object per line) | `logs/{start} -- {end}.trace.jsonl` |

While running, files use `-- active.log` / `-- active.trace.jsonl` suffix, renamed on session stop.

#### JSONL Trace Event Schema

```json
{
  "seq": 42,
  "ts": "2026-04-13T04:30:00.123",
  "epoch_ms": 1776230400123,
  "process_id": 12345,
  "thread_id": 67890,
  "thread_name": "MainThread",
  "source": "backend.pipeline",
  "event": "process_text_started",
  "details": {
    "text_chars": 28,
    "capture_tts_future": false
  }
}
```

#### Global Trace Functions

| Function | Usage |
|---|---|
| `trace_event(component, kind, **details)` | Log a structured trace event via the active session logger |
| `trace_exception(component, exc, **details)` | Log an exception with automatic stack trace extraction |

The `_sanitize_for_json()` helper handles serialization of `Path` objects, `bytes` (as `<bytes:N>`), strings (truncated at 4000 chars), and nested containers.

---

## 5. Natural Language Processing

### 5.1 Intent Classifier

**File:** `nlp/intent_classifier.py`
**Class:** `IntentClassifier`
**Lines of Code:** 533

A multi-model intent classification engine that maps raw user text to one of 14 predefined intent categories.

#### Supported Intent Categories (Exact from Code)

| Intent | Description | Example |
|---|---|---|
| `launch_app` | Open or start desktop apps | "Open Chrome" |
| `close_app` | Close or quit running apps | "Close Spotify" |
| `web_search` | Search the web for information | "Look up latest AI news" |
| `open_website` | Navigate to a URL | "Open github.com" |
| `play_media` | Play music or media | "Play lo-fi on YouTube" |
| `system_volume` | Change or query system volume | "Set volume to 40" |
| `system_brightness` | Change or query screen brightness | "Set brightness to 65" |
| `power_control` | System power commands | "Turn off monitor" |
| `system_settings` | Open Windows settings pages | "Open Bluetooth settings" |
| `general_qa` | General questions and conversation | "Explain transformers in NLP" |
| `vision_query` | Analyze camera, screenshots, images | "What do you see on screen" |
| `file_control` | Find/open local files | "Find my resume PDF" |
| `clipboard_action` | Read/copy/paste clipboard | "Read clipboard" |
| `stop_cancel` | Stop current assistant action | "Cancel that" |

#### Available Model Backends

| Model Name | Aliases | Engine | Runtime Label |
|---|---|---|---|
| `DistilBERT` | `distilbert`, `distil_bert`, `distil bert` | ONNX Runtime | `distilbert_onnx_cpu` / `distilbert_onnx_gpu` |
| `LinearSVC` | `linearsvc`, `linear_svc`, `linear svc` | scikit-learn (joblib) | `linearsvc` |
| `BiLSTM` | `bilstm`, `bi_lstm`, `bi lstm` | (falls back to LinearSVC) | `bilstm_fallback` |

#### ONNX Runtime Initialization

```
_init_onnx_runtime()
    │
    ├── Load ONNX session options (graph optimization, memory patterns)
    ├── Query available providers (ort.get_available_providers())
    ├── Create CPU session (always)
    ├── Attempt GPU session (if CUDAExecutionProvider available):
    │   ├── Windows CUDA preflight check (windows_cuda_runtime_ready())
    │   ├── Create GPU session with [CUDA, CPU] providers
    │   └── Verify CUDAExecutionProvider is active in session
    ├── Select initial active session (CPU or GPU based on startup query)
    ├── Load DistilBert tokenizer (from_pretrained, local_files_only by default)
    └── Set runtime label: distilbert_onnx_cpu or distilbert_onnx_gpu
```

#### ONNX Session Selection Logic

```python
_select_onnx_session_key(texts)
    ├── compute_mode == "cpu" → "cpu"
    ├── compute_mode == "gpu" → "gpu" if available, else "cpu"
    ├── device_hint == "gpu" and GPU available → "gpu"
    ├── device_hint == "cpu" → "cpu"
    └── auto mode → choose_device_for_query() → "gpu" if complex
```

#### Fallback Runtime

When ONNX is unavailable or disabled (`JARVIS_ENABLE_ONNX_INTENT=0` or `JARVIS_DISABLE_ONNX_INTENT=1`):
1. Load `linearsvc.pkl` from `ML_MODELS_DIR`
2. Use `decision_function()` to produce per-class scores
3. Map sklearn class indices to label encoder indices via `_resolve_class_index()`

#### Inference Pipeline

```
predict(text)
    │
    ├── Small-talk override check (exact match / prefix match):
    │   • "hi", "hello", "hey", "yo", "how are you", ...
    │   • Returns general_qa with confidence=1.0 immediately
    │
    ├── _run([text]):
    │   ├── ONNX available? → Tokenize → Session.run() → raw logits
    │   └── else → _fallback_logits() via LinearSVC
    │
    ├── Softmax normalization
    ├── Extract best_idx and confidence
    ├── If ONNX used and confidence < 0.55 → override to general_qa
    └── Build score_map {intent: probability} for all classes
```

#### Batch Prediction

```python
predict_batch(texts: List[str]) -> List[Dict[str, object]]
```

Processes multiple texts in a single ONNX inference call, applying small-talk overrides per item and distributing elapsed time evenly across results.

#### Diagnostics Output

```json
{
  "intent": "launch_app",
  "confidence": 0.87,
  "all_scores": {"launch_app": 0.87, "general_qa": 0.06, "web_search": 0.04, ...},
  "latency_ms": 23.4,
  "runtime": "distilbert_onnx_cpu",
  "provider": "CPUExecutionProvider"
}
```

---

### 5.2 Entity Extractor

**File:** `nlp/entity_extractor.py`
**Class:** `EntityExtractor`
**Lines of Code:** 261

Extracts structured parameters from natural language commands using a combination of spaCy NER, regex patterns, entity ruler patterns, and heuristic rules.

#### spaCy Model Loading

```
_load_spacy_model()
    ├── Check JARVIS_ENABLE_SPACY == "1"
    ├── Check Python < 3.13 (spaCy compatibility)
    ├── If JARVIS_USE_SPACY_TRF == "1" → try en_core_web_trf first
    └── Fallback to en_core_web_sm
```

#### Entity Ruler Patterns

The extractor installs a custom `entity_ruler` before spaCy's NER component with patterns for:

**APP_NAME labels (16 apps):** `chrome`, `firefox`, `spotify`, `vscode`, `notepad`, `calculator`, `discord`, `steam`, `photoshop`, `vlc`, `word`, `excel`, `powerpoint`, `task manager`, `file explorer`, `paint`

**PLATFORM_NAME labels (8 platforms):** `spotify`, `youtube`, `netflix`, `prime`, `soundcloud`, `local`, `drive`, `pc`

#### Compiled Regex Patterns

| Pattern | Purpose | Example Match |
|---|---|---|
| `URL_RE` | Detect URLs | `https://example.com`, `www.site.net` |
| `SEARCH_RE` | Extract search queries | "search for **Python tutorials**" |
| `MEDIA_RE` | Extract media title + platform | "play **Bohemian Rhapsody** on **Spotify**" |
| `VOLUME_RE` | Extract numeric percentages | "**50** **%**" |
| `BRIGHTNESS_RE` | Extract brightness percentages | "**70** **percent**" |
| `CAPITALIZED_APP_RE` | Detect capitalized app names | "open **Visual Studio Code**" |
| `APP_COMMAND_RE` | Extract app name after command verb | "launch **google chrome**" |
| `APP_SUFFIX_SPLIT_RE` | Strip trailing politeness | "**please**", "**for me**", "**thanks**" |

#### Extraction Logic by Intent

| Intent | Extracted Slots |
|---|---|
| `launch_app` / `close_app` | `app_name` (spaCy NER → regex → catalog scan) |
| `web_search` | `search_query` (regex or full normalized text) |
| `open_website` | `website_url` (URL regex → app_name fallback) |
| `play_media` | `media_title`, `platform` (regex → spaCy PLATFORM_NAME) |
| `system_volume` | `volume_level`, `direction` (numeric/keyword + up/down) |
| `system_brightness` | `brightness_level`, `direction` |
| `power_control` | `power_command` (14 keyword matches) |
| `system_settings` | `setting_name` (9 setting keywords) |
| `clipboard_action` | `clipboard_action` (`read`/`copy`/`paste`) |
| Any | `person_name` (spaCy PERSON entity) |

#### Power Commands Recognized

`shutdown`, `restart`, `sleep`, `hibernate`, `lock`, `turn off monitor`, `monitor off`, `wifi on`, `wifi off`, `bluetooth on`, `bluetooth off`, `airplane mode on`, `airplane mode off`, `battery saver on`, `battery saver off`

#### Volume/Brightness Keywords

`max`, `min`, `half`, `zero`, `mute`, `unmute` + any numeric value 0-100

---

### 5.3 Router

**File:** `nlp/router.py`
**Class:** `Router`
**Lines of Code:** 817

The Router is the decision engine that maps classified intents to concrete actions. It owns instances of `WebcamCapture`, `ScreenCapture`, and optionally `CNNImageClassifier`.

#### Router Constructor State

| Attribute | Type | Default | Description |
|---|---|---|---|
| `llm` | `QwenBridge | None` | Passed | LLM reference for generative queries |
| `cancel_callback` | `Callable` | Passed | Pipeline's cancel function |
| `_llm_retry_cooldown_s` | `float` | `1.5` | Seconds between LLM re-init attempts |
| `_response_verbosity` | `str` | `"normal"` | Current verbosity setting |
| `realtime_web_enabled` | `bool` | `False` | Verified web mode toggle |
| `_cnn` | `CNNImageClassifier | None` | `None` | Lazy-loaded CNN (attempted once) |
| `_camera` | `WebcamCapture` | Instance | Webcam capture utility |
| `_screen_capture` | `ScreenCapture` | Instance | Screen capture utility |
| `_last_route_result` | `Dict | None` | `None` | Compact record of last route for LLM context |

#### Routing Priority (Exact Order)

```
route(intent_result, entities, raw_text, context, compute_hint)
    │
    ├── 1. FAST-PATH CHECK (_route_fast_paths):
    │   ├── time_control.looks_like_time_query() → time_control.handle_time_query()
    │   ├── system_info.looks_like_system_info_query() → system_info.get_system_awareness()
    │   ├── "rescan app/refresh app index" → app_control.rescan_app_index()
    │   ├── "rescan music/refresh music index" → media_control.rescan_media_index()
    │   ├── "current volume/what is the volume" → system_control.get_volume()
    │   ├── "current brightness/what is the brightness" → system_control.get_brightness()
    │   ├── "pause music/song/track" → media_control.pause()
    │   ├── "resume music/song/track" → media_control.resume()
    │   ├── "stop music/song/track" → media_control.stop()
    │   ├── "next track/song" → media_control.next_track()
    │   ├── "previous track/song" → media_control.previous_track()
    │   ├── "turn off monitor/monitor off" → system_control.power_action("monitor_off")
    │   ├── "wifi on/off" → system_control.toggle_wifi()
    │   ├── "bluetooth on/off" → system_control.toggle_bluetooth()
    │   ├── "airplane mode on/off" → system_control.toggle_airplane_mode()
    │   └── "battery saver on/off" → system_control.toggle_battery_saver()
    │
    ├── 2. SMALL-TALK CHECK (predefined responses):
    │   Exact: "hi", "hello", "hey", "yo", "how are you", "good morning", "thanks", etc.
    │   Prefixes: "hi ", "hello ", "hey ", "how are you", "how r you"
    │
    ├── 3. INTENT-BASED ROUTING:
    │   ├── launch_app → app_control.launch_app(entities["app_name"])
    │   ├── close_app → app_control.close_app(entities["app_name"])
    │   ├── web_search:
    │   │   ├── Small-talk override → treat as general_qa
    │   │   ├── Verified web eligible → realtime_web.verified_answer()
    │   │   └── Direct search → web_control.search(query, platform)
    │   ├── open_website → web_control.open_url(url_or_name)
    │   ├── play_media → media_control.play(title, platform)
    │   ├── system_volume → system_control.set_volume(level, direction)
    │   ├── system_brightness → system_control.set_brightness(level, direction)
    │   ├── power_control → system_control.power_action(command)
    │   ├── system_settings → system_control.open_settings(setting_name)
    │   ├── file_control → file_control.handle(entities)
    │   ├── clipboard_action → clipboard_control.handle(entities)
    │   ├── vision_query:
    │   │   ├── Not actually visual? → reclassify as general_qa
    │   │   ├── mode="image" → analyze_image_file(file_path)
    │   │   ├── mode="camera" → analyze_camera_capture()
    │   │   └── mode="screen" → _fallback_screen_analysis()
    │   └── stop_cancel → cancel_callback() or default stop response
    │
    ├── 4. VERIFIED WEB FALLBACK:
    │   ├── Check _should_use_verified_web() for general_qa
    │   └── realtime_web.verified_answer(raw_text, llm)
    │
    ├── 5. LLM GENERATION:
    │   ├── _ensure_llm() (with 1.5s retry cooldown)
    │   ├── _build_general_system_prompt() (see below)
    │   └── llm.generate(raw_text, context, device_hint, system_prompt)
    │
    └── 6. FALLBACK RESPONSE (if LLM unavailable):
        └── Predefined response based on text pattern
```

#### System Prompt Builder (`_build_general_system_prompt`)

When routing to the LLM, the router constructs a detailed system prompt including:

```
You are JARVIS (Just A Really Very Intelligent System), a local desktop assistant...
Hybrid runtime policy:
- System operations are executed by deterministic handlers outside the LLM.
- Never claim you changed system state unless an executed result is provided.
- If a user asks for an operation that was not executed, clarify limitations.
- For informational and conversational questions, provide the best direct answer.
- Do not prepend time/date unless the user asked for it.
[verbosity rule based on current mode]
Current runtime time: [formatted datetime]
Realtime web mode: enabled/disabled
Current intent hint: [intent]
Extracted entities: [key=value pairs]
Compute hint: [cpu/gpu/auto]
Latest user input: [clipped raw text]
[Last route result if available]
```

#### Verified Web Decision Logic (`_should_use_verified_web`)

```
_should_use_verified_web(raw_text, intent)
    ├── realtime_web_enabled must be True
    ├── Not empty, not small talk
    ├── Must contain explicit markers: "search web", "look up", "latest",
    │   "news", "today", "current", "update", "recent", "price",
    │   "weather", "score", "stock"
    ├── Intent must be web_search or general_qa
    └── realtime_web.looks_like_research_query() must return True
```

#### Vision Request Detection (`_is_visual_request`)

Checks entities for `vision_mode` in `{screen, image, camera}`, `file_path` presence, or regex patterns:
- `\b(screen|screenshot|display|monitor)\b`
- `\b(image|photo|picture|pic)\b`
- `\b(camera|webcam)\b`
- `\bwhat do you see\b`
- `\bdescribe (this|the) (image|photo|screen)\b`

#### Image Analysis Pipeline

```
analyze_image_file(file_path, prompt)
    │
    ├── CNN classification (if available): _ensure_cnn() → classify_image()
    ├── Image preprocessing:
    │   ├── PIL.Image.open() → RGB conversion
    │   ├── Thumbnail to max_edge (JARVIS_VISION_MAX_EDGE, default 512, range 384-1024)
    │   └── Save to PNG byte buffer
    ├── LLM vision inference (if available):
    │   ├── Build VL prompt with sanitization rules
    │   ├── llm.generate(prompt, [], device_hint="gpu", image_bytes=...)
    │   ├── Validate response (not an error message)
    │   └── _sanitize_vision_response() — strip time/date prefixes
    └── Fallback cascade:
        ├── LLM usable → return LLM response with CNN hints
        ├── CNN usable → return CNN fallback with VL unavailable note
        └── Neither → return "Image analysis is unavailable"
```

---

### 5.4 Text Preprocessor

**File:** `nlp/preprocessor.py`
**Lines of Code:** 28

Normalizes raw user input before any downstream NLP processing.

#### Processing Steps

| Step | Code | Example |
|---|---|---|
| Lowercase | `.strip().lower()` | "Open Chrome" → "open chrome" |
| Normalize percent words | `normalize_percent_words()` | "maximum" → "max"; "percent" → "%" |
| Strip special punctuation | `_PUNCT_RE.sub(" ", ...)` | Removes chars not in `[\w\s\-\.\/:%]` |
| Collapse whitespace | `_WHITESPACE_RE.sub(" ", ...)` | Multiple spaces → single space |

---

### 5.5 Conversation Normalizer

**File:** `nlp/conversation_normalizer.py`
**Lines of Code:** 76

Strips conversational filler and polite phrases to isolate the actionable command core.

#### Filler Detection Regex

```python
_FILLER_RE = r"\b(?:um+|uh+|hmm+|er+|ah+)\b"  # Speech disfluencies
```

#### Assistant Name Prefix Regex

```python
_ASSISTANT_PREFIX_RE = r"^\s*(?:hey|hi|hello|ok|okay|yo)?\s*(?:jarvis|friday|assistant|aria)\b[\s,!:;-]*"
```

#### Leading Polite Phrase Patterns (7 patterns, up to 8 iterations)

| Pattern | Example Stripped |
|---|---|
| `do me a favor (and)` | "Do me a favor and open Chrome" |
| `can/could/would/will/shall you` | "Could you open Chrome" |
| `would you mind` | "Would you mind opening Chrome" |
| `i need/want you to` | "I need you to open Chrome" |
| `i would/i'd like you to` | "I'd like you to open Chrome" |
| `help me (to)` | "Help me find my resume" |
| `please/kindly/just` | "Please open Chrome" |

#### Trailing Polite Phrase Patterns (2 patterns, up to 6 iterations)

| Pattern | Example Stripped |
|---|---|
| `please/thanks/thank you` | "Open Chrome please" |
| `for me/if you can/if possible/right now` | "Open Chrome for me" |

#### Full Normalization Pipeline

```python
normalize_command_text(text)
    ├── Lowercase and collapse whitespace
    ├── Remove speech fillers (um, uh, hmm, er, ah)
    ├── Strip assistant name prefix (jarvis, friday, assistant, aria)
    ├── Strip leading polite phrases (up to 8 iterations)
    ├── Strip trailing polite phrases (up to 6 iterations)
    └── Final whitespace collapse
```

#### Examples

| Input | Normalized Output |
|---|---|
| "Hey JARVIS, could you please open Chrome for me?" | "open chrome" |
| "Um, I'd like you to search for weather in Tokyo thanks" | "search for weather in tokyo" |
| "Okay Jarvis just tell me the time" | "tell me the time" |
| "Do me a favor and set volume to 50 if you can" | "set volume to 50" |

---

## 6. Large Language Model Integration

### 6.1 QwenBridge

**File:** `llm/qwen_bridge.py`
**Class:** `QwenBridge`
**Lines of Code:** 1,089

The primary interface for local LLM inference, supporting both **in-process** and **subprocess worker** execution modes.

#### Execution Modes

| Mode | Description | Use Case |
|---|---|---|
| **In-Process** | `llama-cpp` loaded directly in the main Python process | Development, debugging |
| **Worker Subprocess** | Isolated `qwen_worker.py` process communicating via JSON-over-stdio | Production (default, more stable) |

#### Model Resolution

```
_resolve_model_path()
    │
    ├── $JARVIS_LLM_MODEL_PATH set? → Use explicit path
    │
    ├── Qwen2.5-VL GGUF exists? → Use VL model (supports vision)
    │
    ├── Qwen text-only GGUF exists? → Use text fallback
    │
    └── Return preferred path (will error on load)
```

#### Auto Model Switching

The `_auto_model_switch` feature dynamically switches between the **vision model** (Qwen2.5-VL) and the lighter **text-only model** based on request type:

- **Image requests** → Loads VL model + mmproj file
- **Text requests** → Loads lightweight text-only model for faster inference
- **Model switch** → Gracefully shuts down current runtime, loads new model

#### GPU Offloading Strategy

```
_select_gpu_for_request(user_message, device_hint)
    │
    ├── Explicit hint ("cpu"/"gpu") → Use hint
    │
    ├── compute_mode = "cpu" → Always CPU
    │
    ├── compute_mode = "gpu" → GPU if supported
    │
    └── compute_mode = "auto" →
         ├── Query complexity analysis (choose_device_for_query)
         ├── GPU retry cooldown (60s after failure)
         ├── Auto-switch cooldown (15s between switches)
         └── Simple query streak counter (switch to CPU after 3)
```

#### Incomplete Response Completion

The bridge detects truncated LLM responses and automatically issues a continuation prompt:

1. Check if response ends mid-sentence (missing terminal punctuation)
2. Issue continuation: "Continue from your previous answer without repeating yourself."
3. Merge the continuation with the original response, deduplicating overlap

#### Key Configuration

| Parameter | Environment Variable | Default | Description |
|---|---|---|---|
| Thread count | `JARVIS_LLM_THREADS` | `4` | CPU threads for inference |
| Worker ready timeout | `JARVIS_LLM_WORKER_READY_TIMEOUT_S` | `25` | Seconds to wait for worker startup |
| Response timeout | `JARVIS_LLM_WORKER_RESPONSE_TIMEOUT_S` | `45` | Seconds to wait for generation |
| Vision timeout | `JARVIS_LLM_VISION_TIMEOUT_S` | `240` | Extended timeout for vision inference |
| CPU max tokens | `JARVIS_LLM_MAX_TOKENS_CPU` | `160` | Token limit when running on CPU |
| Context turns | `JARVIS_LLM_CONTEXT_TURNS` | `2` | Conversation history turns sent to LLM |
| GPU retry cooldown | `JARVIS_GPU_RETRY_COOLDOWN` | `60` | Seconds before retrying GPU after failure |
| Auto-switch cooldown | `JARVIS_AUTO_COMPUTE_SWITCH_COOLDOWN` | `15` | Seconds between compute mode switches |

#### Adaptive Max Token Budget

| User Message Length (words) | Image Mode | Max Tokens |
|---|---|---|
| ≤ 8 | No | 96 |
| 9 – 18 | No | 144 |
| 19 – 32 | No | 220 |
| > 32 | No | 320 |
| Any | Yes | 128 |

---

### 6.2 Qwen Worker Process

**File:** `llm/qwen_worker.py`
**Lines of Code:** 284

An isolated subprocess that loads the LLM in its own process space, communicating with the parent via JSON-over-stdio. This architecture:

- Prevents GPU driver crashes from taking down the main UI process
- Allows force-termination of stuck inference without affecting the application
- Isolates `llama-cpp` memory from the PyQt6 event loop

#### Communication Protocol

**Startup:**
```json
// Worker → Parent (success)
{"ok": true, "event": "ready", "backend": "cpu", "gpu_offload_supported": false, "supports_vision": true}

// Worker → Parent (failure)
{"ok": false, "event": "init", "error": "Missing model file: ..."}
```

**Generation Request:**
```json
// Parent → Worker
{"type": "generate", "user_message": "What is gravity?", "context": [...], "max_tokens": 144}

// Worker → Parent (success)
{"ok": true, "text": "Gravity is a fundamental force..."}

// Worker → Parent (error)
{"ok": false, "error": "..."}
```

**Shutdown:**
```json
// Parent → Worker
{"type": "shutdown"}

// Worker → Parent
{"ok": true, "event": "bye"}
```

---

### 6.3 Model Resolution & Auto-Switch

#### Supported Model Files

| File | Description | Required |
|---|---|---|
| `Qwen2.5-VL-3B-Instruct-Q8_0.gguf` | Vision-Language model (multimodal) | For vision features |
| `qwen2.5-3b-instruct-q4_k_m.gguf` | Text-only instruct model | For text-only queries |
| `mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf` | Multimodal vision projector | For vision features |

#### Inference Parameters

| Parameter | Value | Purpose |
|---|---|---|
| `n_ctx` | 4096 | Context window size (tokens) |
| `temperature` | 0.7 | Sampling randomness |
| `top_p` | 0.9 | Nucleus sampling threshold |
| `repeat_penalty` | 1.1 | Repetition suppression |
| `chat_format` | `chatml` | Template for text-only models |

---

## 7. Speech Processing

### 7.1 Speech-to-Text (STT)

**File:** `speech/stt.py`
**Class:** `SpeechToText`
**Lines of Code:** 336

Uses **faster-whisper** (CTranslate2 backend) for high-accuracy English transcription.

#### Model Configuration

| Compute Mode | Model | Device | Compute Type |
|---|---|---|---|
| GPU | `medium.en` | CUDA | float16 |
| CPU | `small.en` | CPU | int8 |
| Auto | `small.en` (CPU default) | CPU | int8 |
| Auto + `JARVIS_STT_USE_CUDA=1` | `medium.en` (CUDA first) | CUDA → CPU fallback | float16 → int8 |

#### Audio Capture Parameters

| Parameter | Value |
|---|---|
| Sample Rate | 16,000 Hz |
| Block Duration | 200 ms |
| Block Samples | 3,200 |
| Silence Threshold | 0.01 RMS |
| Silence Limit | 6 blocks (1.2s silence triggers stop) |
| VAD Filter | Enabled |
| Language | English (`en`) |

#### Error Recovery

The STT engine implements automatic CUDA-to-CPU fallback:

1. Attempt transcription on CUDA
2. On CUDA runtime error → log error, reset to CPU mode
3. Re-initialize with `small.en` on CPU
4. Retry transcription
5. Log recovery status

#### API Methods

| Method | Description |
|---|---|
| `warmup()` | Pre-load the model with a dummy inference pass |
| `listen_once(timeout=10)` | Record from microphone until silence, return transcript |
| `transcribe(audio_array, sample_rate)` | Transcribe pre-recorded audio array |
| `set_compute_mode(mode)` | Hot-swap between CPU and GPU |
| `get_status()` | Return `{available, initialized, reason, device}` |

#### Transcription Result Format

```python
{
    "text": "What time is it in Tokyo?",
    "language": "en",
    "confidence": 0.92,
    "duration_ms": 1247.3,
    "error": ""
}
```

---

### 7.2 Text-to-Speech (TTS)

**File:** `speech/tts.py`
**Class:** `TextToSpeech`
**Lines of Code:** 377

Multi-backend TTS engine with three synthesis backends and configurable voice profiles.

#### Backend Priority (Auto Mode)

```
1. Edge TTS (Microsoft Neural Voices)  ← Primary
       │ On failure ↓
2. Kokoro-82M (Local Neural TTS)       ← Secondary
       │ On failure ↓
3. Windows SAPI (System Voices)        ← Fallback
```

#### Voice Profiles

| Profile | Edge TTS Voice | Kokoro Voice | SAPI Hint | Rate |
|---|---|---|---|---|
| **Female** | `en-IE-EmilyNeural` | `af_heart` | `Zira` | 0 |
| **Male** | `en-GB-RyanNeural` | `am_adam` | `David` | -1 |

#### Backend Details

**Edge TTS:**
- Microsoft's neural voices via `edge-tts` library
- Streaming synthesis with chunk reassembly
- Volume normalization to 0.9 peak
- Supports `+0%` rate and `+0Hz` pitch customization

**Kokoro-82M:**
- Fully offline neural TTS via `KPipeline` API
- HuggingFace Hub integration (`hexgrad/Kokoro-82M`)
- Supports voice candidates with automatic fallback
- Configurable device (`JARVIS_TTS_DEVICE`: `cpu` or `cuda`)
- Voice pack caching validation before synthesis

**Windows SAPI:**
- PowerShell-invoked `System.Speech.Synthesis`
- Voice selection by hint string (partial match)
- Configurable rate per profile
- Text passed via stdin to avoid escaping issues
- 120-second timeout

#### Key Environment Variables

| Variable | Default | Description |
|---|---|---|
| `JARVIS_TTS_BACKEND` | `auto` | Force backend: `auto`, `edge`, `kokoro`, `sapi` |
| `JARVIS_TTS_DEVICE` | `cpu` | Kokoro device: `cpu` or `cuda` |
| `JARVIS_TTS_REPO` | `hexgrad/Kokoro-82M` | HuggingFace repo for Kokoro model |
| `JARVIS_TTS_ALLOW_KOKORO_DOWNLOAD` | `0` | Allow first-run Kokoro model download |
| `JARVIS_TTS_NO_PLAYBACK` | `0` | Skip audio playback (for testing) |

#### API Methods

| Method | Description |
|---|---|
| `speak_async(text)` | Submit text for async synthesis, returns Future |
| `stop()` | Immediately halt current playback |
| `set_profile(profile)` | Switch between female/male voice |

---

### 7.3 Wake Word Detection

**File:** `speech/wake_word.py`
**Class:** `WakeWordDetector(QObject)`
**Lines of Code:** 284

Provides continuous background listening for wake word activation with dual detection modes.

#### Detection Modes

| Mode | Engine | Description |
|---|---|---|
| **Neural** | openWakeWord (ONNX) | Real-time audio stream analysis with ML model |
| **Phrase Fallback** | faster-whisper STT | Periodic transcription checking for activation phrases |

#### Activation Flow

```
Background Listening Loop
    │
    ├── [Neural Mode]
    │   ├── Audio chunk from InputStream (1280 samples, 80ms)
    │   ├── Predict wake word score
    │   ├── Score ≥ sensitivity threshold?
    │   │   ├── Yes → Emit wake_word_detected signal
    │   │   │        → Play chime
    │   │   │        → Pause stream
    │   │   │        → Capture follow-up command
    │   │   │        → Resume stream
    │   │   └── No → Continue listening
    │   └── Loop
    │
    └── [Phrase Fallback Mode]
        ├── STT listen_once(timeout=4s)
        ├── Check for activation phrase in transcript
        ├── Phrase found?
        │   ├── Yes → Emit wake_word_detected signal
        │   │        → Strip phrase prefix
        │   │        → Process remaining text as command
        │   └── No → Continue listening
        └── Loop
```

#### Qt Signals

| Signal | Description |
|---|---|
| `transcript_ready(str)` | Emitted with transcribed follow-up text |
| `state_changed(str)` | Emitted on state transitions (`IDLE`, `LISTENING`) |
| `wake_acknowledged(str)` | Emitted with acknowledgment prompt text |

---

### 7.4 Wake Word Configuration

**File:** `speech/wakeword_config.py`
**Lines of Code:** 97

Manages persistent wake word settings stored at `%APPDATA%/JARVIS/wakeword.json`.

#### Default Configuration (Exact from Code)

```json
{
  "enabled": false,
  "sensitivity": 0.35,
  "activation_phrases": ["jarvis", "hey jarvis"],
  "strict_phrase_prefix": false,
  "auto_restart_after_response": true,
  "follow_up_after_response": true,
  "follow_up_timeout": 8,
  "max_followup_turns": 1
}
```

#### Parameter Validation Ranges

| Parameter | Type | Min | Max | Default |
|---|---|---|---|---|
| `sensitivity` | float | 0.1 | 0.95 | 0.35 |
| `activation_phrases` | list | — | 8 items | `["jarvis", "hey jarvis"]` |
| `follow_up_timeout` | int | 3 | 20 | 8 |
| `max_followup_turns` | int | 0 | 3 | 1 |

#### Utility Function

`parse_phrases_csv(value)` — Splits comma-separated string into lowercase phrase list.

---

## 8. Vision & Multimodal

### 8.1 VisionModel

**File:** `vision/vision_model.py`
**Class:** `VisionModel`
**Lines of Code:** 121

Orchestrates multimodal vision tasks by combining screen/webcam/file capture with the Qwen2.5-VL language model.

#### Capabilities

| Capability | Method | Input | Output |
|---|---|---|---|
| **Screen Analysis** | `describe_screen(prompt)` | Full screenshot | Natural language description |
| **Image Analysis** | `describe_image(path, prompt)` | Image file path | Natural language description |
| **Webcam Analysis** | Via pipeline | Webcam frame | Natural language description |

#### Vision Prompt Engineering

```
"Analyze only the attached image and answer the request directly.
If details are uncertain, state uncertainty instead of guessing.
Do not prepend the current date/time unless asked or visible in the image.
User request: {user_prompt}"
```

#### Vision Response Sanitization

The `_sanitize_vision_response()` method:
1. Strip leading colons, dashes, spaces
2. If user didn't ask about time/date: remove leading time-stamp lines like "Today is..." or "The current time..."
3. Rejoin remaining lines

---

### 8.2 Screen Capture

**File:** `vision/screen_capture.py`
**Class:** `ScreenCapture`
**Lines of Code:** 32

Provides full-screen and region-based screenshot capture using `pyautogui`.

| Method | Description |
|---|---|
| `capture_full(save=True)` | Full monitor screenshot → `(image, path)` |
| `capture_region(x, y, w, h, save=True)` | Region-bounded screenshot |

Output directory: `data/captures/`
Filename format: `screen_{timestamp_ms}.png` or `region_{timestamp_ms}.png`

---

### 8.3 Webcam Capture

**File:** `vision/webcam.py`
**Class:** `WebcamCapture`
**Lines of Code:** 32

Captures single frames from the default webcam using OpenCV.

| Method | Returns |
|---|---|
| `capture_frame()` | `(success: bool, path: str | None, message: str)` |

- Default camera index: `0`
- Output: `data/captures/webcam_{timestamp_ms}.png`
- Graceful error handling for missing webcam or OpenCV

---

### 8.4 CNN Image Classifier

**File:** `vision/cnn_classifier.py`
**Class:** `CNNImageClassifier`
**Lines of Code:** 106

A custom CNN-based image classification system serving as fallback when Qwen2.5-VL is unavailable.

#### Features

- Loads trained weights from `ml/models/cnn_vision/cnn_scratch.pt`
- Reads class labels from `ml/models/cnn_vision/labels.json`
- Top-K prediction with confidence scores
- Warm-up inference on initialization (128×128 dummy tensor)
- Softmax probability output per class

#### Classification Result

```python
{
    "success": True,
    "response_text": "Top match: cat (94.2% confidence).",
    "data": {
        "image_path": "photo.jpg",
        "predictions": [
            {"label": "cat", "confidence": 0.942},
            {"label": "dog", "confidence": 0.038},
            {"label": "bird", "confidence": 0.012}
        ]
    }
}
```

---

### 8.5 CNN Architecture (ScratchVisionCNN)

**File:** `vision/cnn_scratch.py`
**Class:** `ScratchVisionCNN(nn.Module)`
**Lines of Code:** 67

A custom 4-layer convolutional neural network built from scratch using PyTorch.

#### Architecture

```
Input (3 × 128 × 128)
    │
    ├── Conv2d(3→32, 3×3, pad=1) + BatchNorm2d(32) + ReLU + MaxPool2d(2×2)    → 32 × 64 × 64
    ├── Conv2d(32→64, 3×3, pad=1) + BatchNorm2d(64) + ReLU + MaxPool2d(2×2)   → 64 × 32 × 32
    ├── Conv2d(64→128, 3×3, pad=1) + BatchNorm2d(128) + ReLU + MaxPool2d(2×2)  → 128 × 16 × 16
    ├── Conv2d(128→256, 3×3, pad=1) + BatchNorm2d(256) + ReLU + AdaptiveAvgPool2d(1×1) → 256 × 1 × 1
    │
    ├── Flatten → 256
    ├── Dropout(0.35)
    ├── Linear(256→256) + ReLU
    ├── Dropout(0.25)
    └── Linear(256→num_classes)
```

#### Image Transform Pipeline

| Transform | Training | Evaluation |
|---|---|---|
| Resize | RandomResizedCrop(128, scale=0.6–1.0) | Resize(128×128) |
| Flip | RandomHorizontalFlip(0.5) | — |
| Color Jitter | brightness=0.15, contrast=0.15 | — |
| ToTensor | ✓ | ✓ |
| Normalize | ImageNet (μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225]) | ImageNet |

---

## 9. Action Modules

### 9.1 Application Control

**File:** `actions/app_control.py`
**Lines of Code:** 441

Comprehensive application lifecycle management with intelligent fuzzy matching and filesystem scanning.

#### Built-in Application Registry (60+ apps)

| Category | Applications |
|---|---|
| **Browsers** | Chrome, Firefox, Edge, Opera, Brave, Chrome Canary |
| **Development** | VS Code, IntelliJ, PyCharm, Android Studio, Postman, Git Bash |
| **Communication** | Discord, Telegram, WhatsApp, Slack, Teams, Zoom, Skype |
| **Media** | Spotify, VLC, Audacity |
| **Creative** | Photoshop, Illustrator, Premiere, After Effects, Blender |
| **Gaming** | Steam, Epic, Battle.net, Origin, Uplay |
| **Office** | Word, Excel, PowerPoint, Outlook, OneNote |
| **System** | Task Manager, Control Panel, Registry Editor, Calculator, Paint |
| **Terminals** | PowerShell, CMD, Windows Terminal |

#### Application Index System

```
App Index Generation
    │
    ├── 1. Seed with built-in APP_MAP (60+ entries)
    ├── 2. Scan Start Menu shortcuts (.lnk files)
    │   ├── %APPDATA%\Microsoft\Windows\Start Menu\Programs\
    │   └── %ProgramData%\Microsoft\Windows\Start Menu\Programs\
    ├── 3. Scan executable directories (.exe files)
    │   ├── C:\Program Files\
    │   ├── C:\Program Files (x86)\
    │   └── %LOCALAPPDATA%\Programs\
    ├── 4. Deduplicate (prefer .exe over .lnk)
    └── 5. Persist to %APPDATA%\JARVIS\app_index.json
```

#### Application Launch Strategy

```
launch_app(app_name)
    ├── 1. Fuzzy match against app index (SequenceMatcher ≥ 0.72)
    ├── 2. Try os.startfile() (handles .lnk, App Paths, associations)
    ├── 3. Try subprocess.Popen() (direct executable)
    ├── 4. Try _find_executable() (filesystem search)
    └── 5. Try os.startfile(app_name) (system shell fallback)
```

#### API

| Function | Description |
|---|---|
| `launch_app(app_name)` | Launch app by name with fuzzy matching |
| `close_app(app_name)` | Terminate processes matching app name |
| `ensure_app_index(force_rescan)` | Build/rebuild app index |
| `rescan_app_index()` | Force full re-scan of installed apps |

---

### 9.2 System Control

**File:** `actions/system_control.py`
**Lines of Code:** 307

Deep Windows system integration for hardware and OS control.

#### Volume Control (pycaw — Windows Core Audio API)

| Command | Description |
|---|---|
| `set_volume(50)` | Set to 50% |
| `set_volume("up", step=10)` | Increase by 10% |
| `set_volume("down", step=15)` | Decrease by 15% |
| `set_volume("mute")` | Mute audio |
| `set_volume("unmute")` | Unmute audio |
| `set_volume("max")` | Set to 100% |
| `get_volume()` | Query current level & mute state |

#### Brightness Control (screen-brightness-control)

| Command | Description |
|---|---|
| `set_brightness(70)` | Set to 70% |
| `set_brightness("up", step=10)` | Increase by 10% |
| `set_brightness("down")` | Decrease by default step |
| `get_brightness()` | Query current level |

#### Power Actions

| Action | Command | Confirmation |
|---|---|---|
| Shutdown | `shutdown /s /t 5` | Double-confirm |
| Restart | `shutdown /r /t 5` | Double-confirm |
| Sleep | `rundll32 powrprof.dll,SetSuspendState` | None |
| Hibernate | `shutdown /h` | Double-confirm |
| Lock | `LockWorkStation()` (Win32 API) | None |
| Monitor Off | `SendMessage(HWND_BROADCAST, WM_SYSCOMMAND, SC_MONITORPOWER, 2)` | None |

#### Network Controls

| Feature | Implementation |
|---|---|
| Wi-Fi Toggle | PowerShell `Enable/Disable-NetAdapter` with regex matching |
| Bluetooth Toggle | PowerShell `Start/Stop-Service bthserv` |
| Airplane Mode | Opens `ms-settings:network-airplanemode` |
| Battery Saver | Opens `ms-settings:batterysaver` |

#### Windows Settings Launcher

| Name | Settings URI |
|---|---|
| `wifi` | `ms-settings:network-wifi` |
| `bluetooth` | `ms-settings:bluetooth` |
| `display` | `ms-settings:display` |
| `sound` | `ms-settings:sound` |
| `apps` | `ms-settings:appsfeatures` |
| `updates` | `ms-settings:windowsupdate` |
| `privacy` | `ms-settings:privacy` |
| `battery` | `ms-settings:batterysaver` |
| `airplane` | `ms-settings:network-airplanemode` |

---

### 9.3 Media Control

**File:** `actions/media_control.py`
**Lines of Code:** 414

Full-featured media playback system with local library management and cross-platform streaming support.

#### Local Media Library

- **Indexing:** Recursively scans all drive letters for audio files
- **Supported formats:** `.mp3`, `.flac`, `.wav`, `.m4a`, `.ogg`, `.aac`, `.wma`, `.opus`
- **Search:** Fuzzy title matching (SequenceMatcher ≥ 0.70)
- **Playback engine:** pygame.mixer
- **Controls:** Play, pause, resume, stop, next, previous
- **Persistent index:** `%APPDATA%/JARVIS/media_index.json`

#### Streaming Platform Integration

| Platform | Search Method | Native App Support |
|---|---|---|
| **Spotify** | `spotify:search:` URI scheme | ✓ (native app detection) |
| **YouTube** | Direct video URL extraction (regex from search results) | — |
| **YouTube Music** | `music.youtube.com/search` | — |
| **Apple Music** | `music.apple.com/search` | ✓ (iTunes detection) |
| **SoundCloud** | `soundcloud.com/search` | — |
| **Deezer** | `deezer.com/search` | — |
| **Tidal** | `listen.tidal.com/search` | — |
| **Amazon Music** | `music.amazon.com/search` | — |
| **Gaana** | `gaana.com/search` | — |
| **JioSaavn** | `jiosaavn.com/search` | — |

#### Playback Resolution Order

```
play(title, platform="auto")
    │
    ├── platform == "local" or "auto"?
    │   ├── Search local media library
    │   ├── Match found? → Play via pygame.mixer
    │   └── No match → Fall through to YouTube
    │
    ├── platform == "spotify"?
    │   ├── Spotify app installed? → Native URI launch
    │   └── Not installed → Spotify Web
    │
    ├── platform == "youtube"?
    │   ├── Extract first video URL from search
    │   ├── Found? → Open with autoplay
    │   └── Not found → Open search results page
    │
    └── Other platform → Open web search URL
```

---

### 9.4 Web Control

**File:** `actions/web_control.py`
**Lines of Code:** 180

Browser-based web navigation and search with 100+ pre-mapped websites.

#### Search Platforms

| Platform | URL Template |
|---|---|
| Google | `google.com/search?q={}` |
| YouTube | `youtube.com/results?search_query={}` |
| Reddit | `reddit.com/search/?q={}` |
| Twitter/X | `twitter.com/search?q={}` |
| Amazon | `amazon.in/s?k={}` |
| GitHub | `github.com/search?q={}` |
| Stack Overflow | `stackoverflow.com/search?q={}` |
| Wikipedia | `en.wikipedia.org/wiki/Special:Search?search={}` |
| Google Maps | `maps.google.com/search?q={}` |

#### Website Registry (100+ sites by category)

| Category | Sites |
|---|---|
| **Productivity** | Gmail, Drive, Docs, Sheets, Slides, Calendar, Notion, Figma, Canva |
| **Social** | Instagram, Facebook, LinkedIn, Pinterest, Quora, X/Twitter |
| **Development** | GitHub, GitLab, Bitbucket, npm, PyPI, Docker Hub, Stack Overflow |
| **Cloud** | AWS, Azure, GCP, DigitalOcean, Vercel, Netlify, Cloudflare |
| **AI** | ChatGPT, Claude, Perplexity, HuggingFace, Kaggle, Colab |
| **Education** | Udemy, Coursera, edX, MIT OCW, LeetCode, GeeksForGeeks |
| **Entertainment** | Netflix, Prime Video, Hotstar, Disney+, Spotify |
| **Communication** | WhatsApp Web, Telegram Web, Discord, Slack, Teams, Zoom |
| **News/Finance** | BBC, CNN, TradingView, CoinMarketCap, Binance |
| **Shopping** | Amazon, Flipkart, BookMyShow, Zomato, Swiggy |
| **Travel** | Booking, Airbnb, Skyscanner, IRCTC |

---

### 9.5 Real-Time Web Intelligence

**File:** `actions/realtime_web.py`
**Lines of Code:** 521

Multi-source web intelligence engine that produces verified, citation-backed AI overviews.

#### Data Sources

| Source | API | Data Type |
|---|---|---|
| **DuckDuckGo** | Instant Answer API (JSON) | Abstracts, related topics |
| **Wikipedia** | REST API + OpenSearch | Article summaries |
| **Wikidata** | SPARQL entity search | Entity descriptions |
| **Google News** | RSS feed parser | Headlines with links |

#### Research Query Detection

The `looks_like_research_query()` heuristic checks for:

1. **Query length** ≥ 4 tokens (short queries stay local)
2. **Not small talk** (excludes greetings, thanks)
3. **Not system commands** (excludes "open", "play", "volume", etc.)
4. **Web markers** present: "search web", "look up", "find online"
5. **Freshness markers** + analysis terms: "latest news about", "current price of"

#### Synthesis Pipeline

```
verified_answer(query)
    │
    ├── 1. Collect snippets from all sources (parallel)
    ├── 2. Deduplicate by signature normalization
    ├── 3. Score cross-source consistency:
    │       ├── "high"     (overlap ratio > 0.22)
    │       ├── "moderate" (overlap ratio > 0.10)
    │       └── "low"      (overlap ratio ≤ 0.10)
    │
    ├── 4. LLM Synthesis (if available):
    │       ├── Build evidence string with [1], [2] citations
    │       ├── Prompt: "Create a concise AI overview..."
    │       ├── Validate response (not an LLM error message)
    │       └── Return synthesized overview with citations
    │
    ├── 5. Extractive Fallback (if LLM unavailable):
    │       ├── Extract first sentence from each source
    │       ├── Deduplicate by normalized signature
    │       └── Concatenate top 3 sentences
    │
    └── 6. Format response with:
            ├── AI overview text
            ├── Consistency rating
            └── Numbered source list with URLs
```

---

### 9.6 System Information

**File:** `actions/system_info.py`
**Lines of Code:** 137

Provides comprehensive hardware and connectivity awareness.

#### Reported Metrics

| Metric | Collection Method |
|---|---|
| **Wi-Fi SSID** | `netsh wlan show interfaces` |
| **Bluetooth Device** | `Get-PnpDevice -Class Bluetooth` |
| **Battery %** | `psutil.sensors_battery()` |
| **Power Source** | Plugged in / On battery |
| **OS Version** | `platform.system()` + `platform.release()` |
| **CPU** | `platform.processor()` |
| **Core Count** | Physical and logical cores |
| **RAM** | Total and available (GB) |

#### System Info Query Detection

Keyword-based heuristic detects queries about:
- System status, PC info, device status
- Battery, charging, plugged status
- Specs, hardware, RAM, processor, CPU
- Wi-Fi, wireless, Bluetooth connectivity info

---

### 9.7 Time & World Clock

**File:** `actions/time_control.py`
**Lines of Code:** 448

Comprehensive time and date system with 130+ location-to-timezone mappings.

#### Time Resolution Strategy

```
get_current_time(location)
    │
    ├── 1. zoneinfo.ZoneInfo (IANA database)
    ├── 2. WorldTimeAPI (online fallback)
    └── 3. Fixed UTC offset table (offline fallback)
```

#### World Clock Coverage (130+ locations)

| Region | Cities |
|---|---|
| **Americas** | New York, Chicago, Denver, LA, San Francisco, Seattle, Toronto, Vancouver, Mexico City, São Paulo, Buenos Aires |
| **Europe** | London, Paris, Berlin, Amsterdam, Madrid, Rome, Zurich, Vienna, Prague, Stockholm, Moscow, Istanbul |
| **Africa** | Cairo, Lagos, Nairobi, Johannesburg, Casablanca |
| **Middle East** | Dubai, Riyadh, Doha, Tehran |
| **South Asia** | Delhi, Mumbai, Kolkata, Chennai, Bengaluru, Karachi, Kathmandu |
| **East Asia** | Beijing, Shanghai, Tokyo, Seoul, Hong Kong, Taipei |
| **Southeast Asia** | Bangkok, Singapore, Kuala Lumpur, Jakarta, Manila |
| **Oceania** | Sydney, Melbourne, Brisbane, Auckland, Perth |

#### Time Query Detection

Regex-based detection for:
- "what time is it", "time in Tokyo", "current time"
- "what's the date", "today's date"
- "world clock"
- Location extraction via prepositional patterns ("in", "at", "for")

---

### 9.8 File Control

**File:** `actions/file_control.py`
**Lines of Code:** 60

File search and operations across user directories.

#### Actions

| Action | Description |
|---|---|
| `open` / `launch` | Open file with default application (`os.startfile`) |
| `read` / `show` | Read text content (first 1,500 characters) |
| `find` | Locate file and return path |

#### Search Directories

1. `%USERPROFILE%` (Home)
2. `%USERPROFILE%\Desktop`
3. `%USERPROFILE%\Documents`
4. `%USERPROFILE%\Downloads`

---

### 9.9 Clipboard Control

**File:** `actions/clipboard_control.py`
**Lines of Code:** 46

System clipboard interaction using `tkinter` (cross-platform, no additional dependencies).

| Action | Description |
|---|---|
| `read` | Read current clipboard content |
| `copy` / `write` | Write text to clipboard |
| `paste` | Retrieve clipboard content for pasting |

---

## 10. Persistent Memory

### 10.1 SQLite Store

**File:** `memory/sqlite_store.py`
**Class:** `SQLiteStore`
**Lines of Code:** 121

Relational database for structured conversation history, user preferences, and usage statistics.

Database path: `%APPDATA%/JARVIS/jarvis_memory.db`

#### Database Schema

**Table: `conversation_history`**

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER (PK, AUTO) | Row identifier |
| `timestamp` | REAL | Unix epoch |
| `role` | TEXT | `user` or `assistant` |
| `text` | TEXT | Message content |
| `intent` | TEXT | Classified intent (nullable) |
| `confidence` | REAL | Classification confidence (nullable) |

**Table: `user_preferences`**

| Column | Type | Description |
|---|---|---|
| `key` | TEXT (PK) | Preference identifier |
| `value` | TEXT | Stored value |
| `updated_at` | REAL | Last update timestamp |

**Table: `app_usage_stats`**

| Column | Type | Description |
|---|---|---|
| `app_name` | TEXT (PK) | Application name |
| `launch_count` | INTEGER | Total launches |
| `last_used` | REAL | Last launch timestamp |

#### Key Methods

| Method | Description |
|---|---|
| `save_turn(role, text, intent, confidence)` | Persist a conversation turn |
| `get_history(limit=50)` | Retrieve recent conversation history |
| `set_preference(key, value)` | Upsert a user preference |
| `get_preference(key, default)` | Read a user preference |
| `increment_app_usage(app_name)` | Track application launch frequency |
| `get_frequent_apps(n=5)` | Retrieve most-used applications |

---

### 10.2 Vector Store

**File:** `memory/vector_store.py`
**Class:** `VectorStore`
**Lines of Code:** 105

Embedding-based semantic memory using ChromaDB for similarity search.

#### Features

| Feature | Description |
|---|---|
| **Backend** | ChromaDB PersistentClient |
| **Collection** | `jarvis_conversations` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384D) |
| **Fallback** | SHA-256–seeded deterministic random vectors |
| **Persist Directory** | `%APPDATA%/JARVIS/chroma/` |
| **Telemetry** | Disabled (`anonymized_telemetry=False`) |

#### API

| Method | Description |
|---|---|
| `add_memory(text, metadata)` | Store text with embedding and metadata |
| `search_similar(query, n=3)` | Find top-N semantically similar memories |

---

## 11. User Interface

### 11.1 Main Window

**File:** `ui/main_window.py`
**Class:** `JarvisMainWindow(QMainWindow)`
**Lines of Code:** 1,663

The primary GUI built with PyQt6, providing a professional desktop assistant interface.

#### Window Configuration

| Property | Value |
|---|---|
| Default size | 1180 × 740 px |
| Minimum size | 920 × 620 px |
| Title | "JARVIS - Just A Really Very Intelligent System" |
| Position | Centered on primary screen |
| Close behavior | Minimize to system tray |

#### Custom Signals

| Signal | Description |
|---|---|
| `log_requested(str)` | Request to append a log entry |
| `voice_capture_failed(str)` | Voice capture encountered an error |
| `voice_capture_finished()` | Voice capture completed |
| `text_task_finished()` | Text processing task completed |
| `voice_task_finished()` | Voice processing task completed |

#### Layout Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    JARVIS Main Window                     │
│  ┌──────────┬───────────────────────┬──────────────────┐ │
│  │          │                       │                  │ │
│  │ Sidebar  │    Center Panel       │  Conversation    │ │
│  │          │                       │     Panel        │ │
│  │ • State  │  ┌─────────────────┐  │                  │ │
│  │ • Metrics│  │   Status Orb    │  │  ┌────────────┐ │ │
│  │ • Intents│  │   (animated)    │  │  │ Chat Widget│ │ │
│  │ • Model  │  └─────────────────┘  │  │            │ │ │
│  │   Select │                       │  │ [user]     │ │ │
│  │ • Intent │  ┌─────────────────┐  │  │ [JARVIS]   │ │ │
│  │   Panel  │  │ LLM Status      │  │  │ [user]     │ │ │
│  │ • Logs   │  │ Processing Hint │  │  │ [JARVIS]   │ │ │
│  │ • Config │  └─────────────────┘  │  │            │ │ │
│  │          │                       │  └────────────┘ │ │
│  │          │  ┌─────────────────┐  │                  │ │
│  │          │  │ Waveform Widget │  │                  │ │
│  │          │  └─────────────────┘  │                  │ │
│  │          │                       │                  │ │
│  │          │  ┌─────────────────┐  │                  │ │
│  │          │  │ Attachment Bar  │  │                  │ │
│  │          │  │ [thumb] [label] │  │                  │ │
│  │          │  └─────────────────┘  │                  │ │
│  │          │                       │                  │ │
│  │          │  [+] [ Input Box ] [⎙][▶/■][🎤]       │ │
│  └──────────┴───────────────────────┴──────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

#### Plus Menu Actions

| Action | Description |
|---|---|
| **Verified Web Mode** | Checkable toggle for web-verified AI overview mode |
| **Attach Image** | Open file dialog to select an image |
| **Attach Screenshot** | Capture current screen as attachment |
| **Attach Camera Snapshot** | Capture webcam frame as attachment |
| **View Last Image** | Open full-size preview of last attached image |
| **Remove Attached Image** | Clear the current attachment |

#### Input Methods

| Method | Trigger | Description |
|---|---|---|
| **Text** | Enter key or Send button | Type commands in the input box |
| **Voice (Hold)** | Hold mic button | Push-to-talk with silence detection |
| **Voice (Wake)** | Say "JARVIS" / "Hey JARVIS" | Hands-free continuous listening |
| **Image + Text** | Attach image + type prompt | Multimodal query with vision |

#### Hold-to-Talk Mechanics

1. Mouse press on mic button → pause wake word, start `sounddevice.InputStream`
2. Audio chunks captured at 16kHz, 1 channel, 1024 block size
3. Mouse release → stop stream, concatenate chunks
4. Submit audio to `pipeline.process_recorded_audio(array, 16000)`
5. Resume wake word listener if was previously active

#### Processing Hint System

A delay-triggered processing hint appears after a configurable timeout when tasks run longer than expected, showing "Processing, this may take a while..." in amber text.

#### Settings Dialog

The settings dialog (embedded in main window) provides controls for:
- TTS enable/disable toggle
- Voice profile selection (Female/Male)
- Response verbosity (Brief/Normal/Detailed)
- Compute mode selection (Auto/CPU/GPU)
- Wake word enable/disable
- Wake word sensitivity slider (0.1–0.95)
- Activation phrases editor
- Follow-up timeout spinner (3–20s)
- Max follow-up turns spinner (0–3)
- Strict phrase prefix toggle

---

### 11.2 System Tray

**File:** `ui/system_tray.py`
**Class:** `JarvisSystemTray(QSystemTrayIcon)`
**Lines of Code:** 91

#### Tray Menu

| Action | Description |
|---|---|
| **Show Window** | Restore main window from tray |
| **Toggle Wake Word** | Enable/disable background wake word listening |
| **Quit** | Force-quit the application |

Double-click: Restore window
Tooltip states: "JARVIS - Active", "JARVIS - Wake word listening", "JARVIS - Wake word starting"

---

### 11.3 Theme & Styling System

**File:** `ui/theme.py`
**Lines of Code:** 36

Defines the complete visual design language for the JARVIS interface.

#### Color Palette

| Constant | Color | RGBA | Usage |
|---|---|---|---|
| `CYAN` | Bright cyan | `(0, 245, 255)` | Active elements, borders |
| `CYAN_DIM` | Dim cyan | `(0, 184, 200)` | Idle state orb |
| `CYAN_GHOST` | Ghost cyan | `(0, 245, 255, 18)` | Subtle highlights |
| `ORANGE` | Processing orange | `(255, 107, 43)` | Processing state |
| `GREEN` | Speaking green | `(0, 255, 136)` | Speaking state |
| `YELLOW` | Warning yellow | `(255, 215, 0)` | Warnings |
| `RED` | Error red | `(255, 51, 85)` | Errors |
| `BG_MAIN` | Near-black blue | `(3, 10, 15)` | Main background |
| `BG_PANEL` | Dark panel | `(6, 15, 24, 220)` | Panel backgrounds |
| `BORDER` | Cyan border | `(0, 245, 255, 46)` | Border lines |
| `TEXT` | Light text | `(200, 234, 240)` | Primary text |
| `TEXT_DIM` | Dim text | `(90, 138, 150)` | Secondary text |

#### Typography

| Constant | Font Family | Size | Weight |
|---|---|---|---|
| `FONT_DISPLAY` | Orbitron | 10pt | Bold |
| `FONT_MONO` | Share Tech Mono | 10pt | Normal |
| `FONT_BODY` | Rajdhani | 11pt | Normal |

Custom fonts are loaded from `assets/fonts/` via `QFontDatabase.addApplicationFont()`.

#### Qt Stylesheet (`assets/style.qss`)

```css
QWidget {
    color: #c8eaf0;                 /* Light cyan text */
    background: transparent;
}

QLineEdit {
    background: rgba(6, 15, 24, 220);
    border: 1px solid rgba(0, 245, 255, 46);
    border-radius: 8px;
    padding: 8px;
    selection-background-color: rgba(0, 245, 255, 60);
}

QPushButton {
    background: rgba(0, 184, 200, 30);
    border: 1px solid rgba(0, 245, 255, 60);
    border-radius: 8px;
    padding: 8px 12px;
}

QPushButton:hover {
    background: rgba(0, 245, 255, 40);
}

QComboBox {
    background: rgba(6, 15, 24, 220);
    border: 1px solid rgba(0, 245, 255, 46);
    border-radius: 6px;
    padding: 6px;
}

QScrollArea { border: none; }
```

---

### 11.4 Animation Utilities

**File:** `ui/animations.py`
**Lines of Code:** 28

Provides reusable animation factories for UI transitions.

| Function | Description | Properties |
|---|---|---|
| `create_fade_animation(widget, duration=320)` | Opacity 0→1 fade-in | `windowOpacity`, OutCubic easing |
| `create_slide_animation(widget, start, end, duration=320)` | Position slide | `pos`, OutCubic easing |

---

### 11.5 OrbWidget — Animated Status Orb

**File:** `ui/widgets/orb_widget.py`
**Class:** `OrbWidget(QWidget)`
**Lines of Code:** 193

A GPU-rendered animated status indicator with particle effects and state-reactive visuals.

#### Minimum Size: 300×300 pixels

#### State Colors

| State | Color | Effect |
|---|---|---|
| `IDLE` | `CYAN_DIM` (0, 184, 200) | Gentle pulsing glow + rotating dashed orbit ring |
| `LISTENING` | `CYAN` (0, 245, 255) | Expanding ripple rings + audio level bars (10 bars) |
| `PROCESSING` | `ORANGE` (255, 107, 43) | Dual counter-rotating conic gradient arcs |
| `SPEAKING` | `GREEN` (0, 255, 136) | Modulated wave boundary (56-point path with 8-harmonic modulation) |

#### Animation System

- **Frame rate:** 33ms timer (~30 FPS, coarse timer type)
- **Rotation:** 52°/s continuous ring rotation
- **Phase:** 4.2 rad/s oscillation for effects
- **Particles:** 6 orbiting particles (radius 68-120px, speed 0.6-1.5 rad/s, size 2-5px)
- **Scale animation:** QPropertyAnimation (250ms) for state transitions (1.0 → 1.1 on LISTENING)

#### Rendering Layers (paint order)

1. **Glow** — QRadialGradient halo (2× radius, alpha 100→0)
2. **Core orb** — QRadialGradient fill (alpha 180→60) with pulse modulation
3. **Orbit ring** — Rotating dashed ellipse (1.3× radius)
4. **Particles** — 6 orbiting dots
5. **State-specific overlay** — Ripples, arcs, or wave path

---

### 11.6 ChatWidget — Conversation Display

**File:** `ui/widgets/chat_widget.py`
**Classes:** `ChatBubble(QWidget)`, `ChatWidget(QScrollArea)`
**Lines of Code:** 187

#### ChatBubble

Each message is rendered as a custom-painted rounded rectangle:

| Property | User Message | JARVIS Message |
|---|---|---|
| Fill color | `rgba(6, 48, 58, 210)` | `rgba(6, 15, 24, 220)` |
| Border color | `rgba(0, 245, 255, 50)` | `rgba(0, 245, 255, 100)` |
| Border radius | 12px | 12px |
| Role label | "USER" | "JARVIS" |
| Alignment | Right-aligned | Left-aligned |

Features:
- Word-wrap enabled with `TextSelectableByMouse`
- Optional intent label below message text
- Fade-in animation (opacity 0→1, 280ms) on creation
- Dynamic width sizing (max 480px, 84% of viewport)

#### ChatWidget (Container)

- Auto-scroll to bottom with sticky behavior
- Max 50 messages (oldest pruned via `deleteLater()`)
- Multi-pass scroll scheduling: delays at 0, 40, 100, 200, 320ms
- Viewport-responsive bubble width recalculation on resize

---

### 11.7 Sidebar — Diagnostics Panel

**File:** `ui/widgets/sidebar.py`
**Class:** `Sidebar(QWidget)`
**Lines of Code:** 210

Width: 220-360px

#### Components (top to bottom)

1. **Logo** — "JARVIS" in Orbitron bold, centered
2. **Status Label** — Current state ("IDLE", "LISTENING", etc.)
3. **MetricsStatusBar** — Real-time CPU/GPU/RAM
4. **Intent Identifier Toggle** — Collapsible panel:
   - **Recent Intents** — Last 5 classified intents with progress bars
   - **Top Candidates** — Top 3 candidate intents with confidence %
   - **Runtime Info** — Model runtime, provider, latency
   - **Supported Intents Catalog** — All 14 intents with descriptions
   - **Intent Model Selector** — ComboBox: LinearSVC, BiLSTM, DistilBERT
5. **Settings Button** — Opens settings dialog
6. **Logs Button** — Opens log viewer

#### Intent Info Catalog (Exact from Code)

| Intent | Description | Example |
|---|---|---|
| `launch_app` | Open or start desktop apps | "can you please launch chrome" |
| `close_app` | Close or quit running apps | "please close spotify for me" |
| `web_search` | Search the web for information | "look up latest ai news" |
| `open_website` | Open URLs and websites | "open github.com" |
| `play_media` | Play, pause, and control media | "play lofi on youtube" |
| `system_volume` | Change or query system volume | "do me a favor and set volume to 40" |
| `system_brightness` | Change or query screen brightness | "set brightness to 65 please" |
| `power_control` | Power and device toggles | "turn off monitor" |
| `system_settings` | Open Windows settings pages | "open bluetooth settings" |
| `general_qa` | General questions and conversation | "explain transformers in nlp" |
| `vision_query` | Analyze camera, screenshots, images | "what do you see on screen" |
| `file_control` | Find/open local files | "find my resume pdf" |
| `clipboard_action` | Read/copy/paste clipboard | "read clipboard" |
| `stop_cancel` | Stop current assistant action | "cancel that" |

---

### 11.8 WaveformWidget — Audio Visualizer

**File:** `ui/widgets/waveform_widget.py`
**Class:** `WaveformWidget(QWidget)`
**Lines of Code:** 133

Real-time audio waveform visualization with three rendering modes.

#### Configuration

| Parameter | Value |
|---|---|
| Minimum height | 80px |
| Bar count | 40 |
| Frame rate | 33ms (~30 FPS) |
| Sample rate | 16,000 Hz |
| Ring buffer | 2,048 samples |
| Smoothing factor | 0.78 (exponential) |

#### Rendering Modes

| State | Data Source | Visual Style |
|---|---|---|
| **IDLE** | Sine wave formula | Low-amplitude breathing bars (every 2nd frame) |
| **LISTENING** | Live FFT from microphone | Real-time frequency spectrum (every 3rd frame) |
| **SPEAKING** | TTS level + sine modulation | Animated sine wave bars |

#### FFT Processing

```python
_fft_targets()
    ├── Extract last 1024 samples from ring buffer
    ├── np.fft.rfft() → complex spectrum
    ├── np.abs() → magnitude spectrum
    ├── Split into 40 equal bins
    ├── Mean per bin → normalized values
    └── Peak-normalize to [0, 1]
```

#### Rendering

- Horizontal gradient: cyan alpha 60→220→60
- Round-cap vertical bars centered vertically
- Bar width: max(2, viewport_width / (bars × 1.8))
- Max bar height: 45% of widget height

---

### 11.9 MetricsStatusBar — System Metrics

**File:** `ui/widgets/status_bar.py`
**Class:** `MetricsStatusBar(QWidget)`
**Lines of Code:** 74

Real-time hardware utilization display with asynchronous GPU probing.

#### Displayed Metrics

| Metric | Source | Update Interval |
|---|---|---|
| CPU % | `psutil.cpu_percent(interval=None)` | 800ms |
| GPU % | `GPUtil.getGPUs()[0].load × 100` | 2000ms (background thread) |
| RAM % | `psutil.virtual_memory().percent` | 800ms |

GPU probing runs in a dedicated daemon thread (`_gpu_probe_loop`) with a `threading.Event`-based 2-second sleep to avoid UI blocking from slow GPU queries.

---

## 12. Model Setup & Asset Management

**File:** `setup_models.py`
**Lines of Code:** 791

Automated model download and configuration utility.

#### Managed Assets

| Asset | Source | Size (approx.) | Purpose |
|---|---|---|---|
| GloVe 6B 100d | Stanford NLP | ~130 MB | Word embeddings for NLP |
| spaCy `en_core_web_sm` | spaCy Hub | ~12 MB | NER and tokenization |
| Qwen2.5-VL GGUF | HuggingFace | ~2–4 GB | Vision-language model |
| Qwen2.5 Instruct GGUF | HuggingFace | ~1–3 GB | Text-only model |
| mmproj GGUF | HuggingFace | ~600 MB | Vision projector |
| DistilBERT ONNX | Custom train | ~250 MB | Intent classifier |

#### Usage

```bash
# Download all models
python setup_models.py

# Configure llama-cpp for GPU
python setup_models.py --configure-llama-gpu

# Verify model installation
python setup_models.py --verify
```

---

## 13. Build & Packaging

**File:** `build.py`
**Lines of Code:** 37

Uses **Nuitka** to compile the entire Python project into a standalone Windows executable.

#### Build Command

```bash
python build.py
```

#### Nuitka Configuration

| Parameter | Value |
|---|---|
| Mode | `--standalone` |
| Output | `JARVIS.exe` |
| Icon | `assets/icon.ico` |
| Plugin | `pyqt6` |
| Data includes | `ml/`, `data/`, `assets/` |
| Module includes | All project packages |
| Console | Disabled (`--disable-console`) |

#### Output Structure

```
dist/
└── JARVIS.exe             # Standalone executable
    ├── ml/models/          # Bundled ML models
    ├── data/               # Runtime data
    └── assets/             # Static assets
```

---

## 14. Application Entry Point

**File:** `main.py`
**Lines of Code:** 103

#### Startup Sequence

```
main.py → _run()
    │
    ├── 1. Initialize SessionTerminalLogger (dual stream capture)
    ├── 2. Trace "startup_begin" event
    ├── 3. multiprocessing.freeze_support() (for Nuitka builds)
    ├── 4. Set environment safeguards:
    │   ├── KMP_DUPLICATE_LIB_OK=TRUE  (OpenMP clash prevention)
    │   ├── OMP_NUM_THREADS=1
    │   ├── OPENBLAS_NUM_THREADS=1
    │   └── MKL_NUM_THREADS=1
    ├── 5. If JARVIS_DEV=1: install debug import hooks
    │   ├── Log first import of "torch" / "torch.*"
    │   └── Log first import of "sentence_transformers"
    ├── 6. Create QApplication
    ├── 7. Set app name "JARVIS", org "JARVIS"
    ├── 8. Enable High DPI pixmaps (if supported)
    ├── 9. Load QSS stylesheet from assets/style.qss
    ├── 10. Create JarvisMainWindow
    ├── 11. Show window
    ├── 12. Start pipeline (window.start_pipeline())
    ├── 13. Enter Qt event loop (app.exec())
    │
    └── Finally:
        ├── Trace "shutdown_begin"
        ├── Stop session logger
        └── Print session and trace log paths
```

---

## 15. Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `JARVIS_DEV` | `0` | Enable debug mode (verbose logging, import tracing) |
| `JARVIS_COMPUTE_MODE` | `auto` | Global compute mode: `auto`, `cpu`, `gpu` |
| `JARVIS_INTENT_MODEL` | `DistilBERT` | Intent classifier model: `DistilBERT`, `LinearSVC`, `BiLSTM` |
| `JARVIS_ENABLE_WAKEWORD` | `0` | Enable wake word backend |
| `JARVIS_ENABLE_VECTOR_STORE` | `0` | Enable ChromaDB vector store |
| `JARVIS_ENABLE_SPACY` | `0` | Enable spaCy NER in entity extractor |
| `JARVIS_USE_SPACY_TRF` | `0` | Use transformer-based spaCy model |
| `JARVIS_ENABLE_ONNX_INTENT` | `1` | Enable ONNX Runtime for intent classification |
| `JARVIS_DISABLE_ONNX_INTENT` | `0` | Force-disable ONNX intent runtime |
| `JARVIS_ALLOW_ONLINE_MODEL_DOWNLOAD` | `0` | Allow HuggingFace model downloads |
| `JARVIS_LLM_MODEL_PATH` | (auto) | Explicit LLM model file path |
| `JARVIS_LLM_MMPROJ_PATH` | (auto) | Explicit mmproj file path |
| `JARVIS_LLM_THREADS` | `4` | LLM inference thread count |
| `JARVIS_LLM_MAX_TOKENS_CPU` | `160` | Max tokens for CPU-bound generation |
| `JARVIS_LLM_CONTEXT_TURNS` | `2` | Conversation context turns sent to LLM |
| `JARVIS_LLM_SUBPROCESS` | `1` | Force subprocess worker mode |
| `JARVIS_LLM_AUTO_MODEL_SWITCH` | `1` | Auto-switch between VL and text models |
| `JARVIS_LLM_WORKER_READY_TIMEOUT_S` | `25` | Worker startup timeout |
| `JARVIS_LLM_WORKER_RESPONSE_TIMEOUT_S` | `45` | Worker response timeout |
| `JARVIS_LLM_VISION_TIMEOUT_S` | `240` | Vision inference timeout |
| `JARVIS_AUTO_COMPUTE_SWITCH_COOLDOWN` | `15` | CPU/GPU switch cooldown (seconds) |
| `JARVIS_GPU_RETRY_COOLDOWN` | `60` | GPU retry cooldown after failure |
| `JARVIS_ENABLE_LLM_GPU` | `0` | Enable GPU for LLM worker |
| `JARVIS_STT_USE_CUDA` | `0` | Use CUDA for STT |
| `JARVIS_STT_FORCE_CUDA` | `0` | Force CUDA regardless of detection |
| `JARVIS_STT_ALLOW_UNVERIFIED_CUDA` | `0` | Skip DLL verification for CUDA |
| `JARVIS_INTENT_PROVIDER` | `auto` | Intent runtime provider override |
| `JARVIS_TTS_BACKEND` | `auto` | TTS backend selection |
| `JARVIS_TTS_DEVICE` | `cpu` | Kokoro TTS device |
| `JARVIS_TTS_REPO` | `hexgrad/Kokoro-82M` | Kokoro model repository |
| `JARVIS_TTS_ALLOW_KOKORO_DOWNLOAD` | `0` | Allow Kokoro model download |
| `JARVIS_TTS_NO_PLAYBACK` | `0` | Skip TTS audio playback |
| `JARVIS_VISION_MAX_EDGE` | `512` | Max image edge for vision preprocessing (range: 384-1024) |
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | Prevent OpenMP library clash crashes |
| `OMP_NUM_THREADS` | `1` | OpenMP thread limit |
| `OPENBLAS_NUM_THREADS` | `1` | OpenBLAS thread limit |
| `MKL_NUM_THREADS` | `1` | MKL thread limit |

---

## 16. Performance Targets & Benchmarks

#### Defined Targets (PerformanceTargets dataclass)

| Metric | Target |
|---|---|
| Wake word detection | < 500 ms |
| STT transcription (5s audio) | < 800 ms |
| Intent classification | < 8 ms |
| Entity extraction | < 5 ms |
| App launch | < 300 ms |
| System control | < 50 ms |
| LLM first-token latency | < 1,000 ms |
| TTS synthesis start | < 200 ms |

#### Observed Benchmarks

| Metric | Target | Typical (CPU) | Typical (GPU) |
|---|---|---|---|
| Wake word detection | < 500 ms | ~80 ms | — |
| STT transcription (5s audio) | < 800 ms | ~1,200 ms | ~600 ms |
| Intent classification (ONNX) | < 8 ms | ~25 ms | ~10 ms |
| Entity extraction | < 5 ms | ~15 ms | — |
| Router (deterministic action) | < 50 ms | ~10 ms | — |
| LLM first-token latency | < 1,000 ms | ~800 ms | ~400 ms |
| LLM full response (96 tokens) | < 8,000 ms | ~6,000 ms | ~2,000 ms |
| TTS synthesis (Edge) | < 200 ms | ~500 ms | — |
| Vision analysis (screen) | < 15,000 ms | ~12,000 ms | ~5,000 ms |
| App launch | < 300 ms | ~100 ms | — |
| System control (volume/brightness) | < 50 ms | ~30 ms | — |

---

## 17. Dependency Stack

### Core Dependencies

| Package | Version | Purpose |
|---|---|---|
| PyQt6 | ≥ 6.6 | GUI framework |
| llama-cpp-python | ≥ 0.2.0 | GGUF model inference |
| faster-whisper | ≥ 1.0 | Speech-to-text |
| edge-tts | ≥ 6.1 | Neural TTS |
| spacy | ≥ 3.7 | NLP pipeline |
| onnxruntime | ≥ 1.16 | ONNX model inference |
| transformers | ≥ 4.36 | Tokenizers (DistilBERT) |
| numpy | ≥ 1.24 | Numerical operations |
| scikit-learn | — | LinearSVC, LabelEncoder |
| joblib | — | Model (de)serialization |
| sounddevice | ≥ 0.4 | Audio I/O |
| soundfile | ≥ 0.12 | Audio file I/O |
| psutil | ≥ 5.9 | Process/system monitoring |
| pyautogui | ≥ 0.9 | Screen capture |
| pygame | ≥ 2.5 | Media playback |
| pycaw | ≥ 20230407 | Windows audio control |
| screen-brightness-control | ≥ 0.22 | Display brightness |
| pygetwindow | ≥ 0.0.9 | Window management |
| chromadb | ≥ 0.4 | Vector database |
| sentence-transformers | ≥ 2.2 | Text embeddings |
| Pillow | ≥ 10.0 | Image processing |
| opencv-python | ≥ 4.8 | Webcam capture |
| torch | ≥ 2.1 | CNN model inference |
| torchvision | ≥ 0.16 | Image transforms |
| GPUtil | — | GPU utilization monitoring |

### Optional Dependencies

| Package | Purpose |
|---|---|
| kokoro | Offline neural TTS (Kokoro-82M) |
| openwakeword | Neural wake word detection |
| ctranslate2 | CUDA acceleration for STT |
| nuitka | Production build compiler |
| spacy[transformers] | Transformer-based NER |

---

## 18. Static Assets

### Font Assets (`assets/fonts/`)

| File | Family | Usage |
|---|---|---|
| `Orbitron-Bold.ttf` | Orbitron | Display headings, logo text |
| `ShareTechMono-Regular.ttf` | Share Tech Mono | Status labels, code-like text |

### Sound Assets (`assets/sounds/`)

| File | Usage |
|---|---|
| `wake_chime.wav` | Played when wake word is detected |
| `error.wav` | Played on error conditions |

### Other Assets

| File | Usage |
|---|---|
| `icon.ico` | Application window and tray icon |
| `style.qss` | Global Qt stylesheet (dark theme) |

---

## 19. Data Flow Diagrams

### Voice Input Flow

```
Microphone → sounddevice.InputStream
    │
    ├── [Wake Word Path]
    │   WakeWordDetector._listen_loop()
    │       → openWakeWord.predict() / STT.listen_once()
    │       → transcript_ready signal
    │       → Pipeline._handle_wake_transcript()
    │       → _wake_voice_session_worker(initial_text)
    │           ├── process_text(text, capture_tts_future=True)
    │           ├── Wait for TTS completion
    │           ├── Listen for follow-up (configurable turns)
    │           └── Resume wake listener
    │
    └── [Hold-to-Talk Path]
        JarvisMainWindow mouse press on mic button
            → Record audio chunks (1024 blocksize, 16kHz)
            → Mouse release → stop stream
            → pipeline.process_recorded_audio(array, 16000)
                → SpeechToText.transcribe(audio_array)
                → JarvisPipeline.process_text(transcript)
```

### Text Input Flow

```
QLineEdit.returnPressed
    │
    ▼
JarvisMainWindow._submit_text()
    │
    ├── Text → JarvisPipeline.process_text(text)
    │            │
    │            ├── preprocessor.clean(text)
    │            ├── normalize_command_text(cleaned)
    │            ├── choose_device_for_query(cleaned, compute_mode)
    │            ├── IntentClassifier.predict(cleaned)
    │            ├── EntityExtractor.extract_entities(cleaned, intent)
    │            ├── Emit intent_classified + intent_diagnostics signals
    │            ├── ContextManager.add_turn("user", text)
    │            ├── _persist_turn_async(role="user", ...)
    │            ├── Router.route(intent, entities, text, context, hint)
    │            │     ├── [Fast-Path] → time, sysinfo, media cmds
    │            │     ├── [Small-Talk] → predefined responses
    │            │     ├── [Deterministic] → Action module function
    │            │     ├── [Verified Web] → realtime_web.verified_answer()
    │            │     └── [Generative] → QwenBridge.generate(text, context)
    │            ├── enforce_response_verbosity(response, mode)
    │            ├── ContextManager.add_turn("assistant", response)
    │            ├── _persist_turn_async(role="assistant", ...)
    │            ├── Emit new_message signal
    │            └── TextToSpeech.speak_async(response)
    │
    └── Image attached?
         └── pipeline.analyze_image_file(image_path, text)
              └── Router.analyze_image_file(path, prompt)
                   ├── CNN classify (fallback)
                   ├── Image preprocessing (PIL → PNG bytes)
                   └── QwenBridge.generate(prompt, [], image_bytes=...)
```

### Memory Persistence Flow

```
JarvisPipeline._persist_turn_async()
    → ThreadPoolExecutor.submit(_persist_turn_to_stores)
    │
    ├── SQLiteStore.save_turn(role, text, intent, confidence)
    │       → INSERT INTO conversation_history
    │
    ├── VectorStore.add_memory(text, {"role": role, "intent": intent})
    │       → sentence-transformers.encode()
    │       → ChromaDB.collection.add()
    │
    └── ContextManager.embed_recent_missing(limit=4)
            → Find turns without embeddings
            → all-MiniLM-L6-v2 encoding
            → Store embeddings on ConversationTurn objects
```

---

## 20. Security & Privacy Considerations

| Aspect | Implementation |
|---|---|
| **Data Locality** | All processing occurs on-device. No external API calls for core functionality. |
| **Network Access** | Limited to: Edge TTS (Microsoft), web searches (user-initiated), model downloads (first-run only, opt-in via `JARVIS_ALLOW_ONLINE_MODEL_DOWNLOAD`) |
| **Conversation Storage** | Local SQLite database. No cloud sync. |
| **Vector Embeddings** | Local ChromaDB with telemetry disabled (`anonymized_telemetry=False`). Opt-in only (`JARVIS_ENABLE_VECTOR_STORE=1`). |
| **System Access** | Volume, brightness, power require explicit user commands. Destructive actions (shutdown, restart, hibernate) require double-confirmation. |
| **Wake Word Audio** | Audio is processed in real-time memory and discarded after transcription. No persistent audio recording. |
| **Screen Captures** | Saved locally in `data/captures/`. No automatic upload. |
| **Webcam Access** | Only captured on explicit user request. Camera is released immediately after single-frame capture. |
| **Environment Variables** | Sensitive paths and configurations are stored locally, not embedded in code. |
| **OpenMP Isolation** | Thread counts for OMP, OpenBLAS, and MKL are limited to 1 to prevent library clashes. |
| **Process Isolation** | LLM runs in isolated subprocess to prevent GPU driver faults from crashing the UI. |
| **spaCy Model Downloads** | Controlled by `JARVIS_ENABLE_SPACY` and `JARVIS_USE_SPACY_TRF` flags; disabled by default. |

---

## 21. Glossary

| Term | Definition |
|---|---|
| **GGUF** | GPT-Generated Unified Format — quantized model file format for llama.cpp |
| **mmproj** | Multimodal projector — maps visual embeddings to the LLM's text embedding space |
| **ONNX** | Open Neural Network Exchange — portable ML model format |
| **CTranslate2** | Optimized inference engine for Transformer models (used by faster-whisper) |
| **ChromaDB** | Open-source embedding database for semantic search |
| **SAPI** | Speech Application Programming Interface — Windows built-in TTS |
| **pycaw** | Python Core Audio Windows — library for Windows audio endpoint control |
| **Nuitka** | Python-to-C compiler for creating standalone executables |
| **openWakeWord** | Open-source wake word detection library using ONNX models |
| **QwenBridge** | JARVIS's abstraction layer over the local Qwen LLM |
| **Router** | The decision engine that maps classified intents to action modules |
| **Pipeline** | The central orchestrator (`JarvisPipeline`) that coordinates all subsystems |
| **Deterministic Action** | A pre-programmed Python script executed for known intents (no LLM involved) |
| **Generative Action** | A query routed to the local LLM for open-ended reasoning |
| **Fast-Path** | Ultra-low-latency route in the router that bypasses intent classification for obvious commands |
| **Verbosity Enforcement** | Hard sentence/word/character limits applied to LLM responses before output |
| **Session Telemetry** | Structured JSONL trace logging with exception hooks and performance instrumentation |
| **_TeeTextStream** | Custom I/O stream that mirrors all writes to both console and log file simultaneously |
| **Compute Hint** | Per-request device recommendation (cpu/gpu) based on query complexity analysis |

---

*This document was generated for the JARVIS project. All system capabilities described are fully implemented and production-ready.*
]]>

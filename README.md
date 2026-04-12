# JARVIS

Just A Really Very Intelligent System.

JARVIS is a Windows-native desktop assistant built with PyQt6, deterministic action routing, local intent models, and local Qwen-based LLM/Vision support.

## Highlights

- Conversational intent understanding for natural phrasing (for example, "can you please launch chrome").
- Hybrid execution model: deterministic system/app/web actions first, LLM for general reasoning.
- Local multimodal support with Qwen2.5-VL (with robust fallback behavior when runtime fails).
- Wakeword controls and advanced wakeword behavior directly in Settings (no raw JSON editor required).
- Runtime status telemetry and latency logging for debugging and optimization.

## Requirements

- Windows 10/11
- Python 3.12 recommended
- NVIDIA GPU recommended (4GB+ VRAM), but CPU fallback is supported

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Download/setup model assets:

```powershell
python setup_models.py
```

4. Run the app:

```powershell
python main.py
```

## Optional GPU Runtime Setup (llama-cpp)

If you want GPU-accelerated local Qwen runtime, install a CUDA-enabled `llama-cpp-python` wheel that matches your CUDA stack.

Example:

```powershell
python -m pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python==0.3.20 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

## Vision Runtime Notes

JARVIS uses Qwen2.5-VL with conservative image preprocessing for stability on mid-range GPUs.

- `JARVIS_VISION_MAX_EDGE` (default `512`): controls max image edge before multimodal inference.
- `JARVIS_LLM_VISION_TIMEOUT_S` (default `240`): vision worker timeout.
- `JARVIS_LLM_SUBPROCESS` (default `1`): keeps LLM in worker process for better UI stability.

If multimodal runtime fails, JARVIS reports the error and may use classifier fallback for continuity.

## Project Structure

- `main.py`: app entrypoint
- `core/`: orchestration, runtime settings, telemetry
- `nlp/`: preprocessing, intent classification, entity extraction, routing
- `llm/`: Qwen bridge + worker runtime
- `speech/`: STT, TTS, wakeword behavior
- `vision/`: screen/webcam helpers and CNN fallback
- `actions/`: deterministic action modules
- `ui/`: main window, sidebar, widgets, tray
- `ml/`: dataset/training/evaluation utilities

## Build EXE

```powershell
python build.py
```

Output: `dist/JARVIS.exe`

## GitHub-Ready Packaging

This repository now includes `.gitignore` configured to exclude:

- virtual environments
- logs and trace outputs
- runtime captures
- downloaded/generated model binaries
- temporary smoke scripts
- reference-only folder content

Recommended upload set:

- source directories (`actions`, `core`, `llm`, `nlp`, `speech`, `ui`, `vision`, `memory`, `ml` scripts)
- `main.py`, `setup_models.py`, `build.py`, `requirements.txt`, `README.md`, `.gitignore`
- lightweight data files needed for bootstrap (for example, intent CSV files)

## Troubleshooting

- If the app starts but LLM is unavailable, check model paths under `models/` and run `python setup_models.py`.
- If GPU is detected but not used, verify CUDA/runtime DLL compatibility with installed `llama-cpp-python` wheel.
- If wakeword is not triggering, confirm wakeword is enabled in Settings and phrases/sensitivity are configured.

## Runtime Logs

Performance logs are written to `%APPDATA%/JARVIS/performance.log`.


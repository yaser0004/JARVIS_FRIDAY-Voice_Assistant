 # Project Guidelines

## Build And Run
- Use Python 3.12 when possible for best compatibility.
- Prefer environment-safe commands:
  - `python -m pip install --upgrade pip`
  - `python -m pip install --upgrade torch==2.3.0+cu118 torchvision==0.18.0+cu118 --index-url https://download.pytorch.org/whl/cu118`
  - `python -m pip install -r requirements.txt`
- One-time model bootstrap:
  - `python setup_models.py`
- Run app:
  - `python main.py`
- Packaging:
  - `python build.py`

## Model Training And Evaluation
- Dataset refresh:
  - `python ml/dataset.py`
- Train intent models:
  - `python ml/train_ml.py`
  - `python ml/train_bilstm.py`
  - `python ml/train_distilbert.py`
- Optional vision fallback model:
  - `python ml/train_cnn_vision.py`
  - The script auto-converts CIFAR-10 raw batches from `data/vision_dataset/cifar-10-batches-py` when needed.
- Compare metrics and latency:
  - `python ml/evaluate.py`

## Testing Strategy
- There is no dedicated first-party automated test suite in this repo.
- For behavior changes, run focused smoke checks with small Python scripts against pipeline entry points:
  - `core/pipeline.py` `JarvisPipeline.initialize()`
  - `core/pipeline.py` `JarvisPipeline.process_text()`
- Validate import safety after larger edits, especially in optional-runtime modules.

## Architecture Boundaries
- Keep orchestration in `core/pipeline.py`; do not bypass it from UI flows.
- Keep intent logic in `nlp/intent_classifier.py` and routing in `nlp/router.py`.
- Keep side-effect actions inside `actions/` modules.
- Keep UI state updates signal-driven via PyQt in `ui/`.

## Project Conventions
- Favor graceful degradation over hard crashes:
  - Optional components (LLM, wake word, vision, STT/TTS backends) should fail with clear fallback behavior.
- Avoid blocking the UI thread; use worker threads for long-running operations.
- Keep runtime toggles environment-driven (for example, intent model, ONNX usage, LLM worker mode).
- Treat `%APPDATA%/JARVIS/performance.log` as the latency/debug signal source for runtime performance checks.

## Runtime Pitfalls
- Windows may show ONNX CUDA provider warnings; CPU fallback is expected when CUDA runtime dependencies are missing.
- Wake-word detection is intentionally opt-in for stability.
- In-process llama-cpp can be unstable in some UI runtime contexts; keep subprocess worker fallback behavior intact.
- Python 3.13 has package/toolchain constraints for several optional features; prefer Python 3.12 for local LLM and full multimodal support.

## Documentation
- For full setup details and project context, see [README.md](../README.md).
- Keep this file minimal and workspace-wide; link to docs instead of duplicating long guidance.

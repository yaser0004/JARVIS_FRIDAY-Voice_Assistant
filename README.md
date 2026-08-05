# JARVIS

Just A Really Very Intelligent System.

JARVIS is a desktop voice assistant built with PyQt6, deterministic action
routing, local intent models, and a local Gemma 4 E2B LLM/Vision backend
(via Google's LiteRT-LM runtime). It runs on **Windows and Linux**: all
OS-specific behavior lives behind a single platform adapter layer
(`platform_adapters/`).

## Highlights

- Conversational intent understanding for natural phrasing (for example, "can you please launch chrome").
- Hybrid execution model: deterministic system/app/web actions first, LLM for general reasoning.
- Local multimodal support with Gemma 4 E2B (text + vision in one model, with a CNN classifier fallback when the runtime fails).
- Wakeword controls and advanced wakeword behavior directly in Settings (no raw JSON editor required).
- Runtime status telemetry and latency logging for debugging and optimization.

## Requirements

- Windows 10/11, or a Linux desktop (developed against GNOME/X11)
- Python 3.12 recommended
- NVIDIA GPU recommended (4GB+ VRAM), but CPU fallback is supported

## Quick Start

1. Create and activate a virtual environment.
2. Install runtime dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

3. Set up model assets and train the intent classifier:

```bash
python setup_models.py
```

This downloads GloVe/spaCy/Gemma assets **and** builds the intent dataset plus
the LinearSVC classifier. Without that classifier the assistant falls back to
keyword-only routing, so it is not optional. To do only that step (no large
downloads):

```bash
python setup_models.py --only-intent-models
```

4. Run the app:

```bash
python main.py
```

## Linux setup

Core functionality works out of the box. Individual capabilities shell out to
standard desktop tools; when one is missing, JARVIS reports which package to
install rather than failing silently.

| Capability | Tool | Package (Debian/Ubuntu/Pop!_OS) | Without it |
|---|---|---|---|
| Audio capture/playback | PortAudio | `libportaudio2` | **Required** — speech input/output unavailable |
| Volume control | `pactl` | `pulseaudio-utils` | Volume commands fail with a hint |
| Brightness | `brightnessctl` | `brightnessctl` | Brightness commands fail with a hint |
| Power / lock | `systemctl`, `loginctl` | systemd (present by default) | Power commands fail |
| Wi-Fi / Bluetooth | `nmcli`, `bluetoothctl` | `network-manager`, `bluez` | Status reads as "Not connected"; toggles open Settings |
| Open files / URLs | `xdg-open` | `xdg-utils` | File/URL opening fails |
| App switching | `wmctrl` | `wmctrl` | "Switch to X" reports unsupported (X11 only; not available on Wayland) |
| Settings pages | `gnome-control-center` | `gnome-control-center` | Settings shortcuts fail |
| Offline TTS fallback | `spd-say` / `espeak-ng` | `speech-dispatcher` / `espeak-ng` | No speech if Edge TTS and Kokoro both fail |
| Screenshot on Wayland | `gnome-screenshot` | `gnome-screenshot` | Screen capture fails under Wayland |

```bash
sudo apt install libportaudio2 pulseaudio-utils brightnessctl wmctrl xdg-utils \
                 network-manager bluez speech-dispatcher gnome-screenshot
```

## Optional GPU Runtime Setup (LiteRT-LM)

`litert-lm-api` ships prebuilt wheels with both CPU and GPU backends — no
compiling from source required. GPU acceleration runs on **Vulkan**
(Windows/Linux) or **Metal** (macOS), not CUDA directly, so it works across
AMD/NVIDIA/Intel GPUs as long as a Vulkan driver is present. On Windows the
GPU path additionally needs `dxil.dll` and `dxcompiler.dll`
(DirectXShaderCompiler) next to the Python executable; without them JARVIS
falls back to CPU automatically. If GPU generation fails silently, check
Settings for the compute-mode/backend status and confirm your GPU driver
supports Vulkan.

## Vision Runtime Notes

JARVIS uses Gemma 4 E2B (via LiteRT-LM) for both text and vision — one model
handles both, so there's no separate vision model to install. A lightweight
CNN classifier (`vision/cnn_classifier.py`) provides a low-confidence fallback
if the LLM vision path is unavailable.

- `JARVIS_VISION_MAX_EDGE` (default `512`): controls max image edge before multimodal inference.
- `JARVIS_LLM_VISION_TIMEOUT_S` (default `240`): vision worker timeout.
- `JARVIS_LLM_SUBPROCESS` (default `1`): keeps LLM in worker process for better UI stability.

If multimodal runtime fails, JARVIS reports the error and falls back to the CNN classifier for continuity.

## Privacy note

The default TTS backend (`auto`) tries Edge TTS first, which sends the response
text to Microsoft's servers. For fully local speech synthesis set
`JARVIS_TTS_BACKEND=kokoro` (or `sapi` on Windows).

## Project Structure

- `main.py`: app entrypoint
- `core/`: orchestration, runtime settings, telemetry
- `nlp/`: preprocessing, intent classification, entity extraction, routing
- `platform_adapters/`: OS backends (`windows.py`, `linux.py`) selected once at import
- `llm/`: Gemma 4 E2B bridge (LiteRT-LM) + worker runtime
- `speech/`: STT, TTS, wakeword behavior
- `vision/`: screen/webcam helpers and CNN fallback
- `actions/`: deterministic action modules (platform-neutral; they call the adapter)
- `ui/`: main window, sidebar, widgets, tray
- `ml/`: dataset/training/evaluation utilities
- `tests/`: pytest suite (no model artifacts or audio hardware required)

## Tests

```bash
python -m pytest tests/ -q
```

## Models and benchmarks

Training and evaluation tooling is a separate dependency set:

```bash
python -m pip install -r requirements-train.txt
```

The dataset splitter holds out whole base sentences **before** augmentation, so
`data/intents_test.csv` contains no variant of any training sentence:

```bash
python -m ml.dataset      # writes data/intents_train.csv + data/intents_test.csv
python -m ml.train_ml     # LinearSVC + LogReg
python -m ml.evaluate     # scores the held-out test set
```

Current held-out results (`ml/results/comparison_report.json`):

| Model | Accuracy | Macro-F1 | Median latency | Size |
|---|---|---|---|---|
| LinearSVC | 0.831 | 0.821 | 0.37 ms | 0.42 MB |
| BiLSTM | not retrained | — | — | — |
| DistilBERT | not retrained | — | — | — |

Earlier versions of this repo reported ~100% accuracy for all three models.
That was a measurement bug (the evaluation set overlapped the training data);
see `RESOURCES/audit-report.md`. DistilBERT and BiLSTM need to be retrained
against the corrected split before their numbers mean anything.

## Build

```bash
python build.py
```

Output: `dist/JARVIS.exe` on Windows, `dist/jarvis` on Linux.

## Troubleshooting

- If the app starts but LLM is unavailable, check model paths under `models/` and run `python setup_models.py`.
- If every command is answered conversationally instead of acted on, the intent classifier is missing: run `python setup_models.py --only-intent-models`.
- If GPU is detected but not used, confirm a Vulkan driver is installed (Windows/Linux) and, on Windows, that `dxil.dll`/`dxcompiler.dll` are available.
- If wakeword is not triggering, confirm wakeword is enabled in Settings and phrases/sensitivity are configured.
- On Linux, if a system command reports a missing tool, install the package named in the error (see the Linux setup table).

## Runtime Logs

Performance logs are written to `%APPDATA%/JARVIS/performance.log` on Windows
and `$XDG_DATA_HOME/JARVIS/performance.log` (default
`~/.local/share/JARVIS/performance.log`) on Linux.

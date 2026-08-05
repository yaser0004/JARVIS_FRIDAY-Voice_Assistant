# JARVIS/FRIDAY — Engineering Audit and Linux Port

Audit date: 2026-07-26. Scope: the entire repository (~19k lines of Python).
Every finding below was verified by reading the code; the critical ones were
reproduced. Findings marked **FIXED** were addressed in this pass; the rest are
listed as remaining technical debt at the end.

Verification environment: Pop!_OS (GNOME, X11), Python 3.12.3, local venv with
runtime dependencies. Windows-specific code paths could not be executed here —
see "Windows non-regression" for how they were protected instead.

---

## Severity summary

| # | Finding | Severity | Effort | Status |
|---|---------|----------|--------|--------|
| 1 | `actions/system_control.py` imported `comtypes`/`pycaw`/`screen_brightness_control` at module scope — the app could not import at all on Linux | Critical | M | **FIXED** |
| 2 | Command injection: media titles from speech passed through `cmd /c start` | Critical | S | **FIXED** |
| 3 | Benchmarks were train-on-test: `evaluate.py` re-split the training file (79% of "test" rows were training rows) | Critical | M | **FIXED** |
| 4 | Augmentation happened before the split, leaking trivial variants across it (~79% of test cores present in train) | Critical | M | **FIXED** |
| 5 | Fresh clone had no intent model and no step to build one — every utterance became `general_qa` | Critical | S | **FIXED** |
| 6 | `ui/main_window.py` `mouseReleaseEvent` read an uninitialized `_drag_pos` → `AttributeError` on any mouse release | Critical | S | **FIXED** |
| 7 | `close_app` matched a ≥4-char token anywhere in name+exe+**cmdline** → "close code" could kill unrelated processes | High | S | **FIXED** |
| 8 | `set_compute_mode` ran a 45s-timeout subprocess probe synchronously on the UI thread | High | S | **FIXED** |
| 9 | `realtime_web` chained up to 6 HTTP calls × 5s serially (~30s worst case) on the turn thread | High | M | **FIXED** |
| 10 | Intent confidence threshold applied only on the ONNX path; LinearSVC labels used unthresholded | High | S | **FIXED** |
| 11 | Normalization order destroyed the apostrophes/case that the normalizer and entity extractor depend on | High | S | **FIXED** |
| 12 | Keyword collisions routed questions into device actions ("sleep", "sound", "document") | High | M | **FIXED** |
| 13 | TensorFlow + 5 other training-only packages installed as runtime deps and bundled into the exe | High | S | **FIXED** |
| 14 | `main.py` opened `assets/style.qss` by relative path with no guard — fatal on a different CWD | High | S | **FIXED** |
| 15 | No offline TTS fallback on Linux (SAPI path is Windows-only) | Medium | S | **FIXED** |
| 16 | `pyautogui` screenshots fail on Wayland with no fallback | Medium | S | **PARTIAL** |
| 17 | Power-confirmation state never expired — a stale "shutdown" could be confirmed minutes later | Medium | S | **FIXED** |
| 18 | Raw wake-word transcripts written to the trace log while sibling events logged only lengths | Medium | S | **FIXED** |
| 19 | Fake-success actions: clipboard "paste" never pasted; airplane/battery-saver returned success without acting | Medium | S | **PARTIAL** |
| 20 | `vision/vision_model.py` entirely dead; dead signal, dead `predict_batch`, committed dead 3.5 MB `bilstm.keras` | Medium | S | **FIXED** |
| 21 | `_response()` duplicated in 11 modules; urlopen wrapper duplicated 4× | Medium | S | **FIXED** |
| 22 | BiLSTM offered in the UI but silently falls back to LinearSVC | Medium | S | **FIXED** |
| 23 | `config.py` wrote to `~/JARVIS` on Linux instead of an XDG directory | Medium | S | **FIXED** |
| 24 | Fuzzy filename match from speech could open arbitrary executables under `$HOME` | Medium | S | **FIXED** |
| 25 | Unreachable duplicate `weather_query` branch in the router dispatch | Low | S | **FIXED** |
| 26 | `log_performance` opened and closed the log file on every metric | Low | S | **FIXED** |
| 27 | `evaluate.py` double-counted an unused int8 export in reported model size; latency included cold-start | Low | S | **FIXED** |
| 28 | Write-only memory subsystem: nothing ever reads the SQLite or vector stores | Medium | L | **DEFERRED** |
| 29 | ~100 broad `except Exception` blocks; three parallel logging systems, no stdlib `logging` | Medium | L | **DEFERRED** |
| 30 | Documentation contradicted the code (offline claims, <8 ms target, model sizes) | Medium | S | **FIXED** |

---

## Critical findings in detail

### 1. Linux import-time crash (FIXED)
`actions/system_control.py` imported `ctypes`, `comtypes`, `pycaw`, and
`screen_brightness_control` at module scope. The router imports the action
modules, and the pipeline imports the router, so the entire application failed
to import on Linux before any code ran.

Fix: all OS-touching code moved behind `platform_adapters/` (see the port
section). `python -c "import core.pipeline"` now succeeds on Linux.

### 2. Command injection from speech (FIXED)
`actions/media_control.py` built `subprocess.Popen(["cmd", "/c", "start", "", uri])`
where the URI embedded a spoken media title escaped with `quote_plus`.
`quote_plus` is a URL encoder, not a shell escaper, and `cmd /c` interprets
`&`, `|`, and `^`. A spoken title could therefore break out of the intended
argument.

Fix: both call sites use `webbrowser.open`, which hands the URI to the OS
handler with no shell hop (and is portable). Regression test:
`tests/test_action_security.py::test_spotify_search_does_not_shell_out`.

### 3 & 4. The benchmarks measured memorization, not generalization (FIXED)
Two independent leaks produced the reported 100% accuracies:

- `ml/evaluate.py` built its "test set" with a fresh `train_test_split` over
  `intents_augmented.csv`, unrelated to the `GroupShuffleSplit` the models were
  actually trained on. Reproduced: **664/840 = 79% of evaluation rows were in
  the LinearSVC training set.**
- `ml/dataset.py` augmented before splitting, and the split grouped by exact
  string only. Trivial polite/typo variants of one sentence therefore landed on
  both sides. Reproduced: **436/553 = 78.8% of test "core" sentences also
  appeared in train.**

The whole dataset derives from ~350 hand-written base sentences inflated to
4200 rows, so a char-ngram TF-IDF + LinearSVC memorizes it trivially. 100% was
inevitable and meant nothing.

Fix: `ml/dataset.py` now holds out whole **base sentences** per intent *before*
any augmentation, writing `data/intents_train.csv` (train bases + their
variants, with a `base_text` column) and `data/intents_test.csv` (raw held-out
bases, unaugmented). `split_utils` groups by `base_text`. `evaluate.py` loads
the held-out file directly and skips models whose artifacts are absent instead
of crashing. Train rows that coincidentally reproduce a held-out sentence are
dropped.

**Honest result on truly held-out data: LinearSVC 83.1% accuracy / 0.821 macro-F1**
(previously reported 1.0000). Median inference 0.37 ms, 0.42 MB.
DistilBERT and BiLSTM are reported as `skipped` until retrained — that needs
TensorFlow/torch and was not run here.

### 5. A fresh clone could not classify anything (FIXED)
Every trained artifact is gitignored, and `setup_models.py` never trained one.
With no `linearsvc.pkl` and no ONNX export, `IntentClassifier._fallback_logits`
returned `general_qa` with **confidence 1.0** for every input — a fabricated
certainty that also suppressed the router's keyword recovery.

Fix: `setup_models.py` gained `train_intent_models()` (also available as
`--only-intent-models`), which builds the dataset and trains LinearSVC when the
artifact is missing — seconds, using dependencies already required at runtime.
Separately, the no-model fallback now returns a flat distribution, so the
prediction is honestly low-confidence and the router ladder engages.

---

## Linux port: architecture

Windows logic was **moved, not rewritten**, so Windows behavior is unchanged.

```
platform_adapters/
  __init__.py   # selects the backend once, by sys.platform
  windows.py    # moved verbatim from system_control/app_control/system_info
  linux.py      # new
```

Action modules keep everything platform-neutral (parsing, fuzzy matching,
confirmation flow, response formatting) and call `backend.<function>`. There is
no abstract base class: two implementations selected once at import need only a
shared function surface.

| Capability | Windows | Linux |
|---|---|---|
| Volume get/set/mute | pycaw COM | `pactl` on `@DEFAULT_SINK@` |
| Brightness | `screen_brightness_control` | `brightnessctl -m` |
| Power / lock / monitor | `shutdown.exe`, `rundll32`, `ctypes.windll` | `systemctl poweroff\|reboot\|suspend\|hibernate`, `loginctl lock-session`, `xset dpms force off` |
| Settings pages | `ms-settings:` URIs | `gnome-control-center <panel>` |
| Wi-Fi / Bluetooth toggle | PowerShell `Get-NetAdapter`, `bthserv` | `nmcli radio wifi`, `bluetoothctl power` |
| Wi-Fi / Bluetooth name | `netsh wlan`, `Get-PnpDevice` | `nmcli -t -f active,ssid`, `bluetoothctl devices Connected` |
| Open file / URI | `os.startfile` | `xdg-open` |
| App index | `.exe` map + Start-Menu `.lnk` scan | XDG `.desktop` scan (skips `NoDisplay`) |
| App launch | `os.startfile` → `Popen` | `gtk-launch` → `Exec=` with field codes stripped |
| Window list / activate | `pygetwindow` | `wmctrl -l` / `wmctrl -i -a` (X11 only) |

Graceful degradation: a missing Linux tool raises `RuntimeError` naming the
package (or returns a failure tuple), which the action layer already converts
into a spoken error. Nothing crashes; the user is told what to install.

Also ported: an offline Linux TTS fallback (`spd-say`, then `espeak-ng`),
a `gnome-screenshot` fallback for screen capture, XDG config directory
resolution, and `sys_platform == "win32"` markers so `pip install -r
requirements.txt` now succeeds on Linux.

### Windows non-regression
Windows paths cannot execute on this machine, so they are protected by:
- **Move-only discipline** — `platform_adapters/windows.py` contains the
  original function bodies; verify with `git diff --color-moved=dimmed-zebra`.
- Action-module diffs contain only import and delegation changes.
- `requirements.txt` keeps identical Windows pins, now behind platform markers.
- `core/config.py` still resolves `%APPDATA%` first, so the Windows data
  directory is byte-identical.

---

## Verification performed on Linux

```
$ python -c "import core.pipeline"          # was ImportError, now clean
$ python main.py                             # boots, UI up, pipeline ready
$ python -m ml.dataset && python -m ml.train_ml && python -m ml.evaluate
$ python -m pytest tests/ -q                 # 68 passed
```

Live pipeline turns (real `JarvisPipeline`, offscreen Qt):

| Utterance | Result |
|---|---|
| "what time is it" | time handler → correct local time |
| "can you please tell me what's the time right now" | time handler (was falling through to the LLM) |
| "how much sleep do I need" | stays a question (was `power_control`) |
| "turn the volume down" | `system_volume` with direction=down |
| "put the pc to sleep" | `power_control` → suspend |

Backend smoke: `pactl` volume read/write, `nmcli` SSID, `.desktop` app scan.
`_collect_sources` returned 4 sources in 0.54s and 3 in 1.16s where the serial
version chained 4–6 round trips.

Two normalizer bugs were found *by* the new tests and fixed: `i'd like you to`
was never stripped (the pattern required whitespace before the contraction),
and trailing fillers survived a trailing `?` ("open spotify for me?").

---

## Latency changes

| Change | Effect |
|---|---|
| `_collect_sources` fan-out parallelized with an 8s overall deadline | ~30s worst case → ~1–8s |
| LLM prewarm moved off the path to `ready` | UI accepts input immediately (0.18s to ready) |
| Compute-capability probe moved off the UI thread | no multi-second GUI freeze from Settings |
| One long-lived asyncio loop for Edge TTS | no loop construction per utterance |
| `log_performance` keeps one append handle | no open/close per metric on a hot path |

---

## Remaining technical debt

Deliberately not addressed in this pass, with reasons:

1. **Write-only memory subsystem.** Only `save_turn` and `add_memory` are ever
   called; the vector store is off by default and nothing reads either store.
   `ContextManager.resolve_reference` has no caller, so per-turn embeddings are
   computed and discarded. Wiring retrieval is a feature decision, not a fix.
   Also note `sentence-transformers` is disabled on Python ≥3.13, where
   embeddings silently degrade to hash-seeded random vectors.
2. **Logging unification.** Three systems coexist (trace JSONL,
   `performance.log`, `print` teed by the session logger) and stdlib `logging`
   is unused. ~100 broad `except Exception` blocks still swallow root causes.
   Both are wide mechanical changes best done as their own pass.
3. **Wayland.** Screen capture falls back to `gnome-screenshot`, not the
   xdg-desktop-portal API. Window switching needs `wmctrl` (X11); on Wayland it
   returns an explanatory failure.
4. **Clipboard paste.** Still cannot inject keystrokes; the response text is now
   honest about it instead of claiming a paste. Real paste needs `xdotool`/
   `SendInput`.
5. **Airplane mode / battery saver.** Both still only open a settings page while
   returning `success=True`. Neither OS exposes a stable unprivileged toggle;
   the response text says a manual switch is needed, but the success flag is
   still misleading to programmatic callers.
6. **DistilBERT / BiLSTM retraining.** The training scripts now read the honest
   split, but retraining needs TensorFlow/torch and a GPU. Until then their rows
   in `comparison_report.json` read `skipped`. BiLSTM has no inference runtime at
   all and is no longer offered in the UI.
7. **Router structure.** Still a long keyword ladder rather than a data-driven
   intent table. The three worst collisions are fixed and pinned by tests; the
   rest of the ladder is untouched.
8. **QA suites.** `qa/full_system_tester.py` and `qa/ui_torture_test.py` are
   genuine tests with real assertions, but remain Windows-coupled at runtime
   (pycaw, brightness control). They were not ported.
9. **Remaining duplication.** `_max_tokens_for_request`/`_build_messages` are
   duplicated verbatim between `llm/qwen_bridge.py` and `llm/qwen_worker.py`;
   wake-phrase helpers, the CUDA DLL list, the verbosity normalizer, and the
   small-talk set each exist in 2–3 places. Only the duplication on the port
   surface was consolidated (`actions/common.py`).
10. **Edge TTS privacy.** The default `auto` backend tries Edge TTS first, which
    sends response text to Microsoft. Behavior was left as-is and documented;
    set `JARVIS_TTS_BACKEND=kokoro` for local-only synthesis.
11. **`realtime_web` disabled by default.** Verified-web answering stays off
    unless enabled in Settings. Turning it on by default is a behavior change
    that was out of scope.

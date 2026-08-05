import argparse
import os
import importlib.util
import re
import socket
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
GLOVE_DIR = DATA_DIR / "glove"
MODELS_DIR = ROOT / "models"
GLOVE_ZIP = DATA_DIR / "glove.6B.zip"
GLOVE_FILE_NAME = "glove.6B.100d.txt"
GEMMA_FILE_NAME = "gemma-4-E2B-it.litertlm"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_LOG_EVERY_BYTES = 25 * 1024 * 1024
SPACY_DOWNLOAD_TIMEOUT_SECONDS = 60 * 60
SPINNER_UPDATE_SECONDS = 1.0
GLOVE_URLS = (
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
)
GEMMA_URLS = (
    f"https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/{GEMMA_FILE_NAME}?download=true",
    f"https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/{GEMMA_FILE_NAME}",
)


def _log_step(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()


def _download_label(label: str, attempt: int, max_attempts: int, source_url: str) -> str:
    host = source_url.split("/")[2] if "//" in source_url else source_url
    return f"{label} {attempt}/{max_attempts} ({host})"


def _root_cause(exc: Exception | None) -> Exception | None:
    current = exc
    while current is not None and getattr(current, "__cause__", None) is not None:
        current = current.__cause__
    return current


def _run_checked(command: list[str], env: dict[str, str] | None = None, dry_run: bool = False) -> bool:
    rendered = " ".join(command)
    _log_step(f"Running: {rendered}")
    if dry_run:
        return True

    try:
        subprocess.run(command, check=True, env=env)
        return True
    except subprocess.CalledProcessError as exc:
        _log_step(f"Command failed (exit {exc.returncode}): {rendered}")
        return False



@contextmanager
def _indeterminate_progress(label: str):
    if tqdm is None:
        _log_step(f"{label}...")
        yield
        return

    stop_event = Event()
    progress = tqdm(
        total=None,
        desc=label,
        unit="s",
        dynamic_ncols=True,
        mininterval=0.2,
        leave=True,
    )

    def _ticker() -> None:
        while not stop_event.wait(SPINNER_UPDATE_SECONDS):
            progress.update(SPINNER_UPDATE_SECONDS)

    worker = Thread(target=_ticker, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop_event.set()
        worker.join(timeout=2)
        progress.close()


def _get_response_total_bytes(response, existing_bytes: int) -> int | None:
    content_range = response.headers.get("Content-Range", "")
    if content_range:
        match = re.search(r"/(\d+)$", content_range)
        if match:
            return int(match.group(1))

    content_length = response.headers.get("Content-Length", "")
    if content_length.isdigit():
        length = int(content_length)
        status = getattr(response, "status", None)
        if status == 206:
            return existing_bytes + length
        return length

    return None


def _download_with_progress(url: str, partial_path: Path, label: str, attempt: int, max_attempts: int) -> None:
    existing_bytes = partial_path.stat().st_size if partial_path.exists() else 0
    headers = {"User-Agent": "JARVIS-setup/1.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    request = Request(url, headers=headers)
    with urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        status = getattr(response, "status", None)
        if existing_bytes > 0 and status == 200:
            # Some servers ignore range requests; restart cleanly if that happens.
            _log_step("Resume not supported by server, restarting current file download.")
            _safe_unlink(partial_path)
            existing_bytes = 0

        total_bytes = _get_response_total_bytes(response, existing_bytes)
        desc = _download_label(label, attempt, max_attempts, url)
        progress = None
        if tqdm is not None:
            progress = tqdm(
                total=total_bytes,
                initial=existing_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                leave=True,
                dynamic_ncols=True,
                miniters=1,
                mininterval=0.2,
            )

        mode = "ab" if existing_bytes > 0 else "wb"
        bytes_written = existing_bytes
        next_log_at = existing_bytes + DOWNLOAD_LOG_EVERY_BYTES
        try:
            with partial_path.open(mode) as out_file:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break

                    out_file.write(chunk)
                    bytes_written += len(chunk)

                    if progress is not None:
                        progress.update(len(chunk))
                    elif bytes_written >= next_log_at:
                        _log_step(f"{desc}: downloaded {bytes_written / (1024 * 1024):.1f} MB")
                        next_log_at += DOWNLOAD_LOG_EVERY_BYTES
        finally:
            if progress is not None:
                progress.close()

        if total_bytes is not None and partial_path.stat().st_size < total_bytes:
            raise RuntimeError(
                f"Incomplete download: expected {total_bytes} bytes, got {partial_path.stat().st_size} bytes"
            )


def _download_from_mirrors(urls: list[str], destination: Path, label: str, attempts_per_url: int = 3) -> None:
    partial_path = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None

    for url in urls:
        for attempt in range(1, attempts_per_url + 1):
            _log_step(f"{label}: starting download attempt {attempt}/{attempts_per_url} from {url}")
            try:
                _download_with_progress(url, partial_path, label, attempt, attempts_per_url)
                partial_path.replace(destination)
                return
            except Exception as exc:
                last_error = exc
                _log_step(f"{label}: download attempt failed: {exc}")
                if attempt < attempts_per_url:
                    time.sleep(min(2 * attempt, 6))

        _log_step(f"{label}: mirror failed: {url}")

    raise RuntimeError(f"{label} failed across all mirrors") from last_error


def _is_valid_glove_file(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False

    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                if len(parts) == 101:
                    return True
    except Exception:
        return False

    return False


def _extract_glove_zip(target_file: Path) -> None:
    _log_step(f"Extracting {GLOVE_FILE_NAME}...")
    with zipfile.ZipFile(GLOVE_ZIP, "r") as zf:
        if GLOVE_FILE_NAME not in zf.namelist():
            raise RuntimeError(f"{GLOVE_FILE_NAME} not found in archive")
        zf.extract(GLOVE_FILE_NAME, path=GLOVE_DIR)

    if not _is_valid_glove_file(target_file):
        raise RuntimeError("Extracted GloVe file is empty or invalid")


def _download_glove_zip() -> None:
    custom_url = os.getenv("JARVIS_GLOVE_URL", "").strip()
    urls = [custom_url] if custom_url else []
    urls.extend(GLOVE_URLS)

    last_error: Exception | None = None
    try:
        _download_from_mirrors(urls, GLOVE_ZIP, "GloVe zip", attempts_per_url=3)
        return
    except Exception as exc:
        last_error = exc

    hint = ""
    root = _root_cause(last_error)
    if isinstance(root, URLError) and isinstance(getattr(root, "reason", None), socket.gaierror):
        hint = " DNS lookup failed; check network/VPN/DNS settings."

    mirror_list = ", ".join(urls)
    raise RuntimeError(
        "Unable to download GloVe from all mirrors."
        f"{hint} You can download manually from: {mirror_list}"
        f" and save it as {GLOVE_ZIP}."
    ) from last_error


def download_glove() -> None:
    GLOVE_DIR.mkdir(parents=True, exist_ok=True)
    target_file = GLOVE_DIR / GLOVE_FILE_NAME
    if _is_valid_glove_file(target_file):
        _log_step("GloVe 100d already present.")
        return

    if target_file.exists():
        _log_step("Existing GloVe file is empty or invalid; re-downloading.")
        _safe_unlink(target_file)

    if GLOVE_ZIP.exists() and GLOVE_ZIP.stat().st_size > 0:
        _log_step("Found existing GloVe zip; trying local extraction first...")
        try:
            _extract_glove_zip(target_file)
            _safe_unlink(GLOVE_ZIP)
            _log_step("GloVe ready.")
            return
        except Exception as exc:
            _log_step(f"Existing GloVe zip is invalid; re-downloading: {exc}")
            _safe_unlink(GLOVE_ZIP)

    try:
        _download_glove_zip()
        _extract_glove_zip(target_file)
        _safe_unlink(GLOVE_ZIP)
        _log_step("GloVe ready.")
    except Exception as exc:
        _safe_unlink(GLOVE_ZIP)
        _log_step(f"Skipping GloVe setup: {exc}")
        _log_step("BiLSTM training can continue with randomly initialized embeddings.")


def download_spacy_model() -> None:
    preferred_models = ["en_core_web_trf", "en_core_web_sm"]
    if os.sys.version_info >= (3, 13):
        preferred_models = ["en_core_web_sm", "en_core_web_trf"]

    for model_name in preferred_models:
        if _module_available(model_name):
            _log_step(f"spaCy model {model_name} already present.")
            return

    for model_name in preferred_models:
        _log_step(f"Downloading spaCy model {model_name}...")
        try:
            with _indeterminate_progress(f"spaCy {model_name}"):
                subprocess.run(
                    [os.sys.executable, "-m", "spacy", "download", model_name],
                    check=True,
                    timeout=SPACY_DOWNLOAD_TIMEOUT_SECONDS,
                )
            _log_step(f"spaCy model {model_name} ready.")
            return
        except subprocess.TimeoutExpired:
            _log_step(f"spaCy model {model_name} download timed out after {SPACY_DOWNLOAD_TIMEOUT_SECONDS}s")
        except Exception as exc:
            _log_step(f"Failed to download spaCy model {model_name}: {exc}")

    _log_step("Skipping spaCy model download: no compatible model could be installed.")


def download_gemma_litertlm() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / GEMMA_FILE_NAME
    if target.exists() and target.stat().st_size > 0:
        _log_step("Gemma 4 E2B .litertlm already present.")
        return

    if target.exists() and target.stat().st_size == 0:
        _log_step("Existing Gemma 4 E2B .litertlm is empty; re-downloading.")
        _safe_unlink(target)

    custom_url = os.getenv("JARVIS_GEMMA_URL", "").strip()
    urls = [custom_url] if custom_url else []
    urls.extend(GEMMA_URLS)

    _log_step("Downloading Gemma 4 E2B (.litertlm, ~2.6 GB)...")
    _download_from_mirrors(urls, target, "Gemma 4 E2B .litertlm", attempts_per_url=3)
    _log_step("Gemma 4 E2B download complete.")


def warm_model_caches() -> None:
    _log_step("Warming faster-whisper cache (medium.en)...")
    try:
        from faster_whisper import WhisperModel

        with _indeterminate_progress("faster-whisper medium.en"):
            WhisperModel("medium.en", device="cpu", compute_type="int8")
    except Exception as exc:
        _log_step(f"Skipping faster-whisper warmup: {exc}")

    _log_step("Warming openWakeWord cache...")
    from openwakeword.model import Model as OWWModel
    from openwakeword.utils import download_models as download_oww_models

    try:
        with _indeterminate_progress("openWakeWord ONNX warmup"):
            OWWModel(inference_framework="onnx")
    except Exception as exc:
        message = str(exc)
        if "NO_SUCHFILE" in message or "File doesn't exist" in message or "No such file" in message:
            _log_step(f"openWakeWord ONNX warmup failed, downloading models and retrying: {exc}")
            try:
                with _indeterminate_progress("openWakeWord model download"):
                    download_oww_models()
                with _indeterminate_progress("openWakeWord ONNX retry"):
                    OWWModel(inference_framework="onnx")
            except Exception as retry_exc:
                _log_step(f"Skipping openWakeWord warmup: {retry_exc}")
        else:
            _log_step(f"Skipping openWakeWord warmup: {exc}")

    _log_step("Warming Kokoro cache...")
    try:
        import kokoro

        with _indeterminate_progress("Kokoro warmup"):
            kokoro.generate("Model warmup.", voice="af_sarah", speed=1.1)
    except Exception as exc:
        _log_step(f"Skipping Kokoro warmup: {exc}")

    _log_step("Warming MiniLM cache...")
    try:
        from sentence_transformers import SentenceTransformer

        with _indeterminate_progress("MiniLM cache warmup"):
            SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as exc:
        _log_step(f"Skipping MiniLM warmup: {exc}")

    _log_step("Model cache warmup complete.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JARVIS setup utility")
    parser.add_argument(
        "--only-intent-models",
        action="store_true",
        help="Build the intent dataset and train the classifier only (no large downloads).",
    )
    return parser


def train_intent_models() -> bool:
    """Build the intent dataset and train the LinearSVC classifier if absent.

    Without these artifacts the runtime classifier has no model and every
    utterance falls back to general_qa, so a fresh clone needs this step.
    Training uses sklearn/pandas (already runtime deps) and takes seconds.
    """
    linearsvc_path = ROOT / "ml" / "models" / "linearsvc.pkl"
    if linearsvc_path.exists():
        _log_step("Intent classifier already trained; skipping.")
        return True

    train_csv = ROOT / "data" / "intents_train.csv"
    if not train_csv.exists():
        _log_step("Building intent dataset (train/test split before augmentation)...")
        if not _run_checked([sys.executable, "-m", "ml.dataset"]):
            return False

    _log_step("Training intent classifier (LinearSVC)...")
    if not _run_checked([sys.executable, "-m", "ml.train_ml"]):
        _log_step("Intent classifier training failed; JARVIS will rely on router keyword recovery.")
        return False
    return True


def main() -> int:
    args = _build_arg_parser().parse_args()

    if args.only_intent_models:
        return 0 if train_intent_models() else 2

    _log_step("Starting JARVIS model setup (estimated download 4-6 GB)...")
    train_intent_models()
    download_glove()
    download_spacy_model()
    download_gemma_litertlm()
    warm_model_caches()

    _log_step("All model setup steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


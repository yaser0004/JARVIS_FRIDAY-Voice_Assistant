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
QWEN_FILE_NAME = "Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
QWEN_MMPROJ_FILE_NAME = "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
DEFAULT_LLAMA_CPP_VERSION = "0.3.20"
DEFAULT_NUMPY_VERSION = "1.26.4"
DEFAULT_OLLAMA_VISION_MODEL = "qwen2.5vl:3b"
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60
DOWNLOAD_LOG_EVERY_BYTES = 25 * 1024 * 1024
SPACY_DOWNLOAD_TIMEOUT_SECONDS = 60 * 60
SPINNER_UPDATE_SECONDS = 1.0
GLOVE_URLS = (
    "https://nlp.stanford.edu/data/glove.6B.zip",
    "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
)
QWEN_URLS = (
    f"https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/{QWEN_FILE_NAME}?download=true",
    f"https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/{QWEN_FILE_NAME}",
)
QWEN_MMPROJ_URLS = (
    f"https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/{QWEN_MMPROJ_FILE_NAME}?download=true",
    f"https://huggingface.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/resolve/main/{QWEN_MMPROJ_FILE_NAME}",
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


def _run_capture(
    command: list[str], timeout_seconds: int = 20, env: dict[str, str] | None = None
) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
            env=env,
        )
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except Exception as exc:
        return 1, "", str(exc)


def _detect_cuda_path() -> Path | None:
    candidates: list[Path] = []

    env_cuda_path = Path(os.getenv("CUDA_PATH", "").strip()) if os.getenv("CUDA_PATH") else None
    if env_cuda_path:
        candidates.append(env_cuda_path)

    where_cmd = ["cmd", "/c", "where", "nvcc"] if os.name == "nt" else ["which", "nvcc"]
    code, stdout, _ = _run_capture(where_cmd)
    if code == 0 and stdout:
        first = stdout.splitlines()[0].strip()
        if first:
            nvcc_path = Path(first)
            if nvcc_path.exists():
                candidates.append(nvcc_path.parent.parent)

    if os.name == "nt":
        toolkit_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        known_versions = ("13.2", "13.1", "13.0", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8")
        for version in known_versions:
            candidates.append(toolkit_root / f"v{version}")

        if toolkit_root.exists():
            discovered = sorted(toolkit_root.glob("v*"), key=lambda p: p.name, reverse=True)
            candidates.extend(discovered)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() and (candidate / "bin" / "nvcc.exe").exists():
            return candidate

    return None


def _probe_tool(command: list[str], label: str) -> tuple[bool, str]:
    code, stdout, stderr = _run_capture(command)
    if code != 0:
        detail = stderr or stdout or "not available"
        return False, f"{label}: {detail}"

    detail = stdout.splitlines()[0].strip() if stdout else f"{label} available"
    return True, detail


def _resolve_nvcc_path(cuda_path: Path | None) -> Path | None:
    if cuda_path is None:
        return None

    exe_name = "nvcc.exe" if os.name == "nt" else "nvcc"
    candidate = cuda_path / "bin" / exe_name
    return candidate if candidate.exists() else None


def _detect_vcvars64_path() -> Path | None:
    if os.name != "nt":
        return None

    vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    if vswhere.exists():
        code, stdout, _ = _run_capture(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            timeout_seconds=30,
        )
        if code == 0 and stdout:
            install_path = Path(stdout.splitlines()[0].strip().strip('"'))
            vcvars = install_path / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if vcvars.exists():
                return vcvars

    fallback_roots = [
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Community"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Professional"),
        Path("C:/Program Files (x86)/Microsoft Visual Studio/2022/Enterprise"),
    ]
    for root in fallback_roots:
        vcvars = root / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
        if vcvars.exists():
            return vcvars

    return None


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


def _run_checked_with_vcvars(
    command: list[str],
    env: dict[str, str] | None = None,
    vcvars_path: Path | None = None,
    dry_run: bool = False,
) -> bool:
    if vcvars_path is None:
        return _run_checked(command, env=env, dry_run=dry_run)

    if os.name != "nt":
        return _run_checked(command, env=env, dry_run=dry_run)

    command_line = subprocess.list2cmdline(command)
    set_cmds: list[str] = []
    for key in ("CUDA_PATH", "CUDAToolkit_ROOT", "CudaToolkitDir"):
        value = "" if env is None else str(env.get(key, "") or "").strip()
        if value:
            set_cmds.append(f'set "{key}={value}"')

    parts = [f'call "{vcvars_path}" >nul']
    parts.extend(set_cmds)
    parts.append(command_line)
    vcvars_cmd = " && ".join(parts)
    rendered = f"cmd /c {vcvars_cmd}"
    _log_step(f"Running via vcvars: {rendered}")
    if dry_run:
        return True

    try:
        subprocess.run(vcvars_cmd, check=True, env=env, shell=True)
        return True
    except subprocess.CalledProcessError as exc:
        _log_step(f"Command failed (exit {exc.returncode}): {rendered}")
        return False


def _probe_llama_gpu_offload(dry_run: bool = False, env: dict[str, str] | None = None) -> tuple[bool, str]:
    if dry_run:
        return False, "dry-run (verification skipped)"

    probe_script = "\n".join(
        [
            "import llama_cpp",
            "fn = getattr(llama_cpp, 'llama_supports_gpu_offload', None)",
            "value = bool(fn() if callable(fn) else False)",
            "print(f'llama_gpu_offload_supported={value}')",
        ]
    )
    code, stdout, stderr = _run_capture([sys.executable, "-c", probe_script], timeout_seconds=60, env=env)
    combined = "\n".join([line for line in [stdout, stderr] if line]).strip()
    if code != 0:
        return False, combined or "Failed to import llama_cpp for verification"

    for line in stdout.splitlines():
        normalized = line.strip().lower()
        if normalized == "llama_gpu_offload_supported=true":
            return True, line.strip()
        if normalized == "llama_gpu_offload_supported=false":
            return False, line.strip()

    return False, combined or "Unable to parse llama GPU offload status"


def configure_llama_gpu(
    llama_version: str = DEFAULT_LLAMA_CPP_VERSION,
    numpy_version: str = DEFAULT_NUMPY_VERSION,
    dry_run: bool = False,
) -> bool:
    _log_step("Starting llama-cpp GPU setup workflow...")

    if os.name != "nt":
        _log_step("This workflow currently targets Windows only.")
        return False

    nvidia_ok, nvidia_detail = _probe_tool(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        "nvidia-smi",
    )
    nvcc_ok, nvcc_detail = _probe_tool(["nvcc", "--version"], "nvcc")
    cmake_ok, cmake_detail = _probe_tool(["cmake", "--version"], "cmake")
    msvc_ok, msvc_detail = _probe_tool(["cmd", "/c", "where", "cl"], "MSVC cl.exe")
    cuda_path = _detect_cuda_path()
    nvcc_path = _resolve_nvcc_path(cuda_path)
    vcvars_path = _detect_vcvars64_path()
    if vcvars_path is not None and not vcvars_path.exists():
        vcvars_path = None

    if not nvcc_ok and nvcc_path is not None:
        nvcc_ok, nvcc_detail = _probe_tool([str(nvcc_path), "--version"], f"nvcc ({nvcc_path})")

    _log_step(f"GPU probe: {'ok' if nvidia_ok else 'missing'} | {nvidia_detail}")
    _log_step(f"CUDA compiler probe: {'ok' if nvcc_ok else 'missing'} | {nvcc_detail}")
    _log_step(f"CMake probe: {'ok' if cmake_ok else 'missing'} | {cmake_detail}")
    _log_step(f"MSVC probe: {'ok' if msvc_ok else 'missing'} | {msvc_detail}")
    _log_step(f"Detected CUDA_PATH: {cuda_path if cuda_path is not None else 'not found'}")
    _log_step(f"Detected nvcc path: {nvcc_path if nvcc_path is not None else 'not found'}")
    _log_step(f"Detected vcvars64: {vcvars_path if vcvars_path is not None else 'not found'}")

    if not nvidia_ok:
        _log_step("Cannot continue: NVIDIA GPU/driver is not available to the shell (nvidia-smi failed).")
        return False
    if not nvcc_ok:
        _log_step("Cannot continue: nvcc was not found. Install CUDA Toolkit (11.8 or newer) and reopen terminal.")
        return False
    if not cmake_ok:
        _log_step("Cannot continue: CMake is required to build llama-cpp-python from source.")
        return False
    if cuda_path is None:
        _log_step("Cannot continue: unable to resolve CUDA_PATH from env/default locations.")
        return False

    if not msvc_ok:
        _log_step(
            "MSVC cl.exe not found on PATH. Build may still fail unless Visual C++ Build Tools are installed and visible."
        )
        if vcvars_path is None:
            _log_step("Cannot continue: vcvars64.bat was not found. Install Visual Studio C++ Build Tools.")
            return False

    build_env = os.environ.copy()
    cmake_args_raw = build_env.get("CMAKE_ARGS", "").strip()
    cmake_tokens = [token for token in cmake_args_raw.split() if token]
    cmake_tokens = [token for token in cmake_tokens if not token.startswith("-DLLAMA_CUBLAS")]
    if not any(token.startswith("-DGGML_CUDA") for token in cmake_tokens):
        cmake_tokens.append("-DGGML_CUDA=on")
    build_env["CMAKE_ARGS"] = " ".join(cmake_tokens)
    build_env["FORCE_CMAKE"] = "1"
    build_env["CUDA_PATH"] = str(cuda_path)
    build_env["CUDAToolkit_ROOT"] = str(cuda_path)
    cuda_toolkit_dir = str(cuda_path).rstrip("\\/") + "\\"
    build_env["CudaToolkitDir"] = cuda_toolkit_dir
    if nvcc_path is not None:
        cuda_bin = str(nvcc_path.parent)
        existing_path = build_env.get("PATH", "")
        if cuda_bin.lower() not in existing_path.lower():
            build_env["PATH"] = f"{cuda_bin}{os.pathsep}{existing_path}" if existing_path else cuda_bin
    cuda_runtime_bin = cuda_path / "bin" / "x64"
    if cuda_runtime_bin.exists():
        runtime_bin = str(cuda_runtime_bin)
        existing_path = build_env.get("PATH", "")
        if runtime_bin.lower() not in existing_path.lower():
            build_env["PATH"] = f"{runtime_bin}{os.pathsep}{existing_path}" if existing_path else runtime_bin

    pip_llama_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--no-cache-dir",
        "--no-binary=llama-cpp-python",
        f"llama-cpp-python=={llama_version}",
    ]
    if not _run_checked_with_vcvars(pip_llama_cmd, env=build_env, vcvars_path=vcvars_path, dry_run=dry_run):
        _log_step("llama-cpp-python rebuild failed. Verify CUDA toolkit + Build Tools installation.")
        return False

    numpy_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        f"numpy=={numpy_version}",
    ]
    if not _run_checked(numpy_cmd, dry_run=dry_run):
        _log_step("NumPy re-pin failed. The environment may be left in a partially compatible state.")
        return False

    supported, detail = _probe_llama_gpu_offload(dry_run=dry_run, env=build_env)
    if dry_run:
        _log_step("Dry run complete. No installation changes were applied.")
        return True

    _log_step(f"llama offload verification: {detail}")
    if supported:
        _log_step("GPU-enabled llama-cpp setup completed successfully.")
        return True

    _log_step("GPU build/install completed, but llama reports GPU offload unavailable.")
    _log_step("Check VS Build Tools, CUDA toolkit compatibility, and rerun this workflow.")
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


def download_qwen_gguf() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    target = MODELS_DIR / QWEN_FILE_NAME
    mmproj_target = MODELS_DIR / QWEN_MMPROJ_FILE_NAME
    if target.exists() and target.stat().st_size > 0:
        _log_step("Qwen2.5-VL GGUF already present.")
    else:
        if target.exists() and target.stat().st_size == 0:
            _log_step("Existing Qwen2.5-VL GGUF is empty; re-downloading.")
            _safe_unlink(target)

        custom_url = os.getenv("JARVIS_QWEN_URL", "").strip()
        urls = [custom_url] if custom_url else []
        urls.extend(QWEN_URLS)

        _log_step("Downloading Qwen2.5-VL-3B-Instruct GGUF...")
        _download_from_mirrors(urls, target, "Qwen2.5-VL GGUF", attempts_per_url=3)
        _log_step("Qwen2.5-VL GGUF download complete.")

    if mmproj_target.exists() and mmproj_target.stat().st_size > 0:
        _log_step("Qwen2.5-VL mmproj already present.")
        return

    if mmproj_target.exists() and mmproj_target.stat().st_size == 0:
        _log_step("Existing Qwen2.5-VL mmproj is empty; re-downloading.")
        _safe_unlink(mmproj_target)

    custom_mmproj_url = os.getenv("JARVIS_QWEN_MMPROJ_URL", "").strip()
    mmproj_urls = [custom_mmproj_url] if custom_mmproj_url else []
    mmproj_urls.extend(QWEN_MMPROJ_URLS)

    _log_step("Downloading Qwen2.5-VL mmproj...")
    _download_from_mirrors(mmproj_urls, mmproj_target, "Qwen2.5-VL mmproj", attempts_per_url=3)
    _log_step("Qwen2.5-VL mmproj download complete.")


def _ollama_server_reachable(base_url: str) -> tuple[bool, str]:
    endpoint = str(base_url or "http://127.0.0.1:11434").rstrip("/") + "/api/tags"
    request = Request(endpoint, headers={"User-Agent": "JARVIS-setup/1.0"})
    try:
        with urlopen(request, timeout=5) as response:
            _ = response.read(256)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def setup_ollama_vision_model(model_name: str = DEFAULT_OLLAMA_VISION_MODEL) -> bool:
    model = str(model_name or "").strip() or DEFAULT_OLLAMA_VISION_MODEL
    ollama_url = str(os.getenv("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434")).strip()

    _log_step(f"Preparing Ollama vision model setup for '{model}'...")
    code, stdout, stderr = _run_capture(["ollama", "--version"], timeout_seconds=20)
    if code != 0:
        detail = stderr or stdout or "command not found"
        _log_step(f"Ollama CLI is unavailable: {detail}")
        _log_step("Install Ollama from https://ollama.com/download and ensure `ollama` is on PATH.")
        return False

    server_ok, server_error = _ollama_server_reachable(ollama_url)
    if not server_ok:
        _log_step(f"Ollama server is not reachable at {ollama_url}: {server_error}")
        _log_step("Start the Ollama server with `ollama serve` and rerun setup.")
        return False

    _log_step(f"Pulling Ollama model '{model}'...")
    if not _run_checked(["ollama", "pull", model]):
        _log_step("Failed to pull the Ollama vision model.")
        return False

    _log_step("Ollama vision model is ready.")
    return True


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

    _log_step("Skipping legacy standalone vision-model warmup (single-model VL mode active).")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JARVIS setup utility")
    parser.add_argument(
        "--configure-llama-gpu",
        action="store_true",
        help="Run GPU build setup for llama-cpp-python after regular model setup.",
    )
    parser.add_argument(
        "--only-llama-gpu",
        action="store_true",
        help="Run only the llama-cpp GPU setup workflow and skip model downloads.",
    )
    parser.add_argument(
        "--llama-version",
        default=DEFAULT_LLAMA_CPP_VERSION,
        help=f"Target llama-cpp-python version (default: {DEFAULT_LLAMA_CPP_VERSION}).",
    )
    parser.add_argument(
        "--numpy-version",
        default=DEFAULT_NUMPY_VERSION,
        help=f"NumPy version to re-pin after llama install (default: {DEFAULT_NUMPY_VERSION}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print GPU setup commands without executing pip installs.",
    )
    parser.add_argument(
        "--setup-ollama-vision",
        action="store_true",
        help="Pull the configured Ollama vision model after standard setup steps.",
    )
    parser.add_argument(
        "--only-ollama-vision",
        action="store_true",
        help="Run only Ollama vision model setup and skip other downloads.",
    )
    parser.add_argument(
        "--ollama-vision-model",
        default=DEFAULT_OLLAMA_VISION_MODEL,
        help=f"Ollama vision model tag to pull (default: {DEFAULT_OLLAMA_VISION_MODEL}).",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    ollama_model = str(args.ollama_vision_model).strip() or DEFAULT_OLLAMA_VISION_MODEL

    if args.only_ollama_vision:
        ok = setup_ollama_vision_model(model_name=ollama_model)
        return 0 if ok else 2

    if args.only_llama_gpu:
        ok = configure_llama_gpu(
            llama_version=str(args.llama_version).strip() or DEFAULT_LLAMA_CPP_VERSION,
            numpy_version=str(args.numpy_version).strip() or DEFAULT_NUMPY_VERSION,
            dry_run=bool(args.dry_run),
        )
        return 0 if ok else 2

    _log_step("Starting JARVIS model setup (estimated download 8-10 GB)...")
    download_glove()
    download_spacy_model()
    download_qwen_gguf()
    warm_model_caches()

    if args.configure_llama_gpu:
        ok = configure_llama_gpu(
            llama_version=str(args.llama_version).strip() or DEFAULT_LLAMA_CPP_VERSION,
            numpy_version=str(args.numpy_version).strip() or DEFAULT_NUMPY_VERSION,
            dry_run=bool(args.dry_run),
        )
        if not ok:
            _log_step("llama GPU setup did not complete; JARVIS will continue using CPU LLM fallback.")
            return 2

    if args.setup_ollama_vision:
        ok = setup_ollama_vision_model(model_name=ollama_model)
        if not ok:
            return 2

    _log_step("All model setup steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


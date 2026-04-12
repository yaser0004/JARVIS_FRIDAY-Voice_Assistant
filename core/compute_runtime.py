from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


COMPUTE_MODE_AUTO = "auto"
COMPUTE_MODE_CPU = "cpu"
COMPUTE_MODE_GPU = "gpu"
COMPUTE_MODES = (COMPUTE_MODE_AUTO, COMPUTE_MODE_CPU, COMPUTE_MODE_GPU)

_COMPLEXITY_HINTS = (
    "explain",
    "compare",
    "difference",
    "summarize",
    "analysis",
    "latest",
    "news",
    "why",
    "how",
    "reason",
    "steps",
    "strategy",
    "multi",
    "research",
)

_WINDOWS_STT_CUDA_DLLS = (
    "cudnn_ops64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn64_9.dll",
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudart64_12.dll",
)

_CUDA_DLL_DIR_HANDLES: list[Any] = []


def windows_cuda_runtime_ready() -> tuple[bool, str]:
    if os.name != "nt":
        return True, ""

    missing: List[str] = []
    for dll_name in _WINDOWS_STT_CUDA_DLLS:
        try:
            ctypes.WinDLL(dll_name)
        except Exception:
            missing.append(dll_name)

    if missing:
        return False, f"Missing CUDA/cuDNN DLLs: {', '.join(missing)}"
    return True, ""


def ensure_windows_cuda_dll_paths() -> list[str]:
    if os.name != "nt":
        return []

    candidates: list[Path] = []
    for key in ("CUDA_PATH", "CUDAToolkit_ROOT"):
        raw = str(os.getenv(key, "") or "").strip()
        if raw:
            root = Path(raw)
            candidates.append(root / "bin")
            candidates.append(root / "bin" / "x64")

    toolkit_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    if toolkit_root.exists():
        discovered = sorted((path for path in toolkit_root.glob("v*") if path.is_dir()), key=lambda p: p.name, reverse=True)
        for root in discovered:
            candidates.append(root / "bin")
            candidates.append(root / "bin" / "x64")

    path_lower = os.environ.get("PATH", "").lower()
    added: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate)
        key = resolved.lower()
        if key in seen:
            continue
        seen.add(key)
        if not candidate.exists():
            continue

        if key not in path_lower:
            os.environ["PATH"] = f"{resolved}{os.pathsep}{os.environ.get('PATH', '')}"
            path_lower = os.environ.get("PATH", "").lower()

        try:
            handle = os.add_dll_directory(resolved)
            _CUDA_DLL_DIR_HANDLES.append(handle)
        except Exception:
            pass

        added.append(resolved)

    return added


def _probe_llama_gpu_offload_subprocess() -> tuple[bool, str]:
    ensure_windows_cuda_dll_paths()
    probe_script = "\n".join(
        [
            "from core.compute_runtime import ensure_windows_cuda_dll_paths",
            "ensure_windows_cuda_dll_paths()",
            "import llama_cpp",
            "fn = getattr(llama_cpp, 'llama_supports_gpu_offload', None)",
            "print('1' if callable(fn) and bool(fn()) else '0')",
        ]
    )

    try:
        completed = subprocess.run(
            [sys.executable, "-c", probe_script],
            check=False,
            timeout=45,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
    except Exception as exc:
        return False, f"llama probe failed: {exc}"

    stdout = str(completed.stdout or "").strip()
    stderr = str(completed.stderr or "").strip()
    if completed.returncode != 0:
        return False, f"llama probe failed: {stderr or stdout or f'exit {completed.returncode}'}"

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    result = lines[-1] if lines else ""
    if result == "1":
        return True, ""
    if result == "0":
        return False, "llama_supports_gpu_offload() returned False."
    return False, f"llama probe failed: unexpected output '{result or stdout}'"


def normalize_compute_mode(mode: str | None) -> str:
    raw = str(mode or "").strip().lower()
    if raw in COMPUTE_MODES:
        return raw
    return COMPUTE_MODE_AUTO


def estimate_query_complexity(text: str) -> Dict[str, Any]:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return {
            "score": 0.0,
            "token_count": 0,
            "keyword_hits": 0,
            "is_complex": False,
        }

    tokens = re.findall(r"[a-z0-9']+", normalized)
    token_count = len(tokens)
    keyword_hits = sum(1 for hint in _COMPLEXITY_HINTS if hint in normalized)
    punctuation_hits = sum(normalized.count(ch) for ch in ("?", ",", ";", ":", "(", ")"))
    conjunction_hits = len(
        re.findall(r"\b(and|or|while|whereas|because|although|however|including|versus|compare)\b", normalized)
    )

    score = (token_count * 0.45) + (keyword_hits * 2.6) + (punctuation_hits * 0.4) + (conjunction_hits * 1.2)
    if token_count >= 16:
        score += 2.0

    is_complex = bool(score >= 10.0 or token_count >= 22 or keyword_hits >= 2)
    return {
        "score": round(score, 2),
        "token_count": token_count,
        "keyword_hits": keyword_hits,
        "is_complex": is_complex,
    }


def choose_device_for_query(text: str, mode: str | None) -> str:
    selected = normalize_compute_mode(mode)
    if selected == COMPUTE_MODE_CPU:
        return COMPUTE_MODE_CPU
    if selected == COMPUTE_MODE_GPU:
        return COMPUTE_MODE_GPU

    complexity = estimate_query_complexity(text)
    return COMPUTE_MODE_GPU if complexity["is_complex"] else COMPUTE_MODE_CPU


def detect_compute_capabilities() -> Dict[str, Any]:
    onnx_providers: List[str] = []
    try:
        import onnxruntime as ort

        onnx_providers = list(ort.get_available_providers())
    except Exception:
        onnx_providers = []

    torch_cuda = False
    torch_device_name = ""
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
        if torch_cuda:
            torch_device_name = str(torch.cuda.get_device_name(0))
    except Exception:
        torch_cuda = False
        torch_device_name = ""

    gpu_count = 0
    gputil_device_name = ""
    try:
        import GPUtil

        gpus = GPUtil.getGPUs()
        gpu_count = len(gpus)
        if gpus:
            gputil_device_name = str(getattr(gpus[0], "name", "") or "")
    except Exception:
        gpu_count = 0
        gputil_device_name = ""

    gpu_name = torch_device_name or gputil_device_name
    gpu_supported = bool(torch_cuda or "CUDAExecutionProvider" in onnx_providers or gpu_count > 0)

    llama_gpu_offload_supported, llama_gpu_reason = _probe_llama_gpu_offload_subprocess()

    stt_cuda_ready = False
    stt_cuda_reason = ""
    try:
        import ctranslate2 as ct2

        if int(ct2.get_cuda_device_count()) < 1:
            stt_cuda_reason = "No CUDA device detected by ctranslate2."
        elif os.name == "nt":
            stt_cuda_ready, stt_cuda_reason = windows_cuda_runtime_ready()
            if not stt_cuda_ready:
                stt_cuda_reason = stt_cuda_reason or "CUDA runtime preflight failed."
            else:
                stt_cuda_ready = True
        else:
            stt_cuda_ready = True
    except Exception as exc:
        stt_cuda_reason = f"ctranslate2 CUDA probe failed: {exc}"

    return {
        "gpu_supported": gpu_supported,
        "gpu_name": gpu_name,
        "onnx_providers": onnx_providers,
        "onnx_cuda_available": "CUDAExecutionProvider" in onnx_providers,
        "torch_cuda_available": torch_cuda,
        "gpu_count": gpu_count,
        "llm_gpu_offload_supported": llama_gpu_offload_supported,
        "llm_gpu_reason": llama_gpu_reason,
        "stt_cuda_ready": stt_cuda_ready,
        "stt_cuda_reason": stt_cuda_reason,
    }


def apply_compute_environment(mode: str | None) -> str:
    selected = normalize_compute_mode(mode)
    os.environ["JARVIS_COMPUTE_MODE"] = selected

    if selected == COMPUTE_MODE_CPU:
        os.environ["JARVIS_ENABLE_LLM_GPU"] = "0"
        os.environ["JARVIS_STT_USE_CUDA"] = "0"
        os.environ["JARVIS_INTENT_PROVIDER"] = "cpu"
        return selected

    if selected == COMPUTE_MODE_GPU:
        os.environ["JARVIS_ENABLE_LLM_GPU"] = "1"
        os.environ["JARVIS_STT_USE_CUDA"] = "1"
        os.environ["JARVIS_INTENT_PROVIDER"] = "gpu"
        return selected

    os.environ["JARVIS_ENABLE_LLM_GPU"] = "1"
    os.environ["JARVIS_STT_USE_CUDA"] = "0"
    os.environ["JARVIS_INTENT_PROVIDER"] = "auto"
    return selected


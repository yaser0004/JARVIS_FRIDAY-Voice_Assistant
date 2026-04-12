from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import QWEN_GGUF_PATH, QWEN_TEXT_FALLBACK_GGUF_PATH, QWEN_VL_MMPROJ_PATH
from core.compute_runtime import ensure_windows_cuda_dll_paths


SYSTEM_PROMPT = (
    "You are JARVIS (Just A Really Very Intelligent System), an advanced AI assistant running locally on the user's Windows PC. "
    "You are intelligent, concise, and helpful. You have access to the user's system. "
    "Keep responses clear, complete, and natural. "
    "Never mention being an AI model."
)


def _emit(payload: Dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _looks_like_vl_model(model_path: Path) -> bool:
    name = model_path.name.lower()
    return "-vl-" in name or "vision" in name


def _resolve_model_path() -> Path:
    explicit_path = os.getenv("JARVIS_LLM_MODEL_PATH", "").strip()
    if explicit_path:
        return Path(explicit_path)

    preferred = Path(QWEN_GGUF_PATH)
    fallback = Path(QWEN_TEXT_FALLBACK_GGUF_PATH)
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    return preferred


def _resolve_mmproj_path() -> Path:
    explicit_path = os.getenv("JARVIS_LLM_MMPROJ_PATH", "").strip()
    if explicit_path:
        return Path(explicit_path)
    return Path(QWEN_VL_MMPROJ_PATH)


def _build_messages(
    user_message: str,
    context: List[Dict[str, str]],
    image_data_uri: Optional[str] = None,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    active_system_prompt = str(system_prompt or "").strip() or SYSTEM_PROMPT
    messages: List[Dict[str, Any]] = [{"role": "system", "content": active_system_prompt}]
    max_context_turns = max(0, int(os.getenv("JARVIS_LLM_CONTEXT_TURNS", "2")))
    turns = context[-max_context_turns:] if max_context_turns > 0 else []
    for turn in turns:
        role = str(turn.get("role", "user"))
        content = str(turn.get("text", ""))
        messages.append({"role": role, "content": content})

    if image_data_uri:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_message},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        )
    else:
        messages.append({"role": "user", "content": user_message})
    return messages


def _max_tokens_for_request(user_message: str, image_mode: bool = False) -> int:
    token_count = len(str(user_message or "").split())
    if image_mode:
        return 128
    if token_count <= 8:
        return 96
    if token_count <= 18:
        return 144
    if token_count <= 32:
        return 220
    return 320


def _init_llm():
    model_path = _resolve_model_path()
    mmproj_path = _resolve_mmproj_path()
    expects_vision = _looks_like_vl_model(model_path) or os.getenv("JARVIS_LLM_EXPECT_VISION", "0").strip() in {
        "1",
        "true",
        "yes",
    }

    if not model_path.exists():
        _emit({"ok": False, "event": "init", "error": f"Missing model file: {model_path}"})
        return None, False, ""

    try:
        ensure_windows_cuda_dll_paths()
        import llama_cpp
        from llama_cpp import Llama
    except Exception as exc:
        _emit({"ok": False, "event": "init", "error": f"llama-cpp-python import failed: {exc}"})
        return None, False, ""

    gpu_offload_supported = False
    supports_gpu_offload = getattr(llama_cpp, "llama_supports_gpu_offload", None)
    if callable(supports_gpu_offload):
        try:
            gpu_offload_supported = bool(supports_gpu_offload())
        except Exception:
            gpu_offload_supported = False

    prefer_gpu = os.getenv("JARVIS_ENABLE_LLM_GPU", "0").strip().lower() in {"1", "true", "yes"}
    if prefer_gpu and not gpu_offload_supported:
        prefer_gpu = False

    thread_count = int(os.getenv("JARVIS_LLM_THREADS", "4"))
    attempts = [{"name": "cpu", "n_gpu_layers": 0}]
    if prefer_gpu:
        attempts.insert(0, {"name": "gpu", "n_gpu_layers": -1})

    last_error = ""
    for cfg in attempts:
        llama_kwargs: Dict[str, Any] = {
            "model_path": str(model_path),
            "n_gpu_layers": cfg["n_gpu_layers"],
            "n_ctx": 4096,
            "n_threads": thread_count,
            "verbose": False,
        }
        supports_vision = False

        if expects_vision:
            if not mmproj_path.exists():
                _emit(
                    {
                        "ok": False,
                        "event": "init",
                        "error": f"Missing vision projector file: {mmproj_path}",
                    }
                )
                return None, False, ""
            try:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler

                llama_kwargs["chat_handler"] = Qwen25VLChatHandler(
                    clip_model_path=str(mmproj_path),
                    verbose=False,
                )
                supports_vision = True
            except Exception as exc:
                _emit(
                    {
                        "ok": False,
                        "event": "init",
                        "error": f"Qwen2.5-VL chat handler setup failed: {exc}",
                    }
                )
                return None, False, ""
        else:
            llama_kwargs["chat_format"] = "chatml"

        try:
            llm = Llama(**llama_kwargs)
            _emit(
                {
                    "ok": True,
                    "event": "ready",
                    "backend": cfg["name"],
                    "gpu_offload_supported": gpu_offload_supported,
                    "supports_vision": supports_vision,
                }
            )
            return llm, supports_vision, str(cfg["name"])
        except Exception as exc:
            last_error = f"{cfg['name']} init failed: {exc}"

    _emit({"ok": False, "event": "init", "error": f"Local LLM initialization failed: {last_error}"})
    return None, False, ""


def main() -> int:
    llm, supports_vision, backend_name = _init_llm()
    if llm is None:
        return 1

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except Exception as exc:
            _emit({"ok": False, "error": f"invalid request: {exc}"})
            continue

        req_type = str(request.get("type", ""))
        if req_type == "shutdown":
            _emit({"ok": True, "event": "bye"})
            return 0

        if req_type != "generate":
            _emit({"ok": False, "error": "unsupported request type"})
            continue

        user_message = str(request.get("user_message", "")).strip()
        context_raw = request.get("context", [])
        context = context_raw if isinstance(context_raw, list) else []
        system_prompt = str(request.get("system_prompt", "") or "").strip()
        requested_max_tokens = request.get("max_tokens")

        image_b64 = str(request.get("image_b64", "") or "").strip()
        if image_b64 and not supports_vision:
            _emit(
                {
                    "ok": False,
                    "error": (
                        "Vision is unavailable in the current runtime. Ensure Qwen2.5-VL GGUF and "
                        "its matching mmproj file are installed (run `python setup_models.py`)."
                    ),
                }
            )
            continue

        image_data_uri: Optional[str] = None
        if image_b64:
            # Validate payload shape early so worker failures are explicit.
            try:
                base64.b64decode(image_b64)
            except Exception as exc:
                _emit({"ok": False, "error": f"invalid image payload: {exc}"})
                continue
            image_data_uri = f"data:image/png;base64,{image_b64}"

        try:
            messages = _build_messages(
                user_message,
                context,
                image_data_uri=image_data_uri,
                system_prompt=system_prompt,
            )
            max_tokens = _max_tokens_for_request(user_message, image_mode=bool(image_b64))
            if isinstance(requested_max_tokens, (int, float)) and int(requested_max_tokens) > 0:
                max_tokens = int(requested_max_tokens)
            if backend_name != "gpu":
                cpu_cap = int(os.getenv("JARVIS_LLM_MAX_TOKENS_CPU", "72"))
                max_tokens = min(max_tokens, max(32, cpu_cap))
            response = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stream=False,
            )
            text = str(response["choices"][0]["message"]["content"])
            _emit({"ok": True, "text": text})
        except Exception as exc:
            _emit({"ok": False, "error": str(exc)})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


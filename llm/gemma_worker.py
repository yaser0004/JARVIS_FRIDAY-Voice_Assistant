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

from core.config import CONTEXT_WINDOW_SIZE, GEMMA_LITERTLM_PATH
from llm.system_prompt import DEFAULT_SYSTEM_PROMPT


def _emit(payload: Dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _resolve_model_path() -> Path:
    explicit_path = os.getenv("JARVIS_LLM_MODEL_PATH", "").strip()
    if explicit_path:
        return Path(explicit_path)
    return Path(GEMMA_LITERTLM_PATH)


def _build_context_messages(context: List[Dict[str, str]], litert_lm: Any) -> List[Any]:
    max_context_turns = max(0, int(os.getenv("JARVIS_LLM_CONTEXT_TURNS", str(CONTEXT_WINDOW_SIZE))))
    turns = context[-max_context_turns:] if max_context_turns > 0 else []
    messages = []
    for turn in turns:
        content = str(turn.get("text", "")).strip()
        if not content:
            continue
        contents = litert_lm.Contents.of(content)
        role = str(turn.get("role", "user"))
        if role == "assistant":
            messages.append(litert_lm.Message.model(contents))
        else:
            messages.append(litert_lm.Message.user(contents))
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


def _extract_text(response: Any) -> str:
    try:
        content = response.get("content") or []
        parts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("text")]
        if parts:
            return "".join(parts).strip()
    except Exception:
        pass
    return str(response or "").strip()


def _init_engine():
    import litert_lm

    litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

    model_path = _resolve_model_path()
    if not model_path.exists():
        _emit({"ok": False, "event": "init", "error": f"Missing model file: {model_path}"})
        return None, ""

    prefer_gpu = os.getenv("JARVIS_ENABLE_LLM_GPU", "0").strip().lower() in {"1", "true", "yes"}
    thread_count = int(os.getenv("JARVIS_LLM_THREADS", "4"))

    attempts = [("cpu", litert_lm.Backend.CPU(thread_count=thread_count))]
    if prefer_gpu:
        attempts.insert(0, ("gpu", litert_lm.Backend.GPU()))

    last_error = ""
    for name, backend in attempts:
        try:
            # vision_backend must be set explicitly at construction time, or
            # image messages fail with "Vision executor should not be null".
            engine = litert_lm.Engine(str(model_path), backend=backend, vision_backend=backend)
            _emit(
                {
                    "ok": True,
                    "event": "ready",
                    "backend": name,
                    "gpu_offload_supported": bool(prefer_gpu),
                    "supports_vision": True,
                }
            )
            return engine, name
        except Exception as exc:
            last_error = f"{name} init failed: {exc}"

    _emit({"ok": False, "event": "init", "error": f"Local LLM initialization failed: {last_error}"})
    return None, ""


def main() -> int:
    import litert_lm

    engine, backend_name = _init_engine()
    if engine is None:
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
        system_prompt = str(request.get("system_prompt", "") or "").strip() or DEFAULT_SYSTEM_PROMPT
        requested_max_tokens = request.get("max_tokens")

        image_b64 = str(request.get("image_b64", "") or "").strip()
        image_bytes: Optional[bytes] = None
        if image_b64:
            try:
                image_bytes = base64.b64decode(image_b64)
            except Exception as exc:
                _emit({"ok": False, "error": f"invalid image payload: {exc}"})
                continue

        max_tokens = _max_tokens_for_request(user_message, image_mode=bool(image_bytes))
        if isinstance(requested_max_tokens, (int, float)) and int(requested_max_tokens) > 0:
            max_tokens = int(requested_max_tokens)
        if backend_name != "gpu":
            cpu_cap = int(os.getenv("JARVIS_LLM_MAX_TOKENS_CPU", "72"))
            max_tokens = min(max_tokens, max(32, cpu_cap))

        conversation = None
        try:
            conversation = engine.create_conversation(
                messages=_build_context_messages(context, litert_lm),
                system_message=system_prompt,
                sampler_config=litert_lm.SamplerConfig(temperature=0.7, top_p=0.9),
                max_output_tokens=max_tokens,
            )

            if image_bytes is not None:
                message: Any = litert_lm.Contents.of(
                    user_message,
                    litert_lm.Content.ImageBytes(image_bytes),
                )
            else:
                message = user_message

            response = conversation.send_message(
                message,
                repetition_penalty_config=litert_lm.RepetitionPenaltyConfig(repetition_penalty=1.1),
                max_output_tokens=max_tokens,
            )
            text = _extract_text(response)
            _emit({"ok": True, "text": text})
        except Exception as exc:
            _emit({"ok": False, "error": str(exc)})
        finally:
            if conversation is not None:
                try:
                    conversation.close()
                except Exception:
                    pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

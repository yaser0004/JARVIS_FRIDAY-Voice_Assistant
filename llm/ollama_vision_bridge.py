from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


class OllamaVisionBridge:
    """Vision-only bridge that talks to an Ollama server over HTTP."""

    _VISION_MODEL_HINTS = (
        "vision",
        "llava",
        "moondream",
        "minicpm",
        "gemma3",
        "qwen2.5vl",
    )
    _DEFAULT_FALLBACK_PRIORITY = (
        "gemma3:4b",
        "moondream:latest",
        "minicpm-v:latest",
        "llava:latest",
    )

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        configured_url = str(base_url or os.getenv("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434")).strip()
        self.base_url = self._normalize_base_url(configured_url)
        self.model = str(model or os.getenv("JARVIS_OLLAMA_VISION_MODEL", "qwen2.5vl:3b")).strip()

        if timeout_s is None:
            timeout_s = float(os.getenv("JARVIS_OLLAMA_VISION_TIMEOUT_S", "240"))
        self.timeout_s = max(15.0, float(timeout_s))

        self.status_timeout_s = max(1.0, float(os.getenv("JARVIS_OLLAMA_STATUS_TIMEOUT_S", "2")))
        self._last_error = ""
        self.last_used_model = ""

    @staticmethod
    def _normalize_base_url(value: str) -> str:
        url = str(value or "").strip() or "http://127.0.0.1:11434"
        return url.rstrip("/") + "/"

    @staticmethod
    def _canonical_model_name(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())

    @staticmethod
    def _looks_like_vision_model(model_name: str) -> bool:
        lowered = str(model_name or "").strip().lower()
        if not lowered:
            return False
        return any(hint in lowered for hint in OllamaVisionBridge._VISION_MODEL_HINTS)

    @staticmethod
    def _is_memory_error(message: str) -> bool:
        lowered = str(message or "").lower()
        return "requires more system memory" in lowered or "insufficient memory" in lowered

    @staticmethod
    def _resolve_candidate_name(candidate: str, installed_models: List[str]) -> str:
        requested = str(candidate or "").strip().lower()
        if not requested:
            return ""

        for installed in installed_models:
            if installed.lower() == requested:
                return installed

        requested_base = requested.split(":", 1)[0]
        for installed in installed_models:
            if installed.lower().split(":", 1)[0] == requested_base:
                return installed

        return ""

    def _fallback_model_order(self, installed_models: List[str], primary_model: str) -> List[str]:
        models: List[str] = []
        seen: set[str] = set()

        def _add(candidate: str) -> None:
            name = str(candidate or "").strip()
            if not name:
                return
            key = name.lower()
            if key in seen:
                return
            seen.add(key)
            models.append(name)

        _add(primary_model)

        env_models = str(os.getenv("JARVIS_OLLAMA_VISION_FALLBACK_MODELS", "")).strip()
        if env_models:
            for item in env_models.split(","):
                _add(self._resolve_candidate_name(item, installed_models) or item)

        for preferred in self._DEFAULT_FALLBACK_PRIORITY:
            _add(self._resolve_candidate_name(preferred, installed_models) or preferred)

        for candidate in installed_models:
            if self._looks_like_vision_model(candidate):
                _add(candidate)

        return models

    def _request_json(
        self,
        method: str,
        path: str,
        payload: Dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> Dict[str, Any]:
        body = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        request = Request(
            urljoin(self.base_url, path.lstrip("/")),
            data=body,
            headers=headers,
            method=str(method or "GET").upper(),
        )

        try:
            with urlopen(request, timeout=float(timeout_s or self.timeout_s)) as response:
                raw = response.read().decode("utf-8", errors="replace").strip()
        except HTTPError as exc:
            details = ""
            try:
                details = exc.read().decode("utf-8", errors="replace").strip()
            except Exception:
                details = ""
            message = f"HTTP {exc.code} from Ollama"
            if details:
                message = f"{message}: {details}"
            raise RuntimeError(message) from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        if not raw:
            return {}

        try:
            parsed = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Invalid JSON from Ollama: {exc}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("Unexpected Ollama response shape.")
        return parsed

    def _list_models(self, timeout_s: float | None = None) -> List[str]:
        payload = self._request_json("GET", "/api/tags", timeout_s=timeout_s)
        models = payload.get("models", [])
        if not isinstance(models, list):
            return []

        names: List[str] = []
        for item in models:
            if not isinstance(item, dict):
                continue
            candidate = str(item.get("name") or item.get("model") or "").strip()
            if candidate:
                names.append(candidate)
        return names

    def _resolve_model(self, installed_models: List[str]) -> str:
        requested = str(self.model or "").strip()
        if not requested:
            return ""

        requested_lower = requested.lower()
        requested_has_tag = ":" in requested_lower
        requested_base = requested_lower.split(":", 1)[0]
        requested_canon = self._canonical_model_name(requested)

        for candidate in installed_models:
            low = candidate.lower()
            if low == requested_lower:
                return candidate

        for candidate in installed_models:
            canon = self._canonical_model_name(candidate)
            if not canon:
                continue
            if canon == requested_canon:
                return candidate

        # Only allow base-name fallback when the request was untagged.
        if requested_has_tag:
            return ""

        for candidate in installed_models:
            low = candidate.lower()
            base = low.split(":", 1)[0]
            if base == requested_base:
                return candidate

        return ""

    def get_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "available": False,
            "server_reachable": False,
            "model_available": False,
            "model": self.model,
            "resolved_model": "",
            "base_url": self.base_url.rstrip("/"),
            "message": "",
        }

        try:
            installed = self._list_models(timeout_s=self.status_timeout_s)
            status["server_reachable"] = True
            resolved = self._resolve_model(installed)
            status["resolved_model"] = resolved
            status["model_available"] = bool(resolved)
            status["available"] = bool(status["server_reachable"] and status["model_available"])
            if not status["model_available"]:
                status["message"] = (
                    f"Ollama model '{self.model}' is not installed. "
                    f"Run: ollama pull {self.model}"
                )
            else:
                status["message"] = "Ollama vision backend is ready."
            self._last_error = ""
            return status
        except Exception as exc:
            message = str(exc)
            self._last_error = message
            status["message"] = message
            return status

    def is_available(self) -> bool:
        return bool(self.get_status().get("available", False))

    def analyze_image(self, user_prompt: str, image_bytes: bytes) -> str:
        if not image_bytes:
            raise ValueError("image_bytes is required for vision analysis.")

        self.last_used_model = ""
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        prompt_text = str(user_prompt or "Describe this image in detail.").strip() or "Describe this image in detail."

        try:
            installed_models = self._list_models(timeout_s=self.status_timeout_s)
        except Exception as exc:
            raise RuntimeError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        primary_model = self._resolve_model(installed_models)
        if not primary_model:
            raise RuntimeError(
                f"Ollama model '{self.model}' is not installed. Run: ollama pull {self.model}"
            )

        candidates = self._fallback_model_order(installed_models, primary_model)
        errors: List[str] = []
        memory_blocked_primary = False

        for model_name in candidates:
            payload = {
                "model": model_name,
                "stream": False,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt_text,
                        "images": [image_b64],
                    }
                ],
            }

            try:
                response = self._request_json("POST", "/api/chat", payload=payload, timeout_s=self.timeout_s)
                if response.get("error"):
                    raise RuntimeError(str(response.get("error")))

                message = response.get("message", {})
                if not isinstance(message, dict):
                    raise RuntimeError("Unexpected Ollama chat response payload.")

                content = str(message.get("content") or "").strip()
                if not content:
                    raise RuntimeError("Ollama returned an empty vision response.")

                self.last_used_model = model_name
                return content
            except Exception as exc:
                error_text = str(exc).strip() or "unknown error"
                errors.append(f"{model_name}: {error_text}")
                if model_name.lower() == primary_model.lower() and self._is_memory_error(error_text):
                    memory_blocked_primary = True
                continue

        if memory_blocked_primary:
            raise RuntimeError(
                "Requested Ollama vision model could not fit in available memory and no fallback vision model succeeded. "
                + " | ".join(errors)
            )

        raise RuntimeError("Ollama vision request failed. " + " | ".join(errors))

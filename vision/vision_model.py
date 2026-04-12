from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict

from core.config import log_performance
from llm.qwen_bridge import QwenBridge
from vision.screen_capture import ScreenCapture


class VisionModel:
    def __init__(self, compute_mode: str = "auto") -> None:
        self.capture = ScreenCapture()
        self.bridge = QwenBridge(compute_mode=compute_mode)

    def set_compute_mode(self, mode: str) -> None:
        self.bridge.set_compute_mode(mode)

    @staticmethod
    def _prepare_image_bytes(image_path: Path) -> tuple[bytes | None, str]:
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
                rgb.thumbnail((1344, 1344))
                buffer = io.BytesIO()
                rgb.save(buffer, format="PNG")
                return buffer.getvalue(), ""
        except Exception as exc:
            return None, str(exc)

    def _answer(self, image_path: Path, prompt: str) -> tuple[str, bool]:
        image_bytes, image_error = self._prepare_image_bytes(image_path)
        if image_bytes is None:
            return (
                "The image could not be prepared for multimodal analysis. "
                f"Details: {image_error}",
                False,
            )

        prompt_text = (prompt or "Describe this image in detail.").strip()
        if not prompt_text:
            prompt_text = "Describe this image in detail."

        vl_prompt = (
            "Analyze only the attached image and answer the request directly. "
            "If details are uncertain, state uncertainty instead of guessing. "
            "Do not prepend the current date/time unless asked or visible in the image. "
            f"User request: {prompt_text}"
        )
        answer = self.bridge.generate(vl_prompt, context=[], device_hint="gpu", image_bytes=image_bytes)
        text = str(answer).strip()
        lower = text.lower()
        usable = bool(text) and not lower.startswith("local llm error")
        if "vision is unavailable in the current runtime" in lower:
            usable = False

        if usable:
            return text, True
        if not text:
            text = "The multimodal vision runtime is currently unavailable."
        return text, False

    def describe_screen(self, prompt: str = "Describe what is currently visible on the screen.") -> Dict[str, Any]:
        start = time.perf_counter()
        _, path = self.capture.capture_full(save=True)
        if path is None:
            return {
                "success": False,
                "response_text": "I could not capture the screen.",
                "data": {},
            }
        answer, available = self._answer(path, prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        log_performance("vision_screen", latency_ms)
        return {
            "success": available,
            "response_text": answer,
            "data": {"image_path": str(path), "latency_ms": latency_ms, "model_available": available},
        }

    def close(self) -> None:
        self.bridge.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def describe_image(
        self, image_path: str | Path | None, prompt: str = "Describe this image in detail."
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        if image_path is None:
            return {
                "success": False,
                "response_text": "No image path was provided.",
                "data": {},
            }

        path = Path(image_path)
        if not path.exists():
            return {
                "success": False,
                "response_text": "The image file could not be found.",
                "data": {"image_path": str(path)},
            }

        answer, available = self._answer(path, prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        log_performance("vision_image", latency_ms)
        return {
            "success": available,
            "response_text": answer,
            "data": {"image_path": str(path), "latency_ms": latency_ms, "model_available": available},
        }

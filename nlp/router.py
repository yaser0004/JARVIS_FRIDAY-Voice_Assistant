from __future__ import annotations

import io
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from actions import (
    app_control,
    clipboard_control,
    file_control,
    realtime_web,
    system_control,
    system_info,
    time_control,
    web_control,
)
from core.config import log_performance
from core.session_logging import trace_event, trace_exception
from llm.qwen_bridge import QwenBridge
from vision.cnn_classifier import CNNImageClassifier
from vision.screen_capture import ScreenCapture
from vision.webcam import WebcamCapture


class Router:
    def __init__(
        self,
        llm: QwenBridge | None = None,
        cancel_callback=None,
    ) -> None:
        self.llm = llm
        self.cancel_callback = cancel_callback
        self._last_llm_init_attempt = 0.0
        self._llm_retry_cooldown_s = 1.5
        self._response_verbosity = "normal"
        self._last_route_result: Dict[str, Any] | None = None
        self.realtime_web_enabled = False
        self._cnn: CNNImageClassifier | None = None
        self._cnn_attempted = False
        self._camera = WebcamCapture()
        self._screen_capture = ScreenCapture()
        self._trace("initialized", llm_ready=bool(self.llm is not None))

    def _trace(self, event: str, **details: Any) -> None:
        trace_event("backend.router", event, **details)

    def _ensure_llm(self) -> bool:
        if self.llm is not None:
            try:
                if self.llm.is_ready():
                    self._trace("ensure_llm_ready", source="existing")
                    return True
            except Exception:
                pass

            is_available = getattr(self.llm, "is_available", None)
            if callable(is_available):
                try:
                    if bool(is_available()):
                        self._trace("ensure_llm_available", source="existing")
                        return True
                except Exception:
                    pass

        now = time.monotonic()
        if now - self._last_llm_init_attempt < self._llm_retry_cooldown_s:
            self._trace("ensure_llm_throttled")
            return False

        self._last_llm_init_attempt = now
        self._trace("ensure_llm_init_attempt")
        try:
            self.llm = QwenBridge()
            ready = False
            available = False
            try:
                ready = bool(self.llm.is_ready())
            except Exception:
                ready = False
            try:
                available = bool(self.llm.is_available())
            except Exception:
                available = False

            self._trace("ensure_llm_init_result", ready=bool(ready), available=bool(available))
            # Freshly initialized bridges are often "available" before runtime warmup.
            # Allow generation to proceed so the bridge can spin up the runtime on demand.
            return bool(ready or available)
        except Exception as exc:
            trace_exception("backend.router", exc, event="ensure_llm_init_failed")
            log_performance("llm_init_error", 0.0, str(exc))
            self.llm = None
            return False

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(str(text).strip().lower().split())

    def _is_small_talk(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        if not normalized:
            return False

        exact = {
            "hi",
            "hello",
            "hey",
            "hey there",
            "yo",
            "how are you",
            "how are you doing",
            "how r you",
            "good morning",
            "good afternoon",
            "good evening",
            "what's up",
            "whats up",
            "thanks",
            "thank you",
        }
        if normalized in exact:
            return True

        prefixes = ("hi ", "hello ", "hey ", "how are you", "how r you")
        return any(normalized.startswith(prefix) for prefix in prefixes)

    def _should_use_verified_web(self, raw_text: str, intent: str) -> bool:
        if not self.realtime_web_enabled:
            return False

        normalized = self._normalize_text(raw_text)
        if not normalized or self._is_small_talk(normalized):
            return False

        explicit_markers = (
            "search web",
            "web search",
            "look up",
            "lookup",
            "from the web",
            "on the web",
            "latest",
            "news",
            "today",
            "current",
            "update",
            "recent",
            "price",
            "weather",
            "score",
            "stock",
        )
        if not any(marker in normalized for marker in explicit_markers):
            return False

        # Use the realtime-web extractor only for explicit web/fresh-information asks.
        if intent not in {"web_search", "general_qa"}:
            return False

        return realtime_web.looks_like_research_query(raw_text)

    @staticmethod
    def _normalize_verbosity_mode(verbosity: str | None) -> str:
        value = (verbosity or "normal").strip().lower()
        aliases = {
            "short": "brief",
            "concise": "brief",
            "terse": "brief",
            "default": "normal",
            "regular": "normal",
            "long": "detailed",
            "verbose": "detailed",
            "thorough": "detailed",
            "comprehensive": "detailed",
        }
        normalized = aliases.get(value, value)
        if normalized not in {"brief", "normal", "detailed"}:
            return "normal"
        return normalized

    @staticmethod
    def _clip_for_prompt(value: Any, max_chars: int = 220) -> str:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    def _remember_route_result(self, route_path: str, payload: Dict[str, Any]) -> None:
        if route_path in {"general_llm", "general_fallback", "small_talk"}:
            return

        data = payload.get("data") if isinstance(payload, dict) else {}
        if not isinstance(data, dict):
            data = {}

        compact_data: Dict[str, str] = {}
        for key in [
            "intent",
            "level",
            "requested_level",
            "mode",
            "platform",
            "url",
            "cancelled",
            "error",
        ]:
            if key in data and data.get(key) is not None:
                compact_data[key] = self._clip_for_prompt(data.get(key), max_chars=80)

        self._last_route_result = {
            "route_path": route_path,
            "success": bool(payload.get("success", False)),
            "response_text": self._clip_for_prompt(payload.get("response_text", ""), max_chars=220),
            "data": compact_data,
            "at": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        }

    def _build_general_system_prompt(
        self,
        raw_text: str,
        intent: str,
        entities: Dict[str, Any],
        compute_hint: str | None,
    ) -> str:
        verbosity = self._response_verbosity
        if verbosity == "brief":
            verbosity_rule = "Keep replies to 1-2 short sentences unless the user asks for details."
        elif verbosity == "detailed":
            verbosity_rule = "Give fuller answers with practical detail while staying clear and grounded."
        else:
            verbosity_rule = "Keep replies concise by default (about 2-3 sentences)."

        entity_pairs = []
        for key, value in sorted((entities or {}).items()):
            if value is None:
                continue
            entity_pairs.append(f"{key}={self._clip_for_prompt(value, max_chars=60)}")
        entity_summary = ", ".join(entity_pairs) if entity_pairs else "none"

        lines = [
            "You are JARVIS (Just A Really Very Intelligent System), a local desktop assistant running on the user's Windows PC.",
            "Hybrid runtime policy:",
            "- System operations are executed by deterministic handlers outside the LLM.",
            "- Never claim you changed system state unless an executed result is provided in context.",
            "- If a user asks for an operation that was not executed, ask for a direct command or clarify limitations.",
            "- For informational and conversational questions, provide the best direct answer.",
            "- Do not prepend time/date unless the user asked for it.",
            verbosity_rule,
            f"Current runtime time: {datetime.now().astimezone().strftime('%A, %B %d, %Y %I:%M %p %Z')}.",
            f"Realtime web mode: {'enabled' if self.realtime_web_enabled else 'disabled'}.",
            f"Current intent hint: {intent}.",
            f"Extracted entities: {entity_summary}.",
            f"Compute hint: {str(compute_hint or 'auto').lower()}.",
            f"Latest user input: {self._clip_for_prompt(raw_text, max_chars=180)}",
        ]

        if self._last_route_result:
            last = self._last_route_result
            lines.append("Most recent executed route result (for continuity):")
            lines.append(f"- route: {last.get('route_path', 'unknown')}")
            lines.append(f"- success: {bool(last.get('success', False))}")
            lines.append(f"- at: {last.get('at', 'unknown')}")
            lines.append(f"- spoken result: {last.get('response_text', '')}")
            compact_data = last.get("data", {})
            if isinstance(compact_data, dict) and compact_data:
                data_items = ", ".join(f"{k}={v}" for k, v in sorted(compact_data.items()))
                lines.append(f"- result data: {data_items}")

        return "\n".join(lines)

    def set_realtime_web_enabled(self, enabled: bool) -> None:
        self.realtime_web_enabled = bool(enabled)

    def set_response_verbosity(self, verbosity: str) -> None:
        self._response_verbosity = self._normalize_verbosity_mode(verbosity)
        self._trace("set_response_verbosity", verbosity=self._response_verbosity)

    def _is_visual_request(self, raw_text: str, entities: Dict[str, Any]) -> bool:
        if entities.get("vision_mode") in {"screen", "image", "camera"}:
            return True
        if entities.get("file_path"):
            return True

        normalized = self._normalize_text(raw_text)
        visual_patterns = [
            r"\b(screen|screenshot|display|monitor)\b",
            r"\b(image|photo|picture|pic)\b",
            r"\b(camera|webcam)\b",
            r"\bwhat do you see\b",
            r"\bdescribe (this|the) (image|photo|screen)\b",
        ]
        return any(re.search(pattern, normalized) for pattern in visual_patterns)

    def _route_fast_paths(self, raw_text: str) -> Dict[str, Any] | None:
        normalized = self._normalize_text(raw_text)
        if not normalized:
            return None

        if time_control.looks_like_time_query(normalized):
            return time_control.handle_time_query(raw_text)

        if system_info.looks_like_system_info_query(normalized):
            return system_info.get_system_awareness()

        if any(token in normalized for token in ["rescan app", "refresh app index", "reindex app"]):
            return app_control.rescan_app_index()

        if any(token in normalized for token in ["rescan music", "refresh music index", "reindex music"]):
            from actions import media_control

            return media_control.rescan_media_index()

        if "current volume" in normalized or "what is the volume" in normalized:
            return system_control.get_volume()

        if "current brightness" in normalized or "what is the brightness" in normalized:
            return system_control.get_brightness()

        if "pause" in normalized and any(k in normalized for k in ["music", "song", "track", "playback"]):
            from actions import media_control

            return media_control.pause()
        if "resume" in normalized and any(k in normalized for k in ["music", "song", "track", "playback"]):
            from actions import media_control

            return media_control.resume()
        if "stop" in normalized and any(k in normalized for k in ["music", "song", "track", "playback"]):
            from actions import media_control

            return media_control.stop()
        if any(token in normalized for token in ["next track", "next song", "skip track", "skip song"]):
            from actions import media_control

            return media_control.next_track()
        if any(token in normalized for token in ["previous track", "previous song", "last track"]):
            from actions import media_control

            return media_control.previous_track()

        if any(token in normalized for token in ["turn off monitor", "monitor off"]):
            return system_control.power_action("monitor_off")
        if any(token in normalized for token in ["wifi on", "enable wifi", "turn on wifi"]):
            return system_control.toggle_wifi(True)
        if any(token in normalized for token in ["wifi off", "disable wifi", "turn off wifi"]):
            return system_control.toggle_wifi(False)
        if any(token in normalized for token in ["bluetooth on", "enable bluetooth", "turn on bluetooth"]):
            return system_control.toggle_bluetooth(True)
        if any(token in normalized for token in ["bluetooth off", "disable bluetooth", "turn off bluetooth"]):
            return system_control.toggle_bluetooth(False)
        if any(token in normalized for token in ["airplane mode on", "enable airplane mode", "turn on airplane mode"]):
            return system_control.toggle_airplane_mode(True)
        if any(token in normalized for token in ["airplane mode off", "disable airplane mode", "turn off airplane mode"]):
            return system_control.toggle_airplane_mode(False)
        if any(token in normalized for token in ["battery saver on", "enable battery saver", "turn on battery saver"]):
            return system_control.toggle_battery_saver(True)
        if any(token in normalized for token in ["battery saver off", "disable battery saver", "turn off battery saver"]):
            return system_control.toggle_battery_saver(False)

        return None

    def _ensure_cnn(self) -> bool:
        if self._cnn is not None:
            return self._cnn.is_ready()
        if self._cnn_attempted:
            return False
        self._cnn_attempted = True
        try:
            self._cnn = CNNImageClassifier()
            return self._cnn.is_ready()
        except Exception:
            self._cnn = None
            return False

    def _prepare_image_payload(self, file_path: str) -> tuple[bytes | None, str]:
        try:
            from PIL import Image

            with Image.open(file_path) as img:
                rgb = img.convert("RGB")
                # Smaller payloads significantly improve Qwen-VL stability on 4-6GB GPUs.
                try:
                    max_edge = int(os.getenv("JARVIS_VISION_MAX_EDGE", "512"))
                except Exception:
                    max_edge = 512
                max_edge = max(384, min(1024, max_edge))
                if max(rgb.size) > max_edge:
                    rgb.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)

                buffer = io.BytesIO()
                rgb.save(buffer, format="PNG")
                return buffer.getvalue(), ""
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def _sanitize_vision_response(user_prompt: str, response_text: str) -> str:
        text = str(response_text or "").strip()
        if not text:
            return ""

        text = text.lstrip(":\u2014- ").strip()
        prompt_lower = str(user_prompt or "").lower()
        asked_time_or_date = any(token in prompt_lower for token in ["time", "date", "day", "clock"])
        if not asked_time_or_date:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                lead = lines[0].lower()
                if lead.startswith(("today is", "the current time", "current date", "it is currently")):
                    lines = lines[1:]
            text = " ".join(lines).strip() or text

        return text

    @staticmethod
    def _format_cnn_hints(cnn_result: Dict[str, Any] | None) -> str:
        if not cnn_result or not cnn_result.get("success"):
            return ""

        preds = cnn_result.get("data", {}).get("predictions", [])
        if not preds:
            return ""

        label_text = ", ".join(
            f"{item.get('label')} ({float(item.get('confidence', 0.0)) * 100:.1f}%)"
            for item in preds[:3]
        )
        return f"CNN hints (may be noisy): {label_text}."

    @staticmethod
    def _build_cnn_fallback_response(cnn_result: Dict[str, Any] | None) -> str:
        if not cnn_result or not cnn_result.get("success"):
            return ""

        preds = cnn_result.get("data", {}).get("predictions", [])
        if not preds:
            return ""

        top = preds[0]
        top_label = str(top.get("label") or "unknown object")
        try:
            top_conf = float(top.get("confidence", 0.0)) * 100.0
        except Exception:
            top_conf = 0.0

        return (
            "Multimodal vision is unavailable right now. "
            f"Fallback classifier estimate: {top_label} ({top_conf:.1f}% confidence)."
        )

    def analyze_image_file(self, file_path: str | None, prompt: str = "Describe this image in detail.") -> Dict[str, Any]:
        if not file_path:
            return {
                "success": False,
                "response_text": "Please provide an image file path.",
                "data": {},
            }

        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "response_text": "The image file could not be found.",
                "data": {"image_path": str(path)},
            }

        cnn_result = None
        if self._ensure_cnn() and self._cnn is not None:
            cnn_result = self._cnn.classify_image(file_path)
        cnn_hints = self._format_cnn_hints(cnn_result)
        cnn_fallback_response = self._build_cnn_fallback_response(cnn_result)

        image_bytes, image_error = self._prepare_image_payload(file_path)
        llm_result_text = ""
        llm_error_text = ""
        llm_available = self._ensure_llm() and self.llm is not None
        if llm_available and image_bytes is not None:
            prompt_text = (prompt or "Describe this image in detail.").strip()
            if not prompt_text:
                prompt_text = "Describe this image in detail."

            vl_prompt = (
                "Analyze only the attached image and answer the request directly. "
                "If details are uncertain, state uncertainty instead of guessing. "
                "Do not prepend the current date/time unless asked or visible in the image."
            )
            vl_prompt = f"{vl_prompt}\nUser request: {prompt_text}"

            llm_result_text = self.llm.generate(
                vl_prompt,
                context=[],
                device_hint="gpu",
                image_bytes=image_bytes,
            )

            llm_text_lower = str(llm_result_text).lower()
            llm_answer_usable = bool(llm_result_text) and not llm_text_lower.startswith("local llm")
            if "vision is unavailable in the current runtime" in llm_text_lower:
                llm_answer_usable = False
                llm_error_text = (
                    "I could not run multimodal image analysis because Qwen2.5-VL vision runtime is unavailable. "
                    "Run `python setup_models.py` to install the VL GGUF + mmproj files."
                )
            elif not llm_answer_usable:
                llm_error_text = str(llm_result_text).strip() or "Multimodal image analysis is temporarily unavailable."

            if llm_answer_usable:
                cleaned_response = self._sanitize_vision_response(prompt_text, str(llm_result_text))
                return {
                    "success": True,
                    "response_text": cleaned_response,
                    "data": {
                        "mode": "qwen2.5-vl",
                        "image_path": str(path),
                        "cnn": cnn_result.get("data") if cnn_result else None,
                    },
                }

        if llm_error_text:
            response = llm_error_text
            if cnn_fallback_response:
                response = f"{response}\n{cnn_fallback_response}"
                if image_error:
                    response = f"{response}\nImage preprocessing warning: {image_error}"
                return {
                    "success": True,
                    "response_text": response,
                    "data": {
                        "mode": "cnn_fallback",
                        "image_path": str(path),
                        "cnn": cnn_result.get("data") if cnn_result and cnn_result.get("success") else None,
                    },
                }
            if cnn_hints:
                response = f"{response}\nOptional CNN hint (low confidence): {cnn_hints}"
            if image_error:
                response = f"{response}\nImage preprocessing warning: {image_error}"
            return {
                "success": False,
                "response_text": response,
                "data": {
                    "mode": "qwen2.5-vl-unavailable",
                    "image_path": str(path),
                    "cnn": cnn_result.get("data") if cnn_result and cnn_result.get("success") else None,
                },
            }

        if image_error:
            return {
                "success": False,
                "response_text": f"I could not prepare the image for analysis: {image_error}",
                "data": {"image_path": str(path)},
            }

        if llm_result_text:
            response = str(llm_result_text).strip()
            if cnn_hints:
                response = f"{response}\nOptional CNN hint (low confidence): {cnn_hints}"
            return {
                "success": False,
                "response_text": response,
                "data": {
                    "image_path": str(path),
                    "mode": "qwen2.5-vl",
                    "cnn": cnn_result.get("data") if cnn_result and cnn_result.get("success") else None,
                },
            }

        if cnn_fallback_response:
            response = cnn_fallback_response
            if image_error:
                response = f"{response}\nImage preprocessing warning: {image_error}"
            return {
                "success": True,
                "response_text": response,
                "data": {
                    "mode": "cnn_fallback",
                    "image_path": str(path),
                    "cnn": cnn_result.get("data") if cnn_result and cnn_result.get("success") else None,
                },
            }

        if cnn_hints:
            return {
                "success": False,
                "response_text": (
                    "Multimodal image analysis is unavailable right now. "
                    "Run `python setup_models.py` to install Qwen2.5-VL artifacts.\n"
                    f"Optional CNN hint (low confidence): {cnn_hints}"
                ),
                "data": {
                    "image_path": str(path),
                    "mode": "qwen2.5-vl-unavailable",
                    "cnn": cnn_result.get("data") if cnn_result and cnn_result.get("success") else None,
                },
            }

        return {
            "success": False,
            "response_text": "Image analysis is unavailable right now.",
            "data": {},
        }

    def analyze_camera_capture(self) -> Dict[str, Any]:
        ok, path, message = self._camera.capture_frame()
        if not ok or path is None:
            return {
                "success": False,
                "response_text": f"I could not capture from camera: {message}",
                "data": {},
            }
        result = self.analyze_image_file(str(path), prompt="Describe this camera image.")
        data = dict(result.get("data") or {})
        data.setdefault("camera_path", str(path))
        result["data"] = data
        return result

    def _fallback_screen_analysis(self) -> Dict[str, Any]:
        _, image_path = self._screen_capture.capture_full(save=True)
        if image_path is None:
            return {
                "success": False,
                "response_text": "I could not capture the screen for fallback analysis.",
                "data": {},
            }

        result = self.analyze_image_file(
            str(image_path),
            prompt="Describe what is visible on this screenshot.",
        )
        data = dict(result.get("data") or {})
        data.setdefault("image_path", str(image_path))
        data.setdefault("mode", "screen_fallback")
        result["data"] = data
        return result

    def _fallback_general_response(self, raw_text: str) -> str:
        text = self._normalize_text(raw_text)
        if text in {"hi", "hello", "hey", "hey there", "yo"}:
            return "Hey, I am here and ready. What would you like to do?"
        if text in {"how are you", "how are you doing", "how r you", "what's up", "whats up"}:
            return "I am doing well and ready to help. You can ask me to open apps, control system settings, or search the web."
        if text in {"thanks", "thank you"}:
            return "You are welcome."
        return "I can still help with app control, system settings, media controls, and web actions in this runtime."

    def route(
        self,
        intent_result: Dict[str, Any],
        entities: Dict[str, Any],
        raw_text: str,
        context: List[Dict[str, str]],
        compute_hint: str | None = None,
    ) -> Dict[str, Any]:
        route_started = time.perf_counter()
        intent = intent_result.get("intent", "general_qa")
        self._trace(
            "route_started",
            intent=intent,
            text_chars=len(str(raw_text or "")),
            entity_keys=list(entities.keys()),
            context_turns=len(context),
            compute_hint=compute_hint,
            realtime_web_enabled=bool(self.realtime_web_enabled),
        )

        def done(route_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
            elapsed_ms = (time.perf_counter() - route_started) * 1000
            data = payload.get("data") if isinstance(payload, dict) else {}
            if not isinstance(data, dict):
                data = {}
            self._trace(
                "route_completed",
                route_path=route_path,
                elapsed_ms=round(elapsed_ms, 2),
                success=bool(payload.get("success", False)),
                response_chars=len(str(payload.get("response_text", ""))),
                data_keys=list(data.keys()),
            )
            self._remember_route_result(route_path, payload)
            return payload

        try:
            fast_path = self._route_fast_paths(raw_text)
            if fast_path is not None:
                return done("fast_path", fast_path)

            if self._is_small_talk(raw_text):
                return done(
                    "small_talk",
                    {
                        "success": True,
                        "response_text": self._fallback_general_response(raw_text),
                        "data": {"intent": "small_talk"},
                    },
                )

            if intent == "launch_app":
                return done("launch_app", app_control.launch_app(entities.get("app_name", "")))
            if intent == "close_app":
                return done("close_app", app_control.close_app(entities.get("app_name", "")))
            if intent == "web_search":
                if self._is_small_talk(raw_text):
                    intent = "general_qa"
                elif self._should_use_verified_web(raw_text, intent):
                    verified = realtime_web.verified_answer(raw_text, llm=self.llm)
                    if verified.get("success"):
                        return done("web_search_verified", verified)
                else:
                    return done(
                        "web_search_direct",
                        web_control.search(entities.get("search_query", raw_text), entities.get("platform", "google")),
                    )
            if intent == "open_website":
                return done("open_website", web_control.open_url(entities.get("website_url") or entities.get("app_name", "")))
            if intent == "play_media":
                from actions import media_control

                return done(
                    "play_media",
                    media_control.play(entities.get("media_title", raw_text), entities.get("platform", "auto")),
                )
            if intent == "system_volume":
                return done(
                    "system_volume",
                    system_control.set_volume(entities.get("volume_level", "half"), entities.get("direction")),
                )
            if intent == "system_brightness":
                return done(
                    "system_brightness",
                    system_control.set_brightness(
                        entities.get("brightness_level", "half"),
                        entities.get("direction"),
                    ),
                )
            if intent == "power_control":
                return done("power_control", system_control.power_action(entities.get("power_command", "")))
            if intent == "system_settings":
                return done("system_settings", system_control.open_settings(entities.get("setting_name")))
            if intent == "file_control":
                return done("file_control", file_control.handle(entities))
            if intent == "clipboard_action":
                return done("clipboard_action", clipboard_control.handle(entities))
            if intent == "vision_query":
                if not self._is_visual_request(raw_text, entities):
                    intent = "general_qa"
                else:
                    mode = entities.get("vision_mode", "screen")
                    if "camera" in self._normalize_text(raw_text):
                        mode = "camera"
                    if mode == "image":
                        return done("vision_image", self.analyze_image_file(entities.get("file_path")))
                    if mode == "camera":
                        return done("vision_camera", self.analyze_camera_capture())

                    return done("vision_screen", self._fallback_screen_analysis())

            if intent == "stop_cancel":
                if callable(self.cancel_callback):
                    return done("stop_cancel_callback", self.cancel_callback())
                return done(
                    "stop_cancel_default",
                    {
                        "success": True,
                        "response_text": "Okay, I stopped the current action.",
                        "data": {"cancelled": True},
                    },
                )

            if self._should_use_verified_web(raw_text, intent):
                verified = realtime_web.verified_answer(raw_text, llm=self.llm)
                if verified.get("success"):
                    return done("general_verified_web", verified)

            if not self._ensure_llm() or self.llm is None:
                response = self._fallback_general_response(raw_text)
                return done(
                    "general_fallback",
                    {"success": True, "response_text": response, "data": {"intent": "general_qa"}},
                )

            system_prompt = self._build_general_system_prompt(
                raw_text=raw_text,
                intent=str(intent),
                entities=entities,
                compute_hint=compute_hint,
            )
            try:
                response = self.llm.generate(
                    raw_text,
                    context,
                    device_hint=compute_hint,
                    system_prompt=system_prompt,
                )
            except TypeError:
                response = self.llm.generate(raw_text, context)

            return done(
                "general_llm",
                {"success": True, "response_text": response, "data": {"intent": "general_qa"}},
            )
        except Exception as exc:
            trace_exception("backend.router", exc, event="route_failed", intent=intent)
            log_performance("router_error", 0.0, str(exc))
            return done(
                "route_error",
                {
                    "success": False,
                    "response_text": f"I ran into an issue while handling that request: {exc}",
                    "data": {"error": str(exc), "intent": intent},
                },
            )


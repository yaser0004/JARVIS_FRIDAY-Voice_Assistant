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
    weather_control,
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
        # Keep display-control queries (brightness/dim) in system controls, not vision analysis.
        if re.search(r"\b(?:bright|brightness|brighter|dim|dimmer)\b", normalized) and re.search(
            r"\b(?:screen|display|monitor)\b",
            normalized,
        ):
            return False

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

        if self._looks_like_switch_request(raw_text):
            app_name = self._extract_app_name_from_text(raw_text)
            if app_name:
                return app_control.switch_to_app(app_name)

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
        text = text.replace("\r\n", "\n")
        text = text.replace("**", "")
        text = re.sub(r"^\s*here(?:'s| is)\s+an?\s+analysis[^:]*:\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\[\d+\]", "", text)
        prompt_lower = str(user_prompt or "").lower()
        asked_time_or_date = any(token in prompt_lower for token in ["time", "date", "day", "clock"])
        if not asked_time_or_date:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                lead = lines[0].lower()
                if lead.startswith(("today is", "the current time", "current date", "it is currently")):
                    lines = lines[1:]
            text = "\n".join(lines).strip() or text

        raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_lines: List[str] = []
        seen_keys = set()
        for raw_line in raw_lines:
            line = re.sub(r"^#{1,6}\s*", "", raw_line)
            line = re.sub(r"^[\-*\u2022]+\s+", "", line)
            line = re.sub(r"^\d+[\.)]\s+", "", line)
            line = re.sub(r"(?i)^overall\s+impression\s*:?[\s-]*", "", line)
            line = re.sub(r"\s+", " ", line).strip(" \t:-")
            if not line:
                continue

            lower_line = line.lower()
            if lower_line.startswith(("here is a breakdown", "here's a breakdown")):
                continue
            if lower_line in {
                "overall impression",
                "here is a breakdown of the visible code snippets",
                "here's a breakdown of the visible code snippets",
                "the image shows",
            }:
                continue

            key = re.sub(r"[^a-z0-9]+", " ", lower_line).strip()
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)

            if line and line[-1] not in ".!?":
                line = f"{line}."
            cleaned_lines.append(line)

        if not cleaned_lines:
            compact = re.sub(r"\s+", " ", text).strip()
            return compact

        uncertainty_keywords = (
            "uncertain",
            "uncertainty",
            "cannot determine",
            "can't determine",
            "not enough",
            "unclear",
            "might",
            "may",
        )

        def _line_key(value: str) -> str:
            return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip(" .")

        placeholder_keys = {
            "overview",
            "details",
            "uncertainty",
            "none",
            "n a",
            "na",
            "not available",
            "unknown",
        }

        def _is_placeholder(value: str) -> bool:
            return _line_key(value) in placeholder_keys

        def _dedupe_lines(lines: List[str]) -> List[str]:
            output: List[str] = []
            seen_local = set()
            for item in lines:
                key = _line_key(item)
                if not key or key in seen_local:
                    continue
                seen_local.add(key)
                output.append(item)
            return output

        overview_lines: List[str] = []
        uncertainty_lines: List[str] = []
        detail_lines: List[str] = []
        generic_lines: List[str] = []
        saw_explicit_details = False

        for line in cleaned_lines:
            section_match = re.match(r"^(overview|details|uncertainty)\s*:\s*(.*)$", line, flags=re.IGNORECASE)
            if section_match:
                section = section_match.group(1).lower()
                normalized_line = re.sub(
                    r"^(?:(?:overview|details|uncertainty)\s*:\s*)+",
                    "",
                    line,
                    flags=re.IGNORECASE,
                ).strip(" \t:-")
                if not normalized_line:
                    continue
                if normalized_line and normalized_line[-1] not in ".!?":
                    normalized_line = f"{normalized_line}."

                if section == "overview":
                    if not _is_placeholder(normalized_line):
                        overview_lines.append(normalized_line)
                    continue

                if section == "details":
                    if not _is_placeholder(normalized_line):
                        saw_explicit_details = True
                        detail_lines.append(normalized_line)
                    continue

                if not _is_placeholder(normalized_line):
                    uncertainty_lines.append(normalized_line)
                continue

            lowered = line.lower()
            if lowered.startswith("uncertainty:"):
                payload = line.split(":", 1)[1].strip()
                if payload and not _is_placeholder(payload):
                    uncertainty_lines.append(payload if payload[-1] in ".!?" else f"{payload}.")
                continue

            if any(keyword in lowered for keyword in uncertainty_keywords):
                uncertainty_lines.append(line)
            else:
                generic_lines.append(line)

        overview_lines = _dedupe_lines(overview_lines)
        detail_lines = _dedupe_lines(detail_lines)
        generic_lines = _dedupe_lines(generic_lines)
        uncertainty_lines = [line for line in _dedupe_lines(uncertainty_lines) if not _is_placeholder(line)]

        if not overview_lines and generic_lines:
            overview_lines.append(generic_lines[0])
            generic_lines = generic_lines[1:]

        if generic_lines:
            detail_lines.extend(item for item in generic_lines if not _is_placeholder(item))
            detail_lines = _dedupe_lines(detail_lines)

        if not overview_lines and detail_lines and not saw_explicit_details:
            overview_lines.append(detail_lines[0])
            detail_lines = detail_lines[1:]

        if not overview_lines and not detail_lines:
            compact = re.sub(r"\s+", " ", text).strip()
            return compact

        parts: List[str] = []
        if overview_lines:
            parts.append(f"Overview: {' '.join(overview_lines[:2])}")
        if detail_lines:
            parts.append(f"Details: {' '.join(detail_lines[:4])}")
        if uncertainty_lines:
            parts.append(f"Uncertainty: {' '.join(uncertainty_lines[:2])}")

        if not parts:
            compact = re.sub(r"\s+", " ", text).strip()
            return compact

        return "\n\n".join(parts).strip()

    @staticmethod
    def _vision_backend_preference() -> str:
        value = str(os.getenv("JARVIS_VISION_BACKEND", "ollama")).strip().lower()
        aliases = {
            "llama": "qwen",
            "llama-cpp": "qwen",
            "local": "qwen",
            "worker": "qwen",
        }
        normalized = aliases.get(value, value)
        if normalized not in {"ollama", "qwen", "auto"}:
            return "ollama"
        return normalized

    @staticmethod
    def _build_vision_prompt(user_prompt: str) -> str:
        prompt_text = (user_prompt or "Describe this image in detail.").strip()
        if not prompt_text:
            prompt_text = "Describe this image in detail."

        prefix = (
            "Analyze only the attached image and answer the request directly. "
            "If details are uncertain, state uncertainty instead of guessing. "
            "Do not prepend the current date/time unless asked or visible in the image. "
            "Return clean plain text only: no markdown, no bullet points, no numbered lists. "
            "Use sections in this order only when they have meaningful content: Overview:, Details:, Uncertainty:. "
            "Never repeat section labels inside section text."
        )
        return f"{prefix}\nUser request: {prompt_text}"

    def _run_ollama_vision(self, vision_prompt: str, image_bytes: bytes) -> tuple[str, str, Dict[str, Any]]:
        status: Dict[str, Any] = {}
        try:
            from llm.ollama_vision_bridge import OllamaVisionBridge

            bridge = OllamaVisionBridge()
            status = bridge.get_status()
            if not status.get("server_reachable"):
                return (
                    "",
                    (
                        "Ollama vision backend is unreachable. "
                        "Start Ollama with `ollama serve` and verify JARVIS_OLLAMA_URL."
                    ),
                    status,
                )

            if not status.get("model_available"):
                model_name = str(status.get("model") or "qwen2.5vl:3b")
                return (
                    "",
                    (
                        f"Ollama vision model '{model_name}' is not installed. "
                        f"Run `ollama pull {model_name}`."
                    ),
                    status,
                )

            response_text = bridge.analyze_image(vision_prompt, image_bytes)
            if getattr(bridge, "last_used_model", ""):
                status["resolved_model"] = str(getattr(bridge, "last_used_model") or "")
            return str(response_text or "").strip(), "", status
        except Exception as exc:
            trace_exception("backend.router", exc, event="ollama_vision_failed")
            return "", f"Ollama vision request failed: {exc}", status

    def get_vision_runtime_status(self) -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "preferred_backend": self._vision_backend_preference(),
            "ollama": {
                "available": False,
                "message": "Ollama status has not been checked yet.",
            },
        }
        if status["preferred_backend"] not in {"ollama", "auto"}:
            return status

        try:
            from llm.ollama_vision_bridge import OllamaVisionBridge

            status["ollama"] = OllamaVisionBridge().get_status()
        except Exception as exc:
            status["ollama"] = {
                "available": False,
                "message": f"Ollama vision status probe failed: {exc}",
            }
        return status

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
        vision_errors: List[str] = []
        backend_preference = self._vision_backend_preference()
        prompt_text = (prompt or "Describe this image in detail.").strip()
        if not prompt_text:
            prompt_text = "Describe this image in detail."
        vision_prompt = self._build_vision_prompt(prompt_text)

        if image_bytes is not None and backend_preference in {"ollama", "auto"}:
            ollama_text, ollama_error, ollama_status = self._run_ollama_vision(vision_prompt, image_bytes)
            if ollama_text:
                cleaned_response = self._sanitize_vision_response(prompt_text, str(ollama_text))
                return {
                    "success": True,
                    "response_text": cleaned_response,
                    "data": {
                        "mode": "ollama_vision",
                        "vision_backend": "ollama",
                        "vision_model": str(
                            ollama_status.get("resolved_model") or ollama_status.get("model") or ""
                        ),
                        "image_path": str(path),
                        "cnn": cnn_result.get("data") if cnn_result else None,
                    },
                }
            if ollama_error:
                vision_errors.append(str(ollama_error))

        if image_bytes is not None and backend_preference in {"qwen", "auto"}:
            llm_available = self._ensure_llm() and self.llm is not None
            if llm_available:
                llm_result_text = self.llm.generate(
                    vision_prompt,
                    context=[],
                    device_hint="gpu",
                    image_bytes=image_bytes,
                )

                llm_text_lower = str(llm_result_text).lower()
                llm_answer_usable = bool(llm_result_text) and not llm_text_lower.startswith("local llm")
                if "vision is unavailable in the current runtime" in llm_text_lower:
                    llm_answer_usable = False
                    vision_errors.append(
                        "I could not run multimodal image analysis because Qwen2.5-VL vision runtime is unavailable. "
                        "Run `python setup_models.py` to install the VL GGUF + mmproj files."
                    )
                elif not llm_answer_usable:
                    vision_errors.append(
                        str(llm_result_text).strip() or "Multimodal image analysis is temporarily unavailable."
                    )

                if llm_answer_usable:
                    cleaned_response = self._sanitize_vision_response(prompt_text, str(llm_result_text))
                    return {
                        "success": True,
                        "response_text": cleaned_response,
                        "data": {
                            "mode": "qwen2.5-vl",
                            "vision_backend": "qwen",
                            "image_path": str(path),
                            "cnn": cnn_result.get("data") if cnn_result else None,
                        },
                    }
            else:
                vision_errors.append("Local Qwen vision runtime is unavailable.")

        llm_error_text = "\n".join(item for item in vision_errors if item).strip()

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
                    "mode": "vision_unavailable",
                    "vision_backend": backend_preference,
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
                    "If using Ollama vision, ensure `ollama serve` is running and the model is pulled. "
                    "If using local Qwen vision, run `python setup_models.py`.\n"
                    f"Optional CNN hint (low confidence): {cnn_hints}"
                ),
                "data": {
                    "image_path": str(path),
                    "mode": "vision_unavailable",
                    "vision_backend": backend_preference,
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

    @staticmethod
    def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
        return any(token in text for token in tokens)

    @staticmethod
    def _looks_like_information_request(raw_text: str) -> bool:
        normalized = " ".join(str(raw_text or "").lower().split())
        if not normalized:
            return False

        # Strip common polite/assistant prefixes before question checks.
        prefix_patterns = (
            r"^(?:please|kindly)\s+",
            r"^(?:hey|hi|hello)\s+(?:assistant|jarvis|aria)\s+",
            r"^do\s+me\s+a\s+favor(?:\s+and)?\s+",
            r"^(?:can|could|would|will|shall)\s+you\s+",
        )
        for pattern in prefix_patterns:
            normalized = re.sub(pattern, "", normalized).strip()

        starters = (
            "what ",
            "what's ",
            "whats ",
            "why ",
            "how ",
            "who ",
            "where ",
            "when ",
            "which ",
            "define ",
            "explain ",
            "tell me ",
        )
        if normalized.startswith(starters):
            return True

        question_patterns = (
            r"\bwhat\s+does\b",
            r"\bwhy\s+does\b",
            r"\bhow\s+does\b",
            r"\bhow\s+much\s+is\b",
            r"\bwhat\s+is\b",
            r"\bdifference\s+between\b",
            r"\bpurpose\s+of\b",
            r"\bsigns\s+of\b",
            r"\bcauses\s+of\b",
        )
        return any(re.search(pattern, normalized) for pattern in question_patterns)

    @staticmethod
    def _extract_app_name_from_text(raw_text: str) -> str:
        pattern = re.compile(
            r"\b(?:open|launch|start|run|close|quit|exit|terminate|switch\s+to|focus\s+on|bring)"
            r"(?:\s+(?:the|my|app|application|program|window))?\s+([\w\s\+\.-]{2,80})",
            re.IGNORECASE,
        )
        trailing = re.compile(
            r"\b(?:please|now|for me|if you can|if possible|right now|thanks|thank you|quickly|today|to front)\b",
            re.IGNORECASE,
        )
        match = pattern.search(str(raw_text or ""))
        if not match:
            return ""
        candidate = str(match.group(1) or "").strip().strip(".,?!")
        if not candidate:
            return ""
        candidate = trailing.split(candidate, maxsplit=1)[0]
        candidate = re.sub(r"\b(?:app|application|program|window)\b", " ", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+", " ", candidate).strip().lower()
        if candidate in {"", "this", "that", "it", "current", "active"}:
            return ""
        return candidate

    @staticmethod
    def _looks_like_switch_request(raw_text: str) -> bool:
        normalized = " ".join(str(raw_text or "").lower().split())
        patterns = (
            r"\bswitch\s+to\b",
            r"\bbring\b.*\bto\s+front\b",
            r"\bfocus\s+(?:on|to)\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _recover_intent_from_text(
        self,
        raw_text: str,
        entities: Dict[str, Any],
        predicted_intent: str,
        confidence: float,
    ) -> str:
        intent = str(predicted_intent or "general_qa").strip().lower()
        normalized = self._normalize_text(raw_text)
        weather_terms = (
            "weather",
            "forecast",
            "temperature",
            "rain",
            "snow",
            "humidity",
            "celsius",
            "fahrenheit",
            "degrees",
        )
        search_terms = (
            "search ",
            "look up",
            "lookup",
            "google ",
            "web search",
            "find online",
            "from the web",
            "news",
            "latest",
            "headline",
            "headlines",
            "update",
            "updates",
        )
        switch_request = self._looks_like_switch_request(raw_text)
        info_request = self._looks_like_information_request(raw_text)
        weather_keyword_signal = self._contains_any(normalized, weather_terms)
        weather_signal = weather_keyword_signal or bool(entities.get("weather_unit"))
        explicit_search_signal = bool(entities.get("search_query")) or self._contains_any(normalized, search_terms)

        action_intents = {
            "launch_app",
            "close_app",
            "switch_app",
            "power_control",
            "system_volume",
            "system_brightness",
            "system_settings",
            "file_control",
        }

        # Guard against classifier drift that mislabels search/news requests as weather.
        if intent == "weather_query" and not weather_signal:
            if explicit_search_signal:
                return "web_search"
            return "general_qa"

        # Keep high-confidence non-generic intents from the classifier.
        if intent != "general_qa" and confidence >= 0.35:
            if intent == "web_search" and weather_keyword_signal and not explicit_search_signal:
                pass
            elif switch_request and intent in {"launch_app", "close_app", "web_search"}:
                pass
            elif info_request and intent in action_intents:
                pass
            else:
                return intent

        if self._contains_any(normalized, ("cancel", "stop", "abort", "never mind")):
            return "stop_cancel"

        if self._is_visual_request(raw_text, entities):
            return "vision_query"

        if entities.get("clipboard_action") or "clipboard" in normalized:
            return "clipboard_action"

        if weather_keyword_signal:
            if explicit_search_signal:
                return "web_search"
            if entities.get("weather_location") or re.search(r"\b(?:in|at|for)\s+[a-z]", normalized):
                return "weather_query"
            if "weather" in normalized or "forecast" in normalized:
                return "weather_query"

        power_terms = (
            "shutdown",
            "restart",
            "hibernate",
            "sleep",
            "lock",
            "monitor off",
            "turn off monitor",
            "wifi on",
            "wifi off",
            "bluetooth on",
            "bluetooth off",
            "airplane mode",
            "battery saver",
        )
        power_action_markers = (
            "turn ",
            "enable ",
            "disable ",
            "switch ",
            "restart",
            "shutdown",
            "reboot",
            "lock",
            "sleep",
            "hibernate",
            "put ",
        )
        if entities.get("power_command") or self._contains_any(normalized, power_terms):
            if not info_request:
                return "power_control"

            explicit_power_switch = (
                "turn off monitor",
                "monitor off",
                "wifi on",
                "wifi off",
                "bluetooth on",
                "bluetooth off",
                "airplane mode on",
                "airplane mode off",
                "battery saver on",
                "battery saver off",
            )
            if self._contains_any(normalized, explicit_power_switch) and self._contains_any(
                normalized,
                power_action_markers,
            ):
                return "power_control"

        volume_terms = (
            "volume",
            "mute",
            "unmute",
            "speaker",
            "speakers",
            "sound",
            "audio",
            "louder",
            "quieter",
        )
        if self._contains_any(normalized, volume_terms):
            return "system_volume"

        brightness_terms = (
            "brightness",
            "screen brighter",
            "dim the screen",
            "screen dim",
            "screen bright",
            "display brightness",
        )
        if self._contains_any(normalized, brightness_terms):
            return "system_brightness"

        settings_terms = ("setting", "settings", "open", "show", "go to", "configure")
        if entities.get("setting_name") and self._contains_any(normalized, settings_terms):
            return "system_settings"

        if entities.get("media_title") or normalized.startswith("play "):
            return "play_media"

        website_terms = ("open", "visit", "go to", "website", "site", "browser")
        if entities.get("website_url") and self._contains_any(normalized, website_terms):
            return "open_website"
        if re.search(r"\b(?:https?://|www\.|\w+\.(?:com|org|net|io|edu))(?:\S*)", normalized) and self._contains_any(
            normalized,
            website_terms,
        ):
            return "open_website"

        file_terms = (" file", "folder", "document", "pdf", "txt", "open file", "find file", "search file")
        if self._contains_any(normalized, file_terms):
            return "file_control"

        if explicit_search_signal:
            return "web_search"

        close_terms = ("close ", "quit ", "exit ", "terminate ", "kill ")
        launch_terms = ("open ", "launch ", "start ", "run ")
        switch_terms = ("switch to", "focus on", "bring ")

        if switch_request:
            return "switch_app"

        if entities.get("app_name"):
            if self._contains_any(normalized, close_terms) and not info_request:
                return "close_app"
            if self._contains_any(normalized, launch_terms) and not info_request:
                return "launch_app"
            if self._contains_any(normalized, switch_terms):
                return "switch_app"

        if self._contains_any(normalized, close_terms) and not info_request:
            return "close_app"
        if self._contains_any(normalized, launch_terms) and not info_request:
            return "launch_app"

        return intent

    def route(
        self,
        intent_result: Dict[str, Any],
        entities: Dict[str, Any],
        raw_text: str,
        context: List[Dict[str, str]],
        compute_hint: str | None = None,
    ) -> Dict[str, Any]:
        route_started = time.perf_counter()
        entities = dict(entities or {})
        intent = str(intent_result.get("intent", "general_qa") or "general_qa")
        confidence = float(intent_result.get("confidence", 0.0) or 0.0)
        recovered_intent = self._recover_intent_from_text(raw_text, entities, intent, confidence)
        if recovered_intent != intent:
            self._trace(
                "route_intent_recovered",
                predicted_intent=intent,
                recovered_intent=recovered_intent,
                confidence=confidence,
            )
            intent = recovered_intent
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
                app_name = str(entities.get("app_name") or "").strip().lower()
                if not app_name:
                    app_name = self._extract_app_name_from_text(raw_text)
                return done("launch_app", app_control.launch_app(app_name))
            if intent == "close_app":
                app_name = str(entities.get("app_name") or "").strip().lower()
                if not app_name:
                    app_name = self._extract_app_name_from_text(raw_text)
                return done("close_app", app_control.close_app(app_name))
            if intent == "switch_app":
                app_name = str(entities.get("app_name") or "").strip().lower()
                if not app_name:
                    app_name = self._extract_app_name_from_text(raw_text)
                return done("switch_app", app_control.switch_to_app(app_name))
            if intent == "weather_query":
                normalized = self._normalize_text(raw_text)
                weather_terms = (
                    "weather",
                    "forecast",
                    "temperature",
                    "rain",
                    "snow",
                    "humidity",
                    "celsius",
                    "fahrenheit",
                    "degrees",
                )
                if not self._contains_any(normalized, weather_terms):
                    if self._contains_any(
                        normalized,
                        (
                            "search ",
                            "look up",
                            "lookup",
                            "google ",
                            "web search",
                            "news",
                            "latest",
                            "headline",
                            "headlines",
                            "update",
                            "updates",
                        ),
                    ):
                        self._trace("route_weather_guard_redirect", redirected_to="web_search")
                        intent = "web_search"
                    else:
                        self._trace("route_weather_guard_redirect", redirected_to="general_qa")
                        intent = "general_qa"
                else:
                    weather_entities = dict(entities)
                    weather_entities["raw_text"] = raw_text
                    return done("weather_query", weather_control.handle(weather_entities))

            if intent == "weather_query":
                weather_entities = dict(entities)
                weather_entities["raw_text"] = raw_text
                return done("weather_query", weather_control.handle(weather_entities))
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
                entities.setdefault("search_query", str(raw_text or "").strip())
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


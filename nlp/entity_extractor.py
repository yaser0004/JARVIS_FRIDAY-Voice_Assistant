from __future__ import annotations

import re
import os
import sys
from typing import Dict, Optional

from nlp.conversation_normalizer import normalize_command_text


APP_CATALOG = {
    "chrome",
    "firefox",
    "spotify",
    "vscode",
    "notepad",
    "calculator",
    "discord",
    "steam",
    "photoshop",
    "vlc",
    "word",
    "excel",
    "powerpoint",
    "task manager",
    "file explorer",
    "paint",
}

PLATFORMS = {
    "spotify",
    "youtube",
    "netflix",
    "prime",
    "soundcloud",
    "local",
    "drive",
    "pc",
}

URL_RE = re.compile(r"(https?://\S+)|(www\.\S+)|(\w+\.(com|org|net|io|edu)\S*)", re.IGNORECASE)
SEARCH_RE = re.compile(
    r"(?:search for|look up|google|find|search)\s+(.+)$",
    re.IGNORECASE,
)
MEDIA_RE = re.compile(
    r"play\s+(.+?)(?:\s+on\s+(spotify|youtube|soundcloud|netflix|prime|local|drive|pc))?$",
    re.IGNORECASE,
)
WEATHER_LOCATION_RE = re.compile(
    r"\b(?:weather|forecast|temperature|conditions?)\s+(?:in|at|for)\s+([a-zA-Z][a-zA-Z\s\-\.',]{1,80})",
    re.IGNORECASE,
)
WEATHER_LOCATION_ALT_RE = re.compile(
    r"\b(?:in|at|for)\s+([a-zA-Z][a-zA-Z\s\-\.',]{1,80})\s+(?:weather|forecast|temperature|conditions?)",
    re.IGNORECASE,
)
WEATHER_UNIT_RE = re.compile(r"\b(celsius|fahrenheit|metric|imperial|\d+\s*[cf])\b", re.IGNORECASE)
VOLUME_RE = re.compile(r"(\d{1,3})\s*(%|percent)", re.IGNORECASE)
BRIGHTNESS_RE = re.compile(r"(\d{1,3})\s*(%|percent)", re.IGNORECASE)
CAPITALIZED_APP_RE = re.compile(
    r"\b(?:open|launch|start|close|run|quit|switch\s+to|focus\s+on)\s+([A-Z][\w]*(?:\s+[A-Z][\w]*)*)"
)
APP_COMMAND_RE = re.compile(
    r"\b(?:open|launch|start|close|run|quit|exit|terminate|switch\s+to|focus\s+on|bring)"
    r"(?:\s+(?:the|my|app|application|program|window))?\s+([\w\s\+\.-]{2,80})",
    re.IGNORECASE,
)
APP_SUFFIX_SPLIT_RE = re.compile(
    r"\b(?:please|now|for me|if you can|if possible|right now|thanks|thank you|quickly|today|to front)\b",
    re.IGNORECASE,
)
GENERIC_APP_TARGETS = {"this", "that", "it", "app", "application", "program", "window", "current", "active"}


class EntityExtractor:
    def __init__(self) -> None:
        self.nlp = self._load_spacy_model()
        self._install_entity_ruler()

    @staticmethod
    def _load_spacy_model():
        if os.getenv("JARVIS_ENABLE_SPACY", "0") != "1":
            return None
        if sys.version_info >= (3, 13):
            return None

        use_trf = os.getenv("JARVIS_USE_SPACY_TRF", "0") == "1"
        if use_trf:
            candidates = ["en_core_web_trf", "en_core_web_sm"]
        else:
            candidates = ["en_core_web_sm"]

        for model_name in candidates:
            try:
                import spacy

                return spacy.load(model_name)
            except Exception:
                continue
        return None

    def _install_entity_ruler(self) -> None:
        if self.nlp is None:
            return
        if "entity_ruler" in self.nlp.pipe_names:
            ruler = self.nlp.get_pipe("entity_ruler")
        else:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")

        patterns = []
        for app in APP_CATALOG:
            patterns.append({"label": "APP_NAME", "pattern": app})
        for platform in PLATFORMS:
            patterns.append({"label": "PLATFORM_NAME", "pattern": platform})
        ruler.add_patterns(patterns)

    @staticmethod
    def _extract_level(text: str, regex: re.Pattern[str]) -> Optional[str]:
        match = regex.search(text)
        if match:
            return match.group(1)
        for token in ["max", "min", "half", "zero", "mute", "unmute"]:
            if token in text:
                return token
        return None

    @staticmethod
    def _extract_keyword_number(text: str, keyword: str) -> Optional[str]:
        lowered = str(text or "").lower()
        if keyword not in lowered:
            return None
        numbers = re.findall(r"\b(\d{1,3})\b", lowered)
        if not numbers:
            return None
        return numbers[-1]

    @staticmethod
    def _clean_app_candidate(value: str) -> str:
        candidate = str(value or "").strip().strip(".?!,")
        if not candidate:
            return ""
        candidate = APP_SUFFIX_SPLIT_RE.split(candidate, maxsplit=1)[0]
        candidate = re.sub(r"\b(?:app|application|program|window)\b", " ", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s+", " ", candidate).strip().lower()
        if candidate in GENERIC_APP_TARGETS:
            return ""
        return candidate

    @staticmethod
    def _extract_weather_location(text: str) -> str:
        source = str(text or "")
        if not source:
            return ""

        for regex in (WEATHER_LOCATION_RE, WEATHER_LOCATION_ALT_RE):
            match = regex.search(source)
            if match:
                location = str(match.group(1) or "").strip(" .,!?:;")
                location = re.sub(
                    r"\b(?:please|right now|now|today|tomorrow|currently|current|in\s+celsius|in\s+fahrenheit)\b",
                    "",
                    location,
                    flags=re.IGNORECASE,
                )
                location = re.sub(r"\s+", " ", location).strip(" .,!?:;")
                if location:
                    return location

        fallback = re.sub(
            r"\b(?:what(?:'s| is)?|tell me|show me|give me|how(?:'s| is)|weather|forecast|temperature|conditions?|in|at|for)\b",
            " ",
            source,
            flags=re.IGNORECASE,
        )
        fallback = re.sub(r"\s+", " ", fallback).strip(" .,!?:;")
        if len(fallback.split()) <= 4:
            return fallback
        return ""

    @staticmethod
    def _extract_weather_unit(text: str) -> str:
        source = str(text or "")
        match = WEATHER_UNIT_RE.search(source)
        if not match:
            return ""
        token = str(match.group(1) or "").strip().lower()
        if token in {"fahrenheit", "imperial"} or token.endswith("f"):
            return "F"
        return "C"

    @staticmethod
    def _infer_app_from_text(text: str) -> str:
        lowered = str(text or "").lower()
        for app in sorted(APP_CATALOG, key=len, reverse=True):
            if re.search(rf"\b{re.escape(app)}\b", lowered):
                return app
        return ""

    def extract_entities(self, text: str, intent: str) -> Dict[str, str]:
        text = str(text or "")
        normalized_text = normalize_command_text(text) or text.lower()
        lower_text = normalized_text.lower()
        doc = self.nlp(normalized_text) if self.nlp is not None else None
        slots: Dict[str, str] = {}

        app_match = None
        if doc is not None:
            for ent in doc.ents:
                if ent.label_ == "APP_NAME":
                    app_match = ent.text
                    break

        if app_match:
            slots["app_name"] = app_match.lower()
        else:
            cmd_match = APP_COMMAND_RE.search(normalized_text)
            if cmd_match:
                candidate = self._clean_app_candidate(cmd_match.group(1))
                for app in sorted(APP_CATALOG, key=len, reverse=True):
                    if candidate == app or candidate.startswith(app + " "):
                        candidate = app
                        break
                if candidate:
                    slots["app_name"] = candidate

            cap = CAPITALIZED_APP_RE.search(normalized_text)
            if cap:
                slots["app_name"] = self._clean_app_candidate(cap.group(1))

        if intent in {"launch_app", "close_app", "switch_app"} and "app_name" not in slots:
            inferred_app = self._infer_app_from_text(normalized_text)
            if inferred_app:
                slots["app_name"] = inferred_app

        url_match = URL_RE.search(normalized_text)
        if url_match:
            slots["website_url"] = url_match.group(0)

        search_match = SEARCH_RE.search(normalized_text)
        if search_match:
            slots["search_query"] = search_match.group(1).strip()

        media_match = MEDIA_RE.search(normalized_text)
        if media_match:
            slots["media_title"] = media_match.group(1).strip()
            if media_match.group(2):
                slots["platform"] = media_match.group(2).lower()

        if intent == "weather_query" or any(token in lower_text for token in ["weather", "forecast", "temperature"]):
            weather_location = self._extract_weather_location(normalized_text)
            if weather_location:
                slots["weather_location"] = weather_location
            weather_unit = self._extract_weather_unit(normalized_text)
            if weather_unit:
                slots["weather_unit"] = weather_unit

        if doc is not None:
            for ent in doc.ents:
                if ent.label_ == "PLATFORM_NAME":
                    slots.setdefault("platform", ent.text.lower())
                if ent.label_ in {"PERSON", "PER"}:
                    slots.setdefault("person_name", ent.text)

        volume_level = self._extract_level(normalized_text, VOLUME_RE)
        if volume_level is None and intent == "system_volume":
            volume_level = self._extract_keyword_number(normalized_text, "volume")
        if volume_level is not None:
            slots["volume_level"] = volume_level

        brightness_level = self._extract_level(normalized_text, BRIGHTNESS_RE)
        if brightness_level is None and intent == "system_brightness":
            brightness_level = self._extract_keyword_number(normalized_text, "brightness")
        if brightness_level is not None:
            slots["brightness_level"] = brightness_level

        if any(k in lower_text for k in ["up", "increase", "raise", "turn up"]):
            slots["direction"] = "up"
        elif any(k in lower_text for k in ["down", "decrease", "lower", "turn down"]):
            slots["direction"] = "down"

        for power_cmd in [
            "shutdown",
            "restart",
            "sleep",
            "hibernate",
            "lock",
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
        ]:
            if power_cmd in lower_text:
                slots["power_command"] = power_cmd
                break

        for setting in ["wifi", "bluetooth", "display", "sound", "apps", "updates", "privacy", "battery", "airplane"]:
            if setting in lower_text:
                slots["setting_name"] = setting
                break

        if "clipboard" in lower_text:
            if any(k in lower_text for k in ["read", "show", "what"]):
                slots["clipboard_action"] = "read"
            elif "copy" in lower_text:
                slots["clipboard_action"] = "copy"
            elif "paste" in lower_text:
                slots["clipboard_action"] = "paste"

        if intent == "open_website" and "website_url" not in slots and "app_name" in slots:
            slots["website_url"] = slots["app_name"]

        if intent == "web_search" and "search_query" not in slots:
            slots["search_query"] = normalized_text

        return slots


from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


GEOCODE_API = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT_SECONDS = 7
CACHE_TTL_SECONDS = int(os.getenv("JARVIS_WEATHER_CACHE_SECS", "300"))
DEFAULT_LOCATION = os.getenv("JARVIS_DEFAULT_WEATHER_LOCATION", "")


_CACHE: Dict[Tuple[str, str], Tuple[float, Dict[str, Any]]] = {}


def _response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


def _fetch_json(url: str) -> Dict[str, Any] | None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read().decode("utf-8", errors="ignore")
        return json.loads(raw)
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None
    except Exception:
        return None


def _normalize_location(value: str) -> str:
    lowered = re.sub(r"\s+", " ", str(value or "").strip().lower())
    lowered = re.sub(r"[^a-z0-9\s\-\.',]", "", lowered)
    return lowered.strip(" .,!?:;")


def _clean_location_candidate(value: str) -> str:
    cleaned = str(value or "").strip(" .,!?:;")
    cleaned = re.sub(r"\b(?:please|right now|now|today|tomorrow|currently|current)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(?:in\s+(?:celsius|fahrenheit)|in\s+[cf])\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?:;")
    return cleaned


def _extract_location_from_text(text: str) -> str:
    source = str(text or "").strip()
    if not source:
        return ""

    patterns = [
        r"\b(?:weather|forecast|temperature|conditions?)\s+(?:in|at|for)\s+([a-zA-Z][a-zA-Z\s\-\.',]{1,80})",
        r"\b(?:in|at|for)\s+([a-zA-Z][a-zA-Z\s\-\.',]{1,80})\s+(?:weather|forecast|temperature|conditions?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, source, flags=re.IGNORECASE)
        if match:
            return _clean_location_candidate(match.group(1))

    fallback = source
    fallback = re.sub(r"\b(?:what(?:'s| is)?|tell me|show me|give me|how(?:'s| is))\b", "", fallback, flags=re.IGNORECASE)
    fallback = re.sub(r"\b(?:weather|forecast|temperature|conditions?|for|in|at|please|now|today|tomorrow)\b", " ", fallback, flags=re.IGNORECASE)
    fallback = re.sub(r"\s+", " ", fallback).strip(" .,!?:;")
    return fallback


def _normalize_unit(raw_unit: str) -> Tuple[str, str, str]:
    unit_text = str(raw_unit or "").strip().lower()
    if unit_text in {"f", "fahrenheit", "imperial"}:
        return "fahrenheit", "mph", "F"
    return "celsius", "kmh", "C"


def _geocode(location: str) -> Dict[str, Any] | None:
    params = {
        "name": location,
        "count": 1,
        "language": "en",
        "format": "json",
    }
    url = f"{GEOCODE_API}?{urlencode(params)}"
    payload = _fetch_json(url)
    if not payload:
        return None
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        return None

    top = results[0]
    if not isinstance(top, dict):
        return None
    if top.get("latitude") is None or top.get("longitude") is None:
        return None
    return top


def _weather_code_to_text(code: int) -> str:
    mapping = {
        0: "clear sky",
        1: "mainly clear",
        2: "partly cloudy",
        3: "overcast",
        45: "fog",
        48: "depositing rime fog",
        51: "light drizzle",
        53: "moderate drizzle",
        55: "dense drizzle",
        56: "light freezing drizzle",
        57: "dense freezing drizzle",
        61: "slight rain",
        63: "moderate rain",
        65: "heavy rain",
        66: "light freezing rain",
        67: "heavy freezing rain",
        71: "slight snow",
        73: "moderate snow",
        75: "heavy snow",
        77: "snow grains",
        80: "slight rain showers",
        81: "moderate rain showers",
        82: "violent rain showers",
        85: "slight snow showers",
        86: "heavy snow showers",
        95: "thunderstorm",
        96: "thunderstorm with slight hail",
        99: "thunderstorm with heavy hail",
    }
    return mapping.get(int(code), "variable weather")


def _format_location(geo: Dict[str, Any]) -> str:
    city = str(geo.get("name") or "").strip()
    admin = str(geo.get("admin1") or "").strip()
    country = str(geo.get("country") or "").strip()

    parts = [part for part in [city, admin, country] if part]
    if not parts:
        return "that location"

    compact: list[str] = []
    for part in parts:
        if part not in compact:
            compact.append(part)
    return ", ".join(compact)


def _fetch_current_weather(geo: Dict[str, Any], unit: str) -> Dict[str, Any] | None:
    temperature_unit, wind_speed_unit, _ = _normalize_unit(unit)
    params = {
        "latitude": float(geo.get("latitude")),
        "longitude": float(geo.get("longitude")),
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,precipitation",
        "temperature_unit": temperature_unit,
        "wind_speed_unit": wind_speed_unit,
        "precipitation_unit": "mm",
        "timezone": "auto",
    }
    url = f"{FORECAST_API}?{urlencode(params)}"
    payload = _fetch_json(url)
    if not payload:
        return None

    current = payload.get("current")
    if not isinstance(current, dict):
        return None

    return {
        "current": current,
        "timezone": str(payload.get("timezone") or geo.get("timezone") or ""),
    }


def get_current_weather(location: str, unit: str = "C") -> Dict[str, Any]:
    resolved_location = _clean_location_candidate(location)
    if not resolved_location:
        return _response(False, "Please tell me which place you want the weather for.")

    normalized_location = _normalize_location(resolved_location)
    cache_key = (normalized_location, unit.upper())
    now = time.time()
    cached = _CACHE.get(cache_key)
    if cached and now - cached[0] <= CACHE_TTL_SECONDS:
        return cached[1]

    geo = _geocode(resolved_location)
    if geo is None:
        return _response(False, f"I could not find weather location data for {resolved_location}.")

    weather_payload = _fetch_current_weather(geo, unit)
    if weather_payload is None:
        return _response(False, f"I could not fetch weather for {resolved_location} right now.")

    current = weather_payload["current"]
    _, wind_speed_unit, unit_symbol = _normalize_unit(unit)

    condition = _weather_code_to_text(int(current.get("weather_code") or 0))
    place_name = _format_location(geo)

    temperature = current.get("temperature_2m")
    feels_like = current.get("apparent_temperature")
    humidity = current.get("relative_humidity_2m")
    wind_speed = current.get("wind_speed_10m")
    precipitation = current.get("precipitation")

    response_parts = [
        f"In {place_name}, it is {temperature}{unit_symbol} with {condition}.",
        f"Feels like {feels_like}{unit_symbol}.",
    ]
    if humidity is not None:
        response_parts.append(f"Humidity is {humidity}%.")
    if wind_speed is not None:
        response_parts.append(f"Wind is {wind_speed} {wind_speed_unit}.")
    if precipitation is not None and float(precipitation) > 0:
        response_parts.append(f"Precipitation is {precipitation} mm.")

    payload = _response(
        True,
        " ".join(response_parts),
        {
            "location": place_name,
            "latitude": geo.get("latitude"),
            "longitude": geo.get("longitude"),
            "temperature": temperature,
            "feels_like": feels_like,
            "humidity": humidity,
            "wind_speed": wind_speed,
            "wind_unit": wind_speed_unit,
            "precipitation_mm": precipitation,
            "condition": condition,
            "unit": unit_symbol,
            "timezone": weather_payload.get("timezone"),
            "provider": "open-meteo",
            "observed_at": current.get("time"),
        },
    )

    _CACHE[cache_key] = (now, payload)
    return payload


def handle(entities: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(entities or {})
    raw_text = str(data.get("raw_text") or "")

    location = str(data.get("weather_location") or "").strip()
    if not location and data.get("location"):
        location = str(data.get("location") or "").strip()
    if not location and raw_text:
        location = _extract_location_from_text(raw_text)
    if not location and DEFAULT_LOCATION:
        location = DEFAULT_LOCATION

    if not location:
        return _response(False, "Please mention a location, for example: weather in London.")

    unit = str(data.get("weather_unit") or os.getenv("JARVIS_WEATHER_UNIT", "C"))
    return get_current_weather(location, unit)

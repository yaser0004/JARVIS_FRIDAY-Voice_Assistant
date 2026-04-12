from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Dict, Iterable, List, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except Exception:
    ZoneInfo = None  # type: ignore[assignment]

    class ZoneInfoNotFoundError(Exception):
        pass


def _response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


LOCATION_TIMEZONES: dict[str, str] = {
    "new york": "America/New_York",
    "washington": "America/New_York",
    "washington dc": "America/New_York",
    "boston": "America/New_York",
    "miami": "America/New_York",
    "atlanta": "America/New_York",
    "chicago": "America/Chicago",
    "dallas": "America/Chicago",
    "houston": "America/Chicago",
    "denver": "America/Denver",
    "phoenix": "America/Phoenix",
    "los angeles": "America/Los_Angeles",
    "san francisco": "America/Los_Angeles",
    "seattle": "America/Los_Angeles",
    "las vegas": "America/Los_Angeles",
    "toronto": "America/Toronto",
    "vancouver": "America/Vancouver",
    "montreal": "America/Toronto",
    "mexico city": "America/Mexico_City",
    "bogota": "America/Bogota",
    "lima": "America/Lima",
    "sao paulo": "America/Sao_Paulo",
    "rio": "America/Sao_Paulo",
    "buenos aires": "America/Argentina/Buenos_Aires",
    "santiago": "America/Santiago",
    "reykjavik": "Atlantic/Reykjavik",
    "london": "Europe/London",
    "dublin": "Europe/Dublin",
    "lisbon": "Europe/Lisbon",
    "paris": "Europe/Paris",
    "berlin": "Europe/Berlin",
    "amsterdam": "Europe/Amsterdam",
    "brussels": "Europe/Brussels",
    "madrid": "Europe/Madrid",
    "barcelona": "Europe/Madrid",
    "rome": "Europe/Rome",
    "milan": "Europe/Rome",
    "zurich": "Europe/Zurich",
    "vienna": "Europe/Vienna",
    "prague": "Europe/Prague",
    "warsaw": "Europe/Warsaw",
    "budapest": "Europe/Budapest",
    "stockholm": "Europe/Stockholm",
    "oslo": "Europe/Oslo",
    "copenhagen": "Europe/Copenhagen",
    "helsinki": "Europe/Helsinki",
    "athens": "Europe/Athens",
    "istanbul": "Europe/Istanbul",
    "moscow": "Europe/Moscow",
    "kyiv": "Europe/Kyiv",
    "kiev": "Europe/Kyiv",
    "cairo": "Africa/Cairo",
    "casablanca": "Africa/Casablanca",
    "lagos": "Africa/Lagos",
    "nairobi": "Africa/Nairobi",
    "addis ababa": "Africa/Addis_Ababa",
    "johannesburg": "Africa/Johannesburg",
    "dubai": "Asia/Dubai",
    "abu dhabi": "Asia/Dubai",
    "riyadh": "Asia/Riyadh",
    "doha": "Asia/Qatar",
    "kuwait": "Asia/Kuwait",
    "tehran": "Asia/Tehran",
    "karachi": "Asia/Karachi",
    "lahore": "Asia/Karachi",
    "delhi": "Asia/Kolkata",
    "new delhi": "Asia/Kolkata",
    "mumbai": "Asia/Kolkata",
    "kolkata": "Asia/Kolkata",
    "chennai": "Asia/Kolkata",
    "bengaluru": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata",
    "hyderabad": "Asia/Kolkata",
    "pune": "Asia/Kolkata",
    "kathmandu": "Asia/Kathmandu",
    "dhaka": "Asia/Dhaka",
    "colombo": "Asia/Colombo",
    "bangkok": "Asia/Bangkok",
    "phnom penh": "Asia/Phnom_Penh",
    "jakarta": "Asia/Jakarta",
    "bali": "Asia/Makassar",
    "kuala lumpur": "Asia/Kuala_Lumpur",
    "singapore": "Asia/Singapore",
    "manila": "Asia/Manila",
    "hong kong": "Asia/Hong_Kong",
    "beijing": "Asia/Shanghai",
    "shanghai": "Asia/Shanghai",
    "taipei": "Asia/Taipei",
    "seoul": "Asia/Seoul",
    "tokyo": "Asia/Tokyo",
    "osaka": "Asia/Tokyo",
    "sydney": "Australia/Sydney",
    "melbourne": "Australia/Melbourne",
    "brisbane": "Australia/Brisbane",
    "adelaide": "Australia/Adelaide",
    "perth": "Australia/Perth",
    "auckland": "Pacific/Auckland",
    "wellington": "Pacific/Auckland",
    "india": "Asia/Kolkata",
    "japan": "Asia/Tokyo",
    "china": "Asia/Shanghai",
    "australia": "Australia/Sydney",
    "united kingdom": "Europe/London",
    "uk": "Europe/London",
    "united states": "America/New_York",
    "usa": "America/New_York",
    "canada": "America/Toronto",
    "uae": "Asia/Dubai",
}


# Fallback fixed offsets (minutes from UTC) used only when IANA zone data is unavailable.
FIXED_OFFSETS_MINUTES: dict[str, int] = {
    "America/New_York": -300,
    "America/Chicago": -360,
    "America/Denver": -420,
    "America/Phoenix": -420,
    "America/Los_Angeles": -480,
    "America/Toronto": -300,
    "America/Vancouver": -480,
    "America/Mexico_City": -360,
    "America/Bogota": -300,
    "America/Lima": -300,
    "America/Sao_Paulo": -180,
    "America/Argentina/Buenos_Aires": -180,
    "America/Santiago": -180,
    "Atlantic/Reykjavik": 0,
    "Europe/London": 0,
    "Europe/Dublin": 0,
    "Europe/Lisbon": 0,
    "Europe/Paris": 60,
    "Europe/Berlin": 60,
    "Europe/Amsterdam": 60,
    "Europe/Brussels": 60,
    "Europe/Madrid": 60,
    "Europe/Rome": 60,
    "Europe/Zurich": 60,
    "Europe/Vienna": 60,
    "Europe/Prague": 60,
    "Europe/Warsaw": 60,
    "Europe/Budapest": 60,
    "Europe/Stockholm": 60,
    "Europe/Oslo": 60,
    "Europe/Copenhagen": 60,
    "Europe/Helsinki": 120,
    "Europe/Athens": 120,
    "Europe/Istanbul": 180,
    "Europe/Moscow": 180,
    "Europe/Kyiv": 120,
    "Africa/Cairo": 120,
    "Africa/Casablanca": 0,
    "Africa/Lagos": 60,
    "Africa/Nairobi": 180,
    "Africa/Addis_Ababa": 180,
    "Africa/Johannesburg": 120,
    "Asia/Dubai": 240,
    "Asia/Riyadh": 180,
    "Asia/Qatar": 180,
    "Asia/Kuwait": 180,
    "Asia/Tehran": 210,
    "Asia/Karachi": 300,
    "Asia/Kolkata": 330,
    "Asia/Kathmandu": 345,
    "Asia/Dhaka": 360,
    "Asia/Colombo": 330,
    "Asia/Bangkok": 420,
    "Asia/Phnom_Penh": 420,
    "Asia/Jakarta": 420,
    "Asia/Makassar": 480,
    "Asia/Kuala_Lumpur": 480,
    "Asia/Singapore": 480,
    "Asia/Manila": 480,
    "Asia/Hong_Kong": 480,
    "Asia/Shanghai": 480,
    "Asia/Taipei": 480,
    "Asia/Seoul": 540,
    "Asia/Tokyo": 540,
    "Australia/Sydney": 600,
    "Australia/Melbourne": 600,
    "Australia/Brisbane": 600,
    "Australia/Adelaide": 570,
    "Australia/Perth": 480,
    "Pacific/Auckland": 720,
}


TIME_TRIGGER_RE = re.compile(r"\b(time|clock)\b", re.IGNORECASE)
DATE_TRIGGER_RE = re.compile(r"\b(date|day)\b", re.IGNORECASE)
WORLD_CLOCK_RE = re.compile(r"\bworld\s*clock\b", re.IGNORECASE)
LOCATION_RE = re.compile(r"\b(?:in|at|for|of)\s+([a-zA-Z][a-zA-Z\s\-\.'`]{1,60})")
WORLD_TIME_API = "https://worldtimeapi.org/api/timezone/{zone}"
REQUEST_TIMEOUT_SECONDS = 6


def _normalize_location(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9\s\-]", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _format_datetime(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _zone_datetime(zone_name: str) -> tuple[datetime | None, str]:
    if ZoneInfo is not None:
        try:
            now = datetime.now(ZoneInfo(zone_name))
            return now, now.tzname() or zone_name
        except ZoneInfoNotFoundError:
            pass
        except Exception:
            pass

    online = _zone_datetime_online(zone_name)
    if online is not None:
        return online

    offset = FIXED_OFFSETS_MINUTES.get(zone_name)
    if offset is None:
        return None, zone_name

    now_utc = datetime.now(timezone.utc)
    tz = timezone(timedelta(minutes=offset))
    now = now_utc.astimezone(tz)
    sign = "+" if offset >= 0 else "-"
    hours = abs(offset) // 60
    mins = abs(offset) % 60
    return now, f"UTC{sign}{hours:02d}:{mins:02d}"


def _zone_datetime_online(zone_name: str) -> tuple[datetime, str] | None:
    url = WORLD_TIME_API.format(zone=zone_name)
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8", errors="ignore"))
    except (URLError, json.JSONDecodeError, TimeoutError, ValueError):
        return None
    except Exception:
        return None

    raw_dt = str(payload.get("datetime") or "").strip()
    if not raw_dt:
        return None

    try:
        dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
    except Exception:
        return None

    return dt, str(payload.get("abbreviation") or zone_name)


def _resolve_location(raw: str) -> tuple[str, str] | None:
    query = _normalize_location(raw)
    if not query:
        return None

    if query in LOCATION_TIMEZONES:
        return query, LOCATION_TIMEZONES[query]

    best: tuple[float, str] | None = None
    for key in LOCATION_TIMEZONES:
        if query in key or key in query:
            score = 0.95
        else:
            score = SequenceMatcher(a=query, b=key).ratio()
        if best is None or score > best[0]:
            best = (score, key)

    if best is None or best[0] < 0.70:
        return None

    resolved_key = best[1]
    return resolved_key, LOCATION_TIMEZONES[resolved_key]


def _extract_locations(text: str) -> List[str]:
    found = [m.group(1).strip() for m in LOCATION_RE.finditer(text)]
    if found:
        return found

    normalized = _normalize_location(text)
    direct_hits = [key for key in LOCATION_TIMEZONES if re.search(rf"\b{re.escape(key)}\b", normalized)]
    if direct_hits:
        return direct_hits

    chunks = re.split(r"[,/]|\band\b", text, flags=re.IGNORECASE)
    guessed: List[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        if len(chunk.split()) > 0 and len(chunk) <= 40:
            guessed.append(chunk)
    return guessed


def _query_has_location_hint(text: str) -> bool:
    if LOCATION_RE.search(text):
        return True

    normalized = _normalize_location(text)
    if not normalized:
        return False

    return any(re.search(rf"\b{re.escape(key)}\b", normalized) for key in LOCATION_TIMEZONES)


def get_current_time(location: str | None = None) -> Dict[str, Any]:
    if not location:
        now = datetime.now().astimezone()
        tz_name = now.tzname() or "local"
        return _response(True, f"The current time is {_format_datetime(now)} ({tz_name}).", {"location": "local"})

    resolved = _resolve_location(location)
    if resolved is None:
        return _response(False, f"I do not have a supported location mapping for {location}.")

    city_key, zone_name = resolved
    dt, tz_label = _zone_datetime(zone_name)
    if dt is None:
        return _response(False, f"I could not resolve timezone data for {city_key}.")

    pretty = city_key.title()
    return _response(True, f"The current time in {pretty} is {_format_datetime(dt)} ({tz_label}).", {"location": city_key, "timezone": zone_name})


def get_current_date(location: str | None = None) -> Dict[str, Any]:
    if not location:
        now = datetime.now().astimezone()
        return _response(True, f"Today's date is {now.strftime('%Y-%m-%d')}.", {"location": "local"})

    resolved = _resolve_location(location)
    if resolved is None:
        return _response(False, f"I do not have a supported location mapping for {location}.")

    city_key, zone_name = resolved
    dt, _ = _zone_datetime(zone_name)
    if dt is None:
        return _response(False, f"I could not resolve timezone data for {city_key}.")

    pretty = city_key.title()
    return _response(True, f"The current date in {pretty} is {dt.strftime('%Y-%m-%d')}.", {"location": city_key, "timezone": zone_name})


def world_clock(locations: Iterable[str] | None = None) -> Dict[str, Any]:
    if locations is None:
        locations = [
            "new york",
            "london",
            "paris",
            "dubai",
            "delhi",
            "singapore",
            "tokyo",
            "sydney",
        ]

    lines: List[str] = []
    resolved_items: List[Dict[str, str]] = []
    for item in locations:
        resolved = _resolve_location(item)
        if resolved is None:
            continue
        city_key, zone_name = resolved
        dt, tz_label = _zone_datetime(zone_name)
        if dt is None:
            continue
        lines.append(f"{city_key.title()}: {dt.strftime('%Y-%m-%d %H:%M:%S')} ({tz_label})")
        resolved_items.append({"location": city_key, "timezone": zone_name})

    if not lines:
        return _response(False, "I could not resolve any requested locations for world clock.")

    return _response(True, "World clock:\n" + "\n".join(lines), {"locations": resolved_items})


def looks_like_time_query(text: str) -> bool:
    value = text.strip()
    normalized = _normalize_location(value)

    if "time complexity" in normalized:
        return False

    if WORLD_CLOCK_RE.search(value) or DATE_TRIGGER_RE.search(value):
        return True

    time_patterns = [
        r"\bwhat(?:'s| is)? the time\b",
        r"\bwhat time is it\b",
        r"\btell me(?:\s+the)? time\b",
        r"\btime now\b",
        r"\bcurrent time\b",
        r"\btime in\b",
        r"\blocal time\b",
        r"\bclock\b",
    ]
    if any(re.search(pattern, normalized) for pattern in time_patterns):
        return True

    return normalized in {"time", "date", "day"}


def handle_time_query(text: str) -> Dict[str, Any]:
    value = text.strip()
    if not value:
        return _response(False, "Please provide a time or date query.")

    has_location_hint = _query_has_location_hint(value)

    if WORLD_CLOCK_RE.search(value):
        candidates = _extract_locations(value)
        return world_clock(candidates if candidates else None)

    if DATE_TRIGGER_RE.search(value) and not TIME_TRIGGER_RE.search(value):
        candidates = _extract_locations(value) if has_location_hint else []
        if candidates:
            return get_current_date(candidates[0])
        return get_current_date()

    candidates = _extract_locations(value) if has_location_hint else []
    if candidates:
        return get_current_time(candidates[0])
    return get_current_time()

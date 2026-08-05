from __future__ import annotations

import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

import psutil

from actions.common import response as _response
from core.config import APPDATA_DIR
from platform_adapters import backend


APP_INDEX_PATH = APPDATA_DIR / "app_index.json"
INDEX_VERSION = 1


def _normalize_key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


def _sanitize_app_target(value: str) -> str:
    cleaned = _normalize_key(str(value or ""))
    cleaned = re.sub(
        r"\b(?:opened|running|currently|current|active|the|my|this|that|it|app|application|program|window)\b",
        " ",
        cleaned,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned in {"", "this", "that", "it", "current", "active", "opened", "running"}:
        return ""
    return cleaned


def _window_match_score(query: str, title: str) -> float:
    normalized_query = _normalize_key(query)
    normalized_title = _normalize_key(title)
    if not normalized_query or not normalized_title:
        return 0.0
    if normalized_query == normalized_title:
        return 1.0
    if normalized_query in normalized_title:
        return 0.95
    if normalized_title in normalized_query:
        return 0.9
    return SequenceMatcher(a=normalized_query, b=normalized_title).ratio()


def _app_record(kind: str, path: str, source: str) -> Dict[str, str]:
    return {
        "kind": kind,
        "path": path,
        "source": source,
    }


def _load_index() -> Dict[str, Any] | None:
    if not APP_INDEX_PATH.exists():
        return None
    try:
        with APP_INDEX_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if int(data.get("version", 0)) != INDEX_VERSION:
            return None
        if not isinstance(data.get("apps"), dict):
            return None
        return data
    except Exception:
        return None


def _save_index(index: Dict[str, Any]) -> None:
    APPDATA_DIR.mkdir(parents=True, exist_ok=True)
    with APP_INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def _register_app(index_map: Dict[str, Dict[str, str]], alias: str, record: Dict[str, str]) -> None:
    key = _normalize_key(alias)
    if not key:
        return
    existing = index_map.get(key)
    if existing is None:
        index_map[key] = record
        return

    # Prefer the platform's native launch kind (exe / command) when aliases collide.
    preferred = backend.DEFAULT_APP_KIND
    if existing.get("kind") != preferred and record.get("kind") == preferred:
        index_map[key] = record


def build_app_index() -> Dict[str, Any]:
    apps: Dict[str, Dict[str, str]] = {}

    for alias, target in backend.DEFAULT_APP_MAP.items():
        _register_app(apps, alias, _app_record(backend.DEFAULT_APP_KIND, target, "builtin"))

    scan_stats = backend.scan_apps(lambda alias, record: _register_app(apps, alias, record))

    return {
        "version": INDEX_VERSION,
        "generated_at": time.time(),
        "stats": {"total_apps": len(apps), **scan_stats},
        "apps": apps,
    }


def ensure_app_index(force_rescan: bool = False) -> Dict[str, Any]:
    if not force_rescan:
        loaded = _load_index()
        if loaded is not None:
            return loaded

    index = build_app_index()
    _save_index(index)
    return index


def rescan_app_index() -> Dict[str, Any]:
    index = ensure_app_index(force_rescan=True)
    stats = index.get("stats", {})
    return _response(
        True,
        f"App index refreshed. Indexed {stats.get('total_apps', 0)} applications.",
        {"index_path": str(APP_INDEX_PATH), **stats},
    )


def _select_app_record(app_name: str) -> Dict[str, str] | None:
    query = _normalize_key(app_name)
    if not query:
        return None

    apps = ensure_app_index().get("apps", {})
    if query in apps:
        return apps[query]

    candidates: list[tuple[float, Dict[str, str]]] = []
    for key, record in apps.items():
        if query in key or key in query:
            score = 0.95
        else:
            score = SequenceMatcher(a=query, b=key).ratio()
        if score >= 0.72:
            candidates.append((score, record))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _find_executable(app_name: str) -> str | None:
    record = _select_app_record(app_name)
    if record is not None and record.get("kind") == backend.DEFAULT_APP_KIND:
        return str(record.get("path", ""))

    normalized = app_name.strip().lower()
    if normalized in backend.DEFAULT_APP_MAP:
        return backend.DEFAULT_APP_MAP[normalized]

    return backend.find_executable(app_name)


def launch_app(app_name: str) -> Dict[str, Any]:
    try:
        normalized = (app_name or "").strip().lower()
        if not normalized:
            return _response(False, "Please specify an app name.")

        ensure_app_index()
        launch_errors: list[str] = []

        selected = _select_app_record(normalized)
        if selected is not None:
            target_path = str(selected.get("path", ""))
            kind = str(selected.get("kind", ""))
            if target_path:
                launched, method = backend.launch_target(target_path)
                if launched:
                    return _response(True, f"Launching {normalized}.", {"target": target_path, "kind": kind})
                launch_errors.append(f"{target_path}: {method}")

        executable = _find_executable(normalized)
        if executable:
            launched, method = backend.launch_target(executable)
            if launched:
                return _response(True, f"Launching {normalized}.", {"executable": executable, "method": method})
            launch_errors.append(f"{executable}: {method}")

        launched, method = backend.launch_target(normalized)
        if launched:
            return _response(True, f"Attempting to launch {normalized} via system shell.", {"method": method})

        detail = "; ".join(launch_errors[-2:]) if launch_errors else method
        return _response(False, f"I could not launch {app_name}: {detail}", {"errors": launch_errors})
    except Exception as exc:
        return _response(False, f"I could not launch {app_name}: {exc}", {"error": str(exc)})


def close_app(app_name: str) -> Dict[str, Any]:
    try:
        target = _sanitize_app_target(app_name)
        if not target:
            return _response(False, "Please specify an app name to close.")

        record = _select_app_record(target)
        target_tokens = {_normalize_key(target).replace(" ", "")}
        if record is not None:
            path = str(record.get("path", ""))
            stem = Path(path).stem.lower().replace(" ", "")
            if stem:
                target_tokens.add(stem)

        for alias in backend.DEFAULT_APP_MAP:
            if target in alias:
                target_tokens.add(alias.replace(" ", ""))

        killed = 0
        for proc in psutil.process_iter(["name", "pid", "exe"]):
            name = (proc.info.get("name") or "").lower()
            exe_path = (proc.info.get("exe") or "").lower()

            proc_tokens = {
                name.replace(" ", ""),
                Path(name).stem.replace(" ", ""),
                Path(exe_path).stem.replace(" ", ""),
            }

            should_kill = False
            for token in target_tokens:
                if not token:
                    continue
                if token in proc_tokens:
                    should_kill = True
                    break

            if should_kill:
                try:
                    proc.terminate()
                    killed += 1
                except Exception:
                    continue

        if killed == 0:
            return _response(False, f"I could not find {target} running.")
        return _response(True, f"Closed {killed} process(es) for {target}.", {"killed": killed})
    except Exception as exc:
        return _response(False, f"I could not close {app_name}: {exc}", {"error": str(exc)})


def list_running_apps() -> List[str]:
    names = set()
    for proc in psutil.process_iter(["name"]):
        name = proc.info.get("name")
        if name:
            names.add(name)
    return sorted(names)


def switch_to_app(app_name: str) -> Dict[str, Any]:
    try:
        target = _sanitize_app_target(app_name)
        if not target:
            return _response(False, "Please specify an app name to switch to.")

        ranked: list[tuple[float, Any]] = []
        for title, handle in backend.list_windows():
            score = _window_match_score(target, title)
            if score >= 0.55:
                ranked.append((score, handle))

        if not ranked:
            ranked = [(0.6, handle) for _, handle in backend.find_windows_by_title(target)]

        if not ranked:
            return _response(False, f"I could not find an open window for {target}.")

        ranked.sort(key=lambda item: item[0], reverse=True)
        backend.activate_window(ranked[0][1])
        return _response(True, f"Switched to {target}.")
    except Exception as exc:
        return _response(False, f"I could not switch to {app_name}: {exc}", {"error": str(exc)})


def app_index_summary() -> Dict[str, Any]:
    index = ensure_app_index()
    stats = index.get("stats", {})
    return _response(
        True,
        f"Indexed {stats.get('total_apps', 0)} apps.",
        {"index_path": str(APP_INDEX_PATH), **stats},
    )


def is_app_available(app_name: str) -> bool:
    return _select_app_record(app_name) is not None

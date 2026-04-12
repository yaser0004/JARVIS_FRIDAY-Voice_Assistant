from __future__ import annotations

import json
import os
import re
import subprocess
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List

import psutil
import pygetwindow as gw

from core.config import APPDATA_DIR


APP_MAP = {
    "chrome": "chrome.exe",
    "firefox": "firefox.exe",
    "edge": "msedge.exe",
    "spotify": "Spotify.exe",
    "vscode": "Code.exe",
    "visual studio code": "Code.exe",
    "discord": "Discord.exe",
    "notepad": "notepad.exe",
    "notepad++": "notepad++.exe",
    "calculator": "calc.exe",
    "file explorer": "explorer.exe",
    "task manager": "taskmgr.exe",
    "paint": "mspaint.exe",
    "word": "WINWORD.EXE",
    "excel": "EXCEL.EXE",
    "powerpoint": "POWERPNT.EXE",
    "outlook": "OUTLOOK.EXE",
    "onenote": "ONENOTE.EXE",
    "vlc": "vlc.exe",
    "steam": "Steam.exe",
    "photoshop": "Photoshop.exe",
    "illustrator": "Illustrator.exe",
    "premiere": "Adobe Premiere Pro.exe",
    "after effects": "AfterFX.exe",
    "blender": "blender.exe",
    "obs": "obs64.exe",
    "telegram": "Telegram.exe",
    "whatsapp": "WhatsApp.exe",
    "slack": "slack.exe",
    "teams": "Teams.exe",
    "zoom": "Zoom.exe",
    "skype": "Skype.exe",
    "chrome canary": "chrome.exe",
    "opera": "opera.exe",
    "brave": "brave.exe",
    "epic": "EpicGamesLauncher.exe",
    "battle.net": "Battle.net.exe",
    "origin": "Origin.exe",
    "uplay": "upc.exe",
    "intellij": "idea64.exe",
    "pycharm": "pycharm64.exe",
    "android studio": "studio64.exe",
    "postman": "Postman.exe",
    "git bash": "git-bash.exe",
    "powershell": "powershell.exe",
    "cmd": "cmd.exe",
    "terminal": "wt.exe",
    "snipping tool": "SnippingTool.exe",
    "camera": "WindowsCamera.exe",
    "photos": "PhotosApp.exe",
    "movies": "Video.UI.exe",
    "spotify launcher": "SpotifyLauncher.exe",
    "audacity": "audacity.exe",
    "acrobat": "Acrobat.exe",
    "filezilla": "filezilla.exe",
    "winrar": "WinRAR.exe",
    "7zip": "7zFM.exe",
    "control panel": "control.exe",
    "registry editor": "regedit.exe",
}

SEARCH_ROOTS = [
    os.getenv("APPDATA", ""),
    os.getenv("LOCALAPPDATA", ""),
    r"C:\Program Files",
    r"C:\Program Files (x86)",
]

APP_INDEX_PATH = APPDATA_DIR / "app_index.json"
INDEX_VERSION = 1

EXECUTABLE_SCAN_ROOTS = [
    os.path.join(os.getenv("ProgramFiles", r"C:\Program Files")),
    os.path.join(os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")),
    os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs"),
]

SHORTCUT_SCAN_ROOTS = [
    os.path.join(os.getenv("APPDATA", ""), r"Microsoft\Windows\Start Menu\Programs"),
    os.path.join(os.getenv("ProgramData", r"C:\ProgramData"), r"Microsoft\Windows\Start Menu\Programs"),
    os.path.join(str(Path.home()), "Desktop"),
]


def _response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


def _normalize_key(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()
    return re.sub(r"\s+", " ", cleaned)


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

    # Prefer executable paths over shortcuts when aliases collide.
    if existing.get("kind") != "exe" and record.get("kind") == "exe":
        index_map[key] = record


def _scan_shortcuts(index_map: Dict[str, Dict[str, str]]) -> int:
    found = 0
    for root in SHORTCUT_SCAN_ROOTS:
        if not root:
            continue
        root_path = Path(root)
        if not root_path.exists():
            continue
        try:
            for candidate in root_path.rglob("*.lnk"):
                alias = candidate.stem.replace("_", " ").replace("-", " ").strip()
                if not alias:
                    continue
                _register_app(index_map, alias, _app_record("shortcut", str(candidate), "scan"))
                found += 1
        except Exception:
            continue
    return found


def _scan_executables(index_map: Dict[str, Dict[str, str]]) -> int:
    found = 0
    for root in EXECUTABLE_SCAN_ROOTS:
        if not root:
            continue
        root_path = Path(root)
        if not root_path.exists():
            continue
        try:
            for candidate in root_path.rglob("*.exe"):
                name = candidate.stem.strip()
                if not name:
                    continue
                if name.lower() in {"uninstall", "unins000"}:
                    continue
                _register_app(index_map, name, _app_record("exe", str(candidate), "scan"))
                found += 1
        except Exception:
            continue
    return found


def build_app_index() -> Dict[str, Any]:
    apps: Dict[str, Dict[str, str]] = {}

    for alias, executable in APP_MAP.items():
        _register_app(apps, alias, _app_record("exe", executable, "builtin"))

    shortcut_count = _scan_shortcuts(apps)
    exe_count = _scan_executables(apps)

    return {
        "version": INDEX_VERSION,
        "generated_at": time.time(),
        "stats": {
            "total_apps": len(apps),
            "shortcuts_scanned": shortcut_count,
            "executables_scanned": exe_count,
        },
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
    if record is not None and record.get("kind") == "exe":
        return str(record.get("path", ""))

    normalized = app_name.strip().lower()
    if normalized in APP_MAP:
        return APP_MAP[normalized]

    needle = normalized.replace(" ", "")
    for root in SEARCH_ROOTS:
        if not root:
            continue
        root_path = Path(root)
        if not root_path.exists():
            continue
        try:
            for candidate in root_path.rglob("*.exe"):
                if needle in candidate.stem.lower().replace(" ", ""):
                    return str(candidate)
        except Exception:
            continue
    return None


def _try_launch_target(target_path: str) -> tuple[bool, str]:
    target = str(target_path or "").strip()
    if not target:
        return False, "empty launch target"

    # Preferred on Windows: ShellExecute handles shortcuts, App Paths, and file associations.
    try:
        if os.path.isabs(target):
            abs_target = Path(target)
            if abs_target.exists():
                os.startfile(str(abs_target))
                return True, "startfile"
        else:
            os.startfile(target)
            return True, "startfile"
    except Exception as start_exc:
        start_error = str(start_exc)
    else:
        start_error = ""

    try:
        subprocess.Popen([target], shell=False)
        return True, "subprocess"
    except Exception as popen_exc:
        popen_error = str(popen_exc)

    return False, f"startfile: {start_error}; subprocess: {popen_error}"


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
                launched, method = _try_launch_target(target_path)
                if launched:
                    return _response(True, f"Launching {normalized}.", {"target": target_path, "kind": kind})
                launch_errors.append(f"{target_path}: {method}")

        executable = _find_executable(normalized)
        if executable:
            launched, method = _try_launch_target(executable)
            if launched:
                return _response(True, f"Launching {normalized}.", {"executable": executable, "method": method})
            launch_errors.append(f"{executable}: {method}")

        launched, method = _try_launch_target(normalized)
        if launched:
            return _response(True, f"Attempting to launch {normalized} via system shell.", {"method": method})

        detail = "; ".join(launch_errors[-2:]) if launch_errors else method
        return _response(False, f"I could not launch {app_name}: {detail}", {"errors": launch_errors})
    except Exception as exc:
        return _response(False, f"I could not launch {app_name}: {exc}", {"error": str(exc)})


def close_app(app_name: str) -> Dict[str, Any]:
    try:
        target = (app_name or "").strip().lower()
        if not target:
            return _response(False, "Please specify an app name to close.")

        record = _select_app_record(target)
        target_tokens = {_normalize_key(target).replace(" ", "")}
        if record is not None:
            path = str(record.get("path", ""))
            stem = Path(path).stem.lower().replace(" ", "")
            if stem:
                target_tokens.add(stem)

        for alias in APP_MAP:
            if target in alias:
                target_tokens.add(alias.replace(" ", ""))

        killed = 0
        for proc in psutil.process_iter(["name", "pid", "exe", "cmdline"]):
            name = (proc.info.get("name") or "").lower()
            exe_path = (proc.info.get("exe") or "").lower()
            cmdline = " ".join(proc.info.get("cmdline") or []).lower()

            proc_tokens = {
                name.replace(" ", ""),
                Path(exe_path).stem.replace(" ", ""),
            }
            proc_blob = (name + " " + exe_path + " " + cmdline).replace(" ", "")

            should_kill = False
            for token in target_tokens:
                if not token:
                    continue
                if token in proc_blob or token in proc_tokens:
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
        matches = gw.getWindowsWithTitle(app_name)
        if not matches:
            return _response(False, f"I could not find a window for {app_name}.")
        window = matches[0]
        if window.isMinimized:
            window.restore()
        window.activate()
        return _response(True, f"Switched to {app_name}.")
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

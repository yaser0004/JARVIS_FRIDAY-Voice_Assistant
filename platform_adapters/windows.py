"""Windows backend.

Bodies are moved verbatim from the original action modules (system_control,
app_control, system_info) so Windows behavior stays identical; only the
function boundaries are new. Functions raise on failure unless documented
otherwise; the action layer turns exceptions into spoken error responses.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pygetwindow as gw
import screen_brightness_control as sbc
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


def _run_command(command: list[str], timeout: int = 8) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        output = (result.stdout or result.stderr or "").strip()
        return result.returncode == 0, output
    except Exception as exc:
        return False, str(exc)


def _run_powershell(command: str, timeout: int = 10) -> tuple[bool, str]:
    return _run_command(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ],
        timeout=timeout,
    )


# --- volume -----------------------------------------------------------------

def _get_volume_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return interface.QueryInterface(IAudioEndpointVolume)


def get_volume() -> Tuple[int, bool]:
    volume = _get_volume_interface()
    current = int(round(volume.GetMasterVolumeLevelScalar() * 100))
    muted = bool(volume.GetMute())
    return current, muted


def set_volume_level(percent: int) -> None:
    volume = _get_volume_interface()
    volume.SetMasterVolumeLevelScalar(percent / 100.0, None)
    volume.SetMute(0, None)


def set_mute(muted: bool) -> None:
    _get_volume_interface().SetMute(1 if muted else 0, None)


# --- brightness -------------------------------------------------------------

def get_brightness() -> int:
    values = sbc.get_brightness()
    if isinstance(values, list) and values:
        return int(round(sum(int(v) for v in values) / len(values)))
    return int(values)


def set_brightness(percent: int) -> None:
    sbc.set_brightness(percent)


# --- power ------------------------------------------------------------------

def shutdown() -> None:
    subprocess.run(["shutdown", "/s", "/t", "5"], check=False)


def restart() -> None:
    subprocess.run(["shutdown", "/r", "/t", "5"], check=False)


def sleep() -> None:
    subprocess.run(
        ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
        check=False,
    )


def hibernate() -> None:
    subprocess.run(["shutdown", "/h"], check=False)


def lock() -> None:
    ctypes.windll.user32.LockWorkStation()


def monitor_off() -> None:
    ctypes.windll.user32.SendMessageW(0xFFFF, 0x0112, 0xF170, 2)


# --- connectivity -----------------------------------------------------------

def toggle_wifi(enable: bool) -> Tuple[bool, str]:
    return _toggle_adapter(enable, "Wi[- ]?Fi|WLAN|Wireless")


def _toggle_adapter(enable: bool, adapter_regex: str) -> Tuple[bool, str]:
    action = "Enable" if enable else "Disable"
    command = (
        f"$adapters = Get-NetAdapter | Where-Object {{$_.Status -ne 'Unknown' -and $_.Name -match '{adapter_regex}'}}; "
        "if (-not $adapters) { exit 2 }; "
        f"$adapters | {action}-NetAdapter -Confirm:$false"
    )
    return _run_powershell(command)


def toggle_bluetooth(enable: bool) -> Tuple[bool, str]:
    # Bluetooth toggling may require elevated privileges and varies by driver.
    service_action = "Start-Service" if enable else "Stop-Service"
    command = f"{service_action} bthserv -ErrorAction Stop"
    return _run_powershell(command)


def current_wifi_name() -> str | None:
    ok, out = _run_powershell("netsh wlan show interfaces")
    if not ok or not out:
        return None

    for line in out.splitlines():
        if "SSID" in line and "BSSID" not in line:
            _, _, value = line.partition(":")
            ssid = value.strip()
            if ssid and ssid.lower() != "not connected":
                return ssid
    return None


def current_bluetooth_name() -> str | None:
    cmd = (
        "Get-PnpDevice -Class Bluetooth | "
        "Where-Object {$_.Status -eq 'OK'} | "
        "Select-Object -ExpandProperty FriendlyName"
    )
    ok, out = _run_powershell(cmd)
    if not ok or not out:
        return None
    names = [line.strip() for line in out.splitlines() if line.strip()]
    if not names:
        return None
    # Keep the first user-facing bluetooth device/adapter name.
    return names[0]


# --- settings pages ---------------------------------------------------------

_SETTINGS_PAGES = {
    "wifi": "network-wifi",
    "network": "network",
    "bluetooth": "bluetooth",
    "airplane": "network-airplanemode",
    "display": "display",
    "sound": "sound",
    "apps": "appsfeatures",
    "updates": "windowsupdate",
    "privacy": "privacy",
    "battery": "batterysaver",
}


def open_settings_page(key: str) -> str:
    page = _SETTINGS_PAGES.get(key, key)
    os.startfile(f"ms-settings:{page}")
    return page


# --- open files / URIs ------------------------------------------------------

def open_path(target: str) -> None:
    os.startfile(str(target))


# --- application index / launch ---------------------------------------------

DEFAULT_APP_KIND = "exe"

DEFAULT_APP_MAP = {
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


def scan_apps(register: Callable[[str, Dict[str, str]], None]) -> Dict[str, int]:
    shortcut_count = 0
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
                register(alias, {"kind": "shortcut", "path": str(candidate), "source": "scan"})
                shortcut_count += 1
        except Exception:
            continue

    exe_count = 0
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
                register(name, {"kind": "exe", "path": str(candidate), "source": "scan"})
                exe_count += 1
        except Exception:
            continue

    return {"shortcuts_scanned": shortcut_count, "executables_scanned": exe_count}


def find_executable(app_name: str) -> str | None:
    needle = app_name.strip().lower().replace(" ", "")
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


def launch_target(target_path: str) -> Tuple[bool, str]:
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


# --- window management --------------------------------------------------------

def list_windows() -> List[Tuple[str, Any]]:
    windows = []
    for window in gw.getAllWindows():
        title = str(getattr(window, "title", "") or "").strip()
        if title:
            windows.append((title, window))
    return windows


def find_windows_by_title(title: str) -> List[Tuple[str, Any]]:
    return [(str(getattr(window, "title", "") or "").strip(), window) for window in gw.getWindowsWithTitle(title)]


def activate_window(handle: Any) -> None:
    if handle.isMinimized:
        handle.restore()
    handle.activate()

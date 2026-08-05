"""Linux backend.

Relies on the standard desktop CLI surface — pactl, brightnessctl, systemctl,
loginctl, nmcli, bluetoothctl, xdg-open, wmctrl — with graceful degradation:
when a tool is missing, functions raise RuntimeError naming the package, and
the action layer turns that into a spoken error. Window management needs X11
(wmctrl); on Wayland it degrades to an explanatory failure.
"""

from __future__ import annotations

import configparser
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple


def _run_command(command: list[str], timeout: int = 8) -> tuple[bool, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        output = (result.stdout or result.stderr or "").strip()
        return result.returncode == 0, output
    except Exception as exc:
        return False, str(exc)


def _require(tool: str, package_hint: str | None = None) -> str:
    path = shutil.which(tool)
    if path is None:
        hint = package_hint or tool
        raise RuntimeError(f"'{tool}' is not installed (install the {hint} package)")
    return path


def _run_checked(command: list[str], timeout: int = 8) -> str:
    _require(command[0])
    ok, output = _run_command(command, timeout=timeout)
    if not ok:
        raise RuntimeError(output or f"{' '.join(command)} failed")
    return output


# --- volume (PulseAudio/PipeWire via pactl) -----------------------------------

def get_volume() -> Tuple[int, bool]:
    _require("pactl", "pulseaudio-utils")
    out = _run_checked(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
    match = re.search(r"(\d+)%", out)
    if not match:
        raise RuntimeError(f"could not parse volume from: {out}")
    level = int(match.group(1))
    ok, mute_out = _run_command(["pactl", "get-mute", "@DEFAULT_SINK@"])
    muted = ok and "yes" in mute_out.lower()
    return min(level, 100), muted


def set_volume_level(percent: int) -> None:
    _require("pactl", "pulseaudio-utils")
    _run_checked(["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{int(percent)}%"])
    _run_command(["pactl", "set-mute", "@DEFAULT_SINK@", "0"])


def set_mute(muted: bool) -> None:
    _require("pactl", "pulseaudio-utils")
    _run_checked(["pactl", "set-mute", "@DEFAULT_SINK@", "1" if muted else "0"])


# --- brightness (brightnessctl) -------------------------------------------------

def get_brightness() -> int:
    _require("brightnessctl")
    # -m: device,class,current,percent,max e.g. "intel_backlight,backlight,48000,50%,96000"
    out = _run_checked(["brightnessctl", "-m"])
    for part in out.split(","):
        part = part.strip()
        if part.endswith("%") and part[:-1].isdigit():
            return int(part[:-1])
    raise RuntimeError(f"could not parse brightness from: {out}")


def set_brightness(percent: int) -> None:
    _require("brightnessctl")
    _run_checked(["brightnessctl", "set", f"{int(percent)}%"])


# --- power (systemd) ------------------------------------------------------------

def shutdown() -> None:
    _run_checked(["systemctl", "poweroff"])


def restart() -> None:
    _run_checked(["systemctl", "reboot"])


def sleep() -> None:
    _run_checked(["systemctl", "suspend"])


def hibernate() -> None:
    _run_checked(["systemctl", "hibernate"])


def lock() -> None:
    _run_checked(["loginctl", "lock-session"])


def monitor_off() -> None:
    if not os.getenv("DISPLAY"):
        raise RuntimeError("turning the monitor off needs an X11 session (xset dpms)")
    _run_checked(["xset", "dpms", "force", "off"])


# --- connectivity (NetworkManager / BlueZ) ---------------------------------------

def toggle_wifi(enable: bool) -> Tuple[bool, str]:
    if shutil.which("nmcli") is None:
        return False, "'nmcli' is not installed (install the network-manager package)"
    return _run_command(["nmcli", "radio", "wifi", "on" if enable else "off"])


def toggle_bluetooth(enable: bool) -> Tuple[bool, str]:
    if shutil.which("bluetoothctl") is None:
        return False, "'bluetoothctl' is not installed (install the bluez package)"
    return _run_command(["bluetoothctl", "power", "on" if enable else "off"])


def current_wifi_name() -> str | None:
    if shutil.which("nmcli") is None:
        return None
    ok, out = _run_command(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"])
    if not ok:
        return None
    for line in out.splitlines():
        active, _, ssid = line.partition(":")
        if active == "yes" and ssid.strip():
            return ssid.strip()
    return None


def current_bluetooth_name() -> str | None:
    if shutil.which("bluetoothctl") is None:
        return None
    ok, out = _run_command(["bluetoothctl", "devices", "Connected"])
    if not ok or not out:
        return None
    for line in out.splitlines():
        # Format: "Device <MAC> <Name>"
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0] == "Device":
            return parts[2].strip()
    return None


# --- settings pages (GNOME control center) ----------------------------------------

_SETTINGS_PANELS = {
    "wifi": "wifi",
    "network": "network",
    "bluetooth": "bluetooth",
    "airplane": "wifi",
    "display": "display",
    "sound": "sound",
    "apps": "applications",
    "updates": "info-overview",
    "privacy": "privacy",
    "battery": "power",
}


def open_settings_page(key: str) -> str:
    panel = _SETTINGS_PANELS.get(key, key)
    launcher = _require("gnome-control-center", "gnome-control-center (or open settings manually)")
    subprocess.Popen([launcher, panel])
    return panel


# --- open files / URIs --------------------------------------------------------------

def open_path(target: str) -> None:
    _require("xdg-open", "xdg-utils")
    subprocess.Popen(["xdg-open", str(target)])


# --- application index / launch ------------------------------------------------------

DEFAULT_APP_KIND = "command"

DEFAULT_APP_MAP = {
    "firefox": "firefox",
    "chrome": "google-chrome",
    "files": "nautilus",
    "file explorer": "nautilus",
    "terminal": "gnome-terminal",
    "calculator": "gnome-calculator",
    "text editor": "gedit",
    "vscode": "code",
    "visual studio code": "code",
    "settings": "gnome-control-center",
    "software center": "gnome-software",
    "screenshot": "gnome-screenshot",
}


def _application_dirs() -> Iterator[Path]:
    home_data = os.getenv("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
    data_dirs = os.getenv("XDG_DATA_DIRS", "/usr/local/share:/usr/share").split(":")
    seen = set()
    for base in [home_data, *data_dirs]:
        directory = Path(base) / "applications"
        if directory.is_dir() and directory not in seen:
            seen.add(directory)
            yield directory


def _read_desktop_entry(path: Path) -> Dict[str, str] | None:
    parser = configparser.RawConfigParser(strict=False, interpolation=None)
    try:
        parser.read(path, encoding="utf-8")
    except Exception:
        return None
    if not parser.has_section("Desktop Entry"):
        return None
    return dict(parser["Desktop Entry"])


def scan_apps(register: Callable[[str, Dict[str, str]], None]) -> Dict[str, int]:
    found = 0
    for directory in _application_dirs():
        try:
            candidates = sorted(directory.glob("*.desktop"))
        except Exception:
            continue
        for candidate in candidates:
            entry = _read_desktop_entry(candidate)
            if entry is None:
                continue
            if entry.get("nodisplay", "").strip().lower() == "true":
                continue
            if entry.get("hidden", "").strip().lower() == "true":
                continue
            name = entry.get("name", "").strip() or candidate.stem
            register(name, {"kind": "desktop", "path": str(candidate), "source": "scan"})
            found += 1
    return {"desktop_entries_scanned": found}


def find_executable(app_name: str) -> str | None:
    needle = app_name.strip().lower()
    for candidate in (needle, needle.replace(" ", "-"), needle.replace(" ", "")):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _desktop_exec_command(path: Path) -> list[str] | None:
    entry = _read_desktop_entry(path)
    if entry is None:
        return None
    exec_line = entry.get("exec", "").strip()
    if not exec_line:
        return None
    try:
        parts = shlex.split(exec_line)
    except ValueError:
        return None
    # Strip freedesktop field codes (%f, %U, ...) — we launch with no arguments.
    command = [part for part in parts if not re.fullmatch(r"%[a-zA-Z]", part)]
    return command or None


def launch_target(target_path: str) -> Tuple[bool, str]:
    target = str(target_path or "").strip()
    if not target:
        return False, "empty launch target"

    errors: list[str] = []

    if target.endswith(".desktop") and Path(target).exists():
        desktop_path = Path(target)
        if shutil.which("gtk-launch"):
            try:
                subprocess.Popen(["gtk-launch", desktop_path.stem])
                return True, "gtk-launch"
            except Exception as exc:
                errors.append(f"gtk-launch: {exc}")
        command = _desktop_exec_command(desktop_path)
        if command:
            try:
                subprocess.Popen(command)
                return True, "desktop-exec"
            except Exception as exc:
                errors.append(f"desktop-exec: {exc}")
        return False, "; ".join(errors) or "unlaunchable desktop entry"

    if os.path.isabs(target) and Path(target).exists():
        try:
            open_path(target)
            return True, "xdg-open"
        except Exception as exc:
            errors.append(f"xdg-open: {exc}")

    resolved = shutil.which(target) or find_executable(target)
    if resolved:
        try:
            subprocess.Popen([resolved])
            return True, "which"
        except Exception as exc:
            errors.append(f"exec: {exc}")

    return False, "; ".join(errors) or "no matching command or desktop entry"


# --- window management (X11 via wmctrl) -----------------------------------------------

def _require_wmctrl() -> None:
    if os.getenv("XDG_SESSION_TYPE", "").lower() == "wayland" and os.getenv("DISPLAY") is None:
        raise RuntimeError("window switching is not supported on Wayland sessions")
    _require("wmctrl")


def list_windows() -> List[Tuple[str, Any]]:
    _require_wmctrl()
    ok, out = _run_command(["wmctrl", "-l"])
    if not ok:
        raise RuntimeError(out or "wmctrl failed")
    windows: List[Tuple[str, Any]] = []
    for line in out.splitlines():
        # Format: "0x04600003 0 hostname Window Title"
        parts = line.split(None, 3)
        if len(parts) == 4 and parts[3].strip():
            windows.append((parts[3].strip(), parts[0]))
    return windows


def find_windows_by_title(title: str) -> List[Tuple[str, Any]]:
    needle = title.strip().lower()
    return [(win_title, handle) for win_title, handle in list_windows() if needle in win_title.lower()]


def activate_window(handle: Any) -> None:
    _require_wmctrl()
    ok, out = _run_command(["wmctrl", "-i", "-a", str(handle)])
    if not ok:
        raise RuntimeError(out or "could not activate window")

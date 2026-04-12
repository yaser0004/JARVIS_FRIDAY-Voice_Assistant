from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Any, Dict, Optional

import screen_brightness_control as sbc
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


_PENDING_POWER_CONFIRMATION: Optional[str] = None


def _response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


def _get_volume_interface():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    return interface.QueryInterface(IAudioEndpointVolume)


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


def _normalize_level(level: int | str, for_brightness: bool = False) -> int | str:
    if isinstance(level, int):
        return max(0, min(100, level))

    value = str(level).strip().lower()
    mapping = {
        "max": 100,
        "min": 10 if for_brightness else 0,
        "half": 50,
        "zero": 0,
        "mute": "mute",
        "unmute": "unmute",
    }
    if value.endswith("%") and value[:-1].isdigit():
        return max(0, min(100, int(value[:-1])))
    if value.isdigit():
        return max(0, min(100, int(value)))
    return mapping.get(value, 50)


def get_volume() -> Dict[str, Any]:
    try:
        volume = _get_volume_interface()
        current = int(round(volume.GetMasterVolumeLevelScalar() * 100))
        muted = bool(volume.GetMute())
        return _response(True, f"Current volume is {current} percent.", {"level": current, "muted": muted})
    except Exception as exc:
        return _response(False, f"I could not read volume: {exc}", {"error": str(exc)})


def set_volume(level: int | str, direction: str = None, step: int = 10) -> Dict[str, Any]:
    try:
        volume = _get_volume_interface()
        current = int(round(volume.GetMasterVolumeLevelScalar() * 100))
        delta = max(1, min(50, int(step or 10)))

        if direction == "up":
            target = min(100, current + delta)
        elif direction == "down":
            target = max(0, current - delta)
        else:
            parsed = _normalize_level(level)
            if parsed == "mute":
                volume.SetMute(1, None)
                return _response(True, "Volume muted.", {"level": 0, "muted": True})
            if parsed == "unmute":
                volume.SetMute(0, None)
                return _response(True, "Volume unmuted.", {"muted": False})
            target = int(parsed)

        volume.SetMasterVolumeLevelScalar(target / 100.0, None)
        volume.SetMute(0, None)
        return _response(True, f"Volume set to {target} percent.", {"level": target, "step": delta})
    except Exception as exc:
        return _response(False, f"I could not change volume: {exc}", {"error": str(exc)})


def get_brightness() -> Dict[str, Any]:
    try:
        values = sbc.get_brightness()
        if isinstance(values, list) and values:
            avg = int(round(sum(int(v) for v in values) / len(values)))
        else:
            avg = int(values)
        return _response(True, f"Current brightness is {avg} percent.", {"level": avg})
    except Exception as exc:
        return _response(False, f"I could not read brightness: {exc}", {"error": str(exc)})


def set_brightness(level: int | str, direction: str = None, step: int = 10) -> Dict[str, Any]:
    try:
        if direction in {"up", "down"}:
            current = get_brightness()
            if not current.get("success"):
                return current
            current_level = int(current.get("data", {}).get("level", 50))
            delta = max(1, min(50, int(step or 10)))
            target = min(100, current_level + delta) if direction == "up" else max(0, current_level - delta)
        else:
            parsed = _normalize_level(level, for_brightness=True)
            if isinstance(parsed, str):
                parsed = 50
            target = int(parsed)

        sbc.set_brightness(target)

        actual_level: int | None = None
        actual = get_brightness()
        if actual.get("success"):
            try:
                actual_level = int(actual.get("data", {}).get("level"))
            except Exception:
                actual_level = None

        if actual_level is None:
            return _response(True, f"Brightness set to {target} percent.", {"level": target, "requested_level": target})

        if abs(actual_level - target) <= 1:
            return _response(
                True,
                f"Brightness set to {actual_level} percent.",
                {"level": actual_level, "requested_level": target},
            )

        return _response(
            True,
            f"Brightness set to {actual_level} percent (requested {target} percent).",
            {"level": actual_level, "requested_level": target},
        )
    except Exception as exc:
        return _response(False, f"I could not change brightness: {exc}", {"error": str(exc)})


def _toggle_adapter(enable: bool, adapter_regex: str, label: str) -> Dict[str, Any]:
    action = "Enable" if enable else "Disable"
    command = (
        f"$adapters = Get-NetAdapter | Where-Object {{$_.Status -ne 'Unknown' -and $_.Name -match '{adapter_regex}'}}; "
        "if (-not $adapters) { exit 2 }; "
        f"$adapters | {action}-NetAdapter -Confirm:$false"
    )
    ok, output = _run_powershell(command)
    if ok:
        state = "enabled" if enable else "disabled"
        return _response(True, f"{label} {state}.")

    # Typical failure is missing admin privilege. Keep graceful fallback.
    try:
        os.startfile("ms-settings:network")
    except Exception:
        pass
    return _response(False, f"I could not toggle {label.lower()} automatically. Opened network settings instead.", {"error": output})


def toggle_wifi(enable: bool) -> Dict[str, Any]:
    return _toggle_adapter(enable, "Wi[- ]?Fi|WLAN|Wireless", "Wi-Fi")


def toggle_bluetooth(enable: bool) -> Dict[str, Any]:
    # Bluetooth toggling may require elevated privileges and varies by driver.
    service_action = "Start-Service" if enable else "Stop-Service"
    command = f"{service_action} bthserv -ErrorAction Stop"
    ok, output = _run_powershell(command)
    if ok:
        state = "enabled" if enable else "disabled"
        return _response(True, f"Bluetooth {state}.")

    try:
        os.startfile("ms-settings:bluetooth")
    except Exception:
        pass
    return _response(False, "I could not toggle Bluetooth automatically. Opened Bluetooth settings instead.", {"error": output})


def toggle_airplane_mode(enable: bool) -> Dict[str, Any]:
    # Windows does not provide a stable non-elevated CLI for airplane mode across builds.
    try:
        os.startfile("ms-settings:network-airplanemode")
    except Exception as exc:
        return _response(False, f"I could not open airplane mode settings: {exc}", {"error": str(exc)})

    desired = "on" if enable else "off"
    return _response(
        True,
        f"Opened airplane mode settings. Please switch airplane mode {desired} there.",
        {"requires_manual_toggle": True, "target_state": desired},
    )


def toggle_battery_saver(enable: bool) -> Dict[str, Any]:
    # Battery saver state is not consistently scriptable without privileged APIs.
    try:
        os.startfile("ms-settings:batterysaver")
    except Exception as exc:
        return _response(False, f"I could not open battery saver settings: {exc}", {"error": str(exc)})

    desired = "on" if enable else "off"
    return _response(
        True,
        f"Opened battery settings. Please switch battery saver {desired} there.",
        {"requires_manual_toggle": True, "target_state": desired},
    )


def power_action(command: str) -> Dict[str, Any]:
    global _PENDING_POWER_CONFIRMATION

    cmd = (command or "").strip().lower().replace(" ", "_")
    dangerous = {"shutdown", "restart", "hibernate"}

    try:
        if cmd in dangerous:
            if _PENDING_POWER_CONFIRMATION != cmd:
                _PENDING_POWER_CONFIRMATION = cmd
                return _response(
                    True,
                    f"Please confirm {cmd} by repeating the command once more.",
                    {"requires_confirmation": True, "command": cmd},
                )
            _PENDING_POWER_CONFIRMATION = None

        if cmd == "shutdown":
            subprocess.run(["shutdown", "/s", "/t", "5"], check=False)
            return _response(True, "Shutting down in 5 seconds.")
        if cmd == "restart":
            subprocess.run(["shutdown", "/r", "/t", "5"], check=False)
            return _response(True, "Restarting in 5 seconds.")
        if cmd == "sleep":
            subprocess.run(
                ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
                check=False,
            )
            return _response(True, "Putting the system to sleep.")
        if cmd == "hibernate":
            subprocess.run(["shutdown", "/h"], check=False)
            return _response(True, "Hibernating now.")
        if cmd == "lock":
            ctypes.windll.user32.LockWorkStation()
            return _response(True, "Locking your workstation.")
        if cmd == "turn_off_monitor":
            ctypes.windll.user32.SendMessageW(0xFFFF, 0x0112, 0xF170, 2)
            return _response(True, "Turning off monitor.")
        if cmd == "monitor_off":
            ctypes.windll.user32.SendMessageW(0xFFFF, 0x0112, 0xF170, 2)
            return _response(True, "Turning off monitor.")

        if cmd.startswith("wifi_"):
            return toggle_wifi(cmd.endswith("on"))
        if cmd.startswith("bluetooth_"):
            return toggle_bluetooth(cmd.endswith("on"))
        if cmd.startswith("airplane_"):
            return toggle_airplane_mode(cmd.endswith("on"))
        if cmd.startswith("battery_saver_"):
            return toggle_battery_saver(cmd.endswith("on"))

        return _response(False, "Unknown power command.", {"command": cmd})
    except Exception as exc:
        return _response(False, f"I could not perform power action: {exc}", {"error": str(exc)})


def open_settings(setting_name: str) -> Dict[str, Any]:
    try:
        mapping = {
            "wifi": "network-wifi",
            "bluetooth": "bluetooth",
            "airplane": "network-airplanemode",
            "display": "display",
            "sound": "sound",
            "apps": "appsfeatures",
            "updates": "windowsupdate",
            "privacy": "privacy",
            "battery": "batterysaver",
        }
        key = (setting_name or "display").strip().lower()
        page = mapping.get(key, key)
        os.startfile(f"ms-settings:{page}")
        return _response(True, f"Opened {key} settings.", {"page": page})
    except Exception as exc:
        return _response(False, f"I could not open settings: {exc}", {"error": str(exc)})

from __future__ import annotations

import platform
import re
import socket
import subprocess
from typing import Any, Dict

import psutil


def _response(success: bool, text: str, data: Any = None) -> Dict[str, Any]:
    return {"success": success, "response_text": text, "data": data}


def _run_powershell(command: str, timeout: int = 8) -> str:
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        out = (result.stdout or "").strip()
        return out
    except Exception:
        return ""


def current_wifi_name() -> str | None:
    out = _run_powershell("netsh wlan show interfaces")
    if not out:
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
    out = _run_powershell(cmd)
    if not out:
        return None
    names = [line.strip() for line in out.splitlines() if line.strip()]
    if not names:
        return None
    # Keep the first user-facing bluetooth device/adapter name.
    return names[0]


def battery_status() -> Dict[str, Any]:
    info = psutil.sensors_battery()
    if info is None:
        return {"available": False, "percent": None, "plugged": None, "seconds_left": None}

    return {
        "available": True,
        "percent": float(info.percent),
        "plugged": bool(info.power_plugged),
        "seconds_left": int(info.secsleft) if info.secsleft is not None else None,
    }


def system_specs() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()}",
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Unknown",
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(vm.total / (1024 ** 3), 2),
        "available_ram_gb": round(vm.available / (1024 ** 3), 2),
    }


def get_system_awareness() -> Dict[str, Any]:
    wifi = current_wifi_name()
    bt = current_bluetooth_name()
    battery = battery_status()
    specs = system_specs()

    battery_text = "Battery unavailable"
    if battery["available"]:
        battery_text = (
            f"Battery: {battery['percent']:.0f}% ({'Plugged in' if battery['plugged'] else 'On battery'})"
        )

    response_lines = [
        f"Wi-Fi: {wifi or 'Not connected'}",
        f"Bluetooth: {bt or 'Not connected'}",
        battery_text,
        f"System: {specs['os']} on {specs['machine']}",
        f"CPU: {specs['processor']} ({specs['cpu_logical_cores']} logical cores)",
        f"RAM: {specs['available_ram_gb']} GB free / {specs['total_ram_gb']} GB total",
    ]

    return _response(True, "\n".join(response_lines), {
        "wifi": wifi,
        "bluetooth": bt,
        "battery": battery,
        "specs": specs,
    })


def looks_like_system_info_query(text: str) -> bool:
    normalized = text.lower().strip()
    if any(token in normalized for token in ["open ", "enable ", "disable ", "turn on", "turn off", "settings"]):
        return False

    if re.search(r"\b(system status|system info|pc status|pc info|device status)\b", normalized):
        return True

    if any(token in normalized for token in ["battery", "plugged", "charging", "battery percent", "battery status"]):
        return True

    if any(token in normalized for token in ["specs", "specifications", "hardware", "ram", "processor", "cpu"]):
        return True

    asks_for_network_info = any(token in normalized for token in ["what", "which", "current", "connected", "name", "status"])
    if asks_for_network_info and any(token in normalized for token in ["wifi", "wi-fi", "wireless", "ssid", "bluetooth"]):
        return True

    return False

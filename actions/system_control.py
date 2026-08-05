from __future__ import annotations

import time
from typing import Any, Dict, Optional

from actions.common import response as _response
from platform_adapters import backend


# (command, monotonic timestamp) — a pending confirmation goes stale after the
# TTL so an old "shutdown" can't be confirmed by an unrelated command minutes later.
_PENDING_POWER_CONFIRMATION: Optional[tuple[str, float]] = None
_POWER_CONFIRMATION_TTL_S = 60.0


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
        current, muted = backend.get_volume()
        return _response(True, f"Current volume is {current} percent.", {"level": current, "muted": muted})
    except Exception as exc:
        return _response(False, f"I could not read volume: {exc}", {"error": str(exc)})


def set_volume(level: int | str, direction: str = None, step: int = 10) -> Dict[str, Any]:
    try:
        current, _ = backend.get_volume()
        delta = max(1, min(50, int(step or 10)))

        if direction == "up":
            target = min(100, current + delta)
        elif direction == "down":
            target = max(0, current - delta)
        else:
            parsed = _normalize_level(level)
            if parsed == "mute":
                backend.set_mute(True)
                return _response(True, "Volume muted.", {"level": 0, "muted": True})
            if parsed == "unmute":
                backend.set_mute(False)
                return _response(True, "Volume unmuted.", {"muted": False})
            target = int(parsed)

        backend.set_volume_level(target)
        return _response(True, f"Volume set to {target} percent.", {"level": target, "step": delta})
    except Exception as exc:
        return _response(False, f"I could not change volume: {exc}", {"error": str(exc)})


def get_brightness() -> Dict[str, Any]:
    try:
        avg = backend.get_brightness()
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

        backend.set_brightness(target)

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


def _open_settings_quietly(key: str) -> None:
    try:
        backend.open_settings_page(key)
    except Exception:
        pass


def toggle_wifi(enable: bool) -> Dict[str, Any]:
    ok, output = backend.toggle_wifi(enable)
    if ok:
        state = "enabled" if enable else "disabled"
        return _response(True, f"Wi-Fi {state}.")

    # Typical failure is missing privilege or tooling. Keep graceful fallback.
    _open_settings_quietly("network")
    return _response(False, "I could not toggle wi-fi automatically. Opened network settings instead.", {"error": output})


def toggle_bluetooth(enable: bool) -> Dict[str, Any]:
    ok, output = backend.toggle_bluetooth(enable)
    if ok:
        state = "enabled" if enable else "disabled"
        return _response(True, f"Bluetooth {state}.")

    _open_settings_quietly("bluetooth")
    return _response(False, "I could not toggle Bluetooth automatically. Opened Bluetooth settings instead.", {"error": output})


def toggle_airplane_mode(enable: bool) -> Dict[str, Any]:
    # No stable non-elevated CLI for airplane mode on either OS; open the settings page.
    try:
        backend.open_settings_page("airplane")
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
        backend.open_settings_page("battery")
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
            pending = _PENDING_POWER_CONFIRMATION
            if pending is not None and time.monotonic() - pending[1] > _POWER_CONFIRMATION_TTL_S:
                pending = None
            if pending is None or pending[0] != cmd:
                _PENDING_POWER_CONFIRMATION = (cmd, time.monotonic())
                return _response(
                    True,
                    f"Please confirm {cmd} by repeating the command once more.",
                    {"requires_confirmation": True, "command": cmd},
                )
            _PENDING_POWER_CONFIRMATION = None

        if cmd == "shutdown":
            backend.shutdown()
            return _response(True, "Shutting down now.")
        if cmd == "restart":
            backend.restart()
            return _response(True, "Restarting now.")
        if cmd == "sleep":
            backend.sleep()
            return _response(True, "Putting the system to sleep.")
        if cmd == "hibernate":
            backend.hibernate()
            return _response(True, "Hibernating now.")
        if cmd == "lock":
            backend.lock()
            return _response(True, "Locking your workstation.")
        if cmd in {"turn_off_monitor", "monitor_off"}:
            backend.monitor_off()
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
        key = (setting_name or "display").strip().lower()
        page = backend.open_settings_page(key)
        return _response(True, f"Opened {key} settings.", {"page": page})
    except Exception as exc:
        return _response(False, f"I could not open settings: {exc}", {"error": str(exc)})

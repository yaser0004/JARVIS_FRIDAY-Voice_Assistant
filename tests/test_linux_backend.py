"""Linux backend builds the right argv and degrades cleanly when tools are absent."""

import subprocess

import pytest

from platform_adapters import linux


@pytest.fixture
def calls(monkeypatch):
    """Record argv and pretend every tool exists and succeeds."""
    recorded = []

    def fake_run(cmd, **kwargs):
        recorded.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    def fake_popen(cmd, **kwargs):
        recorded.append(cmd)
        return None

    monkeypatch.setattr(linux.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(linux.subprocess, "run", fake_run)
    monkeypatch.setattr(linux.subprocess, "Popen", fake_popen)
    return recorded


def test_set_volume_uses_pactl(calls):
    linux.set_volume_level(55)
    assert calls[0] == ["pactl", "set-sink-volume", "@DEFAULT_SINK@", "55%"]


def test_set_mute_uses_pactl(calls):
    linux.set_mute(True)
    assert calls[0] == ["pactl", "set-mute", "@DEFAULT_SINK@", "1"]


def test_set_brightness_uses_brightnessctl(calls):
    linux.set_brightness(30)
    assert calls[0] == ["brightnessctl", "set", "30%"]


@pytest.mark.parametrize(
    "fn,expected",
    [
        ("shutdown", ["systemctl", "poweroff"]),
        ("restart", ["systemctl", "reboot"]),
        ("sleep", ["systemctl", "suspend"]),
        ("hibernate", ["systemctl", "hibernate"]),
        ("lock", ["loginctl", "lock-session"]),
    ],
)
def test_power_commands(calls, fn, expected):
    getattr(linux, fn)()
    assert calls[0] == expected


def test_open_path_uses_xdg_open(calls):
    linux.open_path("/home/user/notes.txt")
    assert calls[0] == ["xdg-open", "/home/user/notes.txt"]


def test_toggle_wifi_uses_nmcli(calls):
    ok, _ = linux.toggle_wifi(False)
    assert ok is True
    assert calls[0] == ["nmcli", "radio", "wifi", "off"]


def test_volume_parsing(monkeypatch):
    monkeypatch.setattr(linux.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    outputs = {
        "get-sink-volume": "Volume: front-left: 45874 /  70% / -9.29 dB",
        "get-mute": "Mute: no",
    }

    def fake_run(cmd, **kwargs):
        key = next((k for k in outputs if k in cmd), None)
        return subprocess.CompletedProcess(cmd, 0, stdout=outputs.get(key, ""), stderr="")

    monkeypatch.setattr(linux.subprocess, "run", fake_run)
    assert linux.get_volume() == (70, False)


def test_brightness_parsing(monkeypatch):
    monkeypatch.setattr(linux.shutil, "which", lambda tool: "/usr/bin/brightnessctl")
    monkeypatch.setattr(
        linux.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, stdout="intel_backlight,backlight,48000,50%,96000", stderr=""
        ),
    )
    assert linux.get_brightness() == 50


# --- graceful degradation when the tool is missing ---------------------------

def test_missing_pactl_raises_with_package_hint(monkeypatch):
    monkeypatch.setattr(linux.shutil, "which", lambda tool: None)
    with pytest.raises(RuntimeError) as exc:
        linux.set_volume_level(50)
    assert "pactl" in str(exc.value)


def test_missing_brightnessctl_raises(monkeypatch):
    monkeypatch.setattr(linux.shutil, "which", lambda tool: None)
    with pytest.raises(RuntimeError):
        linux.get_brightness()


def test_missing_nmcli_returns_failure_not_exception(monkeypatch):
    monkeypatch.setattr(linux.shutil, "which", lambda tool: None)
    ok, message = linux.toggle_wifi(True)
    assert ok is False
    assert "nmcli" in message


def test_missing_nmcli_wifi_name_is_none(monkeypatch):
    monkeypatch.setattr(linux.shutil, "which", lambda tool: None)
    assert linux.current_wifi_name() is None


def test_window_switching_unsupported_on_wayland(monkeypatch):
    monkeypatch.setenv("XDG_SESSION_TYPE", "wayland")
    monkeypatch.delenv("DISPLAY", raising=False)
    with pytest.raises(RuntimeError) as exc:
        linux.list_windows()
    assert "wayland" in str(exc.value).lower()


def test_desktop_entry_exec_strips_field_codes(tmp_path, monkeypatch):
    desktop = tmp_path / "demo.desktop"
    desktop.write_text("[Desktop Entry]\nName=Demo\nExec=demo-app --flag %U\nType=Application\n")
    monkeypatch.setattr(linux.shutil, "which", lambda tool: None)
    assert linux._desktop_exec_command(desktop) == ["demo-app", "--flag"]


def test_scan_apps_skips_nodisplay(tmp_path, monkeypatch):
    apps = tmp_path / "applications"
    apps.mkdir()
    (apps / "visible.desktop").write_text("[Desktop Entry]\nName=Visible\nExec=vis\n")
    (apps / "hidden.desktop").write_text("[Desktop Entry]\nName=Hidden\nExec=hid\nNoDisplay=true\n")
    monkeypatch.setattr(linux, "_application_dirs", lambda: iter([apps]))

    registered = {}
    stats = linux.scan_apps(lambda alias, record: registered.__setitem__(alias, record))
    assert "Visible" in registered
    assert "Hidden" not in registered
    assert stats["desktop_entries_scanned"] == 1

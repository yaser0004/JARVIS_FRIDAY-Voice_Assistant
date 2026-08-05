"""close_app must match process identity, never a substring of the command line.

The old matcher accepted any >=4-char token appearing anywhere in
name+exe+cmdline, so "close code" terminated every process whose command line
merely mentioned "code".
"""

import pytest

from actions import app_control


class FakeProc:
    def __init__(self, name, exe="", cmdline=()):
        self.info = {"name": name, "pid": 1234, "exe": exe, "cmdline": list(cmdline)}
        self.terminated = False

    def terminate(self):
        self.terminated = True


@pytest.fixture
def procs(monkeypatch):
    processes = [
        FakeProc("Code.exe", exe="C:/Users/x/AppData/Local/Programs/Code.exe"),
        FakeProc("python3", exe="/usr/bin/python3", cmdline=["python3", "-m", "pycodestyle", "code_review.py"]),
        FakeProc("firefox", exe="/usr/lib/firefox/firefox"),
        FakeProc("node", exe="/usr/bin/node", cmdline=["node", "vscode-server/out/server-main.js"]),
    ]
    monkeypatch.setattr(app_control.psutil, "process_iter", lambda attrs=None: iter(processes))
    monkeypatch.setattr(app_control, "_select_app_record", lambda name: None)
    monkeypatch.setattr(app_control.backend, "DEFAULT_APP_MAP", {}, raising=False)
    return processes


def test_matches_the_real_process(procs):
    result = app_control.close_app("code")
    assert result["success"] is True
    assert procs[0].terminated is True


def test_does_not_kill_unrelated_cmdline_matches(procs):
    app_control.close_app("code")
    assert procs[1].terminated is False, "a process merely mentioning 'code' must survive"
    assert procs[3].terminated is False, "vscode-server in argv is not the target app"


def test_unknown_app_reports_failure(procs):
    result = app_control.close_app("nonexistentapp")
    assert result["success"] is False
    assert not any(p.terminated for p in procs)


def test_blank_target_is_rejected(procs):
    result = app_control.close_app("   ")
    assert result["success"] is False
    assert not any(p.terminated for p in procs)

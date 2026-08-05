"""Destructive power commands need a fresh confirmation.

The pending-confirmation state used to live forever, so a "shutdown" uttered
once could be completed by an unrelated command minutes later.
"""

import pytest

from actions import system_control


@pytest.fixture(autouse=True)
def clear_pending():
    system_control._PENDING_POWER_CONFIRMATION = None
    yield
    system_control._PENDING_POWER_CONFIRMATION = None


@pytest.fixture
def executed(monkeypatch):
    calls = []
    for name in ("shutdown", "restart", "hibernate", "sleep", "lock"):
        monkeypatch.setattr(
            system_control.backend, name, lambda n=name: calls.append(n), raising=False
        )
    return calls


def test_first_shutdown_only_asks_for_confirmation(executed):
    result = system_control.power_action("shutdown")
    assert result["data"]["requires_confirmation"] is True
    assert executed == []


def test_repeat_within_window_executes(executed):
    system_control.power_action("shutdown")
    result = system_control.power_action("shutdown")
    assert result["success"] is True
    assert executed == ["shutdown"]


def test_confirmation_expires(monkeypatch, executed):
    clock = {"now": 1000.0}
    monkeypatch.setattr(system_control.time, "monotonic", lambda: clock["now"])

    system_control.power_action("shutdown")
    clock["now"] += system_control._POWER_CONFIRMATION_TTL_S + 1

    result = system_control.power_action("shutdown")
    assert result["data"]["requires_confirmation"] is True, "stale confirmation must not execute"
    assert executed == []


def test_different_command_does_not_confirm_previous(executed):
    system_control.power_action("shutdown")
    result = system_control.power_action("restart")
    assert result["data"]["requires_confirmation"] is True
    assert executed == []


def test_non_destructive_command_runs_immediately(executed):
    result = system_control.power_action("lock")
    assert result["success"] is True
    assert executed == ["lock"]


def test_sleep_needs_no_confirmation(executed):
    result = system_control.power_action("sleep")
    assert result["success"] is True
    assert executed == ["sleep"]

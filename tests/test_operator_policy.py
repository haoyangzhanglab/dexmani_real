"""Operator batch semantics: C/D must not fence H, S/Q must.

Offline, no hardware, no pynput.  ``run_operator_control`` drains a keyboard
batch and translates it to shared flags / home.  The policy deployment treats
C/D (PAUSE/DISCARD) as teleop-only no-ops, so a C or D in the same drained
batch must not suppress Home; S/Q must still suppress it.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import mock

import pytest

import dexmani_real.deployment.operator as operator
from dexmani_real.runtime.operator_input import OperatorCommand
from dexmani_real.runtime.safety import SafetyState, StopRequest


class _FakeKeyboardInput:
    """KeyboardInput double: one scripted poll, no pynput session."""

    def __init__(self, scripted):
        self.estop_latched = False
        self.healthy = True
        self._scripted = list(scripted)
        self.stop_event: threading.Event | None = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None

    def drain_signal(self, command: OperatorCommand) -> None:
        del command

    def poll(self, timeout: float = 0.0) -> list[OperatorCommand]:
        del timeout
        signals = list(self._scripted)
        self._scripted = []
        if self.stop_event is not None:
            self.stop_event.set()
        return signals


def _shared() -> SimpleNamespace:
    shared = SimpleNamespace(motion_lock=threading.RLock())
    for name, value in dict(
        is_running=True,
        error_state=False,
        estop_request=False,
        quit_requested=False,
        start_request=False,
        stop_request=int(StopRequest.NONE),
        safety_state=int(SafetyState.ARMED),
        physical_home_completed=False,
    ).items():
        setattr(shared, name, SimpleNamespace(value=value))
    return shared


def _run(monkeypatch: pytest.MonkeyPatch, signals) -> mock.Mock:
    stop_event = threading.Event()
    fake = _FakeKeyboardInput(signals)
    fake.stop_event = stop_event
    home = mock.Mock(return_value=True)
    monkeypatch.setattr(operator, "KeyboardInput", lambda **kw: fake)
    monkeypatch.setattr(operator, "_home", home)
    monkeypatch.setattr(operator, "_request_immediate_stop", mock.Mock())
    monkeypatch.setattr(operator, "_request_immediate_quit", mock.Mock())
    operator.run_operator_control(
        _shared(), mock.Mock(), mock.Mock(), stop_event=stop_event, execute=True
    )
    return home


@pytest.mark.parametrize("noop", [OperatorCommand.PAUSE, OperatorCommand.DISCARD])
def test_noop_command_does_not_block_home(monkeypatch, noop):
    home = _run(monkeypatch, [noop, OperatorCommand.HOME])
    home.assert_called_once()


@pytest.mark.parametrize("fence", [OperatorCommand.STOP, OperatorCommand.QUIT])
def test_control_command_blocks_home_in_same_batch(monkeypatch, fence):
    # Home ordered first proves the suppression is batch-wide, not a S/Q
    # ``return`` racing ahead of H.
    home = _run(monkeypatch, [OperatorCommand.HOME, fence])
    home.assert_not_called()

"""Offline characterization for operator keyboard event and hold semantics."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from dexmani_real.runtime.operator_input import (
    KeyboardInput,
    KeyboardState,
    OperatorCommand,
)


class _FakeListener:
    """Small pynput listener double that never touches a graphical session."""

    latest: "_FakeListener | None" = None

    def __init__(self, *, on_press, on_release, **_kwargs) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.alive = True
        _FakeListener.latest = self

    def start(self) -> None:
        return None

    def wait(self) -> None:
        return None

    def is_alive(self) -> bool:
        return self.alive

    def stop(self) -> None:
        self.alive = False

    def join(self, timeout: float | None = None) -> None:
        del timeout


class _FakeKeyboard:
    class Key:
        esc = object()
        up = object()
        down = object()
        left = object()
        right = object()
        space = object()
        enter = object()
        backspace = object()


def _char(value: str) -> SimpleNamespace:
    return SimpleNamespace(char=value)


@pytest.fixture
def fake_pynput(monkeypatch: pytest.MonkeyPatch):
    module = ModuleType("pynput")
    module.keyboard = SimpleNamespace(
        Listener=_FakeListener,
        Key=_FakeKeyboard.Key,
    )
    monkeypatch.setitem(sys.modules, "pynput", module)
    monkeypatch.setenv("DISPLAY", ":offline-test")
    _FakeListener.latest = None
    return module


def test_keyboard_input_suppresses_os_repeat_until_release(fake_pynput) -> None:
    del fake_pynput
    keyboard = KeyboardInput()
    keyboard.start()
    try:
        listener = _FakeListener.latest
        assert listener is not None

        listener.on_press(_char("b"))
        listener.on_press(_char("b"))
        assert keyboard.poll(0.0) == [OperatorCommand.BEGIN]

        listener.on_release(_char("b"))
        listener.on_press(_char("b"))
        assert keyboard.poll(0.0) == [OperatorCommand.BEGIN]
    finally:
        keyboard.stop()


def test_keyboard_input_esc_is_sticky_and_one_shot(fake_pynput) -> None:
    del fake_pynput
    estop = mock.Mock()
    keyboard = KeyboardInput(estop_callback=estop)
    keyboard.start()
    try:
        listener = _FakeListener.latest
        assert listener is not None
        listener.on_press(_FakeKeyboard.Key.esc)
        listener.on_press(_FakeKeyboard.Key.esc)

        assert keyboard.poll(0.0) == [OperatorCommand.EMERGENCY_STOP]
        assert keyboard.estop_latched
        estop.assert_called_once_with()
    finally:
        keyboard.stop()


def test_keyboard_input_dead_listener_latches_emergency_stop(fake_pynput) -> None:
    del fake_pynput
    estop = mock.Mock()
    keyboard = KeyboardInput(estop_callback=estop)
    keyboard.start()
    try:
        listener = _FakeListener.latest
        assert listener is not None
        listener.alive = False

        assert not keyboard.healthy
        assert keyboard.poll(0.0) == [OperatorCommand.EMERGENCY_STOP]
        assert keyboard.estop_latched
        estop.assert_called_once_with()
    finally:
        keyboard.stop()


def test_keyboard_input_listener_start_failure_is_reported(fake_pynput) -> None:
    del fake_pynput

    class FailingListener(_FakeListener):
        def wait(self) -> None:
            raise RuntimeError("listener startup failed")

    sys.modules["pynput"].keyboard.Listener = FailingListener
    keyboard = KeyboardInput(startup_timeout_s=0.1)
    with pytest.raises(RuntimeError, match="listener failed during startup"):
        keyboard.start()
    assert not keyboard._running
    assert _FakeListener.latest is not None
    assert not _FakeListener.latest.alive


def test_keyboard_state_tracks_held_keys_and_repeatable_events() -> None:
    state = KeyboardState()
    on_press, on_release = state._callbacks(_FakeKeyboard)

    on_press(_char("w"))
    on_press(_char("w"))
    assert state.is_pressed("w")
    assert state.pressed_keys() == ("w",)

    for key in (
        _FakeKeyboard.Key.space,
        _FakeKeyboard.Key.enter,
        _FakeKeyboard.Key.backspace,
        _char("x"),
        _FakeKeyboard.Key.space,
    ):
        on_press(key)
    assert [state.pop_event() for _ in range(5)] == [
        "space",
        "enter",
        "backspace",
        "x",
        "space",
    ]
    assert state.pop_event() is None

    on_release(_char("w"))
    assert not state.is_pressed("w")


def test_keyboard_state_esc_latches_until_stop() -> None:
    estop = mock.Mock()
    state = KeyboardState(estop_callback=estop)
    on_press, on_release = state._callbacks(_FakeKeyboard)

    on_press(_FakeKeyboard.Key.esc)
    on_release(_FakeKeyboard.Key.esc)
    assert state.is_pressed("esc")
    assert state.pressed_keys() == ()
    estop.assert_called_once_with()

    state.stop()
    assert not state.is_pressed("esc")


def test_keyboard_state_dead_listener_clears_held_keys() -> None:
    state = KeyboardState()
    state._running = True
    state._listener = SimpleNamespace(is_alive=lambda: False)
    state._keys.add("w")

    assert not state.healthy
    assert not state.is_pressed("w")

"""Offline characterization for operator keyboard event and hold semantics."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest

from dexmani_real.runtime import operator_input
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand


class _FakeListener:
    """Small pynput listener double that never touches a graphical session."""

    latest: "_FakeListener | None" = None

    def __init__(self, *, on_press, on_release, **kwargs) -> None:
        self.on_press = on_press
        self.on_release = on_release
        self.kwargs = kwargs
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
def fake_pynput(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = ModuleType("pynput")
    module.keyboard = SimpleNamespace(
        Listener=_FakeListener,
        Key=_FakeKeyboard.Key,
    )
    monkeypatch.setitem(sys.modules, "pynput", module)
    monkeypatch.setenv("DISPLAY", ":offline-test")
    _FakeListener.latest = None
    return module


def test_keyboard_input_command_edges_preserve_all_control_dispositions(
    fake_pynput: ModuleType,
) -> None:
    del fake_pynput
    stop = mock.Mock()
    quit = mock.Mock()
    keyboard = KeyboardInput(stop_callback=stop, quit_callback=quit)
    keyboard.start()
    try:
        listener = _FakeListener.latest
        assert listener is not None
        controls = (
            ("b", OperatorCommand.BEGIN),
            ("c", OperatorCommand.PAUSE),
            ("s", OperatorCommand.STOP),
            ("d", OperatorCommand.DISCARD),
            ("h", OperatorCommand.HOME),
            ("q", OperatorCommand.QUIT),
        )
        for key, _ in controls:
            listener.on_press(_char(key))
            listener.on_press(_char(key))

        assert keyboard.poll(0.0) == [signal for _, signal in controls]
        stop.assert_called_once_with()
        quit.assert_called_once_with()

        for key, _ in controls:
            listener.on_release(_char(key))
            listener.on_press(_char(key))
        assert keyboard.poll(0.0) == [signal for _, signal in controls]
    finally:
        keyboard.stop()


def test_keyboard_input_held_keys_and_repeatable_raw_events(
    fake_pynput: ModuleType,
) -> None:
    del fake_pynput
    keyboard = KeyboardInput(capture_commands=False, capture_raw_events=True)
    keyboard.start()
    try:
        listener = _FakeListener.latest
        assert listener is not None
        held_chars = (
            "b",
            "c",
            "s",
            "d",
            "h",
            "q",
            "w",
            "a",
            "r",
            "i",
            "j",
            "k",
            "l",
        )
        held_special = (
            (_FakeKeyboard.Key.up, "up"),
            (_FakeKeyboard.Key.down, "down"),
            (_FakeKeyboard.Key.left, "left"),
            (_FakeKeyboard.Key.right, "right"),
        )
        for key in held_chars:
            listener.on_press(_char(key))
            listener.on_press(_char(key))
        for key, _ in held_special:
            listener.on_press(key)
            listener.on_press(key)

        expected_held = tuple(
            sorted((*held_chars, *(name for _, name in held_special)))
        )
        assert keyboard.pressed_keys() == expected_held
        assert keyboard.poll(0.0) == []

        for key in held_chars:
            listener.on_release(_char(key))
        for key, _ in held_special:
            listener.on_release(key)
        assert keyboard.pressed_keys() == ()

        event_keys = (
            (_char("x"), "x"),
            (_FakeKeyboard.Key.space, "space"),
            (_FakeKeyboard.Key.enter, "enter"),
            (_FakeKeyboard.Key.backspace, "backspace"),
        )
        for key, _ in event_keys:
            listener.on_press(key)
            listener.on_press(key)
            listener.on_release(key)
        assert [keyboard.pop_event() for _ in range(8)] == [
            "x",
            "x",
            "space",
            "space",
            "enter",
            "enter",
            "backspace",
            "backspace",
        ]
        assert keyboard.pop_event() is None
        assert all(not keyboard.is_pressed(name) for _, name in event_keys)
    finally:
        keyboard.stop()


def test_command_mode_does_not_queue_unread_raw_events(fake_pynput: ModuleType) -> None:
    del fake_pynput
    keyboard = KeyboardInput()
    keyboard.start()
    try:
        listener = _FakeListener.latest
        assert listener is not None
        for _ in range(3):
            listener.on_press(_char("x"))
            listener.on_press(_FakeKeyboard.Key.space)
            listener.on_press(_FakeKeyboard.Key.enter)
            listener.on_press(_FakeKeyboard.Key.backspace)

        assert keyboard.pop_event() is None
        assert not keyboard._events
    finally:
        keyboard.stop()


def test_keyboard_input_esc_is_sticky_and_one_shot(fake_pynput: ModuleType) -> None:
    del fake_pynput
    estop = mock.Mock()
    keyboard = KeyboardInput(estop_callback=estop)
    keyboard.start()
    try:
        listener = _FakeListener.latest
        assert listener is not None
        listener.on_press(_FakeKeyboard.Key.esc)
        listener.on_release(_FakeKeyboard.Key.esc)
        listener.on_press(_FakeKeyboard.Key.esc)

        assert keyboard.poll(0.0) == [OperatorCommand.EMERGENCY_STOP]
        assert keyboard.estop_latched
        assert keyboard.is_pressed("esc")
        estop.assert_called_once_with()
    finally:
        keyboard.stop()


def test_held_key_mode_repeats_legacy_estop_callback(fake_pynput: ModuleType) -> None:
    del fake_pynput
    estop = mock.Mock()
    keyboard = KeyboardInput(
        capture_commands=False,
        repeat_estop_callback=True,
        estop_callback=estop,
    )
    keyboard.start()
    try:
        listener = _FakeListener.latest
        assert listener is not None
        listener.on_press(_FakeKeyboard.Key.esc)
        listener.on_press(_FakeKeyboard.Key.esc)
        listener.on_release(_FakeKeyboard.Key.esc)

        assert keyboard.poll(0.0) == [OperatorCommand.EMERGENCY_STOP]
        assert keyboard.is_pressed("esc")
        assert keyboard.pressed_keys() == ()
        assert estop.call_count == 2
    finally:
        keyboard.stop()


def test_keyboard_input_dead_listener_clears_held_keys_and_latches_estop(
    fake_pynput: ModuleType,
) -> None:
    del fake_pynput
    estop = mock.Mock()
    keyboard = KeyboardInput(capture_commands=False, estop_callback=estop)
    keyboard.start()
    try:
        listener = _FakeListener.latest
        assert listener is not None
        listener.on_press(_char("w"))
        assert keyboard.is_pressed("w")
        listener.alive = False

        assert not keyboard.healthy
        assert not keyboard.is_pressed("w")
        assert keyboard.poll(0.0) == [OperatorCommand.EMERGENCY_STOP]
        assert keyboard.estop_latched
        estop.assert_called_once_with()
    finally:
        keyboard.stop()


def test_keyboard_input_quiesce_keeps_hold_tracking_and_clears_callbacks(
    fake_pynput: ModuleType,
) -> None:
    del fake_pynput
    estop = mock.Mock()
    keyboard = KeyboardInput(
        capture_commands=False,
        capture_raw_events=True,
        repeat_estop_callback=True,
        estop_callback=estop,
    )
    keyboard.start()
    try:
        listener = _FakeListener.latest
        assert listener is not None
        listener.on_press(_char("w"))
        listener.on_press(_FakeKeyboard.Key.space)
        keyboard.quiesce()

        assert keyboard.is_pressed("w")
        assert keyboard.pop_event() is None
        listener.on_press(_FakeKeyboard.Key.esc)
        assert keyboard.estop_latched
        estop.assert_not_called()
        assert not keyboard.wait_for_release(timeout_s=0.0)
        listener.on_release(_char("w"))
        listener.on_release(_FakeKeyboard.Key.esc)
        assert keyboard.wait_for_release(timeout_s=0.0)
    finally:
        keyboard.stop()


def test_keyboard_input_echo_lifecycle_and_listener_capture(
    fake_pynput, monkeypatch
) -> None:
    del fake_pynput
    saved = ["saved-terminal-attrs"]
    suppress = mock.Mock(return_value=saved)
    restore = mock.Mock()
    monkeypatch.setattr(operator_input, "_suppress_terminal_echo", suppress)
    monkeypatch.setattr(operator_input, "_restore_terminal_echo", restore)

    keyboard = KeyboardInput()
    keyboard.start()
    listener = _FakeListener.latest
    assert listener is not None
    assert listener.kwargs == {"suppress": False}
    keyboard.stop()
    suppress.assert_called_once_with()
    restore.assert_called_once_with(saved)

    unsuppressed = KeyboardInput(suppress_echo=False)
    unsuppressed.start()
    unsuppressed.stop()
    suppress.assert_called_once_with()
    restore.assert_called_once_with(saved)


def test_keyboard_input_listener_start_failure_is_reported(
    fake_pynput: ModuleType,
) -> None:
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

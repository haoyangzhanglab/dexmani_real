"""Bounded operator decisions for an active recording session."""

from __future__ import annotations

import time
from enum import Enum, auto
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.teleop.keyboard import ControlSignal, KeyboardHandler


class QuitRecordingDecision(Enum):
    SAVE = auto()
    DISCARD = auto()
    SAVE_AND_HOME = auto()
    ESTOP = auto()
    SHUTDOWN = auto()
    TIMEOUT = auto()


def await_quit_recording_decision(
    shared: SharedStorage,
    keyboard: KeyboardHandler,
    *,
    timeout_s: float,
) -> QuitRecordingDecision:
    """Wait for the bounded save/discard decision while keeping policy health live."""
    deadline_s = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline_s:
        if shared.estop_request.value:
            return QuitRecordingDecision.ESTOP
        shared.set_heartbeat("policy", time.monotonic())
        for signal in keyboard.poll(timeout=0.1):
            if signal is ControlSignal.STOP:
                return QuitRecordingDecision.SAVE
            if signal is ControlSignal.DISCARD:
                return QuitRecordingDecision.DISCARD
            if signal is ControlSignal.HOME:
                return QuitRecordingDecision.SAVE_AND_HOME
            if signal is ControlSignal.EMERGENCY_STOP:
                shared.estop_request.value = True
                return QuitRecordingDecision.ESTOP
        if shared.estop_request.value:
            return QuitRecordingDecision.ESTOP
        if not shared.is_running.value:
            return QuitRecordingDecision.SHUTDOWN
    return QuitRecordingDecision.TIMEOUT

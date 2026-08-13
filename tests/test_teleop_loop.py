"""Headless teleop_loop harness — drives the VR teleop coordinator with fakes.

The production ``teleop_loop`` is run in a daemon thread against a real
``SharedStorage``.  The operator/audio/SIGTERM seams are patched (see
``conftest.teleop_fakes``) and sensor/robot workers are replaced by direct ring
writes plus the readiness events the loop waits on at startup.  These tests pin
the coordinator's observable behavior — safety-state transitions, measured-hold
publication, and stale-feedback faulting — so the Phase-1.4 closure extraction
can be validated as zero-behavior-change.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from dexmani_real.robot.arm_loop import ArmLoopConfig
from dexmani_real.robot.safety import SafetyState
from dexmani_real.teleop.config import TeleopConfig
from dexmani_real.teleop.keyboard import ControlSignal
from dexmani_real.teleop.loop import teleop_loop

from tests.fakes.keyboard import FakeKeyboardHandler
from tests.fakes.workers import (
    drain_arm_action_q,
    write_arm_state,
    write_hand_state,
    write_vr_frame,
)
from tests.helpers import make_teleop_config, run_in_thread, stop_loop, wait_until

_HOME_QPOS = np.asarray(ArmLoopConfig().home_qpos, dtype=np.float64)


def _mark_ready(shared, cfg: TeleopConfig) -> None:
    """Set the readiness events teleop_loop waits on before entering its main loop."""
    shared.set_ready("arm")
    shared.set_ready("vr")
    if cfg.runtime.policy.hand_enabled:
        shared.set_ready("hand")
    if cfg.runtime.policy.recording_enabled:
        shared.set_ready("camera")
        shared.set_ready("recorder")


def _start_arm_writer(shared, qpos, interval_s: float = 0.02) -> tuple[threading.Event, threading.Thread]:
    """Write fresh arm state frames at a fixed rate, like a live arm worker."""
    stop = threading.Event()

    def _write_loop() -> None:
        while not stop.is_set():
            write_arm_state(shared, qpos=qpos)
            stop.wait(interval_s)

    thread = threading.Thread(target=_write_loop, daemon=True)
    thread.start()
    return stop, thread


def _start_hand_writer(shared, qpos, interval_s: float = 0.02) -> tuple[threading.Event, threading.Thread]:
    """Write fresh healthy hand state frames at a fixed rate."""
    stop = threading.Event()

    def _write_loop() -> None:
        while not stop.is_set():
            write_hand_state(shared, qpos=qpos)
            stop.wait(interval_s)

    thread = threading.Thread(target=_write_loop, daemon=True)
    thread.start()
    return stop, thread


def _start_vr_writer(shared, interval_s: float = 0.03) -> tuple[threading.Event, threading.Thread]:
    """Write fresh VR frames at a fixed rate, like a live HTS source."""
    stop = threading.Event()

    def _write_loop() -> None:
        while not stop.is_set():
            write_vr_frame(shared, wrist_pos=[0.3, 0.0, 0.25])
            stop.wait(interval_s)

    thread = threading.Thread(target=_write_loop, daemon=True)
    thread.start()
    return stop, thread


def _start_teleop(shared, cfg):
    shared.safety_state.value = int(SafetyState.ARMED)
    _mark_ready(shared, cfg)
    write_vr_frame(shared, wrist_pos=[0.3, 0.0, 0.25])
    thread = run_in_thread(teleop_loop, shared, cfg)
    wait_until(
        lambda: shared.get_heartbeat("policy") > 0.0,
        timeout_s=20.0,
        description="teleop main loop",
    )
    return thread


def _stop_arm_writer(stop_event: threading.Event, thread: threading.Thread) -> None:
    stop_event.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "arm writer thread did not stop"


def test_begin_transitions_to_running(shared, teleop_fakes):
    cfg = make_teleop_config(hand_enabled=False, recording_enabled=False)
    arm_stop, arm_thread = _start_arm_writer(shared, _HOME_QPOS)
    thread = _start_teleop(shared, cfg)
    try:
        FakeKeyboardHandler.last_instance.press(ControlSignal.BEGIN)
        wait_until(
            lambda: shared.safety_state.value == int(SafetyState.RUNNING),
            description="BEGIN -> RUNNING",
        )
    finally:
        stop_loop(shared, thread)
        _stop_arm_writer(arm_stop, arm_thread)
    assert not shared.error_state.value


def test_pause_publishes_measured_hold(shared, teleop_fakes):
    cfg = make_teleop_config(hand_enabled=False, recording_enabled=False)
    arm_stop, arm_thread = _start_arm_writer(shared, _HOME_QPOS)
    thread = _start_teleop(shared, cfg)
    try:
        FakeKeyboardHandler.last_instance.press(ControlSignal.BEGIN)
        wait_until(
            lambda: shared.safety_state.value == int(SafetyState.RUNNING),
            description="BEGIN -> RUNNING",
        )
        FakeKeyboardHandler.last_instance.press(ControlSignal.PAUSE)

        hold = None
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and hold is None:
            item = drain_arm_action_q(shared)
            if item is not None and bool(item["is_hold"][0]):
                hold = item
        assert hold is not None, "PAUSE did not publish a measured arm hold"
        wait_until(
            lambda: shared.safety_state.value == int(SafetyState.ARMED),
            description="PAUSE -> ARMED",
        )
    finally:
        stop_loop(shared, thread)
        _stop_arm_writer(arm_stop, arm_thread)
    assert not shared.error_state.value


def test_stale_arm_feedback_faults(shared, teleop_fakes):
    cfg = make_teleop_config(
        hand_enabled=False,
        recording_enabled=False,
        **{"policy.max_consecutive_errors": 2},
    )
    # No arm writer: after BEGIN the grid reads no fresh arm state and faults.
    thread = _start_teleop(shared, cfg)
    try:
        FakeKeyboardHandler.last_instance.press(ControlSignal.BEGIN)
        wait_until(
            lambda: shared.safety_state.value == int(SafetyState.RUNNING),
            description="BEGIN -> RUNNING",
        )
        wait_until(
            lambda: bool(shared.error_state.value),
            description="stale arm feedback -> fault",
        )
    finally:
        stop_loop(shared, thread)


def test_clean_shutdown(shared, teleop_fakes):
    cfg = make_teleop_config(hand_enabled=False, recording_enabled=False)
    arm_stop, arm_thread = _start_arm_writer(shared, _HOME_QPOS)
    thread = _start_teleop(shared, cfg)
    stop_loop(shared, thread)
    _stop_arm_writer(arm_stop, arm_thread)
    assert not shared.error_state.value


def test_hand_feedback_pause_advances_generation(shared, teleop_fakes):
    cfg = make_teleop_config(hand_enabled=True, recording_enabled=False)
    hand_home = np.deg2rad(np.asarray(cfg.runtime.hand.home_qpos_deg, dtype=np.float64))
    arm_stop, arm_thread = _start_arm_writer(shared, _HOME_QPOS)
    hand_stop, hand_thread = _start_hand_writer(shared, hand_home)
    vr_stop, vr_thread = _start_vr_writer(shared)
    thread = _start_teleop(shared, cfg)
    try:
        initial_generation = int(shared.run_generation.value)
        FakeKeyboardHandler.last_instance.press(ControlSignal.BEGIN)
        wait_until(
            lambda: shared.safety_state.value == int(SafetyState.RUNNING),
            description="BEGIN -> RUNNING (hand enabled)",
        )
        # Flip hand feedback to unhealthy: the coordinator pauses without
        # publishing and advances the control-run generation.
        hand_stop.set()
        hand_thread.join(timeout=2.0)
        write_hand_state(shared, qpos=hand_home, send_healthy=False)
        wait_until(
            lambda: int(shared.run_generation.value) > initial_generation,
            description="hand feedback pause advances run generation",
        )
    finally:
        stop_loop(shared, thread)
        _stop_arm_writer(arm_stop, arm_thread)
        vr_stop.set()
        vr_thread.join(timeout=2.0)
        if hand_thread.is_alive():
            hand_stop.set()
            hand_thread.join(timeout=2.0)
    assert int(shared.run_generation.value) > initial_generation

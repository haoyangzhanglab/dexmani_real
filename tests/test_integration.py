"""Full arm+teleop integration harness — both loop bodies on one SharedStorage.

Runs ``arm_loop`` (against the fake xArm SDK) and ``teleop_loop`` (patched
operator/audio/SIGTERM seams) as two daemon threads on a single SharedStorage,
so the coordinator's holds and HOME requests are actually consumed, applied,
and acknowledged by the worker.  This covers the two remaining Phase-1.4
closures that the single-loop harnesses could not: ``_complete_reanchor``
(fresh-feedback hold release) and ``_handoff_control_hold_to_home`` /
``_do_configured_teleop_home`` (HOME round-trip).
"""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from dexmani_real.robot.arm_loop import ArmLoopConfig, arm_loop
from dexmani_real.robot.safety import SafetyState
from dexmani_real.teleop.keyboard import ControlSignal
from dexmani_real.teleop.loop import teleop_loop

from tests.fakes.keyboard import FakeKeyboardHandler
from tests.fakes.workers import write_vr_frame
from tests.helpers import make_teleop_config, run_in_thread, stop_loop, wait_until


def _arm_record(shared):
    latest = shared.arm_state_ring.read_latest()
    return None if latest is None else latest[0][0]


def _start_arm(shared, fake_cls):
    """Start arm_loop and wait for Mode-6 ready State 2."""
    shared.safety_state.value = int(SafetyState.ARMED)
    arm_thread = run_in_thread(arm_loop, shared, ArmLoopConfig())
    assert shared.arm_ready.wait(timeout=8.0), "arm_loop did not set arm_ready"
    wait_until(
        lambda: fake_cls.last_instance is not None
        and fake_cls.last_instance.mode == 6
        and fake_cls.last_instance.state == 2,
        description="arm Mode-6 ready",
    )
    return arm_thread, fake_cls.last_instance


def _start_teleop(shared, cfg):
    """Start teleop_loop (arm_ready is already set by arm_loop)."""
    shared.vr_ready.set()
    thread = run_in_thread(teleop_loop, shared, cfg)
    wait_until(
        lambda: shared.policy_heartbeat_s.value > 0.0,
        timeout_s=20.0,
        description="teleop main loop",
    )
    return thread


def _start_vr_writer(shared, interval_s: float = 0.03):
    stop = threading.Event()

    def _loop() -> None:
        while not stop.is_set():
            write_vr_frame(shared, wrist_pos=[0.3, 0.0, 0.25])
            stop.wait(interval_s)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return stop, thread


def test_home_roundtrip_through_arm_worker(shared, arm_fakes, teleop_fakes, capsys):
    """HOME handoff: teleop plans+queues HOME_SENTINEL, arm_loop executes+ACKs."""
    cfg = make_teleop_config(hand_enabled=False, recording_enabled=False)
    arm_thread, fake = _start_arm(shared, arm_fakes)

    home = np.asarray(cfg.runtime.arm.home_qpos, dtype=np.float64)
    fake.qpos = home.copy()
    fake.qpos[0] += 0.1  # small J1 offset -> a real multi-milestone homing path

    write_vr_frame(shared, wrist_pos=[0.3, 0.0, 0.25])
    teleop_thread = _start_teleop(shared, cfg)
    gen0 = int(shared.run_generation.value)
    try:
        FakeKeyboardHandler.last_instance.press(ControlSignal.HOME)
        # arm_loop servos the planned milestones to the canonical home.
        wait_until(
            lambda: np.allclose(fake.qpos, home, atol=1e-4),
            timeout_s=20.0,
            description="arm_loop servos canonical home",
        )
        # wait_for_arm_home consumed the HomeResult and reported success.
        wait_until(
            lambda: "home reached" in capsys.readouterr().out,
            timeout_s=5.0,
            description="teleop wait_for_arm_home success",
        )
        assert len(fake.servo_calls) > 0, "homing milestones were not servoed"
        assert int(shared.run_generation.value) > gen0, "HOME did not advance run_generation"
    finally:
        stop_loop(shared, teleop_thread)
        stop_loop(shared, arm_thread)
    assert not shared.error_state.value


def test_vr_stale_hold_releases_after_reanchor(shared, arm_fakes, teleop_fakes, caplog):
    """vr_stale hold -> _enter_measured_hold -> fresh re-anchor (_complete_reanchor)."""
    caplog.set_level(logging.INFO)
    cfg = make_teleop_config(hand_enabled=False, recording_enabled=False)
    arm_thread, fake = _start_arm(shared, arm_fakes)

    # Start the arm at the canonical (in-workspace) home pose so the begin
    # re-anchor and any IK target remain workspace-safe; zeros(7) sits outside
    # the teleop workspace and makes even the hold publish get rejected.
    fake.qpos = np.asarray(cfg.runtime.arm.home_qpos, dtype=np.float64).copy()

    vr_stop, vr_thread = _start_vr_writer(shared)
    teleop_thread = _start_teleop(shared, cfg)
    try:
        FakeKeyboardHandler.last_instance.press(ControlSignal.BEGIN)
        wait_until(
            lambda: shared.safety_state.value == int(SafetyState.RUNNING),
            description="BEGIN -> RUNNING",
        )

        base_seq = int((_arm_record(shared) or {"last_cmd_seq": 0})["last_cmd_seq"])
        # Stop VR: after the stale threshold the coordinator enters a measured hold.
        vr_stop.set()
        vr_thread.join(timeout=2.0)
        wait_until(
            lambda: (r := _arm_record(shared)) is not None
            and int(r["last_cmd_is_hold"]) == 1
            and int(r["last_cmd_seq"]) > base_seq,
            timeout_s=8.0,
            description="vr_stale hold applied by arm_loop",
        )

        # Restart VR: once the hold apply timer elapses and fresh feedback is
        # anchored, _complete_reanchor releases the hold.
        vr_stop, vr_thread = _start_vr_writer(shared)
        wait_until(
            lambda: "released vr_stale hold after fresh re-anchor" in caplog.text,
            timeout_s=8.0,
            description="_complete_reanchor releases the vr_stale hold",
        )
    finally:
        vr_stop.set()
        vr_thread.join(timeout=2.0)
        stop_loop(shared, teleop_thread)
        stop_loop(shared, arm_thread)
    assert not shared.error_state.value

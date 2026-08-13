"""Headless arm_loop harness — exercises the arm worker against a fake xArm SDK.

These tests run the production ``arm_loop`` body in a daemon thread against a
real ``SharedStorage``, with ``xarm.wrapper.XArmAPI`` replaced by
``FakeXArmAPI``.  They pin the observable worker behaviors (state publication,
servo application, STOP/RESUME priority control, homing, and collision fault
latching) so the Phase-1.4 closure extraction can be validated as a
zero-behavior-change refactor.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.arm_loop import ArmLoopConfig, arm_loop
from dexmani_real.robot.homing import HOME_SENTINEL, HomeRequest
from dexmani_real.robot.safety import SafetyState
from dexmani_real.robot.types import ArmControlKind

from tests.fakes.workers import make_arm_command, make_arm_control_request
from tests.helpers import run_in_thread, stop_loop, wait_until


def _start_arm(shared, *, safety: SafetyState = SafetyState.DISARMED):
    shared.safety_state.value = int(safety)
    thread = run_in_thread(arm_loop, shared, ArmLoopConfig())
    return thread


def _wait_connected(fake_cls):
    wait_until(lambda: fake_cls.last_instance is not None, description="arm SDK connect")
    return fake_cls.last_instance


def _wait_mode6_ready(fake):
    wait_until(
        lambda: fake.mode == 6 and fake.state == 2,
        description="Mode-6 ready State 2",
    )


def test_startup_publishes_state_and_sets_ready(shared, arm_fakes):
    thread = _start_arm(shared)
    fake = _wait_connected(arm_fakes)
    assert shared.arm_ready.wait(timeout=8.0), "arm_ready was not set"
    wait_until(
        lambda: shared.arm_state_ring.read_latest() is not None,
        description="initial arm state publication",
    )
    record = shared.arm_state_ring.read_latest()[0][0]
    assert bool(record["connected"]) and bool(record["state_valid"])
    assert np.all(np.isfinite(record["qpos"]))
    stop_loop(shared, thread)
    assert fake.connected is False, "arm_loop should disconnect on clean exit"


def test_servo_endpoint_applies_target(shared, arm_fakes):
    thread = _start_arm(shared, safety=SafetyState.ARMED)
    fake = _wait_connected(arm_fakes)
    assert shared.arm_ready.wait(timeout=8.0)
    _wait_mode6_ready(fake)

    cfg = ArmLoopConfig()
    target = np.asarray(cfg.home_qpos, dtype=np.float64).copy()
    target[0] += 0.05
    shared.arm_action_q.put(make_arm_command(shared, target, action_id=1))

    wait_until(lambda: len(fake.servo_calls) > 0, description="servo endpoint applied")
    np.testing.assert_allclose(fake.servo_calls[0], target, atol=1e-9)
    # The applied endpoint is published back through the state ring.
    wait_until(
        lambda: shared.arm_state_ring.read_latest() is not None
        and int(shared.arm_state_ring.read_latest()[0][0]["last_cmd_seq"]) == 1,
        description="endpoint state publication",
    )
    stop_loop(shared, thread)


def test_decelerated_stop_then_resume(shared, arm_fakes):
    thread = _start_arm(shared, safety=SafetyState.ARMED)
    fake = _wait_connected(arm_fakes)
    assert shared.arm_ready.wait(timeout=8.0)
    _wait_mode6_ready(fake)

    make_arm_control_request(shared, ArmControlKind.DECELERATED_STOP, action_id=1)
    wait_until(lambda: 6 in fake.state_calls, description="decelerated State-6 request")
    assert fake.mode == 6

    make_arm_control_request(shared, ArmControlKind.RESUME, action_id=2)
    # Resume re-enters Mode 6 behind a measured hold and reports ready State 2.
    wait_until(
        lambda: fake.mode == 6 and fake.state == 2,
        description="resume behind measured hold",
    )
    stop_loop(shared, thread)


def test_home_single_point_confirm_dwell(shared, arm_fakes):
    thread = _start_arm(shared, safety=SafetyState.ARMED)
    fake = _wait_connected(arm_fakes)
    assert shared.arm_ready.wait(timeout=8.0)
    _wait_mode6_ready(fake)

    cfg = ArmLoopConfig()
    home = np.asarray(cfg.home_qpos, dtype=np.float64)
    fake.qpos = home.copy()

    request = HomeRequest(
        request_id=1,
        waypoints=home.reshape(1, -1),
        final_qpos=home,
        execution_timeout_s=10.0,
    )
    shared.arm_action_q.put((HOME_SENTINEL, request))

    result = shared.arm_home_result_q.get(timeout=10.0)
    assert result.request_id == 1
    assert result.success, result.reason
    np.testing.assert_allclose(result.final_qpos, home, atol=1e-6)
    stop_loop(shared, thread)


def test_home_multi_milestone_executes_mode0(shared, arm_fakes):
    thread = _start_arm(shared, safety=SafetyState.ARMED)
    fake = _wait_connected(arm_fakes)
    assert shared.arm_ready.wait(timeout=8.0)
    _wait_mode6_ready(fake)

    cfg = ArmLoopConfig()
    home = np.asarray(cfg.home_qpos, dtype=np.float64)
    start = home.copy()
    start[0] += 0.1  # well inside the ±6.28 rad joint-0 range
    fake.qpos = start.copy()

    request = HomeRequest(
        request_id=2,
        waypoints=np.stack([start, home]),
        final_qpos=home,
        execution_timeout_s=10.0,
    )
    shared.arm_action_q.put((HOME_SENTINEL, request))

    result = shared.arm_home_result_q.get(timeout=10.0)
    assert result.success, result.reason
    np.testing.assert_allclose(result.final_qpos, home, atol=1e-3)
    # The final milestone is delivered through the SDK servo path (Mode 0 MoveJoint).
    assert len(fake.servo_calls) > 0
    np.testing.assert_allclose(fake.servo_calls[-1], home, atol=1e-9)
    stop_loop(shared, thread)


def test_collision_fault_c31_latches_error(shared, arm_fakes):
    thread = _start_arm(shared, safety=SafetyState.ARMED)
    fake = _wait_connected(arm_fakes)
    assert shared.arm_ready.wait(timeout=8.0)
    _wait_mode6_ready(fake)

    fake.error_code = 31  # collision fault
    wait_until(lambda: bool(shared.error_state.value), description="C31 fault latch")
    stop_loop(shared, thread)

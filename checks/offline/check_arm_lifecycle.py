"""Phase A: arm lifecycle leaf helpers, homing abort, and the single finalizer.

Covers doc §11.1 (controller state / stop confirmation) and §11.2 (homing
abort classification and the worker-side finalizer).  Each case runs against
``FakeArm`` and a lightweight ``SimpleNamespace`` shared-storage stand-in, so no
hardware or real ``SharedStorage`` is required.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

import _bootstrap  # noqa: F401  (repo root on sys.path)
from _fakes import FakeArm

from dexmani_real.robot.arm_loop import (
    _finalize_home_result,
    _request_still_current_and_armed,
)
from dexmani_real.robot.arm_sdk import (
    controller_state_allows_motion,
    enter_mode0,
    enter_mode6,
    stop_controller,
)
from dexmani_real.robot.homing import _shared_abort_reason_impl, send_arm_home
from dexmani_real.robot.safety import SafetyState
from dexmani_real.shm.shared_storage import HomeOutcome, HomeRequest, HomeResult

_ARMED = int(SafetyState.ARMED)
_DISARMED = int(SafetyState.DISARMED)
_FAULT = int(SafetyState.FAULT)


def _v(value: bool | int) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def _shared(**overrides) -> SimpleNamespace:
    base = dict(
        is_running=_v(True),
        estop_request=_v(False),
        error_state=_v(False),
        safety_state=_v(_ARMED),
        run_generation=_v(5),
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _request() -> HomeRequest:
    return HomeRequest(
        request_id=1,
        run_generation=5,
        waypoints=np.zeros((1, 7), dtype=np.float64),
        final_qpos=np.zeros(7, dtype=np.float64),
        execution_timeout_s=10.0,
    )


def _result(outcome: HomeOutcome, reason: str = "") -> HomeResult:
    return HomeResult(
        request_id=1,
        outcome=outcome,
        reason=reason,
        final_qpos=np.zeros(7, dtype=np.float64),
        completed_at_s=0.0,
    )


def _test_controller_state() -> None:
    for state in (0, 1, 2):
        assert controller_state_allows_motion(state), f"state {state} must allow motion"
    for state in (3, 4, 5, -1, 99):
        assert not controller_state_allows_motion(state), (
            f"state {state} must fail closed"
        )


def _test_enter_mode() -> None:
    # Enter Mode 6 from a stopped controller succeeds and refreshes heartbeat.
    polled: list[int] = []
    arm = FakeArm(state=4, mode=6, error_code=0)
    enter_mode6(arm, on_poll=lambda: polled.append(1))
    assert arm.state == 0, "enter_mode6 must leave a movable state"
    assert polled, "enter_mode6 must poll the heartbeat callback"

    # Setter failure fails closed (raises, never silently continues).
    arm_bad = FakeArm(state=4, mode=6, error_code=0)
    arm_bad.fail("set_mode", 1)
    try:
        enter_mode6(arm_bad)
    except RuntimeError:
        pass
    else:
        raise AssertionError("enter_mode6 must raise when set_mode fails")

    arm_bad2 = FakeArm(state=4, mode=6, error_code=0)
    arm_bad2.fail("set_state", 1)
    try:
        enter_mode0(arm_bad2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("enter_mode0 must raise when set_state fails")


def _test_stop_controller() -> None:
    # Confirms State 4 even with a latched controller error (no error clear).
    stop = stop_controller(FakeArm(state=4, mode=6, error_code=24))
    assert stop.confirmed, stop.reason

    # Emergency flag routes through emergency_stop first.
    arm = FakeArm(state=4, mode=6, error_code=0)
    assert stop_controller(arm, emergency=True).confirmed
    assert "emergency_stop" in arm.call_order()

    # A set_state(4) that reports failure still confirms via a live read when
    # the controller is already at State 4 (and does not drop the fault).
    arm_stuck = FakeArm(state=4, mode=6, error_code=0)
    arm_stuck.fail("set_state", 1)
    stop_stuck = stop_controller(arm_stuck)
    assert stop_stuck.confirmed, stop_stuck.reason


def _test_abort_reason() -> None:
    assert _shared_abort_reason_impl(_shared()) is None

    for kwargs in (
        dict(safety_state=_v(_DISARMED)),
        dict(run_generation=_v(6)),
        dict(estop_request=_v(True)),
        dict(is_running=_v(False)),
    ):
        outcome, reason = _shared_abort_reason_impl(
            _shared(**kwargs), expected_generation=5
        )
        assert outcome is HomeOutcome.CANCELLED, (kwargs, reason)

    for kwargs in (
        dict(safety_state=_v(_FAULT)),
        dict(error_state=_v(True)),
    ):
        outcome, reason = _shared_abort_reason_impl(
            _shared(**kwargs), expected_generation=5
        )
        assert outcome is HomeOutcome.FAILED, (kwargs, reason)


def _test_request_current_and_armed() -> None:
    req = _request()
    assert _request_still_current_and_armed(_shared(), req) is True
    assert _request_still_current_and_armed(_shared(run_generation=_v(6)), req) is False
    assert _request_still_current_and_armed(_shared(safety_state=_v(_DISARMED)), req) is False
    assert _request_still_current_and_armed(_shared(error_state=_v(True)), req) is False


def _test_finalize() -> None:
    req = _request()

    # FAILED: sticky fault + stop + never accept motion.
    sh = _shared()
    terminal, accept = _finalize_home_result(
        sh, FakeArm(state=0, mode=0), req, _result(HomeOutcome.FAILED, "boom"),
        on_poll=lambda: None,
    )
    assert terminal.outcome is HomeOutcome.FAILED
    assert accept is False
    assert sh.error_state.value is True

    # CANCELLED with confirmed stop: no new fault, no motion.
    sh = _shared()
    terminal, accept = _finalize_home_result(
        sh, FakeArm(state=0, mode=0), req, _result(HomeOutcome.CANCELLED, "e-stop"),
        on_poll=lambda: None,
    )
    assert terminal.outcome is HomeOutcome.CANCELLED
    assert accept is False
    assert sh.error_state.value is False, "clean cancellation must not add a fault"

    # SUCCESS: restore Mode 6 and resume motion acceptance.
    sh = _shared()
    terminal, accept = _finalize_home_result(
        sh, FakeArm(state=0, mode=0), req, _result(HomeOutcome.SUCCESS, "home"),
        on_poll=lambda: None,
    )
    assert terminal.outcome is HomeOutcome.SUCCESS
    assert accept is True
    assert sh.error_state.value is False

    # SUCCESS but the generation changed before restore: upgrade to FAILED.
    sh = _shared(run_generation=_v(6))
    terminal, accept = _finalize_home_result(
        sh, FakeArm(state=0, mode=0), req, _result(HomeOutcome.SUCCESS, "home"),
        on_poll=lambda: None,
    )
    assert terminal.outcome is HomeOutcome.FAILED
    assert accept is False
    assert sh.error_state.value is True

    # CANCELLED whose stop cannot be confirmed upgrades to FAILED + sticky.
    arm_stuck = FakeArm(state=0, mode=0)
    arm_stuck.fail("set_state", 1)  # set_state reports failure, state stays 0
    sh = _shared()
    terminal, accept = _finalize_home_result(
        sh, arm_stuck, req, _result(HomeOutcome.CANCELLED, "e-stop"),
        on_poll=lambda: None,
    )
    assert terminal.outcome is HomeOutcome.FAILED
    assert accept is False
    assert sh.error_state.value is True


def _test_home_producer_gate() -> None:
    home = np.zeros(7, dtype=np.float64)

    # HOME is a motion command produced only from ARMED; RUNNING/DISARMED/FAULT
    # are rejected before planning or queue publication.
    assert send_arm_home(_shared(safety_state=_v(int(SafetyState.RUNNING))), home, verbose=False) is False
    assert send_arm_home(_shared(safety_state=_v(_DISARMED)), home, verbose=False) is False
    assert send_arm_home(_shared(safety_state=_v(_FAULT)), home, verbose=False) is False


def _test_outcome_migration() -> None:
    assert int(HomeOutcome.SUCCESS) == 0
    assert int(HomeOutcome.CANCELLED) == 1
    assert int(HomeOutcome.FAILED) == 2
    assert _result(HomeOutcome.SUCCESS).success is True
    assert _result(HomeOutcome.CANCELLED).success is False
    assert _result(HomeOutcome.FAILED).success is False


def main() -> int:
    _test_controller_state()
    _test_enter_mode()
    _test_stop_controller()
    _test_abort_reason()
    _test_request_current_and_armed()
    _test_finalize()
    _test_outcome_migration()
    _test_home_producer_gate()
    print("check_arm_lifecycle: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

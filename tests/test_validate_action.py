"""Tests for validate_action: gate paths + degraded-hand semantics (no hardware)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.robot.types import RobotAction
from dexmani_real.robot.validate import validate_action


def _make_action(arm_cmd: np.ndarray | None = None, hand_cmd: np.ndarray | None = None) -> RobotAction:
    return RobotAction(
        arm_qpos_cmd=arm_cmd if arm_cmd is not None else np.zeros(7, dtype=np.float64),
        hand_qpos_cmd=hand_cmd if hand_cmd is not None else np.zeros(12, dtype=np.float64),
    )


def _stub_clamp(pos: np.ndarray) -> np.ndarray:
    """Identity — tests don't exercise workspace clamping."""
    return pos


def _stub_robot(
    *,
    arm_error: bool = False,
    arm_connected: bool = True,
    hand_connected: bool = False,
    hand_error: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        arm=SimpleNamespace(
            is_error=lambda: arm_error,
            is_connected=lambda: arm_connected,
            qpos_min_soft=np.array([-np.pi] * 7, dtype=np.float64),
            qpos_max_soft=np.array([np.pi] * 7, dtype=np.float64),
        ),
        hand=SimpleNamespace(
            connected_flag=hand_connected,
            error_state=hand_error,
            config=SimpleNamespace(
                qpos_min=np.array([-np.pi] * 12, dtype=np.float64),
                qpos_max=np.array([np.pi] * 12, dtype=np.float64),
            ),
        ),
        clamp_workspace_pos=_stub_clamp,
        is_error=lambda: arm_error or hand_error,
    )


# ── Check 1: error state — degraded-hand semantics ──


def test_hand_absent_and_errored_passes():
    """Hand absent (failed connect → error_state=True but connected_flag=False) must pass."""
    robot = _stub_robot(hand_connected=False, hand_error=True)
    ok, reason = validate_action(robot, _make_action())
    assert ok, f"expected pass for absent hand, got: {reason}"


def test_hand_connected_and_errored_fails():
    """Hand connected with error_state=True must fail."""
    robot = _stub_robot(hand_connected=True, hand_error=True)
    ok, reason = validate_action(robot, _make_action())
    assert not ok
    assert "hand error" in reason


def test_arm_error_fails():
    robot = _stub_robot(arm_error=True)
    ok, reason = validate_action(robot, _make_action())
    assert not ok
    assert "arm error" in reason


def test_arm_not_connected_fails():
    robot = _stub_robot(arm_connected=False)
    ok, reason = validate_action(robot, _make_action())
    assert not ok
    assert "not connected" in reason


def test_no_error_passes():
    robot = _stub_robot()
    ok, reason = validate_action(robot, _make_action())
    assert ok


# ── Check 3: torque gate ──


def test_torque_over_limit_fails():
    robot = _stub_robot()
    # _ARM_TORQUE_LIMIT_NM[0] = 50 Nm — exceed J1
    tau = np.zeros(7, dtype=np.float64)
    tau[0] = 60.0
    ok, reason = validate_action(robot, _make_action(), actual_arm_tau=tau)
    assert not ok
    assert "torque" in reason


def test_torque_nan_passes():
    """NaN tau silently skips the gate."""
    robot = _stub_robot()
    tau = np.full(7, np.nan, dtype=np.float64)
    ok, reason = validate_action(robot, _make_action(), actual_arm_tau=tau)
    assert ok


def test_torque_within_limit_passes():
    robot = _stub_robot()
    tau = np.full(7, 10.0, dtype=np.float64)
    ok, reason = validate_action(robot, _make_action(), actual_arm_tau=tau)
    assert ok


# ── Check 4: temperature gate ──


def test_temp_over_limit_fails():
    robot = _stub_robot()
    temps = np.full(7, 75.0, dtype=np.float64)  # limit is 70°C
    ok, reason = validate_action(robot, _make_action(), actual_arm_temps=temps)
    assert not ok
    assert "temperature" in reason


def test_temp_nan_passes():
    robot = _stub_robot()
    temps = np.full(7, np.nan, dtype=np.float64)
    ok, _ = validate_action(robot, _make_action(), actual_arm_temps=temps)
    assert ok


# ── Check 7: arm joint-limit clip (in-place) ──


def test_arm_joint_limit_clip():
    """validate_action passes through commands unchanged (absolute joint-limit clip
    is applied downstream in ArmInnerLoop._send_target, per H3 fix)."""
    robot = _stub_robot()
    cmd = np.array([3.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    action = _make_action(arm_cmd=cmd)
    ok, reason = validate_action(robot, action)
    assert ok, f"finite commands should pass: {reason}"
    # Commands are not mutated in-place — clipping is in _send_target.
    assert action.arm_qpos_cmd[0] == 3.0
    assert action.arm_qpos_cmd[1] == -3.0


# ── Default args: all-optional → passes ──


def test_validate_action_default_args():
    """validate_action with no optional args is backward-compatible (checks 3/4/5/6 skip)."""
    robot = _stub_robot()
    ok, reason = validate_action(robot, _make_action())
    assert ok, f"expected ok with defaults, got: {reason}"

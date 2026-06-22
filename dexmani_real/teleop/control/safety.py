"""Per-frame safety checks on RobotState. Stateless, shared by teleop and deploy."""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.types import RobotState, _ARM_TORQUE_LIMIT_NM

# Default thresholds (teleop). Deploy may use stricter values.
# _ARM_TORQUE_LIMIT_NM is defined in robot/types.py (hardware property, not teleop policy).
# Conservative defaults based on XHand specs. Validate against the actual
# motor datasheet before production use.
_HAND_CURRENT_LIMIT_MA = 500.0
_HAND_TEMP_LIMIT_C = 70.0
# Retarget validity range aligned with XHand hardware joint limits (2026-06-22).
# Previous hardcoded [-0.5, 2.5] was too narrow on the low end (-0.5 > XHand
# thumb min -0.698 rad, causing false negatives) and too wide on the high end
# (2.5 > XHand max 1.92 rad, risking false positives).
# New bounds match XHand qpos_min.min() ≈ -0.698 rad and qpos_max.max() ≈ 1.92 rad,
# with a small safety margin (±0.05 rad) to allow for numerical noise at IK boundaries.
_RETARGET_VALID_MIN = -0.75   # rad, ~43°, XHand thumb min=-40° (-0.698 rad)
_RETARGET_VALID_MAX = 2.0     # rad, ~115°, XHand max=110° (1.92 rad)


def check_arm_torque(
    state: RobotState,
    torque_limit_nm: np.ndarray | float = _ARM_TORQUE_LIMIT_NM,
) -> bool:
    tau = state.arm_tau
    if not np.all(np.isfinite(tau)):
        return False
    if isinstance(torque_limit_nm, np.ndarray):
        if len(tau) != len(torque_limit_nm):
            return False
        return not np.any(np.abs(tau) >= torque_limit_nm)
    return float(np.max(np.abs(tau))) < torque_limit_nm


def check_hand_current(
    state: RobotState,
    current_limit_ma: float = _HAND_CURRENT_LIMIT_MA,
) -> bool:
    cur = state.hand_current
    if not np.all(np.isfinite(cur)):
        return False
    return float(np.max(cur)) < current_limit_ma


def check_hand_temperature(
    state: RobotState,
    temp_limit_c: float = _HAND_TEMP_LIMIT_C,
) -> bool:
    temp = state.hand_temperature
    if not np.all(np.isfinite(temp)):
        return False
    return float(np.max(temp)) < temp_limit_c


def check_hand_comm(state: RobotState) -> bool:
    return not state.hand_error


def check_arm_joint_limits(
    state: RobotState,
    qpos_min: np.ndarray,
    qpos_max: np.ndarray,
) -> bool:
    """Check arm_qpos is within hardware joint limits."""
    qpos = state.arm_qpos
    if not np.all(np.isfinite(qpos)):
        return False
    return not (np.any(qpos < qpos_min) or np.any(qpos > qpos_max))


def check_hand_joint_limits(
    state: RobotState,
    qpos_min: np.ndarray,
    qpos_max: np.ndarray,
) -> bool:
    """Check hand_qpos is within hardware joint limits."""
    qpos = state.hand_qpos
    if not np.all(np.isfinite(qpos)):
        return False
    return not (np.any(qpos < qpos_min) or np.any(qpos > qpos_max))


def check_retarget_valid(
    hand_qpos: np.ndarray,
    physio_min: float = _RETARGET_VALID_MIN,
    physio_max: float = _RETARGET_VALID_MAX,
) -> bool:
    """Check retargeted hand_qpos is within hardware-aligned range.

    Default bounds match XHand joint limits with safety margin (±0.05 rad).
    Callers can override for other hand hardware or tighter safety policies.
    """
    if not np.all(np.isfinite(hand_qpos)):
        return False
    if np.any(hand_qpos < physio_min) or np.any(hand_qpos > physio_max):
        return False
    return True

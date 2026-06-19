"""Per-frame safety checks on RobotState. Stateless, shared by teleop and deploy."""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.interface import RobotState

# Default thresholds (teleop). Deploy may use stricter values.
# Per-joint torque limits from URDF: J1-J2=50, J3-J5=30, J6-J7=20 Nm.
# Using per-joint limits protects the wrist (J6-J7) whose rated torque is
# only 20 Nm — a unified 50 Nm threshold would allow 2.5× rated torque
# before triggering a warning.
_ARM_TORQUE_LIMIT_NM = np.array([50.0, 50.0, 30.0, 30.0, 30.0, 20.0, 20.0])
_HAND_CURRENT_LIMIT_MA = 500.0  # TODO: verify against XHand motor datasheet
_HAND_TEMP_LIMIT_C = 70.0       # TODO: verify against XHand motor datasheet
_RETARGET_PHYSIO_MIN = -0.5
_RETARGET_PHYSIO_MAX = 2.5


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
    physio_min: float = _RETARGET_PHYSIO_MIN,
    physio_max: float = _RETARGET_PHYSIO_MAX,
) -> bool:
    if not np.all(np.isfinite(hand_qpos)):
        return False
    if np.any(hand_qpos < physio_min) or np.any(hand_qpos > physio_max):
        return False
    return True

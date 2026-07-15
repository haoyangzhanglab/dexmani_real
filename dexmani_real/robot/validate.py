"""Pre-send validation — centralized safety gate for teleop actions.

Extracted from RobotInterface.validate_action() to remove the circular
dependency on teleop.control.safety (now deleted).

Ref: ManiUniCon validate_action pattern.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from dexmani_real.utils.log import get_logger
from dexmani_real.robot.types import RobotAction, _ARM_TEMP_LIMIT_C, _ARM_TORQUE_LIMIT_NM

logger = get_logger(__name__)


def validate_action(
    robot,  # RobotInterface (avoid circular import)
    action: RobotAction,
    *,
    actual_arm_qpos: np.ndarray | None = None,
    actual_arm_tau: np.ndarray | None = None,
    actual_arm_temps: np.ndarray | None = None,
    env_collision_check: Callable[[np.ndarray], bool] | None = None,
) -> tuple[bool, str]:
    """Centralized pre-send validation.

    Checks (fail-fast order):
      1. SDK error state (arm + hand is_error)
      2. Arm connection
      3. Torque gating  — per-joint threshold check
      4. Temperature gating — 70 °C per-joint threshold
      5. Env collision check — caller-provided predicate (if wired)
      6. Workspace clamp — per-axis clip (non-fatal)
      7. Arm joint-limit clipping (qpos_min/max)
      8. Hand joint-limit clipping (qpos_min/max)

    All new parameters are optional — None skips the check (backward
    compatible with existing call sites).
    """
    # 1. Hardware error state
    if robot.is_error():
        return False, "robot error state"

    # 2. Arm connection
    if not robot.arm.is_connected():
        return False, "arm not connected"

    # 3. Torque gating (per-joint)
    if actual_arm_tau is not None:
        tau = np.asarray(actual_arm_tau, dtype=np.float64)
        if tau.shape == (7,) and np.all(np.isfinite(tau)):
            over_idx = np.where(np.abs(tau) > _ARM_TORQUE_LIMIT_NM)[0]
            if len(over_idx) > 0:
                return False, f"torque limit exceeded: joints={over_idx.tolist()}"

    # 4. Temperature gating (per-joint)
    if actual_arm_temps is not None:
        temps = np.asarray(actual_arm_temps, dtype=np.float64)
        if temps.shape == (7,) and np.all(np.isfinite(temps)):
            over_idx = np.where(temps > _ARM_TEMP_LIMIT_C)[0]
            if len(over_idx) > 0:
                return False, f"temperature limit exceeded: joints={over_idx.tolist()}"

    # 5. Env collision check (caller-provided predicate)
    if env_collision_check is not None and actual_arm_qpos is not None:
        try:
            if not env_collision_check(actual_arm_qpos):
                return False, "env collision detected"
        except Exception as e:
            logger.warning("env_collision_check raised: %s — skipping", e)

    # 6. Workspace clamp (non-fatal)
    if action.target_eef_pos is not None:
        action.target_eef_pos[:] = robot.clamp_workspace_pos(action.target_eef_pos)

    # 7. Joint-limit clipping (arm) — soft limits, strictly inside the firmware
    #    reduced range so boundary-clipped commands never trip a reduced-mode
    #    fault (see xarm7._inset_joint_limits)
    arm_lo = robot.arm.qpos_min_soft
    arm_hi = robot.arm.qpos_max_soft
    action.arm_qpos_cmd[:] = np.clip(action.arm_qpos_cmd, arm_lo, arm_hi)

    # 8. Joint-limit clipping (hand)
    hand_lo = robot.hand.config.qpos_min
    hand_hi = robot.hand.config.qpos_max
    if action.hand_qpos_cmd is not None:
        action.hand_qpos_cmd[:] = np.clip(action.hand_qpos_cmd, hand_lo, hand_hi)

    return True, "ok"

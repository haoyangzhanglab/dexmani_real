"""Pre-send validation — centralized safety gate for teleop actions.

Extracted from RobotInterface.validate_action() to remove the circular
dependency on teleop.control.safety (now deleted).

Ref: ManiUniCon validate_action pattern.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.types import _ARM_TORQUE_LIMIT_NM, RobotAction
from dexmani_real.utils.log import get_logger

# Per-motor hand current limit — defense-in-depth for seized/stalled finger motors.
# Matches XHand firmware tor_max (xhand.py:180).
_HAND_CURRENT_LIMIT_MA = 300.0

logger = get_logger(__name__)


def validate_action(
    robot,  # RobotInterface (avoid circular import)
    action: RobotAction,
    *,
    actual_arm_qpos: np.ndarray | None = None,
    actual_arm_qvel: np.ndarray | None = None,
    actual_arm_tau: np.ndarray | None = None,
    actual_hand_current: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Centralized pre-send validation.

    Checks (fail-fast order):
      1. SDK error state (arm + hand is_error)
      2. Arm connection
      3. Torque gating  — per-joint threshold check
      4. Hand current gating — per-motor threshold check
      5. Arm joint-limit clipping — moved to ArmInnerLoop._send_target

    Removed from this gate (covered elsewhere):
      - Workspace clamp → TeleopPipeline Stage 3 (before IK)
      - Hand joint-limit clip → XHand.send_action() internal _limit_joint_range
      - Self-collision check → IK Stage (_check_teleop_collision_gate) + firmware error 22
      - Env collision check → IK Stage (_check_teleop_collision_gate)

    All optional parameters — None skips the check (backward compatible).
    """
    # 1. Hardware error state — hand errors gate only when the hand is connected:
    #    arm-only / degraded mode must not fail because XHand.is_error() is True
    #    for a merely-absent hand (xhand.py:395).  connected_flag (not
    #    is_connected()) so a connected-then-errored hand is still caught.
    if robot.arm.is_error():
        return False, "arm error state"
    if robot.hand.connected_flag and robot.hand.error_state:
        return False, "hand error state"

    # 2. Arm connection
    if not robot.arm.is_connected():
        return False, "arm not connected"

    # 2.5. Action NaN guard — reject NaN/Inf joint commands before any dispatch
    if not np.all(np.isfinite(action.arm_qpos_cmd)):
        return False, "action arm_qpos_cmd NaN/Inf"
    if not np.all(np.isfinite(action.hand_qpos_cmd)):
        return False, "action hand_qpos_cmd NaN/Inf"

    # 3. Torque gating (per-joint)
    if actual_arm_tau is not None:
        tau = np.asarray(actual_arm_tau, dtype=np.float64)
        if tau.shape == (7,) and np.all(np.isfinite(tau)):
            over_idx = np.where(np.abs(tau) > _ARM_TORQUE_LIMIT_NM)[0]
            if len(over_idx) > 0:
                return False, f"torque limit exceeded: joints={over_idx.tolist()}"
        else:
            logger.error("Torque data shape mismatch or contains NaN — rejecting action")
            return False, "torque data invalid"

    # 4. Arm velocity NaN guard — arm_qvel was previously unchecked (arm_qpos/arm_tau
    #    were gated but velocity NaN could silently propagate into HDF5 recordings).
    #    Symmetric with torque check: reject on NaN/Inf, skip when None.
    if actual_arm_qvel is not None:
        qvel = np.asarray(actual_arm_qvel, dtype=np.float64)
        if qvel.shape == (7,) and np.all(np.isfinite(qvel)):
            pass  # nominal
        else:
            logger.error("Arm velocity data shape mismatch or contains NaN — rejecting action")
            return False, "arm velocity data invalid"

    # 4.5. Hand current gating (per-motor) — defense-in-depth for
    # seized/stalled finger motors; complements XHand firmware overcurrent
    # protection (tor_max=300 mA).
    if actual_hand_current is not None:
        hc = np.asarray(actual_hand_current, dtype=np.float64)
        if hc.shape == (12,) and np.all(np.isfinite(hc)):
            over_idx = np.where(np.abs(hc) > _HAND_CURRENT_LIMIT_MA)[0]
            if len(over_idx) > 0:
                return False, f"hand current limit exceeded: motors={over_idx.tolist()}"
        else:
            logger.error("Hand current data shape mismatch or contains NaN — rejecting action")
            return False, "hand current data invalid"

    # 4. Joint-limit clipping (arm) — moved to ArmInnerLoop._send_target.
    #    Absolute clip is applied there, before the per-step delta clamp,
    #    consolidating all joint-safety mechanics in one place.

    return True, "ok"

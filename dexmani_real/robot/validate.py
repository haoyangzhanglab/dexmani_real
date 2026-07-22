"""Pre-send validation — centralized safety gate for teleop actions.

Extracted from RobotInterface.validate_action() to remove the circular
dependency on teleop.control.safety (now deleted).

Ref: ManiUniCon validate_action pattern.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.types import _ARM_TEMP_LIMIT_C, _ARM_TORQUE_LIMIT_NM, RobotAction


def validate_action(
    robot,  # RobotInterface (avoid circular import)
    action: RobotAction,
    *,
    actual_arm_qpos: np.ndarray | None = None,
    actual_arm_tau: np.ndarray | None = None,
    actual_arm_temps: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Centralized pre-send validation.

    Checks (fail-fast order):
      1. SDK error state (arm + hand is_error)
      2. Arm connection
      3. Torque gating  — per-joint threshold check
      4. Temperature gating — 70 °C per-joint threshold
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

    # 5. Joint-limit clipping (arm) — moved to ArmInnerLoop._send_target.
    #    Absolute clip is applied there, before the per-step delta clamp,
    #    consolidating all joint-safety mechanics in one place.

    return True, "ok"

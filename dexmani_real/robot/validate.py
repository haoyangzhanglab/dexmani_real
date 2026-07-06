"""Pre-send validation — centralized safety gate for teleop actions.

Extracted from RobotInterface.validate_action() to remove the circular
dependency on teleop.control.safety (now deleted).

Ref: ManiUniCon validate_action pattern.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from dexmani_real.utils.log import get_logger
from dexmani_real.robot.types import RobotAction

logger = get_logger(__name__)


def validate_action(
    robot,  # RobotInterface (avoid circular import)
    action: RobotAction,
    *,
    actual_arm_qpos: np.ndarray | None = None,
    env_collision_check: Callable[[np.ndarray], bool] | None = None,
) -> tuple[bool, str]:
    """Centralized pre-send validation.

    Checks (fail-fast order):
      1. SDK error state (arm + hand is_error)
      2. Arm connection
      3. Workspace FK check — validates the command target position
         (action.arm_qpos_cmd FK), not the current actual position.
         This allows recovery commands that move the arm back into
         workspace when it has drifted outside bounds.
      4. Environment collision (defence in depth, S4) — independent
         second check beyond the IK-layer collision gate.  Uses the
         CollisionModel Tier-1 fast check (~17μs).

    Returns (ok, reason_string).
    """
    # 1. Hardware error state
    if robot.is_error():
        return False, "robot error state"

    # 2. Arm connection
    if not robot.arm.is_connected():
        return False, "arm not connected"

    # 3. Workspace bounds — validate the command target position.
    #    Using command FK (not actual position FK) so that recovery
    #    commands moving the arm back into workspace are not blocked
    #    when the arm has drifted outside bounds.
    cmd_eef = robot.kinematics.compute_eef_pose_world(action.arm_qpos_cmd)
    if not robot.workspace.check(cmd_eef.p):
        return False, "workspace position violation"

    # 4. Environment collision — defence in depth (S4)
    if env_collision_check is not None:
        qpos_for_col = actual_arm_qpos if (actual_arm_qpos is not None and np.all(np.isfinite(actual_arm_qpos))) else action.arm_qpos_cmd
        if env_collision_check(qpos_for_col):
            return False, "environment collision (pre-send gate)"

    return True, "ok"

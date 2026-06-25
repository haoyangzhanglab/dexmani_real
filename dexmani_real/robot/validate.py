"""Pre-send validation — centralized safety gate for teleop actions.

Extracted from RobotInterface.validate_action() to remove the circular
dependency on teleop.control.safety (now deleted).

Ref: ManiUniCon validate_action pattern.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.utils.log import get_logger
from dexmani_real.robot.types import RobotAction

logger = get_logger(__name__)


def validate_action(
    robot,  # RobotInterface (avoid circular import)
    action: RobotAction,
    *,
    actual_arm_qpos: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Centralized pre-send validation.

    Checks (fail-fast order):
      1. SDK error state (arm + hand is_error)
      2. Arm connection
      3. Workspace FK soft clamp — uses actual_arm_qpos (from inner loop)
         when available, falls back to action.arm_qpos_cmd (command).
         Real-time position is preferred because the "hold last command"
         fallback may drift past workspace bounds if commands progressively
         diverge from actual positions.

    Returns (ok, reason_string).
    """
    # 1. Hardware error state
    if robot.is_error():
        return False, "robot error state"

    # 2. Arm connection
    if not robot.arm.is_connected():
        return False, "arm not connected"

    # 3. Workspace bounds — prefer actual position over command
    qpos_for_fk = actual_arm_qpos
    if qpos_for_fk is None or not np.all(np.isfinite(qpos_for_fk)):
        qpos_for_fk = action.arm_qpos_cmd

    arm_eef = robot.kinematics.compute_eef_pose_world(qpos_for_fk)
    if not robot.workspace.check(arm_eef.p):
        return False, "workspace position violation"

    return True, "ok"

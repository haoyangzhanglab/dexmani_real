"""Pre-send validation — centralized safety gate for teleop actions.

Extracted from RobotInterface.validate_action() to remove the circular
dependency on teleop.control.safety (now deleted).

Ref: ManiUniCon validate_action pattern.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.log import get_logger
from dexmani_real.robot.types import RobotAction, _ARM_TORQUE_LIMIT_NM

logger = get_logger(__name__)


def validate_action(
    robot,  # RobotInterface (avoid circular import)
    action: RobotAction,
) -> tuple[bool, str]:
    """Centralized pre-send validation.

    Checks (fail-fast order):
      1. SDK error state (arm + hand is_error)
      2. Workspace FK soft clamp (out-of-bounds → hold)

    Returns (ok, reason_string).
    """
    # 1. Hardware error state
    if robot.is_error():
        return False, "robot error state"

    # 2. Arm connection
    if not robot.arm.is_connected():
        return False, "arm not connected"

    # 3. Workspace bounds (computed from action command)
    arm_eef = robot.kinematics.compute_eef_pose_world(action.arm_qpos_cmd)
    if not robot.workspace.check(arm_eef.p):
        return False, "workspace position violation"

    return True, "ok"

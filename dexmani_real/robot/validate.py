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
      3. Arm joint-limit clipping (qpos_min/max)
      4. Hand joint-limit clipping (qpos_min/max)

    Returns (ok, reason_string).
    """
    # 1. Hardware error state
    if robot.is_error():
        return False, "robot error state"

    # 2. Arm connection
    if not robot.arm.is_connected():
        return False, "arm not connected"

    # 3. Joint-limit clipping (arm)
    arm_lo = robot.arm.config.qpos_min
    arm_hi = robot.arm.config.qpos_max
    action.arm_qpos_cmd[:] = np.clip(action.arm_qpos_cmd, arm_lo, arm_hi)

    # 4. Joint-limit clipping (hand)
    hand_lo = robot.hand.config.qpos_min
    hand_hi = robot.hand.config.qpos_max
    if action.hand_qpos_cmd is not None:
        action.hand_qpos_cmd[:] = np.clip(action.hand_qpos_cmd, hand_lo, hand_hi)

    return True, "ok"

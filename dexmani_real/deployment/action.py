"""Interpret physical model output as source-neutral intent."""

import numpy as np

from dexmani_real.planning.kinematics.pose import Pose, rot6d_to_quat_wxyz
from dexmani_real.robot.action import ActionIntent


def physical_action_dim(action_mode):
    if action_mode == "joint":
        return 19
    if action_mode == "eef":
        return 21
    raise ValueError("action_mode must be joint or eef")


def policy_action_intent(action, action_mode):
    """Split one row of an admitted policy future into a physical intent."""
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (physical_action_dim(action_mode),):
        raise ValueError("physical action must have the expected shape")
    if action_mode == "joint":
        return ActionIntent(action_mode, action[:7], action[7:])
    return ActionIntent(
        action_mode, Pose(action[:3], rot6d_to_quat_wxyz(action[3:9])), action[9:]
    )

"""Interpret physical model output as source-neutral intent."""

import numpy as np

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
    split = 7 if action_mode == "joint" else 9
    return ActionIntent(action_mode, action[:split], action[split:])

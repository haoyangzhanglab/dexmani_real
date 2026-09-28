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
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (physical_action_dim(action_mode),) or not np.isfinite(action).all():
        raise ValueError("physical action must have the expected shape and finite values")
    split = 7 if action_mode == "joint" else 9
    return ActionIntent(action_mode, action[:split].copy(), action[split:].copy())

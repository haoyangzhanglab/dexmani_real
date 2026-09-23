"""Decode physical policy actions using current robot geometry and limits."""

import numpy as np

from dexmani_real.planning import Pose, XArm7MotionPlanner
from dexmani_real.planning.kinematics.ik import make_online_ik_config
from dexmani_real.planning.kinematics.pose import rot6d_to_quat_wxyz
from dexmani_real.robot.projection import project_hand_command


def physical_action_dim(action_mode):
    if action_mode == "joint":
        return 19
    if action_mode == "eef":
        return 21
    raise ValueError("action_mode must be joint or eef")


def make_action_planner(action_mode, runtime, *, control_dt_s):
    physical_action_dim(action_mode)
    if action_mode == "joint":
        return None
    return XArm7MotionPlanner.create_default(
        teleop_profile=make_online_ik_config(runtime, control_dt_s=control_dt_s)
    )


def decode_policy_action(
    action,
    action_mode,
    current_arm_qpos,
    *,
    previous_arm_command_qpos,
    planner,
    workspace,
    hand_qpos_min_rad,
    hand_qpos_max_rad,
):
    action = np.asarray(action)
    if action.shape != (physical_action_dim(action_mode),) or not np.isfinite(action).all():
        raise ValueError("physical action must have the expected shape and finite values")
    raw_hand = action[7:19] if action_mode == "joint" else action[9:21]
    hand = project_hand_command(
        raw_hand, qpos_min_rad=hand_qpos_min_rad, qpos_max_rad=hand_qpos_max_rad
    )
    hand_clip = float(np.max(np.abs(hand - raw_hand)))
    if action_mode == "joint":
        return action[:7], hand, None, hand_clip
    planner.set_hand_qpos(hand)
    position = np.clip(action[:3], workspace[:, 0], workspace[:, 1])
    intent = np.concatenate((position, action[3:9]))
    result = planner.solve_teleop_ik(
        Pose(p=position, q=rot6d_to_quat_wxyz(action[3:9])),
        current_arm_qpos,
        current_arm_qpos if previous_arm_command_qpos is None else previous_arm_command_qpos,
    )
    return result.qpos if result.success else None, hand, intent, hand_clip

"""Current raw teleop targets for nominal-rate physical replay."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.data.raw.reader import EpisodeReader
from dexmani_real.geometry.kinematics import eef_pose_sequence


@dataclass
class TrajectoryData:
    episode_path: str
    num_frames: int
    fps: float
    task_label: str
    action_arm_joint: np.ndarray
    action_hand_joint: np.ndarray
    arm_qpos: np.ndarray
    hand_qpos: np.ndarray
    arm_ee: np.ndarray


def load_trajectory(episode_path):
    with EpisodeReader(episode_path) as reader:
        meta = reader.meta
        if meta["collection_source"] != "teleop":
            raise ValueError("physical replay requires a teleop episode")
        reader.require_fields(
            "action_arm_joint_target",
            "action_hand_joint_target",
            "arm_qpos",
            "hand_qpos",
            "timestamp",
        )
        arm, hand, aq, hq = [
            reader[k][:]
            for k in (
                "action_arm_joint_target",
                "action_hand_joint_target",
                "arm_qpos",
                "hand_qpos",
            )
        ]
        if len(arm) == 0 or not all(np.isfinite(v).all() for v in (arm, hand, aq, hq)):
            raise ValueError("replay requires nonempty finite targets and robot states")
        reader.validate_completion()
        reader.validate_rows(required_tactile=(), require_camera=False)
        return TrajectoryData(
            str(reader.path),
            len(arm),
            reader.control_hz,
            meta["task_label"],
            arm,
            hand,
            aq,
            hq,
            eef_pose_sequence(aq),
        )


def verify_replay_preflight(trajectory, runtime):
    if not runtime.hand.enabled:
        raise ValueError("physical replay requires XHand")
    for values, lower, upper in (
        (trajectory.action_arm_joint, runtime.arm.joint_limit_lower, runtime.arm.joint_limit_upper),
        (trajectory.action_hand_joint, runtime.hand.qpos_min_rad, runtime.hand.qpos_max_rad),
    ):
        if (
            not np.isfinite(values).all()
            or np.any(values < np.asarray(lower) - 1e-9)
            or np.any(values > np.asarray(upper) + 1e-9)
        ):
            raise ValueError("recorded targets violate configured absolute limits")

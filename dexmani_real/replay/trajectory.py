"""Current raw teleop targets for nominal-rate physical replay."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.recording.storage.schema import validate_dispatch_rows


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
        collection_source = meta["collection_source"]
        if isinstance(collection_source, bytes):
            collection_source = collection_source.decode("utf-8")
        if collection_source != "teleop":
            raise ValueError("physical replay requires a teleop episode")
        reader.require_fields(
            "action_arm_joint_target",
            "action_hand_joint_target",
            "arm_qpos",
            "hand_qpos",
            "dispatch_status",
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
        errors = validate_dispatch_rows(reader["dispatch_status"][:])
        if errors:
            raise ValueError("; ".join(errors))
        timing = reader.read_timing()
        reader.validate_arm_execution(timing["rows"])
        return TrajectoryData(
            str(reader.path),
            len(arm),
            reader.control_hz,
            str(meta["task_label"]),
            arm,
            hand,
            aq,
            hq,
            compute_eef_pose_history_xarm_base(aq),
        )


def verify_replay_preflight(trajectory, runtime):
    if not runtime.control.hand_enabled:
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

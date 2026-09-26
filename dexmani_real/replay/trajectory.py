"""Current raw teleop targets for nominal-rate physical replay."""

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base
from dexmani_real.recording.storage.reader import EpisodeReader


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


def resolve_episode_path(raw_path: str) -> tuple[str, str]:
    """Validate and name one published episode directory."""
    path = Path(raw_path)
    if not path.is_dir():
        raise ValueError(f"episode must be a published directory: {path}")
    if not (path / "data.h5").is_file():
        raise ValueError(f"episode directory is missing data.h5: {path}")
    return str(path), path.name


def load_trajectory(episode_path):
    path, _ = resolve_episode_path(episode_path)
    with EpisodeReader(path) as reader:
        h5 = reader.h5f
        meta = h5["meta"].attrs
        collection_source = meta["collection_source"]
        if isinstance(collection_source, bytes):
            collection_source = collection_source.decode("utf-8")
        if collection_source != "teleop":
            raise ValueError("physical replay requires a teleop episode")
        if not bool(meta["episode_valid"]):
            raise ValueError("physical replay rejects an invalid episode")
        if not np.all(h5["frame_valid"][:]):
            raise ValueError("physical replay cannot reproduce failed control rows")
        arm, hand, aq, hq = [
            h5[k][:]
            for k in (
                "action_arm_joint_target",
                "action_hand_joint_target",
                "arm_qpos",
                "hand_qpos",
            )
        ]
        if len(arm) == 0 or not all(np.isfinite(v).all() for v in (arm, hand, aq, hq)):
            raise ValueError("replay requires nonempty finite targets and robot states")
        return TrajectoryData(
            path,
            len(arm),
            reader.control_hz,
            str(meta.get("task_label", "")),
            arm,
            hand,
            aq,
            hq,
            compute_eef_pose_history_xarm_base(aq),
        )


def verify_replay_preflight(trajectory, runtime):
    limiting_hz = min(runtime.arm.loop_hz, runtime.hand.loop_hz)
    if trajectory.fps > limiting_hz and not math.isclose(
        trajectory.fps, limiting_hz, rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError(
            f"Replay rate {trajectory.fps:g} Hz exceeds limiting worker rate {limiting_hz:g} Hz"
        )
    if not runtime.policy.hand_enabled:
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

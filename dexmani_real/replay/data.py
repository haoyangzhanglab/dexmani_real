"""Raw 轨迹读取与回放采样缓存。"""

from dataclasses import dataclass

import numpy as np

from dexmani_real.data.raw.reader import EpisodeReader
from dexmani_real.robot.model import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE


@dataclass
class Trajectory:
    episode_path: str
    fps: float
    task_label: str
    action_arm_joint: np.ndarray
    action_hand_joint: np.ndarray
    arm_qpos: np.ndarray
    hand_qpos: np.ndarray

    def __len__(self):
        return len(self.arm_qpos)


def load_trajectory(episode_path):
    with EpisodeReader(episode_path) as reader:
        meta = reader.meta
        if meta["collection_source"] != "teleop":
            raise ValueError("physical replay requires a teleop episode")
        reader.validate_completion()
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
        reader.validate_rows(required_tactile=(), require_camera=False)
        return Trajectory(
            str(reader.path),
            reader.control_hz,
            meta["task_label"],
            arm,
            hand,
            aq,
            hq,
        )


def verify_replay_preflight(trajectory, runtime):
    if not runtime.hand.enabled:
        raise ValueError("physical replay requires XHand")
    for values, lower, upper in (
        (trajectory.action_arm_joint, runtime.arm.joint_limit_lower, runtime.arm.joint_limit_upper),
        (trajectory.action_hand_joint, runtime.hand.qpos_min_rad, runtime.hand.qpos_max_rad),
    ):
        if np.any(values < np.asarray(lower) - 1e-9) or np.any(values > np.asarray(upper) + 1e-9):
            raise ValueError("recorded targets violate configured absolute limits")


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.count = 0
        self.data = {
            name: np.full((capacity, *shape), np.nan)
            for name, shape in (
                ("arm_qpos", ARM_JOINT_SHAPE),
                ("arm_candidate", ARM_JOINT_SHAPE),
                ("hand_qpos", HAND_JOINT_SHAPE),
                ("hand_candidate", HAND_JOINT_SHAPE),
            )
        }
        for name in (
            "arm_observation_ns",
            "hand_observation_ns",
            "submit_started_ns",
            "submit_completed_ns",
        ):
            self.data[name] = np.full(capacity, -1, dtype=np.int64)

    def append(self, row, command):
        index = self.count
        for name, state, target in (
            ("arm", row.arm, command.arm_qpos),
            ("hand", row.hand, command.hand_qpos),
        ):
            self.data[f"{name}_qpos"][index] = state["qpos"][0]
            self.data[f"{name}_observation_ns"][index] = state["timestamp_ns"][0]
            if target is not None:
                self.data[f"{name}_candidate"][index] = target
        self.count += 1
        return index

    def to_dict(self):
        return {name: values[: self.count].copy() for name, values in self.data.items()}

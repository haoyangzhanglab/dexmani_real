"""读取标称频率目标序列及首行实测起点，保存本次回放的真实采样。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dexmani_real.data.raw import RawReader, require_published_raw
from dexmani_real.robot.model import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.utils.io import publish


@dataclass
class Trajectory:
    episode_path: str
    fps: float
    task_label: str
    action_arm_joint: np.ndarray
    action_hand_joint: np.ndarray
    initial_arm_qpos: np.ndarray
    initial_hand_qpos: np.ndarray

    def __len__(self):
        return len(self.action_arm_joint)


def load_trajectory(episode_path):
    require_published_raw(episode_path)
    with RawReader(episode_path) as reader:
        meta = reader.meta
        arm = reader["action_arm_joint_target"][:]
        hand = reader["action_hand_joint_target"][:]
        if len(arm) == 0 or not all(np.isfinite(v).all() for v in (arm, hand)):
            raise ValueError("replay requires nonempty finite targets")
        initial_arm = reader["arm_qpos"][0]
        initial_hand = reader["hand_qpos"][0]
        if not all(np.isfinite(v).all() for v in (initial_arm, initial_hand)):
            raise ValueError("replay requires finite initial robot states")
        return Trajectory(
            str(reader.path),
            reader.control_hz,
            meta["task_label"],
            arm,
            hand,
            initial_arm,
            initial_hand,
        )


def validate_trajectory(trajectory, runtime):
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

    def append(self, row, arm_target, hand_target):
        index = self.count
        for name, state, target in (
            ("arm", row.arm, arm_target),
            ("hand", row.hand, hand_target),
        ):
            self.data[f"{name}_qpos"][index] = state["qpos"][0]
            self.data[f"{name}_observation_ns"][index] = state["timestamp_ns"][0]
            self.data[f"{name}_candidate"][index] = target
        self.count += 1
        return index

    def to_dict(self):
        return {name: values[: self.count].copy() for name, values in self.data.items()}


def save_replay_data(replay_data, output_dir):
    output = Path(output_dir)
    temporary = output / "replay_data.tmp.npz"
    np.savez_compressed(temporary, **replay_data)
    publish(temporary, output / "replay_data.npz")

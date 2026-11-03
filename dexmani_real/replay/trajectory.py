"""一次读取当前设备顺序的绝对关节目标；不依赖存档模态。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np

from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig


@dataclass
class Trajectory:
    control_hz: float
    action_arm_joint: np.ndarray
    action_hand_joint: np.ndarray

    def __len__(self) -> int:
        return len(self.action_arm_joint)


def load_trajectory(episode_path: str | Path, runtime: RealConfig) -> Trajectory:
    validate_robot_config(runtime)
    if not runtime.hand.enabled:
        raise ValueError("replay requires XHand")
    with h5py.File(Path(episode_path).expanduser() / "data.h5", "r") as data:
        meta = data["meta"].attrs
        if meta["joint_units"] != "rad":
            raise ValueError("replay joint_units must be rad")
        for field, names in (
            ("arm_joint_names", XARM7_JOINT_NAMES),
            ("hand_joint_names", XHAND_SDK_JOINT_NAMES),
        ):
            if json.loads(meta[field]) != list(names):
                raise ValueError(f"replay {field} order mismatch")
        hz = float(meta["control_hz"])
        validate_control_rate(runtime, hz)
        arrays = []
        for field, size, lower, upper in (
            (
                "action_arm_joint_target",
                7,
                runtime.arm.joint_limit_lower,
                runtime.arm.joint_limit_upper,
            ),
            (
                "action_hand_joint_target",
                12,
                runtime.hand.qpos_min_rad,
                runtime.hand.qpos_max_rad,
            ),
        ):
            values = data[field][:]
            if values.dtype.kind not in "iuf":
                raise ValueError(f"{field} must contain real numeric targets")
            values = values.astype(np.float64)
            if (
                values.ndim != 2
                or values.shape[1] != size
                or not len(values)
                or not np.isfinite(values).all()
            ):
                raise ValueError(
                    f"{field} requires nonempty finite targets of shape (N, {size})"
                )
            if np.any(values < lower) or np.any(values > upper):
                raise ValueError(f"{field} violates configured absolute limits")
            arrays.append(values)
        if len(arrays[0]) != len(arrays[1]):
            raise ValueError("replay target lengths differ")
    return Trajectory(hz, *arrays)

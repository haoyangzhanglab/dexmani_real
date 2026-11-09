"""校验当前 Raw 时间与动作契约，读取尝试提交的绝对关节目标。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.data.raw import (
    RawReader,
    require_continuous_recording,
    require_published_raw,
)
from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config

if TYPE_CHECKING:
    from dexmani_real.config.real import RealConfig


TARGET_LIMIT_TOLERANCE_RAD = 1e-6


@dataclass
class Trajectory:
    control_hz: float
    action_arm_joint: np.ndarray
    action_hand_joint: np.ndarray

    def __len__(self) -> int:
        return len(self.action_arm_joint)


def load_trajectory(episode_path: str | Path, runtime: RealConfig) -> Trajectory:
    """校验配置范围，并归一 1e-6 rad 内的文件边界舍入误差。"""
    validate_robot_config(runtime)
    if not runtime.hand.enabled:
        raise ValueError("replay requires XHand")
    require_published_raw(episode_path)
    with RawReader(episode_path) as reader:
        require_continuous_recording(reader.meta, reader["timing/rows"][:])
        hz = reader.control_hz
        validate_control_rate(runtime, hz)
        arrays = []
        for field, lower, upper in (
            (
                "action_arm_joint_target",
                runtime.arm.joint_limit_lower,
                runtime.arm.joint_limit_upper,
            ),
            (
                "action_hand_joint_target",
                runtime.hand.qpos_min_rad,
                runtime.hand.qpos_max_rad,
            ),
        ):
            # RawReader 统一检查当前动作语义、关节顺序及与时间行数一致的形状。
            values = reader[field][:].astype(np.float64)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f"{field} requires nonempty finite targets")
            lower_bound, upper_bound = np.asarray(lower), np.asarray(upper)
            if np.any(values < lower_bound - TARGET_LIMIT_TOLERANCE_RAD) or np.any(
                values > upper_bound + TARGET_LIMIT_TOLERANCE_RAD
            ):
                raise ValueError(f"{field} violates configured absolute limits")
            # 仅归一文件精度造成的边界误差；真正越界已在上面拒绝。
            np.clip(values, lower_bound, upper_bound, out=values)
            arrays.append(values)
    return Trajectory(hz, *arrays)

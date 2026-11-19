"""校验 Raw 时间与动作契约，读取已确认提交的绝对关节目标。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.data.raw import RawReader
from dexmani_real.robot.commands import validate_hand_target
from dexmani_real.runtime.preflight import (
    validate_robot_config,
    validate_xhand_target_rate,
)

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


def load_trajectory(episode_path: str | Path, runtime: RealConfig, *, segment: int | None = None) -> Trajectory:
    """按 Raw 顺序读取目标，逐行校验 XHand 操作限位。"""
    validate_robot_config(runtime)
    if not runtime.hand.enabled:
        raise ValueError("replay requires XHand")
    with RawReader(episode_path) as reader:
        reader.read_timing()
        starts = reader.segment_starts
        if not len(starts):
            raise ValueError("replay requires nonempty targets")
        if segment is None and len(starts) == 1:
            segment = 0
        ends = np.r_[starts[1:], reader.num_frames]
        if segment is None or not 0 <= segment < len(starts):
            choices = "; ".join(
                f"{i}: rows [{int(start)}, {int(end)}) ({int(end - start)} rows)"
                for i, (start, end) in enumerate(zip(starts, ends, strict=True))
            )
            raise ValueError(f"select one continuous segment with --segment (zero-based): {choices}")
        start, end = int(starts[segment]), int(ends[segment])
        reader.require_confirmed_actions(start, end)
        hz = reader.control_hz
        validate_xhand_target_rate(hz)
        # RawReader 检查绝对目标语义、关节顺序、单位及行形状。
        arm = reader["action_arm_joint_target"][start:end]
        finite = np.isfinite(arm).all(axis=1)
        if not finite.all():
            raise ValueError(f"action_arm_joint_target requires finite targets at row {start + np.flatnonzero(~finite)[0]}")
        lower, upper = np.asarray(runtime.arm.joint_limit_lower), np.asarray(runtime.arm.joint_limit_upper)
        outside = (arm < lower - TARGET_LIMIT_TOLERANCE_RAD) | (arm > upper + TARGET_LIMIT_TOLERANCE_RAD)
        if outside.any():
            row = start + np.flatnonzero(outside.any(axis=1))[0]
            raise ValueError(f"action_arm_joint_target violates configured absolute limits at row {row}")
        np.clip(arm, lower, upper, out=arm)

        hand = reader["action_hand_joint_target"][start:end]
        lower, upper = np.asarray(runtime.hand.qpos_min_rad), np.asarray(runtime.hand.qpos_max_rad)
        for row, target in enumerate(hand, start=start):
            try:
                validate_hand_target(target, lower, upper)
            except ValueError as exc:
                raise ValueError(f"action_hand_joint_target at row {row}: {exc}") from exc
    return Trajectory(hz, arm, hand)

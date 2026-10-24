"""In-memory capture of measured robot state during one physical replay."""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE


class ReplayRecorder:
    """Pre-allocated, single-threaded capture buffer for replay evaluation."""

    def __init__(self, capacity: int) -> None:
        self.count = 0
        self.arm_qpos = np.full((capacity, *ARM_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.arm_candidate = np.full((capacity, *ARM_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.arm_observation_ns = np.full(capacity, -1, dtype=np.int64)
        self.hand_observation_ns = np.full(capacity, -1, dtype=np.int64)
        self.submit_started_ns = np.full(capacity, -1, dtype=np.int64)
        self.submit_completed_ns = np.full(capacity, -1, dtype=np.int64)
        self.hand_qpos = np.full((capacity, *HAND_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.dispatch_status = np.zeros((capacity, 2), dtype=np.uint8)
        self.hand_candidate = np.full((capacity, *HAND_JOINT_SHAPE), np.nan, dtype=np.float64)

    def record(
        self,
        idx: int,
        row,
        command,
        dispatch,
        *,
        submit_started_ns: int = -1,
        submit_completed_ns: int = -1,
    ) -> None:
        """记录候选与真实 dispatch；未发送候选不能成为已提交命令。"""
        if idx != self.count or idx >= len(self.arm_qpos):
            raise ValueError("replay rows must be sequential and within capacity")
        self.dispatch_status[idx] = int(dispatch.arm), int(dispatch.hand)
        self.arm_qpos[idx] = row.arm["qpos"][0]
        if command.arm_qpos is not None:
            self.arm_candidate[idx] = command.arm_qpos
        # Source clocks belong to the consumed samples; submission does not imply arrival.
        self.arm_observation_ns[idx] = row.arm["timestamp_ns"][0]
        self.hand_observation_ns[idx] = row.hand["timestamp_ns"][0]
        self.submit_started_ns[idx] = submit_started_ns
        self.submit_completed_ns[idx] = submit_completed_ns
        self.hand_qpos[idx] = row.hand["qpos"][0]
        if command.hand_qpos is not None:
            self.hand_candidate[idx] = command.hand_qpos
        self.count = idx + 1

    def to_dict(self) -> dict[str, np.ndarray]:
        """导出实测前缀和候选；是否发送由同一行的 dispatch 表达。"""
        count = self.count
        return {
            "dispatch_status": self.dispatch_status[:count].copy(),
            "arm_qpos": self.arm_qpos[:count].copy(),
            "hand_qpos": self.hand_qpos[:count].copy(),
            "hand_candidate": self.hand_candidate[:count].copy(),
            "arm_candidate": self.arm_candidate[:count].copy(),
            "arm_observation_ns": self.arm_observation_ns[:count].copy(),
            "hand_observation_ns": self.hand_observation_ns[:count].copy(),
            "submit_started_ns": self.submit_started_ns[:count].copy(),
            "submit_completed_ns": self.submit_completed_ns[:count].copy(),
        }

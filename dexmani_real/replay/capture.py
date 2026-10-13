"""In-memory capture of measured robot state during one physical replay."""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE


class ReplayRecorder:
    """Pre-allocated, single-threaded capture buffer for replay evaluation."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._count = 0
        self.arm_qpos = np.full((capacity, *ARM_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.arm_cmd = np.full((capacity, *ARM_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.arm_observation_ns = np.full(capacity, -1, dtype=np.int64)
        self.hand_observation_ns = np.full(capacity, -1, dtype=np.int64)
        self.submit_started_ns = np.full(capacity, -1, dtype=np.int64)
        self.submit_completed_ns = np.full(capacity, -1, dtype=np.int64)
        self.hand_qpos = np.full((capacity, *HAND_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.dispatch_status = np.zeros((capacity, 2), dtype=np.uint8)
        self.hand_cmd = np.full((capacity, *HAND_JOINT_SHAPE), np.nan, dtype=np.float64)

    def record(
        self,
        idx: int,
        arm_qpos: np.ndarray,
        arm_cmd: np.ndarray,
        hand_cmd: np.ndarray,
        *,
        hand_qpos: np.ndarray,
        dispatch_status: tuple[int, int],
        arm_observation_ns: int,
        hand_observation_ns: int,
        submit_started_ns: int = -1,
        submit_completed_ns: int = -1,
    ) -> None:
        """Capture one replay row, preserving rejected candidates for diagnosis."""
        if idx < 0:
            raise ValueError("replay frame index must be non-negative")
        if idx >= self.capacity:
            return
        self.dispatch_status[idx] = dispatch_status
        self.arm_qpos[idx] = arm_qpos
        self.arm_cmd[idx] = arm_cmd
        # Source clocks belong to the consumed samples; submission does not imply arrival.
        self.arm_observation_ns[idx] = arm_observation_ns
        self.hand_observation_ns[idx] = hand_observation_ns
        self.submit_started_ns[idx] = submit_started_ns
        self.submit_completed_ns[idx] = submit_completed_ns
        self.hand_qpos[idx] = hand_qpos
        self.hand_cmd[idx] = hand_cmd
        self._count = idx + 1

    def to_dict(self) -> dict[str, np.ndarray]:
        """Copy the physical prefix with NaN EEF fields pending offline derivation."""
        count = self._count
        return {
            "dispatch_status": self.dispatch_status[:count].copy(),
            "arm_qpos": self.arm_qpos[:count].copy(),
            "hand_qpos": self.hand_qpos[:count].copy(),
            "hand_cmd": self.hand_cmd[:count].copy(),
            # Preserve output fields before any fallible offline pose derivation.
            "eef_pos": np.full((count, 3), np.nan, dtype=np.float64),
            "eef_quat_wxyz": np.full((count, 4), np.nan, dtype=np.float64),
            "eef_rot6d": np.full((count, 6), np.nan, dtype=np.float64),
            "arm_cmd": self.arm_cmd[:count].copy(),
            "arm_observation_ns": self.arm_observation_ns[:count].copy(),
            "hand_observation_ns": self.hand_observation_ns[:count].copy(),
            "submit_started_ns": self.submit_started_ns[:count].copy(),
            "submit_completed_ns": self.submit_completed_ns[:count].copy(),
        }

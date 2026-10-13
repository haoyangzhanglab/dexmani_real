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
        self.arm_tracking_error = np.full(capacity, np.nan, dtype=np.float64)
        self.timestamps = np.full(capacity, np.nan, dtype=np.float64)
        self.hand_qpos = np.full((capacity, *HAND_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.dispatch_status = np.zeros((capacity, 2), dtype=np.uint8)
        self.hand_cmd = np.full((capacity, *HAND_JOINT_SHAPE), np.nan, dtype=np.float64)

    def record(
        self,
        idx: int,
        arm_qpos: np.ndarray,
        arm_cmd: np.ndarray,
        hand_cmd: np.ndarray,
        ts: float,
        *,
        hand_qpos: np.ndarray,
        dispatch_status: tuple[int, int],
        arm_tracking_error: float | None = None,
    ) -> None:
        """Capture one replay row, preserving rejected candidates for diagnosis."""
        if idx < 0:
            raise ValueError("replay frame index must be non-negative")
        if idx >= self.capacity:
            return
        self.dispatch_status[idx] = dispatch_status
        self.arm_qpos[idx] = arm_qpos
        self.arm_cmd[idx] = arm_cmd
        self.timestamps[idx] = ts
        if arm_tracking_error is not None:
            self.arm_tracking_error[idx] = arm_tracking_error
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
            "arm_tracking_error": self.arm_tracking_error[:count].copy(),
            "timestamp": self.timestamps[:count].copy(),
        }

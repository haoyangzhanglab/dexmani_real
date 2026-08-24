"""In-memory capture of measured robot state during one physical replay."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dexmani_real.planning.poses import rot6d_to_rotmat
from dexmani_real.robot_spec import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE

_SAFETY_REASON_BYTES = 256


class ReplayRecorder:
    """Pre-allocated, single-threaded capture buffer for replay evaluation."""

    def __init__(self, capacity: int, has_hand: bool = False) -> None:
        self.capacity = capacity
        self.has_hand = has_hand
        self._count = 0
        self.arm_qpos = np.full((capacity, *ARM_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.eef_pos = np.full((capacity, 3), np.nan, dtype=np.float64)
        self.eef_quat_wxyz = np.full((capacity, 4), np.nan, dtype=np.float64)
        self.eef_rot6d = np.full((capacity, 6), np.nan, dtype=np.float64)
        self.arm_cmd = np.full((capacity, *ARM_JOINT_SHAPE), np.nan, dtype=np.float64)
        self.arm_sent_cmd = np.full(
            (capacity, *ARM_JOINT_SHAPE), np.nan, dtype=np.float64
        )
        self.arm_tracking_error = np.full(capacity, np.nan, dtype=np.float64)
        self.timestamps = np.full(capacity, np.nan, dtype=np.float64)
        self.flag_safety_reject = np.zeros(capacity, dtype=bool)
        self.safety_reject_reason: list[str | None] = [None] * capacity
        self.hand_qpos: np.ndarray | None = None
        self.hand_cmd: np.ndarray | None = None
        if has_hand:
            self.hand_qpos = np.full(
                (capacity, *HAND_JOINT_SHAPE), np.nan, dtype=np.float64
            )
            self.hand_cmd = np.full(
                (capacity, *HAND_JOINT_SHAPE), np.nan, dtype=np.float64
            )

    def record(
        self,
        idx: int,
        arm_qpos: np.ndarray,
        eef_pos: np.ndarray,
        eef_rot6d: np.ndarray,
        arm_cmd: np.ndarray,
        hand_cmd: np.ndarray | None,
        ts: float,
        arm_sent_cmd: np.ndarray | None = None,
        arm_tracking_error: float | None = None,
        safety_reject_reason: str | None = None,
        hand_qpos: np.ndarray | None = None,
    ) -> None:
        """Capture one replay row, preserving rejected candidates for diagnosis."""
        if idx < 0:
            raise ValueError("replay frame index must be non-negative")
        if idx >= self.capacity:
            return
        self.arm_qpos[idx] = arm_qpos
        self.eef_pos[idx] = eef_pos
        self.eef_rot6d[idx] = eef_rot6d
        try:
            quat_xyzw = Rotation.from_matrix(rot6d_to_rotmat(eef_rot6d)).as_quat()
            self.eef_quat_wxyz[idx] = np.array(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
            )
        except ValueError:
            self.eef_quat_wxyz[idx] = np.full(4, np.nan)
        self.arm_cmd[idx] = arm_cmd
        self.timestamps[idx] = ts
        if arm_sent_cmd is not None:
            self.arm_sent_cmd[idx] = arm_sent_cmd
        if arm_tracking_error is not None:
            self.arm_tracking_error[idx] = arm_tracking_error
        if self.hand_qpos is not None and hand_qpos is not None:
            self.hand_qpos[idx] = hand_qpos
        if self.hand_cmd is not None and hand_cmd is not None:
            self.hand_cmd[idx] = hand_cmd
        if safety_reject_reason is not None:
            self.flag_safety_reject[idx] = True
            self.safety_reject_reason[idx] = safety_reject_reason
        self._count = idx + 1

    @property
    def count(self) -> int:
        return self._count

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return copies of the populated prefix without pickle-only values."""
        count = self._count
        reasons = np.array(
            [
                reason.encode() if reason else b""
                for reason in self.safety_reject_reason[:count]
            ],
            dtype=f"S{_SAFETY_REASON_BYTES}",
        )
        result = {
            "arm_qpos": self.arm_qpos[:count].copy(),
            "eef_pos": self.eef_pos[:count].copy(),
            "eef_quat_wxyz": self.eef_quat_wxyz[:count].copy(),
            "eef_rot6d": self.eef_rot6d[:count].copy(),
            "arm_cmd": self.arm_cmd[:count].copy(),
            "arm_sent_cmd": self.arm_sent_cmd[:count].copy(),
            "arm_tracking_error": self.arm_tracking_error[:count].copy(),
            "timestamp": self.timestamps[:count].copy(),
            "flag_safety_reject": self.flag_safety_reject[:count].copy(),
            "safety_reject_reason": reasons,
        }
        if self.hand_qpos is not None:
            result["hand_qpos"] = self.hand_qpos[:count].copy()
        if self.hand_cmd is not None:
            result["hand_cmd"] = self.hand_cmd[:count].copy()
        return result

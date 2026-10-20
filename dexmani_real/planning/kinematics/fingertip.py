"""Shared xArm-base fingertip geometry for offline processing and deployment."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rotmat, rot6d_to_rotmat
from dexmani_real.robot.model import (
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
)


def compute_fingertip_history_xarm_base(
    eef_pose_history: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    hand_fk: Any,
    handbase_position_eef_m: np.ndarray | tuple[float, ...],
    handbase_quat_eef_wxyz: np.ndarray | tuple[float, ...],
) -> np.ndarray:
    """Return aligned float32 [T,5,3] base-frame tips from the consumed EEF history."""
    poses = np.asarray(eef_pose_history, dtype=np.float64)
    hand = np.asarray(hand_qpos, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 9 or hand.shape != (len(poses), HAND_JOINT_SHAPE[0]):
        raise ValueError("aligned EEF/hand histories have invalid shapes")
    if not np.isfinite(poses).all() or not np.isfinite(hand).all():
        raise ValueError("EEF/hand histories must be finite")
    mount_p = np.asarray(handbase_position_eef_m, dtype=np.float64)
    if mount_p.shape != (3,) or not np.isfinite(mount_p).all():
        raise ValueError("hand mount position must be finite shape (3,)")
    mount_rotation = quat_wxyz_to_rotmat(handbase_quat_eef_wxyz)
    tips = np.empty((len(poses), *HAND_FINGERTIP_SHAPE), dtype=np.float32)
    for index, pose in enumerate(poses):
        rotation = rot6d_to_rotmat(pose[3:])
        tips_hand = np.asarray(
            hand_fk.compute_tip_positions_in_handbase(hand[index]), dtype=np.float64
        )
        if tips_hand.shape != HAND_FINGERTIP_SHAPE or not np.isfinite(tips_hand).all():
            raise RuntimeError("hand FK produced malformed fingertip positions")
        # Row vectors: hand base -> EEF -> xArm base. Keep float64 until storage.
        tips[index] = (tips_hand @ mount_rotation.T + mount_p) @ rotation.T + pose[:3]
    if not np.isfinite(tips).all():
        raise RuntimeError("fingertip transform produced non-finite positions")
    return tips

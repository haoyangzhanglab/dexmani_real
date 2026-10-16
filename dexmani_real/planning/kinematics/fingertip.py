"""Shared arm-base fingertip geometry for recording and deployment."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base
from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rotmat, rot6d_to_rotmat
from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
)


def compute_fingertip_history_xarm_base(
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    hand_fk: Any,
    handbase_position_eef_m: np.ndarray,
    handbase_quat_eef_wxyz: np.ndarray,
    arm_fk: Any | None = None,
    eef_pose_history: np.ndarray | None = None,
) -> np.ndarray:
    """Return aligned float32 [T,5,3] base-frame tips, reusing eef_pose_history or computing one Arm FK per row."""
    arm = np.asarray(arm_qpos, dtype=np.float64)
    hand = np.asarray(hand_qpos, dtype=np.float64)
    if (
        arm.ndim != 2
        or arm.shape[1] != ARM_JOINT_SHAPE[0]
        or hand.shape != (len(arm), HAND_JOINT_SHAPE[0])
    ):
        raise ValueError("aligned arm/hand qpos histories have invalid shapes")
    if not np.isfinite(arm).all() or not np.isfinite(hand).all():
        raise ValueError("arm/hand qpos histories must be finite")
    mount_p = np.asarray(handbase_position_eef_m, dtype=np.float64)
    if mount_p.shape != (3,) or not np.isfinite(mount_p).all():
        raise ValueError("hand mount position must be finite shape (3,)")
    mount_rotation = quat_wxyz_to_rotmat(handbase_quat_eef_wxyz)
    if eef_pose_history is not None:
        poses = np.asarray(eef_pose_history, dtype=np.float64)
        if poses.shape != (len(arm), 9) or not np.all(np.isfinite(poses)):
            raise ValueError("eef_pose_history must be finite with shape (T, 9)")
    else:
        if arm_fk is None:
            raise ValueError("arm_fk is required when eef_pose_history is not provided")
        poses = compute_eef_pose_history_xarm_base(arm, arm_fk=arm_fk)
    tips = np.empty((len(arm), *HAND_FINGERTIP_SHAPE), dtype=np.float32)
    for index, pose in enumerate(poses):
        rotation = rot6d_to_rotmat(pose[3:])
        tips_hand = np.asarray(hand_fk.compute_tip_positions_in_handbase(hand[index]), dtype=np.float64)
        if tips_hand.shape != HAND_FINGERTIP_SHAPE or not np.isfinite(tips_hand).all():
            raise RuntimeError("hand FK produced malformed fingertip positions")
        # Row vectors: hand base -> EEF -> xArm base. Keep float64 until storage.
        tips[index] = (tips_hand @ mount_rotation.T + mount_p) @ rotation.T + pose[:3]
    if not np.isfinite(tips).all():
        raise RuntimeError("fingertip transform produced non-finite positions")
    return tips

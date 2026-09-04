"""Shared arm-base fingertip geometry for recording and deployment."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.ipc.schema import (
    ARM_JOINT_SHAPE,
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
)
from dexmani_real.planning.poses import (
    compose_pose,
    normalize_quat_wxyz,
    quat_wxyz_to_rotmat,
    rot6d_to_quat_wxyz,
)
from dexmani_real.planning.types import Pose


def compute_fingertip_points_xarm_base(
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    arm_fk: Any | None,
    hand_fk: Any,
    handbase_position_eef_m: np.ndarray,
    handbase_quat_eef_wxyz: np.ndarray,
    eef_position_xarm_base_m: np.ndarray | None = None,
    eef_rot6d_xarm_base: np.ndarray | None = None,
) -> np.ndarray:
    """Return fingertips in xArm base, preserving SDK finger order.

    Callers that already computed the EEF pose may provide both EEF fields to
    avoid a duplicate arm FK. Otherwise the pose is derived from ``arm_qpos``.
    """
    arm = np.asarray(arm_qpos, dtype=np.float64)
    hand = np.asarray(hand_qpos, dtype=np.float64)
    mount_p = np.asarray(handbase_position_eef_m, dtype=np.float64)
    mount_q = normalize_quat_wxyz(handbase_quat_eef_wxyz)
    if arm.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(arm)):
        raise ValueError(f"arm_qpos must be finite {ARM_JOINT_SHAPE}")
    if hand.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(hand)):
        raise ValueError(f"hand_qpos must be finite {HAND_JOINT_SHAPE}")
    if mount_p.shape != (3,) or not np.all(np.isfinite(mount_p)):
        raise ValueError("hand mount position must be finite shape (3,)")
    if not hand_fk.is_ready():
        raise RuntimeError("hand fingertip FK is not ready")
    if (eef_position_xarm_base_m is None) != (eef_rot6d_xarm_base is None):
        raise ValueError("EEF position and rot6d must be provided together")
    if eef_position_xarm_base_m is None:
        if arm_fk is None:
            raise ValueError("arm_fk is required when the EEF pose is not provided")
        eef_pos, eef_rot6d = arm_fk.compute(arm)
    else:
        eef_pos = np.asarray(eef_position_xarm_base_m, dtype=np.float64)
        eef_rot6d = np.asarray(eef_rot6d_xarm_base, dtype=np.float64)
        if eef_pos.shape != (3,) or not np.all(np.isfinite(eef_pos)):
            raise ValueError("EEF position must be finite shape (3,)")
        if eef_rot6d.shape != (6,) or not np.all(np.isfinite(eef_rot6d)):
            raise ValueError("EEF rot6d must be finite shape (6,)")
    eef_pose = Pose(
        p=np.asarray(eef_pos, dtype=np.float64), q=rot6d_to_quat_wxyz(eef_rot6d)
    )
    handbase_pose = compose_pose(eef_pose, Pose(p=mount_p, q=mount_q))
    tips_hand = np.asarray(
        hand_fk.compute_tip_positions_in_handbase(hand), dtype=np.float64
    )
    if tips_hand.shape != HAND_FINGERTIP_SHAPE or not np.all(np.isfinite(tips_hand)):
        raise RuntimeError("hand FK produced malformed fingertip positions")
    rotation = quat_wxyz_to_rotmat(handbase_pose.q)
    tips = tips_hand @ rotation.T + handbase_pose.p
    if tips.shape != HAND_FINGERTIP_SHAPE or not np.all(np.isfinite(tips)):
        raise RuntimeError("fingertip transform produced malformed positions")
    return tips


__all__ = ["compute_fingertip_points_xarm_base"]

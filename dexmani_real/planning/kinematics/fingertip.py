"""Shared xArm-base fingertip and hand-cloud geometry from measured joint histories."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.planning.kinematics.pose import quat_wxyz_to_rotmat, rot6d_to_rotmat
from dexmani_real.robot.model import (
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
)


def compute_hand_geometry_history_xarm_base(
    eef_pose_history: np.ndarray,
    hand_qpos: np.ndarray,
    *,
    hand_fk: Any,
    handbase_position_eef_m: np.ndarray | tuple[float, ...],
    handbase_quat_eef_wxyz: np.ndarray | tuple[float, ...],
    hand_cloud=None,
    include_fingertips: bool = True,
) -> dict[str, np.ndarray]:
    """Derive requested base-frame tips/cloud with one hand FK update per measured row."""
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
    if not include_fingertips and hand_cloud is None:
        return {}
    frame_ids = list(hand_fk.fingertip_frame_ids) if include_fingertips else []
    tip_count = len(frame_ids)
    if hand_cloud is not None:
        frame_ids.extend(hand_cloud.frame_ids)
    arrays = {}
    if include_fingertips:
        arrays["fingertip_xyz"] = np.empty((len(poses), *HAND_FINGERTIP_SHAPE), dtype=np.float32)
    if hand_cloud is not None:
        arrays["hand_pointcloud"] = np.empty((len(poses), len(hand_cloud.points), 6), dtype=np.float32)
    for index, pose in enumerate(poses):
        rotation = rot6d_to_rotmat(pose[3:])
        base_from_hand = np.eye(4)
        base_from_hand[:3, :3] = rotation @ mount_rotation
        base_from_hand[:3, 3] = rotation @ mount_p + pose[:3]
        placements = hand_fk.compute_frame_placements(hand[index], frame_ids)
        if include_fingertips:
            tips_hand = placements[:tip_count, :3, 3]
            # Row vectors: hand base -> xArm base. Keep float64 until storage.
            arrays["fingertip_xyz"][index] = (
                tips_hand @ base_from_hand[:3, :3].T + base_from_hand[:3, 3]
            )
        if hand_cloud is not None:
            arrays["hand_pointcloud"][index] = hand_cloud.transform(
                placements[tip_count:], base_from_hand
            )
    if include_fingertips and not np.isfinite(arrays["fingertip_xyz"]).all():
        raise RuntimeError("fingertip geometry is not finite float32")
    return arrays

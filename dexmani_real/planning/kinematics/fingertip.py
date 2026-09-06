"""Shared arm-base fingertip geometry for recording and deployment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any

import numpy as np

from dexmani_real.ipc.schema import (
    ARM_JOINT_SHAPE,
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
)
from dexmani_real.planning.kinematics.pose import (
    Pose,
    compose_pose,
    normalize_quat_wxyz,
    quat_wxyz_to_rotmat,
    rot6d_to_quat_wxyz,
)

FINGERTIP_POINTS_DERIVATION = "fk_from_processed_joint_state"
FINGERTIP_POLICY_ID = "arm_hand_fk_from_joint_state_v1"


def _require_sha256(value: str, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_mount_rotation_row_major(quat_wxyz: np.ndarray) -> list[float]:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("hand mount quaternion must be finite shape (4,)")
    if float(np.linalg.norm(quat)) <= 1e-12:
        raise ValueError("hand mount quaternion norm must be positive")
    rotation = np.round(quat_wxyz_to_rotmat(normalize_quat_wxyz(quat)), decimals=12)
    rotation[np.abs(rotation) < 5e-13] = 0.0
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("hand mount rotation is invalid")
    return [float(value) for value in rotation.reshape(-1)]


def compute_fingertip_geometry_sha256(
    *,
    arm_fk_urdf_sha256: str,
    arm_eef_frame: str,
    hand_fk_urdf_sha256: str,
    hand_sdk_to_urdf_idx: Sequence[int],
    fingertip_link_names: Sequence[str],
    handbase_position_eef_m: np.ndarray,
    handbase_quat_eef_wxyz: np.ndarray,
) -> str:
    """Return the stable identity of the geometry used by fingertip FK.

    File hashing belongs to the caller so processing can freeze the identity at
    the same boundary that generates the persisted fingertip points.
    """
    mount_position = np.asarray(handbase_position_eef_m, dtype=np.float64)
    if mount_position.shape != (3,) or not np.all(np.isfinite(mount_position)):
        raise ValueError("hand mount position must be finite shape (3,)")
    if not isinstance(arm_eef_frame, str) or not arm_eef_frame.strip():
        raise ValueError("arm_eef_frame must be non-empty")
    mapping = tuple(int(value) for value in hand_sdk_to_urdf_idx)
    if len(mapping) != 12 or set(mapping) != set(range(12)):
        raise ValueError("hand_sdk_to_urdf_idx must be a 12-joint permutation")
    links = tuple(fingertip_link_names)
    if len(links) != 5 or any(
        not isinstance(name, str) or not name.strip() for name in links
    ):
        raise ValueError("fingertip_link_names must contain five non-empty names")
    payload = {
        "arm_eef_frame": arm_eef_frame,
        "arm_fk_urdf_sha256": _require_sha256(
            arm_fk_urdf_sha256, label="arm_fk_urdf_sha256"
        ),
        "fingertip_link_names": list(links),
        "hand_fk_urdf_sha256": _require_sha256(
            hand_fk_urdf_sha256, label="hand_fk_urdf_sha256"
        ),
        "hand_sdk_to_urdf_idx": list(mapping),
        "handbase_position_eef_m": [float(value) for value in mount_position],
        "handbase_rotation_eef_row_major": _canonical_mount_rotation_row_major(
            handbase_quat_eef_wxyz
        ),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


__all__ = [
    "FINGERTIP_POINTS_DERIVATION",
    "FINGERTIP_POLICY_ID",
    "compute_fingertip_geometry_sha256",
    "compute_fingertip_points_xarm_base",
]

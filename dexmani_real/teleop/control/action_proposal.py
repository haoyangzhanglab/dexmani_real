"""Compute Cartesian teleoperation intent without side effects.

TeleopController supplies the mapped pose and previous accepted target.
Target dispatch and recording belong to execute_control_step.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.planning.kinematics.pose import quat_multiply
from dexmani_real.utils.geometry import normalize_quat_wxyz


@dataclass(frozen=True)
class EefTargetProposal:
    """One xarm_base EEF target after explicit smoothing."""

    position_xarm_base_m: np.ndarray
    quat_xarm_base_wxyz: np.ndarray


def _quat_to_rotvec(q: np.ndarray) -> np.ndarray:
    """Convert a normalized wxyz quaternion to a shortest-arc rotation vector."""
    sign = np.asarray(q, dtype=np.float64)
    if sign[0] < 0:
        sign = -sign
    w, x, y, z = np.clip(sign[0], -1.0, 1.0), sign[1], sign[2], sign[3]
    sin_half = np.sqrt(x * x + y * y + z * z)
    if sin_half < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(sin_half, w)
    return angle * np.array([x, y, z], dtype=np.float64) / sin_half


def ema_smooth_pose(
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
    prev_pos: np.ndarray,
    prev_quat_wxyz: np.ndarray,
    alpha_pos: float,
    alpha_rot: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth an EEF target before IK with independent position/rotation EMA.

    Positions are (3,) meters; quaternions are (4,) WXYZ. Rotation scales the
    shortest relative rotation vector from the previous smoothed orientation.
    Each alpha must be in [0, 1]; 1 disables smoothing for that component.
    Returns float64 (position, quaternion) arrays.
    """
    if not (0 <= alpha_pos <= 1 and 0 <= alpha_rot <= 1):
        raise ValueError("EMA alpha must be in [0, 1]")

    pos = alpha_pos * np.asarray(target_pos, dtype=np.float64) + (1.0 - alpha_pos) * np.asarray(
        prev_pos, dtype=np.float64
    )

    # Interpolate relative rotation and choose the quaternion sign from adjacency.
    prev_quat = normalize_quat_wxyz(prev_quat_wxyz, name="prev_quat_wxyz")
    target_quat = normalize_quat_wxyz(target_quat_wxyz, name="target_quat_wxyz")
    if float(np.dot(prev_quat, target_quat)) < 0.0:
        target_quat = -target_quat
    prev_conjugate = prev_quat * np.array([1.0, -1.0, -1.0, -1.0])
    relative_quat = normalize_quat_wxyz(quat_multiply(prev_conjugate, target_quat), name="relative")
    rv = alpha_rot * _quat_to_rotvec(relative_quat)

    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        quat = prev_quat.copy()
    else:
        axis = rv / angle
        half = angle / 2.0
        delta_quat = np.array(
            [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
            dtype=np.float64,
        )
        quat = normalize_quat_wxyz(quat_multiply(prev_quat, delta_quat), name="smoothed")

    return pos, quat


def _finite_vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array.copy()


def compute_target_eef_pose(
    mapped_position_xarm_base_m: np.ndarray,
    mapped_quat_xarm_base_wxyz: np.ndarray,
    *,
    previous_position_xarm_base_m: np.ndarray | None,
    previous_quat_xarm_base_wxyz: np.ndarray | None,
    ema_alpha_position: float,
    ema_alpha_rotation: float,
) -> EefTargetProposal:
    """Smooth one mapped xarm_base EEF target."""
    position_xarm_base_m = _finite_vector(
        mapped_position_xarm_base_m, (3,), "mapped_position_xarm_base_m"
    )
    quat_xarm_base_wxyz = normalize_quat_wxyz(mapped_quat_xarm_base_wxyz)
    if previous_position_xarm_base_m is not None and previous_quat_xarm_base_wxyz is not None:
        position_xarm_base_m, quat_xarm_base_wxyz = ema_smooth_pose(
            position_xarm_base_m,
            quat_xarm_base_wxyz,
            previous_position_xarm_base_m,
            previous_quat_xarm_base_wxyz,
            ema_alpha_position,
            ema_alpha_rotation,
        )

    return EefTargetProposal(
        position_xarm_base_m=position_xarm_base_m,
        quat_xarm_base_wxyz=quat_xarm_base_wxyz,
    )

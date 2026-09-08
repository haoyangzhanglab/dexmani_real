"""Compute bounded teleoperation action proposals without side effects.

The control-grid owner supplies observations and temporal state, then remains
responsible for safety-gated command publication and recording. Keeping this
module pure makes proposal behavior testable without shared memory or hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dexmani_real.teleop.control_loop.hand_control import (
    HandRetargetObservationCache,
    compute_hand_command,
    smoothstep_hand_ramp,
)
from dexmani_real.utils.limits import limit_hand_target_delta


@dataclass(frozen=True)
class EefTargetProposal:
    """One mapped EEF target in the world frame, before and after filtering."""

    position_world_m: np.ndarray
    quat_world_wxyz: np.ndarray
    smoothing_state_incomplete: bool


@dataclass(frozen=True)
class HandJointProposal:
    """One hand proposal plus the next ramp state; no command is published."""

    qpos_rad: np.ndarray
    retarget_succeeded: bool
    next_ramp_start_qpos_rad: np.ndarray | None
    next_ramp_step: int


@dataclass(frozen=True)
class ArmJointProposal:
    """One IK result after firmware limits and command-to-command step limits."""

    qpos_rad: np.ndarray
    validation_issue: str | None


def _normalize_quat(q: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(q, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite (4,) quaternion")
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError(f"{name} quaternion norm is too small")
    return value / norm


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


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
    """EMA in Cartesian space: position R³ + rotation vector so(3).

    Smooths a 6-DOF EEF target pose before IK.  Position uses standard
    Euclidean EMA; orientation converts the quaternion to a rotation
    vector (axis * angle), applies EMA in so(3), then converts back.

    Rotation-vector EMA naturally takes the short geodesic path on S³
    (magnitude = angle ∈ [0, π]) without the overhead of scipy Slerp.

    Position and rotation are smoothed with independent factors because
    they have different noise profiles and human motion bandwidths:
    position benefits from higher α (lower latency), rotation from lower
    α (stronger filtering of orientation jitter).

    Args:
        target_pos: (3,) target EEF position in meters.
        target_quat_wxyz: (4,) target EEF orientation quaternion (w, x, y, z).
        prev_pos: (3,) previous smoothed position.
        prev_quat_wxyz: (4,) previous smoothed orientation quaternion.
        alpha_pos: Smoothing factor for position in [0, 1].  1.0 = no smoothing.
        alpha_rot: Smoothing factor for rotation in [0, 1].  1.0 = no smoothing.

    Returns:
        ``(pos_smoothed, quat_wxyz_smoothed)`` — both float64 copies.
    """
    alpha_pos = float(np.clip(alpha_pos, 0.0, 1.0))
    alpha_rot = float(np.clip(alpha_rot, 0.0, 1.0))

    pos = alpha_pos * np.asarray(target_pos, dtype=np.float64) + (1.0 - alpha_pos) * np.asarray(
        prev_pos, dtype=np.float64
    )

    # Interpolate relative rotation and choose the quaternion sign from adjacency.
    prev_quat = _normalize_quat(prev_quat_wxyz, name="prev_quat_wxyz")
    target_quat = _normalize_quat(target_quat_wxyz, name="target_quat_wxyz")
    if float(np.dot(prev_quat, target_quat)) < 0.0:
        target_quat = -target_quat
    prev_conjugate = prev_quat * np.array([1.0, -1.0, -1.0, -1.0])
    relative_quat = _normalize_quat(_quat_multiply(prev_conjugate, target_quat), name="relative")
    rv = alpha_rot * _quat_to_rotvec(relative_quat)

    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        quat = prev_quat.copy()
    else:
        axis = rv / angle
        half = angle / 2.0
        delta_quat = np.array(
            [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)], dtype=np.float64
        )
        quat = _normalize_quat(_quat_multiply(prev_quat, delta_quat), name="smoothed")

    return pos, quat


def _finite_vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite array with shape {shape}")
    return array.copy()


def compute_target_eef_pose(
    mapped_position_world_m: np.ndarray,
    mapped_quat_world_wxyz: np.ndarray,
    *,
    previous_position_world_m: np.ndarray | None,
    previous_quat_world_wxyz: np.ndarray | None,
    workspace_bounds_world_m: np.ndarray,
    ema_alpha_position: float,
    ema_alpha_rotation: float,
) -> EefTargetProposal:
    """Smooth and clamp one mapped world-frame EEF target."""
    raw_position_world_m = _finite_vector(
        mapped_position_world_m, (3,), "mapped_position_world_m"
    )
    raw_quat_world_wxyz = _finite_vector(
        mapped_quat_world_wxyz, (4,), "mapped_quat_world_wxyz"
    )
    workspace = np.asarray(workspace_bounds_world_m, dtype=np.float64)
    if (
        workspace.shape != (3, 2)
        or not np.all(np.isfinite(workspace))
        or np.any(workspace[:, 0] > workspace[:, 1])
    ):
        raise ValueError(
            "workspace_bounds_world_m must be finite shape (3, 2) with lower <= upper"
        )

    position_world_m = raw_position_world_m.copy()
    quat_world_wxyz = raw_quat_world_wxyz.copy()
    smoothing_state_incomplete = (
        previous_position_world_m is not None and previous_quat_world_wxyz is None
    )
    if previous_position_world_m is not None and previous_quat_world_wxyz is not None:
        position_world_m, quat_world_wxyz = ema_smooth_pose(
            position_world_m,
            quat_world_wxyz,
            previous_position_world_m,
            previous_quat_world_wxyz,
            ema_alpha_position,
            ema_alpha_rotation,
        )

    position_world_m = np.clip(position_world_m, workspace[:, 0], workspace[:, 1])
    return EefTargetProposal(
        position_world_m=position_world_m,
        quat_world_wxyz=quat_world_wxyz,
        smoothing_state_incomplete=smoothing_state_incomplete,
    )


def compute_hand_joint_proposal(
    hand_retargeter: Any,
    vr_frame: dict[str, Any],
    previous_hand_qpos_rad: np.ndarray,
    *,
    hand_available: bool,
    retarget_cache: HandRetargetObservationCache,
    ramp_start_qpos_rad: np.ndarray | None,
    ramp_step: int,
    ramp_total_frames: int,
    command_lower_rad: np.ndarray,
    command_upper_rad: np.ndarray,
    max_delta_rad_per_tick: np.ndarray | float,
) -> HandJointProposal:
    """Retarget and shape one hand proposal without publishing it.

    The final target is bounded against the previously published hand endpoint,
    matching the reject-only per-grid contract for the published endpoint.
    The hand worker separately bounds from measured feedback before its SDK
    call, so this proposal limit never weakens the actuator safety boundary.
    """
    hand_qpos_rad, retarget_succeeded = compute_hand_command(
        hand_retargeter,
        vr_frame,
        previous_hand_qpos_rad,
        hand_available,
        retarget_cache,
    )

    next_ramp_start_qpos_rad = ramp_start_qpos_rad
    next_ramp_step = ramp_step
    if ramp_start_qpos_rad is not None and ramp_step < ramp_total_frames:
        hand_qpos_rad = smoothstep_hand_ramp(
            ramp_start_qpos_rad,
            hand_qpos_rad,
            ramp_step,
            ramp_total_frames,
        )
        next_ramp_step += 1
        if next_ramp_step >= ramp_total_frames:
            next_ramp_start_qpos_rad = None
    elif ramp_start_qpos_rad is not None:
        next_ramp_start_qpos_rad = None
        next_ramp_step = 0

    hand_qpos_rad = np.clip(
        hand_qpos_rad,
        np.asarray(command_lower_rad, dtype=np.float64),
        np.asarray(command_upper_rad, dtype=np.float64),
    )
    hand_qpos_rad = limit_hand_target_delta(
        hand_qpos_rad,
        previous_hand_qpos_rad,
        max_delta_rad_per_tick,
    )

    return HandJointProposal(
        qpos_rad=np.asarray(hand_qpos_rad, dtype=np.float64).copy(),
        retarget_succeeded=retarget_succeeded,
        next_ramp_start_qpos_rad=(
            None
            if next_ramp_start_qpos_rad is None
            else np.asarray(next_ramp_start_qpos_rad, dtype=np.float64).copy()
        ),
        next_ramp_step=next_ramp_step,
    )


def compute_arm_joint_proposal(
    ik_qpos_rad: np.ndarray,
    previous_arm_qpos_rad: np.ndarray | None,
    *,
    joint_lower_rad: np.ndarray,
    joint_upper_rad: np.ndarray,
    max_delta_rad_per_tick: np.ndarray | float | None,
    compute_qpos_delta: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> ArmJointProposal:
    """Clamp one IK result to joint and command-to-command delta limits."""
    arm_qpos_rad = np.asarray(ik_qpos_rad, dtype=np.float64).copy()
    arm_qpos_rad = np.clip(arm_qpos_rad, joint_lower_rad, joint_upper_rad)

    if (
        max_delta_rad_per_tick is not None
        and previous_arm_qpos_rad is not None
        and np.all(np.isfinite(previous_arm_qpos_rad))
    ):
        arm_delta_rad = compute_qpos_delta(arm_qpos_rad, previous_arm_qpos_rad)
        arm_delta_rad = np.clip(
            arm_delta_rad,
            -max_delta_rad_per_tick,
            max_delta_rad_per_tick,
        )
        arm_qpos_rad = (
            np.asarray(previous_arm_qpos_rad, dtype=np.float64) + arm_delta_rad
        )
        arm_qpos_rad = np.clip(arm_qpos_rad, joint_lower_rad, joint_upper_rad)

    validation_issue = None if np.all(np.isfinite(arm_qpos_rad)) else "arm_cmd NaN/Inf"
    return ArmJointProposal(
        qpos_rad=np.asarray(arm_qpos_rad, dtype=np.float64).copy(),
        validation_issue=validation_issue,
    )

"""Compute bounded teleoperation action proposals without side effects.

The coordinator supplies all observations and temporal state, then remains
responsible for safety-gated command publication and recording. Keeping this
module pure makes proposal behavior testable without shared memory or hardware.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dexmani_real.teleop.hand_control import (
    HandRetargetObservationCache,
    compute_hand_command,
    get_raw_hand_command,
    sanitize_hand_command,
    smoothstep_hand_ramp,
)
from dexmani_real.utils.smoothing import ema_smooth_pose


@dataclass(frozen=True)
class EefTargetProposal:
    """One mapped EEF target in the world frame, before and after filtering."""

    position_world_m: np.ndarray
    quat_world_wxyz: np.ndarray
    raw_position_world_m: np.ndarray
    raw_quat_world_wxyz: np.ndarray
    position_before_workspace_clamp_world_m: np.ndarray
    smoothing_state_incomplete: bool


@dataclass(frozen=True)
class HandJointProposal:
    """One hand proposal plus the next ramp state; no command is published."""

    qpos_rad: np.ndarray
    raw_qpos_rad: np.ndarray
    retarget_succeeded: bool
    validation_issue: str | None
    next_ramp_start_qpos_rad: np.ndarray | None
    next_ramp_step: int
    compute_time_ms: float


@dataclass(frozen=True)
class ArmJointProposal:
    """One IK result after firmware limits and command-to-command step limits."""

    qpos_rad: np.ndarray
    raw_qpos_rad: np.ndarray
    validation_issue: str | None


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

    position_before_workspace_clamp_world_m = position_world_m.copy()
    position_world_m = np.clip(position_world_m, workspace[:, 0], workspace[:, 1])
    return EefTargetProposal(
        position_world_m=position_world_m,
        quat_world_wxyz=quat_world_wxyz,
        raw_position_world_m=raw_position_world_m,
        raw_quat_world_wxyz=raw_quat_world_wxyz,
        position_before_workspace_clamp_world_m=(
            position_before_workspace_clamp_world_m
        ),
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
    mechanical_lower_rad: np.ndarray,
    mechanical_upper_rad: np.ndarray,
) -> HandJointProposal:
    """Retarget, shape, and validate one hand proposal without publishing it.

    The final target is bounded against the previously published hand endpoint,
    matching the learned-policy coordinator's reject-only per-grid contract.
    The hand worker separately bounds from measured feedback before its SDK
    call, so this proposal limit never weakens the actuator safety boundary.
    """
    compute_started_s = time.perf_counter()
    hand_qpos_rad, retarget_succeeded = compute_hand_command(
        hand_retargeter,
        vr_frame,
        previous_hand_qpos_rad,
        hand_available,
        retarget_cache,
    )
    compute_time_ms = (time.perf_counter() - compute_started_s) * 1000.0
    raw_qpos_rad = get_raw_hand_command(
        hand_retargeter, hand_qpos_rad, retarget_succeeded
    ).copy()

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
    validation_issue: str | None = None
    try:
        hand_qpos_rad = sanitize_hand_command(
            hand_qpos_rad,
            command_lower_rad,
            command_upper_rad,
            mechanical_lower_rad,
            mechanical_upper_rad,
        )
    except ValueError as exc:
        validation_issue = str(exc)
        hand_qpos_rad = _finite_vector(
            previous_hand_qpos_rad,
            hand_qpos_rad.shape,
            "previous_hand_qpos_rad",
        )
        retarget_succeeded = False

    hand_qpos_rad = limit_hand_target_delta(
        hand_qpos_rad,
        previous_hand_qpos_rad,
        max_delta_rad_per_tick,
    )

    return HandJointProposal(
        qpos_rad=np.asarray(hand_qpos_rad, dtype=np.float64).copy(),
        raw_qpos_rad=np.asarray(raw_qpos_rad, dtype=np.float64).copy(),
        retarget_succeeded=retarget_succeeded,
        validation_issue=validation_issue,
        next_ramp_start_qpos_rad=(
            None
            if next_ramp_start_qpos_rad is None
            else np.asarray(next_ramp_start_qpos_rad, dtype=np.float64).copy()
        ),
        next_ramp_step=next_ramp_step,
        compute_time_ms=compute_time_ms,
    )


def limit_hand_target_delta(
    target_qpos_rad: np.ndarray,
    previous_qpos_rad: np.ndarray,
    max_delta_rad_per_tick: np.ndarray | float,
) -> np.ndarray:
    """Bound a policy-grid hand endpoint relative to its prior endpoint."""
    target = np.asarray(target_qpos_rad, dtype=np.float64)
    previous = np.asarray(previous_qpos_rad, dtype=np.float64)
    if target.shape != previous.shape:
        raise ValueError("hand target and previous endpoint must have the same shape")
    if not np.all(np.isfinite(target)) or not np.all(np.isfinite(previous)):
        raise ValueError("hand target and previous endpoint must be finite")
    max_delta = np.broadcast_to(
        np.asarray(max_delta_rad_per_tick, dtype=np.float64), target.shape
    )
    if not np.all(np.isfinite(max_delta)) or np.any(max_delta <= 0.0):
        raise ValueError("hand max_delta_rad_per_tick must be finite and positive")
    return previous + np.clip(target - previous, -max_delta, max_delta)


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
    raw_qpos_rad = np.asarray(ik_qpos_rad, dtype=np.float64).copy()
    arm_qpos_rad = np.clip(raw_qpos_rad, joint_lower_rad, joint_upper_rad)

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
        raw_qpos_rad=raw_qpos_rad,
        validation_issue=validation_issue,
    )

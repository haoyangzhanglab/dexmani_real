from __future__ import annotations

import math
from typing import Iterable

import mplib as mp
import numpy as np
from transforms3d.quaternions import qconjugate, qmult, qnorm, rotate_vector


XARM7_DOF = 7


def transform_pose(parent: mp.Pose, child: mp.Pose) -> mp.Pose:
    parent_pos = np.asarray(parent.p, dtype=float)
    parent_quat = np.asarray(parent.q, dtype=float)
    child_pos = np.asarray(child.p, dtype=float)
    child_quat = np.asarray(child.q, dtype=float)

    parent_quat = parent_quat / qnorm(parent_quat)
    child_quat = child_quat / qnorm(child_quat)
    pos = parent_pos + rotate_vector(child_pos, parent_quat, is_normalized=True)
    quat = np.asarray(qmult(parent_quat, child_quat), dtype=float)
    return mp.Pose(p=pos, q=quat / qnorm(quat))


def relative_pose(frame_world: mp.Pose, pose_world: mp.Pose) -> mp.Pose:
    frame_pos = np.asarray(frame_world.p, dtype=float)
    frame_quat = np.asarray(frame_world.q, dtype=float)
    pose_pos = np.asarray(pose_world.p, dtype=float)
    pose_quat = np.asarray(pose_world.q, dtype=float)

    frame_quat = frame_quat / qnorm(frame_quat)
    pose_quat = pose_quat / qnorm(pose_quat)
    frame_quat_inv = np.asarray(qconjugate(frame_quat), dtype=float)

    pos = rotate_vector(pose_pos - frame_pos, frame_quat_inv, is_normalized=True)
    quat = np.asarray(qmult(frame_quat_inv, pose_quat), dtype=float)
    return mp.Pose(p=pos, q=quat / qnorm(quat))


def pose_error(pose1: mp.Pose, pose2: mp.Pose) -> tuple[float, float]:
    pos1 = np.asarray(pose1.p, dtype=float)
    pos2 = np.asarray(pose2.p, dtype=float)
    quat1 = np.asarray(pose1.q, dtype=float)
    quat2 = np.asarray(pose2.q, dtype=float)

    quat1 = quat1 / qnorm(quat1)
    quat2 = quat2 / qnorm(quat2)
    dot = min(1.0, max(-1.0, abs(float(np.dot(quat1, quat2)))))
    return float(np.linalg.norm(pos1 - pos2)), 2.0 * math.acos(dot)


def wrap_to_reference(
    qpos: np.ndarray,
    ref_qpos: np.ndarray,
    joint_limits: np.ndarray,
    equivalent_joint_indices: Iterable[int],
) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=float).copy()
    ref_qpos = np.asarray(ref_qpos, dtype=float)
    joint_limits = np.asarray(joint_limits, dtype=float)
    period = 2.0 * np.pi

    for i in equivalent_joint_indices:
        q = qpos[i]
        low, high = joint_limits[i]
        min_shift = math.ceil((low - q) / period)
        max_shift = math.floor((high - q) / period)
        if min_shift > max_shift:
            continue
        shift = round((ref_qpos[i] - q) / period)
        qpos[i] = q + min(max(shift, min_shift), max_shift) * period
    return qpos


def unwrap_path_to_reference(
    path: np.ndarray,
    ref_qpos: np.ndarray,
    joint_limits: np.ndarray,
    equivalent_joint_indices: Iterable[int],
) -> np.ndarray:
    path = np.asarray(path, dtype=float)
    if len(path) == 0:
        return path

    out = np.empty_like(path)
    indices = tuple(equivalent_joint_indices)
    out[0] = wrap_to_reference(path[0], ref_qpos, joint_limits, indices)
    for i in range(1, len(path)):
        out[i] = wrap_to_reference(path[i], out[i - 1], joint_limits, indices)
    return out



def evaluate_path(
    path: np.ndarray,
    start_qpos: np.ndarray,
    goal_qpos: np.ndarray,
    last_goal_qpos: np.ndarray | None = None,
) -> tuple[bool, tuple[float, float, float], dict[str, list[float] | float | int | str | None]]:
    path = np.asarray(path, dtype=float)
    start_qpos = np.asarray(start_qpos, dtype=float)
    goal_qpos = np.asarray(goal_qpos, dtype=float)

    if start_qpos.shape != (XARM7_DOF,) or goal_qpos.shape != (XARM7_DOF,):
        raise ValueError("xArm7 path evaluation expects start_qpos and goal_qpos with shape (7,).")
    if path.ndim != 2 or path.shape[1] != XARM7_DOF:
        raise ValueError("xArm7 path evaluation expects path with shape (T, 7).")
    if last_goal_qpos is not None and np.asarray(last_goal_qpos, dtype=float).shape != (XARM7_DOF,):
        raise ValueError("xArm7 path evaluation expects last_goal_qpos with shape (7,).")

    max_step_deg = np.asarray((35, 30, 35, 30, 35, 30, 45), dtype=float)
    max_excursion_deg = np.asarray((140, 90, 140, 120, 150, 120, 180), dtype=float)
    max_total_motion_deg = np.asarray((220, 140, 220, 180, 220, 180, 270), dtype=float)
    motion_weights = np.asarray((3.0, 2.5, 2.5, 1.5, 1.5, 1.0, 1.0), dtype=float)
    risk_weights = np.asarray((2.0, 2.0, 2.0, 1.5, 1.0, 1.0, 0.5), dtype=float)

    if len(path) < 2:
        max_step = np.zeros_like(start_qpos)
        max_excursion = np.zeros_like(start_qpos) if len(path) == 0 else np.abs(path[0] - start_qpos)
        total_motion = np.zeros_like(start_qpos)
    else:
        step = np.abs(np.diff(path, axis=0))
        max_step = np.max(step, axis=0)
        max_excursion = np.max(np.abs(path - start_qpos[None, :]), axis=0)
        total_motion = np.sum(step, axis=0)

    reason = None
    if np.any(max_step > np.deg2rad(max_step_deg)):
        reason = "step_limit"
    elif np.any(max_excursion > np.deg2rad(max_excursion_deg)):
        reason = "excursion_limit"
    elif np.any(total_motion > np.deg2rad(max_total_motion_deg)):
        reason = "total_motion_limit"

    total_motion_score = float(np.sum(motion_weights * total_motion))
    excursion_score = float(np.sum(motion_weights * max_excursion))
    risk_score = float(np.sum(risk_weights * max_excursion))

    branch_score = 0.0
    if last_goal_qpos is not None:
        branch_score = 0.02 * float(np.sum(np.abs(goal_qpos - np.asarray(last_goal_qpos, dtype=float))))

    debug = {
        "reason": reason,
        "n_waypoints": int(len(path)),
        "max_step_deg": np.rad2deg(max_step).round(2).tolist(),
        "max_excursion_deg": np.rad2deg(max_excursion).round(2).tolist(),
        "total_motion_deg": np.rad2deg(total_motion).round(2).tolist(),
        "score_total_motion": round(total_motion_score, 6),
        "score_excursion": round(excursion_score, 6),
        "score_branch": round(branch_score, 6),
    }
    return reason is None, (total_motion_score + risk_score, excursion_score, branch_score), debug

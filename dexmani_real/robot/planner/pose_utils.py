from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .planner_types import Pose
except ImportError:
    from planner_types import Pose


def ensure_qpos(qpos: np.ndarray, dof: int, name: str) -> np.ndarray:
    array = np.asarray(qpos, dtype=np.float64).reshape(-1)
    if array.shape[0] != dof:
        raise ValueError(f"{name} must have length {dof}, got {array.shape[0]}.")
    return array.copy()


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def pose_to_rotation(pose: Pose) -> Rotation:
    return Rotation.from_quat(wxyz_to_xyzw(pose.q))


def rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    return xyzw_to_wxyz(rotation.as_quat())


def compose_pose(a: Pose, b: Pose) -> Pose:
    rotation_a = pose_to_rotation(a)
    rotation_b = pose_to_rotation(b)
    p = a.p + rotation_a.apply(b.p)
    q = rotation_to_wxyz(rotation_a * rotation_b)
    return Pose(p=p, q=q)


def invert_pose(pose: Pose) -> Pose:
    rotation = pose_to_rotation(pose)
    inverse_rotation = rotation.inv()
    p = inverse_rotation.apply(-pose.p)
    q = rotation_to_wxyz(inverse_rotation)
    return Pose(p=p, q=q)


def compute_pose_error(target: Pose, actual: Pose) -> tuple[float, float]:
    position_error = float(np.linalg.norm(target.p - actual.p))
    rotation_target = pose_to_rotation(target)
    rotation_actual = pose_to_rotation(actual)
    rotation_error = float((rotation_target * rotation_actual.inv()).magnitude())
    return position_error, rotation_error


def pose_error_vector(target: Pose, actual: Pose, max_pos_step: float, max_rot_step: float) -> np.ndarray:
    position_error = target.p - actual.p
    position_norm = np.linalg.norm(position_error)
    if position_norm > max_pos_step > 0:
        position_error = position_error / position_norm * max_pos_step

    rotation_error = pose_to_rotation(target) * pose_to_rotation(actual).inv()
    rotation_vector = rotation_error.as_rotvec()
    rotation_norm = np.linalg.norm(rotation_vector)
    if rotation_norm > max_rot_step > 0:
        rotation_vector = rotation_vector / rotation_norm * max_rot_step

    return np.concatenate([position_error, rotation_vector])


def interpolate_qpos_path(start: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    start = np.asarray(start, dtype=np.float64).reshape(-1)
    goal = np.asarray(goal, dtype=np.float64).reshape(-1)
    delta = goal - start
    step_count = int(np.ceil(np.max(np.abs(delta)) / max(max_step, 1e-12)))
    step_count = max(step_count, 1)
    weights = np.linspace(0.0, 1.0, step_count + 1)
    return start[None, :] + weights[:, None] * delta[None, :]


def resample_qpos_path(path: np.ndarray, target_length: int) -> np.ndarray:
    path = np.asarray(path, dtype=np.float64)
    if target_length <= 0:
        raise ValueError("target_length must be positive.")
    if len(path) == target_length:
        return path.copy()
    if len(path) == 1:
        return np.repeat(path, target_length, axis=0)
    source_t = np.linspace(0.0, 1.0, len(path))
    target_t = np.linspace(0.0, 1.0, target_length)
    columns = [np.interp(target_t, source_t, path[:, index]) for index in range(path.shape[1])]
    return np.stack(columns, axis=1)

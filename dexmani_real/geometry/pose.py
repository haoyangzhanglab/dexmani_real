"""WXYZ 四元数、列序 rot6d 与刚体变换。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

WXYZ_TO_XYZW = np.array([1, 2, 3, 0], dtype=np.intp)
XYZW_TO_WXYZ = np.array([3, 0, 1, 2], dtype=np.intp)
ROT6D_MIN_BASIS_NORM = 1e-4
ROT6D_MIN_ORTHOGONAL_RATIO = 1e-4

ROTATION_ATOL = 1e-6
HOMOGENEOUS_ATOL = 1e-9
QUAT_NORM_EPS = 1e-12


def validate_rigid_transform(value, *, label="transform"):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite (4, 4) transform")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=HOMOGENEOUS_ATOL, rtol=0):
        raise ValueError(f"{label} must have homogeneous final row")
    rotation = matrix[:3, :3]
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
    if orthogonality_error > ROTATION_ATOL or determinant_error > ROTATION_ATOL:
        raise ValueError(f"{label} must have a proper SO(3) rotation")
    return matrix.copy()


def _quaternion(value, name):
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"{name} must be a finite (4,) WXYZ quaternion")
    norm = float(np.linalg.norm(quat))
    if not np.isfinite(norm) or norm <= QUAT_NORM_EPS:
        raise ValueError(f"{name} quaternion norm is invalid")
    return quat, norm


def normalize_quat_wxyz(value, *, name="quaternion"):
    quat, norm = _quaternion(value, name)
    return quat / norm


def validate_unit_quaternion_wxyz(value, *, name="quaternion"):
    quat, norm = _quaternion(value, name)
    if abs(norm - 1.0) > ROTATION_ATOL:
        raise ValueError(f"{name} must be unit length within {ROTATION_ATOL}")
    return quat.copy()


@dataclass
class Pose:
    """位置单位米，四元数顺序 WXYZ。"""

    p: np.ndarray
    q: np.ndarray

    def __init__(self, p: Any, q: Any) -> None:
        self.p = np.asarray(p, dtype=np.float64).reshape(3)
        self.q = normalize_quat_wxyz(q)
        if not np.isfinite(self.p).all():
            raise ValueError("Pose position must be finite")

    def copy(self) -> "Pose":
        return Pose(p=self.p.copy(), q=self.q.copy())


def wxyz_to_xyzw(q: ArrayLike) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q[WXYZ_TO_XYZW]


def xyzw_to_wxyz(q: ArrayLike) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return q[XYZW_TO_WXYZ]


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def compute_pose_error(target: Pose, actual: Pose) -> tuple[float, float]:
    position_error = float(np.linalg.norm(target.p - actual.p))
    q_dot = abs(float(np.dot(target.q, actual.q)))
    rotation_error = 2.0 * np.arccos(np.clip(q_dot, 0.0, 1.0))
    return position_error, rotation_error


def rot6d_to_quat_wxyz(r6: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quat_xyzw = Rotation.from_matrix(rot6d_to_rotmat(r6)).as_quat()
    return xyzw_to_wxyz(quat_xyzw)


def encode_eef_pose(transform: np.ndarray) -> np.ndarray:
    """将内部 SE(3) 编码为外部 XYZ+列序 rot6d，支持 (...,4,4)。"""
    return np.concatenate(
        (transform[..., :3, 3], transform[..., :3, 0], transform[..., :3, 1]), axis=-1,
    )


def rot6d_to_rotmat(r6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt 将 (...,6) 列序 rot6d 还原为 (...,3,3) 旋转矩阵。"""
    values = np.asarray(r6, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 6 or not np.isfinite(values).all():
        raise ValueError("rot6d must have finite shape (...,6)")
    first, second = values[..., :3], values[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if (
        not np.isfinite([first_norm, second_norm]).all()
        or np.any(np.minimum(first_norm, second_norm) < ROT6D_MIN_BASIS_NORM)
    ):
        raise ValueError("rot6d contains a zero or unrepresentable basis")

    unit_first = first / first_norm
    orthogonal_second = second - np.sum(second * unit_first, axis=-1, keepdims=True) * unit_first
    orthogonal_norm = np.linalg.norm(orthogonal_second, axis=-1, keepdims=True)
    if (
        not np.isfinite(orthogonal_norm).all()
        or np.any(orthogonal_norm / second_norm < ROT6D_MIN_ORTHOGONAL_RATIO)
    ):
        raise ValueError("rot6d contains a near-collinear basis")
    unit_second = orthogonal_second / orthogonal_norm
    return np.stack((unit_first, unit_second, np.cross(unit_first, unit_second)), axis=-1)


def forward_from_quat_wxyz(q_wxyz: ArrayLike) -> np.ndarray:
    """返回旋转后的 FLU +X 朝向。"""
    from scipy.spatial.transform import Rotation

    quat_xyzw = wxyz_to_xyzw(q_wxyz)
    return Rotation.from_quat(quat_xyzw).apply(np.array([1.0, 0.0, 0.0]))

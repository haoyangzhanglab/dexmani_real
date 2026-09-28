"""Strict SO(3), SE(3) and WXYZ quaternion mathematics; no fallback values."""

import numpy as np

ROTATION_ATOL = 1e-6
HOMOGENEOUS_ATOL = 1e-9
QUAT_NORM_EPS = 1e-12


def validate_rotation_matrix(value, *, name="rotation"):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite (3, 3) rotation")
    residual = matrix.T @ matrix - np.eye(3)
    orthogonality_error = float(np.max(np.abs(residual)))
    determinant_error = abs(float(np.linalg.det(matrix)) - 1.0)
    if orthogonality_error > ROTATION_ATOL or determinant_error > ROTATION_ATOL:
        raise ValueError(f"{name} must be a proper SO(3) rotation")
    return matrix.copy()


def validate_rigid_transform(value, *, label="transform"):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite (4, 4) transform")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=HOMOGENEOUS_ATOL, rtol=0):
        raise ValueError(f"{label} must have homogeneous final row")
    validate_rotation_matrix(matrix[:3, :3], name=label)
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

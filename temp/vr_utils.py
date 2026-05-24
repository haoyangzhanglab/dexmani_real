"""Math and hand-feature utilities for Quest hand tracking.

Conventions
-----------
- Quaternion order is always ``wxyz``.
- Euler helpers use fixed-axis RPY with ``axes='sxyz'``.
- ``HandData.landmarks`` are treated as a 21-joint MediaPipe-style hand
  skeleton after SDK Unity-left-handed -> FLU conversion.
- ``hand_data_to_retargeting_joints`` returns wrist-relative landmarks. It
  does not estimate a palm canonical frame and it is not used for EEF pose
  control.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import (
    axangle2quat,
    mat2quat,
    qinverse,
    qmult,
    qnorm,
    quat2axangle,
    quat2mat,
    rotate_vector,
)

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # wxyz

IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)
EULER_AXES = "sxyz"

HAND_JOINT_NAMES = (
    "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "little_proximal", "little_intermediate", "little_distal", "little_tip",
)

FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")
FINGER_CHAINS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "little": (0, 17, 18, 19, 20),
}


class HandDataLike(Protocol):
    side: str
    wrist_quat: Quat

    def landmarks_np(self) -> np.ndarray: ...


def as_vec3(v: Vec3 | np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"expected shape (3,), got {arr.shape}")
    return arr


def as_quat(q: Quat | np.ndarray) -> np.ndarray:
    arr = np.asarray(q, dtype=np.float64)
    if arr.shape != (4,):
        raise ValueError(f"expected shape (4,), got {arr.shape}")
    return arr


def tuple3(v: Vec3 | np.ndarray) -> Vec3:
    arr = as_vec3(v)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def tuple4(q: Quat | np.ndarray) -> Quat:
    arr = as_quat(q)
    return (float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3]))


def vec_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec_scale(v: Vec3, scale: float) -> Vec3:
    return (v[0] * scale, v[1] * scale, v[2] * scale)


def vec_norm(v: Vec3 | np.ndarray) -> float:
    return float(np.linalg.norm(as_vec3(v)))


def normalize_array(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-12 or not math.isfinite(norm):
        return np.asarray(fallback, dtype=np.float64).copy()
    return np.asarray(v / norm, dtype=np.float64)


def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-12 or not math.isfinite(denom):
        return 0.0
    cosine = float(np.dot(u, v)) / denom
    return math.acos(max(-1.0, min(1.0, cosine)))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def quat_normalize(q: Quat | np.ndarray, fallback: Quat = IDENTITY_QUAT) -> Quat:
    arr = as_quat(q)
    norm = float(qnorm(arr))
    if norm < 1e-12 or not math.isfinite(norm):
        return fallback
    return tuple4(arr / norm)


def quat_mul(a: Quat, b: Quat) -> Quat:
    return quat_normalize(tuple4(np.asarray(qmult(as_quat(a), as_quat(b)), dtype=np.float64)))


def quat_inverse(q: Quat) -> Quat:
    return tuple4(np.asarray(qinverse(as_quat(quat_normalize(q))), dtype=np.float64))


def quat_dot(a: Quat, b: Quat) -> float:
    return float(np.dot(as_quat(a), as_quat(b)))


def quat_match_hemisphere(q: Quat, reference: Quat) -> Quat:
    return (-q[0], -q[1], -q[2], -q[3]) if quat_dot(q, reference) < 0.0 else q


def quat_angle(q: Quat) -> float:
    qn = quat_normalize(q)
    return 2.0 * math.atan2(vec_norm((qn[1], qn[2], qn[3])), abs(qn[0]))


def quat_slerp_from_identity(q: Quat, scale: float) -> Quat:
    """Scale a rotation quaternion from identity toward ``q``."""
    if scale <= 0.0:
        return IDENTITY_QUAT
    qn = quat_normalize(q)
    if qn[0] < 0.0:
        qn = (-qn[0], -qn[1], -qn[2], -qn[3])
    if scale >= 1.0:
        return qn
    axis, angle = quat2axangle(as_quat(qn))
    angle = float(angle)
    if abs(angle) < 1e-12:
        return IDENTITY_QUAT
    return quat_normalize(tuple4(np.asarray(axangle2quat(axis, angle * scale), dtype=np.float64)))


def quat_rotate_vector(q: Quat, v: Vec3) -> Vec3:
    out = rotate_vector(as_vec3(v), as_quat(quat_normalize(q)), is_normalized=True)
    return tuple3(np.asarray(out, dtype=np.float64))


def change_basis_rotation(delta_quat: Quat, target_from_source: Quat) -> Quat:
    """Map a relative rotation from source basis to target basis.

    If ``target_from_source`` maps source-frame vectors into target-frame
    vectors, the equivalent target-frame relative rotation is
    ``R_ts R_delta R_ts^T``.
    """
    basis = quat_normalize(target_from_source)
    return quat_normalize(quat_mul(quat_mul(basis, delta_quat), quat_inverse(basis)))


def quat_wxyz_to_xyzw(q: Quat) -> tuple[float, float, float, float]:
    qn = quat_normalize(q)
    return (qn[1], qn[2], qn[3], qn[0])


def quat_wxyz_to_matrix(q: Quat) -> np.ndarray:
    return np.asarray(quat2mat(as_quat(quat_normalize(q))), dtype=np.float64)


def matrix_to_quat_wxyz(matrix: np.ndarray) -> Quat:
    mat = np.asarray(matrix, dtype=np.float64)
    if mat.shape != (3, 3):
        raise ValueError(f"expected shape (3, 3), got {mat.shape}")
    return quat_normalize(tuple4(np.asarray(mat2quat(mat), dtype=np.float64)))


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> Quat:
    q = euler2quat(roll, pitch, yaw, axes=EULER_AXES)
    return quat_normalize(tuple4(np.asarray(q, dtype=np.float64)))


def quat_to_rpy(q: Quat) -> tuple[float, float, float]:
    rpy = quat2euler(as_quat(quat_normalize(q)), axes=EULER_AXES)
    return (wrap_pi(float(rpy[0])), wrap_pi(float(rpy[1])), wrap_pi(float(rpy[2])))


def lefranx_vr_to_robot_matrix() -> np.ndarray:
    """Return the LeFranX-style VR-to-robot-base rotation matrix.

    Mapping:
        robot_x = vr_z
        robot_y = -vr_x
        robot_z = vr_y
    """
    return np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


def lefranx_vr_to_robot_quat() -> Quat:
    return matrix_to_quat_wxyz(lefranx_vr_to_robot_matrix())


def clamp_relative_orientation(
    quat: Quat,
    home_quat: Quat,
    orientation_limits: tuple[float, float, float, float, float, float] | None,
    max_orientation_angle: float,
) -> Quat:
    """Clamp ``quat`` around ``home_quat`` using LeFranX-style relative order.

    Relative convention:
        q_rel = q_current * inverse(q_home)
        q_current = q_rel * q_home
    """
    home = quat_normalize(home_quat)
    rel = quat_normalize(quat_mul(quat, quat_inverse(home)))

    if orientation_limits is not None:
        roll, pitch, yaw = quat_to_rpy(rel)
        r_min, r_max, p_min, p_max, y_min, y_max = orientation_limits
        rel = quat_from_rpy(
            clamp(roll, r_min, r_max),
            clamp(pitch, p_min, p_max),
            clamp(yaw, y_min, y_max),
        )

    if max_orientation_angle > 0.0:
        angle = quat_angle(rel)
        if angle > max_orientation_angle and angle > 1e-12:
            rel = quat_slerp_from_identity(rel, max_orientation_angle / angle)

    return quat_normalize(quat_mul(rel, home))


def validate_landmarks(points: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(points, dtype=np.float64))
    if arr.shape != (21, 3):
        raise ValueError(f"expected 21 landmarks with shape (21, 3), got {arr.shape}")
    return arr


def hand_data_to_wrist_relative_joints(data: HandDataLike) -> np.ndarray:
    """Return 21 landmarks translated so landmark[0] is the origin.

    The output keeps the orientation of ``HandData.landmarks``. It does not use
    ``wrist_quat`` and does not estimate a palm canonical frame.
    """
    points = validate_landmarks(data.landmarks_np())
    local = points - points[0]
    local[0] = 0.0
    return np.ascontiguousarray(local, dtype=np.float64)


def hand_data_to_wrist_local_joints(data: HandDataLike) -> np.ndarray:
    """Return 21 landmarks in the SDK wrist pose local frame.

    This is useful for debug/ablation. The default retargeting representation
    remains wrist-relative because LeFranX-style hand retargeting treats the 21
    keypoints as the hand branch skeleton, separate from EEF pose control.
    """
    relative = hand_data_to_wrist_relative_joints(data)
    world_from_wrist = quat_wxyz_to_matrix(quat_normalize(data.wrist_quat))
    wrist_from_world = world_from_wrist.T
    local = relative @ wrist_from_world.T
    local[0] = 0.0
    return np.ascontiguousarray(local, dtype=np.float64)


def hand_data_to_retargeting_joints(data: HandDataLike) -> np.ndarray:
    """Return 21x3 human hand joints for external hand retargeting.

    Current convention:
        - MediaPipe-style 21-joint topology.
        - origin = landmark[0] wrist.
        - orientation = same as ``HandData.landmarks`` after SDK conversion.
        - no palm canonical frame estimation.
        - no EEF / arm pose semantics.

    External dex-retargeting code can build its own ``ref_value`` from this
    array using target human indices from the retargeting config.
    """
    return hand_data_to_wrist_relative_joints(data)


def hand_data_to_finger_curl_vector(data: HandDataLike) -> np.ndarray:
    """Return fixed-order finger bend angles in radians.

    Order: thumb, index, middle, ring, little. Each default finger chain has
    five landmarks, so each finger contributes three interior bend angles and
    the output shape is ``(15,)``.

    Convention:
        straight finger ~= 0
        larger angle = more curled
        unit = radians
    """
    points = validate_landmarks(data.landmarks_np())
    curls: list[float] = []
    for finger in FINGER_ORDER:
        chain = FINGER_CHAINS[finger]
        for i in range(1, len(chain) - 1):
            previous_point = points[chain[i - 1]]
            current_point = points[chain[i]]
            next_point = points[chain[i + 1]]
            incoming = current_point - previous_point
            outgoing = next_point - current_point
            curls.append(angle_between(incoming, outgoing))

    result = np.asarray(curls, dtype=np.float64)
    if result.shape != (15,):
        raise RuntimeError(f"finger curl vector must have shape (15,), got {result.shape}")
    return np.ascontiguousarray(result, dtype=np.float64)

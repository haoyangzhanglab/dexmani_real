"""Pure landmark geometry shared by TAG and DexPilot adapters.

One landmark-space adaptation compensates the human-robot kinematic mismatch:
``adaptive_retargeting_xhand`` scales the pinky chain (MCP→PIP→DIP→TIP) by a
constant per-backend ``pinky_scale`` (plus an optional palm-baseline offset).
"""

from __future__ import annotations

__all__ = [
    "adaptive_retargeting_xhand",
    "validate_landmarks",
]


import numpy as np

_PINKY_MCP = 17
_PINKY_PIP = 18
_PINKY_DIP = 19
_PINKY_TIP = 20

_CONTIGUOUS_BONES = tuple(
    (parent, child)
    for chain in (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    for parent, child in zip(chain, chain[1:])
)

# Right-hand operator→MANO rotation; Unity→FLU conversion occurs in the VR receiver.
_OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


def validate_landmarks(keypoint_3d_array: np.ndarray) -> tuple[bool, str]:
    """Apply the fail-closed geometric gate before touching temporal state."""
    points = np.asarray(keypoint_3d_array, dtype=np.float64)
    if points.shape != (21, 3):
        return False, f"shape {points.shape} != (21, 3)"
    if not np.all(np.isfinite(points)):
        return False, "contains NaN/Inf"
    shortest_bone = min(
        float(np.linalg.norm(points[child] - points[parent])) for parent, child in _CONTIGUOUS_BONES
    )
    if shortest_bone < 0.002:
        return False, "a retargeting bone is shorter than 2 mm"
    return True, ""


def _estimate_palm_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Estimate a palm coordinate frame (3×3 rotation matrix) from 21 hand landmarks.

    Uses wrist + index MCP + middle MCP to fit a palm plane via SVD, then
    constructs a right-handed orthonormal frame with x pointing from middle
    MCP to wrist. The caller must first run ``validate_landmarks()``.

    """
    keypoint_3d_array = np.asarray(keypoint_3d_array, dtype=np.float64)
    if keypoint_3d_array.shape != (21, 3):
        raise ValueError(
            f"keypoint_3d_array must have shape (21, 3), got {keypoint_3d_array.shape}"
        )

    eps = 1e-8
    points = keypoint_3d_array[[0, 5, 9], :].copy()

    first, second = points[1] - points[0], points[2] - points[0]
    lengths = np.linalg.norm(first), np.linalg.norm(second)
    if (
        min(lengths) < 0.01
        or np.linalg.norm(np.cross(first, second)) < 0.1 * lengths[0] * lengths[1]
    ):
        raise ValueError("wrist/index/middle palm triangle is degenerate")
    x_vector = points[0] - points[2]  # middle MCP → wrist
    points_centered = points - np.mean(points, axis=0, keepdims=True)

    try:
        _, _, v = np.linalg.svd(points_centered)
    except np.linalg.LinAlgError as exc:
        raise ValueError("palm SVD failed") from exc

    normal = v[2, :]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < eps:
        raise ValueError("palm normal is degenerate")
    normal = normal / normal_norm

    x = x_vector - np.sum(x_vector * normal) * normal
    x_norm = np.linalg.norm(x)
    if x_norm < eps:
        raise ValueError("palm longitudinal axis is degenerate")
    x = x / x_norm

    z = np.cross(x, normal)
    z_norm = np.linalg.norm(z)
    if z_norm < eps:
        raise ValueError("palm lateral axis is degenerate")
    z = z / z_norm

    # Keep the lateral axis sign consistent with index→middle.
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1.0
        z *= -1.0

    return np.stack([x, normal, z], axis=1)


def adaptive_retargeting_xhand(
    landmarks: np.ndarray,
    *,
    scale: float,
    palm_scale: float,
) -> np.ndarray:
    """Return a copy of (21, 3) MANO landmarks with the pinky chain scaled.

    scale multiplies MCP-to-TIP segments; palm_scale multiplies wrist-to-MCP
    (1 leaves the baseline unchanged).
    """
    landmarks = np.asarray(landmarks, dtype=np.float64).copy()
    if landmarks.shape != (21, 3) or not np.all(np.isfinite(landmarks)):
        raise ValueError("landmarks must be a finite (21, 3) array")
    raw = landmarks.copy()

    if palm_scale != 1.0:
        landmarks[_PINKY_MCP] = landmarks[0] + (raw[_PINKY_MCP] - landmarks[0]) * palm_scale

    mcp_to_pip = raw[_PINKY_PIP] - raw[_PINKY_MCP]
    landmarks[_PINKY_PIP] = landmarks[_PINKY_MCP] + mcp_to_pip * scale

    pip_to_dip = raw[_PINKY_DIP] - raw[_PINKY_PIP]
    landmarks[_PINKY_DIP] = landmarks[_PINKY_PIP] + pip_to_dip * scale

    dip_to_tip = raw[_PINKY_TIP] - raw[_PINKY_DIP]
    landmarks[_PINKY_TIP] = landmarks[_PINKY_DIP] + dip_to_tip * scale

    return landmarks

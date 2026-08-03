"""Hand geometry utilities shared across teleop and test code."""

import numpy as np

__all__ = ["OPERATOR2MANO_RIGHT", "estimate_frame_from_hand_points"]

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Estimate palm coordinate frame (3x3 rotation matrix) from 21 hand landmarks via SVD.

    Uses wrist + index MCP + middle MCP (3 points) to fit a palm plane,
    then constructs a right-handed orthonormal frame with x pointing from
    middle MCP to wrist.

    Ref: LeFranX vr_hand_detector_adapter.py:293-342
    """
    keypoint_3d_array = np.asarray(keypoint_3d_array, dtype=np.float64)
    if keypoint_3d_array.shape != (21, 3):
        raise ValueError(f"keypoint_3d_array must have shape (21, 3), got {keypoint_3d_array.shape}")

    eps = 1e-8

    # LeFranX: 3-point method — wrist, index MCP, middle MCP
    points = keypoint_3d_array[[0, 5, 9], :].copy()

    if not np.all(np.isfinite(points)):
        return np.eye(3, dtype=np.float64)

    # Compute vector from middle MCP to wrist
    x_vector = points[0] - points[2]

    # Normal fitting with SVD on centered points
    points_centered = points - np.mean(points, axis=0, keepdims=True)

    try:
        _, _, v = np.linalg.svd(points_centered)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)

    normal = v[2, :]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < eps:
        return np.eye(3, dtype=np.float64)
    normal = normal / normal_norm

    # Gram-Schmidt Orthonormalize
    x = x_vector - np.sum(x_vector * normal) * normal
    x_norm = np.linalg.norm(x)
    if x_norm < eps:
        return np.eye(3, dtype=np.float64)
    x = x / x_norm

    z = np.cross(x, normal)
    z_norm = np.linalg.norm(z)
    if z_norm < eps:
        return np.eye(3, dtype=np.float64)
    z = z / z_norm

    # LeFranX: use index_mcp -> middle_mcp as lateral reference
    # (points[1] - points[2]) is the vector from middle MCP to index MCP
    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1.0
        z *= -1.0

    frame = np.stack([x, normal, z], axis=1)
    return frame

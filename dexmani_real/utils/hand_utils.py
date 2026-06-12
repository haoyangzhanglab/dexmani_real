"""Hand geometry utilities shared across teleop and test code."""

import numpy as np

OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)

OPERATOR2MANO_LEFT = np.array(
    [
        [0, 0, -1],
        [1, 0, 0],
        [0, -1, 0],
    ]
)


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Estimate palm coordinate frame (3x3 rotation matrix) from 21 hand landmarks via SVD.

    Uses wrist + 4 MCP joints to fit a palm plane, then constructs a right-handed
    orthonormal frame with x pointing from middle MCP to wrist.
    """
    keypoint_3d_array = np.asarray(keypoint_3d_array, dtype=np.float64)
    assert keypoint_3d_array.shape == (21, 3)

    eps = 1e-8

    wrist = keypoint_3d_array[0]
    index_mcp = keypoint_3d_array[5]
    middle_mcp = keypoint_3d_array[9]
    ring_mcp = keypoint_3d_array[13]
    pinky_mcp = keypoint_3d_array[17]

    palm_points = np.stack(
        [wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp],
        axis=0,
    )

    if not np.all(np.isfinite(palm_points)):
        return np.eye(3, dtype=np.float64)

    palm_center = palm_points.mean(axis=0, keepdims=True)
    centered = palm_points - palm_center

    if np.linalg.norm(centered) < eps:
        return np.eye(3, dtype=np.float64)

    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.eye(3, dtype=np.float64)

    normal = vh[-1, :]
    normal_norm = np.linalg.norm(normal)
    if normal_norm < eps:
        return np.eye(3, dtype=np.float64)
    normal = normal / normal_norm

    x_vector = wrist - middle_mcp
    x = x_vector - np.dot(x_vector, normal) * normal
    x_norm = np.linalg.norm(x)

    if x_norm < eps:
        x_vector = wrist - palm_center.reshape(3)
        x = x_vector - np.dot(x_vector, normal) * normal
        x_norm = np.linalg.norm(x)

    if x_norm < eps:
        return np.eye(3, dtype=np.float64)

    x = x / x_norm

    z = np.cross(x, normal)
    z_norm = np.linalg.norm(z)
    if z_norm < eps:
        return np.eye(3, dtype=np.float64)
    z = z / z_norm

    lateral_ref = index_mcp - pinky_mcp
    lateral_ref = lateral_ref - np.dot(lateral_ref, x) * x
    lateral_norm = np.linalg.norm(lateral_ref)

    if lateral_norm > eps:
        lateral_ref = lateral_ref / lateral_norm
        if np.dot(z, lateral_ref) < 0:
            normal *= -1.0
            z *= -1.0

    normal = np.cross(z, x)
    normal_norm = np.linalg.norm(normal)
    if normal_norm < eps:
        return np.eye(3, dtype=np.float64)
    normal = normal / normal_norm

    frame = np.stack([x, normal, z], axis=1)

    try:
        u, _, vh = np.linalg.svd(frame)
        frame = u @ vh
    except np.linalg.LinAlgError:
        pass

    if np.linalg.det(frame) < 0:
        frame[:, 2] *= -1.0

    return frame

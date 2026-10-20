"""TAG/DexPilot landmark frames and pinky-chain/wrist→MCP scale compensation."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

CONTIGUOUS_BONES = tuple(
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
OPERATOR2MANO_RIGHT = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


def validate_landmarks(points: np.ndarray) -> tuple[bool, str]:
    """Apply the fail-closed geometric gate before touching temporal state."""
    if points.shape != (21, 3):
        return False, f"shape {points.shape} != (21, 3)"
    if not np.all(np.isfinite(points)):
        return False, "contains NaN/Inf"
    shortest_bone = min(
        float(np.linalg.norm(points[child] - points[parent])) for parent, child in CONTIGUOUS_BONES
    )
    if shortest_bone < 0.002:
        return False, "a retargeting bone is shorter than 2 mm"
    return True, ""


def estimate_palm_frame(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """Construct the admitted palm triangle's right-handed frame, X toward wrist."""
    eps = 1e-8
    points = keypoint_3d_array[[0, 5, 9], :].copy()

    first, second = points[1] - points[0], points[2] - points[0]
    lengths = np.linalg.norm(first), np.linalg.norm(second)
    normal = np.cross(first, second)
    normal_norm = np.linalg.norm(normal)
    if min(lengths) < 0.01 or normal_norm < 0.1 * lengths[0] * lengths[1]:
        raise ValueError("wrist/index/middle palm triangle is degenerate")
    x_vector = points[0] - points[2]  # middle MCP → wrist
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


def scale_pinky(
    raw: np.ndarray,
    *,
    scale: float,
    palm_scale: float,
) -> np.ndarray:
    """Copy validated MANO landmarks; scale MCP→TIP segments and wrist→MCP baseline separately."""
    landmarks = raw.copy()

    if palm_scale != 1.0:
        landmarks[PINKY_MCP] = landmarks[0] + (raw[PINKY_MCP] - landmarks[0]) * palm_scale

    mcp_to_pip = raw[PINKY_PIP] - raw[PINKY_MCP]
    landmarks[PINKY_PIP] = landmarks[PINKY_MCP] + mcp_to_pip * scale

    pip_to_dip = raw[PINKY_DIP] - raw[PINKY_PIP]
    landmarks[PINKY_DIP] = landmarks[PINKY_PIP] + pip_to_dip * scale

    dip_to_tip = raw[PINKY_TIP] - raw[PINKY_DIP]
    landmarks[PINKY_TIP] = landmarks[PINKY_DIP] + dip_to_tip * scale

    return landmarks


def prepare_landmarks(landmarks, *, pinky_scale, pinky_palm_scale):
    """Validate operator geometry, then return MANO landmarks with the configured pinky scaling."""
    if landmarks is None:
        return None
    landmarks = np.asarray(landmarks, dtype=np.float64)
    valid, reason = validate_landmarks(landmarks)
    if not valid:
        logger.warning("VR landmarks rejected (%s) — no target produced", reason)
        return None
    try:
        wrist_rot = estimate_palm_frame(landmarks)
        mano = landmarks @ wrist_rot @ OPERATOR2MANO_RIGHT
    except ValueError:
        logger.warning("Coordinate transform failed — no target produced")
        return None
    mano = scale_pinky(mano, scale=pinky_scale, palm_scale=pinky_palm_scale)
    if not np.isfinite(mano).all():
        raise ValueError("transformed hand landmarks must be finite")
    return mano

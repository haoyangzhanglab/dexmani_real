"""右手掌坐标系与小指尺度补偿。"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

BONE_CHILDREN = np.arange(1, 21)
BONE_PARENTS = BONE_CHILDREN - 1
BONE_PARENTS[::4] = 0
OPERATOR_TO_MANO = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])


def estimate_palm_frame(landmarks):
    first, second = landmarks[[5, 9]] - landmarks[0]
    lengths = np.linalg.norm(first), np.linalg.norm(second)
    normal = np.cross(first, second)
    normal_norm = np.linalg.norm(normal)
    if min(lengths) < 0.01 or normal_norm < 0.1 * lengths[0] * lengths[1]:
        return None

    normal /= normal_norm
    x = -second + np.dot(second, normal) * normal
    x /= np.linalg.norm(x)
    z = np.cross(x, normal)
    z /= np.linalg.norm(z)
    if np.dot(z, first - second) < 0:
        normal *= -1
        z *= -1
    return np.column_stack((x, normal, z))


def prepare_landmarks(landmarks, *, pinky_scale, pinky_palm_scale):
    """输入已由 VR 接收器校验；这里只拒绝退化的手部几何。"""
    bones = landmarks[BONE_CHILDREN] - landmarks[BONE_PARENTS]
    if np.any(np.linalg.norm(bones, axis=1) < 0.002):
        logger.warning("VR landmarks rejected: a bone is shorter than 2 mm")
        return None
    palm = estimate_palm_frame(landmarks)
    if palm is None:
        logger.warning("VR landmarks rejected: degenerate palm triangle")
        return None

    mano = landmarks @ palm @ OPERATOR_TO_MANO
    # 小指的 wrist→MCP 与 MCP→TIP 使用独立尺度。
    pinky = mano[17].copy()
    base = mano[0] + (pinky - mano[0]) * pinky_palm_scale
    mano[17:] = base + (mano[17:] - pinky) * pinky_scale
    if not np.isfinite(mano).all():
        raise ValueError("transformed hand landmarks must be finite")
    return mano

"""HTS 右手 21 点：掌坐标、固定骨长标定和 XHand 的降维几何目标。"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dexmani_real.config.teleop import HandRetargetingConfig

BONE_CHILDREN = np.arange(1, 21)
BONE_PARENTS = BONE_CHILDREN - 1
BONE_PARENTS[::4] = 0
FINGER_CHAINS = BONE_CHILDREN.reshape(5, 4)


class HandGeometryRejected(ValueError):
    """当前输入无法定义可用的手部几何。"""


@dataclass(frozen=True)
class HandGeometry:
    """中立几何；五指顺序为 Thumb/Index/Middle/Ring/Pinky，基部数组仅含前两指。"""

    neutral_directions: np.ndarray  # (5, 2, 3), neutral hand-base directions
    base_axes: np.ndarray  # (2, 3)
    base_indices: np.ndarray  # (2,), SDK order
    flex_indices: np.ndarray  # (5, 2), SDK order
    flex_axes: np.ndarray  # (5, 2, 3), neutral hand-base axes
    flex_coupling: np.ndarray  # (5,), relative signs of the parallel flexion axes
    limit_centers: np.ndarray  # (12,), rad
    base_directions: np.ndarray  # (2, 3), perpendicular to base_axes


def project_direction(vectors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """输入为单位方向或骨长加权平均方向；退化不产生任意替代方向。"""
    projected = vectors - np.sum(vectors * normals, axis=-1, keepdims=True) * normals
    norms = np.linalg.norm(projected, axis=-1, keepdims=True)
    if np.any(norms < 1e-6):
        raise HandGeometryRejected("hand direction projection is degenerate")
    return projected / norms


def signed_angle(initial: np.ndarray, target: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return np.arctan2(
        np.sum(axis * np.cross(initial, target), axis=-1),
        np.sum(initial * target, axis=-1),
    )


def rotate_vectors(vectors: np.ndarray, axes: np.ndarray, angles: np.ndarray) -> np.ndarray:
    cosine, sine = np.cos(angles)[..., None], np.sin(angles)[..., None]
    return (
        vectors * cosine + np.cross(axes, vectors) * sine
        + axes * np.sum(axes * vectors, axis=-1, keepdims=True) * (1 - cosine)
    )


def prepare_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """将米制右手输入转换到 XHand 对齐的掌系。"""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise HandGeometryRejected("right-hand landmarks must be finite with shape (21, 3)")
    lengths = np.linalg.norm(points[BONE_CHILDREN] - points[BONE_PARENTS], axis=1)
    if np.any(lengths < 0.002) or np.any(lengths > 0.18):
        raise HandGeometryRejected("hand bone lengths must be between 2 and 180 mm")
    centered = points - points[0]
    forward = centered[9]
    side = centered[5] - centered[17]
    forward = forward / np.linalg.norm(forward)
    side = side - np.dot(side, forward) * forward
    span = np.linalg.norm(side)
    if span < 0.01:
        raise HandGeometryRejected("degenerate palm: projected MCP span is below 10 mm")
    side /= span
    # 由有序解剖标记确定法向，不用 SVD 的任意符号，也不引入反射矩阵。
    normal = np.cross(side, forward)
    return centered @ np.column_stack((normal, side, forward))


def hand_lengths(points: np.ndarray) -> np.ndarray:
    """三个真实指骨长度及掌宽；不使用随屈曲变化的 MCP→TIP 直线距离。"""
    lengths = np.linalg.norm(np.diff(points[FINGER_CHAINS], axis=1), axis=2)
    return np.r_[lengths.ravel(), np.linalg.norm(points[5] - points[17])]


@dataclass(frozen=True)
class HandCalibration:
    bone_lengths: np.ndarray
    palm_width: float

    @classmethod
    def from_measurements(
        cls, measurements: Sequence[np.ndarray] | np.ndarray, config: HandRetargetingConfig,
    ) -> "HandCalibration":
        values = np.asarray(measurements, dtype=np.float64)
        if len(values) < config.calibration_samples:
            raise ValueError(f"hand calibration requires {config.calibration_samples} frames")
        median = np.median(values, axis=0)
        relative_mad = np.median(np.abs(values - median), axis=0) / median
        if np.max(relative_mad) > config.calibration_max_relative_mad:
            raise ValueError("hand calibration bone lengths are unstable")
        return cls(median[:15].reshape(5, 3), float(median[15]))

    def _finger_directions(self, unit_bones: np.ndarray) -> np.ndarray:
        """拇指使用最后两段，其余手指合并远端两段。"""
        directions = unit_bones[:, [0, 2]]
        directions[0] = unit_bones[0, 1:]
        distal_lengths = self.bone_lengths[1:, 1:]
        # 按标定骨长加权并保持无量纲，便于统一判断投影退化。
        directions[1:, 1] = np.sum(
            distal_lengths[:, :, None] * unit_bones[1:, 1:], axis=1,
        ) / distal_lengths.sum(axis=1)[:, None]
        return directions

    def project_qpos(
        self, points: np.ndarray, geometry: HandGeometry, *, tolerance: float,
    ) -> np.ndarray:
        """将 HTS 骨段方向投影为 SDK 顺序的 XHand 关节角。"""
        bones = np.diff(points[FINGER_CHAINS], axis=1)
        lengths = np.linalg.norm(bones, axis=2)
        palm_width = np.linalg.norm(points[5] - points[17])
        if (
            np.any(np.abs(lengths / self.bone_lengths - 1.0) > tolerance)
            or abs(palm_width / self.palm_width - 1.0) > tolerance
        ):
            raise HandGeometryRejected("hand bone lengths changed beyond the calibrated tolerance")
        unit_bones = bones / lengths[:, :, None]
        directions = self._finger_directions(unit_bones)
        base = project_direction(unit_bones[0, 0], geometry.base_axes[0])
        angles = np.empty(2)
        angles[0] = signed_angle(geometry.base_directions[0], base, geometry.base_axes[0])
        # 两段共同确定无向屈曲平面，避免近端跨 90° 时方位翻转。
        index = directions[1]
        c = index @ geometry.base_directions[1]
        s = index @ np.cross(geometry.base_axes[1], geometry.base_directions[1])
        x = float(np.sum(c * c - s * s))
        y = float(2 * np.sum(c * s))
        if np.hypot(x, y) < 1e-12:
            raise HandGeometryRejected("index flexion plane is degenerate")
        angles[1] = 0.5 * np.arctan2(y, x)
        directions[:2] = rotate_vectors(
            directions[:2], geometry.base_axes[:, None], -angles[:, None],
        )
        directions = project_direction(directions, geometry.flex_axes[:, :1])
        # 平行轴使远段绝对角可直接减去近端旋转；保留反向轴符号。
        flex = signed_angle(geometry.neutral_directions, directions, geometry.flex_axes)
        flex[:, 1] -= geometry.flex_coupling * flex[:, 0]
        requested_q = np.empty_like(geometry.limit_centers)
        requested_q[geometry.base_indices] = angles
        requested_q[geometry.flex_indices] = flex
        # 选择离操作区间最近的 2π 分支；保留越限请求的任务空间信息。
        center = geometry.limit_centers
        return center + np.arctan2(np.sin(requested_q - center), np.cos(requested_q - center))

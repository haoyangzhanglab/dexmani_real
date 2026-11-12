"""HTS 右手 21 点：掌坐标、固定骨长标定和 XHand 的降维几何目标。"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from dexmani_real.config.teleop import HandRetargetingConfig
from dexmani_real.geometry.kinematics import HandFK

BONE_CHILDREN = np.arange(1, 21)
BONE_PARENTS = BONE_CHILDREN - 1
BONE_PARENTS[::4] = 0
FINGER_CHAINS = np.arange(1, 21).reshape(5, 4)


class HandGeometryRejected(ValueError):
    """新鲜输入无法定义当前手部几何目标，可短暂保持上一命令。"""


@dataclass(frozen=True)
class HandTargets:
    positions: np.ndarray  # (5, 3, 3): requested FK base / intermediate / tip
    directions: np.ndarray  # (5, 2, 3): requested FK proximal / terminal
    activation: np.ndarray  # (4,): thumb–index/middle/ring/pinky
    pinch_distances: np.ndarray  # (4,), m


@dataclass(frozen=True)
class HandGeometry:
    """中立几何；五指顺序为 Thumb/Index/Middle/Ring/Pinky，基部数组仅含前两指。"""

    positions: np.ndarray  # (5, 3, 3), m
    segments: np.ndarray  # (5, 2), m
    base_axes: np.ndarray  # (2, 3)
    base_indices: np.ndarray  # (2,), SDK order
    flex_indices: np.ndarray  # (5, 2), SDK order
    flex_axes: np.ndarray  # (5, 2, 3), neutral hand-base axes
    limits: np.ndarray  # (12, 2), operating limits, rad
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
    """已准入的米制右手输入；输出掌系 (+X 掌侧、+Y 拇指侧、+Z 指向中指)。"""
    points = np.asarray(landmarks, dtype=np.float64)
    lengths = np.linalg.norm(points[BONE_CHILDREN] - points[BONE_PARENTS], axis=1)
    if np.any(lengths < 0.002) or np.any(lengths > 0.18):
        raise HandGeometryRejected("hand bone lengths must be between 2 and 180 mm")
    centered = points - points[0]
    forward = centered[9]
    side = centered[5] - centered[17]
    forward = forward / np.linalg.norm(forward)
    side = side - np.dot(side, forward) * forward
    if np.linalg.norm(side) < 0.01:
        raise HandGeometryRejected("degenerate palm: index/pinky span is below 10 mm")
    side /= np.linalg.norm(side)
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

    def targets(
        self, points: np.ndarray, geometry: HandGeometry, config: HandRetargetingConfig,
        *, fk: HandFK, frame_ids: Sequence[int],
    ) -> HandTargets:
        """由未裁剪请求的 FK 生成全部目标；安全边界由求解器约束。"""
        bones = np.diff(points[FINGER_CHAINS], axis=1)
        lengths = np.linalg.norm(bones, axis=2)
        measured = np.r_[lengths.ravel(), np.linalg.norm(points[5] - points[17])]
        reference = np.r_[self.bone_lengths.ravel(), self.palm_width]
        if np.max(np.abs(measured / reference - 1.0)) > config.bone_length_tolerance:
            raise HandGeometryRejected("hand bone lengths changed beyond the calibrated tolerance")
        unit_bones = bones / lengths[:, :, None]
        proximal = unit_bones[:, 0].copy()
        proximal[0] = unit_bones[0, 1]
        distal = unit_bones[:, 2].copy()
        # 归一化权重使退化阈值无量纲，并保留两段标定骨长的比例。
        distal[1:] = np.sum(
            self.bone_lengths[1:, 1:, None] * unit_bones[1:, 1:], axis=1,
        ) / self.bone_lengths[1:, 1:].sum(axis=1)[:, None]
        base = project_direction(unit_bones[0, 0], geometry.base_axes[0])
        angles = np.empty(2)
        angles[0] = signed_angle(geometry.base_directions[0], base, geometry.base_axes[0])
        # 两段共同确定无向屈曲平面，避免近端跨 90° 时方位翻转。
        index = np.stack((proximal[1], distal[1]))
        c = index @ geometry.base_directions[1]
        s = index @ np.cross(geometry.base_axes[1], geometry.base_directions[1])
        x, y = np.sum(c * c - s * s), 2 * np.sum(c * s)
        if np.hypot(x, y) < 1e-12:
            raise HandGeometryRejected("index flexion plane is degenerate")
        angles[1] = 0.5 * np.arctan2(y, x)
        directions = np.stack((proximal, distal), axis=1)
        directions[:2] = rotate_vectors(
            directions[:2], geometry.base_axes[:, None], -angles[:, None],
        )
        directions = project_direction(directions, geometry.flex_axes[:, :1])
        neutral = np.diff(geometry.positions, axis=1) / geometry.segments[:, :, None]
        # 平行轴使远段绝对角可直接减去近端旋转；保留反向轴符号。
        flex = signed_angle(neutral, directions, geometry.flex_axes)
        flex[:, 1] -= np.sum(geometry.flex_axes[:, 0] * geometry.flex_axes[:, 1], axis=1) * flex[:, 0]
        requested = np.empty(12)
        requested[geometry.base_indices] = angles
        requested[geometry.flex_indices] = flex
        # 选择离操作区间最近的 2π 分支；保留越限请求的任务空间信息。
        center = geometry.limits.mean(axis=1)
        requested = center + np.arctan2(np.sin(requested - center), np.cos(requested - center))
        targets = fk.forward(requested, frame_ids)[:, :3, 3].reshape(5, 3, 3)
        segments = np.diff(targets, axis=1)
        directions = segments / np.linalg.norm(segments, axis=2)[:, :, None]
        distances = np.linalg.norm(targets[1:, 2] - targets[0, 2], axis=1)
        x = np.clip(
            (config.pinch_start_m - distances) / (config.pinch_start_m - config.pinch_full_m),
            0.0, 1.0,
        )
        activation = x * x * (3.0 - 2.0 * x)
        return HandTargets(
            positions=targets, directions=directions,
            activation=activation, pinch_distances=distances,
        )

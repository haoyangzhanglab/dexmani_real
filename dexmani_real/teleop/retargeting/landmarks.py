"""HTS 右手 21 点：掌坐标、固定骨长标定和 XHand 的降维几何目标。"""

from dataclasses import dataclass

import numpy as np

BONE_CHILDREN = np.arange(1, 21)
BONE_PARENTS = BONE_CHILDREN - 1
BONE_PARENTS[::4] = 0
FINGER_CHAINS = np.arange(1, 21).reshape(5, 4)
TIP_INDICES = np.array([4, 8, 12, 16, 20])


@dataclass(frozen=True)
class HandTargets:
    positions: np.ndarray
    directions: np.ndarray  # (5, 2, 3): proximal / terminal
    activation: np.ndarray
    pinch_distances: np.ndarray
    thumb_base_observable: bool


def prepare_landmarks(landmarks):
    """米制、右手输入；输出右手掌系 (+X 掌侧、+Y 拇指侧、+Z 指向中指)。"""
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("hand landmarks must be a finite (21, 3) array in metres")
    lengths = np.linalg.norm(points[BONE_CHILDREN] - points[BONE_PARENTS], axis=1)
    if np.any(lengths < 0.002) or np.any(lengths > 0.18):
        raise ValueError("hand bone lengths must be between 2 and 180 mm")
    centered = points - points[0]
    forward = centered[9]
    side = centered[5] - centered[17]
    forward = forward / np.linalg.norm(forward)
    side = side - np.dot(side, forward) * forward
    if np.linalg.norm(side) < 0.01:
        raise ValueError("degenerate palm: index/pinky span is below 10 mm")
    side /= np.linalg.norm(side)
    # 由有序解剖标记确定法向，不用 SVD 的任意符号，也不引入反射矩阵。
    normal = np.cross(side, forward)
    return centered @ np.column_stack((normal, side, forward))


def hand_lengths(points):
    """三个真实指骨长度及掌宽；不使用随屈曲变化的 MCP→TIP 直线距离。"""
    lengths = np.linalg.norm(np.diff(points[FINGER_CHAINS], axis=1), axis=2)
    return np.r_[lengths.ravel(), np.linalg.norm(points[5] - points[17])]


@dataclass(frozen=True)
class HandCalibration:
    bone_lengths: np.ndarray
    palm_width: float
    sample_count: int

    @classmethod
    def from_measurements(cls, measurements, config):
        values = np.asarray(measurements, dtype=np.float64)
        if len(values) < config.calibration_samples:
            raise ValueError(f"hand calibration requires {config.calibration_samples} frames")
        median = np.median(values, axis=0)
        relative_mad = np.median(np.abs(values - median), axis=0) / median
        if np.max(relative_mad) > config.calibration_max_relative_mad:
            raise ValueError("hand calibration bone lengths are unstable")
        # MAD 对少量异常值不敏感；每个样本还须满足运行时的骨长容差。
        if np.max(np.abs(values / median - 1.0)) > config.bone_length_tolerance:
            raise ValueError("hand bone lengths changed beyond the calibrated tolerance")
        return cls(median[:15].reshape(5, 3), float(median[15]), len(values))

    def targets(
        self, points, robot_points, config, *, robot_segments, robot_palm_width,
        thumb_base_xy, thumb_base_radius,
    ):
        """校验标定骨长并映射三点/指：thumb MCP/IP/TIP；其余 MCP/PIP/TIP。"""
        chains = points[FINGER_CHAINS]
        bones = np.diff(chains, axis=1)
        lengths = np.linalg.norm(bones, axis=2)
        measured = np.r_[lengths.ravel(), np.linalg.norm(points[5] - points[17])]
        reference = np.r_[self.bone_lengths.ravel(), self.palm_width]
        if np.max(np.abs(measured / reference - 1.0)) > config.bone_length_tolerance:
            raise ValueError("hand bone lengths changed beyond the calibrated tolerance")
        unit_bones = bones / lengths[:, :, None]
        base = unit_bones[0, 0, :2]
        norm = np.linalg.norm(base)
        observable = bool(norm > 1e-6)
        targets = robot_points.copy()
        # 拇指第一轴只改变 MCP 的 XY 方位；高度和轨迹半径由 URDF 固定。
        if observable:
            targets[0, 0, :2] = thumb_base_xy + base / norm * thumb_base_radius
        # 否则沿用中立占位坐标；Objective 显式关闭不可辨识的拇指位置项。
        targets[0, 1] = targets[0, 0] + bones[0, 1] * robot_segments[0, 0] / self.bone_lengths[0, 1]
        targets[0, 2] = targets[0, 1] + bones[0, 2] * robot_segments[0, 1] / self.bone_lengths[0, 2]
        targets[1:, 1] = (
            targets[1:, 0] + bones[1:, 0]
            * (robot_segments[1:, 0] / self.bone_lengths[1:, 0])[:, None]
        )
        # 位置目标合并 PIP→DIP→TIP，末段方向另行保留。
        targets[1:, 2] = (
            targets[1:, 1] + (chains[1:, 3] - chains[1:, 1])
            * (robot_segments[1:, 1] / self.bone_lengths[1:, 1:].sum(axis=1))[:, None]
        )
        directions = unit_bones[:, [0, 2]]
        directions[0, 0] = unit_bones[0, 1]  # 拇指近段为 MCP→IP。
        palm_scale = robot_palm_width / self.palm_width
        distances = np.linalg.norm(points[TIP_INDICES[1:]] - points[4], axis=1) * palm_scale
        x = np.clip(
            (config.pinch_start_m - distances) / (config.pinch_start_m - config.pinch_full_m),
            0.0, 1.0,
        )
        activation = x * x * (3.0 - 2.0 * x)
        # 距离接近不足以证明对掌；末段夹角 60→90 度逐渐启用辅助。
        opposition = np.degrees(np.arccos(np.clip(unit_bones[1:, 2] @ unit_bones[0, 2], -1, 1)))
        x = np.clip((opposition - 60.0) / 30.0, 0.0, 1.0)
        activation *= x * x * (3.0 - 2.0 * x)
        return HandTargets(targets, directions, activation, distances, observable)

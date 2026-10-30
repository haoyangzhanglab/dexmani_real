"""HTS 右手 21 点：掌坐标、固定骨长标定和 XHand 的降维几何目标。"""

from dataclasses import dataclass

import numpy as np

BONE_CHILDREN = np.arange(1, 21)
BONE_PARENTS = BONE_CHILDREN - 1
BONE_PARENTS[::4] = 0
FINGER_CHAINS = np.arange(1, 21).reshape(5, 4)
TIP_INDICES = np.array([4, 8, 12, 16, 20])


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
    def from_metadata(cls, metadata):
        if metadata.get("method") != "adaptive_xhand_v1":
            raise ValueError("unsupported hand calibration method")
        bones = np.asarray(metadata["human_bone_lengths_m"], dtype=np.float64)
        width, count = metadata["human_palm_width_m"], metadata["sample_count"]
        if (
            bones.shape != (5, 3) or not np.isfinite(bones).all()
            or np.any(bones < 0.002) or np.any(bones > 0.18)
            or not np.isfinite(width) or not 0.01 <= width <= 0.36
            or type(count) is not int or count < 2
        ):
            raise ValueError("invalid saved hand morphology calibration")
        return cls(bones.copy(), float(width), count)

    @classmethod
    def from_frames(cls, frames, *, min_samples, max_relative_mad):
        values = np.array([hand_lengths(prepare_landmarks(frame)) for frame in frames])
        if len(values) < min_samples:
            raise ValueError(f"hand calibration requires {min_samples} distinct frames")
        median = np.median(values, axis=0)
        relative_mad = np.median(np.abs(values - median), axis=0) / median
        if np.max(relative_mad) > max_relative_mad:
            raise ValueError("hand calibration bone lengths are unstable")
        return cls(median[:15].reshape(5, 3), float(median[15]), len(values))

    def validate_frame(self, points, tolerance):
        reference = np.r_[self.bone_lengths.ravel(), self.palm_width]
        if np.max(np.abs(hand_lengths(points) / reference - 1.0)) > tolerance:
            raise ValueError("hand bone lengths changed beyond the calibrated tolerance")

    def targets(self, points, robot_points, thumb_cmc, config):
        """三点/指：thumb MCP/IP/TIP；其余 MCP/PIP/TIP（合并人手 DIP）。"""
        chains = points[FINGER_CHAINS]
        bones = np.diff(chains, axis=1)
        robot_segments = np.linalg.norm(np.diff(robot_points, axis=1), axis=2)
        targets = robot_points.copy()
        thumb_metacarpal = np.linalg.norm(robot_points[0, 0] - thumb_cmc)
        targets[0, 0] = thumb_cmc + bones[0, 0] * thumb_metacarpal / self.bone_lengths[0, 0]
        targets[0, 1] = targets[0, 0] + bones[0, 1] * robot_segments[0, 0] / self.bone_lengths[0, 1]
        targets[0, 2] = targets[0, 1] + bones[0, 2] * robot_segments[0, 1] / self.bone_lengths[0, 2]
        targets[1:, 1] = (
            targets[1:, 0] + bones[1:, 0]
            * (robot_segments[1:, 0] / self.bone_lengths[1:, 0])[:, None]
        )
        # 先从原始点计算合并向量，再缩放；不会逐点覆盖造成骨段翻转。
        targets[1:, 2] = (
            targets[1:, 1] + (chains[1:, 3] - chains[1:, 1])
            * (robot_segments[1:, 1] / self.bone_lengths[1:, 1:].sum(axis=1))[:, None]
        )
        directions = bones[:, 2] / np.linalg.norm(bones[:, 2], axis=1)[:, None]
        palm_scale = np.linalg.norm(robot_points[1, 0] - robot_points[4, 0]) / self.palm_width
        distances = np.linalg.norm(points[TIP_INDICES[1:]] - points[4], axis=1) * palm_scale
        x = np.clip(
            (config.pinch_start_m - distances) / (config.pinch_start_m - config.pinch_full_m),
            0.0, 1.0,
        )
        activation = x * x * (3.0 - 2.0 * x)
        pinch_distances = (1.0 - activation) * distances + activation * config.pinch_gap_m
        return targets, directions, activation, pinch_distances

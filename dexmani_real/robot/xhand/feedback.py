"""XHand SDK 状态、SDK / 255 触觉缓存与反馈时效。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.model import (
    HAND_FINGER_COUNT,
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    XHAND_TACTILE_SCALE_DIVISOR,
)

TACTILE_CHANNELS = ("aggregate", "dense")

HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", HAND_JOINT_SHAPE),
        ("tactile_aggregate", "<f4", HAND_TACTILE_SUM_SHAPE),
        ("tactile_dense", "<f4", HAND_TACTILE_FORCE_SHAPE),
        # 每种触觉分别保留各指最后有效样本的查询起点，0 表示无有效样本。
        ("tactile_aggregate_timestamp_ns", "<u8", (HAND_FINGER_COUNT,)),
        ("tactile_dense_timestamp_ns", "<u8", (HAND_FINGER_COUNT,)),
        # 过流状态与关节位置共用来源时间，独立于逐指触觉缓存。
        ("overcurrent", "?"),
        ("overcurrent_joint_mask", "?", HAND_JOINT_SHAPE),
        ("timestamp_ns", "<u8"),
    ]
)


@dataclass
class XHandState:
    """触觉采用 SDK 单位，可应用 Bias。单指缺失为 NaN，整种模态缺失为 None。"""

    qpos: np.ndarray
    tactile_aggregate: np.ndarray | None
    tactile_dense: np.ndarray | None
    timestamp_ns: int
    overcurrent: bool
    overcurrent_joint_mask: np.ndarray


class XHandFeedback:
    """只读反馈缓存；各指独立更新，新触觉样本转换为 SDK / 255。"""

    def __init__(self) -> None:
        self.latest: np.ndarray | None = None

    def clear_tactile(self) -> None:
        """显式清零开始后旧触觉不再可用；保留独立关节反馈。"""
        if self.latest is None:
            return
        frame = self.latest.copy()
        for name in TACTILE_CHANNELS:
            frame[f"tactile_{name}"] = np.nan
            frame[f"tactile_{name}_timestamp_ns"] = 0
        frame.flags.writeable = False
        self.latest = frame

    def consume(self, state: XHandState | None) -> np.ndarray | None:
        if state is None:
            return self.latest

        if self.latest is None:
            frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
            frame["tactile_aggregate"] = np.nan
            frame["tactile_dense"] = np.nan
        else:
            frame = self.latest.copy()
        frame["qpos"] = state.qpos
        frame["overcurrent"] = state.overcurrent
        frame["overcurrent_joint_mask"] = state.overcurrent_joint_mask
        for name in TACTILE_CHANNELS:
            values = getattr(state, f"tactile_{name}")
            if values is not None:
                valid = np.isfinite(values).reshape(HAND_FINGER_COUNT, -1).all(axis=1)
                frame[f"tactile_{name}"][0, valid] = values[valid] / XHAND_TACTILE_SCALE_DIVISOR
                frame[f"tactile_{name}_timestamp_ns"][0, valid] = state.timestamp_ns
        frame["timestamp_ns"] = state.timestamp_ns
        frame.flags.writeable = False
        self.latest = frame
        return frame

    def snapshot(self) -> np.ndarray | None:
        """只读缓存；来源年龄由观测或实际发送边界裁决。"""
        return self.latest


def tactile_validity(frame, channel, *, max_age_s, now_ns):
    """返回当前时刻的五指有效性，完整模态要求五指均有效。"""
    if frame is None:
        return np.zeros(HAND_FINGER_COUNT, dtype=bool)
    stamps = frame[f"tactile_{channel}_timestamp_ns"][0]
    oldest_ns = max(0, now_ns - int(max_age_s * 1e9))
    return (stamps > oldest_ns) & (stamps <= now_ns)


def expire_tactile(frame, *, max_age_s, now_ns):
    """为观测裁掉过期触觉，保留缓存自身与关节反馈。"""
    if frame is None:
        return None
    result = frame
    for channel in TACTILE_CHANNELS:
        timestamps = f"tactile_{channel}_timestamp_ns"
        expired = (frame[timestamps][0] != 0) & ~tactile_validity(frame, channel, max_age_s=max_age_s, now_ns=now_ns)
        if expired.any():
            if result is frame:
                result = frame.copy()
            result[f"tactile_{channel}"][0, expired] = np.nan
            result[timestamps][0, expired] = 0
    result.flags.writeable = False
    return result

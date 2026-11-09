"""XHand SDK Native 状态、SDK / 255 触觉缓存与反馈时效。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.model import (
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
    XHAND_TACTILE_SCALE_DIVISOR,
)

HAND_STATE_DTYPE = np.dtype(
    [
        ("qpos", "<f8", HAND_JOINT_SHAPE),
        ("tactile_aggregate", "<f4", HAND_TACTILE_SUM_SHAPE),
        ("tactile_dense", "<f4", HAND_TACTILE_FORCE_SHAPE),
        # 每个触觉通道最后一份完整有效帧的来源时间。
        ("tactile_aggregate_timestamp_ns", "<u8"),
        ("tactile_dense_timestamp_ns", "<u8"),
        # 关节来源时间独立于触觉缓存。
        ("timestamp_ns", "<u8"),
    ]
)


@dataclass
class XHandState:
    """关节可信；触觉为 SDK Native 完整有效帧或 None；时间为主机 SDK 查询起点，非硬件采样时刻。"""

    qpos: np.ndarray
    tactile_aggregate: np.ndarray | None
    tactile_dense: np.ndarray | None
    timestamp_ns: int


class XHandFeedback:
    """只读反馈缓存；两种触觉仅在新有效帧入缓存时转换为 SDK / 255。"""

    def __init__(self):
        self.latest: np.ndarray | None = None

    def clear_tactile(self) -> None:
        """显式清零开始后旧触觉不再可用；保留独立关节反馈。"""
        if self.latest is None:
            return
        frame = self.latest.copy()
        for name in ("aggregate", "dense"):
            frame[f"tactile_{name}"] = np.nan
            frame[f"tactile_{name}_timestamp_ns"] = 0
        frame.flags.writeable = False
        self.latest = frame

    def consume(self, state: XHandState | None) -> np.ndarray | None:
        if state is None:
            return self.snapshot()

        if self.latest is None:
            frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
            frame["tactile_aggregate"] = np.nan
            frame["tactile_dense"] = np.nan
        else:
            frame = self.latest.copy()
        frame["qpos"] = state.qpos
        for name in ("aggregate", "dense"):
            values = getattr(state, f"tactile_{name}")
            if values is not None:
                frame[f"tactile_{name}"] = (
                    np.asarray(values, dtype=np.float32)
                    / np.float32(XHAND_TACTILE_SCALE_DIVISOR)
                )
                frame[f"tactile_{name}_timestamp_ns"] = state.timestamp_ns
        frame["timestamp_ns"] = state.timestamp_ns
        frame.flags.writeable = False
        self.latest = frame
        return self.snapshot()

    def snapshot(self) -> np.ndarray | None:
        """只读缓存；来源年龄由观测或实际发送边界裁决。"""
        return self.latest

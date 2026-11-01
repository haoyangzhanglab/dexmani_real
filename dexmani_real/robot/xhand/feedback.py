"""XHand 标准化状态、触觉缓存与反馈时效。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dexmani_real.config.xhand import INITIAL_STATE_TIMEOUT_S
from dexmani_real.robot.model import (
    HAND_JOINT_SHAPE,
    HAND_TACTILE_FORCE_SHAPE,
    HAND_TACTILE_SUM_SHAPE,
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
    """关节可信；触觉为准确形状、有限值的 float32 完整帧，或无新帧的 None。"""

    qpos: np.ndarray
    tactile_aggregate: np.ndarray | None
    tactile_dense: np.ndarray | None


class XHandFeedback:
    def __init__(self):
        self.latest: np.ndarray | None = None
        self.initial_deadline_ns = 0

    def reset(self, now_ns: int, *, deadline_ns: int | None = None) -> None:
        self.latest = None
        self.initial_deadline_ns = (
            now_ns + int(INITIAL_STATE_TIMEOUT_S * 1e9) if deadline_ns is None else deadline_ns
        )

    def consume(self, state: XHandState | None, *, source_ns: int) -> np.ndarray | None:
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
                frame[f"tactile_{name}"] = values
                frame[f"tactile_{name}_timestamp_ns"] = source_ns
        frame["timestamp_ns"] = source_ns
        frame.flags.writeable = False
        self.latest = frame
        return self.snapshot()

    def snapshot(self) -> np.ndarray | None:
        """只读缓存；来源年龄由观测或实际发送边界裁决。"""
        return self.latest

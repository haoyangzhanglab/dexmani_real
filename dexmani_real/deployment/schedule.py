"""预测请求、动作窗口与控制槽。"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    slot: int
    handoff_slot: int
    deadline_ns: int
    result: np.ndarray | None = None
    request_cursor: int = 0


@dataclass
class ActionChunk:
    actions: np.ndarray
    anchor_slot: int  # async 切片首步的槽号。
    deadline_ns: int  # 沿用原预测的截止。


def control_slot(*, now_ns, next_step_ns, last_slot, dt_ns):
    if now_ns < next_step_ns:
        return None, 0
    skipped = (now_ns - next_step_ns) // dt_ns
    slot = last_slot + 1 + skipped
    return slot, skipped

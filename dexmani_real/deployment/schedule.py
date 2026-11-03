"""async/RTC 的时间锚点动作窗口。"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    """slot 是观测锚点，handoff_slot 是最早或约定交接时刻。"""

    slot: int
    handoff_slot: int | None
    result: np.ndarray | None = None


@dataclass
class ActionChunk:
    actions: np.ndarray
    anchor_slot: int


def window_end(plan, action_steps):
    return plan.anchor_slot + min(action_steps, len(plan.actions))


def handoff_plan(query, *, slot, mode):
    if query is None:
        return None, None
    bootstrap = query.handoff_slot is None
    handoff = slot if bootstrap and query.result is not None else query.handoff_slot
    if handoff is None or slot < handoff:
        return None, None
    if mode != "async" and slot != handoff:
        return None, "handoff_slot_missed"
    if query.result is None:
        return (None, None) if mode == "async" else (None, "handoff_result_not_ready")
    # RTC bootstrap 尚无既有前缀承诺。
    offset = 0 if bootstrap and mode == "rtc" else slot - query.slot
    if not 0 <= offset < len(query.result):
        return None, "handoff_prediction_exhausted"
    return ActionChunk(query.result[offset:].copy(), slot), None


def prefetch_handoff(plan, *, slot, mode, action_steps, prefetch_steps, query_pending):
    end = window_end(plan, action_steps)
    prefetch_slot = end - prefetch_steps
    if slot < prefetch_slot or (mode == "rtc" and slot != prefetch_slot):
        return None, None
    if query_pending:
        return None, None if mode == "async" else "prefetch_unavailable"
    return end, None


def control_slot(*, now_ns, next_step_ns, last_slot, dt_ns):
    """返回当前时间槽及跳过的槽数。"""
    if now_ns < next_step_ns:
        return None, next_step_ns, 0
    skipped = (now_ns - next_step_ns) // dt_ns
    slot = last_slot + 1 + skipped
    return slot, next_step_ns + (skipped + 1) * dt_ns, skipped


def plan_rejection(plan, slot, mode, action_steps):
    end = plan.anchor_slot + len(plan.actions) if mode == "async" else window_end(plan, action_steps)
    if not plan.anchor_slot <= slot < end:
        return "plan_exhausted_without_handoff"
    return None

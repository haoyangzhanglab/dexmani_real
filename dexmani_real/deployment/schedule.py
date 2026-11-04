"""普通 async 的时间窗口，以及异步请求快照。"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Prediction:
    slot: int
    handoff_slot: int | None
    result: np.ndarray | None = None
    request_cursor: int = 0


@dataclass
class ActionChunk:
    actions: np.ndarray
    anchor_slot: int


def handoff_plan(query: Prediction | None, *, slot: int) -> ActionChunk | None:
    if query is None or query.result is None:
        return None
    if query.handoff_slot is not None and slot < query.handoff_slot:
        return None
    offset = slot - query.slot
    return ActionChunk(query.result[offset:], slot)


def prefetch_handoff(plan, *, slot, action_steps, prefetch_steps):
    end = plan.anchor_slot + min(action_steps, len(plan.actions))
    return end if slot >= end - prefetch_steps else None


def control_slot(*, now_ns, next_step_ns, last_slot, dt_ns):
    if now_ns < next_step_ns:
        return None, next_step_ns, 0
    skipped = (now_ns - next_step_ns) // dt_ns
    slot = last_slot + 1 + skipped
    return slot, next_step_ns + (skipped + 1) * dt_ns, skipped


def plan_exhausted(plan, slot):
    return slot >= plan.anchor_slot + len(plan.actions)

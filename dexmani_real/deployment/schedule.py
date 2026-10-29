"""sync、async 与 RTC 的动作窗口调度。"""

from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.commands import RobotCommand


@dataclass
class Prediction:
    """slot 是观测锚点，handoff_slot 是最早交接时刻。"""

    run_id: int
    slot: int
    handoff_slot: int | None
    result: np.ndarray | None = None


@dataclass
class ActionChunk:
    """actions 从 anchor_slot 索引，async 可继续使用 segment_end 后的有效尾部。"""

    actions: np.ndarray
    anchor_slot: int
    segment_end: int
    frozen: dict[int, RobotCommand]


def handoff_plan(
    query, *, run_id, slot, mode, action_steps, authorized,
):
    if query is None or query.run_id != run_id:
        return None, None
    bootstrap = query.handoff_slot is None
    handoff = slot if bootstrap and query.result is not None else query.handoff_slot
    if handoff is None or slot < handoff:
        return None, None
    if not authorized:
        return None, "handoff_authority_revoked"
    ordinary_async = mode == "async"
    if not ordinary_async and slot != handoff:
        return None, "handoff_slot_missed"
    if query.result is None:
        if ordinary_async:
            return None, None
        return None, "handoff_result_not_ready"
    segment_end = slot + action_steps
    if ordinary_async:
        if not 0 <= slot - query.slot < len(query.result):
            return None, "handoff_prediction_exhausted"
        segment_end = min(segment_end, query.slot + len(query.result))
    return (
        ActionChunk(
            actions=query.result,
            anchor_slot=slot if bootstrap and mode == "rtc" else query.slot,
            segment_end=segment_end,
            frozen={},
        ),
        None,
    )


def prefetch_handoff(
    plan, *, slot, mode, prefetch_steps, query_pending,
):
    if mode == "sync":
        return None, None
    prefetch_slot = plan.segment_end - prefetch_steps
    if mode == "async":
        if slot < prefetch_slot:
            return None, None
    elif slot != prefetch_slot:
        return None, None
    if query_pending:
        return None, None if mode == "async" else "prefetch_unavailable"
    return plan.segment_end, None


def control_slot(*, now_ns, next_step_ns, last_slot, dt_ns, synchronous):
    """返回到期控制步与下一期限，不补发错过的动作。"""
    if now_ns < next_step_ns:
        return None, next_step_ns, 0
    skipped = (now_ns - next_step_ns) // dt_ns
    # sync 的动作序号只随真实控制步推进，async 则按预测的时间锚点索引。
    slot = last_slot + 1 + (0 if synchronous else skipped)
    return slot, next_step_ns + (skipped + 1) * dt_ns, skipped


def plan_rejection(plan, slot, mode):
    if mode == "async":
        if not 0 <= slot - plan.anchor_slot < len(plan.actions):
            return "plan_exhausted_without_handoff"
    elif slot >= plan.segment_end:
        return "plan_exhausted_without_handoff"
    return None

"""sync、async 与 RTC 的动作窗口调度。"""

from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.commands import RobotCommand
from dexmani_real.runtime.state import RunEndReason


@dataclass
class PolicyQuery:
    """slot 是观测锚点，handoff_slot 是最早交接时刻。"""

    query_id: int
    run_id: int
    slot: int
    expires_at_ns: int
    handoff_slot: int | None
    result: np.ndarray | None = None


@dataclass
class ActionPlan:
    """actions 从 anchor_slot 索引，async 可继续使用 segment_end 后的有效尾部。"""

    actions: np.ndarray
    anchor_slot: int
    segment_end: int
    expires_at_ns: int
    frozen: dict[int, RobotCommand]


@dataclass(frozen=True)
class ControlRejection:
    detail: str
    run_end_reason: RunEndReason = RunEndReason.EXECUTOR_BOUNDARY


def handoff_plan(
    query, *, run_id, slot, mode, action_steps, now_ns, authorized,
):
    if query is None or query.run_id != run_id:
        return None, None
    bootstrap = query.handoff_slot is None
    handoff = slot if bootstrap and query.result is not None else query.handoff_slot
    if handoff is None or slot < handoff:
        return None, None
    if not authorized:
        return None, ControlRejection("handoff_authority_revoked")
    ordinary_async = mode == "async" and not bootstrap
    if not ordinary_async and slot != handoff:
        return None, ControlRejection("handoff_slot_missed")
    if query.result is None:
        if ordinary_async:
            return None, None
        return None, ControlRejection("handoff_result_not_ready")
    segment_end = slot + action_steps
    if ordinary_async:
        if not 0 <= slot - query.slot < len(query.result):
            return None, ControlRejection("handoff_prediction_exhausted")
        if now_ns >= query.expires_at_ns:
            return None, ControlRejection("handoff_decision_expired")
        segment_end = min(segment_end, query.slot + len(query.result))
    return (
        ActionPlan(
            actions=query.result,
            anchor_slot=slot if bootstrap else query.slot,
            segment_end=segment_end,
            expires_at_ns=query.expires_at_ns,
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
        return None, None if mode == "async" else ControlRejection("prefetch_unavailable")
    return plan.segment_end, None


def control_slot(*, now_ns, next_step_ns, last_slot, dt_ns, synchronous, resume_ready):
    """返回到期控制步与下一期限，不补发错过的动作。"""
    if synchronous and resume_ready:
        next_step_ns = min(next_step_ns, now_ns)
    if now_ns < next_step_ns:
        return None, next_step_ns, 0
    slot = last_slot + 1
    skipped = (now_ns - next_step_ns) // dt_ns
    if skipped:
        return slot, next_step_ns, skipped
    return slot, (now_ns if synchronous else next_step_ns) + dt_ns, 0


def run_budget(*, run_id, started_ns, max_running_s, wait_started_ns, max_wait_s):
    deadlines = []
    if run_id is not None:
        if max_running_s is not None:
            deadlines.append((started_ns + int(max_running_s * 1e9), "duration"))
        if wait_started_ns is not None:
            deadlines.append((wait_started_ns + int(max_wait_s * 1e9), "wait"))
    # 同时到期时优先报告等待超时。
    return min(deadlines, key=lambda item: (item[0], item[1] != "wait")) if deadlines else None


def plan_rejection(plan, slot, mode, now_ns):
    if mode == "async":
        if not 0 <= slot - plan.anchor_slot < len(plan.actions):
            return ControlRejection("plan_exhausted_without_handoff")
        if now_ns >= plan.expires_at_ns:
            return ControlRejection("submission_deadline_expired")
    elif slot >= plan.segment_end:
        return ControlRejection("plan_exhausted_without_handoff")
    return None

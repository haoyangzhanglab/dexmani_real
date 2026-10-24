"""Pure Policy scheduling; the episode supplies clocks and handles execution I/O."""

from dataclasses import dataclass

import numpy as np

from dexmani_real.robot.commands import RobotCommand
from dexmani_real.runtime.state import RunEndReason


@dataclass
class PolicyQuery:
    """slot anchors the observation; handoff_slot is the earliest planned takeover."""

    query_id: int
    run_id: int
    slot: int
    expires_at_ns: int
    handoff_slot: int | None
    result: np.ndarray | None = None


@dataclass
class ActionPlan:
    """actions index from anchor_slot; async may consume the tail past nominal segment_end."""

    actions: np.ndarray
    anchor_slot: int
    segment_end: int
    expires_at_ns: int
    frozen: dict[int, RobotCommand]


@dataclass(frozen=True)
class ControlRejection:
    """A scheduling/target decision; only the episode owner applies termination."""

    detail: str
    run_end_reason: RunEndReason = RunEndReason.EXECUTOR_BOUNDARY


def decision_expires_at_ns(timestamps, now_ns, max_age_ns):
    """An exclusive deadline preserves age <= budget on the integer-nanosecond clock."""
    if not timestamps or any(stamp <= 0 or stamp > now_ns for stamp in timestamps):
        return None
    expires_at_ns = min(timestamps) + max_age_ns + 1
    return expires_at_ns if now_ns < expires_at_ns else None


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
    plan, *, slot, mode, prefetch_steps, query_pending, model_pending, history_ready,
):
    if mode == "sync":
        return None, None
    prefetch_slot = plan.segment_end - prefetch_steps
    if mode == "async":
        if slot < prefetch_slot:
            return None, None
    elif slot != prefetch_slot:
        return None, None
    if query_pending or model_pending:
        return None, None if mode == "async" else ControlRejection("prefetch_unavailable")
    if not history_ready:
        return None, ControlRejection("prefetch_unavailable")
    return plan.segment_end, None


def control_slot(*, now_ns, next_step_ns, last_slot, dt_ns, synchronous, resume_ready):
    """Return the due slot, next deadline and missed count without catch-up sends."""
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
    # WAIT wins a tie: no action was supplied before its strict deadline.
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


def rtc_prefix_request(plan, slot, prefetch_steps):
    """Owned guidance tail plus the slots whose physical commands must be committed."""
    return plan.actions[slot - plan.anchor_slot:].copy(), range(slot, slot + prefetch_steps)

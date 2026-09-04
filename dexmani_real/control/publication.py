"""Physical command preparation, realtime publication, and blocking acceptance."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.ipc.schema import COUPLED_COMMAND_DTYPE, HAND_JOINT_SHAPE
from dexmani_real.runtime.safety import (
    PUBLISH_REASON_ESTOP,
    PUBLISH_REASON_EXPIRED,
    PUBLISH_REASON_FAULT,
    PUBLISH_REASON_GENERATION,
    PUBLISH_REASON_RUNTIME_STOPPED,
    PUBLISH_REASON_SAFETY_STATE,
    CoupledCommandTicket,
    SafetyState,
    cancel_coupled_command_if_current,
    coupled_command_ticket_is_current,
    publish_coupled_command_if_motion_permitted,
    read_motion_permit,
)
from dexmani_real.utils.feedback import (
    FeedbackIssue,
    FeedbackIssueCode,
    diagnose_arm_feedback,
    diagnose_hand_feedback,
)
from dexmani_real.utils.limits import (
    canonicalize_policy_hand_endpoint_roundoff,
    limit_hand_target_delta,
)
from dexmani_real.utils.limits import (
    validate_hand_command_bounds as _validate_hand_bounds,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PublishResult:
    """Compact result of the realtime IPC publication boundary."""

    published: bool
    ticket: CoupledCommandTicket | None = None
    reason: str = ""


@dataclass(frozen=True)
class AcceptanceResult:
    """Result of an explicitly blocking worker/SDK acceptance wait."""

    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class PreparedCommand:
    """A physically checked command, or its preparation rejection."""

    candidate: ActionCandidate | None = None
    reason: str = ""
    gate_code: GateRejectCode | None = None
    feedback_issue: FeedbackIssue | None = None
    unavailable: bool = False
    fatal: bool = False
    hand_roundoff_canonicalized: bool = False

    @property
    def accepted(self) -> bool:
        return self.candidate is not None


@dataclass(frozen=True)
class _ArmFeedbackSnapshot:
    qpos: np.ndarray
    accepted_action_id: int
    accepted_monotonic_ns: int = 0


@dataclass(frozen=True)
class _HandFeedbackSnapshot:
    qpos: np.ndarray
    accepted_action_id: int
    accepted_monotonic_ns: int = 0


def validate_hand_command_bounds(
    hand_cmd: np.ndarray,
    operational_lower: np.ndarray,
    operational_upper: np.ndarray,
    mechanical_lower: np.ndarray,
    mechanical_upper: np.ndarray,
) -> np.ndarray:
    """Reject a target outside the operational or rated XHand envelope."""
    return _validate_hand_bounds(
        hand_cmd,
        operational_lower,
        operational_upper,
        mechanical_lower,
        mechanical_upper,
        hand_defaults.mechanical_qpos_min_rad,
        hand_defaults.mechanical_qpos_max_rad,
    )


def motion_rejection_reason(
    shared: Any,
    *,
    check_is_running: bool = True,
    required_safety_state: SafetyState | None = None,
) -> str:
    """Return why the runtime cannot accept motion, or an empty string."""
    if required_safety_state is not None and not isinstance(
        required_safety_state, SafetyState
    ):
        raise TypeError("required_safety_state must be a SafetyState or None")
    if bool(shared.estop_request.value):
        return PUBLISH_REASON_ESTOP
    if bool(shared.error_state.value):
        return PUBLISH_REASON_FAULT
    if check_is_running and not bool(shared.is_running.value):
        return PUBLISH_REASON_RUNTIME_STOPPED
    permit = read_motion_permit(shared)
    if not permit.allows_motion:
        return f"{PUBLISH_REASON_SAFETY_STATE}: {permit.state.name}"
    if required_safety_state is not None and permit.state is not required_safety_state:
        return (
            f"{PUBLISH_REASON_SAFETY_STATE}: expected {required_safety_state.name}, "
            f"got {permit.state.name}"
        )
    return ""


def _read_arm_feedback(
    shared: Any,
    *,
    max_age_s: float,
) -> tuple[_ArmFeedbackSnapshot | None, str, FeedbackIssue | None]:
    result = shared.arm_state_ring.read_latest()
    if result is None:
        return None, "arm feedback unavailable", None
    record = result[0][0]
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    issue = diagnose_arm_feedback(
        connected=bool(record["connected"]),
        error_code=int(record["error_code"]),
        state_valid=bool(record["state_valid"]),
        source_monotonic_ns=int(record["source_monotonic_ns"]),
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=max_age_s,
        qpos=qpos,
        qvel=np.asarray(record["qvel"], dtype=np.float64),
    )
    if issue is not None:
        return None, f"arm feedback is unhealthy: {issue.detail}", issue
    return (
        _ArmFeedbackSnapshot(
            qpos=qpos.copy(),
            accepted_action_id=int(record["last_cmd_seq"]),
            accepted_monotonic_ns=int(record["last_cmd_accepted_monotonic_ns"]),
        ),
        "",
        None,
    )


def read_hand_feedback(
    shared: Any,
    *,
    max_age_s: float,
) -> tuple[_HandFeedbackSnapshot | None, str, FeedbackIssue | None]:
    result = shared.hand_state_ring.read_latest()
    if result is None:
        return None, "hand feedback unavailable", None
    record = result[0][0]
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    issue = diagnose_hand_feedback(
        connected=bool(record["connected"]),
        state_valid=bool(record["state_valid"]),
        source_monotonic_ns=int(record["source_monotonic_ns"]),
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=max_age_s,
        qpos=qpos,
    )
    if issue is not None:
        return None, f"hand feedback is unhealthy: {issue.detail}", issue
    if qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(qpos)):
        return None, "hand measured qpos is malformed", None
    return (
        _HandFeedbackSnapshot(
            qpos=qpos.copy(),
            accepted_action_id=int(record["accepted_target_action_id"]),
            accepted_monotonic_ns=int(record["accepted_target_monotonic_ns"]),
        ),
        "",
        None,
    )


def build_action_candidate(
    shared: Any,
    arm_qpos: np.ndarray | None,
    hand_qpos: np.ndarray | None,
    *,
    run_generation: int | None = None,
    is_hold: bool = False,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    scheduled_target_monotonic_ns: int | None = None,
    now_ns: int | None = None,
    action_validity_s: float = 0.5,
    valid_until_monotonic_ns: int | None = None,
) -> ActionCandidate | None:
    """Build one immutable, structurally valid joint command."""
    if run_generation is not None and (
        isinstance(run_generation, (bool, np.bool_))
        or not isinstance(run_generation, (int, np.integer))
        or int(run_generation) < 0
    ):
        raise ValueError("run_generation must be a non-negative integer or None")
    if not np.isfinite(action_validity_s) or action_validity_s <= 0.0:
        raise ValueError("action_validity_s must be finite and positive")
    with shared.arm_command_seq.get_lock():
        action_id = int(shared.arm_command_seq.value) + 1
        shared.arm_command_seq.value = action_id
    now_ns = int(time.monotonic_ns() if now_ns is None else now_ns)
    if observation_anchor_monotonic_ns is not None:
        anchor_ns = int(observation_anchor_monotonic_ns)
        if anchor_ns <= 0 or anchor_ns > now_ns:
            return None
    target_ns = now_ns
    scheduled_ns = int(
        target_ns
        if scheduled_target_monotonic_ns is None
        else scheduled_target_monotonic_ns
    )
    if scheduled_ns <= 0:
        return None
    delivery_deadline_ns = now_ns + int(float(action_validity_s) * 1e9)
    if valid_until_monotonic_ns is not None:
        delivery_deadline_ns = min(delivery_deadline_ns, int(valid_until_monotonic_ns))
    if delivery_deadline_ns < target_ns:
        return None
    return ActionCandidate(
        observation_id=action_id if observation_id is None else int(observation_id),
        run_generation=(
            int(shared.run_generation.value)
            if run_generation is None
            else int(run_generation)
        ),
        action_id=action_id,
        created_monotonic_ns=now_ns,
        scheduled_target_monotonic_ns=scheduled_ns,
        target_monotonic_ns=target_ns,
        valid_until_monotonic_ns=delivery_deadline_ns,
        arm_qpos=arm_qpos,
        hand_qpos=hand_qpos,
        is_hold=is_hold,
    )


def prepare_command(
    shared: Any,
    candidate: ActionCandidate,
    *,
    gate: SafetyGate,
    arm_feedback_max_age_s: float,
    hand_feedback_max_age_s: float,
    arm_delta_reference_qpos: np.ndarray | None = None,
    hand_delta_reference_qpos: np.ndarray | None = None,
    hand_mechanical_lower_rad: np.ndarray | None = None,
    hand_mechanical_upper_rad: np.ndarray | None = None,
    hand_command_max_delta_rad_per_tick: float | np.ndarray | None = None,
    canonicalize_policy_hand_roundoff: bool = False,
) -> PreparedCommand:
    """Read valid feedback, check physical safety, and shape a learned hand target."""
    arm_feedback, reason, issue = _read_arm_feedback(
        shared, max_age_s=arm_feedback_max_age_s
    )
    if arm_feedback is None:
        unavailable = issue is None or issue.code is FeedbackIssueCode.STALE
        return PreparedCommand(
            reason=reason,
            feedback_issue=issue,
            unavailable=unavailable,
            fatal=not unavailable,
        )

    hand_feedback: _HandFeedbackSnapshot | None = None
    if candidate.hand_qpos is not None:
        hand_feedback, reason, issue = read_hand_feedback(
            shared, max_age_s=hand_feedback_max_age_s
        )
        if hand_feedback is None:
            unavailable = issue is None or issue.code is FeedbackIssueCode.STALE
            return PreparedCommand(
                reason=reason,
                feedback_issue=issue,
                unavailable=unavailable,
                fatal=not unavailable,
            )

    mechanical_lower = np.asarray(
        (
            hand_defaults.mechanical_qpos_min_rad
            if hand_mechanical_lower_rad is None
            else hand_mechanical_lower_rad
        ),
        dtype=np.float64,
    )
    mechanical_upper = np.asarray(
        (
            hand_defaults.mechanical_qpos_max_rad
            if hand_mechanical_upper_rad is None
            else hand_mechanical_upper_rad
        ),
        dtype=np.float64,
    )
    hand_roundoff_canonicalized = False
    if candidate.hand_qpos is not None and canonicalize_policy_hand_roundoff:
        try:
            hand_qpos, hand_roundoff_canonicalized = (
                canonicalize_policy_hand_endpoint_roundoff(
                    candidate.hand_qpos,
                    gate.hand_low,
                    gate.hand_high,
                    mechanical_lower,
                    mechanical_upper,
                    hand_defaults.mechanical_qpos_min_rad,
                    hand_defaults.mechanical_qpos_max_rad,
                )
            )
        except ValueError as exc:
            return PreparedCommand(reason=str(exc))
        if hand_roundoff_canonicalized:
            candidate = replace(candidate, hand_qpos=hand_qpos)

    gate_result = gate.validate(
        candidate,
        current_arm_qpos=arm_feedback.qpos,
        current_hand_qpos=(hand_feedback.qpos if hand_feedback is not None else None),
        arm_delta_reference_qpos=arm_delta_reference_qpos,
        hand_delta_reference_qpos=hand_delta_reference_qpos,
    )
    if not gate_result.accepted:
        return PreparedCommand(
            reason=gate_result.reason,
            gate_code=gate_result.code,
            fatal=gate_result.code
            in {
                GateRejectCode.COLLISION_CHECK_FAILED,
                GateRejectCode.WORKSPACE_CHECK_FAILED,
            },
            hand_roundoff_canonicalized=hand_roundoff_canonicalized,
        )

    if (
        candidate.hand_qpos is not None
        and hand_command_max_delta_rad_per_tick is not None
    ):
        assert hand_feedback is not None
        shaped_hand_qpos = limit_hand_target_delta(
            candidate.hand_qpos,
            hand_feedback.qpos,
            hand_command_max_delta_rad_per_tick,
        )
        if not np.array_equal(shaped_hand_qpos, candidate.hand_qpos):
            candidate = replace(candidate, hand_qpos=shaped_hand_qpos)
            shaped_result = gate.validate_shaped_hand(
                candidate,
                current_arm_qpos=arm_feedback.qpos,
                current_hand_qpos=hand_feedback.qpos,
                hand_delta_reference_qpos=hand_delta_reference_qpos,
            )
            if not shaped_result.accepted:
                return PreparedCommand(
                    reason=shaped_result.reason,
                    gate_code=shaped_result.code,
                    fatal=shaped_result.code is GateRejectCode.COLLISION_CHECK_FAILED,
                    hand_roundoff_canonicalized=hand_roundoff_canonicalized,
                )

    if candidate.hand_qpos is not None:
        try:
            validate_hand_command_bounds(
                candidate.hand_qpos,
                gate.hand_low,
                gate.hand_high,
                mechanical_lower,
                mechanical_upper,
            )
        except ValueError as exc:
            return PreparedCommand(
                reason=str(exc),
                hand_roundoff_canonicalized=hand_roundoff_canonicalized,
            )
    return PreparedCommand(
        candidate=candidate,
        hand_roundoff_canonicalized=hand_roundoff_canonicalized,
    )


def prepare_joint_command(
    shared: Any,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None = None,
    *,
    gate: SafetyGate,
    is_hold: bool = False,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    action_validity_s: float = 0.5,
    arm_delta_reference_qpos: np.ndarray | None = None,
    hand_delta_reference_qpos: np.ndarray | None = None,
    hand_mechanical_lower_rad: np.ndarray | None = None,
    hand_mechanical_upper_rad: np.ndarray | None = None,
    arm_feedback_max_age_s: float,
    hand_feedback_max_age_s: float,
) -> PreparedCommand:
    """Build and physically validate raw joint targets without publishing."""
    try:
        candidate = build_action_candidate(
            shared,
            arm_qpos,
            hand_qpos,
            is_hold=is_hold,
            observation_id=observation_id,
            observation_anchor_monotonic_ns=observation_anchor_monotonic_ns,
            action_validity_s=action_validity_s,
        )
    except (TypeError, ValueError) as exc:
        return PreparedCommand(reason=str(exc), fatal=True)
    if candidate is None:
        return PreparedCommand(
            reason="invalid observation anchor or closed window", fatal=True
        )
    return prepare_command(
        shared,
        candidate,
        gate=gate,
        arm_feedback_max_age_s=arm_feedback_max_age_s,
        hand_feedback_max_age_s=hand_feedback_max_age_s,
        arm_delta_reference_qpos=arm_delta_reference_qpos,
        hand_delta_reference_qpos=hand_delta_reference_qpos,
        hand_mechanical_lower_rad=hand_mechanical_lower_rad,
        hand_mechanical_upper_rad=hand_mechanical_upper_rad,
    )


def _make_coupled_command(candidate: ActionCandidate) -> np.ndarray:
    frame = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
    frame["run_generation"][0] = candidate.run_generation
    frame["observation_id"][0] = candidate.observation_id
    frame["action_id"][0] = candidate.action_id
    frame["created_monotonic_ns"][0] = candidate.created_monotonic_ns
    frame["scheduled_target_monotonic_ns"][0] = candidate.scheduled_target_monotonic_ns
    frame["target_monotonic_ns"][0] = candidate.target_monotonic_ns
    frame["valid_until_monotonic_ns"][0] = candidate.valid_until_monotonic_ns
    frame["is_hold"][0] = int(candidate.is_hold)
    if candidate.arm_qpos is not None:
        frame["arm_present"][0] = 1
        frame["arm_qpos"][0] = candidate.arm_qpos
    if candidate.hand_qpos is not None:
        frame["hand_present"][0] = 1
        frame["hand_qpos"][0] = candidate.hand_qpos
    return frame


def command_publishability_reason(
    shared: Any,
    candidate: ActionCandidate,
    *,
    check_is_running: bool = True,
    required_safety_state: SafetyState | None = None,
    minimum_delivery_window_s: float = 0.0,
) -> str:
    """Check runtime, generation, and validity-window publication invariants."""
    minimum_delivery_window_ns = _minimum_delivery_window_ns(minimum_delivery_window_s)
    return _command_publishability_reason(
        shared,
        candidate,
        check_is_running=check_is_running,
        required_safety_state=required_safety_state,
        minimum_delivery_window_ns=minimum_delivery_window_ns,
    )


def _minimum_delivery_window_ns(minimum_delivery_window_s: float) -> int:
    """Convert the public delivery-window requirement to monotonic nanoseconds."""
    if not np.isfinite(minimum_delivery_window_s) or minimum_delivery_window_s < 0.0:
        raise ValueError("minimum_delivery_window_s must be finite and non-negative")
    return int(minimum_delivery_window_s * 1e9)


def _command_publishability_reason(
    shared: Any,
    candidate: ActionCandidate,
    *,
    check_is_running: bool,
    required_safety_state: SafetyState | None,
    minimum_delivery_window_ns: int,
) -> str:
    """Check the pre-publication state using a resolved delivery window."""
    reason = motion_rejection_reason(
        shared,
        check_is_running=check_is_running,
        required_safety_state=required_safety_state,
    )
    if reason:
        return reason
    permit = read_motion_permit(shared)
    if int(candidate.run_generation) != permit.run_generation:
        return PUBLISH_REASON_GENERATION
    if (
        candidate.valid_until_monotonic_ns - time.monotonic_ns()
        <= minimum_delivery_window_ns
    ):
        return PUBLISH_REASON_EXPIRED
    return ""


def publish_command(
    shared: Any,
    candidate: ActionCandidate,
    *,
    check_is_running: bool = True,
    required_safety_state: SafetyState | None = None,
    minimum_delivery_window_s: float = 0.0,
) -> PublishResult:
    """Publish one checked command without waiting for worker acknowledgement."""
    minimum_delivery_window_ns = _minimum_delivery_window_ns(minimum_delivery_window_s)
    reason = _command_publishability_reason(
        shared,
        candidate,
        check_is_running=check_is_running,
        required_safety_state=required_safety_state,
        minimum_delivery_window_ns=minimum_delivery_window_ns,
    )
    if reason:
        return PublishResult(False, reason=reason)
    ticket, rejection_reason = publish_coupled_command_if_motion_permitted(
        shared,
        expected_run_generation=int(candidate.run_generation),
        frame=_make_coupled_command(candidate),
        required_state=required_safety_state,
        minimum_delivery_window_ns=minimum_delivery_window_ns,
    )
    if ticket is None:
        return PublishResult(False, reason=rejection_reason)
    return PublishResult(True, ticket=ticket)


def wait_command_accepted(
    shared: Any,
    *,
    ticket: CoupledCommandTicket,
    action_id: int,
    wait_for_arm: bool,
    wait_for_hand: bool,
    timeout_s: float,
    arm_feedback_max_age_s: float,
    hand_feedback_max_age_s: float,
    check_is_running: bool = True,
    abort_requested: Callable[[], bool] | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> AcceptanceResult:
    """Block until the requested workers report SDK acceptance of one command."""
    if not wait_for_arm and not wait_for_hand:
        raise ValueError("acceptance wait requires at least one worker")
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("acceptance timeout must be finite and positive")
    if action_id < 0 or ticket.ring_sequence <= 0:
        raise ValueError("acceptance identity must be non-negative and published")
    deadline_s = time.monotonic() + timeout_s
    while time.monotonic() < deadline_s:
        if abort_requested is not None and abort_requested():
            cancel_coupled_command_if_current(shared, ticket=ticket)
            return AcceptanceResult(False, "acceptance aborted")
        reason = motion_rejection_reason(shared, check_is_running=check_is_running)
        if reason:
            cancel_coupled_command_if_current(shared, ticket=ticket)
            return AcceptanceResult(False, reason)
        if heartbeat is not None:
            heartbeat()

        arm_accepted = not wait_for_arm
        if wait_for_arm:
            arm_feedback, reason, _ = _read_arm_feedback(
                shared, max_age_s=arm_feedback_max_age_s
            )
            if arm_feedback is None:
                cancel_coupled_command_if_current(shared, ticket=ticket)
                return AcceptanceResult(False, reason)
            if arm_feedback.accepted_action_id > action_id:
                return AcceptanceResult(False, "arm acceptance was superseded")
            arm_accepted = arm_feedback.accepted_action_id == action_id
            if arm_accepted and arm_feedback.accepted_monotonic_ns <= 0:
                cancel_coupled_command_if_current(shared, ticket=ticket)
                return AcceptanceResult(False, "arm acceptance timestamp is missing")

        hand_accepted = not wait_for_hand
        if wait_for_hand:
            hand_feedback, reason, _ = read_hand_feedback(
                shared, max_age_s=hand_feedback_max_age_s
            )
            if hand_feedback is None:
                cancel_coupled_command_if_current(shared, ticket=ticket)
                return AcceptanceResult(False, reason)
            if hand_feedback.accepted_action_id > action_id:
                return AcceptanceResult(False, "hand acceptance was superseded")
            hand_accepted = hand_feedback.accepted_action_id == action_id
            if hand_accepted and hand_feedback.accepted_monotonic_ns <= 0:
                cancel_coupled_command_if_current(shared, ticket=ticket)
                return AcceptanceResult(False, "hand acceptance timestamp is missing")

        if arm_accepted and hand_accepted:
            return AcceptanceResult(True)
        if not coupled_command_ticket_is_current(shared, ticket=ticket):
            return AcceptanceResult(
                False, "command ownership was revoked or superseded"
            )
        time.sleep(0.005)

    cancel_coupled_command_if_current(shared, ticket=ticket)
    return AcceptanceResult(False, f"command was not accepted within {timeout_s:.3f}s")

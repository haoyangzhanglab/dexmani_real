"""Controller-side validation, serialization, publication, and acknowledgement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.ipc.schema import (
    ARM_JOINT_SHAPE,
    COUPLED_COMMAND_DTYPE,
    HAND_JOINT_SHAPE,
)
from dexmani_real.runtime.safety import (
    CoupledCommandTicket,
    cancel_coupled_command_if_current,
    coupled_command_ticket_is_current,
    publish_coupled_command_if_motion_permitted,
    read_motion_permit,
)
from dexmani_real.utils.feedback import validate_arm_feedback, validate_hand_feedback
from dexmani_real.utils.limits import (
    validate_hand_command_bounds as _validate_hand_bounds,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


def validate_hand_command_bounds(
    hand_cmd: np.ndarray,
    operational_lower: np.ndarray,
    operational_upper: np.ndarray,
    mechanical_lower: np.ndarray,
    mechanical_upper: np.ndarray,
) -> np.ndarray:
    """Apply the canonical rated XHand envelope at the publication boundary."""
    return _validate_hand_bounds(
        hand_cmd,
        operational_lower,
        operational_upper,
        mechanical_lower,
        mechanical_upper,
        hand_defaults.mechanical_qpos_min_rad,
        hand_defaults.mechanical_qpos_max_rad,
    )


class CommandPublishStatus(str, Enum):
    """Result of the controller-side candidate/publication boundary."""

    PUBLISHED = "published"
    APPLIED = "applied"
    NO_SAFETY_GATE = "no safety gate"
    INVALID_CANDIDATE = "invalid candidate"
    INVALID_OBSERVATION_ANCHOR = "invalid observation anchor"
    RUNTIME_STOPPED = "runtime stopped"
    ESTOP_REQUESTED = "e-stop requested"
    STICKY_FAULT = "sticky fault"
    SAFETY_STATE_GATED = "safety state gated"
    RUN_GENERATION_GATED = "run generation gated"
    ARM_FEEDBACK_UNAVAILABLE = "arm feedback unavailable"
    ARM_FEEDBACK_UNHEALTHY = "arm feedback unhealthy"
    HAND_FEEDBACK_UNAVAILABLE = "hand feedback unavailable"
    HAND_FEEDBACK_UNHEALTHY = "hand feedback unhealthy"
    HAND_PREFLIGHT_REJECTED = "hand preflight rejected"
    GATE_REJECTED = "safety gate rejected"
    TEMPORAL_WINDOW_CLOSED = "temporal window closed"
    ACK_SUPERSEDED = "acknowledgement superseded"
    ACK_TIMEOUT = "acknowledgement timeout"


@dataclass(frozen=True)
class CommandPublishResult:
    """Typed publication outcome; callers retain ownership of disposition."""

    status: CommandPublishStatus
    candidate: ActionCandidate | None = None
    detail: str = ""
    gate_code: GateRejectCode | None = None
    ticket: CoupledCommandTicket | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in (
            CommandPublishStatus.PUBLISHED,
            CommandPublishStatus.APPLIED,
        )

    @property
    def runtime_gated(self) -> bool:
        return self.status in (
            CommandPublishStatus.RUNTIME_STOPPED,
            CommandPublishStatus.ESTOP_REQUESTED,
            CommandPublishStatus.STICKY_FAULT,
            CommandPublishStatus.SAFETY_STATE_GATED,
            CommandPublishStatus.RUN_GENERATION_GATED,
        )

    @property
    def reason(self) -> str:
        return self.detail or self.status.value


def check_runtime_gate(
    shared: Any,
    *,
    check_is_running: bool = True,
) -> CommandPublishResult | None:
    """Reject publication outside an active, non-faulted runtime.

    This controller-side check reduces stale queue/ring traffic.  Workers still
    re-check the same lifecycle state immediately before their SDK boundary.
    """
    if bool(shared.estop_request.value):
        return CommandPublishResult(CommandPublishStatus.ESTOP_REQUESTED)
    if bool(shared.error_state.value):
        return CommandPublishResult(CommandPublishStatus.STICKY_FAULT)
    if check_is_running and not bool(shared.is_running.value):
        return CommandPublishResult(CommandPublishStatus.RUNTIME_STOPPED)
    permit = read_motion_permit(shared)
    if not permit.allows_motion:
        return CommandPublishResult(
            CommandPublishStatus.SAFETY_STATE_GATED,
            detail=f"safety state {permit.state.name} does not accept motion commands",
        )
    return None


@dataclass(frozen=True)
class _ArmFeedbackSnapshot:
    qpos: np.ndarray
    last_cmd_seq: int


@dataclass(frozen=True)
class _HandFeedbackSnapshot:
    qpos: np.ndarray
    accepted_target_action_id: int


def _arm_feedback_snapshot(
    shared: Any,
    candidate: ActionCandidate | None,
    *,
    arm_feedback_max_age_s: float,
) -> tuple[_ArmFeedbackSnapshot | None, CommandPublishResult | None]:
    """Read the arm fields required by publication and acknowledgement.

    The frame must be connected, controller-error-free, valid, fresh, and
    finite before its joint positions can reach the safety gate or ACK path.
    """
    result = shared.arm_state_ring.read_latest()
    if result is None:
        return None, CommandPublishResult(
            CommandPublishStatus.ARM_FEEDBACK_UNAVAILABLE,
            candidate=candidate,
        )
    record = result[0][0]
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    issue = validate_arm_feedback(
        connected=bool(record["connected"]),
        error_code=int(record["error_code"]),
        state_valid=bool(record["state_valid"]),
        source_monotonic_ns=int(record["source_monotonic_ns"]),
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=arm_feedback_max_age_s,
        qpos=qpos,
        qvel=np.asarray(record["qvel"], dtype=np.float64),
    )
    if issue is not None:
        return None, CommandPublishResult(
            CommandPublishStatus.ARM_FEEDBACK_UNHEALTHY,
            candidate=candidate,
            detail=f"arm feedback is unhealthy: {issue}",
        )
    return _ArmFeedbackSnapshot(qpos.copy(), int(record["last_cmd_seq"])), None


def read_hand_feedback(
    shared: Any,
    candidate: ActionCandidate | None,
    *,
    hand_feedback_max_age_s: float,
) -> tuple[_HandFeedbackSnapshot | None, CommandPublishResult | None]:
    """Read one fully healthy hand command/feedback snapshot fail-closed.

    Delegates connected/state-valid checks, source-timestamp existence, future
    timestamp, and ``max_age`` freshness to :func:`validate_hand_feedback`.
    """
    if not np.isfinite(hand_feedback_max_age_s) or hand_feedback_max_age_s <= 0.0:
        raise ValueError("hand_feedback_max_age_s must be finite and positive")
    result = shared.hand_state_ring.read_latest()
    if result is None:
        return None, CommandPublishResult(
            CommandPublishStatus.HAND_FEEDBACK_UNAVAILABLE,
            candidate=candidate,
        )
    record = result[0][0]
    issue = validate_hand_feedback(
        connected=bool(record["connected"]),
        state_valid=bool(record["state_valid"]),
        source_monotonic_ns=int(record["source_monotonic_ns"]),
        now_monotonic_ns=time.monotonic_ns(),
        max_age_s=hand_feedback_max_age_s,
        qpos=np.asarray(record["qpos"], dtype=np.float64),
    )
    if issue is not None:
        return None, CommandPublishResult(
            CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY,
            candidate=candidate,
            detail=f"hand feedback is unhealthy: {issue}",
        )
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    if qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(qpos)):
        return None, CommandPublishResult(
            CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY,
            candidate=candidate,
            detail="hand measured qpos is malformed",
        )
    return (
        _HandFeedbackSnapshot(
            qpos.copy(),
            int(record["accepted_target_action_id"]),
        ),
        None,
    )


def _make_coupled_command(candidate: ActionCandidate) -> np.ndarray:
    """Serialize one action into the coherent arm/hand IPC record."""
    frame = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
    frame["run_generation"][0] = candidate.run_generation
    frame["observation_id"][0] = candidate.observation_id
    frame["action_id"][0] = candidate.action_id
    frame["created_monotonic_ns"][0] = candidate.created_monotonic_ns
    frame["scheduled_target_monotonic_ns"][0] = candidate.scheduled_target_monotonic_ns
    frame["target_monotonic_ns"][0] = candidate.target_monotonic_ns
    frame["valid_until_monotonic_ns"][0] = candidate.valid_until_monotonic_ns
    frame["is_hold"][0] = int(bool(candidate.is_hold))
    if candidate.arm_qpos is not None:
        frame["arm_present"][0] = 1
        frame["arm_qpos"][0] = candidate.arm_qpos
    if candidate.hand_qpos is not None:
        frame["hand_present"][0] = 1
        frame["hand_qpos"][0] = candidate.hand_qpos
    return frame


def send_command(
    shared: Any,
    candidate: ActionCandidate,
    *,
    check_is_running: bool = True,
) -> CommandPublishResult:
    """Publish one coherent arm/hand record to the actuator IPC ring.

    Returns a typed transport outcome. Callers decide whether a rejection
    means hold, drop, command quiescence, run abort, or global fault.
    """
    runtime_rejection = check_runtime_gate(
        shared,
        check_is_running=check_is_running,
    )
    if runtime_rejection is not None:
        return CommandPublishResult(
            runtime_rejection.status,
            candidate=candidate,
            detail=runtime_rejection.detail,
        )
    permit = read_motion_permit(shared)
    if int(candidate.run_generation) != permit.run_generation:
        return CommandPublishResult(
            CommandPublishStatus.RUN_GENERATION_GATED,
            candidate=candidate,
            detail="candidate generation no longer owns the motion permit",
        )

    now_ns = time.monotonic_ns()
    if candidate.valid_until_monotonic_ns <= now_ns:
        logger.error(
            "send_command: action_id=%d temporal window closed", candidate.action_id
        )
        return CommandPublishResult(
            CommandPublishStatus.TEMPORAL_WINDOW_CLOSED,
            candidate=candidate,
        )

    frame = _make_coupled_command(candidate)
    ticket = publish_coupled_command_if_motion_permitted(
        shared,
        expected_run_generation=int(candidate.run_generation),
        frame=frame,
    )
    if ticket is None:
        return CommandPublishResult(
            CommandPublishStatus.RUN_GENERATION_GATED,
            candidate=candidate,
            detail="motion permit was revoked before IPC publication",
        )

    return CommandPublishResult(
        CommandPublishStatus.PUBLISHED,
        candidate=candidate,
        ticket=ticket,
    )


def build_action_candidate(
    shared: Any,
    arm_qpos: np.ndarray | None,
    hand_qpos: np.ndarray | None,
    *,
    is_hold: bool = False,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    scheduled_target_monotonic_ns: int | None = None,
    now_ns: int | None = None,
    action_validity_s: float = 0.5,
    valid_until_monotonic_ns: int | None = None,
) -> ActionCandidate | None:
    """Build an ``ActionCandidate`` from raw joint targets.

    Allocates a fresh monotonic ``action_id`` from ``shared.arm_command_seq``
    and stamps an immediate delivery target. ``scheduled_target_monotonic_ns``
    retains the source plan's control-grid time even when the coordinator is
    publishing a due endpoint later. The delivery validity is the earlier of
    ``action_validity_s`` and an optional immutable caller deadline.
    """
    with shared.arm_command_seq.get_lock():
        action_id = int(shared.arm_command_seq.value) + 1
        shared.arm_command_seq.value = action_id
    now_ns = int(time.monotonic_ns() if now_ns is None else now_ns)
    if observation_anchor_monotonic_ns is not None:
        anchor_ns = int(observation_anchor_monotonic_ns)
        if anchor_ns <= 0 or anchor_ns > now_ns:
            logger.warning(
                "build_action_candidate: action_id=%d rejected: invalid observation anchor",
                action_id,
            )
            return None
    target_ns = now_ns
    scheduled_ns = int(
        target_ns
        if scheduled_target_monotonic_ns is None
        else scheduled_target_monotonic_ns
    )
    if scheduled_ns <= 0:
        logger.warning(
            "build_action_candidate: action_id=%d rejected: invalid scheduled target",
            action_id,
        )
        return None
    delivery_deadline_ns = now_ns + int(float(action_validity_s) * 1e9)
    if valid_until_monotonic_ns is not None:
        delivery_deadline_ns = min(delivery_deadline_ns, int(valid_until_monotonic_ns))
    if delivery_deadline_ns < target_ns:
        logger.warning(
            "build_action_candidate: action_id=%d rejected: delivery window is closed",
            action_id,
        )
        return None
    return ActionCandidate(
        observation_id=action_id if observation_id is None else int(observation_id),
        run_generation=int(shared.run_generation.value),
        action_id=action_id,
        created_monotonic_ns=now_ns,
        scheduled_target_monotonic_ns=scheduled_ns,
        target_monotonic_ns=target_ns,
        valid_until_monotonic_ns=delivery_deadline_ns,
        arm_qpos=(None if arm_qpos is None else np.asarray(arm_qpos, dtype=np.float64)),
        hand_qpos=(
            None if hand_qpos is None else np.asarray(hand_qpos, dtype=np.float64)
        ),
        is_hold=is_hold,
    )


def validate_and_send_candidate(
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
) -> CommandPublishResult:
    """Validate a pre-built candidate through the gate and publish it.

    Checks runtime and actuator feedback, runs :meth:`SafetyGate.validate`,
    preflights a coupled hand target, and publishes via :func:`send_command`.
    This is the publication tail shared by VR teleop, keyboard/replay, and the
    learned-policy coordinator.

    Returns:
        A typed result that distinguishes policy-semantic gate rejection from
        runtime, feedback, and transport failures.
    """
    action_id = int(candidate.action_id)
    runtime_rejection = check_runtime_gate(shared)
    if runtime_rejection is not None:
        return CommandPublishResult(
            runtime_rejection.status,
            candidate=candidate,
            detail=runtime_rejection.detail,
        )

    arm_feedback, feedback_rejection = _arm_feedback_snapshot(
        shared,
        candidate,
        arm_feedback_max_age_s=arm_feedback_max_age_s,
    )
    if feedback_rejection is not None:
        logger.warning(
            "validate_and_send_candidate: action_id=%d rejected: %s",
            action_id,
            feedback_rejection.reason,
        )
        return feedback_rejection
    assert arm_feedback is not None

    # Read hand feedback before the gate so delta/collision checks see the
    # current hand state; the same snapshot then serves the coupled-hand preflight.
    hand_feedback: _HandFeedbackSnapshot | None = None
    if candidate.hand_qpos is not None:
        hand_feedback, feedback_rejection = read_hand_feedback(
            shared, candidate, hand_feedback_max_age_s=hand_feedback_max_age_s
        )
        if feedback_rejection is not None:
            logger.warning(
                "validate_and_send_candidate: action_id=%d rejected: %s",
                action_id,
                feedback_rejection.reason,
            )
            return feedback_rejection
        assert hand_feedback is not None

    permit = read_motion_permit(shared)
    if int(candidate.run_generation) != permit.run_generation:
        return CommandPublishResult(
            CommandPublishStatus.RUN_GENERATION_GATED,
            candidate=candidate,
            detail="candidate generation no longer owns the motion permit",
        )
    gate_result = gate.validate(
        candidate,
        current_arm_qpos=arm_feedback.qpos,
        current_hand_qpos=(hand_feedback.qpos if hand_feedback is not None else None),
        arm_delta_reference_qpos=arm_delta_reference_qpos,
        hand_delta_reference_qpos=hand_delta_reference_qpos,
        run_generation=permit.run_generation,
    )
    if not gate_result.accepted:
        reason = gate_result.reason or "unspecified"
        logger.warning(
            "validate_and_send_candidate: action_id=%d rejected by safety gate: %s",
            action_id,
            reason,
        )
        return CommandPublishResult(
            CommandPublishStatus.GATE_REJECTED,
            candidate=candidate,
            detail=reason,
            gate_code=gate_result.code,
        )

    if candidate.hand_qpos is not None:
        assert hand_feedback is not None
        mechanical_lower = (
            np.asarray(hand_defaults.mechanical_qpos_min_rad, dtype=np.float64)
            if hand_mechanical_lower_rad is None
            else np.asarray(hand_mechanical_lower_rad, dtype=np.float64)
        )
        mechanical_upper = (
            np.asarray(hand_defaults.mechanical_qpos_max_rad, dtype=np.float64)
            if hand_mechanical_upper_rad is None
            else np.asarray(hand_mechanical_upper_rad, dtype=np.float64)
        )
        try:
            validate_hand_command_bounds(
                candidate.hand_qpos,
                gate.hand_low,
                gate.hand_high,
                mechanical_lower,
                mechanical_upper,
            )
        except ValueError as exc:
            logger.warning(
                "validate_and_send_candidate: action_id=%d rejected by hand preflight: %s",
                action_id,
                exc,
            )
            return CommandPublishResult(
                CommandPublishStatus.HAND_PREFLIGHT_REJECTED,
                candidate=candidate,
                detail=str(exc),
            )

    return send_command(shared, candidate)


def publish_joint_targets(
    shared: Any,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None = None,
    *,
    is_hold: bool = False,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    safety_gate: SafetyGate | None = None,
    wait_applied: bool = False,
    apply_timeout_s: float = 0.5,
    action_validity_s: float = 0.5,
    arm_delta_reference_qpos: np.ndarray | None = None,
    hand_delta_reference_qpos: np.ndarray | None = None,
    hand_mechanical_lower_rad: np.ndarray | None = None,
    hand_mechanical_upper_rad: np.ndarray | None = None,
    arm_feedback_max_age_s: float,
    hand_feedback_max_age_s: float,
) -> CommandPublishResult:
    """Validate a joint-space target through the gate and publish it.

    This is a convenience wrapper used by keyboard teleop, calibration, and replay — it
    builds an ``ActionCandidate`` from raw joint arrays, runs
    the full validation pipeline, and calls :func:`send_command`.  When the
    candidate carries a hand target, a coupled-hand preflight
    (:func:`validate_hand_command_bounds`) additionally rejects-whole the rated
    mechanical envelope *before* the arm endpoint is enqueued, so a rejected
    hand command cannot desync the arm from the hand.

    ``hand_mechanical_lower_rad`` / ``hand_mechanical_upper_rad`` default to the
    rated device envelope. ``action_validity_s`` bounds how long a worker may
    continue a measured-state-bounded hand ramp toward this endpoint.

    Returns a typed validation/publication result. On success, ``candidate``
    contains the immutable target that was published (and, when
    ``wait_applied`` is true, acknowledged by arm feedback — plus hand
    feedback when the candidate carries a hand target).
    """
    if safety_gate is None:
        logger.error("publish_joint_targets: no safety gate configured")
        return CommandPublishResult(CommandPublishStatus.NO_SAFETY_GATE)
    gate = safety_gate

    runtime_rejection = check_runtime_gate(shared)
    if runtime_rejection is not None:
        return runtime_rejection
    if not np.isfinite(action_validity_s) or action_validity_s <= 0.0:
        raise ValueError("action_validity_s must be finite and positive")

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
        logger.warning("publish_joint_targets: invalid candidate: %s", exc)
        return CommandPublishResult(
            CommandPublishStatus.INVALID_CANDIDATE,
            detail=str(exc),
        )
    if candidate is None:
        return CommandPublishResult(CommandPublishStatus.INVALID_OBSERVATION_ANCHOR)
    action_id = int(candidate.action_id)

    publish_result = validate_and_send_candidate(
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
    if not publish_result.succeeded:
        return publish_result
    published = publish_result.candidate
    if published is None:
        return CommandPublishResult(
            CommandPublishStatus.INVALID_CANDIDATE,
            detail="successful publication omitted its candidate",
        )
    if wait_applied:
        if not (np.isfinite(apply_timeout_s) and apply_timeout_s > 0):
            raise ValueError("apply_timeout_s must be finite and positive")
        ticket = publish_result.ticket
        if ticket is None:
            return CommandPublishResult(
                CommandPublishStatus.INVALID_CANDIDATE,
                candidate=published,
                detail="successful publication omitted its command ticket",
            )
        deadline_s = time.monotonic() + float(apply_timeout_s)
        # Coupled arm/hand candidates use one action_id and require both acknowledgements.
        with_hand = published.hand_qpos is not None
        while time.monotonic() < deadline_s:
            runtime_rejection = check_runtime_gate(shared)
            if runtime_rejection is not None:
                cancel_coupled_command_if_current(shared, ticket=ticket)
                return CommandPublishResult(
                    runtime_rejection.status,
                    candidate=published,
                    detail=runtime_rejection.detail,
                )
            arm_feedback, feedback_rejection = _arm_feedback_snapshot(
                shared,
                published,
                arm_feedback_max_age_s=arm_feedback_max_age_s,
            )
            if feedback_rejection is not None:
                cancel_coupled_command_if_current(shared, ticket=ticket)
                return feedback_rejection
            assert arm_feedback is not None
            if arm_feedback.last_cmd_seq > action_id:
                logger.warning(
                    "publish_joint_targets: action_id=%d was superseded by arm action_id=%d",
                    action_id,
                    arm_feedback.last_cmd_seq,
                )
                return CommandPublishResult(
                    CommandPublishStatus.ACK_SUPERSEDED,
                    candidate=published,
                )
            arm_ok = arm_feedback.last_cmd_seq == action_id
            if not with_hand:
                if arm_ok:
                    return CommandPublishResult(
                        CommandPublishStatus.APPLIED,
                        candidate=published,
                    )
            else:
                hand_feedback, feedback_rejection = read_hand_feedback(
                    shared, published, hand_feedback_max_age_s=hand_feedback_max_age_s
                )
                if feedback_rejection is not None:
                    cancel_coupled_command_if_current(shared, ticket=ticket)
                    return feedback_rejection
                assert hand_feedback is not None
                hand_action_id = hand_feedback.accepted_target_action_id
                if hand_action_id > action_id:
                    logger.warning(
                        "publish_joint_targets: action_id=%d was superseded by hand action_id=%d",
                        action_id,
                        hand_action_id,
                    )
                    return CommandPublishResult(
                        CommandPublishStatus.ACK_SUPERSEDED,
                        candidate=published,
                    )
                if arm_ok and hand_action_id == action_id:
                    return CommandPublishResult(
                        CommandPublishStatus.APPLIED,
                        candidate=published,
                    )
            if not coupled_command_ticket_is_current(shared, ticket=ticket):
                logger.warning(
                    "publish_joint_targets: action_id=%d lost command ownership before acknowledgement",
                    action_id,
                )
                return CommandPublishResult(
                    CommandPublishStatus.ACK_SUPERSEDED,
                    candidate=published,
                    detail="published command was revoked or superseded before acknowledgement",
                )
            time.sleep(0.005)
        logger.warning(
            "publish_joint_targets: action_id=%d was not acknowledged within %.3fs",
            action_id,
            apply_timeout_s,
        )
        cancel_coupled_command_if_current(shared, ticket=ticket)
        return CommandPublishResult(
            CommandPublishStatus.ACK_TIMEOUT,
            candidate=published,
        )
    return publish_result

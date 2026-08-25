"""Controller-side validation, serialization, publication, and acknowledgement."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.config.defaults import policy as policy_defaults
from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.ipc.schema import (
    ARM_COMMAND_DTYPE,
    ARM_JOINT_SHAPE,
    HAND_COMMAND_DTYPE,
    HAND_JOINT_SHAPE,
)
from dexmani_real.runtime.safety import SafetyState
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
    ARM_FEEDBACK_UNAVAILABLE = "arm feedback unavailable"
    ARM_FEEDBACK_UNHEALTHY = "arm feedback unhealthy"
    HAND_FEEDBACK_UNAVAILABLE = "hand feedback unavailable"
    HAND_FEEDBACK_UNHEALTHY = "hand feedback unhealthy"
    HAND_PREFLIGHT_REJECTED = "hand preflight rejected"
    GATE_REJECTED = "safety gate rejected"
    TEMPORAL_WINDOW_CLOSED = "temporal window closed"
    PREPARE_TIMEOUT = "prepare timeout"
    ACK_SUPERSEDED = "acknowledgement superseded"
    ACK_TIMEOUT = "acknowledgement timeout"


@dataclass(frozen=True)
class CommandPublishResult:
    """Typed publication outcome; callers retain ownership of disposition."""

    status: CommandPublishStatus
    candidate: ActionCandidate | None = None
    detail: str = ""
    gate_code: GateRejectCode | None = None

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
    state_value = int(shared.safety_state.value)
    if state_value not in (int(SafetyState.ARMED), int(SafetyState.RUNNING)):
        try:
            state_name = SafetyState(state_value).name
        except ValueError:
            state_name = f"UNKNOWN({state_value})"
        return CommandPublishResult(
            CommandPublishStatus.SAFETY_STATE_GATED,
            detail=f"safety state {state_name} does not accept motion commands",
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


def _make_arm_command(
    candidate: ActionCandidate, now_monotonic_ns: int, target_monotonic_ns: int
) -> np.ndarray:
    """Serialize an ActionCandidate into an ARM_COMMAND_DTYPE record.

    Carries the same identity/timing prefix as ``_make_hand_command`` so a
    STOP/FAULT generation bump invalidates arm and hand at the same boundary.
    """
    if candidate.arm_qpos is None:
        raise ValueError("candidate has no arm command")
    frame = np.zeros(1, dtype=ARM_COMMAND_DTYPE)
    frame["run_generation"][0] = candidate.run_generation
    frame["observation_id"][0] = candidate.observation_id
    frame["action_id"][0] = candidate.action_id
    frame["created_monotonic_ns"][0] = now_monotonic_ns
    frame["target_monotonic_ns"][0] = target_monotonic_ns
    frame["valid_until_monotonic_ns"][0] = candidate.valid_until_monotonic_ns
    frame["is_hold"][0] = int(bool(candidate.is_hold))
    frame["qpos_cmd"][0] = candidate.arm_qpos
    return frame


def _make_hand_command(
    candidate: ActionCandidate, now_monotonic_ns: int, target_monotonic_ns: int
) -> np.ndarray:
    """Serialize an ActionCandidate into a HAND_COMMAND_DTYPE record."""
    if candidate.hand_qpos is None:
        raise ValueError("candidate has no hand command")
    frame = np.zeros(1, dtype=HAND_COMMAND_DTYPE)
    frame["run_generation"][0] = candidate.run_generation
    frame["observation_id"][0] = candidate.observation_id
    frame["action_id"][0] = candidate.action_id
    frame["created_monotonic_ns"][0] = now_monotonic_ns
    frame["target_monotonic_ns"][0] = target_monotonic_ns
    # Preserve the candidate's ownership window. A measured-state bounded hand
    # servo may need several worker ticks before the exact endpoint is sent.
    frame["valid_until_monotonic_ns"][0] = candidate.valid_until_monotonic_ns
    frame["is_hold"][0] = int(bool(candidate.is_hold))
    frame["qpos_cmd"][0] = candidate.hand_qpos
    return frame


def send_command(
    shared: Any,
    candidate: ActionCandidate,
    *,
    prepare_timeout_s: float | None = None,
) -> CommandPublishResult:
    """Publish arm and/or hand commands to the actuator IPC primitives.

    Fire-and-forget: the arm command goes into the bounded queue, the hand
    command overwrites the latest-wins ring.  No ACKs, no commit protocol.

    Returns a typed transport outcome. Callers decide whether a rejection
    means hold, drop, command quiescence, run abort, or global fault.
    """
    timeout = (
        prepare_timeout_s
        if prepare_timeout_s is not None
        else policy_defaults.action_prepare_timeout_s
    )
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("prepare_timeout_s must be finite and positive")

    runtime_rejection = check_runtime_gate(shared)
    if runtime_rejection is not None:
        return CommandPublishResult(
            runtime_rejection.status,
            candidate=candidate,
            detail=runtime_rejection.detail,
        )

    now_ns = time.monotonic_ns()
    lead_time_s = float(getattr(shared, "action_lead_time_s", 0.05))
    target_ns = now_ns + int(lead_time_s * 1e9)

    # Candidate validity and worker delivery have separate time boundaries.
    if target_ns <= now_ns or candidate.valid_until_monotonic_ns < now_ns:
        logger.error(
            "send_command: action_id=%d temporal window closed", candidate.action_id
        )
        return CommandPublishResult(
            CommandPublishStatus.TEMPORAL_WINDOW_CLOSED,
            candidate=candidate,
        )

    deadline_ns = now_ns + int(timeout * 1e9)
    remaining_s = (deadline_ns - time.monotonic_ns()) * 1e-9
    if remaining_s <= 0:
        return CommandPublishResult(
            CommandPublishStatus.PREPARE_TIMEOUT,
            candidate=candidate,
        )

    if candidate.arm_qpos is not None:
        shared.arm_cmd_ring.write(_make_arm_command(candidate, now_ns, target_ns))

    # Both actuator transports are latest-wins seqlock rings; publication is not atomic.
    if candidate.hand_qpos is not None:
        hand_frame = _make_hand_command(candidate, now_ns, target_ns)
        shared.hand_cmd_ring.write(hand_frame)

    return CommandPublishResult(
        CommandPublishStatus.PUBLISHED,
        candidate=candidate,
    )


def build_action_candidate(
    shared: Any,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None,
    *,
    is_hold: bool = False,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    now_ns: int | None = None,
    action_validity_s: float = 0.5,
) -> ActionCandidate | None:
    """Build an ``ActionCandidate`` from raw joint targets.

    Allocates a fresh monotonic ``action_id`` from ``shared.arm_command_seq``
    and stamps the target/valid-until timestamps from
    ``shared.action_lead_time_s`` and ``action_validity_s``.  Returns ``None``
    when the optional observation anchor is non-positive or in the future.
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
    return ActionCandidate(
        observation_id=action_id if observation_id is None else int(observation_id),
        run_generation=int(shared.run_generation.value),
        action_id=action_id,
        created_monotonic_ns=now_ns,
        target_monotonic_ns=now_ns + int(float(shared.action_lead_time_s) * 1e9),
        valid_until_monotonic_ns=now_ns + int(float(action_validity_s) * 1e9),
        arm_qpos=np.asarray(arm_qpos, dtype=np.float64),
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
    prepare_timeout_s: float = 0.06,
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

    gate_result = gate.validate(
        candidate,
        current_arm_qpos=arm_feedback.qpos,
        current_hand_qpos=(hand_feedback.qpos if hand_feedback is not None else None),
        run_generation=int(shared.run_generation.value),
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

    return send_command(shared, candidate, prepare_timeout_s=prepare_timeout_s)


def publish_joint_targets(
    shared: Any,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None = None,
    *,
    is_hold: bool = False,
    prepare_timeout_s: float = 0.05,
    observation_id: int | None = None,
    observation_anchor_monotonic_ns: int | None = None,
    safety_gate: SafetyGate | None = None,
    wait_applied: bool = False,
    apply_timeout_s: float = 0.5,
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
    rated device envelope.

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

    try:
        candidate = build_action_candidate(
            shared,
            arm_qpos,
            hand_qpos,
            is_hold=is_hold,
            observation_id=observation_id,
            observation_anchor_monotonic_ns=observation_anchor_monotonic_ns,
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
        prepare_timeout_s=prepare_timeout_s,
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
        deadline_s = time.monotonic() + float(apply_timeout_s)
        # Coupled arm/hand candidates use one action_id and require both acknowledgements.
        with_hand = published.hand_qpos is not None
        while time.monotonic() < deadline_s:
            runtime_rejection = check_runtime_gate(shared)
            if runtime_rejection is not None:
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
                return feedback_rejection
            assert arm_feedback is not None
            arm_ok = arm_feedback.last_cmd_seq >= action_id
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
            time.sleep(0.005)
        logger.warning(
            "publish_joint_targets: action_id=%d was not acknowledged within %.3fs",
            action_id,
            apply_timeout_s,
        )
        return CommandPublishResult(
            CommandPublishStatus.ACK_TIMEOUT,
            candidate=published,
        )
    return publish_result

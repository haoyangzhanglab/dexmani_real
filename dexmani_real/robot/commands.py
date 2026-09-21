"""Physical command preparation, realtime publication, and blocking acceptance."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.ipc.command_stream import command_stream_capacity_locked
from dexmani_real.ipc.schema import COUPLED_COMMAND_DTYPE
from dexmani_real.runtime.safety import (
    PUBLISH_REASON_ESTOP,
    PUBLISH_REASON_EXPIRED,
    _expire_command_locked,
    PUBLISH_REASON_FAULT,
    PUBLISH_REASON_FIFO_FULL,
    PUBLISH_REASON_GENERATION,
    PUBLISH_REASON_RUNTIME_STOPPED,
    PUBLISH_REASON_SAFETY_STATE,
    CommittedCommand,
    SafetyState,
    cancel_coupled_command_if_current,
    coupled_command_is_current,
    PUBLISH_REASON_NO_CONSUMER,
    read_motion_permit,
)
from dexmani_real.utils.feedback import (
    FeedbackIssue,
    FeedbackIssueCode,
    diagnose_arm_feedback,
    diagnose_feedback_timestamp_order,
    diagnose_hand_feedback,
)
from dexmani_real.utils.log import get_logger, ThrottledWarner

logger = get_logger(__name__)
_warn_full = ThrottledWarner(interval_s=2.0, logger=logger)


@dataclass(frozen=True)
class ActionCandidate:
    """One current command candidate proposed by a control producer.

    Publication confirms the candidate still belongs to the active
    ``run_generation`` and commits it once to the ordered command FIFO. The
    candidate is an owner-owned immutable numeric snapshot: a FULL commit
    result retries this exact object without rebuilding it, re-solving IK, or
    re-clipping its targets, and retains its original execution deadline through every retry.
    """

    run_generation: int
    expires_monotonic_ns: int
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None
    is_hold: bool = False


_JOINT_LIMIT_TOLERANCE_RAD = 1e-12


def _hand_joint_limit_detail(
    hand_qpos_rad: np.ndarray, lower_rad: np.ndarray, upper_rad: np.ndarray
) -> str:
    outside = (hand_qpos_rad < lower_rad - _JOINT_LIMIT_TOLERANCE_RAD) | (
        hand_qpos_rad > upper_rad + _JOINT_LIMIT_TOLERANCE_RAD
    )
    return f"hand_joint_limit:j{np.flatnonzero(outside)[0]}"


class GateRejectCode(str, Enum):
    """Stable machine-readable rejection reasons from :class:`SafetyGate`."""

    INVALID_TARGET = "invalid joint target"
    ARM_JOINT_LIMIT = "arm joint limit violation"
    HAND_JOINT_LIMIT = "hand joint limit violation"
    COLLISION_TRANSITION = "collision on arm/hand transition"
    COLLISION_CHECK_FAILED = "collision transition check failed"
    WORKSPACE = "workspace"
    WORKSPACE_CHECK_FAILED = "workspace check failed"


@dataclass(frozen=True)
class GateResult:
    """Typed outcome of one safety-gate validation."""

    accepted: bool
    code: GateRejectCode | None = None
    detail: str = ""

    @property
    def reason(self) -> str:
        return self.detail or ("" if self.code is None else self.code.value)


class SafetyGate:
    """Fail-closed validation of physical limits, workspace, and collision."""

    def __init__(
        self,
        *,
        arm_joint_lower_rad: tuple[float, ...],
        arm_joint_upper_rad: tuple[float, ...],
        hand_joint_lower_rad: tuple[float, ...],
        hand_joint_upper_rad: tuple[float, ...],
        workspace_check: Callable[[np.ndarray, np.ndarray], bool] | None = None,
        collision_check: (
            Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool] | None
        ) = None,
    ) -> None:
        arm_low = np.asarray(arm_joint_lower_rad, dtype=np.float64)
        arm_high = np.asarray(arm_joint_upper_rad, dtype=np.float64)
        hand_low = np.asarray(hand_joint_lower_rad, dtype=np.float64)
        hand_high = np.asarray(hand_joint_upper_rad, dtype=np.float64)
        if arm_low.shape != ARM_JOINT_SHAPE or arm_high.shape != ARM_JOINT_SHAPE:
            raise ValueError("arm joint limits must have seven entries")
        if hand_low.shape != HAND_JOINT_SHAPE or hand_high.shape != HAND_JOINT_SHAPE:
            raise ValueError("hand joint limits must have twelve entries")
        bounds = np.concatenate((arm_low, arm_high, hand_low, hand_high))
        if (
            not np.all(np.isfinite(bounds))
            or np.any(arm_low > arm_high)
            or np.any(hand_low > hand_high)
        ):
            raise ValueError("joint limits must be finite and ordered")
        self.arm_low = arm_low
        self.arm_high = arm_high
        self.hand_low = hand_low
        self.hand_high = hand_high
        self.workspace_check = workspace_check
        self.collision_check = collision_check

    def validate(
        self,
        candidate: ActionCandidate,
        *,
        current_arm_qpos: np.ndarray,
        current_hand_qpos: np.ndarray | None = None,
    ) -> GateResult:
        """Validate one candidate without modifying it or external state.

        Workspace and collision transitions start at measured feedback.
        """
        # Sensor readers own measured feedback; this gate admits outgoing targets.
        if candidate.arm_qpos is None and candidate.hand_qpos is None:
            return GateResult(
                False, GateRejectCode.INVALID_TARGET, "no actuator target"
            )
        for name, target, shape in (
            ("arm", candidate.arm_qpos, ARM_JOINT_SHAPE),
            ("hand", candidate.hand_qpos, HAND_JOINT_SHAPE),
        ):
            if target is not None and (
                target.shape != shape or not np.all(np.isfinite(target))
            ):
                return GateResult(
                    False, GateRejectCode.INVALID_TARGET, f"{name} target shape/finite"
                )
        arm_start = current_arm_qpos
        arm_end = arm_start.copy() if candidate.arm_qpos is None else candidate.arm_qpos
        hand_end = candidate.hand_qpos
        hand_start: np.ndarray | None = None
        if hand_end is not None:
            assert current_hand_qpos is not None
            hand_start = current_hand_qpos
        if candidate.arm_qpos is not None and (
            np.any(arm_end < self.arm_low) or np.any(arm_end > self.arm_high)
        ):
            return GateResult(False, GateRejectCode.ARM_JOINT_LIMIT)
        if hand_end is not None and (
            np.any(hand_end < self.hand_low - _JOINT_LIMIT_TOLERANCE_RAD)
            or np.any(hand_end > self.hand_high + _JOINT_LIMIT_TOLERANCE_RAD)
        ):
            return GateResult(
                False,
                GateRejectCode.HAND_JOINT_LIMIT,
                _hand_joint_limit_detail(hand_end, self.hand_low, self.hand_high),
            )
        if self.workspace_check is not None and candidate.arm_qpos is not None:
            try:
                if not self.workspace_check(arm_start, arm_end):
                    return GateResult(False, GateRejectCode.WORKSPACE)
            except Exception:
                logger.warning(
                    "SafetyGate: workspace check failed closed", exc_info=True
                )
                return GateResult(False, GateRejectCode.WORKSPACE_CHECK_FAILED)
        # Arm/hand transition collision (requires both current + target hand).
        if (
            self.collision_check is not None
            and candidate.arm_qpos is not None
            and hand_end is not None
            and hand_start is not None
        ):
            try:
                if not self.collision_check(arm_start, arm_end, hand_start, hand_end):
                    return GateResult(False, GateRejectCode.COLLISION_TRANSITION)
            except Exception:
                logger.warning(
                    "SafetyGate: collision transition check failed closed",
                    exc_info=True,
                )
                return GateResult(False, GateRejectCode.COLLISION_CHECK_FAILED)
        return GateResult(True)


def planner_action_safety_gate(
    *,
    planner: Any,
    arm_joint_lower_rad: tuple[float, ...],
    arm_joint_upper_rad: tuple[float, ...],
    hand_joint_lower_rad: tuple[float, ...],
    hand_joint_upper_rad: tuple[float, ...],
    collision_check: (
        Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], bool] | None
    ) = None,
) -> SafetyGate:
    """Build a safety gate using the planner's segment workspace check.

    Collision checking is enabled by the caller that owns the command path.
    """
    return SafetyGate(
        arm_joint_lower_rad=arm_joint_lower_rad,
        arm_joint_upper_rad=arm_joint_upper_rad,
        hand_joint_lower_rad=hand_joint_lower_rad,
        hand_joint_upper_rad=hand_joint_upper_rad,
        workspace_check=planner.is_workspace_segment_safe,
        collision_check=collision_check,
    )


@dataclass(frozen=True)
class PublishResult:
    """Compact result of the realtime IPC publication boundary.

    ``fifo_depth`` is the committed backlog observed by a rejected FULL
    commit, for the producer's visible ``[WAIT]`` report.
    """

    published: bool
    command: CommittedCommand | None = None
    reason: str = ""
    fifo_depth: int = 0


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

    @property
    def accepted(self) -> bool:
        return self.candidate is not None


@dataclass(frozen=True)
class _ActuatorFeedback:
    qpos: np.ndarray
    accepted_monotonic_ns: int = 0
    source_monotonic_ns: int = 0
    ring_commit_monotonic_ns: int = 0
    # Generation/sequence identity of the acceptance watermark: a stale
    # generation's ACK can never satisfy a current-epoch acceptance wait.
    accepted_generation: int = 0
    accepted_sequence: int = 0


@dataclass(frozen=True)
class CommandFeedbackSnapshot:
    """One immutable feedback selection reused across a single command dispatch.

    Arm and hand are independent ring reads; ``read_command_feedback`` then
    captures one post-selection ``validation_now_ns`` and revalidates both
    modalities' provenance (``source <= ring_commit <= validation_now``) and
    freshness against it before constructing this immutable snapshot. This is
    the only feedback a policy dispatch may use for decode/IK, SafetyGate, and
    publication preparation — never re-read the rings mid-dispatch.
    """

    arm_qpos: np.ndarray
    arm_source_monotonic_ns: int
    hand_qpos: np.ndarray | None = None
    hand_source_monotonic_ns: int | None = None
    arm_ring_commit_monotonic_ns: int = 0
    hand_ring_commit_monotonic_ns: int | None = None
    validation_now_ns: int = 0


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
    max_age_s: float | None,
    now_monotonic_ns: int | None = None,
) -> tuple[_ActuatorFeedback | None, str, FeedbackIssue | None]:
    result = shared.arm_state_ring.read_latest()
    if result is None:
        return None, "arm feedback unavailable", None
    record = result[0][0]
    ring_commit_ns = int(result[1])
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
    source_ns = int(record["source_monotonic_ns"])
    issue = diagnose_arm_feedback(
        connected=bool(record["connected"]),
        error_code=int(record["error_code"]),
        state_valid=bool(record["state_valid"]),
        source_monotonic_ns=source_ns,
        now_monotonic_ns=now_ns,
        max_age_s=max_age_s,
        qpos=qpos,
        qvel=np.asarray(record["qvel"], dtype=np.float64),
    )
    if issue is not None:
        return None, f"arm feedback is unhealthy: {issue.detail}", issue
    return (
        _ActuatorFeedback(
            qpos=qpos.copy(),
            accepted_monotonic_ns=int(record["last_cmd_accepted_monotonic_ns"]),
            source_monotonic_ns=source_ns,
            ring_commit_monotonic_ns=ring_commit_ns,
            accepted_generation=int(record["last_cmd_generation"]),
            accepted_sequence=int(record["last_cmd_accepted_sequence"]),
        ),
        "",
        None,
    )


def read_hand_feedback(
    shared: Any,
    *,
    max_age_s: float | None,
    now_monotonic_ns: int | None = None,
) -> tuple[_ActuatorFeedback | None, str, FeedbackIssue | None]:
    result = shared.hand_state_ring.read_latest()
    if result is None:
        return None, "hand feedback unavailable", None
    record = result[0][0]
    ring_commit_ns = int(result[1])
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    now_ns = time.monotonic_ns() if now_monotonic_ns is None else int(now_monotonic_ns)
    source_ns = int(record["source_monotonic_ns"])
    issue = diagnose_hand_feedback(
        connected=bool(record["connected"]),
        state_valid=bool(record["state_valid"]),
        source_monotonic_ns=source_ns,
        now_monotonic_ns=now_ns,
        max_age_s=max_age_s,
        qpos=qpos,
    )
    if issue is not None:
        return None, f"hand feedback is unhealthy: {issue.detail}", issue
    return (
        _ActuatorFeedback(
            qpos=qpos.copy(),
            accepted_monotonic_ns=int(record["accepted_target_monotonic_ns"]),
            source_monotonic_ns=source_ns,
            ring_commit_monotonic_ns=ring_commit_ns,
            accepted_generation=int(record["accepted_target_generation"]),
            accepted_sequence=int(record["accepted_target_sequence"]),
        ),
        "",
        None,
    )


def read_command_feedback(
    shared: Any,
    *,
    require_hand: bool,
    arm_max_age_s: float | None,
    hand_max_age_s: float | None,
) -> tuple[CommandFeedbackSnapshot | None, str, FeedbackIssue | None]:
    """Select one immutable feedback snapshot for a single command dispatch.

    Each modality is read and validated against its own post-read local now
    (never a pre-captured common ``now``, which can race a concurrent producer
    into a false ``FUTURE_TIMESTAMP``). After both modalities are selected, one
    post-selection ``validation_now_ns`` is captured and both are revalidated
    against it for provenance (``source <= ring_commit <= validation_now``), so
    an out-of-order commit is never admitted. Returned arrays are ownership
    copies, so later ring writes cannot mutate the selected state.

    ``*_max_age_s=None`` removes the age-only veto (truthfulness and causality
    still apply): policy dispatch relies on worker supervision for liveness,
    while teleop/replay callers keep explicit live-feedback thresholds.
    """
    arm_feedback, reason, issue = _read_arm_feedback(
        shared, max_age_s=arm_max_age_s
    )
    if arm_feedback is None:
        return None, reason, issue

    hand_feedback: _ActuatorFeedback | None = None
    if require_hand:
        hand_feedback, reason, issue = read_hand_feedback(
            shared, max_age_s=hand_max_age_s
        )
        if hand_feedback is None:
            return None, reason, issue

    validation_now_ns = time.monotonic_ns()

    issue = diagnose_feedback_timestamp_order(
        source_monotonic_ns=arm_feedback.source_monotonic_ns,
        ring_commit_monotonic_ns=arm_feedback.ring_commit_monotonic_ns,
        validation_now_ns=validation_now_ns,
        max_age_s=arm_max_age_s,
        modality="arm",
    )
    if issue is not None:
        return None, f"arm feedback is unhealthy: {issue.detail}", issue

    hand_qpos: np.ndarray | None = None
    hand_source_ns: int | None = None
    hand_commit_ns: int | None = None
    if hand_feedback is not None:
        issue = diagnose_feedback_timestamp_order(
            source_monotonic_ns=hand_feedback.source_monotonic_ns,
            ring_commit_monotonic_ns=hand_feedback.ring_commit_monotonic_ns,
            validation_now_ns=validation_now_ns,
            max_age_s=hand_max_age_s,
            modality="hand",
        )
        if issue is not None:
            return None, f"hand feedback is unhealthy: {issue.detail}", issue
        hand_qpos = hand_feedback.qpos
        hand_source_ns = hand_feedback.source_monotonic_ns
        hand_commit_ns = hand_feedback.ring_commit_monotonic_ns

    return (
        CommandFeedbackSnapshot(
            arm_qpos=arm_feedback.qpos,
            arm_source_monotonic_ns=arm_feedback.source_monotonic_ns,
            hand_qpos=hand_qpos,
            hand_source_monotonic_ns=hand_source_ns,
            arm_ring_commit_monotonic_ns=arm_feedback.ring_commit_monotonic_ns,
            hand_ring_commit_monotonic_ns=hand_commit_ns,
            validation_now_ns=validation_now_ns,
        ),
        "",
        None,
    )


def prepare_joint_command(
    shared: Any,
    arm_qpos: np.ndarray,
    hand_qpos: np.ndarray | None = None,
    *,
    gate: SafetyGate,
    expires_monotonic_ns: int,
    run_generation: int | None = None,
    is_hold: bool = False,
    arm_feedback_max_age_s: float,
    hand_feedback_max_age_s: float,
    feedback_snapshot: CommandFeedbackSnapshot | None = None,
) -> PreparedCommand:
    """Copy and check a target once; retry this snapshot unchanged after FIFO FULL.

    A supplied feedback snapshot is reused by decode, IK and these checks.
    Mechanical/SDK validity remains checked by the actuator-owning worker.
    """
    try:
        candidate = ActionCandidate(
            expires_monotonic_ns=expires_monotonic_ns,
            run_generation=(int(shared.run_generation.value)
                            if run_generation is None else run_generation),
            arm_qpos=np.array(arm_qpos, dtype=np.float64, copy=True),
            hand_qpos=(None if hand_qpos is None else
                       np.array(hand_qpos, dtype=np.float64, copy=True)),
            is_hold=is_hold,
        )
    except (TypeError, ValueError) as exc:
        return PreparedCommand(reason=str(exc), fatal=True)
    current_arm_qpos: np.ndarray
    current_hand_qpos: np.ndarray | None
    if feedback_snapshot is None:
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

        hand_feedback: _ActuatorFeedback | None = None
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
        current_arm_qpos = arm_feedback.qpos
        current_hand_qpos = hand_feedback.qpos if hand_feedback is not None else None
    else:
        current_arm_qpos = feedback_snapshot.arm_qpos
        current_hand_qpos = feedback_snapshot.hand_qpos

    gate_result = gate.validate(
        candidate,
        current_arm_qpos=current_arm_qpos,
        current_hand_qpos=current_hand_qpos,
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
        )

    return PreparedCommand(
        candidate=candidate,
    )


def command_publishability_reason(
    shared: Any,
    candidate: ActionCandidate,
    *,
    check_is_running: bool = True,
    required_safety_state: SafetyState | None = None,
) -> str:
    """Check the runtime and generation publication invariants without committing.

    This is the ``execute=False`` rehearsal of :func:`publish_command`; it
    deliberately does not evaluate FIFO capacity, which is only meaningful at
    the atomic commit point.
    """
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
    return ""


def publish_command(
    shared: Any,
    candidate: ActionCandidate,
    *,
    required_safety_state: SafetyState,
) -> PublishResult:
    """Commit one checked command to the ordered FIFO without waiting for
    worker acknowledgement.

    A FULL result is recoverable backpressure: the caller keeps this exact
    candidate and retries from its main loop cadence.
    """
    frame = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
    if not 0 < candidate.expires_monotonic_ns < 2**64:
        raise ValueError("command deadline must be positive uint64 monotonic nanoseconds")
    frame["expires_monotonic_ns"][0] = candidate.expires_monotonic_ns
    frame["run_generation"][0] = candidate.run_generation
    frame["is_hold"][0] = int(candidate.is_hold)
    if candidate.arm_qpos is not None:
        frame["arm_present"][0] = 1
        frame["arm_qpos"][0] = candidate.arm_qpos
    if candidate.hand_qpos is not None:
        frame["hand_present"][0] = 1
        frame["hand_qpos"][0] = candidate.hand_qpos
    with shared.motion_lock:
        if bool(shared.estop_request.value):
            return PublishResult(False, reason=PUBLISH_REASON_ESTOP)
        if bool(shared.error_state.value):
            return PublishResult(False, reason=PUBLISH_REASON_FAULT)
        if not bool(shared.is_running.value):
            return PublishResult(False, reason=PUBLISH_REASON_RUNTIME_STOPPED)
        state = SafetyState(int(shared.safety_state.value))
        if state not in (SafetyState.ARMED, SafetyState.RUNNING):
            return PublishResult(False, reason=f"{PUBLISH_REASON_SAFETY_STATE}: {state.name}")
        if state is not required_safety_state:
            return PublishResult(False, reason=(
                f"{PUBLISH_REASON_SAFETY_STATE}: expected {required_safety_state.name}, "
                f"got {state.name}"))
        if int(shared.run_generation.value) != candidate.run_generation:
            return PublishResult(False, reason=PUBLISH_REASON_GENERATION)
        if _expire_command_locked(shared, candidate.run_generation, candidate.expires_monotonic_ns):
            return PublishResult(False, reason=PUBLISH_REASON_EXPIRED)
        has_capacity, backlog, any_attached = command_stream_capacity_locked(shared)
        if not has_capacity:
            result = PublishResult(False, reason=(
                PUBLISH_REASON_FIFO_FULL if any_attached else PUBLISH_REASON_NO_CONSUMER
            ), fifo_depth=backlog)
        else:
            sequence = int(shared.coupled_cmd_ring.write(frame))
            result = PublishResult(True, command=CommittedCommand(
                run_generation=candidate.run_generation,
                sequence=sequence,
                published_monotonic_ns=time.monotonic_ns(),
            ))
    # Logging must not hold the motion lock and delay an operator's stop fence.
    if result.reason == PUBLISH_REASON_FIFO_FULL:
        _warn_full("command FIFO full depth=%d; retaining target", result.fifo_depth)
    return result


def wait_command_accepted(
    shared: Any,
    *,
    command: CommittedCommand,
    wait_for_arm: bool,
    wait_for_hand: bool,
    timeout_s: float,
    arm_feedback_max_age_s: float,
    hand_feedback_max_age_s: float,
    check_is_running: bool = True,
    abort_requested: Callable[[], bool] | None = None,
    heartbeat: Callable[[], None] | None = None,
) -> AcceptanceResult:
    """Block until the requested workers report SDK acceptance of one command.

    Acceptance is judged by ordered consumption inside the command's own run
    generation: a worker only advances its
    acceptance watermark after SDK-accepting every targeted record in commit
    order, so a same-generation watermark at or beyond this command's FIFO
    sequence proves ordered acceptance of this command. A stale-generation
    ACK can never satisfy the
    wait. Explicit waits are for home/replay/calibration boundaries only;
    ordinary streaming never blocks here.
    """
    if not wait_for_arm and not wait_for_hand:
        raise ValueError("acceptance wait requires at least one worker")
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("acceptance timeout must be finite and positive")
    if command.sequence <= 0:
        raise ValueError("acceptance requires a published command sequence")
    generation = int(command.run_generation)
    sequence = int(command.sequence)
    deadline_s = time.monotonic() + timeout_s
    while time.monotonic() < deadline_s:
        if abort_requested is not None and abort_requested():
            cancel_coupled_command_if_current(shared, command=command)
            return AcceptanceResult(False, "acceptance aborted")
        reason = motion_rejection_reason(shared, check_is_running=check_is_running)
        if reason:
            cancel_coupled_command_if_current(shared, command=command)
            return AcceptanceResult(False, reason)
        if heartbeat is not None:
            heartbeat()

        arm_accepted = not wait_for_arm
        if wait_for_arm:
            arm_feedback, reason, _ = _read_arm_feedback(
                shared, max_age_s=arm_feedback_max_age_s
            )
            if arm_feedback is None:
                cancel_coupled_command_if_current(shared, command=command)
                return AcceptanceResult(False, reason)
            arm_accepted = arm_feedback.accepted_generation == generation and (
                arm_feedback.accepted_sequence >= sequence
            )
            if arm_accepted and arm_feedback.accepted_monotonic_ns <= 0:
                cancel_coupled_command_if_current(shared, command=command)
                return AcceptanceResult(False, "arm acceptance timestamp is missing")

        hand_accepted = not wait_for_hand
        if wait_for_hand:
            hand_feedback, reason, _ = read_hand_feedback(
                shared, max_age_s=hand_feedback_max_age_s
            )
            if hand_feedback is None:
                cancel_coupled_command_if_current(shared, command=command)
                return AcceptanceResult(False, reason)
            hand_accepted = hand_feedback.accepted_generation == generation and (
                hand_feedback.accepted_sequence >= sequence
            )
            if hand_accepted and hand_feedback.accepted_monotonic_ns <= 0:
                cancel_coupled_command_if_current(shared, command=command)
                return AcceptanceResult(False, "hand acceptance timestamp is missing")

        # A matching old SDK return remains historical acceptance, never a
        # successful wait after that operation lost its motion generation.
        if not coupled_command_is_current(shared, command=command):
            return AcceptanceResult(
                False, "command generation was revoked before acceptance"
            )
        if arm_accepted and hand_accepted:
            return AcceptanceResult(True)
        time.sleep(0.005)

    cancel_coupled_command_if_current(shared, command=command)
    return AcceptanceResult(False, f"command was not accepted within {timeout_s:.3f}s")

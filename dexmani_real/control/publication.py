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
from dexmani_real.ipc.schema import COUPLED_COMMAND_DTYPE
from dexmani_real.runtime.safety import (
    PUBLISH_REASON_ESTOP,
    PUBLISH_REASON_FAULT,
    PUBLISH_REASON_FIFO_FULL,
    PUBLISH_REASON_GENERATION,
    PUBLISH_REASON_RUNTIME_STOPPED,
    PUBLISH_REASON_SAFETY_STATE,
    CommittedCommand,
    SafetyState,
    cancel_coupled_command_if_current,
    coupled_command_is_current,
    publish_coupled_command_if_motion_permitted,
    read_motion_permit,
)
from dexmani_real.utils.feedback import (
    FeedbackIssue,
    FeedbackIssueCode,
    diagnose_arm_feedback,
    diagnose_feedback_timestamp_order,
    diagnose_hand_feedback,
)
from dexmani_real.utils.limits import (
    canonicalize_policy_hand_endpoint_roundoff,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


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


class PublishWaitTracker:
    """One visible ``[WAIT]``/``[RESUME]`` pair per continuous FIFO-full span.

    Each command producer owns one tracker. Entering backpressure prints once;
    repeated FULL retries of the same kept candidate stay silent. A span ends
    exactly one way: either the kept candidate commits (``[RESUME]`` with the
    elapsed wait) or lifecycle revokes it (``[DROP]`` naming the kept action
    and the reason). Every terminal path must call one of the two, so the next
    span starts from a clean state and each decision keeps exactly one line.
    """

    def __init__(self, label: str) -> None:
        self._label = label
        self._waiting_since_ns: int | None = None
        self._keep_action_id = 0

    @property
    def waiting(self) -> bool:
        return self._waiting_since_ns is not None

    def note_full(self, depth: int, keep_action: int) -> None:
        if self._waiting_since_ns is None:
            logger.warning(
                "[WAIT] %s command_fifo full depth=%d keep_action=%d",
                self._label,
                int(depth),
                int(keep_action),
            )
            self._waiting_since_ns = time.monotonic_ns()
            self._keep_action_id = int(keep_action)

    def note_committed(self) -> None:
        if self._waiting_since_ns is None:
            return
        wait_ms = (time.monotonic_ns() - self._waiting_since_ns) / 1e6
        logger.info(
            "[RESUME] %s command_fifo wait_ms=%.0f dropped=0",
            self._label,
            wait_ms,
        )
        self._reset()

    def note_dropped(self, reason: str, *, report: bool = True) -> None:
        """Report the kept candidate revoked by lifecycle while waiting."""
        if self._waiting_since_ns is None:
            return
        wait_ms = (time.monotonic_ns() - self._waiting_since_ns) / 1e6
        if report:
            logger.warning(
                "[DROP] %s command_fifo keep_action=%d wait_ms=%.0f dropped=1 reason=%s",
                self._label,
                self._keep_action_id,
                wait_ms,
                reason,
            )
        self._reset()

    def _reset(self) -> None:
        self._waiting_since_ns = None
        self._keep_action_id = 0


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
    source_monotonic_ns: int = 0
    ring_commit_monotonic_ns: int = 0
    # Generation/sequence identity of the acceptance watermark: a stale
    # generation's ACK can never satisfy a current-epoch acceptance wait.
    accepted_generation: int = 0
    accepted_sequence: int = 0


@dataclass(frozen=True)
class _HandFeedbackSnapshot:
    qpos: np.ndarray
    accepted_action_id: int
    accepted_monotonic_ns: int = 0
    source_monotonic_ns: int = 0
    ring_commit_monotonic_ns: int = 0
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
    the pre-publication freshness recheck — never re-read the rings mid-dispatch.
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
) -> tuple[_ArmFeedbackSnapshot | None, str, FeedbackIssue | None]:
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
        _ArmFeedbackSnapshot(
            qpos=qpos.copy(),
            accepted_action_id=int(record["last_cmd_seq"]),
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
) -> tuple[_HandFeedbackSnapshot | None, str, FeedbackIssue | None]:
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
        _HandFeedbackSnapshot(
            qpos=qpos.copy(),
            accepted_action_id=int(record["accepted_target_action_id"]),
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

    hand_feedback: _HandFeedbackSnapshot | None = None
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


def build_action_candidate(
    shared: Any,
    arm_qpos: np.ndarray | None,
    hand_qpos: np.ndarray | None,
    *,
    run_generation: int | None = None,
    is_hold: bool = False,
) -> ActionCandidate:
    """Assign command identity and copy targets into an immutable candidate.

    The candidate carries no delivery lease or timing authority: it stays
    committable until its run generation is revoked, and a FULL commit result
    retries this exact numeric snapshot unchanged. Action IDs come from the
    shared monotonic counter and may have gaps; the FIFO queue sequence is
    produced later by the commit and is never mixed with them.
    """
    if run_generation is not None and (
        isinstance(run_generation, (bool, np.bool_))
        or not isinstance(run_generation, (int, np.integer))
        or int(run_generation) < 0
    ):
        raise ValueError("run_generation must be a non-negative integer or None")
    with shared.arm_command_seq.get_lock():
        action_id = int(shared.arm_command_seq.value) + 1
        shared.arm_command_seq.value = action_id
    return ActionCandidate(
        run_generation=(
            int(shared.run_generation.value)
            if run_generation is None
            else int(run_generation)
        ),
        action_id=action_id,
        arm_qpos=(
            None
            if arm_qpos is None
            else np.array(arm_qpos, dtype=np.float64, copy=True)
        ),
        hand_qpos=(
            None
            if hand_qpos is None
            else np.array(hand_qpos, dtype=np.float64, copy=True)
        ),
        is_hold=is_hold,
    )


def prepare_command(
    shared: Any,
    candidate: ActionCandidate,
    *,
    gate: SafetyGate,
    arm_feedback_max_age_s: float,
    hand_feedback_max_age_s: float,
    hand_delta_reference_qpos: np.ndarray | None = None,
    hand_mechanical_lower_rad: np.ndarray | None = None,
    hand_mechanical_upper_rad: np.ndarray | None = None,
    canonicalize_policy_hand_roundoff: bool = False,
    feedback_snapshot: CommandFeedbackSnapshot | None = None,
) -> PreparedCommand:
    """Check one candidate without shaping it, against valid current feedback.

    ``feedback_snapshot is None`` reads/validates arm and (when required) hand
    feedback internally, as before. When a snapshot is supplied, it is used
    as-is for SafetyGate's current state without any additional ring read —
    the caller already selected and validated it for this dispatch.
    """
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
        current_arm_qpos = arm_feedback.qpos
        current_hand_qpos = hand_feedback.qpos if hand_feedback is not None else None
    else:
        current_arm_qpos = feedback_snapshot.arm_qpos
        current_hand_qpos = feedback_snapshot.hand_qpos

    hand_roundoff_canonicalized = False
    if candidate.hand_qpos is not None and canonicalize_policy_hand_roundoff:
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
        try:
            hand_qpos, hand_roundoff_canonicalized = (
                canonicalize_policy_hand_endpoint_roundoff(
                    candidate.hand_qpos,
                    gate.hand_low,
                    gate.hand_high,
                    mechanical_lower,
                    mechanical_upper,
                )
            )
        except ValueError as exc:
            return PreparedCommand(reason=str(exc))
        if hand_roundoff_canonicalized:
            candidate = replace(candidate, hand_qpos=hand_qpos)

    gate_result = gate.validate(
        candidate,
        current_arm_qpos=current_arm_qpos,
        current_hand_qpos=current_hand_qpos,
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
    hand_delta_reference_qpos: np.ndarray | None = None,
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
        )
    except (TypeError, ValueError) as exc:
        return PreparedCommand(reason=str(exc), fatal=True)
    return prepare_command(
        shared,
        candidate,
        gate=gate,
        arm_feedback_max_age_s=arm_feedback_max_age_s,
        hand_feedback_max_age_s=hand_feedback_max_age_s,
        hand_delta_reference_qpos=hand_delta_reference_qpos,
    )


def _make_coupled_command(candidate: ActionCandidate) -> np.ndarray:
    frame = np.zeros(1, dtype=COUPLED_COMMAND_DTYPE)
    frame["run_generation"][0] = candidate.run_generation
    frame["action_id"][0] = candidate.action_id
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
    command, rejection_reason, fifo_depth = publish_coupled_command_if_motion_permitted(
        shared,
        expected_run_generation=int(candidate.run_generation),
        frame=_make_coupled_command(candidate),
        required_state=required_safety_state,
    )
    if command is None:
        return PublishResult(False, reason=rejection_reason, fifo_depth=fifo_depth)
    return PublishResult(True, command=command)


def wait_command_accepted(
    shared: Any,
    *,
    command: CommittedCommand,
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
    """Block until the requested workers report SDK acceptance of one command.

    Acceptance is judged by ordered consumption inside the command's own run
    generation, never by action-ID supersession: a worker only advances its
    acceptance watermark after SDK-accepting every targeted record in commit
    order, so a same-generation watermark at or beyond this command's FIFO
    sequence proves ordered acceptance of this command. A larger action ID
    alone proves nothing, and a stale-generation ACK can never satisfy the
    wait. Explicit waits are for home/replay/calibration boundaries only;
    ordinary streaming never blocks here.
    """
    if not wait_for_arm and not wait_for_hand:
        raise ValueError("acceptance wait requires at least one worker")
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError("acceptance timeout must be finite and positive")
    if action_id < 0 or command.sequence <= 0:
        raise ValueError("acceptance identity must be non-negative and published")
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
                arm_feedback.accepted_action_id == action_id
                or arm_feedback.accepted_sequence >= sequence
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
                hand_feedback.accepted_action_id == action_id
                or hand_feedback.accepted_sequence >= sequence
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

"""Physical command preparation, realtime publication, and blocking acceptance."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.robot.model import ARM_JOINT_SHAPE, HAND_JOINT_SHAPE
from dexmani_real.ipc.schema import ROBOT_COMMAND_DTYPE
from dexmani_real.runtime.safety import (
    PUBLISH_REASON_ESTOP, PUBLISH_REASON_FAULT, PUBLISH_REASON_PENDING,
    PUBLISH_REASON_RUN, PUBLISH_REASON_RUNTIME_STOPPED, PUBLISH_REASON_SAFETY_STATE,
    SafetyState, cancel_coupled_command_if_current, coupled_command_is_current,
    read_motion_permit,
)
from dexmani_real.utils.feedback import (
    FeedbackIssue,
    FeedbackIssueCode,
    diagnose_arm_feedback,
    diagnose_feedback_timestamp_order,
    diagnose_hand_feedback,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RobotCommand:
    """Owned immutable targets. ID zero denotes an unpublished checked proposal."""

    run_id: int
    arm_qpos: np.ndarray | None = None
    hand_qpos: np.ndarray | None = None
    command_id: int = 0
    issued_monotonic_ns: int = 0

    def __post_init__(self) -> None:
        for name in ("arm_qpos", "hand_qpos"):
            value = getattr(self, name)
            if value is not None:
                value = np.array(value, dtype=np.float64, copy=True)
                value.flags.writeable = False
                object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CommandAdoption:
    arm_adopted: bool = False
    hand_adopted: bool = False
    arm_monotonic_ns: int = 0
    hand_monotonic_ns: int = 0
    arm_known: bool = False
    hand_known: bool = False

    def complete(self, command: RobotCommand) -> bool:
        return bool(command.command_id and
                    (command.arm_qpos is None or self.arm_adopted) and
                    (command.hand_qpos is None or self.hand_adopted))

    def fields(self, command: RobotCommand) -> dict[str, int | bool]:
        return dict(
            command_id=command.command_id, command_run_id=command.run_id,
            command_issued_monotonic_ns=command.issued_monotonic_ns,
            command_arm_present=command.arm_qpos is not None,
            command_hand_present=command.hand_qpos is not None,
            arm_command_adopted=self.arm_adopted,
            hand_command_adopted=self.hand_adopted,
            arm_command_adopted_monotonic_ns=self.arm_monotonic_ns,
            hand_command_adopted_monotonic_ns=self.hand_monotonic_ns,
        )


def read_command_adoption(shared: Any, command: RobotCommand, *, boundary_ns: int = 0) -> CommandAdoption:
    """Read exact identities; a fresh source sample proves post-boundary absence.

    Positive historical facts remain useful even if the sensor is now unhealthy.
    Negative facts require healthy post-boundary feedback, never a republished
    stale payload. Each SDK owner publishes state serially with its SDK calls.
    """
    values = {}
    for name in ("arm", "hand"):
        present = getattr(command, f"{name}_qpos") is not None
        result = getattr(shared, f"{name}_state_ring").read_latest() if present else None
        adopted, stamp, known = False, 0, not present
        if result is not None:
            record = result[0][0]
            adopted = bool(command.command_id and
                int(record["last_adopted_run_id"]) == command.run_id and
                int(record["last_adopted_command_id"]) == command.command_id and
                int(record["last_adopted_monotonic_ns"]) > 0)
            stamp = int(record["last_adopted_monotonic_ns"]) if adopted else 0
            known = bool(record["state_valid"] and
                         int(record["source_monotonic_ns"]) > boundary_ns)
        values.update({f"{name}_adopted": adopted, f"{name}_monotonic_ns": stamp,
                       f"{name}_known": known})
    return CommandAdoption(**values)


def read_robot_command(shared: Any) -> RobotCommand | None:
    result = shared.robot_command_ring.read_latest()
    if result is None:
        return None
    record = result[0][0]
    return RobotCommand(
        command_id=int(record["command_id"]), run_id=int(record["run_id"]),
        issued_monotonic_ns=int(record["issued_monotonic_ns"]),
        arm_qpos=record["arm_qpos"] if record["arm_present"] else None,
        hand_qpos=record["hand_qpos"] if record["hand_present"] else None,
    )


def command_admission_ready(shared: Any) -> bool:
    with shared.motion_lock:
        if shared.pending_record_command_id.value:
            return False
        previous = read_robot_command(shared)
        return (previous is None or previous.run_id != int(shared.run_id.value)
                or read_command_adoption(shared, previous).complete(previous))


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
        candidate: RobotCommand,
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
    """Publication outcome; the reason distinguishes lifecycle from executor lag."""
    published: bool
    command: RobotCommand | None = None
    reason: str = ""


@dataclass(frozen=True)
class AcceptanceResult:
    """Result of an explicitly blocking worker/SDK acceptance wait."""

    accepted: bool
    reason: str = ""


@dataclass(frozen=True)
class PreparedCommand:
    """A physically checked command, or its preparation rejection."""

    candidate: RobotCommand | None = None
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
    source_monotonic_ns: int = 0
    ring_commit_monotonic_ns: int = 0


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
            source_monotonic_ns=source_ns,
            ring_commit_monotonic_ns=ring_commit_ns,
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
            source_monotonic_ns=source_ns,
            ring_commit_monotonic_ns=ring_commit_ns,
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
    run_id: int | None = None,
    arm_feedback_max_age_s: float,
    hand_feedback_max_age_s: float,
    feedback_snapshot: CommandFeedbackSnapshot | None = None,
) -> PreparedCommand:
    """Copy targets and apply the producer safety boundary using one feedback snapshot.

    A supplied feedback snapshot is reused by decode, IK and these checks.
    Mechanical/SDK validity remains checked by the actuator-owning worker.
    """
    try:
        candidate = RobotCommand(
            run_id=(int(shared.run_id.value)
                            if run_id is None else run_id),
            arm_qpos=None if arm_qpos is None else np.array(arm_qpos, dtype=np.float64, copy=True),
            hand_qpos=(None if hand_qpos is None else
                       np.array(hand_qpos, dtype=np.float64, copy=True)),
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
    shared: Any, candidate: RobotCommand, *, check_is_running: bool = True,
    required_safety_state: SafetyState | None = None,
) -> str:
    reason = motion_rejection_reason(shared, check_is_running=check_is_running,
                                     required_safety_state=required_safety_state)
    if reason:
        return reason
    return "" if candidate.run_id == int(shared.run_id.value) else PUBLISH_REASON_RUN


def publish_command(
    shared: Any, candidate: RobotCommand, *, required_safety_state: SafetyState,
    account_recording: bool = False,
) -> PublishResult:
    """Publish one checked target, never overwriting unaccounted physical evidence."""
    with shared.motion_lock:
        reason = command_publishability_reason(shared, candidate,
                                               required_safety_state=required_safety_state)
        if reason:
            return PublishResult(False, reason=reason)
        if not command_admission_ready(shared):
            return PublishResult(False, reason=PUBLISH_REASON_PENDING)
        if candidate.arm_qpos is None and candidate.hand_qpos is None:
            raise ValueError("command requires an actuator")
        for name, shape in (("arm", ARM_JOINT_SHAPE), ("hand", HAND_JOINT_SHAPE)):
            target = getattr(candidate, f"{name}_qpos")
            if target is not None and (target.shape != shape or not np.isfinite(target).all()):
                raise ValueError(f"invalid {name} command target")
            if target is not None and not shared.is_ready(name):
                return PublishResult(False, reason=f"{name} worker unavailable")
        command_id = int(shared.next_command_id.value)
        if not 0 < command_id < 2**64 - 1:
            raise OverflowError("command ID lifetime exhausted")
        shared.next_command_id.value = command_id + 1
        command = RobotCommand(run_id=candidate.run_id, command_id=command_id,
            issued_monotonic_ns=time.monotonic_ns(), arm_qpos=candidate.arm_qpos,
            hand_qpos=candidate.hand_qpos)
        frame = np.zeros(1, dtype=ROBOT_COMMAND_DTYPE)
        frame["command_id"] = command.command_id
        frame["run_id"] = command.run_id
        frame["issued_monotonic_ns"] = command.issued_monotonic_ns
        for name in ("arm", "hand"):
            target = getattr(command, f"{name}_qpos")
            frame[f"{name}_present"] = target is not None
            if target is not None:
                frame[f"{name}_qpos"] = target
        shared.robot_command_ring.write(frame)
        if account_recording:
            shared.pending_record_command_id.value = command.command_id
            shared.pending_record_revoked_ns.value = 0
        return PublishResult(True, command)


def wait_command_adopted(
    shared: Any, *, command: RobotCommand, wait_for_arm: bool, wait_for_hand: bool,
    timeout_s: float, arm_feedback_max_age_s: float, hand_feedback_max_age_s: float,
    check_is_running: bool = True, abort_requested: Callable[[], bool] | None = None,
    heartbeat: Callable[[], None] | None = None, hand_reached: bool = False,
) -> AcceptanceResult:
    """Wait for exact SDK adoption, or explicitly for the hand's exact endpoint.

    Endpoint acceptance is not measured convergence. Dedicated arm home keeps
    its separate settled-state and mode-restoration witnesses.
    """
    if not (wait_for_arm or wait_for_hand) or command.command_id <= 0:
        raise ValueError("wait requires a published command and an actuator")
    if not np.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("wait timeout must be finite and positive")
    deadline = time.monotonic() + timeout_s
    reason = "command adoption timeout"
    while time.monotonic() < deadline:
        if abort_requested is not None and abort_requested():
            reason = "command wait aborted"
            break
        reason = motion_rejection_reason(shared, check_is_running=check_is_running)
        if reason or not coupled_command_is_current(shared, command=command):
            reason = reason or "command run revoked"
            break
        if heartbeat is not None:
            heartbeat()
        adopted = read_command_adoption(shared, command)
        hand_done = adopted.hand_adopted
        if hand_reached and wait_for_hand:
            result = shared.hand_state_ring.read_latest()
            hand_done = bool(result is not None and
                int(result[0]["last_reached_run_id"][0]) == command.run_id and
                int(result[0]["last_reached_command_id"][0]) == command.command_id and
                int(result[0]["last_reached_monotonic_ns"][0]) > 0)
        healthy = True
        for name, needed, max_age in (("arm", wait_for_arm, arm_feedback_max_age_s),
                                      ("hand", wait_for_hand, hand_feedback_max_age_s)):
            if needed:
                reader = _read_arm_feedback if name == "arm" else read_hand_feedback
                feedback, failure, _ = reader(shared, max_age_s=max_age)
                if feedback is None:
                    reason, healthy = failure, False
                    break
        if not healthy:
            break
        if (not wait_for_arm or adopted.arm_adopted) and (not wait_for_hand or hand_done):
            # A revocation racing the reads wins over a blocking operation's success.
            if coupled_command_is_current(shared, command=command):
                return AcceptanceResult(True)
            break
        time.sleep(0.005)
    cancel_coupled_command_if_current(shared, command=command)
    return AcceptanceResult(False, reason or "command adoption timeout")

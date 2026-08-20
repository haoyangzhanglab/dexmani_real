"""Unified safety gate — the single validation boundary for all action paths.

Every coordinator (teleop, keyboard, replay, calibration) must
route candidates through :class:`SafetyGate` before :func:`send_command` writes
to the actuator IPC primitives.  Workers trust the gate and apply commands
immediately with hardware-boundary checks (safety state, dtype, finite values,
command/mechanical limits, lifecycle metadata, and SDK return code).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.config.defaults import policy as policy_defaults
from dexmani_real.policy.runtime import ActionCandidate
from dexmani_real.robot.safety import SafetyState
from dexmani_real.utils.hand_health import validate_hand_feedback
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import (
    ARM_COMMAND_DTYPE,
    ARM_JOINT_SHAPE,
    HAND_COMMAND_DTYPE,
    HAND_JOINT_SHAPE,
)

logger = get_logger(__name__)


def advance_run_generation(shared: Any) -> int:
    """Invalidate candidates prepared before the current control run."""
    lock_getter = getattr(shared.run_generation, "get_lock", None)
    if callable(lock_getter):
        with lock_getter():
            shared.run_generation.value = int(shared.run_generation.value) + 1
            return int(shared.run_generation.value)
    shared.run_generation.value = int(shared.run_generation.value) + 1
    return int(shared.run_generation.value)


class GateRejectCode(str, Enum):
    """Stable machine-readable reasons emitted by :class:`SafetyGate`."""

    UNSUPPORTED_CONTRACT = "unsupported representation/units/frame"
    RUN_GENERATION_MISMATCH = "run generation mismatch"
    INVALID_CURRENT_ARM_SHAPE = "invalid current arm joint state shape"
    NONFINITE_CURRENT_ARM = "current arm joint state contains NaN/Inf"
    INVALID_CANDIDATE_SHAPE = "invalid candidate joint shape"
    NONFINITE_CANDIDATE = "candidate contains NaN/Inf"
    ARM_JOINT_LIMIT = "arm joint limit violation"
    HAND_JOINT_LIMIT = "hand joint limit violation"
    WORKSPACE = "workspace"
    WORKSPACE_CHECK_FAILED = "workspace check failed"


@dataclass(frozen=True)
class GateResult:
    """Outcome of :meth:`SafetyGate.validate`."""

    accepted: bool
    code: GateRejectCode | None = None
    detail: str = ""

    @property
    def reason(self) -> str:
        """Human-readable rejection reason for logs and operator messages."""
        if self.detail:
            return self.detail
        return "" if self.code is None else self.code.value


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


def _publication_runtime_gate(
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
    last_cmd_qpos: np.ndarray
    last_cmd_seq: int


def _arm_feedback_snapshot(
    shared: Any,
    candidate: ActionCandidate | None,
) -> tuple[_ArmFeedbackSnapshot | None, CommandPublishResult | None]:
    """Read the arm fields required by publication and acknowledgement.

    Readiness is owned by the runtime gate (``is_running``/``error_state``/
    ``safety_state``), not by the arm frame: this only supplies the current
    joint positions for :meth:`SafetyGate.validate` and the acknowledgement
    wait, failing closed on a missing or malformed frame.
    """
    result = shared.arm_state_ring.read_latest()
    if result is None:
        return None, CommandPublishResult(
            CommandPublishStatus.ARM_FEEDBACK_UNAVAILABLE,
            candidate=candidate,
        )
    record = result[0][0]
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    if qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(qpos)):
        return None, CommandPublishResult(
            CommandPublishStatus.ARM_FEEDBACK_UNHEALTHY,
            candidate=candidate,
            detail="arm feedback has invalid shape or non-finite values",
        )
    return _ArmFeedbackSnapshot(qpos.copy(), int(record["last_cmd_seq"])), None


def _hand_feedback_snapshot(
    shared: Any,
    candidate: ActionCandidate | None,
    *,
    hand_feedback_max_age_s: float,
) -> tuple[_HandFeedbackSnapshot | None, CommandPublishResult | None]:
    """Read one fully healthy hand command/feedback snapshot fail-closed.

    Delegates the five health flags, source-timestamp existence, future
    timestamp, and ``max_age`` freshness to :func:`validate_hand_feedback`;
    the worker's last accepted command is then shape/finite-checked on its own
    (that predicate does not know about ``last_cmd_qpos``).
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
        error_state=bool(record["error_state"]),
        state_valid=bool(record["state_valid"]),
        send_healthy=bool(record["send_healthy"]),
        read_healthy=bool(record["read_healthy"]),
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
    last_cmd_qpos = np.asarray(record["last_cmd_qpos"], dtype=np.float64)
    if (
        last_cmd_qpos.shape != HAND_JOINT_SHAPE
        or not np.all(np.isfinite(last_cmd_qpos))
    ):
        return None, CommandPublishResult(
            CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY,
            candidate=candidate,
            detail="hand last accepted command is malformed",
        )
    return (
        _HandFeedbackSnapshot(
            last_cmd_qpos.copy(),
            int(record["last_cmd_seq"]),
        ),
        None,
    )


class SafetyGate:
    """Single validation boundary for all action paths.

    Pipeline (short-circuit, fail-closed):

    1. **Well-formed** — representation, shapes, finite values
    2. **Joint limits** — commanded actuators only; hold actuators skip
    3. **Workspace** — optional segment check

    xArm Mode 6 firmware enforces velocity, acceleration, and collision limits
    as the final backstop.  Collision-free homing paths are planned
    independently through ``plan_joint_home_path`` /
    ``plan_band_alignment_path``, which call the collision model directly
    rather than through the SafetyGate.

    The controller validates ``run_generation`` and the proposal validity
    window before publication.  Workers re-check generation and expiry before
    applying the resulting fixed command.  The gate never silently converts a
    rejected motion into a different joint-space endpoint.
    """

    def __init__(
        self,
        *,
        arm_joint_lower_rad: tuple[float, ...],
        arm_joint_upper_rad: tuple[float, ...],
        hand_joint_lower_rad: tuple[float, ...],
        hand_joint_upper_rad: tuple[float, ...],
    ) -> None:
        _arm_low = np.asarray(arm_joint_lower_rad, dtype=np.float64)
        _arm_high = np.asarray(arm_joint_upper_rad, dtype=np.float64)
        _hand_low = np.asarray(hand_joint_lower_rad, dtype=np.float64)
        _hand_high = np.asarray(hand_joint_upper_rad, dtype=np.float64)
        if _arm_low.shape != ARM_JOINT_SHAPE or _arm_high.shape != ARM_JOINT_SHAPE:
            raise ValueError("arm joint limits must have seven entries")
        if _hand_low.shape != HAND_JOINT_SHAPE or _hand_high.shape != HAND_JOINT_SHAPE:
            raise ValueError("hand joint limits must have twelve entries")
        concat = np.concatenate((_arm_low, _arm_high, _hand_low, _hand_high))
        if (
            not np.all(np.isfinite(concat))
            or np.any(_arm_low > _arm_high)
            or np.any(_hand_low > _hand_high)
        ):
            raise ValueError("joint limits must be finite and ordered")

        self.arm_low = _arm_low
        self.arm_high = _arm_high
        self.hand_low = _hand_low
        self.hand_high = _hand_high

    # -- callback set after construction (avoids circular imports) ----------
    workspace_check: Any = None  # Callable[[np.ndarray, np.ndarray], bool] | None

    def validate(
        self,
        candidate: ActionCandidate,
        *,
        current_arm_qpos: np.ndarray,
        run_generation: int,
    ) -> GateResult:
        """Run the full validation pipeline.

        Args:
            candidate: The proposed ``ActionCandidate``.
            current_arm_qpos: Latest measured arm joint positions [rad].
            run_generation: Expected control-run generation.

        Returns:
            A typed accept/reject result. The candidate is never modified.
        """
        # 1 ── Well-formed ────────────────────────────────────────────
        if (
            candidate.representation != "joint_position"
            or candidate.units != "rad"
            or candidate.frame != "robot_joint"
        ):
            return GateResult(False, GateRejectCode.UNSUPPORTED_CONTRACT)

        if candidate.run_generation != run_generation:
            return GateResult(False, GateRejectCode.RUN_GENERATION_MISMATCH)

        arm_start = np.asarray(current_arm_qpos, dtype=np.float64)
        if arm_start.shape != ARM_JOINT_SHAPE:
            return GateResult(False, GateRejectCode.INVALID_CURRENT_ARM_SHAPE)
        if not np.all(np.isfinite(arm_start)):
            return GateResult(False, GateRejectCode.NONFINITE_CURRENT_ARM)

        arm_end = (
            arm_start.copy()
            if candidate.arm_qpos is None
            else np.asarray(candidate.arm_qpos, dtype=np.float64).copy()
        )
        hand_end = (
            None
            if candidate.hand_qpos is None
            else np.asarray(candidate.hand_qpos, dtype=np.float64).copy()
        )
        if arm_end.shape != ARM_JOINT_SHAPE or (
            hand_end is not None and hand_end.shape != HAND_JOINT_SHAPE
        ):
            return GateResult(False, GateRejectCode.INVALID_CANDIDATE_SHAPE)
        if not np.all(np.isfinite(arm_end)) or (
            hand_end is not None and not np.all(np.isfinite(hand_end))
        ):
            return GateResult(False, GateRejectCode.NONFINITE_CANDIDATE)

        # 2 ── Joint limits (commanded actuators only) ─────────────────
        if candidate.arm_qpos is not None and (
            np.any(arm_end < self.arm_low) or np.any(arm_end > self.arm_high)
        ):
            return GateResult(False, GateRejectCode.ARM_JOINT_LIMIT)
        if hand_end is not None and (
            np.any(hand_end < self.hand_low - 1e-12)
            or np.any(hand_end > self.hand_high + 1e-12)
        ):
            return GateResult(False, GateRejectCode.HAND_JOINT_LIMIT)

        # 3 ── Workspace (optional) ────────────────────────────────────
        if self.workspace_check is not None and candidate.arm_qpos is not None:
            try:
                if not self.workspace_check(arm_start, arm_end):
                    return GateResult(False, GateRejectCode.WORKSPACE)
            except Exception:
                logger.warning(
                    "SafetyGate: workspace check failed closed", exc_info=True
                )
                return GateResult(False, GateRejectCode.WORKSPACE_CHECK_FAILED)

        return GateResult(True)


# Command serialization and publication.


def _make_arm_command(
    candidate: ActionCandidate, now_monotonic_ns: int
) -> np.ndarray:
    """Serialize an ActionCandidate into an ARM_COMMAND_DTYPE record."""
    if candidate.arm_qpos is None:
        raise ValueError("candidate has no arm command")
    frame = np.zeros(1, dtype=ARM_COMMAND_DTYPE)
    frame["action_id"][0] = candidate.action_id
    frame["created_monotonic_ns"][0] = now_monotonic_ns
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
    frame["valid_until_monotonic_ns"][0] = target_monotonic_ns + int(3e8)
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

    runtime_rejection = _publication_runtime_gate(shared)
    if runtime_rejection is not None:
        return CommandPublishResult(
            runtime_rejection.status,
            candidate=candidate,
            detail=runtime_rejection.detail,
        )

    now_ns = time.monotonic_ns()
    lead_time_s = float(getattr(shared, "action_lead_time_s", 0.05))
    target_ns = now_ns + int(lead_time_s * 1e9)

    # The candidate validity window protects the policy boundary.  The publish
    # boundary owns the worker delivery target because queueing adds latency.
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
        shared.arm_cmd_ring.write(_make_arm_command(candidate, now_ns))

    # Both actuator transports are latest-wins seqlock rings
    # (``SharedMemoryRingBuffer``): ``write`` overwrites the oldest slot and
    # returns the new sequence number, so there is no ``Full``/backpressure and
    # no error-return channel.  The arm-then-hand ordering is therefore
    # non-atomic by design, and no rollback is performed or claimed; if a
    # transport ever gains a failing write, a coordinated stop/fault path is
    # required here instead of returning ``PUBLISHED``.
    if candidate.hand_qpos is not None:
        hand_frame = _make_hand_command(candidate, now_ns, target_ns)
        shared.hand_cmd_ring.write(hand_frame)

    return CommandPublishResult(
        CommandPublishStatus.PUBLISHED,
        candidate=candidate,
    )


# Worker-side validation.


def _worker_command_is_current(
    command: np.ndarray,
    *,
    expected_run_generation: int | None,
    now_monotonic_ns: int | None,
) -> bool:
    """Validate the lifecycle metadata of a fixed hand command."""
    if expected_run_generation is not None and int(command["run_generation"][0]) != int(
        expected_run_generation
    ):
        return False
    if now_monotonic_ns is not None:
        valid_until_ns = int(command["valid_until_monotonic_ns"][0])
        if valid_until_ns <= 0 or int(now_monotonic_ns) > valid_until_ns:
            return False
    return True


def worker_validate_arm(
    command: np.ndarray,
    *,
    armed_at_seq: int = 0,
    now_monotonic_ns: int | None = None,
    max_command_age_s: float = 0.3,
) -> bool:
    """Minimal hardware-level check for an arm endpoint from the ring.

    The arm transport is latest-wins, so freshness is decided by two rules
    instead of a generation/expiry protocol: the endpoint must have been created
    after motion was armed (``action_id > armed_at_seq``), and it must not be
    older than ``max_command_age_s``.  The gate already validated limits and
    geometry.
    """
    if not (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == ARM_COMMAND_DTYPE
        and np.all(np.isfinite(command["qpos_cmd"][0]))
    ):
        return False
    if int(command["action_id"][0]) <= int(armed_at_seq):
        return False
    if now_monotonic_ns is not None:
        created_ns = int(command["created_monotonic_ns"][0])
        if created_ns <= 0:
            return False
        age_s = (int(now_monotonic_ns) - created_ns) * 1e-9
        if age_s < 0.0 or age_s > max_command_age_s:
            return False
    return True


def worker_validate_hand(
    command: np.ndarray,
    *,
    qpos_lower_rad: np.ndarray,
    qpos_upper_rad: np.ndarray,
    mechanical_lower_rad: np.ndarray,
    mechanical_upper_rad: np.ndarray,
    expected_run_generation: int | None = None,
    now_monotonic_ns: int | None = None,
) -> bool:
    """Hardware-boundary check for a hand command from the ring.

    Returns True when the command is well-formed, belongs to the active run,
    has not expired, and lies inside both operational and rated mechanical
    limits. These redundant checks protect direct/home publishers and IPC
    corruption; they never modify the endpoint.
    """
    command_lower = np.asarray(qpos_lower_rad, dtype=np.float64)
    command_upper = np.asarray(qpos_upper_rad, dtype=np.float64)
    mechanical_lower = np.asarray(mechanical_lower_rad, dtype=np.float64)
    mechanical_upper = np.asarray(mechanical_upper_rad, dtype=np.float64)
    rated_lower = np.asarray(hand_defaults.mechanical_qpos_min_rad, dtype=np.float64)
    rated_upper = np.asarray(hand_defaults.mechanical_qpos_max_rad, dtype=np.float64)
    limits_well_formed = (
        command_lower.shape == HAND_JOINT_SHAPE
        and command_upper.shape == HAND_JOINT_SHAPE
        and mechanical_lower.shape == HAND_JOINT_SHAPE
        and mechanical_upper.shape == HAND_JOINT_SHAPE
        and np.all(
            np.isfinite(
                np.concatenate(
                    (command_lower, command_upper, mechanical_lower, mechanical_upper)
                )
            )
        )
        and np.all(command_lower <= command_upper)
        and np.all(mechanical_lower <= mechanical_upper)
        and np.all(mechanical_lower >= rated_lower)
        and np.all(mechanical_upper <= rated_upper)
        and np.all(command_lower >= mechanical_lower)
        and np.all(command_upper <= mechanical_upper)
    )
    qpos_cmd = (
        np.asarray(command["qpos_cmd"][0], dtype=np.float64)
        if isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == HAND_COMMAND_DTYPE
        else np.empty(0, dtype=np.float64)
    )
    well_formed = (
        isinstance(command, np.ndarray)
        and command.shape == (1,)
        and command.dtype == HAND_COMMAND_DTYPE
        and qpos_cmd.shape == HAND_JOINT_SHAPE
        and np.all(np.isfinite(qpos_cmd))
        and limits_well_formed
        and np.all(qpos_cmd >= command_lower - 1e-12)
        and np.all(qpos_cmd <= command_upper + 1e-12)
        and np.all(qpos_cmd >= mechanical_lower - 1e-12)
        and np.all(qpos_cmd <= mechanical_upper + 1e-12)
    )
    return bool(
        well_formed
        and _worker_command_is_current(
            command,
            expected_run_generation=expected_run_generation,
            now_monotonic_ns=now_monotonic_ns,
        )
    )


# Gate factory.


def planner_action_safety_gate(
    *,
    planner: Any,
    arm_joint_lower_rad: tuple[float, ...],
    arm_joint_upper_rad: tuple[float, ...],
    hand_joint_lower_rad: tuple[float, ...],
    hand_joint_upper_rad: tuple[float, ...],
) -> SafetyGate:
    """Build a :class:`SafetyGate` wired to the planner's workspace callback."""
    gate = SafetyGate(
        arm_joint_lower_rad=arm_joint_lower_rad,
        arm_joint_upper_rad=arm_joint_upper_rad,
        hand_joint_lower_rad=hand_joint_lower_rad,
        hand_joint_upper_rad=hand_joint_upper_rad,
    )
    gate.workspace_check = planner.is_workspace_segment_safe
    return gate


# Convenience publication.


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
        hand_qpos=None if hand_qpos is None else np.asarray(hand_qpos, dtype=np.float64),
        is_hold=is_hold,
    )


def validate_and_send_candidate(
    shared: Any,
    candidate: ActionCandidate,
    *,
    gate: SafetyGate,
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
    runtime_rejection = _publication_runtime_gate(shared)
    if runtime_rejection is not None:
        return CommandPublishResult(
            runtime_rejection.status,
            candidate=candidate,
            detail=runtime_rejection.detail,
        )

    arm_feedback, feedback_rejection = _arm_feedback_snapshot(shared, candidate)
    if feedback_rejection is not None:
        logger.warning(
            "validate_and_send_candidate: action_id=%d rejected: %s",
            action_id,
            feedback_rejection.reason,
        )
        return feedback_rejection
    assert arm_feedback is not None

    gate_result = gate.validate(
        candidate,
        current_arm_qpos=arm_feedback.qpos,
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
        hand_feedback, feedback_rejection = _hand_feedback_snapshot(
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

    runtime_rejection = _publication_runtime_gate(shared)
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
        # A candidate carrying a hand_qpos is published to both actuators under
        # the same action_id (see send_command), so a synchronous caller must
        # observe BOTH applied before returning.  Arm-only candidates keep the
        # existing arm last_cmd_seq >= action_id gate.
        with_hand = published.hand_qpos is not None
        while time.monotonic() < deadline_s:
            runtime_rejection = _publication_runtime_gate(shared)
            if runtime_rejection is not None:
                return CommandPublishResult(
                    runtime_rejection.status,
                    candidate=published,
                    detail=runtime_rejection.detail,
                )
            arm_feedback, feedback_rejection = _arm_feedback_snapshot(shared, published)
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
                hand_feedback, feedback_rejection = _hand_feedback_snapshot(
                    shared, published, hand_feedback_max_age_s=hand_feedback_max_age_s
                )
                if feedback_rejection is not None:
                    return feedback_rejection
                assert hand_feedback is not None
                hand_seq = hand_feedback.last_cmd_seq
                if hand_seq > action_id:
                    logger.warning(
                        "publish_joint_targets: action_id=%d was superseded by hand action_id=%d",
                        action_id,
                        hand_seq,
                    )
                    return CommandPublishResult(
                        CommandPublishStatus.ACK_SUPERSEDED,
                        candidate=published,
                    )
                if arm_ok and hand_seq == action_id:
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


# Hand command bounds preflight.


def validate_hand_command_bounds(
    hand_cmd: np.ndarray,
    operational_lower: np.ndarray,
    operational_upper: np.ndarray,
    mechanical_lower: np.ndarray,
    mechanical_upper: np.ndarray,
) -> np.ndarray:
    """Validate one hand target against operational + rated mechanical bounds;
    reject-whole, never clip.

    Shared preflight for every coupled hand path (teleop, replay, return-home).
    Normal action producers reach it through ``validate_and_send_candidate``;
    hand-home also reuses it before publishing the exact home endpoint.  Raises
    ``ValueError`` on any violation and returns a copy otherwise.
    """
    command = np.asarray(hand_cmd, dtype=np.float64)
    op_lower = np.asarray(operational_lower, dtype=np.float64)
    op_upper = np.asarray(operational_upper, dtype=np.float64)
    mech_lower = np.asarray(mechanical_lower, dtype=np.float64)
    mech_upper = np.asarray(mechanical_upper, dtype=np.float64)
    rated_lower = np.asarray(hand_defaults.mechanical_qpos_min_rad, dtype=np.float64)
    rated_upper = np.asarray(hand_defaults.mechanical_qpos_max_rad, dtype=np.float64)

    if command.shape != HAND_JOINT_SHAPE:
        raise ValueError(
            f"hand command must have shape {HAND_JOINT_SHAPE}, got {command.shape}"
        )
    for label, value in (
        ("operational lower", op_lower),
        ("operational upper", op_upper),
        ("mechanical lower", mech_lower),
        ("mechanical upper", mech_upper),
    ):
        if value.shape != HAND_JOINT_SHAPE:
            raise ValueError(f"hand {label} limits must have shape {HAND_JOINT_SHAPE}")
    if not np.all(
        np.isfinite(np.concatenate((command, op_lower, op_upper, mech_lower, mech_upper)))
    ):
        raise ValueError("hand command and limit arrays must be finite")
    if np.any(op_lower > op_upper) or np.any(mech_lower > mech_upper):
        raise ValueError("hand operational and mechanical limits must be ordered")
    if np.any(mech_lower < rated_lower) or np.any(mech_upper > rated_upper):
        raise ValueError("hand mechanical limits cannot exceed the rated device envelope")
    if np.any(op_lower < mech_lower) or np.any(op_upper > mech_upper):
        raise ValueError("hand operational limits must be inside mechanical limits")
    if np.any(command < op_lower - 1e-12) or np.any(command > op_upper + 1e-12):
        raise ValueError("hand command violates operational joint limits")
    if np.any(command < mech_lower - 1e-12) or np.any(command > mech_upper + 1e-12):
        raise ValueError("hand command violates rated mechanical joint limits")

    return command.copy()


# Hand homing utility.


def publish_hand_home_and_wait_applied(
    shared: Any,
    home_qpos: np.ndarray,
    *,
    command_lower_rad: np.ndarray,
    command_upper_rad: np.ndarray,
    mechanical_lower_rad: np.ndarray,
    mechanical_upper_rad: np.ndarray,
    hand_feedback_max_age_s: float,
    timeout_s: float = 1.0,
    heartbeat: bool = False,
    check_is_running: bool = True,
    verbose: bool = True,
    abort_requested: Any = None,
) -> bool:
    """Publish exact hand-home and wait only for worker/SDK acceptance.

    The configured endpoint must lie inside both the operational command box
    and the rated mechanical box. Success means every SDK send, including the
    exact final home endpoint, was acknowledged. Measured qpos is deliberately
    not compared with the target because contact and steady-state position
    error are valid.
    """
    if not np.isfinite(timeout_s) or timeout_s <= 0.0:
        raise ValueError(
            "hand home command acknowledgement timeout must be finite and positive"
        )
    # Bound validation is shared with the other coupled hand paths; a
    # violation raises (reject-whole, never clip).
    target = validate_hand_command_bounds(
        home_qpos,
        command_lower_rad,
        command_upper_rad,
        mechanical_lower_rad,
        mechanical_upper_rad,
    )
    runtime_rejection = _publication_runtime_gate(
        shared,
        check_is_running=check_is_running,
    )
    if runtime_rejection is not None:
        logger.warning("hand home rejected by runtime gate: %s", runtime_rejection.reason)
        return False
    command_lower = np.asarray(command_lower_rad, dtype=np.float64)
    command_upper = np.asarray(command_upper_rad, dtype=np.float64)
    deadline_s = time.monotonic() + timeout_s
    hand_feedback, feedback_rejection = _hand_feedback_snapshot(
        shared, None, hand_feedback_max_age_s=hand_feedback_max_age_s
    )
    if feedback_rejection is not None:
        logger.warning("hand home rejected: %s", feedback_rejection.reason)
        return False
    assert hand_feedback is not None
    start = hand_feedback.last_cmd_qpos
    if np.any(start < command_lower - 1e-12) or np.any(start > command_upper + 1e-12):
        logger.warning(
            "hand home rejected: last accepted hand command violates operational limits"
        )
        return False

    # The exact home endpoint is published as a single command, not spread
    # over milestones.
    milestone_count = 1

    last_action_id = 0
    acknowledged = False
    for milestone_index in range(1, milestone_count + 1):
        if time.monotonic() >= deadline_s:
            break
        runtime_rejection = _publication_runtime_gate(
            shared,
            check_is_running=check_is_running,
        )
        if runtime_rejection is not None:
            logger.warning("hand home stopped by runtime gate: %s", runtime_rejection.reason)
            return False
        milestone = target.copy()

        with shared.arm_command_seq.get_lock():
            action_id = int(shared.arm_command_seq.value) + 1
            shared.arm_command_seq.value = action_id
        last_action_id = action_id
        now_ns = time.monotonic_ns()
        frame = np.zeros(1, dtype=HAND_COMMAND_DTYPE)
        frame["run_generation"][0] = int(shared.run_generation.value)
        frame["observation_id"][0] = action_id
        frame["action_id"][0] = action_id
        frame["created_monotonic_ns"][0] = now_ns
        frame["target_monotonic_ns"][0] = now_ns
        frame["valid_until_monotonic_ns"][0] = now_ns + int(
            max(0.3, deadline_s - time.monotonic() + 0.1) * 1e9
        )
        frame["is_hold"][0] = 0
        frame["qpos_cmd"][0] = milestone
        shared.hand_cmd_ring.write(frame)

        acknowledged = False
        while time.monotonic() < deadline_s:
            if abort_requested is not None and abort_requested():
                return False
            if _publication_runtime_gate(
                shared,
                check_is_running=check_is_running,
            ) is not None:
                return False
            if heartbeat:
                shared.set_heartbeat("policy", time.monotonic())

            hand_feedback, feedback_rejection = _hand_feedback_snapshot(
                shared, None, hand_feedback_max_age_s=hand_feedback_max_age_s
            )
            if feedback_rejection is not None:
                logger.warning("hand home acknowledgement stopped: %s", feedback_rejection.reason)
                return False
            assert hand_feedback is not None
            applied_id = hand_feedback.last_cmd_seq
            if applied_id > action_id:
                logger.warning(
                    "hand home action_id=%d was superseded by action_id=%d before acknowledgement",
                    action_id,
                    applied_id,
                )
                return False
            if applied_id == action_id:
                acknowledged = True
                break
            time.sleep(0.01)
        if not acknowledged:
            break

    if last_action_id and milestone_index == milestone_count and acknowledged:
        if verbose:
            print(
                f"  hand: home command accepted (action_id={last_action_id})",
                flush=True,
            )
        return True

    logger.warning(
        "hand home action_id=%d was not acknowledged within %.3fs",
        last_action_id,
        timeout_s,
    )
    return False

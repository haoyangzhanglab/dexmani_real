"""Deployment coordinator — the sole learned-policy robot-action producer.

The inference worker writes proposals to ``policy_chunk_ring``; this coordinator
is the only process that turns a proposal into a robot command. It selects the
chunk, schedules the due endpoint (one per control tick), runs the shared
candidate publication boundary (SafetyGate -> send_command), and owns the
policy semantic watchdog and the ``RUNNING <-> ARMED`` control-source state.

It never dumps a whole chunk into the arm queue or hand ring and never
interpolates between model steps.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.defaults import arm as arm_defaults
from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.config.defaults import policy as policy_defaults
from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    CommandPublishResult,
    CommandPublishStatus,
    build_action_candidate,
    poll_coupled_command_acknowledgement,
    validate_and_send_candidate,
)
from dexmani_real.control.safety_gate import GateRejectCode, planner_action_safety_gate
from dexmani_real.deployment.config import PolicyDeploymentConfig
from dexmani_real.deployment.contracts import ActionChunk
from dexmani_real.deployment.metrics import (
    ACK_FAILURE,
    ACK_LATENCY_MS,
    ACK_TIMEOUT,
    ACKNOWLEDGED,
    APPLIED_ACTION_STEPS,
    CHUNKS_GENERATION_DROPPED,
    CHUNKS_INGESTED,
    CHUNKS_STALE,
    COMMAND_SILENCE_ABORT,
    ENDPOINTS_COMMITTED,
    ENDPOINTS_DUE,
    ENDPOINTS_FATAL_REJECTED,
    ENDPOINTS_MOTION_DISCARDED,
    ENDPOINTS_PUBLISHED,
    ENDPOINTS_STALE_DISCARDED,
    ENDPOINTS_TRANSIENT_DEFERRED,
    ENDPOINTS_VALIDATED,
    EPISODE_ACTION_STEPS,
    HAND_POLICY_ENDPOINT_ROUNDOFF_CANONICALIZED,
    HAND_PREFLIGHT_REJECTIONS,
    IK_CHECKER_REJECTS,
    POLICY_ABORTS,
    SAFETY_REJECTED_STEPS,
    Metrics,
    flush_every,
    reject_counter_name,
)
from dexmani_real.deployment.timing import first_future_step_index
from dexmani_real.ipc.channels import RuntimeChannels, read_arm_state_dict
from dexmani_real.ipc.schema import MAX_POLICY_CHUNK_STEPS, POLICY_CHUNK_DTYPE
from dexmani_real.planning import (
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.paths import wrap_nearest_equivalent
from dexmani_real.planning.poses import rot6d_to_quat_wxyz, validate_rot6d_geometry
from dexmani_real.planning.types import IKFailureKind
from dexmani_real.robot_spec import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.runtime.safety import (
    CoupledCommandTicket,
    SafetyState,
    StopRequest,
    begin_requested_motion,
    revoke_motion,
)
from dexmani_real.utils.feedback import FeedbackIssueCode, diagnose_arm_feedback
from dexmani_real.utils.limits import validate_hand_limit_nesting
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.config.runtime import ResolvedRuntimeConfig

logger = get_logger(__name__)
_TARGET_WAIT_HEARTBEAT_S = 0.05
_UINT64_MAX = int(np.iinfo(np.uint64).max)


class PolicyEndpointDisposition(str, Enum):
    """Coordinator action for one typed policy endpoint result."""

    COMMIT = "commit"
    DISCARD_MOTION = "discard_motion"
    DEFER_TRANSIENT = "defer_transient"
    DISCARD_STALE = "discard_stale"
    ABORT_FATAL = "abort_fatal"


def classify_policy_endpoint_disposition(
    result: CommandPublishResult,
    *,
    hand_limit_nesting_valid: bool,
) -> PolicyEndpointDisposition:
    """Classify a publication result without interpreting diagnostic strings.

    The policy coordinator owns what a typed gate/transport outcome means for
    one scheduled endpoint.  Existing teleop callers retain their own
    behavior by not using this classifier.
    """
    if result.succeeded:
        return PolicyEndpointDisposition.COMMIT
    if result.status is CommandPublishStatus.TEMPORAL_WINDOW_CLOSED:
        return PolicyEndpointDisposition.DISCARD_STALE
    if result.status is CommandPublishStatus.HAND_PREFLIGHT_REJECTED:
        return (
            PolicyEndpointDisposition.DISCARD_MOTION
            if hand_limit_nesting_valid
            else PolicyEndpointDisposition.ABORT_FATAL
        )
    if result.status is CommandPublishStatus.GATE_REJECTED:
        if result.gate_code in {
            GateRejectCode.ARM_JOINT_LIMIT,
            GateRejectCode.HAND_JOINT_LIMIT,
            GateRejectCode.ARM_DELTA_LIMIT,
            GateRejectCode.WORKSPACE,
            GateRejectCode.COLLISION_TRANSITION,
        }:
            return PolicyEndpointDisposition.DISCARD_MOTION
        if result.gate_code is GateRejectCode.RUN_GENERATION_MISMATCH:
            return PolicyEndpointDisposition.DEFER_TRANSIENT
        # HAND_DELTA_LIMIT must never occur for learned policy (the coordinator
        # disables this gate); all remaining gate/contract/checker failures are
        # unsafe implementation or input faults.
        return PolicyEndpointDisposition.ABORT_FATAL
    if result.status in {
        CommandPublishStatus.ARM_FEEDBACK_UNAVAILABLE,
        CommandPublishStatus.HAND_FEEDBACK_UNAVAILABLE,
    }:
        return PolicyEndpointDisposition.DEFER_TRANSIENT
    if result.status in {
        CommandPublishStatus.ARM_FEEDBACK_UNHEALTHY,
        CommandPublishStatus.HAND_FEEDBACK_UNHEALTHY,
    }:
        if (
            result.feedback_issue is not None
            and result.feedback_issue.code is FeedbackIssueCode.STALE
        ):
            return PolicyEndpointDisposition.DEFER_TRANSIENT
        return PolicyEndpointDisposition.ABORT_FATAL
    if result.status in {
        CommandPublishStatus.RUNTIME_STOPPED,
        CommandPublishStatus.SAFETY_STATE_GATED,
        CommandPublishStatus.RUN_GENERATION_GATED,
    }:
        return PolicyEndpointDisposition.DEFER_TRANSIENT
    # ESTOP/STICKY are fatal but the coordinator leaves their lifecycle state
    # untouched. Missing gate, malformed candidates, future feedback time,
    # acknowledgement failures, and all unknown outcomes fail closed.
    return PolicyEndpointDisposition.ABORT_FATAL


@dataclass(frozen=True)
class _PendingAcknowledgement:
    """One physical command awaiting arm and hand worker acknowledgement."""

    candidate: ActionCandidate
    ticket: CoupledCommandTicket
    published_monotonic_ns: int
    acceptance_deadline_monotonic_ns: int
    observation_deadline_monotonic_ns: int


class _AcknowledgementAction(str, Enum):
    """One bounded coordinator action after polling a pending command."""

    APPLIED = "applied"
    WAIT = "wait"
    FAULT_TIMEOUT = "fault_timeout"
    FAULT_REJECTED = "fault_rejected"


@dataclass(frozen=True)
class _AcknowledgementDecision:
    """Pure interpretation of one arm/hand acknowledgement observation."""

    action: _AcknowledgementAction
    reason: str = ""
    latency_ms: float | None = None


@dataclass
class _EpisodeActionSteps:
    """Pure per-episode terminal policy-step accounting."""

    max_action_steps: int | None
    episode_action_steps: int = 0
    applied_action_steps: int = 0
    safety_rejected_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_action_steps is not None and (
            type(self.max_action_steps) is not int or self.max_action_steps <= 0
        ):
            raise ValueError("max_action_steps must be a positive integer or null")

    def reset(self) -> None:
        self.episode_action_steps = 0
        self.applied_action_steps = 0
        self.safety_rejected_steps = 0

    def record_applied(self) -> bool:
        self.episode_action_steps += 1
        self.applied_action_steps += 1
        return self.limit_reached

    def record_safety_rejected(self) -> bool:
        self.episode_action_steps += 1
        self.safety_rejected_steps += 1
        return self.limit_reached

    @property
    def limit_reached(self) -> bool:
        return (
            self.max_action_steps is not None
            and self.episode_action_steps >= self.max_action_steps
        )


def _record_terminal_before_finalize(
    record_terminal: Callable[[], bool],
    finalize_terminal: Callable[[], None],
) -> bool:
    """Record/check the limit before any cursor advance or sync request."""
    if record_terminal():
        return True
    finalize_terminal()
    return False


@dataclass
class _SyncExecution:
    """Minimal mutable cursor for one sequential sync chunk."""

    chunk: ActionChunk
    chunk_start_monotonic_ns: int
    step_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, ActionChunk):
            raise TypeError("sync execution requires an ActionChunk")
        if self.chunk_start_monotonic_ns <= 0:
            raise ValueError("sync chunk start must be positive")
        if not 0 <= self.step_index < self.chunk.num_steps:
            raise ValueError("sync step index is outside the chunk")

    def scheduled_target_ns(self, step_dt_ns: int) -> int:
        if step_dt_ns <= 0:
            raise ValueError("sync step_dt_ns must be positive")
        return self.chunk_start_monotonic_ns + self.step_index * int(step_dt_ns)

    def finalize_current(self) -> bool:
        """Advance once and return whether the complete chunk is finalized."""
        self.step_index += 1
        return self.step_index >= self.chunk.num_steps


@dataclass
class _AsyncExecution:
    """Minimal mutable cursor on one absolute logical ActionChunk timeline."""

    chunk: ActionChunk
    step_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, ActionChunk):
            raise TypeError("async execution requires an ActionChunk")
        if not 0 <= self.step_index < self.chunk.num_steps:
            raise ValueError("async step index is outside the chunk")

    def scheduled_target_ns(self, step_dt_ns: int) -> int:
        if step_dt_ns <= 0:
            raise ValueError("async step_dt_ns must be positive")
        return self.chunk.observation_logical_step_monotonic_ns + self.step_index * int(
            step_dt_ns
        )

    def advance_to_first_future(self, *, now_ns: int, step_dt_ns: int) -> bool:
        """Keep a selected target due until its next logical boundary."""
        target_ns = self.scheduled_target_ns(step_dt_ns)
        if now_ns < target_ns or _selected_async_target_is_due(
            target_ns=target_ns,
            step_dt_ns=step_dt_ns,
            now_ns=now_ns,
        ):
            return True
        return self.align_to_first_future(now_ns=now_ns, step_dt_ns=step_dt_ns)

    def align_to_first_future(self, *, now_ns: int, step_dt_ns: int) -> bool:
        """Strictly skip every target before now."""
        first_index = first_future_step_index(
            self.chunk.observation_logical_step_monotonic_ns,
            step_dt_ns,
            now_ns,
            self.chunk.num_steps,
        )
        if first_index is None:
            return False
        self.step_index = max(self.step_index, first_index)
        return self.step_index < self.chunk.num_steps

    def finalize_current(self) -> bool:
        self.step_index += 1
        return self.step_index >= self.chunk.num_steps


def _newer_async_execution(
    current: _AsyncExecution | None,
    incoming: ActionChunk,
    *,
    run_generation: int,
    now_ns: int,
    step_dt_ns: int,
) -> _AsyncExecution | None:
    """Return a usable newer chunk execution, otherwise leave current unchanged."""
    if incoming.run_generation != run_generation:
        return current
    if current is not None and incoming.chunk_id <= current.chunk.chunk_id:
        return current
    first_index = first_future_step_index(
        incoming.observation_logical_step_monotonic_ns,
        step_dt_ns,
        now_ns,
        incoming.num_steps,
    )
    if first_index is None:
        return current
    return _AsyncExecution(incoming, first_index)


def _selected_async_target_is_due(
    *,
    target_ns: int,
    step_dt_ns: int,
    now_ns: int,
) -> bool:
    """Keep a target selected before sleep due until its next grid boundary."""
    if target_ns <= 0 or step_dt_ns <= 0 or now_ns <= 0:
        raise ValueError("async target times must be positive")
    return target_ns <= now_ns < target_ns + step_dt_ns


def _chunk_source_is_stale(
    chunk: ActionChunk,
    *,
    now_monotonic_ns: int,
    max_source_age_ns: int,
) -> bool:
    """Return whether source freshness has closed before endpoint publication."""
    if now_monotonic_ns <= 0 or max_source_age_ns <= 0:
        raise ValueError("source freshness times must be positive")
    return (
        now_monotonic_ns - chunk.observation_latest_source_monotonic_ns
        > max_source_age_ns
    )


def _chunk_source_deadline_ns(
    chunk: ActionChunk,
    *,
    max_source_age_ns: int,
) -> int:
    """Return the immutable source-freshness deadline without uint64 overflow."""
    if not isinstance(chunk, ActionChunk):
        raise TypeError("source deadline requires an ActionChunk")
    if isinstance(max_source_age_ns, (bool, np.bool_)) or not isinstance(
        max_source_age_ns, (int, np.integer)
    ):
        raise TypeError("max_source_age_ns must be an integer")
    max_age_ns = int(max_source_age_ns)
    if max_age_ns <= 0:
        raise ValueError("max_source_age_ns must be positive")
    source_ns = chunk.observation_latest_source_monotonic_ns
    if source_ns > _UINT64_MAX - max_age_ns:
        raise ValueError("source freshness deadline exceeds uint64")
    return source_ns + max_age_ns


def _classify_acknowledgement(
    pending: _PendingAcknowledgement,
    acknowledgement: CommandPublishResult,
    *,
    poll_started_monotonic_ns: int,
    observed_monotonic_ns: int,
) -> _AcknowledgementDecision:
    """Map one non-blocking ACK poll to the coordinator's next action."""
    acknowledgement_status = acknowledgement.status
    if acknowledgement_status is CommandPublishStatus.APPLIED:
        accepted_times = {
            "arm": acknowledgement.arm_accepted_monotonic_ns,
        }
        if pending.candidate.hand_qpos is not None:
            accepted_times["hand"] = acknowledgement.hand_accepted_monotonic_ns
        if any(value is None or value <= 0 for value in accepted_times.values()):
            return _AcknowledgementDecision(
                _AcknowledgementAction.FAULT_REJECTED,
                reason="applied acknowledgement omitted worker acceptance time",
            )
        if any(
            value is not None
            and (
                value < pending.published_monotonic_ns
                or value > observed_monotonic_ns
            )
            for value in accepted_times.values()
        ):
            return _AcknowledgementDecision(
                _AcknowledgementAction.FAULT_REJECTED,
                reason="worker acceptance timestamp is outside the observed interval",
            )
        late_workers = [
            name
            for name, value in accepted_times.items()
            if value is not None
            and value > pending.acceptance_deadline_monotonic_ns
        ]
        if late_workers:
            return _AcknowledgementDecision(
                _AcknowledgementAction.FAULT_TIMEOUT,
                reason=(
                    f"{'/'.join(late_workers)} worker accepted action after deadline"
                ),
            )
        accepted_monotonic_ns = max(
            value for value in accepted_times.values() if value is not None
        )
        return _AcknowledgementDecision(
            _AcknowledgementAction.APPLIED,
            latency_ms=(accepted_monotonic_ns - pending.published_monotonic_ns)
            / 1e6,
        )
    if acknowledgement_status is CommandPublishStatus.ACK_PENDING:
        if poll_started_monotonic_ns < pending.observation_deadline_monotonic_ns:
            return _AcknowledgementDecision(_AcknowledgementAction.WAIT)
        return _AcknowledgementDecision(
            _AcknowledgementAction.FAULT_TIMEOUT,
            reason=f"worker acknowledgement timeout: {acknowledgement.detail}",
        )
    return _AcknowledgementDecision(
        _AcknowledgementAction.FAULT_REJECTED,
        reason=f"arm/hand acknowledgement failed: {acknowledgement_status.value}",
    )


def _command_watchdog_abort_reason(
    *,
    now_monotonic_ns: int,
    run_started_monotonic_ns: int | None,
    last_valid_command_monotonic_ns: int | None,
    first_command_timeout_ns: int,
    command_silence_timeout_ns: int,
) -> str | None:
    """Return the active command watchdog failure without reading shared state."""
    if last_valid_command_monotonic_ns is None:
        if (
            run_started_monotonic_ns is not None
            and now_monotonic_ns - run_started_monotonic_ns > first_command_timeout_ns
        ):
            return "first command timeout"
        return None
    if now_monotonic_ns - last_valid_command_monotonic_ns > command_silence_timeout_ns:
        return "command silence timeout"
    return None


@dataclass(frozen=True)
class CoordinatorConfig:
    """Real-owned timing, safety, and limits required by the coordinator."""

    arm_joint_lower_rad: tuple[float, ...]
    arm_joint_upper_rad: tuple[float, ...]
    workspace_bounds: tuple[
        tuple[float, float], tuple[float, float], tuple[float, float]
    ]
    hand_joint_lower_rad: tuple[float, ...]
    hand_joint_upper_rad: tuple[float, ...]
    hand_mechanical_lower_rad: tuple[float, ...]
    hand_mechanical_upper_rad: tuple[float, ...]
    arm_feedback_max_age_s: float
    hand_feedback_max_age_s: float
    control_hz: float
    execute: bool
    max_source_to_command_age_s: float
    max_command_silence_s: float
    action_validity_s: float
    command_acknowledgement_timeout_s: float
    first_command_timeout_s: float
    inference_mode: str = "sync"
    max_action_steps: int | None = None
    # Full 19-DoF collision model (hand + static boxes) for EE->IK and the
    # transition collision gate; table clearance is not part of the policy
    # safety gate.
    static_boxes: tuple = ()
    ik_max_pose_error_pos_m: float = 0.008
    ik_max_pose_error_rot_rad: float = 0.08
    # The arm worker's canonical final command-jump bound. The learned hand
    # path deliberately disables this gate: its worker retains the measured
    # 0.3-rad/tick ramp.
    arm_max_delta_rad_per_tick: float | None = arm_defaults.max_servo_command_jump_rad
    hand_max_delta_rad_per_tick: float | None = None
    # The learned hand command is shaped from fresh feedback before IPC. This
    # is deliberately separate from the policy SafetyGate's disabled hand
    # delta rejection: the worker can exactly accept the shaped command even
    # when object contact prevents the raw absolute endpoint from converging.
    hand_command_max_delta_rad_per_tick: float = (
        hand_defaults.hand_max_delta_rad_per_tick
    )
    endpoint_delta_tolerance_rad: float = policy_defaults.endpoint_delta_tolerance_rad
    required_start_arm_qpos: tuple[float, ...] | None = None
    start_arm_home_tolerance_rad: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.execute, bool):
            raise TypeError("execute must be a boolean")
        if self.inference_mode not in {"sync", "async"}:
            raise ValueError("inference_mode must be 'sync' or 'async'")
        if self.max_action_steps is not None and (
            type(self.max_action_steps) is not int or self.max_action_steps <= 0
        ):
            raise ValueError("max_action_steps must be a positive integer or null")
        timing_values = (
            self.max_source_to_command_age_s,
            self.max_command_silence_s,
            self.action_validity_s,
            self.command_acknowledgement_timeout_s,
            self.first_command_timeout_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in timing_values):
            raise ValueError("coordinator timing values must be finite and positive")
        if self.command_acknowledgement_timeout_s > self.action_validity_s:
            raise ValueError(
                "command acknowledgement timeout must be positive and no greater "
                "than action validity"
            )
        if self.execute:
            if self.required_start_arm_qpos is None:
                raise ValueError("physical publication requires canonical arm home")
            qpos = np.asarray(self.required_start_arm_qpos, dtype=np.float64)
            if qpos.shape != (7,) or not np.all(np.isfinite(qpos)):
                raise ValueError("required_start_arm_qpos must be a finite 7-vector")
            tolerance = self.start_arm_home_tolerance_rad
            if tolerance is None or not np.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(
                    "physical start home tolerance must be finite and positive"
                )
        if (
            not np.isfinite(self.hand_command_max_delta_rad_per_tick)
            or self.hand_command_max_delta_rad_per_tick <= 0.0
        ):
            raise ValueError(
                "hand command max delta per tick must be finite and positive"
            )

    @classmethod
    def from_runtime(
        cls,
        runtime: "ResolvedRuntimeConfig",
        *,
        execute: bool,
        deployment_config: PolicyDeploymentConfig | None = None,
    ) -> "CoordinatorConfig":
        deployment = deployment_config or PolicyDeploymentConfig()
        return cls(
            arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
            workspace_bounds=runtime.policy.workspace.as_tuple(),
            hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
            hand_mechanical_lower_rad=tuple(runtime.hand.mechanical_qpos_min_rad),
            hand_mechanical_upper_rad=tuple(runtime.hand.mechanical_qpos_max_rad),
            arm_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["arm"]),
            hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
            control_hz=float(runtime.policy.control_hz),
            execute=execute,
            max_source_to_command_age_s=float(
                runtime.policy.max_source_to_command_age_s
            ),
            max_command_silence_s=float(runtime.policy.max_command_silence_s),
            action_validity_s=float(runtime.policy.action_validity_s),
            command_acknowledgement_timeout_s=float(
                runtime.policy.command_acknowledgement_timeout_s
            ),
            first_command_timeout_s=float(runtime.policy.first_command_timeout_s),
            inference_mode=deployment.inference_mode,
            max_action_steps=deployment.max_action_steps,
            static_boxes=tuple(runtime.environment.static_boxes),
            ik_max_pose_error_pos_m=float(runtime.policy.ik_max_pose_error_pos_m),
            ik_max_pose_error_rot_rad=float(runtime.policy.ik_max_pose_error_rot_rad),
            arm_max_delta_rad_per_tick=float(runtime.arm.max_servo_command_jump_rad),
            hand_max_delta_rad_per_tick=None,
            hand_command_max_delta_rad_per_tick=float(
                runtime.hand.hand_max_delta_rad_per_tick
            ),
            endpoint_delta_tolerance_rad=float(
                runtime.policy.endpoint_delta_tolerance_rad
            ),
            required_start_arm_qpos=(tuple(runtime.arm.home_qpos) if execute else None),
            start_arm_home_tolerance_rad=(
                float(runtime.arm.homing.convergence_rad) if execute else None
            ),
        )


def action_chunk_from_record(rec: np.void) -> ActionChunk:
    """Deserialize and ownership-copy one exact ActionChunk IPC record."""
    if not isinstance(rec, np.void) or rec.dtype != POLICY_CHUNK_DTYPE:
        raise ValueError("policy chunk record has an invalid IPC schema")
    n = int(rec["num_steps"])
    if not 0 < n <= MAX_POLICY_CHUNK_STEPS:
        raise ValueError("policy chunk has an invalid num_steps")
    presence: dict[str, bool] = {}
    for name in ("arm_present", "ee_present", "hand_present"):
        value = int(rec[name])
        if value not in (0, 1):
            raise ValueError(f"policy chunk {name} must be 0 or 1")
        presence[name] = bool(value)
    return ActionChunk(
        chunk_id=int(rec["chunk_id"]),
        run_generation=int(rec["run_generation"]),
        observation_id=int(rec["observation_id"]),
        observation_anchor_monotonic_ns=int(rec["observation_anchor_monotonic_ns"]),
        observation_latest_source_monotonic_ns=int(
            rec["observation_latest_source_monotonic_ns"]
        ),
        observation_logical_step_monotonic_ns=int(
            rec["observation_logical_step_monotonic_ns"]
        ),
        inference_started_monotonic_ns=int(rec["inference_started_monotonic_ns"]),
        inference_finished_monotonic_ns=int(rec["inference_finished_monotonic_ns"]),
        num_steps=n,
        arm_present=presence["arm_present"],
        ee_present=presence["ee_present"],
        hand_present=presence["hand_present"],
        arm_qpos=(
            np.array(rec["arm_qpos"][:n], dtype=np.float64, copy=True)
            if presence["arm_present"]
            else None
        ),
        hand_qpos=(
            np.array(rec["hand_qpos"][:n], dtype=np.float64, copy=True)
            if presence["hand_present"]
            else None
        ),
        ee_pos=(
            np.array(rec["ee_pos"][:n], dtype=np.float64, copy=True)
            if presence["ee_present"]
            else None
        ),
        ee_rot6d=(
            np.array(rec["ee_rot6d"][:n], dtype=np.float64, copy=True)
            if presence["ee_present"]
            else None
        ),
    )


def read_latest_action_chunk(shared: RuntimeChannels) -> ActionChunk | None:
    """Read and deserialize the newest parallel ActionChunk transport value."""
    result = shared.policy_chunk_ring.read_latest()
    if result is None:
        return None
    return action_chunk_from_record(result[0][0])


def _physical_start_pose_rejection(
    shared: RuntimeChannels,
    config: CoordinatorConfig,
) -> str | None:
    """Return why B cannot open a physical epoch, or ``None`` at arm home."""
    if not config.execute:
        return None
    if not bool(shared.physical_home_completed.value):
        return (
            "physical home sequence has not completed for the next episode; "
            "press H before B"
        )
    arm_state = read_arm_state_dict(shared)
    if arm_state is None:
        return "arm feedback unavailable; press H after feedback is ready"
    try:
        issue = diagnose_arm_feedback(
            connected=bool(arm_state["connected"]),
            error_code=int(arm_state["error_code"]),
            state_valid=bool(arm_state["state_valid"]),
            source_monotonic_ns=int(arm_state["source_monotonic_ns"]),
            now_monotonic_ns=time.monotonic_ns(),
            max_age_s=config.arm_feedback_max_age_s,
            qpos=np.asarray(arm_state["qpos"], dtype=np.float64),
            qvel=np.asarray(arm_state["qvel"], dtype=np.float64),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return f"malformed arm feedback ({type(exc).__name__}); press H after recovery"
    if issue is not None:
        return f"arm feedback unhealthy ({issue.detail}); press H after recovery"
    assert config.required_start_arm_qpos is not None
    assert config.start_arm_home_tolerance_rad is not None
    current = np.asarray(arm_state["qpos"], dtype=np.float64)
    home = np.asarray(config.required_start_arm_qpos, dtype=np.float64)
    delta = current - home
    max_abs_delta = float(np.max(np.abs(delta)))
    if max_abs_delta <= config.start_arm_home_tolerance_rad:
        return None
    return (
        "arm is not at the training start pose; press H before B: "
        f"current_rad={current.tolist()} home_rad={home.tolist()} "
        f"delta_rad={delta.tolist()} max_abs_delta_rad={max_abs_delta:.9f} "
        f"tolerance_rad={config.start_arm_home_tolerance_rad:.9f}"
    )


def _end_policy_run(
    shared: RuntimeChannels,
    reason: str,
    *,
    abort: bool,
    metrics: Metrics | None = None,
    metric: str | None = None,
    summary_status: str | None = None,
) -> None:
    """End one policy run after ensuring motion is fenced in ARMED.

    Both a clean operator STOP (``abort=False``) and a policy-semantic abort
    (``abort=True``) leave the robot ARMED (command quiescence), never FAULT.
    The keyboard may already have revoked RUNNING before the coordinator sees
    its stop request; that ARMED state is complete and must not bump generation
    a second time.
    Abort counters are flushed immediately because the success-path
    ``flush_every`` is never reached once a run ends (the loop idles in ARMED).
    """
    lifecycle_faulted = (
        bool(shared.error_state.value)
        or bool(shared.estop_request.value)
        or int(shared.safety_state.value) == int(SafetyState.FAULT)
    )
    shared.physical_home_completed.value = False
    if not lifecycle_faulted:
        state = int(shared.safety_state.value)
        if state == int(SafetyState.RUNNING):
            if not revoke_motion(shared, SafetyState.ARMED):
                logger.error(
                    "coordinator: failed to transition RUNNING->ARMED (%s)", reason
                )
        elif state != int(SafetyState.ARMED):
            logger.error(
                "coordinator: cannot finish policy run from safety_state=%d (%s)",
                state,
                reason,
            )
    if abort:
        logger.warning("coordinator: policy run aborted: %s", reason)
        if metrics is not None:
            metrics.increment(POLICY_ABORTS)
            if metric is not None:
                metrics.increment(metric)
            metrics.flush(prefix="coordinator metrics")
            metrics.log_episode_summary(
                status=summary_status or "ABORTED",
                reason=reason,
            )
    else:
        logger.info("coordinator: policy run stopped: %s", reason)
        if metrics is not None:
            metrics.log_episode_summary(
                status=summary_status or "STOPPED",
                reason=reason,
            )


def coordinator_loop(shared: RuntimeChannels, config: CoordinatorConfig) -> None:
    """Coordinator process entry point — the only robot-action producer.

    Idles in ARMED until the operator presses B (``start_request``), runs one
    policy episode in RUNNING, then returns to ARMED on S (``stop_request``) or
    a policy-semantic abort.  Each B advances the run generation, so any
    in-flight chunk or command from a previous run is invalid at the worker.
    """
    if config is None:
        raise ValueError("coordinator_loop requires a CoordinatorConfig")
    validate_hand_limit_nesting(
        config.hand_joint_lower_rad,
        config.hand_joint_upper_rad,
        config.hand_mechanical_lower_rad,
        config.hand_mechanical_upper_rad,
        hand_defaults.mechanical_qpos_min_rad,
        hand_defaults.mechanical_qpos_max_rad,
        label="coordinator hand",
    )

    # Deployment uses the arm-base workspace gate + the full 19-DoF collision
    # model (hand + static boxes) so EE->IK and the transition collision gate
    # see the hand geometry.  Table clearance is not part of the policy gate.
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
            srdf_path=str(XARM7_XHAND_SRDF_PATH),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=np.asarray(config.workspace_bounds, dtype=np.float64),
        ),
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=config.ik_max_pose_error_pos_m,
            max_pose_error_rot_rad=config.ik_max_pose_error_rot_rad,
        ),
        hand_dof=True,
        static_boxes=config.static_boxes,
    )
    gate = planner_action_safety_gate(
        planner=planner,
        arm_joint_lower_rad=config.arm_joint_lower_rad,
        arm_joint_upper_rad=config.arm_joint_upper_rad,
        hand_joint_lower_rad=config.hand_joint_lower_rad,
        hand_joint_upper_rad=config.hand_joint_upper_rad,
        max_arm_delta_rad=config.arm_max_delta_rad_per_tick,
        max_hand_delta_rad=config.hand_max_delta_rad_per_tick,
        endpoint_delta_tolerance_rad=config.endpoint_delta_tolerance_rad,
        collision_check=planner.collision_model.check_transition_collision_free,
    )
    metrics = Metrics()

    shared.set_heartbeat("policy", time.monotonic())
    shared.set_ready("policy")

    # Wait until Main has armed the system (model loaded, hardware ready).
    while shared.is_running.value and int(shared.safety_state.value) != int(
        SafetyState.ARMED
    ):
        # Keep the heartbeat fresh while waiting for arm/inference readiness.
        shared.set_heartbeat("policy", time.monotonic())
        if bool(shared.error_state.value) or bool(shared.estop_request.value):
            return
        time.sleep(0.01)
    if not shared.is_running.value or int(shared.safety_state.value) != int(
        SafetyState.ARMED
    ):
        return

    period_s = 1.0 / float(config.control_hz)
    step_dt_ns = int(round(period_s * 1e9))
    max_source_to_command_age_ns = int(config.max_source_to_command_age_s * 1e9)
    max_silence_ns = int(config.max_command_silence_s * 1e9)
    first_command_timeout_ns = int(config.first_command_timeout_s * 1e9)
    sync_mode = config.inference_mode == "sync"
    scheduler_generation: int | None = None
    last_seen_chunk_key: tuple[int, int] | None = None
    active_execution: _SyncExecution | _AsyncExecution | None = None
    # Silence timeout starts at the first valid endpoint, not first inference.
    last_valid_policy_command_ns: int | None = None
    # RUNNING start time, for the first-command timeout.
    run_started_ns: int | None = None
    pending_acknowledgement: _PendingAcknowledgement | None = None
    previous_arm_command_qpos: np.ndarray | None = None
    episode_steps = _EpisodeActionSteps(config.max_action_steps)
    last_metrics_flush_ns = time.monotonic_ns()

    def reset_scheduler(*, generation: int | None, clear_request: bool) -> None:
        """Clear ActionChunk scheduling state at a lifecycle fence."""
        nonlocal scheduler_generation, last_seen_chunk_key, active_execution
        scheduler_generation = generation
        last_seen_chunk_key = None
        active_execution = None
        if sync_mode and clear_request:
            shared.inference_request.clear()

    def finish_active_endpoint(candidate: ActionCandidate | None) -> None:
        """Finalize one endpoint; sync requests inference only at chunk end."""
        nonlocal active_execution, previous_arm_command_qpos
        if active_execution is None:
            raise RuntimeError("endpoint finalized without an active chunk")
        if candidate is not None:
            if candidate.arm_qpos is None:
                raise RuntimeError("finalized candidate omitted its arm target")
            previous_arm_command_qpos = np.asarray(
                candidate.arm_qpos, dtype=np.float64
            ).copy()
        was_sync = isinstance(active_execution, _SyncExecution)
        if active_execution.finalize_current():
            active_execution = None
            if was_sync:
                shared.inference_request.set()
        elif isinstance(active_execution, _AsyncExecution):
            if not active_execution.align_to_first_future(
                now_ns=time.monotonic_ns(),
                step_dt_ns=step_dt_ns,
            ):
                active_execution = None

    def discard_selected_endpoint() -> None:
        """Finalize one explicit motion/staleness rejection without a token."""
        finish_active_endpoint(None)

    def record_terminal_step(*, applied: bool) -> bool:
        """Account one terminal endpoint and truncate before another selection."""
        nonlocal pending_acknowledgement
        nonlocal last_valid_policy_command_ns, run_started_ns
        nonlocal previous_arm_command_qpos
        reached_limit = (
            episode_steps.record_applied()
            if applied
            else episode_steps.record_safety_rejected()
        )
        metrics.increment(EPISODE_ACTION_STEPS)
        metrics.increment(APPLIED_ACTION_STEPS if applied else SAFETY_REJECTED_STEPS)
        if not reached_limit:
            return False
        _end_policy_run(
            shared,
            "action_step_limit",
            abort=False,
            metrics=metrics,
            summary_status="TRUNCATED",
        )
        pending_acknowledgement = None
        reset_scheduler(generation=None, clear_request=True)
        last_valid_policy_command_ns = None
        run_started_ns = None
        previous_arm_command_qpos = None
        return True

    def fault_physical(reason: str, *, metric: str) -> None:
        """Latch FAULT for a physical publication or acknowledgement failure."""
        nonlocal pending_acknowledgement
        nonlocal last_valid_policy_command_ns, run_started_ns
        nonlocal previous_arm_command_qpos
        shared.error_state.value = True
        shared.physical_home_completed.value = False
        if not revoke_motion(shared, SafetyState.FAULT):
            logger.critical("coordinator: unable to latch FAULT after physical failure")
        logger.critical("coordinator: physical publication failure: %s", reason)
        metrics.increment(POLICY_ABORTS)
        metrics.increment(metric)
        metrics.flush(prefix="coordinator metrics")
        metrics.log_episode_summary(status="FAULTED", reason=reason)
        pending_acknowledgement = None
        reset_scheduler(generation=None, clear_request=True)
        last_valid_policy_command_ns = None
        run_started_ns = None
        previous_arm_command_qpos = None

    def abort_and_reset(reason: str, *, metric: str) -> None:
        """Fail closed and synchronously invalidate every buffered endpoint."""
        nonlocal pending_acknowledgement
        nonlocal last_valid_policy_command_ns, run_started_ns
        nonlocal previous_arm_command_qpos
        if config.execute:
            fault_physical(reason, metric=metric)
            return
        _end_policy_run(shared, reason, abort=True, metrics=metrics, metric=metric)
        pending_acknowledgement = None
        reset_scheduler(generation=None, clear_request=True)
        last_valid_policy_command_ns = None
        run_started_ns = None
        previous_arm_command_qpos = None

    try:
        while shared.is_running.value:
            tick_start = time.monotonic()
            now_ns = time.monotonic_ns()
            shared.set_heartbeat("policy", time.monotonic())

            if bool(shared.quit_requested.value):
                if run_started_ns is not None or int(shared.safety_state.value) == int(
                    SafetyState.RUNNING
                ):
                    _end_policy_run(
                        shared,
                        "operator quit",
                        abort=False,
                        metrics=metrics,
                    )
                else:
                    metrics.log_episode_summary(
                        status="INTERRUPTED",
                        reason="operator quit",
                    )
                pending_acknowledgement = None
                reset_scheduler(generation=None, clear_request=True)
                return

            if bool(shared.error_state.value) or bool(shared.estop_request.value):
                metrics.log_episode_summary(
                    status="FAULTED",
                    reason=(
                        "emergency stop requested"
                        if bool(shared.estop_request.value)
                        else "runtime error state"
                    ),
                )
                shared.physical_home_completed.value = False
                pending_acknowledgement = None
                last_valid_policy_command_ns = None
                run_started_ns = None
                previous_arm_command_qpos = None
                reset_scheduler(generation=None, clear_request=True)
                _sleep_tick(period_s, tick_start)
                continue

            # S fences motion in the keyboard callback before this process can
            # observe the request.  Settle the active episode before entering
            # ARMED idle so its stop time, reason, and counters are not lost.
            if run_started_ns is not None or int(shared.safety_state.value) == int(
                SafetyState.RUNNING
            ):
                raw_stop_request = int(shared.stop_request.value)
                try:
                    stop_request = StopRequest(raw_stop_request)
                except ValueError:
                    abort_and_reset(
                        f"invalid stop request code: {raw_stop_request}",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if stop_request is not StopRequest.NONE:
                    shared.stop_request.value = int(StopRequest.NONE)
                    # A stray B from RUNNING must not auto-restart after this stop.
                    shared.start_request.value = False
                    stop_reason = (
                        "run time limit"
                        if stop_request is StopRequest.RUN_TIME_LIMIT
                        else "operator stop"
                    )
                    if stop_request is StopRequest.RUN_TIME_LIMIT and config.execute:
                        if pending_acknowledgement is not None:
                            fault_physical(
                                "run time limit reached before worker acknowledgement",
                                metric=ACK_TIMEOUT,
                            )
                            continue
                    _end_policy_run(
                        shared,
                        stop_reason,
                        abort=False,
                        metrics=metrics,
                    )
                    pending_acknowledgement = None
                    reset_scheduler(generation=None, clear_request=True)
                    last_valid_policy_command_ns = None
                    run_started_ns = None
                    previous_arm_command_qpos = None
                    _sleep_tick(period_s, tick_start)
                    continue

            # ARMED idle: wait for the operator to request a new run (B).
            if int(shared.safety_state.value) != int(SafetyState.RUNNING):
                # An immediate keyboard fence can become visible just after
                # the stop-request read above. Preserve the active episode and
                # settle its request on the next tick instead of losing it.
                if run_started_ns is not None:
                    _sleep_tick(period_s, tick_start)
                    continue
                if scheduler_generation is not None or active_execution is not None:
                    reset_scheduler(generation=None, clear_request=True)
                pending_acknowledgement = None
                last_valid_policy_command_ns = None
                run_started_ns = None
                previous_arm_command_qpos = None
                if not config.execute:
                    # Shadow has no home lifecycle to acknowledge an S pressed
                    # while already idle. Clear it only with no active episode;
                    # a stop for a prior run is settled by the branch above.
                    with shared.motion_lock:
                        if (
                            int(shared.safety_state.value) == int(SafetyState.ARMED)
                            and not bool(shared.start_request.value)
                            and int(shared.stop_request.value)
                            == int(StopRequest.OPERATOR)
                        ):
                            shared.stop_request.value = int(StopRequest.NONE)
                if not bool(shared.start_request.value):
                    _sleep_tick(period_s, tick_start)
                    continue
                start_pose_rejection = _physical_start_pose_rejection(shared, config)
                if start_pose_rejection is not None:
                    with shared.motion_lock:
                        shared.start_request.value = False
                    logger.warning("coordinator: ignored B: %s", start_pose_rejection)
                    _sleep_tick(period_s, tick_start)
                    continue
                run_epoch = begin_requested_motion(shared)
                if run_epoch is None:
                    # A newer S or lifecycle transition won the same lock.
                    _sleep_tick(period_s, tick_start)
                    continue
                logger.info(
                    "coordinator_loop: RUNNING (run_generation=%d)",
                    run_epoch.generation,
                )
                if config.execute:
                    # H authorizes exactly one physical episode. A subsequent B
                    # remains disabled until another completed H.
                    shared.physical_home_completed.value = False
                last_valid_policy_command_ns = None
                if run_epoch.started_monotonic_ns <= 0:
                    abort_and_reset(
                        "invalid run epoch", metric=ENDPOINTS_FATAL_REJECTED
                    )
                    continue
                run_started_ns = run_epoch.started_monotonic_ns
                previous_arm_command_qpos = None
                metrics.begin_episode(
                    generation=run_epoch.generation,
                    started_monotonic_ns=run_epoch.started_monotonic_ns,
                )
                episode_steps.reset()
                metrics.increment(EPISODE_ACTION_STEPS, 0)
                metrics.increment(APPLIED_ACTION_STEPS, 0)
                metrics.increment(SAFETY_REJECTED_STEPS, 0)
                pending_acknowledgement = None
                reset_scheduler(
                    generation=run_epoch.generation,
                    clear_request=True,
                )
                if sync_mode:
                    shared.inference_request.set()
                _sleep_tick(period_s, tick_start)
                continue

            if scheduler_generation != int(shared.run_generation.value):
                # A lifecycle epoch invalidated the previous scheduler before
                # this tick; never let a stale endpoint survive the boundary.
                reset_scheduler(
                    generation=int(shared.run_generation.value),
                    clear_request=True,
                )

            if pending_acknowledgement is not None:
                acknowledgement = poll_coupled_command_acknowledgement(
                    shared,
                    pending_acknowledgement.candidate,
                    ticket=pending_acknowledgement.ticket,
                    arm_feedback_max_age_s=config.arm_feedback_max_age_s,
                    hand_feedback_max_age_s=config.hand_feedback_max_age_s,
                )
                acknowledgement_observed_ns = time.monotonic_ns()
                acknowledgement_decision = _classify_acknowledgement(
                    pending_acknowledgement,
                    acknowledgement,
                    poll_started_monotonic_ns=now_ns,
                    observed_monotonic_ns=acknowledgement_observed_ns,
                )
                if acknowledgement_decision.action is _AcknowledgementAction.APPLIED:
                    assert acknowledgement_decision.latency_ms is not None
                    acknowledged_candidate = pending_acknowledgement.candidate
                    metrics.increment(ACKNOWLEDGED)
                    metrics.observe(ACK_LATENCY_MS, acknowledgement_decision.latency_ms)
                    metrics.observe_timing(
                        ACK_LATENCY_MS, acknowledgement_decision.latency_ms
                    )
                    logger.debug(
                        "coordinator: action_id=%d acknowledged by arm and hand",
                        acknowledged_candidate.action_id,
                    )
                    pending_acknowledgement = None
                    metrics.increment(ENDPOINTS_COMMITTED)
                    try:
                        reached_limit = _record_terminal_before_finalize(
                            lambda: record_terminal_step(applied=True),
                            lambda: finish_active_endpoint(acknowledged_candidate),
                        )
                    except RuntimeError as exc:
                        fault_physical(
                            f"acknowledgement invariant failed: {exc}",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    if reached_limit:
                        continue
                elif acknowledgement_decision.action is _AcknowledgementAction.WAIT:
                    _sleep_tick(period_s, tick_start)
                    continue
                elif (
                    acknowledgement_decision.action
                    is _AcknowledgementAction.FAULT_TIMEOUT
                ):
                    fault_physical(
                        acknowledgement_decision.reason,
                        metric=ACK_TIMEOUT,
                    )
                    if acknowledgement.status is CommandPublishStatus.APPLIED:
                        _sleep_tick(period_s, tick_start)
                    continue
                else:
                    fault_physical(
                        acknowledgement_decision.reason,
                        metric=ACK_FAILURE,
                    )
                    continue

            watchdog_abort_reason = _command_watchdog_abort_reason(
                now_monotonic_ns=now_ns,
                run_started_monotonic_ns=run_started_ns,
                last_valid_command_monotonic_ns=last_valid_policy_command_ns,
                first_command_timeout_ns=first_command_timeout_ns,
                command_silence_timeout_ns=max_silence_ns,
            )
            if watchdog_abort_reason is not None:
                abort_and_reset(
                    watchdog_abort_reason,
                    metric=COMMAND_SILENCE_ABORT,
                )
                continue

            try:
                newest_chunk = read_latest_action_chunk(shared)
            except Exception as exc:
                abort_and_reset(
                    f"invalid policy chunk IPC record: {exc}",
                    metric=ENDPOINTS_FATAL_REJECTED,
                )
                continue
            if newest_chunk is not None:
                chunk_key = (newest_chunk.run_generation, newest_chunk.chunk_id)
                if chunk_key != last_seen_chunk_key:
                    last_seen_chunk_key = chunk_key
                    if newest_chunk.run_generation != int(shared.run_generation.value):
                        metrics.increment(CHUNKS_GENERATION_DROPPED)
                    elif sync_mode:
                        if active_execution is not None:
                            abort_and_reset(
                                "sync inference published before chunk completion",
                                metric=ENDPOINTS_FATAL_REJECTED,
                            )
                            continue
                        active_execution = _SyncExecution(
                            newest_chunk,
                            chunk_start_monotonic_ns=now_ns,
                        )
                        metrics.increment(CHUNKS_INGESTED)
                    else:
                        current_async = active_execution
                        if current_async is not None and not isinstance(
                            current_async, _AsyncExecution
                        ):
                            abort_and_reset(
                                "async scheduler retained a sync chunk",
                                metric=ENDPOINTS_FATAL_REJECTED,
                            )
                            continue
                        replacement = _newer_async_execution(
                            current_async,
                            newest_chunk,
                            run_generation=int(shared.run_generation.value),
                            now_ns=now_ns,
                            step_dt_ns=step_dt_ns,
                        )
                        if replacement is current_async:
                            if (
                                current_async is None
                                or newest_chunk.chunk_id > current_async.chunk.chunk_id
                            ):
                                metrics.increment(CHUNKS_STALE)
                        else:
                            active_execution = replacement
                            metrics.increment(CHUNKS_INGESTED)

            if active_execution is None:
                _sleep_tick(period_s, tick_start)
                continue
            active_chunk = active_execution.chunk
            try:
                source_deadline_ns = _chunk_source_deadline_ns(
                    active_chunk,
                    max_source_age_ns=max_source_to_command_age_ns,
                )
            except (TypeError, ValueError) as exc:
                abort_and_reset(
                    f"invalid chunk source deadline: {exc}",
                    metric=ENDPOINTS_FATAL_REJECTED,
                )
                continue
            if _chunk_source_is_stale(
                active_chunk,
                now_monotonic_ns=now_ns,
                max_source_age_ns=max_source_to_command_age_ns,
            ):
                was_sync = isinstance(active_execution, _SyncExecution)
                active_execution = None
                if was_sync:
                    shared.inference_request.set()
                metrics.increment(CHUNKS_STALE)
                _sleep_tick(period_s, tick_start)
                continue

            if isinstance(active_execution, _SyncExecution):
                scheduled_target_ns = active_execution.scheduled_target_ns(step_dt_ns)
            else:
                if not active_execution.advance_to_first_future(
                    now_ns=now_ns,
                    step_dt_ns=step_dt_ns,
                ):
                    active_execution = None
                    metrics.increment(CHUNKS_STALE)
                    _sleep_tick(period_s, tick_start)
                    continue
                scheduled_target_ns = active_execution.scheduled_target_ns(step_dt_ns)
            if now_ns < scheduled_target_ns:
                if isinstance(active_execution, _AsyncExecution):
                    time.sleep(
                        min(
                            (scheduled_target_ns - now_ns) / 1e9,
                            _TARGET_WAIT_HEARTBEAT_S,
                        )
                    )
                    # Re-enter through chunk ingest so a newer async result can
                    # replace this selected endpoint before publication.
                    continue
                else:
                    _sleep_tick(period_s, tick_start)
                    continue
            step_index = active_execution.step_index
            active_observation_id = active_chunk.observation_id
            active_observation_anchor_ns = active_chunk.observation_anchor_monotonic_ns
            metrics.increment(ENDPOINTS_DUE)

            assert active_chunk.hand_qpos is not None
            hand_qpos = np.asarray(active_chunk.hand_qpos[step_index], dtype=np.float64)

            _arm_state = read_arm_state_dict(shared)
            if active_chunk.is_ee:
                # EE -> joint via collision-aware IK.
                # hand_qpos is loaded into the collision model first so the solve
                # sees the full 19-DoF geometry.
                if _arm_state is None:
                    metrics.increment(ENDPOINTS_TRANSIENT_DEFERRED)
                    _sleep_tick(period_s, tick_start)
                    continue
                try:
                    arm_issue = diagnose_arm_feedback(
                        connected=bool(_arm_state["connected"]),
                        error_code=int(_arm_state["error_code"]),
                        state_valid=bool(_arm_state["state_valid"]),
                        source_monotonic_ns=int(_arm_state["source_monotonic_ns"]),
                        now_monotonic_ns=time.monotonic_ns(),
                        max_age_s=config.arm_feedback_max_age_s,
                        qpos=np.asarray(_arm_state["qpos"], dtype=np.float64),
                        qvel=np.asarray(_arm_state["qvel"], dtype=np.float64),
                    )
                except Exception as exc:
                    abort_and_reset(
                        f"malformed EE arm feedback: {type(exc).__name__}",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if arm_issue is not None:
                    if arm_issue.code is FeedbackIssueCode.STALE:
                        metrics.increment(ENDPOINTS_TRANSIENT_DEFERRED)
                        _sleep_tick(period_s, tick_start)
                        continue
                    abort_and_reset(
                        f"fatal EE arm feedback: {arm_issue.code.value}",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                assert active_chunk.ee_pos is not None
                assert active_chunk.ee_rot6d is not None
                ee_pos = np.asarray(active_chunk.ee_pos[step_index], dtype=np.float64)
                ee_rot6d = np.asarray(
                    active_chunk.ee_rot6d[step_index], dtype=np.float64
                )
                try:
                    validate_rot6d_geometry(ee_rot6d, label="policy ee_rot6d")
                except ValueError:
                    metrics.increment(ENDPOINTS_MOTION_DISCARDED)
                    try:
                        reached_limit = _record_terminal_before_finalize(
                            lambda: record_terminal_step(applied=False),
                            discard_selected_endpoint,
                        )
                    except RuntimeError:
                        abort_and_reset(
                            "cannot discard malformed EE endpoint",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    if reached_limit:
                        continue
                    _sleep_tick(period_s, tick_start)
                    continue
                try:
                    planner.set_hand_qpos(hand_qpos)
                    ik_result = planner.solve_teleop_ik(
                        Pose(p=ee_pos, q=rot6d_to_quat_wxyz(ee_rot6d)),
                        _arm_state["qpos"],
                        (
                            previous_arm_command_qpos
                            if previous_arm_command_qpos is not None
                            else _arm_state["qpos"]
                        ),
                    )
                except Exception as exc:
                    abort_and_reset(
                        f"unexpected EE IK exception: {type(exc).__name__}",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if not ik_result.success or ik_result.qpos is None:
                    failure_kind = getattr(ik_result, "failure_kind", None)
                    if failure_kind is IKFailureKind.CHECKER_FAILURE:
                        metrics.increment(IK_CHECKER_REJECTS)
                        abort_and_reset(
                            "EE IK collision checker failed",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    if failure_kind is IKFailureKind.INVALID_OUTPUT:
                        abort_and_reset(
                            "EE IK returned non-finite output",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    if failure_kind not in {
                        IKFailureKind.NO_SOLUTION,
                        IKFailureKind.GEOMETRY_REJECTED,
                        IKFailureKind.COLLISION,
                    }:
                        abort_and_reset(
                            "EE IK returned an untyped failure",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    metrics.increment(ENDPOINTS_MOTION_DISCARDED)
                    try:
                        reached_limit = _record_terminal_before_finalize(
                            lambda: record_terminal_step(applied=False),
                            discard_selected_endpoint,
                        )
                    except RuntimeError:
                        abort_and_reset(
                            "cannot discard rejected EE endpoint",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    if reached_limit:
                        continue
                    _sleep_tick(period_s, tick_start)
                    continue
                arm_qpos = np.asarray(ik_result.qpos, dtype=np.float64)
            else:
                assert active_chunk.arm_qpos is not None
                arm_qpos = np.asarray(
                    active_chunk.arm_qpos[step_index], dtype=np.float64
                )
                # Preserve joint-wrap continuity against the command stream.
                arm_reference = previous_arm_command_qpos
                if (
                    arm_reference is None
                    and _arm_state is not None
                    and np.all(np.isfinite(_arm_state["qpos"]))
                ):
                    arm_reference = _arm_state["qpos"]
                if arm_reference is not None:
                    arm_qpos = wrap_nearest_equivalent(
                        arm_qpos,
                        arm_reference,
                        config.arm_joint_lower_rad,
                        config.arm_joint_upper_rad,
                    )

            try:
                candidate = build_action_candidate(
                    shared,
                    arm_qpos,
                    hand_qpos,
                    run_generation=active_chunk.run_generation,
                    is_hold=False,
                    observation_id=active_observation_id,
                    observation_anchor_monotonic_ns=active_observation_anchor_ns,
                    scheduled_target_monotonic_ns=scheduled_target_ns,
                    action_validity_s=float(config.action_validity_s),
                    valid_until_monotonic_ns=source_deadline_ns,
                )
            except (TypeError, ValueError) as exc:
                abort_and_reset(
                    f"candidate contract failure: {type(exc).__name__}",
                    metric=ENDPOINTS_FATAL_REJECTED,
                )
                continue
            if candidate is None:
                if time.monotonic_ns() > source_deadline_ns:
                    try:
                        discard_selected_endpoint()
                    except RuntimeError:
                        abort_and_reset(
                            "stale endpoint could not be finalized",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    metrics.increment(ENDPOINTS_STALE_DISCARDED)
                    _sleep_tick(period_s, tick_start)
                    continue
                abort_and_reset(
                    "invalid candidate or observation anchor",
                    metric=ENDPOINTS_FATAL_REJECTED,
                )
                continue

            try:
                publish_result = validate_and_send_candidate(
                    shared,
                    candidate,
                    gate=gate,
                    arm_feedback_max_age_s=config.arm_feedback_max_age_s,
                    hand_feedback_max_age_s=config.hand_feedback_max_age_s,
                    arm_delta_reference_qpos=previous_arm_command_qpos,
                    hand_mechanical_lower_rad=np.asarray(
                        config.hand_mechanical_lower_rad, dtype=np.float64
                    ),
                    hand_mechanical_upper_rad=np.asarray(
                        config.hand_mechanical_upper_rad, dtype=np.float64
                    ),
                    hand_command_max_delta_rad_per_tick=(
                        config.hand_command_max_delta_rad_per_tick
                    ),
                    canonicalize_policy_hand_roundoff=True,
                    execute=config.execute,
                    required_safety_state=SafetyState.RUNNING,
                    # Leave one full policy tick for both 30 Hz workers to
                    # observe the coupled record.
                    minimum_delivery_window_s=period_s,
                )
            except Exception as exc:
                abort_and_reset(
                    f"publication invariant failed: {type(exc).__name__}",
                    metric=ENDPOINTS_FATAL_REJECTED,
                )
                continue
            if publish_result.hand_roundoff_canonicalized:
                metrics.increment(HAND_POLICY_ENDPOINT_ROUNDOFF_CANONICALIZED)
            disposition = classify_policy_endpoint_disposition(
                publish_result,
                hand_limit_nesting_valid=True,
            )
            if disposition is PolicyEndpointDisposition.COMMIT:
                validated_only = publish_result.status is CommandPublishStatus.VALIDATED
                if (not config.execute) != validated_only:
                    abort_and_reset(
                        "execute flag and publication result disagree",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if (
                    config.execute
                    and publish_result.status is not CommandPublishStatus.PUBLISHED
                ):
                    abort_and_reset(
                        "physical publication did not return a command ticket",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                published_candidate = publish_result.candidate
                if published_candidate is None:
                    abort_and_reset(
                        "successful publication omitted its candidate",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if published_candidate.action_id != candidate.action_id:
                    abort_and_reset(
                        "publication changed candidate action identity",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if validated_only:
                    metrics.increment(ENDPOINTS_VALIDATED)
                    metrics.increment(ENDPOINTS_COMMITTED)
                    try:
                        reached_limit = _record_terminal_before_finalize(
                            lambda: record_terminal_step(applied=True),
                            lambda: finish_active_endpoint(published_candidate),
                        )
                    except RuntimeError as exc:
                        abort_and_reset(
                            f"validation invariant failed: {exc}",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    if reached_limit:
                        continue
                else:
                    metrics.increment(ENDPOINTS_PUBLISHED)
                    ticket = publish_result.ticket
                    if ticket is None:
                        fault_physical(
                            "physical publication omitted its coupled command ticket",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    publication_ns = int(ticket.published_monotonic_ns)
                    if publication_ns <= 0:
                        fault_physical(
                            "physical publication omitted its monotonic timestamp",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    pending_acknowledgement = _PendingAcknowledgement(
                        candidate=published_candidate,
                        ticket=ticket,
                        published_monotonic_ns=publication_ns,
                        acceptance_deadline_monotonic_ns=int(
                            published_candidate.valid_until_monotonic_ns
                        ),
                        observation_deadline_monotonic_ns=(
                            publication_ns
                            + int(config.command_acknowledgement_timeout_s * 1e9)
                        ),
                    )
                    logger.debug(
                        "coordinator: published action_id=%d; awaiting arm/hand acknowledgement",
                        published_candidate.action_id,
                    )
                last_valid_policy_command_ns = now_ns
            elif disposition in {
                PolicyEndpointDisposition.DISCARD_MOTION,
                PolicyEndpointDisposition.DISCARD_STALE,
            }:
                if publish_result.status is CommandPublishStatus.GATE_REJECTED:
                    metrics.increment(reject_counter_name(None))
                    metrics.increment(
                        reject_counter_name(
                            publish_result.gate_code.value
                            if publish_result.gate_code is not None
                            else None
                        )
                    )
                if (
                    publish_result.status
                    is CommandPublishStatus.HAND_PREFLIGHT_REJECTED
                ):
                    metrics.increment(HAND_PREFLIGHT_REJECTIONS)
                if disposition is PolicyEndpointDisposition.DISCARD_MOTION:
                    metrics.increment(ENDPOINTS_MOTION_DISCARDED)
                    try:
                        reached_limit = _record_terminal_before_finalize(
                            lambda: record_terminal_step(applied=False),
                            discard_selected_endpoint,
                        )
                    except RuntimeError:
                        abort_and_reset(
                            "rejected endpoint could not be finalized",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    if reached_limit:
                        continue
                else:
                    try:
                        discard_selected_endpoint()
                    except RuntimeError:
                        abort_and_reset(
                            "stale endpoint could not be finalized",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    metrics.increment(ENDPOINTS_STALE_DISCARDED)
            elif disposition is PolicyEndpointDisposition.DEFER_TRANSIENT:
                metrics.increment(ENDPOINTS_TRANSIENT_DEFERRED)
            else:
                abort_and_reset(
                    f"fatal policy endpoint result: {publish_result.status.value}",
                    metric=ENDPOINTS_FATAL_REJECTED,
                )
                continue

            last_metrics_flush_ns = flush_every(
                metrics, last_ns=last_metrics_flush_ns, prefix="coordinator metrics"
            )
            _sleep_tick(period_s, tick_start)
    finally:
        metrics.log_episode_summary(
            status="INTERRUPTED",
            reason="coordinator loop exited",
        )
        logger.info("coordinator_loop: exited")


def _sleep_tick(period_s: float, tick_start: float) -> None:
    """Sleep for the remainder of one control tick, if any."""
    sleep_s = period_s - (time.monotonic() - tick_start)
    if sleep_s > 0:
        time.sleep(sleep_s)

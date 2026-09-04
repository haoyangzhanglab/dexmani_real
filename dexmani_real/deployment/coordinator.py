"""Deployment coordinator — the sole learned-policy robot-action producer.

The inference worker writes proposals to ``policy_chunk_ring``; this coordinator
is the only process that turns a proposal into a robot command. It selects the
chunk, schedules the due endpoint (one per control tick), runs the shared
candidate publication boundary (SafetyGate -> send_command), and owns the
policy semantic watchdog and the ``RUNNING <-> ARMED`` control-source state.

It never dumps a whole chunk into the arm queue or hand ring and never
interpolates between model steps. Coordination polling is separate from the
policy control grid so worker progress polling and completed inference do not
consume whole action slots.
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
    validate_and_send_candidate,
)
from dexmani_real.control.safety_gate import GateRejectCode, SafetyGate
from dexmani_real.deployment.config import PolicyDeploymentConfig
from dexmani_real.deployment.contracts import ActionChunk
from dexmani_real.deployment.metrics import (
    CHUNKS_GENERATION_DROPPED,
    CHUNKS_INGESTED,
    CHUNKS_STALE,
    COMMAND_PROGRESS_TIMEOUT,
    COMMAND_SILENCE_ABORT,
    ENDPOINT_SCHEDULE_LATENESS_MS,
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
    POLICY_ABORTS,
    PUBLICATION_INTERVAL_MS,
    SAFETY_REJECTED_STEPS,
    SUCCESSFUL_ACTION_STEPS,
    Metrics,
    flush_every,
    reject_counter_name,
)
from dexmani_real.deployment.timing import first_future_step_index
from dexmani_real.ipc.channels import (
    RuntimeChannels,
    read_arm_state_dict,
    read_hand_state_dict,
)
from dexmani_real.ipc.schema import MAX_POLICY_CHUNK_STEPS, POLICY_CHUNK_DTYPE
from dexmani_real.planning import (
    Pose,
    TeleopProfile,
    XArm7MotionPlanner,
    XArm7PlannerConfig,
)
from dexmani_real.planning.arm_fk import make_arm_fk
from dexmani_real.planning.paths import (
    WORKSPACE_BOUNDS_TOLERANCE_M,
    interpolate_waypoints,
    wrap_nearest_equivalent,
)
from dexmani_real.planning.poses import rot6d_to_quat_wxyz, validate_rot6d_geometry
from dexmani_real.planning.types import IKFailureKind
from dexmani_real.robot_spec import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.runtime.safety import (
    SafetyState,
    StopRequest,
    begin_requested_motion,
    revoke_motion,
)
from dexmani_real.utils.feedback import (
    FeedbackIssueCode,
    diagnose_arm_feedback,
    diagnose_hand_feedback,
)
from dexmani_real.utils.limits import validate_hand_limit_nesting
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.rate import LoopRate

if TYPE_CHECKING:
    from dexmani_real.config.runtime import ResolvedRuntimeConfig

logger = get_logger(__name__)
_UINT64_MAX = int(np.iinfo(np.uint64).max)
_POLICY_WORKSPACE_INTERPOLATION_MAX_STEP_RAD = 0.02


class PolicyEndpointDisposition(str, Enum):
    """Coordinator action for one typed policy endpoint result."""

    COMMIT = "commit"
    DISCARD_MOTION = "discard_motion"
    DEFER_TRANSIENT = "defer_transient"
    DISCARD_STALE = "discard_stale"
    ABORT_FATAL = "abort_fatal"


def classify_policy_endpoint_disposition(
    result: CommandPublishResult,
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
        return PolicyEndpointDisposition.DISCARD_MOTION
    if result.status is CommandPublishStatus.GATE_REJECTED:
        if result.gate_code in {
            GateRejectCode.ARM_JOINT_LIMIT,
            GateRejectCode.HAND_JOINT_LIMIT,
            GateRejectCode.ARM_DELTA_LIMIT,
            GateRejectCode.WORKSPACE,
        }:
            return PolicyEndpointDisposition.DISCARD_MOTION
        if result.gate_code is GateRejectCode.RUN_GENERATION_MISMATCH:
            return PolicyEndpointDisposition.DEFER_TRANSIENT
        # HAND_DELTA_LIMIT and collision transition checks are disabled for the
        # learned-policy gate; remaining gate/contract/checker failures are
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
    # malformed candidates and all unknown outcomes fail closed.
    return PolicyEndpointDisposition.ABORT_FATAL


@dataclass
class _CommandProgressWatchdog:
    """Track independent worker acceptance watermarks for one run generation."""

    run_generation: int | None = None
    latest_published_action_id: int | None = None
    arm_accepted_action_id: int | None = None
    hand_accepted_action_id: int | None = None
    arm_no_progress_since_monotonic_ns: int | None = None
    hand_no_progress_since_monotonic_ns: int | None = None

    def reset(self, *, run_generation: int | None) -> None:
        if run_generation is not None and run_generation <= 0:
            raise ValueError("run generation must be positive or null")
        self.run_generation = run_generation
        self.latest_published_action_id = None
        self.arm_accepted_action_id = None
        self.hand_accepted_action_id = None
        self.arm_no_progress_since_monotonic_ns = None
        self.hand_no_progress_since_monotonic_ns = None

    def observe(
        self,
        *,
        run_generation: int,
        arm_accepted_action_id: int | None,
        hand_accepted_action_id: int | None,
        now_monotonic_ns: int,
        timeout_ns: int,
    ) -> str | None:
        """Observe worker progress and return a bounded stall reason, if any."""
        if self.run_generation != run_generation:
            return "command progress generation does not match active run"
        if now_monotonic_ns <= 0 or timeout_ns <= 0:
            raise ValueError("command progress times must be positive")
        for worker, observed_action_id in (
            ("arm", arm_accepted_action_id),
            ("hand", hand_accepted_action_id),
        ):
            if observed_action_id is None:
                continue
            if observed_action_id < 0:
                return f"{worker} command progress is negative"
            previous_action_id = getattr(self, f"{worker}_accepted_action_id")
            if (
                previous_action_id is not None
                and observed_action_id < previous_action_id
            ):
                return f"{worker} command progress regressed"
            setattr(self, f"{worker}_accepted_action_id", observed_action_id)
            if previous_action_id is None:
                continue
            if observed_action_id > previous_action_id:
                waiting_since = (
                    None
                    if self._worker_covers_latest(observed_action_id)
                    else now_monotonic_ns
                )
                setattr(
                    self,
                    f"{worker}_no_progress_since_monotonic_ns",
                    waiting_since,
                )

        for worker in ("arm", "hand"):
            waiting_since = getattr(self, f"{worker}_no_progress_since_monotonic_ns")
            if (
                waiting_since is not None
                and now_monotonic_ns - waiting_since > timeout_ns
            ):
                return f"{worker} worker command progress timeout"
        return None

    def record_publication(
        self,
        *,
        run_generation: int,
        action_id: int,
        published_monotonic_ns: int,
    ) -> None:
        """Extend the target watermark without hiding an existing stall."""
        if self.run_generation != run_generation:
            raise RuntimeError("publication does not match progress generation")
        if action_id <= 0 or published_monotonic_ns <= 0:
            raise ValueError("published action identity and time must be positive")
        previous_published = self.latest_published_action_id
        if previous_published is not None and action_id <= previous_published:
            raise RuntimeError("published action IDs must increase")
        for worker in ("arm", "hand"):
            accepted_action_id = getattr(self, f"{worker}_accepted_action_id")
            if accepted_action_id is None:
                raise RuntimeError(f"{worker} command progress baseline is unavailable")
            if action_id <= accepted_action_id:
                raise RuntimeError(
                    f"published action ID does not advance beyond {worker} baseline"
                )
            waiting_field = f"{worker}_no_progress_since_monotonic_ns"
            if getattr(self, waiting_field) is None:
                setattr(self, waiting_field, published_monotonic_ns)
        self.latest_published_action_id = action_id

    @property
    def latest_publication_is_accepted(self) -> bool:
        latest_action_id = self.latest_published_action_id
        return (
            latest_action_id is not None
            and self.arm_accepted_action_id is not None
            and self.hand_accepted_action_id is not None
            and self.arm_accepted_action_id >= latest_action_id
            and self.hand_accepted_action_id >= latest_action_id
        )

    def _worker_covers_latest(self, accepted_action_id: int) -> bool:
        latest_action_id = self.latest_published_action_id
        return latest_action_id is None or accepted_action_id >= latest_action_id


@dataclass
class _EpisodeActionSteps:
    """Pure per-episode terminal policy-step accounting."""

    max_action_steps: int | None
    episode_action_steps: int = 0
    successful_action_steps: int = 0
    safety_rejected_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_action_steps is not None and (
            type(self.max_action_steps) is not int or self.max_action_steps <= 0
        ):
            raise ValueError("max_action_steps must be a positive integer or null")

    def reset(self) -> None:
        self.episode_action_steps = 0
        self.successful_action_steps = 0
        self.safety_rejected_steps = 0

    def record_successful(self) -> bool:
        self.episode_action_steps += 1
        self.successful_action_steps += 1
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


def _scheduled_endpoint_due_ns(
    scheduled_target_ns: int,
    next_endpoint_due_ns: int | None,
) -> int:
    """Combine a chunk target with the persistent policy publication grid."""
    if next_endpoint_due_ns is None:
        return scheduled_target_ns
    return max(scheduled_target_ns, next_endpoint_due_ns)


def _advance_endpoint_due_ns(
    endpoint_due_ns: int,
    terminal_monotonic_ns: int,
    step_dt_ns: int,
) -> int:
    """Advance one terminal slot without accumulating jitter or catching up."""
    if endpoint_due_ns <= 0 or terminal_monotonic_ns <= 0 or step_dt_ns <= 0:
        raise ValueError("endpoint schedule times must be positive")
    lateness_ns = max(0, terminal_monotonic_ns - endpoint_due_ns)
    if lateness_ns >= step_dt_ns:
        return terminal_monotonic_ns + step_dt_ns
    return endpoint_due_ns + step_dt_ns


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


def _read_healthy_command_progress(
    shared: RuntimeChannels,
    config: "CoordinatorConfig",
    *,
    now_monotonic_ns: int,
) -> tuple[int | None, int | None, str | None]:
    """Read healthy worker watermarks; stale/missing feedback counts as no progress."""
    arm_state = read_arm_state_dict(shared)
    hand_state = read_hand_state_dict(shared)
    arm_action_id: int | None = None
    hand_action_id: int | None = None
    if arm_state is not None:
        arm_issue = diagnose_arm_feedback(
            connected=bool(arm_state["connected"]),
            error_code=int(arm_state["error_code"]),
            state_valid=bool(arm_state["state_valid"]),
            source_monotonic_ns=int(arm_state["source_monotonic_ns"]),
            now_monotonic_ns=now_monotonic_ns,
            max_age_s=config.arm_feedback_max_age_s,
            qpos=np.asarray(arm_state["qpos"], dtype=np.float64),
            qvel=np.asarray(arm_state["qvel"], dtype=np.float64),
        )
        if arm_issue is None:
            arm_action_id = int(arm_state["last_cmd_seq"])
        elif arm_issue.code is not FeedbackIssueCode.STALE:
            return None, None, f"fatal arm feedback: {arm_issue.code.value}"
    if hand_state is not None:
        hand_issue = diagnose_hand_feedback(
            connected=bool(hand_state["connected"]),
            state_valid=bool(hand_state["state_valid"]),
            source_monotonic_ns=int(hand_state["source_monotonic_ns"]),
            now_monotonic_ns=now_monotonic_ns,
            max_age_s=config.hand_feedback_max_age_s,
            qpos=np.asarray(hand_state["qpos"], dtype=np.float64),
        )
        if hand_issue is None:
            hand_action_id = int(hand_state["accepted_target_action_id"])
        elif hand_issue.code is not FeedbackIssueCode.STALE:
            return None, None, f"fatal hand feedback: {hand_issue.code.value}"
    return arm_action_id, hand_action_id, None


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
    coordinator_hz: float
    execute: bool
    max_source_to_command_age_s: float
    max_command_silence_s: float
    action_validity_s: float
    command_progress_timeout_s: float
    first_command_timeout_s: float
    inference_mode: str = "sync"
    max_action_steps: int | None = None
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
        if not np.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise ValueError("control_hz must be finite and positive")
        if (
            not np.isfinite(self.coordinator_hz)
            or self.coordinator_hz < self.control_hz
        ):
            raise ValueError("coordinator_hz must be finite and >= control_hz")
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
            self.command_progress_timeout_s,
            self.first_command_timeout_s,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in timing_values):
            raise ValueError("coordinator timing values must be finite and positive")
        if self.command_progress_timeout_s > self.action_validity_s:
            raise ValueError(
                "command progress timeout must be positive and no greater "
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
            coordinator_hz=float(runtime.policy.coordinator_hz),
            execute=execute,
            max_source_to_command_age_s=float(
                runtime.policy.max_source_to_command_age_s
            ),
            max_command_silence_s=float(runtime.policy.max_command_silence_s),
            action_validity_s=float(runtime.policy.action_validity_s),
            command_progress_timeout_s=float(runtime.policy.command_progress_timeout_s),
            first_command_timeout_s=float(runtime.policy.first_command_timeout_s),
            inference_mode=deployment.inference_mode,
            max_action_steps=deployment.max_action_steps,
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


def _build_policy_planner(config: CoordinatorConfig) -> XArm7MotionPlanner:
    """Build policy kinematics without real-time software collision checks."""
    return XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
            srdf_path=str(XARM7_XHAND_SRDF_PATH),
            base_pose_world=Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0])),
            workspace_bounds=np.asarray(config.workspace_bounds, dtype=np.float64),
        ),
        teleop_profile=TeleopProfile(
            max_pose_error_pos_m=config.ik_max_pose_error_pos_m,
            max_pose_error_rot_rad=config.ik_max_pose_error_rot_rad,
            check_self_collision=False,
        ),
        hand_dof=False,
    )


def _build_policy_workspace_check(
    config: CoordinatorConfig,
) -> Callable[[np.ndarray, np.ndarray], bool]:
    """Build the learned-policy arm-base workspace segment predicate.

    Joint policy keeps its hot-path workspace check independent of MPlib and
    collision resources. ``ArmFK`` returns the EEF in arm-base coordinates,
    which is the configured policy workspace frame.
    """
    workspace_bounds = np.asarray(config.workspace_bounds, dtype=np.float64)
    if (
        workspace_bounds.shape != (3, 2)
        or not np.all(np.isfinite(workspace_bounds))
        or np.any(workspace_bounds[:, 0] > workspace_bounds[:, 1])
    ):
        raise ValueError("workspace bounds must be finite shape (3, 2) and ordered")
    arm_fk = make_arm_fk()

    def is_workspace_segment_safe(
        start_arm_qpos: np.ndarray,
        end_arm_qpos: np.ndarray,
    ) -> bool:
        """Check every 0.02-rad interpolated EEF sample against workspace."""
        try:
            path = interpolate_waypoints(
                np.stack([start_arm_qpos, end_arm_qpos]),
                max_step=_POLICY_WORKSPACE_INTERPOLATION_MAX_STEP_RAD,
            )
            for arm_qpos in path:
                eef_position_base, _ = arm_fk.compute(arm_qpos)
                eef_position_base = np.asarray(eef_position_base, dtype=np.float64)
                if (
                    eef_position_base.shape != (3,)
                    or not np.all(np.isfinite(eef_position_base))
                    or np.any(
                        eef_position_base
                        < workspace_bounds[:, 0] - WORKSPACE_BOUNDS_TOLERANCE_M
                    )
                    or np.any(
                        eef_position_base
                        > workspace_bounds[:, 1] + WORKSPACE_BOUNDS_TOLERANCE_M
                    )
                ):
                    return False
        except Exception:
            return False
        return True

    return is_workspace_segment_safe


def _build_policy_safety_gate(
    config: CoordinatorConfig,
) -> SafetyGate:
    """Build the real-time policy gate; return_home owns collision planning."""
    return SafetyGate(
        arm_joint_lower_rad=config.arm_joint_lower_rad,
        arm_joint_upper_rad=config.arm_joint_upper_rad,
        hand_joint_lower_rad=config.hand_joint_lower_rad,
        hand_joint_upper_rad=config.hand_joint_upper_rad,
        workspace_check=_build_policy_workspace_check(config),
        max_arm_delta_rad=config.arm_max_delta_rad_per_tick,
        max_hand_delta_rad=config.hand_max_delta_rad_per_tick,
        endpoint_delta_tolerance_rad=config.endpoint_delta_tolerance_rad,
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

    gate = _build_policy_safety_gate(config)
    ee_planner: XArm7MotionPlanner | None = None
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

    control_period_s = 1.0 / float(config.control_hz)
    step_dt_ns = int(round(control_period_s * 1e9))
    max_source_to_command_age_ns = int(config.max_source_to_command_age_s * 1e9)
    max_silence_ns = int(config.max_command_silence_s * 1e9)
    command_progress_timeout_ns = int(config.command_progress_timeout_s * 1e9)
    first_command_timeout_ns = int(config.first_command_timeout_s * 1e9)
    sync_mode = config.inference_mode == "sync"
    scheduler_generation: int | None = None
    last_seen_chunk_key: tuple[int, int] | None = None
    active_execution: _SyncExecution | _AsyncExecution | None = None
    # Silence timeout starts at the first valid endpoint, not first inference.
    last_valid_policy_command_ns: int | None = None
    # RUNNING start time, for the first-command timeout.
    run_started_ns: int | None = None
    command_progress = _CommandProgressWatchdog()
    pending_truncation_action_id: int | None = None
    previous_arm_command_qpos: np.ndarray | None = None
    next_endpoint_due_ns: int | None = None
    last_publication_monotonic_ns: int | None = None
    episode_steps = _EpisodeActionSteps(config.max_action_steps)
    last_metrics_flush_ns = time.monotonic_ns()
    rate = LoopRate(
        config.coordinator_hz,
        label="policy coordinator",
        busy_wait=False,
    )

    def reset_scheduler(*, generation: int | None, clear_request: bool) -> None:
        """Clear ActionChunk scheduling state at a lifecycle fence."""
        nonlocal scheduler_generation, last_seen_chunk_key, active_execution
        nonlocal next_endpoint_due_ns, last_publication_monotonic_ns
        nonlocal pending_truncation_action_id
        scheduler_generation = generation
        last_seen_chunk_key = None
        active_execution = None
        next_endpoint_due_ns = None
        last_publication_monotonic_ns = None
        pending_truncation_action_id = None
        command_progress.reset(run_generation=generation)
        if sync_mode and clear_request:
            shared.inference_request.clear()

    def consume_endpoint_slot(
        *,
        endpoint_due_ns: int,
        terminal_monotonic_ns: int,
        publication_monotonic_ns: int | None = None,
    ) -> None:
        """Advance one terminal policy slot and record its timing diagnostics."""
        nonlocal next_endpoint_due_ns, last_publication_monotonic_ns
        schedule_lateness_ms = max(0, terminal_monotonic_ns - endpoint_due_ns) / 1e6
        metrics.observe(ENDPOINT_SCHEDULE_LATENESS_MS, schedule_lateness_ms)
        metrics.observe_timing(ENDPOINT_SCHEDULE_LATENESS_MS, schedule_lateness_ms)
        next_endpoint_due_ns = _advance_endpoint_due_ns(
            endpoint_due_ns,
            terminal_monotonic_ns,
            step_dt_ns,
        )
        if publication_monotonic_ns is None:
            return
        if last_publication_monotonic_ns is not None:
            publication_interval_ms = (
                publication_monotonic_ns - last_publication_monotonic_ns
            ) / 1e6
            metrics.observe(PUBLICATION_INTERVAL_MS, publication_interval_ms)
            metrics.observe_timing(
                PUBLICATION_INTERVAL_MS,
                publication_interval_ms,
            )
        last_publication_monotonic_ns = publication_monotonic_ns

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

    def record_terminal_step(
        *,
        successful: bool,
        wait_for_action_id: int | None = None,
    ) -> bool:
        """Account one terminal endpoint and truncate before another selection."""
        nonlocal pending_truncation_action_id
        nonlocal last_valid_policy_command_ns, run_started_ns
        nonlocal previous_arm_command_qpos
        reached_limit = (
            episode_steps.record_successful()
            if successful
            else episode_steps.record_safety_rejected()
        )
        metrics.increment(EPISODE_ACTION_STEPS)
        metrics.increment(
            SUCCESSFUL_ACTION_STEPS if successful else SAFETY_REJECTED_STEPS
        )
        if not reached_limit:
            return False
        if config.execute and successful:
            if wait_for_action_id is None or wait_for_action_id <= 0:
                raise RuntimeError(
                    "physical action-step limit requires its published action ID"
                )
            pending_truncation_action_id = wait_for_action_id
            return True
        _end_policy_run(
            shared,
            "action_step_limit",
            abort=False,
            metrics=metrics,
            summary_status="TRUNCATED",
        )
        pending_truncation_action_id = None
        reset_scheduler(generation=None, clear_request=True)
        last_valid_policy_command_ns = None
        run_started_ns = None
        previous_arm_command_qpos = None
        return True

    def fault_physical(reason: str, *, metric: str) -> None:
        """Latch FAULT for a physical publication or progress failure."""
        nonlocal pending_truncation_action_id
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
        pending_truncation_action_id = None
        reset_scheduler(generation=None, clear_request=True)
        last_valid_policy_command_ns = None
        run_started_ns = None
        previous_arm_command_qpos = None

    def abort_and_reset(reason: str, *, metric: str) -> None:
        """Fail closed and synchronously invalidate every buffered endpoint."""
        nonlocal pending_truncation_action_id
        nonlocal last_valid_policy_command_ns, run_started_ns
        nonlocal previous_arm_command_qpos
        if config.execute:
            fault_physical(reason, metric=metric)
            return
        _end_policy_run(shared, reason, abort=True, metrics=metrics, metric=metric)
        pending_truncation_action_id = None
        reset_scheduler(generation=None, clear_request=True)
        last_valid_policy_command_ns = None
        run_started_ns = None
        previous_arm_command_qpos = None

    try:
        while shared.is_running.value:
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
                pending_truncation_action_id = None
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
                pending_truncation_action_id = None
                last_valid_policy_command_ns = None
                run_started_ns = None
                previous_arm_command_qpos = None
                reset_scheduler(generation=None, clear_request=True)
                rate.wait()
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
                    _end_policy_run(
                        shared,
                        stop_reason,
                        abort=False,
                        metrics=metrics,
                    )
                    pending_truncation_action_id = None
                    reset_scheduler(generation=None, clear_request=True)
                    last_valid_policy_command_ns = None
                    run_started_ns = None
                    previous_arm_command_qpos = None
                    rate.wait()
                    continue

            # ARMED idle: wait for the operator to request a new run (B).
            if int(shared.safety_state.value) != int(SafetyState.RUNNING):
                # An immediate keyboard fence can become visible just after
                # the stop-request read above. Preserve the active episode and
                # settle its request on the next tick instead of losing it.
                if run_started_ns is not None:
                    rate.wait()
                    continue
                if scheduler_generation is not None or active_execution is not None:
                    reset_scheduler(generation=None, clear_request=True)
                pending_truncation_action_id = None
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
                    rate.wait()
                    continue
                start_pose_rejection = _physical_start_pose_rejection(shared, config)
                if start_pose_rejection is not None:
                    with shared.motion_lock:
                        shared.start_request.value = False
                    logger.warning("coordinator: ignored B: %s", start_pose_rejection)
                    rate.wait()
                    continue
                run_epoch = begin_requested_motion(shared)
                if run_epoch is None:
                    # A newer S or lifecycle transition won the same lock.
                    rate.wait()
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
                metrics.increment(SUCCESSFUL_ACTION_STEPS, 0)
                metrics.increment(SAFETY_REJECTED_STEPS, 0)
                pending_truncation_action_id = None
                reset_scheduler(
                    generation=run_epoch.generation,
                    clear_request=True,
                )
                if sync_mode:
                    shared.inference_request.set()
                rate.wait()
                continue

            if scheduler_generation != int(shared.run_generation.value):
                # A lifecycle epoch invalidated the previous scheduler before
                # this tick; never let a stale endpoint survive the boundary.
                reset_scheduler(
                    generation=int(shared.run_generation.value),
                    clear_request=True,
                )

            if config.execute:
                try:
                    arm_progress_id, hand_progress_id, progress_feedback_fault = (
                        _read_healthy_command_progress(
                            shared,
                            config,
                            now_monotonic_ns=now_ns,
                        )
                    )
                except Exception as exc:
                    fault_physical(
                        f"command progress feedback failed: {type(exc).__name__}",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if progress_feedback_fault is not None:
                    fault_physical(
                        progress_feedback_fault,
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                progress_fault = command_progress.observe(
                    run_generation=int(shared.run_generation.value),
                    arm_accepted_action_id=arm_progress_id,
                    hand_accepted_action_id=hand_progress_id,
                    now_monotonic_ns=now_ns,
                    timeout_ns=command_progress_timeout_ns,
                )
                if progress_fault is not None:
                    fault_physical(
                        progress_fault,
                        metric=COMMAND_PROGRESS_TIMEOUT,
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

            if config.execute and (
                command_progress.arm_accepted_action_id is None
                or command_progress.hand_accepted_action_id is None
            ):
                rate.wait()
                continue
            if pending_truncation_action_id is not None:
                if (
                    command_progress.latest_published_action_id
                    != pending_truncation_action_id
                ):
                    fault_physical(
                        "action-step truncation watermark changed unexpectedly",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if not command_progress.latest_publication_is_accepted:
                    rate.wait()
                    continue
                _end_policy_run(
                    shared,
                    "action_step_limit",
                    abort=False,
                    metrics=metrics,
                    summary_status="TRUNCATED",
                )
                pending_truncation_action_id = None
                reset_scheduler(generation=None, clear_request=True)
                last_valid_policy_command_ns = None
                run_started_ns = None
                previous_arm_command_qpos = None
                rate.wait()
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
                rate.wait()
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
                rate.wait()
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
                    rate.wait()
                    continue
                scheduled_target_ns = active_execution.scheduled_target_ns(step_dt_ns)
            endpoint_due_ns = _scheduled_endpoint_due_ns(
                scheduled_target_ns,
                next_endpoint_due_ns,
            )
            if now_ns < endpoint_due_ns:
                if isinstance(active_execution, _AsyncExecution):
                    rate.wait()
                    # Re-enter through chunk ingest so a newer async result can
                    # replace this selected endpoint before publication.
                    continue
                else:
                    rate.wait()
                    continue
            step_index = active_execution.step_index
            active_observation_id = active_chunk.observation_id
            active_observation_anchor_ns = active_chunk.observation_anchor_monotonic_ns
            metrics.increment(ENDPOINTS_DUE)

            assert active_chunk.hand_qpos is not None
            hand_qpos = np.asarray(active_chunk.hand_qpos[step_index], dtype=np.float64)

            if active_chunk.is_ee:
                # EE -> joint via kinematics-only IK.  Real-time policy motion
                # intentionally does not run software collision checks.
                _arm_state = read_arm_state_dict(shared)
                if _arm_state is None:
                    metrics.increment(ENDPOINTS_TRANSIENT_DEFERRED)
                    rate.wait()
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
                        rate.wait()
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
                    consume_endpoint_slot(
                        endpoint_due_ns=endpoint_due_ns,
                        terminal_monotonic_ns=time.monotonic_ns(),
                    )
                    metrics.increment(ENDPOINTS_MOTION_DISCARDED)
                    try:
                        reached_limit = _record_terminal_before_finalize(
                            lambda: record_terminal_step(successful=False),
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
                    rate.wait()
                    continue
                try:
                    if ee_planner is None:
                        ee_planner = _build_policy_planner(config)
                    ik_result = ee_planner.solve_teleop_ik(
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
                    if failure_kind is IKFailureKind.INVALID_OUTPUT:
                        abort_and_reset(
                            "EE IK returned non-finite output",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    if failure_kind not in {
                        IKFailureKind.NO_SOLUTION,
                        IKFailureKind.GEOMETRY_REJECTED,
                    }:
                        abort_and_reset(
                            "EE IK returned an untyped failure",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    consume_endpoint_slot(
                        endpoint_due_ns=endpoint_due_ns,
                        terminal_monotonic_ns=time.monotonic_ns(),
                    )
                    metrics.increment(ENDPOINTS_MOTION_DISCARDED)
                    try:
                        reached_limit = _record_terminal_before_finalize(
                            lambda: record_terminal_step(successful=False),
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
                    rate.wait()
                    continue
                arm_qpos = np.asarray(ik_result.qpos, dtype=np.float64)
            else:
                assert active_chunk.arm_qpos is not None
                arm_qpos = np.asarray(
                    active_chunk.arm_qpos[step_index], dtype=np.float64
                )
                # Preserve joint-wrap continuity against the command stream.
                arm_reference = previous_arm_command_qpos
                if arm_reference is None:
                    _arm_state = read_arm_state_dict(shared)
                    if _arm_state is not None and np.all(
                        np.isfinite(_arm_state["qpos"])
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
                    rate.wait()
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
                    minimum_delivery_window_s=control_period_s,
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
                    publication_ns = time.monotonic_ns()
                    consume_endpoint_slot(
                        endpoint_due_ns=endpoint_due_ns,
                        terminal_monotonic_ns=publication_ns,
                        publication_monotonic_ns=publication_ns,
                    )
                    metrics.increment(ENDPOINTS_VALIDATED)
                    try:
                        last_valid_policy_command_ns = publication_ns
                        reached_limit = _record_terminal_before_finalize(
                            lambda: record_terminal_step(successful=True),
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
                    consume_endpoint_slot(
                        endpoint_due_ns=endpoint_due_ns,
                        terminal_monotonic_ns=publication_ns,
                        publication_monotonic_ns=publication_ns,
                    )
                    try:
                        command_progress.record_publication(
                            run_generation=published_candidate.run_generation,
                            action_id=published_candidate.action_id,
                            published_monotonic_ns=publication_ns,
                        )
                        last_valid_policy_command_ns = publication_ns
                        reached_limit = _record_terminal_before_finalize(
                            lambda: record_terminal_step(
                                successful=True,
                                wait_for_action_id=published_candidate.action_id,
                            ),
                            lambda: finish_active_endpoint(published_candidate),
                        )
                    except (RuntimeError, ValueError) as exc:
                        fault_physical(
                            f"command progress invariant failed: {exc}",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    logger.debug(
                        "coordinator: published action_id=%d; worker progress is asynchronous",
                        published_candidate.action_id,
                    )
                    if reached_limit:
                        continue
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
                    consume_endpoint_slot(
                        endpoint_due_ns=endpoint_due_ns,
                        terminal_monotonic_ns=time.monotonic_ns(),
                    )
                    metrics.increment(ENDPOINTS_MOTION_DISCARDED)
                    try:
                        reached_limit = _record_terminal_before_finalize(
                            lambda: record_terminal_step(successful=False),
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
            rate.wait()
    finally:
        metrics.log_episode_summary(
            status="INTERRUPTED",
            reason="coordinator loop exited",
        )
        logger.info("coordinator_loop: exited")

"""Deployment coordinator — the sole learned-policy robot-action producer.

The inference worker writes proposals to ``policy_plan_ring``; this coordinator
is the only process that turns a proposal into a robot command. It selects the
plan, schedules the due endpoint (one per control tick), runs the shared
candidate publication boundary (SafetyGate -> send_command), and owns the
policy semantic watchdog and the ``RUNNING <-> ARMED`` control-source state.

It never dumps a whole chunk into the arm queue or hand ring and never
interpolates between model steps.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real.config.defaults import arm as arm_defaults
from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.config.defaults import policy as policy_defaults
from dexmani_real.control.action import ActionCandidate
from dexmani_real.control.publication import (
    CommandPublishStatus,
    PolicyEndpointDisposition,
    build_action_candidate,
    classify_policy_endpoint_disposition,
    poll_coupled_command_acknowledgement,
    validate_and_send_candidate,
)
from dexmani_real.control.safety_gate import planner_action_safety_gate
from dexmani_real.deployment.action_buffer import (
    ActionBuffer,
    BufferCoverage,
    BufferedPlan,
    PushStatus,
    compute_max_buffered_plans,
)
from dexmani_real.deployment.config import (
    DeploymentConfig,
    H4ExecuteBounds,
    TaskExecuteBounds,
)
from dexmani_real.deployment.contracts import JointActionChunk
from dexmani_real.deployment.metrics import (
    COMMAND_SILENCE_ABORT,
    COUPLED_COMMAND_WRITES,
    ENDPOINTS_COALESCED,
    ENDPOINTS_COMMITTED,
    ENDPOINTS_DUE,
    ENDPOINTS_FATAL_REJECTED,
    ENDPOINTS_MOTION_DISCARDED,
    ENDPOINTS_PUBLISHED,
    ENDPOINTS_SHADOW_VALIDATED,
    ENDPOINTS_STALE_DISCARDED,
    ENDPOINTS_TRANSIENT_DEFERRED,
    EXECUTE_ACK_TIMEOUT,
    EXECUTE_ACKNOWLEDGED,
    EXECUTE_PUBLICATION_BOUND_REACHED,
    HAND_POLICY_ENDPOINT_ROUNDOFF_CANONICALIZED,
    HAND_PREFLIGHT_REJECTIONS,
    IK_CHECKER_REJECTS,
    PHYSICAL_HOME_COMPLETED,
    PLAN_AGE_MS,
    PLANS_EVICTED,
    PLANS_GENERATION_DROPPED,
    PLANS_INGESTED,
    PLANS_STALE,
    POLICY_ABORTS,
    SHADOW_COUPLED_WRITE_VIOLATIONS,
    USABLE_HORIZON_MS,
    Metrics,
    bounded_execute_run_receipt_json,
    flush_every,
    reject_counter_name,
    shadow_run_receipt_json,
)
from dexmani_real.deployment.timing import (
    compute_plan_deadline_ns,
    first_valid_index_from_prefix_mask,
    usable_target_mask,
)
from dexmani_real.ipc.channels import RuntimeChannels, read_arm_state_dict
from dexmani_real.ipc.schema import MAX_POLICY_CHUNK_STEPS
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
    begin_motion,
    read_run_epoch,
    revoke_motion,
)
from dexmani_real.utils.feedback import FeedbackIssueCode, diagnose_arm_feedback
from dexmani_real.utils.limits import validate_hand_limit_nesting
from dexmani_real.utils.log import get_logger, write_json_receipt

if TYPE_CHECKING:
    from dexmani_real.config.runtime import ResolvedRuntimeConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class _PendingExecuteAcknowledgement:
    """One physical command awaiting arm and hand worker acknowledgement."""

    candidate: ActionCandidate
    ticket: CoupledCommandTicket
    deadline_monotonic_ns: int


@dataclass(frozen=True)
class CoordinatorConfig:
    """Deployment config plus the safety/limits the coordinator needs to gate.

    Mirrors ``TeleopConfig.from_runtime``: the deployment namespace supplies the
    model/boundary knobs, the runtime namespace supplies the joint limits, hand
    mechanical envelope, and control rate.
    """

    deployment: DeploymentConfig
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
    execution_mode: str
    h4_execute_bounds: H4ExecuteBounds | None = None
    task_execute_bounds: TaskExecuteBounds | None = None
    execute_receipt_dir: str | None = None
    # Full 19-DoF collision model (hand + static boxes) for EE->IK and the
    # transition collision gate (Phase 6/7); table clearance is not part of the
    # policy safety gate.
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
        if self.execution_mode not in {"shadow", "execute", "task"}:
            raise ValueError("execution_mode must be 'shadow', 'execute', or 'task'")
        if self.execution_mode == "execute" and self.h4_execute_bounds is None:
            raise ValueError("execute coordinator requires explicit H4 execute bounds")
        if self.h4_execute_bounds is not None and not isinstance(
            self.h4_execute_bounds, H4ExecuteBounds
        ):
            raise TypeError("h4_execute_bounds must be H4ExecuteBounds or None")
        if self.execution_mode == "shadow" and self.h4_execute_bounds is not None:
            raise ValueError("shadow coordinator must not carry H4 execute bounds")
        if self.execution_mode != "execute" and self.h4_execute_bounds is not None:
            raise ValueError("only H4 execute may carry H4 execute bounds")
        if self.execution_mode == "task" and self.task_execute_bounds is None:
            raise ValueError("task coordinator requires explicit task execute bounds")
        if self.task_execute_bounds is not None and not isinstance(
            self.task_execute_bounds, TaskExecuteBounds
        ):
            raise TypeError("task_execute_bounds must be TaskExecuteBounds or None")
        if self.execution_mode != "task" and self.task_execute_bounds is not None:
            raise ValueError("only task execute may carry task execute bounds")
        if (
            self.execution_mode in {"execute", "task"}
            and not self.deployment.hand_enabled
        ):
            raise ValueError(
                "physical execute coordinator requires hand-enabled deployment"
            )
        if self.execution_mode in {"execute", "task"}:
            if self.required_start_arm_qpos is None:
                raise ValueError(
                    "physical execute coordinator requires canonical arm home"
                )
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

    @property
    def physical_execute_bounds(self) -> H4ExecuteBounds | TaskExecuteBounds | None:
        if self.execution_mode == "execute":
            return self.h4_execute_bounds
        if self.execution_mode == "task":
            return self.task_execute_bounds
        return None

    @classmethod
    def from_runtime(
        cls,
        deployment: DeploymentConfig,
        runtime: "ResolvedRuntimeConfig",
        *,
        execution_mode: str,
        h4_execute_bounds: H4ExecuteBounds | None,
        task_execute_bounds: TaskExecuteBounds | None = None,
        execute_receipt_dir: str | None = None,
    ) -> "CoordinatorConfig":
        return cls(
            deployment=deployment,
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
            execution_mode=execution_mode,
            h4_execute_bounds=h4_execute_bounds,
            task_execute_bounds=task_execute_bounds,
            execute_receipt_dir=execute_receipt_dir,
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
            required_start_arm_qpos=(
                tuple(runtime.arm.home_qpos)
                if execution_mode in {"execute", "task"}
                else None
            ),
            start_arm_home_tolerance_rad=(
                float(runtime.arm.homing.convergence_rad)
                if execution_mode in {"execute", "task"}
                else None
            ),
        )


def _read_latest_plan(shared: RuntimeChannels):
    """Return the latest plan record (scalar structured array) or None."""
    result = shared.policy_plan_ring.read_latest()
    if result is None:
        return None
    return result[0][0]


def _buffered_plan_from_record(
    rec: np.void,
    *,
    max_plan_age_ns: int,
    max_source_to_command_age_ns: int,
) -> BufferedPlan:
    """Copy and strictly validate one IPC plan at the scheduler boundary."""
    required = {
        "plan_id",
        "run_generation",
        "observation_id",
        "observation_latest_source_monotonic_ns",
        "observation_logical_step_monotonic_ns",
        "observation_anchor_monotonic_ns",
        "inference_started_monotonic_ns",
        "inference_finished_monotonic_ns",
        "num_steps",
        "arm_present",
        "ee_present",
        "hand_present",
        "arm_qpos",
        "hand_qpos",
        "ee_pos",
        "ee_rot6d",
        "target_monotonic_ns",
        "valid_mask",
    }
    names = set(getattr(getattr(rec, "dtype", None), "names", ()) or ())
    if not required.issubset(names):
        raise ValueError("policy plan record has an invalid IPC schema")
    n = int(rec["num_steps"])
    if not 0 < n <= MAX_POLICY_CHUNK_STEPS:
        raise ValueError("policy plan has an invalid num_steps")
    if int(rec["hand_present"]) != 1:
        raise ValueError("learned-policy plan must include hand targets")
    arm_present = int(rec["arm_present"])
    ee_present = int(rec["ee_present"])
    if (arm_present, ee_present) not in ((1, 0), (0, 1)):
        raise ValueError("policy plan must contain exactly one arm representation")
    deadline_ns = _plan_deadline_ns(
        rec,
        max_plan_age_ns=max_plan_age_ns,
        max_source_to_command_age_ns=max_source_to_command_age_ns,
    )
    if deadline_ns is None:
        raise ValueError("policy plan has non-causal timestamps")
    target = np.array(rec["target_monotonic_ns"][:n], dtype=np.uint64, copy=True)
    mask = np.array(rec["valid_mask"][:n], dtype=np.uint8, copy=True)
    first_index = first_valid_index_from_prefix_mask(mask)
    if first_index == n:
        raise ValueError("policy plan valid_mask must contain a deliverable suffix")
    hand_qpos = np.array(rec["hand_qpos"][:n], dtype=np.float64, copy=True)
    if not np.all(np.isfinite(hand_qpos)):
        raise ValueError("policy plan hand targets must be finite")
    if arm_present:
        arm_qpos = np.array(rec["arm_qpos"][:n], dtype=np.float64, copy=True)
        if not np.all(np.isfinite(arm_qpos)):
            raise ValueError("policy plan arm targets must be finite")
        chunk = JointActionChunk(arm_qpos, hand_qpos, target, mask)
    else:
        ee_pos = np.array(rec["ee_pos"][:n], dtype=np.float64, copy=True)
        ee_rot6d = np.array(rec["ee_rot6d"][:n], dtype=np.float64, copy=True)
        if not np.all(np.isfinite(ee_pos)) or not np.all(np.isfinite(ee_rot6d)):
            raise ValueError("policy plan EE targets must be finite")
        chunk = JointActionChunk(
            None, hand_qpos, target, mask, ee_pos=ee_pos, ee_rot6d=ee_rot6d
        )
    return BufferedPlan(
        plan_id=int(rec["plan_id"]),
        run_generation=int(rec["run_generation"]),
        observation_id=int(rec["observation_id"]),
        observation_anchor_ns=int(rec["observation_anchor_monotonic_ns"]),
        observation_latest_source_ns=int(rec["observation_latest_source_monotonic_ns"]),
        inference_finished_ns=int(rec["inference_finished_monotonic_ns"]),
        deadline_ns=deadline_ns,
        chunk=chunk,
    )


def _plan_deadline_ns(
    rec,
    *,
    max_plan_age_ns: int,
    max_source_to_command_age_ns: int,
) -> int | None:
    """Return the immutable expiry shared by a plan and its source observation."""
    finished_ns = int(rec["inference_finished_monotonic_ns"])
    started_ns = int(rec["inference_started_monotonic_ns"])
    source_ns = int(rec["observation_latest_source_monotonic_ns"])
    logical_ns = int(rec["observation_logical_step_monotonic_ns"])
    anchor_ns = int(rec["observation_anchor_monotonic_ns"])
    if (
        finished_ns <= 0
        or started_ns <= 0
        or source_ns <= 0
        or logical_ns <= 0
        or anchor_ns <= 0
        or not source_ns <= logical_ns <= anchor_ns <= started_ns <= finished_ns
    ):
        return None
    return compute_plan_deadline_ns(
        finished_ns,
        source_ns,
        max_plan_age_ns,
        max_source_to_command_age_ns,
    )


def _usable_horizon_ms(plan: BufferedPlan, *, now_ns: int) -> float:
    """Return the remaining actionable span of one immutable plan.

    The result is bounded by both the plan/source deadline and its latest valid
    logical target. It therefore measures useful future coverage rather than
    the raw model horizon, and cannot become negative for an already-expired
    endpoint.
    """
    first_index = first_valid_index_from_prefix_mask(plan.chunk.valid_mask)
    usable = usable_target_mask(
        plan.chunk.target_monotonic_ns,
        first_index,
        plan.deadline_ns,
    )
    if not bool(np.any(usable)):
        return 0.0
    latest_target_ns = int(np.max(plan.chunk.target_monotonic_ns[usable == 1]))
    return max(0.0, latest_target_ns - int(now_ns)) / 1e6


def _coupled_command_sequence(shared: RuntimeChannels) -> int:
    """Read the monotonic coupled-ring sequence used by a shadow receipt."""
    sequence = int(shared.coupled_cmd_ring.latest_sequence)
    if sequence < 0:
        raise ValueError("coupled command ring sequence must be non-negative")
    return sequence


def _physical_start_pose_rejection(
    shared: RuntimeChannels,
    config: CoordinatorConfig,
) -> str | None:
    """Return why B cannot open a physical epoch, or ``None`` at arm home."""
    if config.execution_mode not in {"execute", "task"}:
        return None
    if not bool(shared.physical_home_completed.value):
        return (
            "physical home sequence has not completed in this process; "
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
) -> None:
    """Advance the generation and drop RUNNING -> ARMED.

    Both a clean operator STOP (``abort=False``) and a policy-semantic abort
    (``abort=True``) leave the robot ARMED (command quiescence), never FAULT.
    Abort counters are flushed immediately because the success-path
    ``flush_every`` is never reached once a run ends (the loop idles in ARMED),
    and the H0 gate reads these counters.
    """
    lifecycle_faulted = (
        bool(shared.error_state.value)
        or bool(shared.estop_request.value)
        or int(shared.safety_state.value) == int(SafetyState.FAULT)
    )
    if not lifecycle_faulted and not revoke_motion(shared, SafetyState.ARMED):
        logger.error("coordinator: failed to transition RUNNING->ARMED (%s)", reason)
    if abort:
        logger.warning("coordinator: policy run aborted: %s", reason)
        if metrics is not None:
            metrics.increment(POLICY_ABORTS)
            if metric is not None:
                metrics.increment(metric)
            metrics.flush(prefix="coordinator metrics")
    else:
        logger.info("coordinator: policy run stopped: %s", reason)


def coordinator_loop(shared: RuntimeChannels, config: CoordinatorConfig) -> None:
    """Coordinator process entry point — the only robot-action producer.

    Idles in ARMED until the operator presses B (``start_request``), runs one
    policy episode in RUNNING, then returns to ARMED on S (``stop_request``) or
    a policy-semantic abort.  Each B advances the run generation, so any
    in-flight plan or command from a previous run is invalid at the worker.
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
    max_plan_age_ns = int(config.deployment.max_plan_age_s * 1e9)
    max_source_to_command_age_ns = int(
        config.deployment.max_source_to_command_age_s * 1e9
    )
    max_silence_ns = int(config.deployment.max_command_silence_s * 1e9)
    first_command_timeout_ns = int(config.deployment.first_command_timeout_s * 1e9)
    action_buffer = ActionBuffer(
        max_buffered_plans=compute_max_buffered_plans(
            config.deployment.max_plan_age_s,
            config.deployment.inference_hz,
        )
    )
    buffer_generation: int | None = None
    last_seen_plan_key: tuple[int, int] | None = None
    # Silence timeout starts at the first published command, not first inference.
    last_valid_policy_command_ns: int | None = None
    # RUNNING start time, for the first-command timeout.
    run_started_ns: int | None = None
    shadow_run_generation: int | None = None
    shadow_start_coupled_sequence: int | None = None
    execute_run_generation: int | None = None
    execute_start_coupled_sequence: int | None = None
    execute_first_publication_ns: int | None = None
    execute_last_publication_ns: int | None = None
    execute_published_endpoints = 0
    execute_acknowledged_action_id: int | None = None
    execute_pending_acknowledgement: _PendingExecuteAcknowledgement | None = None
    # One process carries one operator-authorized policy session. Physical
    # modes exit after it; shadow remains ARMED for Q but cannot start again.
    policy_session_started = False
    previous_arm_command_qpos: np.ndarray | None = None
    last_metrics_flush_ns = time.monotonic_ns()

    def emit_shadow_receipt(reason: str) -> None:
        """Log one run-total shadow receipt after B has opened an epoch."""
        nonlocal shadow_run_generation, shadow_start_coupled_sequence
        if shadow_run_generation is None or shadow_start_coupled_sequence is None:
            return
        try:
            end_sequence = _coupled_command_sequence(shared)
            receipt = shadow_run_receipt_json(
                run_generation=shadow_run_generation,
                reason=reason,
                coupled_command_start_sequence=shadow_start_coupled_sequence,
                coupled_command_end_sequence=end_sequence,
                metrics=metrics.run_snapshot(),
            )
        except Exception:
            logger.critical("coordinator: cannot render shadow receipt", exc_info=True)
        else:
            logger.info("shadow run receipt: %s", receipt)
            if end_sequence != shadow_start_coupled_sequence:
                logger.critical(
                    "coordinator: shadow run changed coupled command sequence "
                    "(%d -> %d)",
                    shadow_start_coupled_sequence,
                    end_sequence,
                )
        finally:
            shadow_run_generation = None
            shadow_start_coupled_sequence = None

    def emit_execute_receipt(reason: str) -> None:
        """Log one bounded physical-execution receipt after B opened an epoch."""
        nonlocal execute_run_generation, execute_start_coupled_sequence
        if execute_run_generation is None or execute_start_coupled_sequence is None:
            return
        execute_bounds = config.physical_execute_bounds
        assert execute_bounds is not None
        try:
            end_sequence = _coupled_command_sequence(shared)
            receipt = bounded_execute_run_receipt_json(
                execution_mode=config.execution_mode,
                run_generation=execute_run_generation,
                reason=reason,
                coupled_command_start_sequence=execute_start_coupled_sequence,
                coupled_command_end_sequence=end_sequence,
                max_published_endpoints=execute_bounds.max_published_endpoints,
                acknowledgement_timeout_s=execute_bounds.acknowledgement_timeout_s,
                acknowledged_action_id=execute_acknowledged_action_id,
                completed=bool(shared.execute_completed.value),
                metrics=metrics.run_snapshot(),
                timeline_monotonic_ns={
                    name: value
                    for name, value in (
                        ("run_started", run_started_ns),
                        ("first_publication", execute_first_publication_ns),
                        ("last_publication", execute_last_publication_ns),
                        ("receipt_emitted", time.monotonic_ns()),
                    )
                    if value is not None
                },
            )
        except Exception:
            # Receipt rendering is part of the same acceptance boundary as
            # persistence. Never leave a successful command marked complete
            # when its terminal evidence cannot be produced.
            shared.execute_completed.value = False
            shared.error_state.value = True
            revoke_motion(shared, SafetyState.FAULT)
            logger.critical("coordinator: cannot render execute receipt", exc_info=True)
        else:
            logger.info("execute run receipt: %s", receipt)
            if config.execute_receipt_dir is not None:
                try:
                    receipt_path = write_json_receipt(
                        config.execute_receipt_dir, receipt
                    )
                except Exception:
                    # Receipt persistence is part of the acceptance
                    # boundary. Do not report a clean execute without it.
                    shared.execute_completed.value = False
                    shared.error_state.value = True
                    revoke_motion(shared, SafetyState.FAULT)
                    logger.critical(
                        "coordinator: cannot persist physical execute receipt",
                        exc_info=True,
                    )
                else:
                    logger.info("physical execute receipt written: %s", receipt_path)
        finally:
            execute_run_generation = None
            execute_start_coupled_sequence = None

    def emit_run_receipt(reason: str) -> None:
        """Emit the receipt appropriate for the active execution boundary."""
        if config.execution_mode == "shadow":
            emit_shadow_receipt(reason)
        else:
            emit_execute_receipt(reason)

    def fault_execute(reason: str, *, metric: str) -> None:
        """Latch FAULT for a physical acknowledgement/boundary failure."""
        nonlocal buffer_generation, last_seen_plan_key, execute_pending_acknowledgement
        shared.execute_completed.value = False
        shared.error_state.value = True
        if not revoke_motion(shared, SafetyState.FAULT):
            logger.critical("coordinator: unable to latch FAULT after execute failure")
        logger.critical("coordinator: physical execute failure: %s", reason)
        metrics.increment(POLICY_ABORTS)
        metrics.increment(metric)
        metrics.flush(prefix="coordinator metrics")
        emit_execute_receipt(reason)
        execute_pending_acknowledgement = None
        action_buffer.reset(run_generation=int(shared.run_generation.value))
        buffer_generation = None
        last_seen_plan_key = None

    def abort_and_reset(reason: str, *, metric: str) -> None:
        """Fail closed and synchronously invalidate every buffered endpoint."""
        nonlocal buffer_generation, last_seen_plan_key, execute_pending_acknowledgement
        if config.execution_mode in {"execute", "task"}:
            fault_execute(reason, metric=metric)
            return
        _end_policy_run(shared, reason, abort=True, metrics=metrics, metric=metric)
        emit_run_receipt(reason)
        execute_pending_acknowledgement = None
        action_buffer.reset(run_generation=int(shared.run_generation.value))
        buffer_generation = None
        last_seen_plan_key = None

    def fault_shadow_integrity(*, reason: str, write_violation: bool) -> None:
        """Latch FAULT when a shadow no-write proof is violated or unavailable.

        Unlike an ordinary learned endpoint reject, this means either a
        structural no-write invariant has already been broken or the runtime
        cannot establish the ring-sequence evidence needed to prove it. Leaving
        the runtime ARMED would permit a second B and hide the evidence, so this
        path must revoke the permit directly to sticky FAULT.
        """
        nonlocal buffer_generation, last_seen_plan_key
        shared.error_state.value = True
        if not revoke_motion(shared, SafetyState.FAULT):
            logger.critical("coordinator: unable to latch FAULT after shadow failure")
        logger.critical("coordinator: shadow integrity failure: %s", reason)
        if write_violation:
            metrics.increment(SHADOW_COUPLED_WRITE_VIOLATIONS)
        metrics.increment(POLICY_ABORTS)
        metrics.increment(ENDPOINTS_FATAL_REJECTED)
        metrics.flush(prefix="coordinator metrics")
        if shadow_start_coupled_sequence is None:
            logger.critical("coordinator: shadow receipt unavailable: %s", reason)
        else:
            emit_shadow_receipt(reason)
        action_buffer.reset(run_generation=int(shared.run_generation.value))
        buffer_generation = None
        last_seen_plan_key = None

    def fault_shadow_coupled_write() -> None:
        """Latch FAULT after any post-B shadow coupled-ring mutation."""
        fault_shadow_integrity(
            reason="shadow coupled-command write detected",
            write_violation=True,
        )

    try:
        while shared.is_running.value:
            tick_start = time.monotonic()
            now_ns = time.monotonic_ns()
            shared.set_heartbeat("policy", time.monotonic())

            if bool(shared.quit_requested.value):
                if int(shared.safety_state.value) == int(SafetyState.RUNNING):
                    _end_policy_run(shared, "operator quit", abort=False)
                if config.execution_mode in {"execute", "task"}:
                    execute_pending_acknowledgement = None
                    emit_run_receipt("operator quit")
                return

            if bool(shared.error_state.value) or bool(shared.estop_request.value):
                if config.execution_mode in {"execute", "task"}:
                    emit_execute_receipt(
                        "e-stop requested"
                        if bool(shared.estop_request.value)
                        else "error_state set"
                    )
                if buffer_generation is not None:
                    action_buffer.reset(run_generation=int(shared.run_generation.value))
                    buffer_generation = None
                    last_seen_plan_key = None
                _sleep_tick(period_s, tick_start)
                continue

            # ARMED idle: wait for the operator to request a new run (B).
            if int(shared.safety_state.value) != int(SafetyState.RUNNING):
                if buffer_generation is not None:
                    action_buffer.reset(run_generation=int(shared.run_generation.value))
                    buffer_generation = None
                    last_seen_plan_key = None
                execute_pending_acknowledgement = None
                if not bool(shared.start_request.value):
                    _sleep_tick(period_s, tick_start)
                    continue
                shared.start_request.value = False
                if policy_session_started:
                    logger.warning(
                        "coordinator: ignored B after the policy session already started"
                    )
                    _sleep_tick(period_s, tick_start)
                    continue
                start_pose_rejection = _physical_start_pose_rejection(shared, config)
                if start_pose_rejection is not None:
                    logger.warning("coordinator: ignored B: %s", start_pose_rejection)
                    _sleep_tick(period_s, tick_start)
                    continue
                # A stray S from ARMED must not stop the freshly started run.
                shared.stop_request.value = False
                if not begin_motion(shared):
                    logger.error(
                        "coordinator: cannot enter RUNNING (safety_state=%d)",
                        int(shared.safety_state.value),
                    )
                    return
                logger.info(
                    "coordinator_loop: RUNNING (run_generation=%d)",
                    int(shared.run_generation.value),
                )
                policy_session_started = True
                if config.execution_mode in {"execute", "task"}:
                    shared.execute_completed.value = False
                last_valid_policy_command_ns = None
                run_epoch = read_run_epoch(shared)
                if (
                    run_epoch.generation != int(shared.run_generation.value)
                    or run_epoch.started_monotonic_ns <= 0
                ):
                    abort_and_reset(
                        "invalid run epoch", metric=ENDPOINTS_FATAL_REJECTED
                    )
                    continue
                run_started_ns = run_epoch.started_monotonic_ns
                previous_arm_command_qpos = None
                metrics.begin_run()
                if config.execution_mode in {"execute", "task"}:
                    metrics.increment(PHYSICAL_HOME_COMPLETED)
                execute_published_endpoints = 0
                execute_acknowledged_action_id = None
                execute_pending_acknowledgement = None
                execute_first_publication_ns = None
                execute_last_publication_ns = None
                try:
                    start_coupled_sequence = _coupled_command_sequence(shared)
                except Exception as exc:
                    if config.execution_mode == "shadow":
                        fault_shadow_integrity(
                            reason=(
                                "cannot establish shadow coupled-write baseline: "
                                f"{type(exc).__name__}"
                            ),
                            write_violation=False,
                        )
                        continue
                    fault_execute(
                        "cannot establish physical coupled-write baseline: "
                        f"{type(exc).__name__}",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if config.execution_mode == "shadow":
                    shadow_start_coupled_sequence = start_coupled_sequence
                    shadow_run_generation = run_epoch.generation
                else:
                    execute_start_coupled_sequence = start_coupled_sequence
                    execute_run_generation = run_epoch.generation
                action_buffer.reset(run_generation=run_epoch.generation)
                buffer_generation = run_epoch.generation
                last_seen_plan_key = None
                _sleep_tick(period_s, tick_start)
                continue

            # RUNNING: an operator stop ends cleanly; a bounded-run time-limit
            # stop is clean for shadow and fail-closed for physical execute without proof of
            # the required publication/acknowledgement boundary.
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
                if (
                    stop_request is StopRequest.RUN_TIME_LIMIT
                    and config.execution_mode in {"execute", "task"}
                ):
                    if execute_pending_acknowledgement is not None:
                        fault_execute(
                            "run time limit reached before worker acknowledgement",
                            metric=EXECUTE_ACK_TIMEOUT,
                        )
                    elif execute_published_endpoints == 0:
                        fault_execute(
                            "run time limit reached before first publication",
                            metric=COMMAND_SILENCE_ABORT,
                        )
                    else:
                        fault_execute(
                            "publication bound not completed before run time limit",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                    continue
                _end_policy_run(shared, stop_reason, abort=False)
                execute_pending_acknowledgement = None
                emit_run_receipt(stop_reason)
                action_buffer.reset(run_generation=int(shared.run_generation.value))
                buffer_generation = None
                last_seen_plan_key = None
                _sleep_tick(period_s, tick_start)
                continue

            if buffer_generation != int(shared.run_generation.value):
                # A lifecycle epoch invalidated the previous scheduler before
                # this tick; never let a stale endpoint survive the boundary.
                action_buffer.reset(run_generation=int(shared.run_generation.value))
                buffer_generation = int(shared.run_generation.value)
                last_seen_plan_key = None

            if shadow_start_coupled_sequence is not None:
                try:
                    current_coupled_sequence = _coupled_command_sequence(shared)
                except Exception as exc:
                    fault_shadow_integrity(
                        reason=(
                            "cannot inspect shadow coupled-command sequence: "
                            f"{type(exc).__name__}"
                        ),
                        write_violation=False,
                    )
                    continue
                if current_coupled_sequence != shadow_start_coupled_sequence:
                    fault_shadow_coupled_write()
                    continue

            if execute_pending_acknowledgement is not None:
                acknowledgement = poll_coupled_command_acknowledgement(
                    shared,
                    execute_pending_acknowledgement.candidate,
                    ticket=execute_pending_acknowledgement.ticket,
                    arm_feedback_max_age_s=config.arm_feedback_max_age_s,
                    hand_feedback_max_age_s=config.hand_feedback_max_age_s,
                )
                acknowledgement_observed_ns = time.monotonic_ns()
                if acknowledgement.status is CommandPublishStatus.APPLIED:
                    if (
                        acknowledgement_observed_ns
                        > execute_pending_acknowledgement.deadline_monotonic_ns
                    ):
                        fault_execute(
                            "worker acknowledgement arrived after deadline",
                            metric=EXECUTE_ACK_TIMEOUT,
                        )
                        _sleep_tick(period_s, tick_start)
                        continue
                    execute_acknowledged_action_id = int(
                        execute_pending_acknowledgement.candidate.action_id
                    )
                    metrics.increment(EXECUTE_ACKNOWLEDGED)
                    execute_pending_acknowledgement = None
                    execute_bounds = config.physical_execute_bounds
                    assert execute_bounds is not None
                    if (
                        execute_published_endpoints
                        >= execute_bounds.max_published_endpoints
                    ):
                        shared.execute_completed.value = True
                        metrics.increment(EXECUTE_PUBLICATION_BOUND_REACHED)
                        reason = (
                            "H4 publication bound reached"
                            if config.execution_mode == "execute"
                            else "task publication bound reached"
                        )
                        _end_policy_run(shared, reason, abort=False)
                        emit_execute_receipt(reason)
                        action_buffer.reset(
                            run_generation=int(shared.run_generation.value)
                        )
                        buffer_generation = None
                        last_seen_plan_key = None
                        _sleep_tick(period_s, tick_start)
                        continue
                    # A non-final task ACK permits the next due endpoint in
                    # this same control tick; adding another sleep here would
                    # halve the 16 Hz execution rate.
                elif acknowledgement.status is CommandPublishStatus.ACK_PENDING:
                    if now_ns < execute_pending_acknowledgement.deadline_monotonic_ns:
                        _sleep_tick(period_s, tick_start)
                        continue
                    fault_execute(
                        "arm/hand acknowledgement timeout",
                        metric=EXECUTE_ACK_TIMEOUT,
                    )
                    continue
                else:
                    fault_execute(
                        "arm/hand acknowledgement failed: "
                        f"{acknowledgement.status.value}",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue

            if config.execution_mode in {"execute", "task"}:
                execute_bounds = config.physical_execute_bounds
                assert execute_bounds is not None
                if (
                    execute_published_endpoints
                    >= execute_bounds.max_published_endpoints
                ):
                    # A successful final publication must leave a pending
                    # acknowledgement. Reaching this branch means that invariant
                    # was lost; do not permit another action to be built or sent.
                    fault_execute(
                        "publication bound reached without acknowledgement state",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue

            # Abort a run that never produced its first command (the model
            # dropped every plan); command-to-command silence is checked below.
            if (
                last_valid_policy_command_ns is None
                and run_started_ns is not None
                and now_ns - run_started_ns > first_command_timeout_ns
            ):
                abort_and_reset("first command timeout", metric=COMMAND_SILENCE_ABORT)
                continue

            # Watch command-to-command silence; first-inference latency is exempt.
            if (
                last_valid_policy_command_ns is not None
                and now_ns - last_valid_policy_command_ns > max_silence_ns
            ):
                abort_and_reset("command silence timeout", metric=COMMAND_SILENCE_ABORT)
                continue

            rec = _read_latest_plan(shared)
            if rec is not None:
                key = (int(rec["run_generation"]), int(rec["plan_id"]))
                if key != last_seen_plan_key:
                    last_seen_plan_key = key
                    try:
                        buffered_plan = _buffered_plan_from_record(
                            rec,
                            max_plan_age_ns=max_plan_age_ns,
                            max_source_to_command_age_ns=max_source_to_command_age_ns,
                        )
                    except Exception as exc:
                        abort_and_reset(
                            f"invalid policy plan IPC record: {exc}",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    admitted = action_buffer.push(buffered_plan, now_ns=now_ns)
                    if admitted.accepted:
                        metrics.increment(PLANS_INGESTED)
                        plan_age_ms = max(
                            0.0,
                            (now_ns - buffered_plan.inference_finished_ns) / 1e6,
                        )
                        metrics.observe(PLAN_AGE_MS, plan_age_ms)
                        metrics.observe_timing(PLAN_AGE_MS, plan_age_ms)
                        if admitted.evicted_count:
                            metrics.increment(PLANS_EVICTED, admitted.evicted_count)
                    elif admitted.status is PushStatus.WRONG_GENERATION:
                        metrics.increment(PLANS_GENERATION_DROPPED)
                    else:
                        metrics.increment(PLANS_STALE)

            selected = action_buffer.peek_due(now_ns=now_ns)
            if selected.coverage is not BufferCoverage.DUE:
                _sleep_tick(period_s, tick_start)
                continue
            assert (
                selected.plan is not None
                and selected.step_index is not None
                and selected.token is not None
            )
            active_plan = selected.plan
            step_index = selected.step_index
            endpoint_token = selected.token
            metrics.increment(ENDPOINTS_DUE)
            usable_horizon_ms = _usable_horizon_ms(active_plan, now_ns=now_ns)
            metrics.observe(USABLE_HORIZON_MS, usable_horizon_ms)
            metrics.observe_timing(USABLE_HORIZON_MS, usable_horizon_ms)
            if selected.coalesced_count:
                metrics.increment(ENDPOINTS_COALESCED, selected.coalesced_count)

            assert active_plan.chunk.hand_qpos is not None
            hand_qpos = np.asarray(
                active_plan.chunk.hand_qpos[step_index], dtype=np.float64
            )

            _arm_state = read_arm_state_dict(shared)
            if active_plan.chunk.is_ee:
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
                assert active_plan.chunk.ee_pos is not None
                assert active_plan.chunk.ee_rot6d is not None
                ee_pos = np.asarray(
                    active_plan.chunk.ee_pos[step_index], dtype=np.float64
                )
                ee_rot6d = np.asarray(
                    active_plan.chunk.ee_rot6d[step_index], dtype=np.float64
                )
                try:
                    validate_rot6d_geometry(ee_rot6d, label="policy ee_rot6d")
                except ValueError:
                    try:
                        action_buffer.discard(
                            endpoint_token,
                            reason_code="ee_rot6d_geometry",
                        )
                    except RuntimeError:
                        abort_and_reset(
                            "cannot discard malformed EE endpoint",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    metrics.increment(ENDPOINTS_MOTION_DISCARDED)
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
                    try:
                        action_buffer.discard(
                            endpoint_token,
                            reason_code=failure_kind.value,
                        )
                    except RuntimeError:
                        abort_and_reset(
                            "cannot discard rejected EE endpoint",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    metrics.increment(ENDPOINTS_MOTION_DISCARDED)
                    _sleep_tick(period_s, tick_start)
                    continue
                arm_qpos = np.asarray(ik_result.qpos, dtype=np.float64)
            else:
                assert active_plan.chunk.arm_qpos is not None
                arm_qpos = np.asarray(
                    active_plan.chunk.arm_qpos[step_index], dtype=np.float64
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
                    is_hold=False,
                    observation_id=active_plan.observation_id,
                    observation_anchor_monotonic_ns=active_plan.observation_anchor_ns,
                    scheduled_target_monotonic_ns=int(
                        active_plan.chunk.target_monotonic_ns[step_index]
                    ),
                    action_validity_s=float(config.deployment.action_validity_s),
                    valid_until_monotonic_ns=active_plan.deadline_ns,
                )
            except (TypeError, ValueError) as exc:
                abort_and_reset(
                    f"candidate contract failure: {type(exc).__name__}",
                    metric=ENDPOINTS_FATAL_REJECTED,
                )
                continue
            if candidate is None:
                if time.monotonic_ns() >= active_plan.deadline_ns:
                    try:
                        action_buffer.discard(
                            endpoint_token,
                            reason_code="temporal_window_closed",
                        )
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
                    execution_mode=(
                        "shadow" if config.execution_mode == "shadow" else "execute"
                    ),
                    # Leave one full policy tick for both 30 Hz workers to
                    # observe the coupled record. Near-expiry plans are stale;
                    # never extend their immutable source deadline.
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
                shadow_validated = (
                    publish_result.status is CommandPublishStatus.SHADOW_VALIDATED
                )
                if (config.execution_mode == "shadow") != shadow_validated:
                    abort_and_reset(
                        "execution mode and publication result disagree",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if (
                    config.execution_mode in {"execute", "task"}
                    and publish_result.status is not CommandPublishStatus.PUBLISHED
                ):
                    abort_and_reset(
                        "physical execution did not return a publication ticket",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                try:
                    current_coupled_sequence = _coupled_command_sequence(shared)
                except Exception as exc:
                    if config.execution_mode == "shadow":
                        fault_shadow_integrity(
                            reason=(
                                "cannot inspect coupled-command sequence after "
                                f"validation: {type(exc).__name__}"
                            ),
                            write_violation=False,
                        )
                    else:
                        fault_execute(
                            "cannot inspect physical coupled-command sequence after "
                            f"validation: {type(exc).__name__}",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                    continue
                if shadow_start_coupled_sequence is not None:
                    if current_coupled_sequence != shadow_start_coupled_sequence:
                        fault_shadow_coupled_write()
                        continue
                elif config.execution_mode in {"execute", "task"}:
                    assert execute_start_coupled_sequence is not None
                    expected_sequence = (
                        execute_start_coupled_sequence + execute_published_endpoints + 1
                    )
                    if current_coupled_sequence != expected_sequence:
                        fault_execute(
                            "physical publication changed coupled-command sequence by "
                            f"{current_coupled_sequence - execute_start_coupled_sequence}, "
                            f"expected exactly {execute_published_endpoints + 1}",
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
                try:
                    action_buffer.commit(endpoint_token)
                except RuntimeError:
                    abort_and_reset(
                        "published endpoint could not be committed",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if shadow_validated:
                    metrics.increment(ENDPOINTS_SHADOW_VALIDATED)
                else:
                    metrics.increment(ENDPOINTS_PUBLISHED)
                    metrics.increment(COUPLED_COMMAND_WRITES)
                    ticket = publish_result.ticket
                    if ticket is None:
                        fault_execute(
                            "physical publication omitted its coupled command ticket",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    execute_published_endpoints += 1
                    publication_ns = time.monotonic_ns()
                    if execute_first_publication_ns is None:
                        execute_first_publication_ns = publication_ns
                    execute_last_publication_ns = publication_ns
                    execute_bounds = config.physical_execute_bounds
                    assert execute_bounds is not None
                    if (
                        execute_published_endpoints
                        > execute_bounds.max_published_endpoints
                    ):
                        fault_execute(
                            "physical publication count exceeded immutable bound",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    execute_pending_acknowledgement = _PendingExecuteAcknowledgement(
                        candidate=published_candidate,
                        ticket=ticket,
                        deadline_monotonic_ns=(
                            time.monotonic_ns()
                            + int(execute_bounds.acknowledgement_timeout_s * 1e9)
                        ),
                    )
                    logger.info(
                        "coordinator: %s published action_id=%d (%d/%d); awaiting worker acknowledgement",
                        config.execution_mode,
                        published_candidate.action_id,
                        execute_published_endpoints,
                        execute_bounds.max_published_endpoints,
                    )
                metrics.increment(ENDPOINTS_COMMITTED)
                previous_arm_command_qpos = np.asarray(
                    published_candidate.arm_qpos, dtype=np.float64
                ).copy()
                last_valid_policy_command_ns = now_ns
            elif disposition in {
                PolicyEndpointDisposition.DISCARD_MOTION,
                PolicyEndpointDisposition.DISCARD_STALE,
            }:
                if publish_result.status is CommandPublishStatus.GATE_REJECTED:
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
                reason_code = (
                    publish_result.gate_code.value
                    if publish_result.gate_code is not None
                    else publish_result.status.value
                )
                try:
                    action_buffer.discard(endpoint_token, reason_code=reason_code)
                except RuntimeError:
                    abort_and_reset(
                        "rejected endpoint could not be finalized",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
                if disposition is PolicyEndpointDisposition.DISCARD_MOTION:
                    metrics.increment(ENDPOINTS_MOTION_DISCARDED)
                else:
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
        emit_run_receipt("coordinator exit")
        logger.info("coordinator_loop: exited")


def _sleep_tick(period_s: float, tick_start: float) -> None:
    """Sleep for the remainder of one control tick, if any."""
    sleep_s = period_s - (time.monotonic() - tick_start)
    if sleep_s > 0:
        time.sleep(sleep_s)

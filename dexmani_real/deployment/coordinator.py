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
from dexmani_real.deployment.config import DeploymentConfig, H4ExecuteBounds
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
    PLAN_AGE_MS,
    PLANS_EVICTED,
    PLANS_GENERATION_DROPPED,
    PLANS_INGESTED,
    PLANS_STALE,
    POLICY_ABORTS,
    SHADOW_COUPLED_WRITE_VIOLATIONS,
    USABLE_HORIZON_MS,
    Metrics,
    execute_run_receipt_json,
    flush_every,
    reject_counter_name,
    shadow_run_receipt_json,
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
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.config.runtime import ResolvedRuntimeConfig

logger = get_logger(__name__)


@dataclass(frozen=True)
class _H4PendingAcknowledgement:
    """One published H4 command awaiting worker acknowledgement."""

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
    endpoint_delta_tolerance_rad: float = policy_defaults.endpoint_delta_tolerance_rad

    def __post_init__(self) -> None:
        if self.execution_mode not in {"shadow", "execute"}:
            raise ValueError("execution_mode must be 'shadow' or 'execute'")
        if self.execution_mode == "execute" and self.h4_execute_bounds is None:
            raise ValueError("execute coordinator requires explicit H4 execute bounds")
        if self.execution_mode == "shadow" and self.h4_execute_bounds is not None:
            raise ValueError("shadow coordinator must not carry H4 execute bounds")
        if self.execution_mode == "execute" and not self.deployment.hand_enabled:
            raise ValueError("H4 execute coordinator requires hand-enabled deployment")

    @classmethod
    def from_runtime(
        cls,
        deployment: DeploymentConfig,
        runtime: "ResolvedRuntimeConfig",
        *,
        execution_mode: str,
        h4_execute_bounds: H4ExecuteBounds | None,
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
            static_boxes=tuple(runtime.environment.static_boxes),
            ik_max_pose_error_pos_m=float(runtime.policy.ik_max_pose_error_pos_m),
            ik_max_pose_error_rot_rad=float(runtime.policy.ik_max_pose_error_rot_rad),
            arm_max_delta_rad_per_tick=float(runtime.arm.max_servo_command_jump_rad),
            hand_max_delta_rad_per_tick=None,
            endpoint_delta_tolerance_rad=float(
                runtime.policy.endpoint_delta_tolerance_rad
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
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError("policy plan valid_mask must contain only 0 or 1")
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
    return min(
        finished_ns + int(max_plan_age_ns),
        source_ns + int(max_source_to_command_age_ns),
    )


def _usable_horizon_ms(plan: BufferedPlan, *, now_ns: int) -> float:
    """Return the remaining actionable span of one immutable plan.

    The result is bounded by both the plan/source deadline and its latest valid
    logical target. It therefore measures useful future coverage rather than
    the raw model horizon, and cannot become negative for an already-expired
    endpoint.
    """
    valid = np.asarray(plan.chunk.valid_mask, dtype=np.uint8) == 1
    if not np.any(valid):
        return 0.0
    latest_target_ns = int(np.max(plan.chunk.target_monotonic_ns[valid]))
    return max(0.0, min(plan.deadline_ns, latest_target_ns) - int(now_ns)) / 1e6


def _coupled_command_sequence(shared: RuntimeChannels) -> int:
    """Read the monotonic coupled-ring sequence used by a shadow receipt."""
    sequence = int(shared.coupled_cmd_ring.latest_sequence)
    if sequence < 0:
        raise ValueError("coupled command ring sequence must be non-negative")
    return sequence


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
    execute_published_endpoints = 0
    execute_acknowledged_action_id: int | None = None
    h4_pending_acknowledgement: _H4PendingAcknowledgement | None = None
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
        """Log one bounded H4 execute receipt after B has opened an epoch."""
        nonlocal execute_run_generation, execute_start_coupled_sequence
        if execute_run_generation is None or execute_start_coupled_sequence is None:
            return
        assert config.h4_execute_bounds is not None
        try:
            end_sequence = _coupled_command_sequence(shared)
            receipt = execute_run_receipt_json(
                run_generation=execute_run_generation,
                reason=reason,
                coupled_command_start_sequence=execute_start_coupled_sequence,
                coupled_command_end_sequence=end_sequence,
                max_published_endpoints=config.h4_execute_bounds.max_published_endpoints,
                acknowledgement_timeout_s=(
                    config.h4_execute_bounds.acknowledgement_timeout_s
                ),
                acknowledged_action_id=execute_acknowledged_action_id,
                metrics=metrics.run_snapshot(),
            )
        except Exception:
            logger.critical("coordinator: cannot render execute receipt", exc_info=True)
        else:
            logger.info("execute run receipt: %s", receipt)
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
        """Latch FAULT for an H4 acknowledgement/boundary failure."""
        nonlocal buffer_generation, last_seen_plan_key, h4_pending_acknowledgement
        shared.error_state.value = True
        if not revoke_motion(shared, SafetyState.FAULT):
            logger.critical("coordinator: unable to latch FAULT after H4 failure")
        logger.critical("coordinator: H4 execute failure: %s", reason)
        metrics.increment(POLICY_ABORTS)
        metrics.increment(metric)
        metrics.flush(prefix="coordinator metrics")
        emit_execute_receipt(reason)
        h4_pending_acknowledgement = None
        action_buffer.reset(run_generation=int(shared.run_generation.value))
        buffer_generation = None
        last_seen_plan_key = None

    def abort_and_reset(reason: str, *, metric: str) -> None:
        """Fail closed and synchronously invalidate every buffered endpoint."""
        nonlocal buffer_generation, last_seen_plan_key, h4_pending_acknowledgement
        if config.execution_mode == "execute":
            fault_execute(reason, metric=metric)
            return
        _end_policy_run(shared, reason, abort=True, metrics=metrics, metric=metric)
        emit_run_receipt(reason)
        h4_pending_acknowledgement = None
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
                    h4_pending_acknowledgement = None
                    emit_run_receipt("operator quit")
                return

            if bool(shared.error_state.value) or bool(shared.estop_request.value):
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
                h4_pending_acknowledgement = None
                if not bool(shared.start_request.value):
                    _sleep_tick(period_s, tick_start)
                    continue
                shared.start_request.value = False
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
                execute_published_endpoints = 0
                execute_acknowledged_action_id = None
                h4_pending_acknowledgement = None
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
                        "cannot establish H4 coupled-write baseline: "
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

            # RUNNING: an operator or bounded-run time-limit stop ends the run
            # cleanly before the Main process shuts workers down.
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
                _end_policy_run(shared, stop_reason, abort=False)
                h4_pending_acknowledgement = None
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

            if h4_pending_acknowledgement is not None:
                acknowledgement = poll_coupled_command_acknowledgement(
                    shared,
                    h4_pending_acknowledgement.candidate,
                    ticket=h4_pending_acknowledgement.ticket,
                    arm_feedback_max_age_s=config.arm_feedback_max_age_s,
                    hand_feedback_max_age_s=config.hand_feedback_max_age_s,
                )
                if acknowledgement.status is CommandPublishStatus.APPLIED:
                    execute_acknowledged_action_id = int(
                        h4_pending_acknowledgement.candidate.action_id
                    )
                    metrics.increment(EXECUTE_ACKNOWLEDGED)
                    metrics.increment(EXECUTE_PUBLICATION_BOUND_REACHED)
                    h4_pending_acknowledgement = None
                    _end_policy_run(
                        shared,
                        "H4 publication bound reached",
                        abort=False,
                    )
                    emit_execute_receipt("H4 publication bound reached")
                    action_buffer.reset(run_generation=int(shared.run_generation.value))
                    buffer_generation = None
                    last_seen_plan_key = None
                    _sleep_tick(period_s, tick_start)
                    continue
                if acknowledgement.status is CommandPublishStatus.ACK_PENDING:
                    if now_ns < h4_pending_acknowledgement.deadline_monotonic_ns:
                        _sleep_tick(period_s, tick_start)
                        continue
                    fault_execute(
                        "H4 arm/hand acknowledgement timeout",
                        metric=EXECUTE_ACK_TIMEOUT,
                    )
                    continue
                fault_execute(
                    "H4 arm/hand acknowledgement failed: "
                    f"{acknowledgement.status.value}",
                    metric=ENDPOINTS_FATAL_REJECTED,
                )
                continue

            if config.execution_mode == "execute":
                h4_execute_bounds = config.h4_execute_bounds
                assert h4_execute_bounds is not None
                if (
                    execute_published_endpoints
                    >= h4_execute_bounds.max_published_endpoints
                ):
                    # A successful H4 publication must always leave a pending
                    # acknowledgement. Reaching this branch means that invariant
                    # was lost; do not permit another action to be built or sent.
                    fault_execute(
                        "H4 publication bound reached without acknowledgement state",
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
                    canonicalize_policy_hand_roundoff=True,
                    execution_mode=config.execution_mode,
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
                    config.execution_mode == "execute"
                    and publish_result.status is not CommandPublishStatus.PUBLISHED
                ):
                    abort_and_reset(
                        "H4 execution did not return a publication ticket",
                        metric=ENDPOINTS_FATAL_REJECTED,
                    )
                    continue
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
                            "H4 publication omitted its coupled command ticket",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    execute_published_endpoints += 1
                    assert config.h4_execute_bounds is not None
                    if (
                        execute_published_endpoints
                        > config.h4_execute_bounds.max_published_endpoints
                    ):
                        fault_execute(
                            "H4 publication count exceeded immutable bound",
                            metric=ENDPOINTS_FATAL_REJECTED,
                        )
                        continue
                    h4_pending_acknowledgement = _H4PendingAcknowledgement(
                        candidate=candidate,
                        ticket=ticket,
                        deadline_monotonic_ns=(
                            time.monotonic_ns()
                            + int(
                                config.h4_execute_bounds.acknowledgement_timeout_s * 1e9
                            )
                        ),
                    )
                    logger.info(
                        "coordinator: H4 published action_id=%d; awaiting worker acknowledgement",
                        candidate.action_id,
                    )
                metrics.increment(ENDPOINTS_COMMITTED)
                previous_arm_command_qpos = np.asarray(
                    candidate.arm_qpos, dtype=np.float64
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

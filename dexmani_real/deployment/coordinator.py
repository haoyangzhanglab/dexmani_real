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

from dexmani_real.control.publication import (
    CommandPublishStatus,
    build_action_candidate,
    validate_and_send_candidate,
)
from dexmani_real.control.safety_gate import planner_action_safety_gate
from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.deployment.metrics import (
    COMMAND_SILENCE_ABORT,
    ENDPOINTS_COALESCED,
    ENDPOINTS_DUE,
    ENDPOINTS_PUBLISHED,
    HAND_PREFLIGHT_REJECTIONS,
    POLICY_ABORTS,
    SAFETY_REJECTIONS,
    Metrics,
    flush_every,
    reject_counter_name,
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
from dexmani_real.planning.poses import rot6d_to_quat_wxyz
from dexmani_real.robot_spec import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.runtime.safety import SafetyState, advance_run_generation, transition
from dexmani_real.utils.log import get_logger

if TYPE_CHECKING:
    from dexmani_real.config.runtime import ResolvedRuntimeConfig

logger = get_logger(__name__)


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
    # Full 19-DoF collision model (hand + static boxes) for EE->IK and the
    # transition collision gate (Phase 6/7); table clearance is not part of the
    # policy safety gate.
    static_boxes: tuple = ()
    ik_max_pose_error_pos_m: float = 0.008
    ik_max_pose_error_rot_rad: float = 0.08
    # Per-tick delta limits for the learned-policy safety gate (reject, never
    # clip).  ``None`` disables the arm delta check.
    arm_max_delta_rad_per_tick: float | None = np.deg2rad(8.0)
    hand_max_delta_rad_per_tick: float = 0.1

    @classmethod
    def from_runtime(
        cls,
        deployment: DeploymentConfig,
        runtime: "ResolvedRuntimeConfig",
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
            static_boxes=tuple(runtime.environment.static_boxes),
            ik_max_pose_error_pos_m=float(runtime.policy.ik_max_pose_error_pos_m),
            ik_max_pose_error_rot_rad=float(runtime.policy.ik_max_pose_error_rot_rad),
            arm_max_delta_rad_per_tick=runtime.policy.arm_max_delta_rad_per_tick,
            hand_max_delta_rad_per_tick=float(runtime.hand.hand_max_delta_rad_per_tick),
        )


def _read_latest_plan(shared: RuntimeChannels):
    """Return the latest plan record (scalar structured array) or None."""
    result = shared.policy_plan_ring.read_latest()
    if result is None:
        return None
    return result[0][0]


def _adoptable(
    rec,
    *,
    current_generation: int,
    last_observation_id: int,
    now_ns: int,
    max_plan_age_ns: int,
    max_observation_age_ns: int,
) -> tuple[bool, str]:
    """Adoption gate for a fresh plan record. Any failure drops it."""
    if int(rec["run_generation"]) != current_generation:
        return False, "generation mismatch"
    if int(rec["observation_id"]) < last_observation_id:
        return False, "stale observation"
    finished_ns = int(rec["inference_finished_monotonic_ns"])
    anchor_ns = int(rec["observation_anchor_monotonic_ns"])
    if finished_ns <= 0 or now_ns - finished_ns > max_plan_age_ns:
        return False, "plan expired"
    if anchor_ns <= 0 or now_ns - anchor_ns > max_observation_age_ns:
        return False, "observation expired"
    n = int(rec["num_steps"])
    if n <= 0 or n > MAX_POLICY_CHUNK_STEPS:
        return False, "bad num_steps"
    mask = rec["valid_mask"][:n]
    if not np.all((mask == 0) | (mask == 1)):
        return False, "bad valid_mask"
    return True, ""


def _ready_to_replan(active_plan, next_step: int, stride: int) -> bool:
    """Whether a new plan may be admitted (adopt/promote) this tick.

    True when there is no active plan, when the stride of steps has been served,
    or when the active plan is fully consumed (``next_step`` reached its step
    count) — the last case keeps a plan shorter than the stride from stalling
    the scheduler.
    """
    if active_plan is None:
        return True
    return next_step >= stride or next_step >= int(active_plan["num_steps"])


def _select_due_step(
    target_ns: np.ndarray,
    valid_mask: np.ndarray,
    n: int,
    next_step: int,
    now_ns: int,
) -> tuple[int | None, int]:
    """Select the latest due step, coalescing overdue intermediate targets.

    Walks the (strictly increasing) target timeline from ``next_step``, skipping
    invalid steps. Returns ``(selected_index, new_next_step)``; ``selected_index``
    is ``None`` when no step is due yet (the coordinator then publishes nothing).
    ``new_next_step`` always advances past invalid/consumed steps.
    """
    latest_due: int | None = None
    i = next_step
    while i < n:
        if not bool(valid_mask[i]):
            i += 1
            continue
        if int(target_ns[i]) <= now_ns:
            latest_due = i
            i += 1
        else:
            break
    if latest_due is None:
        return None, i
    return latest_due, latest_due + 1


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
    advance_run_generation(shared)
    if not transition(shared, SafetyState.ARMED):
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
    max_observation_age_ns = int(config.deployment.max_observation_age_s * 1e9)
    max_silence_ns = int(config.deployment.max_command_silence_s * 1e9)
    first_command_timeout_ns = int(config.deployment.first_command_timeout_s * 1e9)
    replan_stride_steps = int(config.deployment.replan_stride_steps)

    active_plan = None
    active_plan_id = 0
    pending_plan = None
    pending_plan_id = 0
    last_adopted_observation_id = 0
    next_step = 0
    # Silence timeout starts at the first published command, not first inference.
    last_valid_policy_command_ns: int | None = None
    # RUNNING start time, for the first-command timeout.
    run_started_ns: int | None = None
    last_metrics_flush_ns = time.monotonic_ns()

    try:
        while shared.is_running.value:
            tick_start = time.monotonic()
            now_ns = time.monotonic_ns()
            shared.set_heartbeat("policy", time.monotonic())

            if bool(shared.error_state.value) or bool(shared.estop_request.value):
                _sleep_tick(period_s, tick_start)
                continue

            # ARMED idle: wait for the operator to request a new run (B).
            if int(shared.safety_state.value) != int(SafetyState.RUNNING):
                if not bool(shared.start_request.value):
                    _sleep_tick(period_s, tick_start)
                    continue
                shared.start_request.value = False
                # A stray S from ARMED must not stop the freshly started run.
                shared.stop_request.value = False
                if not transition(shared, SafetyState.RUNNING):
                    logger.error(
                        "coordinator: cannot enter RUNNING (safety_state=%d)",
                        int(shared.safety_state.value),
                    )
                    return
                advance_run_generation(shared)
                logger.info(
                    "coordinator_loop: RUNNING (run_generation=%d)",
                    int(shared.run_generation.value),
                )
                # Reset per-run episode state for the new observation epoch.
                active_plan = None
                active_plan_id = 0
                pending_plan = None
                pending_plan_id = 0
                last_adopted_observation_id = 0
                next_step = 0
                last_valid_policy_command_ns = None
                run_started_ns = time.monotonic_ns()
                _sleep_tick(period_s, tick_start)
                continue

            # RUNNING: operator STOP (S) ends the run cleanly.
            if bool(shared.stop_request.value):
                shared.stop_request.value = False
                # A stray B from RUNNING must not auto-restart after this stop.
                shared.start_request.value = False
                _end_policy_run(shared, "operator stop", abort=False)
                _sleep_tick(period_s, tick_start)
                continue

            # Abort a run that never produced its first command (the model
            # dropped every plan); command-to-command silence is checked below.
            if (
                last_valid_policy_command_ns is None
                and run_started_ns is not None
                and now_ns - run_started_ns > first_command_timeout_ns
            ):
                _end_policy_run(
                    shared,
                    "first command timeout",
                    abort=True,
                    metrics=metrics,
                    metric=COMMAND_SILENCE_ABORT,
                )
                continue

            # Watch command-to-command silence; first-inference latency is exempt.
            if (
                last_valid_policy_command_ns is not None
                and now_ns - last_valid_policy_command_ns > max_silence_ns
            ):
                _end_policy_run(
                    shared,
                    "command silence timeout",
                    abort=True,
                    metrics=metrics,
                    metric=COMMAND_SILENCE_ABORT,
                )
                continue

            rec = _read_latest_plan(shared)
            if rec is not None:
                rec_id = int(rec["plan_id"])
                if rec_id != active_plan_id and rec_id != pending_plan_id:
                    ok, reason = _adoptable(
                        rec,
                        current_generation=int(shared.run_generation.value),
                        last_observation_id=last_adopted_observation_id,
                        now_ns=now_ns,
                        max_plan_age_ns=max_plan_age_ns,
                        max_observation_age_ns=max_observation_age_ns,
                    )
                    if ok:
                        if _ready_to_replan(
                            active_plan, next_step, replan_stride_steps
                        ):
                            # No active plan, the stride is served, or the plan is
                            # consumed: adopt now (supersede any held plan).
                            active_plan = rec
                            active_plan_id = rec_id
                            last_adopted_observation_id = int(rec["observation_id"])
                            next_step = 0
                            pending_plan = None
                            pending_plan_id = 0
                        else:
                            # Hold the newest plan (latest wins) and promote it
                            # once ready, so a mid-plan replan never opens a
                            # command gap (plan §8).
                            pending_plan = rec
                            pending_plan_id = rec_id
                    else:
                        logger.debug("coordinator: plan %d dropped: %s", rec_id, reason)

            # Promote a held plan once the active plan is done enough.
            if pending_plan is not None and _ready_to_replan(
                active_plan, next_step, replan_stride_steps
            ):
                ok, reason = _adoptable(
                    pending_plan,
                    current_generation=int(shared.run_generation.value),
                    last_observation_id=last_adopted_observation_id,
                    now_ns=now_ns,
                    max_plan_age_ns=max_plan_age_ns,
                    max_observation_age_ns=max_observation_age_ns,
                )
                if ok:
                    active_plan = pending_plan
                    active_plan_id = pending_plan_id
                    last_adopted_observation_id = int(pending_plan["observation_id"])
                    next_step = 0
                pending_plan = None
                pending_plan_id = 0

            if active_plan is None:
                _sleep_tick(period_s, tick_start)
                continue

            n = int(active_plan["num_steps"])
            prev_next_step = next_step
            selected, next_step = _select_due_step(
                np.asarray(active_plan["target_monotonic_ns"][:n], dtype=np.uint64),
                np.asarray(active_plan["valid_mask"][:n], dtype=np.uint8),
                n,
                next_step,
                now_ns,
            )
            if selected is None:
                _sleep_tick(period_s, tick_start)
                continue
            metrics.increment(ENDPOINTS_DUE)
            coalesced = int(selected) - int(prev_next_step)
            if coalesced > 0:
                metrics.increment(ENDPOINTS_COALESCED, coalesced)

            hand_qpos: np.ndarray | None = None
            if int(active_plan["hand_present"]) == 1:
                hand_qpos = np.asarray(
                    active_plan["hand_qpos"][selected], dtype=np.float64
                )

            _arm_state = read_arm_state_dict(shared)
            if int(active_plan["ee_present"]) == 1:
                # EE -> joint via collision-aware IK (plan §14.2 decision 3).
                # hand_qpos is loaded into the collision model first so the solve
                # sees the full 19-DoF geometry.
                if _arm_state is None or not np.all(np.isfinite(_arm_state["qpos"])):
                    # No causal arm feedback yet: drop (silence watchdog backstop).
                    _sleep_tick(period_s, tick_start)
                    continue
                ee_pos = np.asarray(active_plan["ee_pos"][selected], dtype=np.float64)
                ee_rot6d = np.asarray(
                    active_plan["ee_rot6d"][selected], dtype=np.float64
                )
                if hand_qpos is not None:
                    planner.set_hand_qpos(hand_qpos)
                ik_result = planner.solve_teleop_ik(
                    Pose(p=ee_pos, q=rot6d_to_quat_wxyz(ee_rot6d)),
                    _arm_state["qpos"],
                    _arm_state["qpos"],
                )
                if not ik_result.success or ik_result.qpos is None:
                    _end_policy_run(
                        shared,
                        f"EE IK failure: {ik_result.reason}",
                        abort=True,
                        metrics=metrics,
                        metric=SAFETY_REJECTIONS,
                    )
                    continue
                arm_qpos = np.asarray(ik_result.qpos, dtype=np.float64)
            else:
                arm_qpos = np.asarray(
                    active_plan["arm_qpos"][selected], dtype=np.float64
                )
                # Canonicalize targets against fresh feedback before publication.
                if _arm_state is not None and np.all(np.isfinite(_arm_state["qpos"])):
                    arm_qpos = wrap_nearest_equivalent(
                        arm_qpos,
                        _arm_state["qpos"],
                        config.arm_joint_lower_rad,
                        config.arm_joint_upper_rad,
                    )

            candidate = build_action_candidate(
                shared,
                arm_qpos,
                hand_qpos,
                is_hold=False,
                observation_id=int(active_plan["observation_id"]),
                observation_anchor_monotonic_ns=int(
                    active_plan["observation_anchor_monotonic_ns"]
                ),
                action_validity_s=float(config.deployment.action_validity_s),
            )
            if candidate is None:
                _sleep_tick(period_s, tick_start)
                continue

            publish_result = validate_and_send_candidate(
                shared,
                candidate,
                gate=gate,
                arm_feedback_max_age_s=config.arm_feedback_max_age_s,
                hand_feedback_max_age_s=config.hand_feedback_max_age_s,
                hand_mechanical_lower_rad=np.asarray(
                    config.hand_mechanical_lower_rad, dtype=np.float64
                ),
                hand_mechanical_upper_rad=np.asarray(
                    config.hand_mechanical_upper_rad, dtype=np.float64
                ),
            )
            if not publish_result.succeeded:
                if publish_result.status == CommandPublishStatus.GATE_REJECTED:
                    # A gate rejection is an invalid model endpoint; abort the run.
                    metrics.increment(
                        reject_counter_name(
                            publish_result.gate_code.value
                            if publish_result.gate_code is not None
                            else None
                        )
                    )
                    _end_policy_run(
                        shared,
                        f"safety gate rejection: {publish_result.reason}",
                        abort=True,
                        metrics=metrics,
                        metric=SAFETY_REJECTIONS,
                    )
                    continue
                if (
                    publish_result.status
                    == CommandPublishStatus.HAND_PREFLIGHT_REJECTED
                ):
                    metrics.increment(HAND_PREFLIGHT_REJECTIONS)
                    _end_policy_run(
                        shared,
                        f"hand command preflight rejection: {publish_result.reason}",
                        abort=True,
                        metrics=metrics,
                    )
                    continue
                # Drop transient feedback failures; the silence watchdog is the backstop.
                _sleep_tick(period_s, tick_start)
                continue

            metrics.increment(ENDPOINTS_PUBLISHED)
            last_valid_policy_command_ns = now_ns

            last_metrics_flush_ns = flush_every(
                metrics, last_ns=last_metrics_flush_ns, prefix="coordinator metrics"
            )
            _sleep_tick(period_s, tick_start)
    finally:
        logger.info("coordinator_loop: exited")


def _sleep_tick(period_s: float, tick_start: float) -> None:
    """Sleep for the remainder of one control tick, if any."""
    sleep_s = period_s - (time.monotonic() - tick_start)
    if sleep_s > 0:
        time.sleep(sleep_s)

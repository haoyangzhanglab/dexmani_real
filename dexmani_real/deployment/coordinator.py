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

import numpy as np

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
from dexmani_real.planning import XArm7MotionPlanner, XArm7PlannerConfig
from dexmani_real.planning.constants import (
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_XHAND_SRDF_PATH,
)
from dexmani_real.planning.path_utils import wrap_nearest_equivalent
from dexmani_real.policy.safety import (
    CommandPublishStatus,
    build_action_candidate,
    planner_action_safety_gate,
    validate_and_send_candidate,
)
from dexmani_real.robot.safety import SafetyState, advance_run_generation, transition
from dexmani_real.shm.shared_storage import SharedStorage, read_arm_state_dict
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import MAX_POLICY_CHUNK_STEPS

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
    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    hand_joint_lower_rad: tuple[float, ...]
    hand_joint_upper_rad: tuple[float, ...]
    hand_mechanical_lower_rad: tuple[float, ...]
    hand_mechanical_upper_rad: tuple[float, ...]
    hand_feedback_max_age_s: float
    control_hz: float

    @classmethod
    def from_runtime(cls, deployment: DeploymentConfig, runtime: object) -> "CoordinatorConfig":
        return cls(
            deployment=deployment,
            arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
            workspace_bounds=runtime.policy.workspace.as_tuple(),
            hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
            hand_mechanical_lower_rad=tuple(runtime.hand.mechanical_qpos_min_rad),
            hand_mechanical_upper_rad=tuple(runtime.hand.mechanical_qpos_max_rad),
            hand_feedback_max_age_s=float(runtime.safety.heartbeat_timeouts["hand"]),
            control_hz=float(runtime.policy.control_hz),
        )


def _read_latest_plan(shared: SharedStorage):
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
    shared: SharedStorage,
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


def coordinator_loop(shared: SharedStorage, config: CoordinatorConfig) -> None:
    """Coordinator process entry point — the only robot-action producer.

    Idles in ARMED until the operator presses B (``start_request``), runs one
    policy episode in RUNNING, then returns to ARMED on S (``stop_request``) or
    a policy-semantic abort.  Each B advances the run generation, so any
    in-flight plan or command from a previous run is invalid at the worker.
    """
    if config is None:
        raise ValueError("coordinator_loop requires a CoordinatorConfig")

    # Deployment uses the same arm-base workspace gate as VR teleoperation.
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(XARM7_XHAND_COLLISION_URDF_PATH),
            srdf_path=str(XARM7_XHAND_SRDF_PATH),
            workspace_bounds=np.asarray(config.workspace_bounds, dtype=np.float64),
        ),
    )
    gate = planner_action_safety_gate(
        planner=planner,
        arm_joint_lower_rad=config.arm_joint_lower_rad,
        arm_joint_upper_rad=config.arm_joint_upper_rad,
        hand_joint_lower_rad=config.hand_joint_lower_rad,
        hand_joint_upper_rad=config.hand_joint_upper_rad,
    )
    metrics = Metrics()

    shared.set_heartbeat("policy", time.monotonic())
    shared.set_ready("policy")

    # Wait until Main has armed the system (model loaded, hardware ready).
    while (
        shared.is_running.value
        and int(shared.safety_state.value) != int(SafetyState.ARMED)
    ):
        # Keep the heartbeat fresh while waiting for arm/inference readiness.
        shared.set_heartbeat("policy", time.monotonic())
        if bool(shared.error_state.value) or bool(shared.estop_request.value):
            return
        time.sleep(0.01)
    if not shared.is_running.value or int(shared.safety_state.value) != int(SafetyState.ARMED):
        return

    period_s = 1.0 / float(config.control_hz)
    max_plan_age_ns = int(config.deployment.max_plan_age_s * 1e9)
    max_observation_age_ns = int(config.deployment.max_observation_age_s * 1e9)
    max_silence_ns = int(config.deployment.max_command_silence_s * 1e9)

    active_plan = None
    active_plan_id = 0
    last_adopted_observation_id = 0
    next_step = 0
    # Silence timeout starts at the first published command, not first inference.
    last_valid_policy_command_ns: int | None = None
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
                last_adopted_observation_id = 0
                next_step = 0
                last_valid_policy_command_ns = None
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
            if rec is not None and int(rec["plan_id"]) != active_plan_id:
                ok, reason = _adoptable(
                    rec,
                    current_generation=int(shared.run_generation.value),
                    last_observation_id=last_adopted_observation_id,
                    now_ns=now_ns,
                    max_plan_age_ns=max_plan_age_ns,
                    max_observation_age_ns=max_observation_age_ns,
                )
                if ok:
                    active_plan = rec
                    active_plan_id = int(rec["plan_id"])
                    last_adopted_observation_id = int(rec["observation_id"])
                    next_step = 0
                else:
                    logger.debug("coordinator: plan %d dropped: %s", int(rec["plan_id"]), reason)

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

            arm_qpos = np.asarray(active_plan["arm_qpos"][selected], dtype=np.float64)
            # Canonicalize targets against fresh feedback before publication.
            _arm_state = read_arm_state_dict(shared)
            if _arm_state is not None and np.all(np.isfinite(_arm_state["qpos"])):
                arm_qpos = wrap_nearest_equivalent(
                    arm_qpos,
                    _arm_state["qpos"],
                    config.arm_joint_lower_rad,
                    config.arm_joint_upper_rad,
                )
            hand_qpos: np.ndarray | None = None
            if int(active_plan["hand_present"]) == 1:
                hand_qpos = np.asarray(active_plan["hand_qpos"][selected], dtype=np.float64)

            candidate = build_action_candidate(
                shared,
                arm_qpos,
                hand_qpos,
                is_hold=False,
                observation_id=int(active_plan["observation_id"]),
                observation_anchor_monotonic_ns=int(active_plan["observation_anchor_monotonic_ns"]),
                action_validity_s=float(config.deployment.action_validity_s),
            )
            if candidate is None:
                _sleep_tick(period_s, tick_start)
                continue

            publish_result = validate_and_send_candidate(
                shared,
                candidate,
                gate=gate,
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
                if publish_result.status == CommandPublishStatus.HAND_PREFLIGHT_REJECTED:
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

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

from dexmani_real import ASSET_DIR
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
from dexmani_real.policy.safety import (
    CommandPublishStatus,
    advance_run_generation,
    build_action_candidate,
    planner_action_safety_gate,
    validate_and_send_candidate,
)
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import SharedStorage
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


def _abort_policy_run(
    shared: SharedStorage,
    reason: str,
    metrics: Metrics | None = None,
    *,
    metric: str | None = None,
) -> None:
    """Advance the generation and drop RUNNING -> ARMED.

    This is a policy-semantic failure, not a hardware fault: the robot is left
    ARMED (command quiescence) rather than FAULT. The abort counters are flushed
    immediately because the success-path ``flush_every`` is never reached once a
    run aborts (the loop idles in ARMED), and the H0 gate reads these counters.
    """
    advance_run_generation(shared)
    if not transition(shared, SafetyState.ARMED):
        logger.error("coordinator: abort failed to transition RUNNING->ARMED")
    logger.warning("coordinator: policy run aborted: %s", reason)
    if metrics is not None:
        metrics.increment(POLICY_ABORTS)
        if metric is not None:
            metrics.increment(metric)
        metrics.flush(prefix="coordinator metrics")


def coordinator_loop(shared: SharedStorage, config: CoordinatorConfig) -> None:
    """Coordinator process entry point — the only robot-action producer."""
    if config is None:
        raise ValueError("coordinator_loop requires a CoordinatorConfig")

    # The deployment path is machine-driven (no operator in the loop), so it must
    # run the same arm-base Cartesian workspace check as VR teleop.  Build a
    # planner wired to the configured workspace bounds and extend the shared
    # SafetyGate boundary with it.  Build before the first heartbeat/ready
    # publish so a planner failure fails closed at readiness rather than mid-run.
    planner = XArm7MotionPlanner(
        XArm7PlannerConfig(
            urdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand_collision.urdf"),
            srdf_path=str(ASSET_DIR / "robots" / "xhand" / "xarm7_xhand.srdf"),
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
    # Publish readiness while still DISARMED/ARMED so Main's
    # wait_subsystem_ready observes a ready worker with a live heartbeat.
    shared.set_ready("policy")

    # RUNNING entry — the coordinator is the policy control source, so there is
    # no operator BEGIN. It waits for Main to arm the system (DISARMED -> ARMED),
    # then self-enters RUNNING and advances the generation once. The advance
    # invalidates any startup-generation plan and makes the inference worker
    # reset its backend before the first proposal.
    while (
        shared.is_running.value
        and int(shared.safety_state.value) != int(SafetyState.ARMED)
    ):
        # Keep the heartbeat fresh while blocked: arm/inference readiness can
        # exceed the 1.0s policy timeout, and run_supervisor starts checking
        # heartbeats immediately after Main arms (mirrors the hand_process wait).
        shared.set_heartbeat("policy", time.monotonic())
        if bool(shared.error_state.value) or bool(shared.estop_request.value):
            return
        time.sleep(0.01)
    if not shared.is_running.value or int(shared.safety_state.value) != int(SafetyState.ARMED):
        return
    if not transition(shared, SafetyState.RUNNING):
        logger.error(
            "coordinator: cannot enter RUNNING (safety_state=%d)",
            int(shared.safety_state.value),
        )
        return
    advance_run_generation(shared)
    run_generation = int(shared.run_generation.value)
    logger.info("coordinator_loop: RUNNING (run_generation=%d)", run_generation)

    period_s = 1.0 / float(config.control_hz)
    max_plan_age_ns = int(config.deployment.max_plan_age_s * 1e9)
    max_observation_age_ns = int(config.deployment.max_observation_age_s * 1e9)
    max_silence_ns = int(config.deployment.max_command_silence_s * 1e9)

    active_plan = None
    active_plan_id = 0
    last_adopted_observation_id = 0
    next_step = 0
    # Command-to-command silence reference. ``None`` until the first publish so
    # the slow first inference (forced reset + encode + infer after the
    # RUNNING-entry generation advance) is not charged against the silence
    # budget.
    last_valid_policy_command_ns: int | None = None
    last_metrics_flush_ns = time.monotonic_ns()
    running = True

    try:
        while shared.is_running.value:
            tick_start = time.monotonic()
            now_ns = time.monotonic_ns()
            shared.set_heartbeat("policy", time.monotonic())

            if not running:
                # Post-abort: idle in ARMED, heartbeat only, await explicit restart.
                _sleep_tick(period_s, tick_start)
                continue
            if bool(shared.error_state.value) or bool(shared.estop_request.value):
                _sleep_tick(period_s, tick_start)
                continue

            # Command silence watchdog: command-to-command only, armed at
            # the first publish, so first-command inference latency is not
            # charged against the budget.
            if (
                last_valid_policy_command_ns is not None
                and now_ns - last_valid_policy_command_ns > max_silence_ns
            ):
                _abort_policy_run(
                    shared, "command silence timeout", metrics, metric=COMMAND_SILENCE_ABORT
                )
                running = False
                active_plan = None
                continue

            # Adopt the latest plan (latest-wins; a higher plan_id supersedes).
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
                    # SafetyGate rejection is a policy-semantic failure:
                    # the model proposed an invalid endpoint. Abort immediately.
                    # Attribute the rejection per gate code so the flush
                    # log shows *which* operation rejected, not just a total.
                    metrics.increment(
                        reject_counter_name(
                            publish_result.gate_code.value
                            if publish_result.gate_code is not None
                            else None
                        )
                    )
                    _abort_policy_run(
                        shared,
                        f"safety gate rejection: {publish_result.reason}",
                        metrics,
                        metric=SAFETY_REJECTIONS,
                    )
                    running = False
                    active_plan = None
                    continue
                if publish_result.status == CommandPublishStatus.HAND_PREFLIGHT_REJECTED:
                    metrics.increment(HAND_PREFLIGHT_REJECTIONS)
                    _abort_policy_run(
                        shared,
                        f"hand command preflight rejection: {publish_result.reason}",
                        metrics,
                    )
                    running = False
                    active_plan = None
                    continue
                # Feedback/transport failure is transient: drop this tick; the
                # silence watchdog is the eventual abort backstop.
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

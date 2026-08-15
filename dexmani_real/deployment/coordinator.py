"""Deployment coordinator — the sole learned-policy robot-action producer.

The inference worker writes proposals to ``policy_plan_ring``; this coordinator
is the only process that turns a proposal into a robot command. It selects the
plan, schedules the due endpoint (one per control tick), runs the shared
candidate publication boundary (SafetyGate -> send_command), and owns the
policy semantic watchdog and the ``RUNNING <-> ARMED`` control-source state.

It never dumps a whole chunk into the arm queue or hand ring (§73/§74) and never
interpolates between model steps (§78).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.policy.safety import (
    SafetyGate,
    advance_run_generation,
    build_action_candidate,
    validate_and_send_candidate,
    validate_hand_command_delta,
)
from dexmani_real.robot.safety import SafetyState, transition
from dexmani_real.shm.shared_storage import SharedStorage
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import HAND_JOINT_SHAPE, MAX_POLICY_CHUNK_STEPS

logger = get_logger(__name__)


@dataclass(frozen=True)
class CoordinatorConfig:
    """Deployment config plus the safety/limits the coordinator needs to gate.

    Mirrors ``TeleopConfig.from_runtime``: the deployment namespace supplies the
    model/boundary knobs, the runtime namespace supplies the joint limits, hand
    mechanical envelope, delta bound, and control rate.
    """

    deployment: DeploymentConfig
    arm_joint_lower_rad: tuple[float, ...]
    arm_joint_upper_rad: tuple[float, ...]
    hand_joint_lower_rad: tuple[float, ...]
    hand_joint_upper_rad: tuple[float, ...]
    hand_mechanical_lower_rad: tuple[float, ...]
    hand_mechanical_upper_rad: tuple[float, ...]
    hand_max_delta_rad: float | None
    control_hz: float

    @classmethod
    def from_runtime(cls, deployment: DeploymentConfig, runtime: object) -> "CoordinatorConfig":
        return cls(
            deployment=deployment,
            arm_joint_lower_rad=tuple(runtime.arm.joint_limit_lower),
            arm_joint_upper_rad=tuple(runtime.arm.joint_limit_upper),
            hand_joint_lower_rad=tuple(runtime.hand.qpos_min_rad),
            hand_joint_upper_rad=tuple(runtime.hand.qpos_max_rad),
            hand_mechanical_lower_rad=tuple(runtime.hand.mechanical_qpos_min_rad),
            hand_mechanical_upper_rad=tuple(runtime.hand.mechanical_qpos_max_rad),
            hand_max_delta_rad=runtime.hand.max_delta_rad,
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
    """Adoption gate for a fresh plan record (§75). Any failure drops it."""
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
    """Select the latest due step, coalescing overdue intermediate targets (§76).

    Walks the (strictly increasing) target timeline from ``next_step``, skipping
    invalid steps. Returns ``(selected_index, new_next_step)``; ``selected_index``
    is ``None`` when no step is due yet (the coordinator then publishes nothing,
    §77). ``new_next_step`` always advances past invalid/consumed steps.
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


def _preflight_hand(hand_qpos: np.ndarray, previous: np.ndarray | None, config: CoordinatorConfig) -> str | None:
    """Run the coupled-hand mechanical/delta preflight; return an error or None.

    The delta reference is the last *published* hand command (mirroring VR
    teleop), so contact/torque-limit lag never stalls the operator.
    """
    # On the first coupled command there is no prior published command, so the
    # command-to-command delta is skipped (bounds are still enforced). The seed
    # reference is the measured hand pose, mirrored from VR teleop, so a valid
    # seed means ``previous`` is non-None and the delta is enforced.
    max_delta = config.hand_max_delta_rad if previous is not None else None
    try:
        validate_hand_command_delta(
            hand_qpos,
            previous,
            np.asarray(config.hand_joint_lower_rad, dtype=np.float64),
            np.asarray(config.hand_joint_upper_rad, dtype=np.float64),
            np.asarray(config.hand_mechanical_lower_rad, dtype=np.float64),
            np.asarray(config.hand_mechanical_upper_rad, dtype=np.float64),
            max_delta,
        )
    except ValueError as exc:
        return str(exc)
    return None


def _seed_hand_reference(shared: SharedStorage) -> np.ndarray | None:
    """Seed the first hand-delta reference from measured hand feedback.

    Mirrors VR teleop (``ctx.prev_hand_qpos`` is seeded from feedback): the first
    coupled hand command is delta-bounded against the current measured pose, so a
    fresh run never aborts on a delta check that has no prior command. Returns
    ``None`` when feedback is unavailable or unhealthy.
    """
    result = shared.hand_state_ring.read_latest()
    if result is None:
        return None
    record = result[0][0]
    if not bool(record["connected"]) or not bool(record["state_valid"]):
        return None
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    if qpos.shape != HAND_JOINT_SHAPE or not np.all(np.isfinite(qpos)):
        return None
    return qpos.copy()


def _abort_policy_run(shared: SharedStorage, reason: str) -> None:
    """Advance the generation and drop RUNNING -> ARMED (§80.2/§82).

    This is a policy-semantic failure, not a hardware fault: the robot is left
    ARMED (command quiescence) rather than FAULT.
    """
    advance_run_generation(shared)
    if not transition(shared, SafetyState.ARMED):
        logger.error("coordinator: abort failed to transition RUNNING->ARMED")
    logger.warning("coordinator: policy run aborted: %s", reason)


def coordinator_loop(shared: SharedStorage, config: CoordinatorConfig) -> None:
    """Coordinator process entry point — the only robot-action producer."""
    if config is None:
        raise ValueError("coordinator_loop requires a CoordinatorConfig")

    gate = SafetyGate(
        arm_joint_lower_rad=config.arm_joint_lower_rad,
        arm_joint_upper_rad=config.arm_joint_upper_rad,
        hand_joint_lower_rad=config.hand_joint_lower_rad,
        hand_joint_upper_rad=config.hand_joint_upper_rad,
    )

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
    last_published_hand_cmd: np.ndarray | None = (
        _seed_hand_reference(shared) if config.deployment.hand_enabled else None
    )
    last_valid_policy_command_ns = time.monotonic_ns()
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

            # Command silence watchdog (§82).
            if now_ns - last_valid_policy_command_ns > max_silence_ns:
                _abort_policy_run(shared, "command silence timeout")
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

            arm_qpos = np.asarray(active_plan["arm_qpos"][selected], dtype=np.float64)
            hand_qpos: np.ndarray | None = None
            if int(active_plan["hand_present"]) == 1:
                hand_qpos = np.asarray(active_plan["hand_qpos"][selected], dtype=np.float64)

            # Coupled-hand preflight before the arm endpoint is enqueued, so a
            # rejected hand command desyncs nothing (§74). Violation aborts.
            if hand_qpos is not None:
                error = _preflight_hand(hand_qpos, last_published_hand_cmd, config)
                if error is not None:
                    _abort_policy_run(shared, f"hand command delta violation: {error}")
                    running = False
                    active_plan = None
                    continue

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

            reject_reason: list[str] = []
            published = validate_and_send_candidate(
                shared,
                candidate,
                gate=gate,
                dt_s=period_s,
                reject_reason_out=reject_reason,
            )
            if published is None:
                if reject_reason:
                    # SafetyGate rejection is a policy-semantic failure (§80.2):
                    # the model proposed an invalid endpoint. Abort immediately.
                    _abort_policy_run(shared, f"safety gate rejection: {reject_reason[0]}")
                    running = False
                    active_plan = None
                    continue
                # Feedback/transport failure is transient: drop this tick; the
                # silence watchdog is the eventual abort backstop.
                _sleep_tick(period_s, tick_start)
                continue

            if hand_qpos is not None:
                last_published_hand_cmd = hand_qpos.copy()
            last_valid_policy_command_ns = now_ns

            _sleep_tick(period_s, tick_start)
    finally:
        logger.info("coordinator_loop: exited")


def _sleep_tick(period_s: float, tick_start: float) -> None:
    """Sleep for the remainder of one control tick, if any."""
    sleep_s = period_s - (time.monotonic() - tick_start)
    if sleep_s > 0:
        time.sleep(sleep_s)

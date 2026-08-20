"""Inference worker: observations -> model proposals -> ``policy_plan_ring``.

The inference worker is the *only* process that touches the backend. It reads
causal observations from the shared rings, runs ``encode -> infer -> decode``,
and publishes the resulting :class:`~dexmani_real.deployment.contracts.JointActionChunk`
to the latest-wins ``policy_plan_ring``. It never writes ``arm_cmd_ring``,
``hand_cmd_ring``, the SDK, ``SafetyState``, or ``run_generation`` — model output
is a proposal, not a robot command.

``inference_loop`` is a plain ``*_loop(shared, config)`` function (not an
``mp.Process`` subclass); lifecycle/supervision stays in the A/B runtime.
"""

from __future__ import annotations

import time

import numpy as np

from dexmani_real.deployment.config import DeploymentConfig
from dexmani_real.deployment.contracts import InferenceContext, JointActionChunk
from dexmani_real.deployment.loader import (
    load_action_adapter,
    load_backend,
    load_observation_adapter,
)
from dexmani_real.deployment.metrics import (
    INFERENCE_FAILURES,
    INFERENCE_MS,
    OBSERVATIONS_BUILT,
    PLANS_CREATED,
    PLANS_GENERATION_DROPPED,
    Metrics,
    flush_every,
)
from dexmani_real.deployment.observation import (
    FrameWindow,
    ObservationBatch,
    parse_observation_fields,
)
from dexmani_real.shm.shared_storage import SharedStorage, new_frame
from dexmani_real.utils.log import get_logger
from dexmani_real.utils.schema import MAX_POLICY_CHUNK_STEPS, POLICY_PLAN_DTYPE

logger = get_logger(__name__)

# Poll delay while required causal feedback is unavailable.
_NO_FEEDBACK_POLL_S = 0.005


def publish_plan(
    shared: SharedStorage,
    *,
    plan_id: int,
    context: InferenceContext,
    chunk: JointActionChunk,
) -> bool:
    """Publish a validated chunk to the latest-wins plan ring.

    Re-reads ``shared.run_generation`` at publish time; if the generation has
    advanced since the observation was captured, the in-flight plan is dropped
    (returns ``False`` without writing) rather than relabeled. An
    over-capacity chunk raises ``ValueError`` (fail, never truncate).
    """
    if int(shared.run_generation.value) != int(context.run_generation):
        return False

    n = int(chunk.arm_qpos.shape[0])
    if n <= 0 or n > MAX_POLICY_CHUNK_STEPS:
        raise ValueError(f"policy chunk has {n} steps; transport capacity is {MAX_POLICY_CHUNK_STEPS}")

    frame = new_frame(POLICY_PLAN_DTYPE)
    frame["plan_id"][0] = np.uint64(plan_id)
    frame["run_generation"][0] = np.uint64(context.run_generation)
    frame["observation_id"][0] = np.uint64(context.observation_id)
    frame["observation_anchor_monotonic_ns"][0] = np.uint64(context.observation_anchor_monotonic_ns)
    frame["inference_started_monotonic_ns"][0] = np.uint64(context.inference_started_monotonic_ns)
    frame["inference_finished_monotonic_ns"][0] = np.uint64(context.inference_finished_monotonic_ns)
    frame["num_steps"][0] = np.uint32(n)
    frame["arm_present"][0] = 1
    frame["hand_present"][0] = 1 if chunk.hand_qpos is not None else 0
    frame["target_monotonic_ns"][0, :n] = chunk.target_monotonic_ns
    frame["arm_qpos"][0, :n] = chunk.arm_qpos
    if chunk.hand_qpos is not None:
        frame["hand_qpos"][0, :n] = chunk.hand_qpos
    frame["valid_mask"][0, :n] = chunk.valid_mask
    shared.policy_plan_ring.write(frame)
    return True


def _read_state_history(
    ring,
    *,
    horizon: int,
    anchor_ns: int,
    values_field: str,
    required_true_fields: tuple[str, ...] = (),
) -> FrameWindow | None:
    """Read the causal (source <= publish <= anchor) state frames, oldest-first."""
    try:
        history = ring.get_last_k(min(int(horizon), ring.maxlen))
    except Exception:
        logger.warning("inference: state history read failed", exc_info=True)
        return None

    values: list[np.ndarray] = []
    sequences: list[int] = []
    sources: list[int] = []
    publishes: list[int] = []
    for data, ring_publish_ns, sequence in history:
        names = data.dtype.names or ()
        if any(field not in names or not bool(data[field][0]) for field in required_true_fields):
            continue
        source_ns = int(data["source_monotonic_ns"][0])
        publish_ns = (
            int(data["publish_monotonic_ns"][0])
            if "publish_monotonic_ns" in names and int(data["publish_monotonic_ns"][0]) > 0
            else int(ring_publish_ns)
        )
        if not (0 < source_ns <= publish_ns <= anchor_ns):
            continue
        values.append(np.asarray(data[values_field][0], dtype=np.float64))
        sequences.append(int(sequence))
        sources.append(source_ns)
        publishes.append(publish_ns)

    if not values:
        return None
    t = len(values)
    return FrameWindow(
        values=np.stack(values),
        source_sequence=np.asarray(sequences, dtype=np.uint64),
        source_monotonic_ns=np.asarray(sources, dtype=np.uint64),
        publish_monotonic_ns=np.asarray(publishes, dtype=np.uint64),
        valid_mask=np.ones(t, dtype=np.uint8),
    )


def _build_observation(
    shared: SharedStorage,
    config: DeploymentConfig,
    *,
    observation_id: int,
    run_generation: int,
    anchor_ns: int,
) -> ObservationBatch:
    """Assemble requested causal modalities from the arm/hand rings.

    The hand state and tactile rings are read only when their corresponding
    ``observation_fields`` are requested. Every selected frame is additionally
    gated by its source/publish timestamps and modality-specific health flags.
    """
    horizon = int(config.observation_horizon)
    arm_history = _read_state_history(
        shared.arm_state_ring,
        horizon=horizon,
        anchor_ns=anchor_ns,
        values_field="qpos",
        required_true_fields=("state_valid",),
    )
    hand_history: FrameWindow | None = None
    hand_current_history: FrameWindow | None = None
    hand_tactile_sum_history: FrameWindow | None = None
    tactile_history: FrameWindow | None = None
    requested = set(
        parse_observation_fields(getattr(config, "observation_fields", "arm_qpos,hand_qpos"))
    )
    hand_state_requested = bool(
        requested
        & {
            "hand_qpos",
            "hand_joint_position",
            "hand_current",
            "hand_joint_torque",
            "hand_tactile_sum",
            "fingertip_force",
        }
    )
    tactile_requested = bool(requested & {"hand_tactile_force", "xhand_tactile"})
    if config.hand_enabled:
        if hand_state_requested:
            hand_history = _read_state_history(
                shared.hand_state_ring,
                horizon=horizon,
                anchor_ns=anchor_ns,
                values_field="qpos",
                required_true_fields=("state_valid", "read_healthy"),
            )
            if requested & {"hand_current", "hand_joint_torque"}:
                hand_current_history = _read_state_history(
                    shared.hand_state_ring,
                    horizon=horizon,
                    anchor_ns=anchor_ns,
                    values_field="current",
                    required_true_fields=("state_valid", "read_healthy"),
                )
            if requested & {"hand_tactile_sum", "fingertip_force"}:
                hand_tactile_sum_history = _read_state_history(
                    shared.hand_state_ring,
                    horizon=horizon,
                    anchor_ns=anchor_ns,
                    values_field="tactile_sum",
                    required_true_fields=("state_valid", "read_healthy", "tactile_sum_valid"),
                )
        if tactile_requested:
            tactile_history = _read_state_history(
                shared.hand_tactile_ring,
                horizon=horizon,
                anchor_ns=anchor_ns,
                values_field="tactile_force",
                required_true_fields=("fresh",),
            )
    return ObservationBatch(
        observation_id=observation_id,
        run_generation=run_generation,
        anchor_monotonic_ns=anchor_ns,
        arm_history=arm_history,
        hand_history=hand_history,
        hand_current_history=hand_current_history,
        hand_tactile_sum_history=hand_tactile_sum_history,
        tactile_history=tactile_history,
    )


def inference_loop(shared: SharedStorage, config: DeploymentConfig) -> None:
    """Inference process entry point — produces proposals, never robot commands.

    Startup order: heartbeat early -> lazy import -> instantiate -> load ->
    mark ready. A load/import/instantiation failure raises out of this function
    and becomes a supervisor-observed process failure; there is no dummy safe
    mode. The main loop reads a fresh generation each tick and calls
    ``backend.reset`` when it changes.
    """
    if config is None:
        raise ValueError("inference_loop requires a DeploymentConfig")

    # Heartbeat before any lazy import so the supervisor never sees a dead gap.
    shared.set_heartbeat("inference", time.monotonic())
    metrics = Metrics()

    backend = load_backend(config.backend_target, config=config)
    observation_adapter = load_observation_adapter(config.observation_adapter_target, config=config)
    action_adapter = load_action_adapter(config.action_adapter_target, config=config)

    backend.load()  # raises -> process failure (no dummy safe mode)

    shared.set_ready("inference")
    # Refresh the heartbeat after model loading, which may exceed the timeout.
    shared.set_heartbeat("inference", time.monotonic())
    logger.info("inference_loop: ready (backend=%s)", config.backend_target)

    step_dt_ns = int(round(1e9 / float(shared.action_control_hz)))
    period_s = 1.0 / float(config.inference_hz)

    plan_id = 0
    observation_id = 0
    last_generation = -1
    last_metrics_flush_ns = time.monotonic_ns()

    try:
        while shared.is_running.value:
            tick_start = time.monotonic()
            # Heartbeat every tick, including no-feedback and slow-inference paths.
            shared.set_heartbeat("inference", time.monotonic())

            run_generation = int(shared.run_generation.value)
            if run_generation != last_generation:
                backend.reset(run_generation=run_generation)
                last_generation = run_generation

            anchor_ns = time.monotonic_ns()
            observation_id += 1
            observation = _build_observation(
                shared,
                config,
                observation_id=observation_id,
                run_generation=run_generation,
                anchor_ns=anchor_ns,
            )
            if observation.arm_history is None:
                time.sleep(_NO_FEEDBACK_POLL_S)
                continue  # no causal arm feedback yet — never publish garbage
            if config.hand_enabled and observation.hand_history is None:
                time.sleep(_NO_FEEDBACK_POLL_S)
                continue
            metrics.increment(OBSERVATIONS_BUILT)

            started_ns = time.monotonic_ns()
            model_input = observation_adapter.encode(observation)
            raw_output = backend.infer(model_input)
            finished_ns = time.monotonic_ns()
            metrics.observe(INFERENCE_MS, (finished_ns - started_ns) / 1e6)

            context = InferenceContext(
                run_generation=run_generation,
                observation_id=observation_id,
                observation_anchor_monotonic_ns=anchor_ns,
                inference_started_monotonic_ns=started_ns,
                inference_finished_monotonic_ns=finished_ns,
                step_dt_ns=step_dt_ns,
            )

            try:
                chunk = action_adapter.decode(raw_output, context=context)
            except ValueError as exc:
                # Drop invalid model results; the coordinator's silence watchdog
                # handles a prolonged absence of valid proposals.
                logger.warning("inference: bad model output dropped: %s", exc)
                metrics.increment(INFERENCE_FAILURES)
                continue

            plan_id += 1
            if publish_plan(shared, plan_id=plan_id, context=context, chunk=chunk):
                metrics.increment(PLANS_CREATED)
            else:
                metrics.increment(PLANS_GENERATION_DROPPED)
                logger.debug("inference: plan %d dropped (generation advanced)", plan_id)

            last_metrics_flush_ns = flush_every(
                metrics, last_ns=last_metrics_flush_ns, prefix="inference metrics"
            )

            elapsed = time.monotonic() - tick_start
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        try:
            backend.close()
        except Exception:
            logger.warning("inference: backend.close raised", exc_info=True)
        logger.info("inference_loop: exited")

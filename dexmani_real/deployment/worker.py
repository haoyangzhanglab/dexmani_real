"""Inference worker: observations -> model proposals -> ``policy_plan_ring``.

The inference worker is the *only* process that touches the model. It reads
causal observations from the shared rings, runs
:meth:`~dexmani_real.deployment.contracts.PolicyRuntime.predict`, and publishes
the resulting :class:`~dexmani_real.deployment.contracts.JointActionChunk` to
the latest-wins ``policy_plan_ring``. It never writes ``coupled_cmd_ring``, the
SDK, ``SafetyState``, or ``run_generation`` — model output is a proposal, not a
robot command.

``inference_loop`` is a plain ``*_loop(shared, config)`` function (not an
``mp.Process`` subclass); lifecycle/supervision stays in the A/B runtime.

The module also owns the ``module:symbol`` lazy loader (``load_policy_runtime``)
so torch/CUDA imports happen only inside the inference child process, never in
the parent.
"""

from __future__ import annotations

import importlib
import time
from dataclasses import replace
from typing import Any, cast

import numpy as np

from dexmani_real.deployment.config import DeploymentConfig, PolicyRuntimeConfig
from dexmani_real.deployment.contracts import (
    InferenceContext,
    JointActionChunk,
    PolicyRuntime,
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
    PointCloudFrame,
    parse_observation_fields,
)
from dexmani_real.ipc.channels import RuntimeChannels, new_frame
from dexmani_real.ipc.schema import (
    MAX_POLICY_CHUNK_STEPS,
    POLICY_PLAN_DTYPE,
    validate_point_cloud_array,
)
from dexmani_real.runtime.safety import SafetyState, read_run_epoch
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Poll delay while required causal feedback is unavailable.
_NO_FEEDBACK_POLL_S = 0.005
# Poll interval while ARMED (no inference) — gentler than the feedback poll.
_ARMED_IDLE_POLL_S = 0.01


# A ``module:symbol`` target is imported and instantiated only when the
# inference child calls these loaders, so the parent process never imports
# torch or initializes CUDA, and a checkpoint/model object never crosses
# ``spawn``. Every failure raises (fail closed); there is no dummy safe mode.
def _split_target(target: str) -> tuple[str, str]:
    if not isinstance(target, str) or ":" not in target:
        raise ValueError(f"loader target must be 'module:symbol', got {target!r}")
    module_name, _, symbol = target.rpartition(":")
    if not module_name.strip() or not symbol.strip():
        raise ValueError(f"loader target must be 'module:symbol', got {target!r}")
    return module_name.strip(), symbol.strip()


def _load(
    target: str,
    protocol: type[Any],
    kind: str,
    config: DeploymentConfig | PolicyRuntimeConfig | None,
) -> Any:
    module_name, symbol = _split_target(target)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - fail closed on any import error
        raise ImportError(
            f"failed to import {kind} module {module_name!r}: {exc}"
        ) from exc

    factory = getattr(module, symbol, None)
    if factory is None:
        raise ImportError(f"{kind} module {module_name!r} has no symbol {symbol!r}")
    try:
        instance = factory(config=config) if config is not None else factory()
    except Exception as exc:  # noqa: BLE001 - fail closed on any construction error
        raise TypeError(f"failed to instantiate {kind} {target!r}: {exc}") from exc

    if not isinstance(instance, protocol):
        raise TypeError(f"{kind} {target!r} does not satisfy {protocol.__name__}")
    return instance


def load_policy_runtime(
    target: str,
    *,
    config: DeploymentConfig | PolicyRuntimeConfig | None = None,
) -> PolicyRuntime:
    """Load a :class:`PolicyRuntime` from ``module:symbol``."""
    return cast(
        PolicyRuntime,
        _load(target, PolicyRuntime, "policy runtime", config),
    )


def publish_plan(
    shared: RuntimeChannels,
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

    if chunk.hand_qpos is None:
        raise ValueError(
            "learned-policy output must contain a hand target for every step"
        )

    if chunk.arm_qpos is not None:
        n = int(chunk.arm_qpos.shape[0])
    elif chunk.ee_pos is not None:
        n = int(chunk.ee_pos.shape[0])
    else:
        n = 0
    if n <= 0 or n > MAX_POLICY_CHUNK_STEPS:
        raise ValueError(
            f"policy chunk has {n} steps; transport capacity is {MAX_POLICY_CHUNK_STEPS}"
        )

    frame = new_frame(POLICY_PLAN_DTYPE)
    frame["plan_id"][0] = np.uint64(plan_id)
    frame["run_generation"][0] = np.uint64(context.run_generation)
    frame["observation_id"][0] = np.uint64(context.observation_id)
    frame["observation_anchor_monotonic_ns"][0] = np.uint64(
        context.observation_anchor_monotonic_ns
    )
    frame["observation_latest_source_monotonic_ns"][0] = np.uint64(
        context.observation_latest_source_monotonic_ns
    )
    frame["observation_logical_step_monotonic_ns"][0] = np.uint64(
        context.observation_logical_step_monotonic_ns
    )
    frame["inference_started_monotonic_ns"][0] = np.uint64(
        context.inference_started_monotonic_ns
    )
    frame["inference_finished_monotonic_ns"][0] = np.uint64(
        context.inference_finished_monotonic_ns
    )
    frame["num_steps"][0] = np.uint32(n)
    frame["hand_present"][0] = 1
    frame["target_monotonic_ns"][0, :n] = chunk.target_monotonic_ns
    if chunk.arm_qpos is not None:
        frame["arm_present"][0] = 1
        frame["ee_present"][0] = 0
        frame["arm_qpos"][0, :n] = chunk.arm_qpos
    else:
        frame["arm_present"][0] = 0
        frame["ee_present"][0] = 1
        frame["ee_pos"][0, :n] = chunk.ee_pos
        frame["ee_rot6d"][0, :n] = chunk.ee_rot6d
    frame["hand_qpos"][0, :n] = chunk.hand_qpos
    frame["valid_mask"][0, :n] = chunk.valid_mask
    shared.policy_plan_ring.write(frame)
    return True


def _read_state_history(
    ring,
    *,
    history_len: int,
    anchor_ns: int,
    values_field: str,
    required_true_fields: tuple[str, ...] = (),
    max_age_ns: int | None = None,
    not_before_ns: int = 0,
) -> FrameWindow | None:
    """Read the causal (source <= publish <= anchor) state frames, oldest-first.

    When ``max_age_ns`` is given, frames older than the bound are dropped so a
    stalled feedback stream cannot feed a stale window to the model.
    """
    try:
        history = ring.get_last_k(min(int(history_len), ring.maxlen))
    except Exception:
        logger.warning("inference: state history read failed", exc_info=True)
        return None

    values: list[np.ndarray] = []
    sequences: list[int] = []
    sources: list[int] = []
    publishes: list[int] = []
    for data, ring_publish_ns, sequence in history:
        names = data.dtype.names or ()
        if any(
            field not in names or not bool(data[field][0])
            for field in required_true_fields
        ):
            continue
        source_ns = int(data["source_monotonic_ns"][0])
        publish_ns = (
            int(data["publish_monotonic_ns"][0])
            if "publish_monotonic_ns" in names
            and int(data["publish_monotonic_ns"][0]) > 0
            else int(ring_publish_ns)
        )
        if not (max(0, int(not_before_ns)) <= source_ns <= publish_ns <= anchor_ns):
            continue
        if max_age_ns is not None and anchor_ns - source_ns > max_age_ns:
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


def _pointcloud_frame_from_record(
    record: np.ndarray,
    ring_publish_ns: int,
    *,
    anchor_ns: int,
    max_age_ns: int,
    num_points: int,
    not_before_ns: int,
) -> PointCloudFrame | None:
    """Extract one causal, fresh ``PointCloudFrame`` from a ring record (or None)."""
    source_ns = int(record["source_monotonic_ns"])
    camera_publish_ns = int(record["camera_publish_monotonic_ns"])
    payload_publish_ns = int(record["publish_monotonic_ns"])
    camera_sequence = int(record["source_camera_sequence"])
    camera_generation = int(record["camera_generation"])
    if not (
        camera_sequence > 0
        and camera_generation > 0
        and 0
        < source_ns
        <= camera_publish_ns
        <= payload_publish_ns
        <= int(ring_publish_ns)
        <= anchor_ns
        and anchor_ns - source_ns <= max_age_ns
        and source_ns >= int(not_before_ns)
    ):
        return None
    try:
        cloud = validate_point_cloud_array(
            record["point_cloud"],
            num_points=num_points,
        )
        return PointCloudFrame(
            values=cloud,
            source_camera_sequence=camera_sequence,
            source_monotonic_ns=source_ns,
            publish_monotonic_ns=int(ring_publish_ns),
            camera_generation=camera_generation,
        )
    except ValueError:
        logger.warning("inference: invalid point-cloud payload dropped", exc_info=True)
        return None


def _read_pointcloud_history(
    shared: RuntimeChannels,
    *,
    anchor_ns: int,
    max_age_ns: int,
    num_points: int,
    history_len: int,
    not_before_ns: int,
) -> tuple[PointCloudFrame, ...]:
    """Read the last ``history_len`` causal, fresh clouds, oldest-first."""
    if history_len <= 0:
        return ()
    try:
        result = shared.pointcloud_ring.get_last_k(
            min(int(history_len), shared.pointcloud_ring.maxlen)
        )
    except Exception:
        logger.warning("inference: point-cloud history read failed", exc_info=True)
        return ()
    frames: list[PointCloudFrame] = []
    for data, ring_publish_ns, _sequence in result:
        frame = _pointcloud_frame_from_record(
            data[0],
            int(ring_publish_ns),
            anchor_ns=anchor_ns,
            max_age_ns=max_age_ns,
            num_points=num_points,
            not_before_ns=not_before_ns,
        )
        if frame is not None:
            frames.append(frame)
    if not frames:
        return ()
    # A camera restart bumps camera_generation (new depth-clock mapping); the
    # T-history must be mutually consistent, so drop any frame from an older
    # generation than the newest frame.
    newest_gen = frames[-1].camera_generation
    return tuple(frame for frame in frames if frame.camera_generation == newest_gen)


def _select_pointcloud_control_grid(
    frames: tuple[PointCloudFrame, ...],
    *,
    run_started_ns: int,
    anchor_ns: int,
    history_len: int,
    step_dt_ns: int,
    max_grid_lag_ns: int,
) -> tuple[tuple[PointCloudFrame, ...], int]:
    """Select a strictly advancing causal window on the latest elapsed grid tick."""
    if not frames or history_len <= 0 or step_dt_ns <= 0:
        return (), 0
    if anchor_ns < run_started_ns:
        return (), 0
    latest_tick = (anchor_ns - run_started_ns) // step_dt_ns
    if latest_tick < history_len - 1:
        return (), 0
    logical_step_ns = run_started_ns + latest_tick * step_dt_ns
    selected: list[PointCloudFrame] = []
    previous_sequence = 0
    for offset in range(history_len - 1, -1, -1):
        desired_ns = logical_step_ns - offset * step_dt_ns
        candidates = [
            frame
            for frame in frames
            if frame.source_monotonic_ns <= desired_ns
            and desired_ns - frame.source_monotonic_ns <= max_grid_lag_ns
        ]
        if not candidates:
            return (), 0
        frame = candidates[-1]
        if frame.source_camera_sequence <= previous_sequence:
            return (), 0
        selected.append(frame)
        previous_sequence = frame.source_camera_sequence
    return tuple(selected), logical_step_ns


def _align_state_history_to_pointclouds(
    state_history: FrameWindow | None,
    pointcloud_history: tuple[PointCloudFrame, ...],
    *,
    max_skew_ns: int,
) -> FrameWindow | None:
    """Causally align one state window to the point-cloud reference timeline.

    For every point-cloud source time, choose the newest valid state at or
    before that time. Future state samples and pairs outside the explicit skew
    budget are rejected rather than interpolated or padded.
    """
    if state_history is None or not pointcloud_history:
        return None
    source_ns = np.asarray(state_history.source_monotonic_ns, dtype=np.int64)
    valid = np.asarray(state_history.valid_mask, dtype=np.uint8) == 1
    selected: list[int] = []
    for pointcloud in pointcloud_history:
        reference_ns = np.int64(pointcloud.source_monotonic_ns)
        candidates = np.flatnonzero(
            valid
            & (source_ns <= reference_ns)
            & (reference_ns - source_ns <= int(max_skew_ns))
        )
        if candidates.size == 0:
            return None
        selected.append(int(candidates[-1]))
    indices = np.asarray(selected, dtype=np.intp)
    return FrameWindow(
        values=state_history.values[indices],
        source_sequence=state_history.source_sequence[indices],
        source_monotonic_ns=state_history.source_monotonic_ns[indices],
        publish_monotonic_ns=state_history.publish_monotonic_ns[indices],
        valid_mask=np.ones(len(indices), dtype=np.uint8),
    )


def _build_observation(
    shared: RuntimeChannels,
    config: DeploymentConfig | PolicyRuntimeConfig,
    *,
    observation_id: int,
    run_generation: int,
    run_started_ns: int,
    anchor_ns: int,
    step_dt_ns: int,
) -> ObservationBatch | None:
    """Assemble requested causal modalities from the arm/hand rings.

    The hand state and tactile rings are read only when their corresponding
    ``observation_fields`` are requested. Every selected frame is additionally
    gated by its source/publish timestamps and modality-specific health flags.
    """
    horizon = int(config.observation_horizon)
    max_age_ns = int(config.max_input_age_s * 1e9)
    max_skew_ns = int(config.max_observation_skew_s * 1e9)
    max_grid_lag_ns = int(config.max_grid_lag_s * 1e9)
    history_span_ns = max(0, horizon - 1) * int(step_dt_ns)
    pointcloud_history_max_age_ns = max_age_ns + history_span_ns + max_grid_lag_ns
    state_history_max_age_ns = pointcloud_history_max_age_ns + max_skew_ns
    hand_history: FrameWindow | None = None
    hand_current_history: FrameWindow | None = None
    hand_tactile_sum_history: FrameWindow | None = None
    tactile_history: FrameWindow | None = None
    pointcloud: PointCloudFrame | None = None
    pointcloud_history: tuple[PointCloudFrame, ...] = ()
    requested = set(
        parse_observation_fields(
            getattr(config, "observation_fields", "arm_qpos,hand_qpos")
        )
    )
    pointcloud_requested = "point_cloud" in requested
    state_history_len = (
        shared.arm_state_ring.maxlen if pointcloud_requested else horizon
    )
    arm_history = _read_state_history(
        shared.arm_state_ring,
        history_len=state_history_len,
        anchor_ns=anchor_ns,
        values_field="qpos",
        required_true_fields=("state_valid",),
        max_age_ns=(state_history_max_age_ns if pointcloud_requested else max_age_ns),
        not_before_ns=run_started_ns,
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
    if pointcloud_requested:
        all_pointclouds = _read_pointcloud_history(
            shared,
            anchor_ns=anchor_ns,
            max_age_ns=pointcloud_history_max_age_ns,
            num_points=int(config.pointcloud_num_points),
            history_len=shared.pointcloud_ring.maxlen,
            not_before_ns=run_started_ns,
        )
        pointcloud_history, logical_step_ns = _select_pointcloud_control_grid(
            all_pointclouds,
            run_started_ns=run_started_ns,
            anchor_ns=anchor_ns,
            history_len=horizon,
            step_dt_ns=step_dt_ns,
            max_grid_lag_ns=max_grid_lag_ns,
        )
        if len(pointcloud_history) == horizon:
            pointcloud = pointcloud_history[-1]
    else:
        logical_step_ns = 0
    if config.hand_enabled:
        if hand_state_requested:
            hand_history = _read_state_history(
                shared.hand_state_ring,
                history_len=(
                    shared.hand_state_ring.maxlen if pointcloud_requested else horizon
                ),
                anchor_ns=anchor_ns,
                values_field="qpos",
                required_true_fields=("state_valid",),
                max_age_ns=(
                    state_history_max_age_ns if pointcloud_requested else max_age_ns
                ),
                not_before_ns=run_started_ns,
            )
            if requested & {"hand_current", "hand_joint_torque"}:
                hand_current_history = _read_state_history(
                    shared.hand_state_ring,
                    history_len=(
                        shared.hand_state_ring.maxlen
                        if pointcloud_requested
                        else horizon
                    ),
                    anchor_ns=anchor_ns,
                    values_field="current",
                    required_true_fields=("state_valid",),
                    max_age_ns=(
                        state_history_max_age_ns if pointcloud_requested else max_age_ns
                    ),
                    not_before_ns=run_started_ns,
                )
            if requested & {"hand_tactile_sum", "fingertip_force"}:
                hand_tactile_sum_history = _read_state_history(
                    shared.hand_state_ring,
                    history_len=(
                        shared.hand_state_ring.maxlen
                        if pointcloud_requested
                        else horizon
                    ),
                    anchor_ns=anchor_ns,
                    values_field="tactile_sum",
                    required_true_fields=(
                        "state_valid",
                        "tactile_sum_valid",
                    ),
                    max_age_ns=(
                        state_history_max_age_ns if pointcloud_requested else max_age_ns
                    ),
                    not_before_ns=run_started_ns,
                )
        if tactile_requested:
            tactile_history = _read_state_history(
                shared.hand_tactile_ring,
                history_len=(
                    shared.hand_tactile_ring.maxlen if pointcloud_requested else horizon
                ),
                anchor_ns=anchor_ns,
                values_field="tactile_force",
                required_true_fields=("fresh",),
                max_age_ns=(
                    state_history_max_age_ns if pointcloud_requested else max_age_ns
                ),
                not_before_ns=run_started_ns,
            )
    if pointcloud_requested and len(pointcloud_history) == horizon:
        arm_history = _align_state_history_to_pointclouds(
            arm_history,
            pointcloud_history,
            max_skew_ns=max_skew_ns,
        )
        if hand_state_requested:
            hand_history = _align_state_history_to_pointclouds(
                hand_history,
                pointcloud_history,
                max_skew_ns=max_skew_ns,
            )
        if hand_current_history is not None:
            hand_current_history = _align_state_history_to_pointclouds(
                hand_current_history,
                pointcloud_history,
                max_skew_ns=max_skew_ns,
            )
        if hand_tactile_sum_history is not None:
            hand_tactile_sum_history = _align_state_history_to_pointclouds(
                hand_tactile_sum_history,
                pointcloud_history,
                max_skew_ns=max_skew_ns,
            )
        if tactile_history is not None:
            tactile_history = _align_state_history_to_pointclouds(
                tactile_history,
                pointcloud_history,
                max_skew_ns=max_skew_ns,
            )
    if pointcloud_requested:
        if pointcloud is None or logical_step_ns <= 0:
            return None
        latest_source_ns = int(pointcloud.source_monotonic_ns)
        if anchor_ns - latest_source_ns > max_age_ns:
            return None
    elif arm_history is not None and arm_history.values.shape[0] > 0:
        latest_source_ns = int(arm_history.source_monotonic_ns[-1])
        logical_step_ns = latest_source_ns
    else:
        return None
    return ObservationBatch(
        observation_id=observation_id,
        run_generation=run_generation,
        run_started_monotonic_ns=run_started_ns,
        anchor_monotonic_ns=anchor_ns,
        latest_source_monotonic_ns=latest_source_ns,
        logical_step_monotonic_ns=logical_step_ns,
        arm_history=arm_history,
        hand_history=hand_history,
        hand_current_history=hand_current_history,
        hand_tactile_sum_history=hand_tactile_sum_history,
        tactile_history=tactile_history,
        pointcloud=pointcloud,
        pointcloud_history=pointcloud_history,
    )


def inference_loop(shared: RuntimeChannels, config: PolicyRuntimeConfig) -> None:
    """Inference process entry point — produces proposals, never robot commands.

    Startup order: heartbeat early -> lazy import -> instantiate -> load ->
    mark ready. A load/import/instantiation failure raises out of this function
    and becomes a supervisor-observed process failure; there is no dummy safe
    mode. The main loop reads a fresh generation each tick and calls
    ``runtime.reset_episode`` when it changes.
    """
    if config is None:
        raise ValueError("inference_loop requires a PolicyRuntimeConfig")

    # Heartbeat before any lazy import so the supervisor never sees a dead gap.
    shared.set_heartbeat("inference", time.monotonic())
    metrics = Metrics()

    runtime = load_policy_runtime(config.runtime_target, config=config)

    runtime.load()  # raises -> process failure (no dummy safe mode)

    shared.set_ready("inference")
    # Refresh the heartbeat after model loading, which may exceed the timeout.
    shared.set_heartbeat("inference", time.monotonic())
    logger.info("inference_loop: ready (runtime=%s)", config.runtime_target)

    step_dt_ns = int(round(1e9 / float(shared.action_control_hz)))
    period_s = 1.0 / float(config.inference_hz)
    requested = set(parse_observation_fields(config.observation_fields))

    plan_id = 0
    observation_id = 0
    last_generation = -1
    last_logical_step_ns = 0
    last_metrics_flush_ns = time.monotonic_ns()

    try:
        while shared.is_running.value:
            tick_start = time.monotonic()
            # Heartbeat every tick, including no-feedback and slow-inference paths.
            shared.set_heartbeat("inference", time.monotonic())

            epoch = read_run_epoch(shared)
            run_generation = epoch.generation
            if run_generation != last_generation:
                runtime.reset_episode()
                last_generation = run_generation
                observation_id = 0  # new observation epoch for the new run
                last_logical_step_ns = 0

            # ARMED = no inference; the coordinator gates RUNNING via B.
            if int(shared.safety_state.value) != int(SafetyState.RUNNING):
                time.sleep(_ARMED_IDLE_POLL_S)
                continue
            if epoch.started_monotonic_ns <= 0:
                raise RuntimeError("RUNNING state has no observation epoch")

            anchor_ns = time.monotonic_ns()
            observation_id += 1
            observation = _build_observation(
                shared,
                config,
                observation_id=observation_id,
                run_generation=run_generation,
                run_started_ns=epoch.started_monotonic_ns,
                anchor_ns=anchor_ns,
                step_dt_ns=step_dt_ns,
            )
            if observation is None:
                time.sleep(_NO_FEEDBACK_POLL_S)
                continue
            horizon = int(config.observation_horizon)
            if (
                observation.arm_history is None
                or observation.arm_history.values.shape[0] != horizon
            ):
                time.sleep(_NO_FEEDBACK_POLL_S)
                continue  # no complete causal arm history yet — never infer
            if config.hand_enabled and (
                observation.hand_history is None
                or observation.hand_history.values.shape[0] != horizon
            ):
                time.sleep(_NO_FEEDBACK_POLL_S)
                continue
            if (
                "point_cloud" in requested
                and len(observation.pointcloud_history) != horizon
            ):
                time.sleep(_NO_FEEDBACK_POLL_S)
                continue
            if observation.logical_step_monotonic_ns <= last_logical_step_ns:
                time.sleep(_NO_FEEDBACK_POLL_S)
                continue
            metrics.increment(OBSERVATIONS_BUILT)
            last_logical_step_ns = observation.logical_step_monotonic_ns

            started_ns = time.monotonic_ns()
            # The adapter timestamps actions on the observation's logical
            # control grid; the causal cut is retained only as provenance.
            predict_context = InferenceContext(
                run_generation=run_generation,
                observation_id=observation_id,
                observation_anchor_monotonic_ns=anchor_ns,
                observation_latest_source_monotonic_ns=(
                    observation.latest_source_monotonic_ns
                ),
                observation_logical_step_monotonic_ns=(
                    observation.logical_step_monotonic_ns
                ),
                inference_started_monotonic_ns=started_ns,
                inference_finished_monotonic_ns=started_ns,
                step_dt_ns=step_dt_ns,
            )
            chunk = runtime.predict(observation, context=predict_context)
            finished_ns = time.monotonic_ns()
            metrics.observe(INFERENCE_MS, (finished_ns - started_ns) / 1e6)

            # Preserve the model's semantic grid. Expired endpoints are masked;
            # they are never shifted forward and made plausible again.
            earliest_target_ns = finished_ns + int(config.command_lead_s * 1e9)
            future_mask = (
                np.asarray(chunk.valid_mask, dtype=np.uint8)
                & (np.asarray(chunk.target_monotonic_ns) > earliest_target_ns)
            ).astype(np.uint8)
            if not bool(np.any(future_mask)):
                metrics.increment(INFERENCE_FAILURES)
                elapsed = time.monotonic() - tick_start
                if period_s > elapsed:
                    time.sleep(period_s - elapsed)
                continue
            chunk = replace(chunk, valid_mask=future_mask)

            context = InferenceContext(
                run_generation=run_generation,
                observation_id=observation_id,
                observation_anchor_monotonic_ns=anchor_ns,
                observation_latest_source_monotonic_ns=(
                    observation.latest_source_monotonic_ns
                ),
                observation_logical_step_monotonic_ns=(
                    observation.logical_step_monotonic_ns
                ),
                inference_started_monotonic_ns=started_ns,
                inference_finished_monotonic_ns=finished_ns,
                step_dt_ns=step_dt_ns,
            )

            plan_id += 1
            if publish_plan(shared, plan_id=plan_id, context=context, chunk=chunk):
                metrics.increment(PLANS_CREATED)
            else:
                metrics.increment(PLANS_GENERATION_DROPPED)
                logger.debug(
                    "inference: plan %d dropped (generation advanced)", plan_id
                )

            last_metrics_flush_ns = flush_every(
                metrics, last_ns=last_metrics_flush_ns, prefix="inference metrics"
            )

            elapsed = time.monotonic() - tick_start
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        try:
            runtime.close()
        except Exception:
            logger.warning("inference: runtime.close raised", exc_info=True)
        logger.info("inference_loop: exited")

"""Inference worker: observations -> model proposals -> ``policy_plan_ring``.

The inference worker is the *only* process that touches the model. It reads
causal observations from the shared rings, runs
:meth:`~dexmani_real.deployment.contracts.PolicyRuntime.predict`, and publishes
the resulting :class:`~dexmani_real.deployment.contracts.JointActionChunk` to
the latest-wins ``policy_plan_ring``. It never writes ``coupled_cmd_ring``, the
SDK, ``SafetyState``, or ``run_generation`` — model output is a proposal, not a
robot command.

``inference_loop`` is a plain ``*_loop(shared, config)`` function (not an
``mp.Process`` subclass); lifecycle and supervision stay in the runtime layer.

Artifact-bound DexMani deployments load through the preflight module's checked
fd/hash/provenance stream loader. There is no configurable runtime loader.
"""

from __future__ import annotations

import math
import time

import numpy as np

from dexmani_real.deployment.config import (
    FIXED_POLICY_RUNTIME_TARGET,
    PolicyRuntimeConfig,
)
from dexmani_real.deployment.contracts import (
    InferenceContext,
    JointActionChunk,
    PolicyPrediction,
    PolicyRuntime,
)
from dexmani_real.deployment.metrics import (
    INFERENCE_FAILURES,
    INFERENCE_MS,
    OBSERVATION_AGE_MS,
    OBSERVATION_SKEW_MS,
    OBSERVATION_WAIT_ARM_HISTORY,
    OBSERVATION_WAIT_GRID_ADVANCE,
    OBSERVATION_WAIT_HAND_HISTORY,
    OBSERVATION_WAIT_POINTCLOUD_GRID,
    OBSERVATION_WAIT_POINTCLOUD_HISTORY,
    OBSERVATION_WAIT_POINTCLOUD_STALE,
    OBSERVATION_WAIT_RGB_GRID,
    OBSERVATION_WAIT_RGB_HISTORY,
    OBSERVATIONS_BUILT,
    PLANS_CREATED,
    PLANS_GENERATION_DROPPED,
    Metrics,
    flush_every,
    inference_run_receipt_json,
)
from dexmani_real.deployment.observation import (
    FrameWindow,
    ObservationBatch,
    PointCloudFrame,
    RgbFrame,
    parse_observation_fields,
)
from dexmani_real.deployment.timing import build_target_grid, first_deliverable_index
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
_MIN_STARTUP_DELIVERABLE_TARGETS = 2


def _load_inference_runtime(config: PolicyRuntimeConfig) -> PolicyRuntime:
    """Load the single Real-owned runtime through the verified stream boundary."""
    if config.artifact is None:
        raise ValueError("inference runtime requires a resolved artifact")
    # Deliberately imported inside the inference child: the parent must not
    # import torch/Policy or initialize CUDA, and model objects never cross spawn.
    from dexmani_real.deployment.preflight import load_verified_policy_runtime

    return load_verified_policy_runtime(config)


def _duration_s_to_ns_ceil(duration_s: float, *, name: str) -> int:
    """Convert a validated non-negative duration without understating latency."""
    if isinstance(duration_s, bool) or not isinstance(
        duration_s, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a number")
    value = float(duration_s)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return math.ceil(value * 1e9)


def _startup_deliverable_target_count(
    *,
    model_latency_s: float,
    steps: int,
    step_dt_ns: int,
    command_lead_s: float,
) -> int:
    """Return theoretical targets remaining after model latency and command lead."""
    # This arbitrary positive origin has no source-time meaning; warmup only
    # qualifies latency against relative executable-grid spacing.
    origin_ns = 1
    targets = build_target_grid(origin_ns, steps, step_dt_ns)
    finished_ns = origin_ns + _duration_s_to_ns_ceil(
        model_latency_s,
        name="model_latency_s",
    )
    first_index = first_deliverable_index(
        targets,
        finished_ns,
        _duration_s_to_ns_ceil(command_lead_s, name="command_lead_s"),
    )
    return len(targets) - first_index


def stamp_prediction_timing(
    prediction: PolicyPrediction,
    *,
    logical_step_ns: int,
    step_dt_ns: int,
    inference_finished_ns: int,
    command_lead_ns: int,
) -> JointActionChunk | None:
    """Assign the fixed logical control grid, masking expired predictions only.

    The model never controls timestamps.  Targets remain exactly
    ``logical_step_ns + i * step_dt_ns``; a late inference may lose a prefix
    or the whole prediction, but it is never shifted forward.
    """
    if not isinstance(prediction, PolicyPrediction):
        raise TypeError("prediction must be a PolicyPrediction")
    values = (
        prediction.arm_qpos if prediction.arm_qpos is not None else prediction.ee_pos
    )
    assert values is not None
    steps = int(values.shape[0])
    targets = build_target_grid(logical_step_ns, steps, step_dt_ns)
    first_index = first_deliverable_index(
        targets,
        inference_finished_ns,
        command_lead_ns,
    )
    if first_index == steps:
        return None
    valid_mask = np.zeros(steps, dtype=np.uint8)
    valid_mask[first_index:] = 1
    return JointActionChunk(
        arm_qpos=prediction.arm_qpos,
        hand_qpos=prediction.hand_qpos,
        ee_pos=prediction.ee_pos,
        ee_rot6d=prediction.ee_rot6d,
        target_monotonic_ns=targets,
        valid_mask=valid_mask,
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
    required_false_fields: tuple[str, ...] = (),
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
        if any(
            field not in names or bool(data[field][0])
            for field in required_false_fields
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


def _rgb_frame_from_camera_record(
    shared: RuntimeChannels,
    header: np.ndarray,
    ring_publish_ns: int,
    sequence: int,
    *,
    anchor_ns: int,
    max_age_ns: int,
    not_before_ns: int,
    expected_shape: tuple[int, int, int],
) -> RgbFrame | None:
    """Copy one verified, causal raw RGB frame from the camera ring."""
    record = header[0]
    source_ns = int(record["source_monotonic_ns"])
    camera_publish_ns = int(record["publish_monotonic_ns"])
    camera_generation = int(record["camera_generation"])
    if not (
        sequence > 0
        and camera_generation > 0
        and int(record["camera_health"]) == 0
        and not bool(record["clock_reset"])
        and 0 < source_ns <= camera_publish_ns <= ring_publish_ns <= anchor_ns
        and anchor_ns - source_ns <= max_age_ns
        and source_ns >= not_before_ns
    ):
        return None
    payload = shared.camera_ring.read_sequence(sequence, modalities=("rgb",))
    if payload is None:
        return None
    payload_header = payload["header"][0]
    if (
        int(payload_header["source_monotonic_ns"]) != source_ns
        or int(payload_header["publish_monotonic_ns"]) != camera_publish_ns
        or int(payload_header["camera_generation"]) != camera_generation
    ):
        return None
    rgb = payload["rgb"]
    if rgb.shape != expected_shape:
        logger.warning(
            "inference: RGB frame shape %s does not match artifact %s",
            rgb.shape,
            expected_shape,
        )
        return None
    try:
        return RgbFrame(
            values=rgb,
            source_camera_sequence=sequence,
            source_monotonic_ns=source_ns,
            publish_monotonic_ns=camera_publish_ns,
            camera_generation=camera_generation,
        )
    except ValueError:
        logger.warning("inference: invalid RGB payload dropped", exc_info=True)
        return None


def _read_rgb_history(
    shared: RuntimeChannels,
    *,
    anchor_ns: int,
    max_age_ns: int,
    history_len: int,
    not_before_ns: int,
    expected_shape: tuple[int, int, int],
) -> tuple[RgbFrame, ...]:
    """Read the verified causal RGB frames still resident in camera shared memory."""
    if history_len <= 0:
        return ()
    try:
        records = shared.camera_ring.get_last_metadata(
            min(history_len, shared.camera_ring.maxlen)
        )
    except Exception:
        logger.warning("inference: RGB history metadata read failed", exc_info=True)
        return ()
    frames: list[RgbFrame] = []
    for header, ring_publish_ns, sequence in records:
        frame = _rgb_frame_from_camera_record(
            shared,
            header,
            int(ring_publish_ns),
            int(sequence),
            anchor_ns=anchor_ns,
            max_age_ns=max_age_ns,
            not_before_ns=not_before_ns,
            expected_shape=expected_shape,
        )
        if frame is not None:
            frames.append(frame)
    if not frames:
        return ()
    newest_generation = frames[-1].camera_generation
    return tuple(
        frame for frame in frames if frame.camera_generation == newest_generation
    )


def _read_rgb_for_pointcloud_history(
    shared: RuntimeChannels,
    pointcloud_history: tuple[PointCloudFrame, ...],
    *,
    anchor_ns: int,
    max_age_ns: int,
    not_before_ns: int,
    expected_shape: tuple[int, int, int],
) -> tuple[RgbFrame, ...]:
    """Read RGB frames with exactly the camera provenance selected for clouds."""
    try:
        metadata_by_sequence = {
            int(sequence): (header, int(ring_publish_ns))
            for header, ring_publish_ns, sequence in shared.camera_ring.get_last_metadata(
                shared.camera_ring.maxlen
            )
        }
    except Exception:
        logger.warning("inference: RGB provenance metadata read failed", exc_info=True)
        return ()
    frames: list[RgbFrame] = []
    for pointcloud in pointcloud_history:
        metadata = metadata_by_sequence.get(pointcloud.source_camera_sequence)
        if metadata is None:
            return ()
        header, ring_publish_ns = metadata
        frame = _rgb_frame_from_camera_record(
            shared,
            header,
            ring_publish_ns,
            pointcloud.source_camera_sequence,
            anchor_ns=anchor_ns,
            max_age_ns=max_age_ns,
            not_before_ns=not_before_ns,
            expected_shape=expected_shape,
        )
        if frame is None or (
            frame.source_monotonic_ns != pointcloud.source_monotonic_ns
            or frame.camera_generation != pointcloud.camera_generation
        ):
            return ()
        frames.append(frame)
    return tuple(frames)


def _select_camera_control_grid(
    frames: tuple[PointCloudFrame | RgbFrame, ...],
    *,
    run_started_ns: int,
    anchor_ns: int,
    history_len: int,
    step_dt_ns: int,
    max_grid_lag_ns: int,
) -> tuple[tuple[PointCloudFrame | RgbFrame, ...], int]:
    """Select a strictly advancing causal visual window on the policy grid."""
    if not frames or history_len <= 0 or step_dt_ns <= 0:
        return (), 0
    if anchor_ns < run_started_ns:
        return (), 0
    latest_tick = (anchor_ns - run_started_ns) // step_dt_ns
    if latest_tick < history_len - 1:
        return (), 0
    logical_step_ns = run_started_ns + latest_tick * step_dt_ns
    selected: list[PointCloudFrame | RgbFrame] = []
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


def _select_pointcloud_control_grid(
    frames: tuple[PointCloudFrame, ...],
    *,
    run_started_ns: int,
    anchor_ns: int,
    history_len: int,
    step_dt_ns: int,
    max_grid_lag_ns: int,
) -> tuple[tuple[PointCloudFrame, ...], int]:
    """Point-cloud typed wrapper retained for the existing worker boundary."""
    selected, logical_step_ns = _select_camera_control_grid(
        frames,
        run_started_ns=run_started_ns,
        anchor_ns=anchor_ns,
        history_len=history_len,
        step_dt_ns=step_dt_ns,
        max_grid_lag_ns=max_grid_lag_ns,
    )
    if not all(isinstance(frame, PointCloudFrame) for frame in selected):
        raise RuntimeError("point-cloud selection returned a non-point-cloud frame")
    return (
        tuple(frame for frame in selected if isinstance(frame, PointCloudFrame)),
        logical_step_ns,
    )


def _align_state_history_to_camera_frames(
    state_history: FrameWindow | None,
    camera_history: tuple[PointCloudFrame | RgbFrame, ...],
    *,
    max_skew_ns: int,
) -> FrameWindow | None:
    """Causally align state to a selected point-cloud or RGB reference timeline.

    For every camera source time, choose the newest valid state at or
    before that time. Future state samples and pairs outside the explicit skew
    budget are rejected rather than interpolated or padded.
    """
    if state_history is None or not camera_history:
        return None
    source_ns = np.asarray(state_history.source_monotonic_ns, dtype=np.int64)
    valid = np.asarray(state_history.valid_mask, dtype=np.uint8) == 1
    selected: list[int] = []
    for camera_frame in camera_history:
        reference_ns = np.int64(camera_frame.source_monotonic_ns)
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
    config: PolicyRuntimeConfig,
    *,
    observation_id: int,
    run_generation: int,
    run_started_ns: int,
    anchor_ns: int,
    step_dt_ns: int,
    metrics: Metrics | None = None,
) -> ObservationBatch | None:
    """Assemble requested causal modalities from the arm/hand rings.

    The hand state and tactile rings are read only when their corresponding
    ``observation_fields`` are requested. Every selected frame is additionally
    gated by its source/publish timestamps and modality-specific health flags.
    """
    deployment = config.deployment
    horizon = int(deployment.observation_horizon)
    max_age_ns = int(deployment.max_input_age_s * 1e9)
    max_skew_ns = int(deployment.max_observation_skew_s * 1e9)
    max_grid_lag_ns = int(deployment.max_grid_lag_s * 1e9)
    history_span_ns = max(0, horizon - 1) * int(step_dt_ns)
    visual_history_max_age_ns = max_age_ns + history_span_ns + max_grid_lag_ns
    state_history_max_age_ns = visual_history_max_age_ns + max_skew_ns
    hand_history: FrameWindow | None = None
    hand_current_history: FrameWindow | None = None
    hand_tactile_sum_history: FrameWindow | None = None
    tactile_history: FrameWindow | None = None
    pointcloud: PointCloudFrame | None = None
    pointcloud_history: tuple[PointCloudFrame, ...] = ()
    requested = set(parse_observation_fields(deployment.observation_fields))
    pointcloud_requested = "point_cloud" in requested
    rgb_requested = "rgb" in requested
    camera_requested = pointcloud_requested or rgb_requested
    rgb_history: tuple[RgbFrame, ...] = ()
    rgb_shape: tuple[int, int, int] | None = None
    if rgb_requested:
        artifact = config.artifact
        if artifact is None or artifact.allocation_contract.rgb_shape is None:
            raise ValueError("RGB observation requires an artifact RGB contract")
        rgb_shape = artifact.allocation_contract.rgb_shape
    state_history_len = shared.arm_state_ring.maxlen if camera_requested else horizon
    arm_history = _read_state_history(
        shared.arm_state_ring,
        history_len=state_history_len,
        anchor_ns=anchor_ns,
        values_field="qpos",
        required_true_fields=("state_valid",),
        max_age_ns=(state_history_max_age_ns if camera_requested else max_age_ns),
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
            max_age_ns=visual_history_max_age_ns,
            num_points=int(deployment.pointcloud_num_points),
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
            if rgb_requested:
                assert rgb_shape is not None
                rgb_history = _read_rgb_for_pointcloud_history(
                    shared,
                    pointcloud_history,
                    anchor_ns=anchor_ns,
                    max_age_ns=visual_history_max_age_ns,
                    not_before_ns=run_started_ns,
                    expected_shape=rgb_shape,
                )
    elif rgb_requested:
        assert rgb_shape is not None
        all_rgb = _read_rgb_history(
            shared,
            anchor_ns=anchor_ns,
            max_age_ns=visual_history_max_age_ns,
            history_len=shared.camera_ring.maxlen,
            not_before_ns=run_started_ns,
            expected_shape=rgb_shape,
        )
        selected_rgb, logical_step_ns = _select_camera_control_grid(
            all_rgb,
            run_started_ns=run_started_ns,
            anchor_ns=anchor_ns,
            history_len=horizon,
            step_dt_ns=step_dt_ns,
            max_grid_lag_ns=max_grid_lag_ns,
        )
        if not all(isinstance(frame, RgbFrame) for frame in selected_rgb):
            raise RuntimeError("RGB selection returned a non-RGB camera frame")
        rgb_history = tuple(
            frame for frame in selected_rgb if isinstance(frame, RgbFrame)
        )
    else:
        logical_step_ns = 0
    if deployment.hand_enabled:
        if hand_state_requested:
            hand_history = _read_state_history(
                shared.hand_state_ring,
                history_len=(
                    shared.hand_state_ring.maxlen if camera_requested else horizon
                ),
                anchor_ns=anchor_ns,
                values_field="qpos",
                required_true_fields=("state_valid",),
                required_false_fields=("qpos_stale",),
                max_age_ns=(
                    state_history_max_age_ns if camera_requested else max_age_ns
                ),
                not_before_ns=run_started_ns,
            )
            if requested & {"hand_current", "hand_joint_torque"}:
                hand_current_history = _read_state_history(
                    shared.hand_state_ring,
                    history_len=(
                        shared.hand_state_ring.maxlen if camera_requested else horizon
                    ),
                    anchor_ns=anchor_ns,
                    values_field="current",
                    required_true_fields=("state_valid",),
                    required_false_fields=("qpos_stale",),
                    max_age_ns=(
                        state_history_max_age_ns if camera_requested else max_age_ns
                    ),
                    not_before_ns=run_started_ns,
                )
            if requested & {"hand_tactile_sum", "fingertip_force"}:
                hand_tactile_sum_history = _read_state_history(
                    shared.hand_state_ring,
                    history_len=(
                        shared.hand_state_ring.maxlen if camera_requested else horizon
                    ),
                    anchor_ns=anchor_ns,
                    values_field="tactile_sum",
                    required_true_fields=(
                        "state_valid",
                        "tactile_sum_valid",
                    ),
                    required_false_fields=("qpos_stale",),
                    max_age_ns=(
                        state_history_max_age_ns if camera_requested else max_age_ns
                    ),
                    not_before_ns=run_started_ns,
                )
        if tactile_requested:
            tactile_history = _read_state_history(
                shared.hand_tactile_ring,
                history_len=(
                    shared.hand_tactile_ring.maxlen if camera_requested else horizon
                ),
                anchor_ns=anchor_ns,
                values_field="tactile_force",
                required_true_fields=("fresh",),
                max_age_ns=(
                    state_history_max_age_ns if camera_requested else max_age_ns
                ),
                not_before_ns=run_started_ns,
            )
    reference_history: tuple[PointCloudFrame | RgbFrame, ...]
    if pointcloud_requested:
        reference_history = pointcloud_history
    else:
        reference_history = rgb_history
    if camera_requested and len(reference_history) == horizon:
        arm_history = _align_state_history_to_camera_frames(
            arm_history,
            reference_history,
            max_skew_ns=max_skew_ns,
        )
        if hand_state_requested:
            hand_history = _align_state_history_to_camera_frames(
                hand_history,
                reference_history,
                max_skew_ns=max_skew_ns,
            )
        if hand_current_history is not None:
            hand_current_history = _align_state_history_to_camera_frames(
                hand_current_history,
                reference_history,
                max_skew_ns=max_skew_ns,
            )
        if hand_tactile_sum_history is not None:
            hand_tactile_sum_history = _align_state_history_to_camera_frames(
                hand_tactile_sum_history,
                reference_history,
                max_skew_ns=max_skew_ns,
            )
        if tactile_history is not None:
            tactile_history = _align_state_history_to_camera_frames(
                tactile_history,
                reference_history,
                max_skew_ns=max_skew_ns,
            )
    if pointcloud_requested:
        if pointcloud is None or logical_step_ns <= 0:
            if metrics is not None:
                metrics.increment(OBSERVATION_WAIT_POINTCLOUD_GRID)
            return None
        latest_source_ns = int(pointcloud.source_monotonic_ns)
        if anchor_ns - latest_source_ns > max_age_ns:
            if metrics is not None:
                metrics.increment(OBSERVATION_WAIT_POINTCLOUD_STALE)
            return None
        if rgb_requested and len(rgb_history) != horizon:
            if metrics is not None:
                metrics.increment(OBSERVATION_WAIT_RGB_HISTORY)
            return None
    elif rgb_requested:
        if len(rgb_history) != horizon or logical_step_ns <= 0:
            if metrics is not None:
                metrics.increment(OBSERVATION_WAIT_RGB_GRID)
            return None
        latest_source_ns = int(rgb_history[-1].source_monotonic_ns)
        if anchor_ns - latest_source_ns > max_age_ns:
            if metrics is not None:
                metrics.increment(OBSERVATION_WAIT_RGB_GRID)
            return None
    elif arm_history is not None and arm_history.values.shape[0] > 0:
        latest_source_ns = int(arm_history.source_monotonic_ns[-1])
        logical_step_ns = latest_source_ns
    else:
        if metrics is not None:
            metrics.increment(OBSERVATION_WAIT_ARM_HISTORY)
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
        rgb_history=rgb_history,
    )


def observation_timing_ms(observation: ObservationBatch) -> tuple[float, float]:
    """Return causal latest-frame age and cross-modality skew in milliseconds.

    Age is measured from the causal cut to the newest source frame.  Skew uses
    the newest valid frame of each modality, rather than the history span of a
    single modality, so a normal ``n_obs_steps`` window is not misreported as
    sensor skew.
    """
    latest_sources: list[int] = []
    for window in (
        getattr(observation, "arm_history", None),
        getattr(observation, "hand_history", None),
        getattr(observation, "hand_current_history", None),
        getattr(observation, "hand_tactile_sum_history", None),
        getattr(observation, "tactile_history", None),
    ):
        if window is None:
            continue
        valid_mask = getattr(window, "valid_mask", None)
        source_ns = getattr(window, "source_monotonic_ns", None)
        if valid_mask is None or source_ns is None:
            continue
        valid = np.asarray(valid_mask, dtype=np.uint8) == 1
        if np.any(valid):
            latest_sources.append(int(np.max(np.asarray(source_ns)[valid])))
    pointcloud = getattr(observation, "pointcloud", None)
    if pointcloud is not None:
        latest_sources.append(int(pointcloud.source_monotonic_ns))
    rgb_history = getattr(observation, "rgb_history", ())
    if rgb_history:
        latest_sources.append(int(rgb_history[-1].source_monotonic_ns))
    if not latest_sources:
        latest_sources.append(int(observation.latest_source_monotonic_ns))
    latest_ns = max(latest_sources)
    anchor_ns = int(
        getattr(
            observation, "anchor_monotonic_ns", observation.logical_step_monotonic_ns
        )
    )
    if latest_ns > anchor_ns:
        raise ValueError("observation source timestamp exceeds causal cut")
    return (
        (anchor_ns - latest_ns) / 1e6,
        (latest_ns - min(latest_sources)) / 1e6,
    )


def inference_loop(shared: RuntimeChannels, config: PolicyRuntimeConfig) -> None:
    """Inference process entry point — produces proposals, never robot commands.

    Startup order: heartbeat early -> verified artifact load -> mark ready.
    A load/import/instantiation failure raises out
    of this function and becomes a supervisor-observed process failure; there
    is no dummy safe mode. The main loop reads a fresh generation each tick and
    calls ``runtime.reset_episode`` when it changes.
    """
    if config is None:
        raise ValueError("inference_loop requires a PolicyRuntimeConfig")

    # Heartbeat before any lazy import so the supervisor never sees a dead gap.
    shared.set_heartbeat("inference", time.monotonic())
    metrics = Metrics()

    runtime = _load_inference_runtime(config)

    try:
        if config.artifact is not None:
            warmup_samples = 5
            stable_samples = 3
            timings_s = runtime.warmup(samples=warmup_samples)
            if len(timings_s) != warmup_samples:
                raise RuntimeError(
                    "policy runtime returned an incomplete warmup receipt"
                )
            if any(not np.isfinite(value) or value < 0.0 for value in timings_s):
                raise RuntimeError("policy runtime returned invalid warmup timing")
            allocation = config.artifact.allocation_contract
            stable_timings_s = timings_s[-stable_samples:]
            step_dt_ns = int(round(1e9 / float(shared.action_control_hz)))
            remaining_targets = tuple(
                _startup_deliverable_target_count(
                    model_latency_s=value,
                    steps=allocation.n_action_steps,
                    step_dt_ns=step_dt_ns,
                    command_lead_s=config.deployment.command_lead_s,
                )
                for value in stable_timings_s
            )
            logger.info(
                "inference warmup model-latency qualification: samples_ms=%s "
                "stable_remaining_targets=%s minimum=%d",
                ",".join(f"{value * 1e3:.3f}" for value in timings_s),
                ",".join(str(value) for value in remaining_targets),
                _MIN_STARTUP_DELIVERABLE_TARGETS,
            )
            if any(
                remaining < _MIN_STARTUP_DELIVERABLE_TARGETS
                for remaining in remaining_targets
            ):
                raise RuntimeError(
                    "policy inference warmup exceeds the viable action window: "
                    f"stable_max_ms={max(stable_timings_s) * 1e3:.3f} "
                    f"stable_remaining_targets={remaining_targets} "
                    f"minimum={_MIN_STARTUP_DELIVERABLE_TARGETS}"
                )
    except BaseException:
        try:
            runtime.close()
        except Exception:
            logger.warning("inference: startup runtime.close raised", exc_info=True)
        raise

    shared.set_ready("inference")
    # Refresh the heartbeat after model loading, which may exceed the timeout.
    shared.set_heartbeat("inference", time.monotonic())
    logger.info("inference_loop: ready (runtime=%s)", FIXED_POLICY_RUNTIME_TARGET)

    step_dt_ns = int(round(1e9 / float(shared.action_control_hz)))
    deployment = config.deployment
    period_s = 1.0 / float(deployment.inference_hz)
    requested = set(parse_observation_fields(deployment.observation_fields))

    plan_id = 0
    observation_id = 0
    last_generation = -1
    metrics_run_generation: int | None = None
    last_logical_step_ns = 0
    last_metrics_flush_ns = time.monotonic_ns()

    def emit_inference_run_receipt(reason: str) -> None:
        """Emit one complete receipt before this run's metrics are cleared."""
        nonlocal metrics_run_generation
        if metrics_run_generation is None:
            return
        receipt = inference_run_receipt_json(
            run_generation=metrics_run_generation,
            reason=reason,
            metrics=metrics.run_snapshot(),
        )
        logger.info("inference run receipt: %s", receipt)
        metrics_run_generation = None

    def wait_for_observation(reason: str | None = None) -> None:
        nonlocal last_metrics_flush_ns
        if reason is not None:
            metrics.increment(reason)
        last_metrics_flush_ns = flush_every(
            metrics,
            last_ns=last_metrics_flush_ns,
            prefix="inference metrics",
        )
        time.sleep(_NO_FEEDBACK_POLL_S)

    try:
        while shared.is_running.value:
            tick_start = time.monotonic()
            # Heartbeat every tick, including no-feedback and slow-inference paths.
            shared.set_heartbeat("inference", time.monotonic())

            epoch = read_run_epoch(shared)
            run_generation = epoch.generation
            if run_generation != last_generation:
                emit_inference_run_receipt("run generation advanced")
                runtime.reset_episode()
                last_generation = run_generation
                observation_id = 0  # new observation epoch for the new run
                last_logical_step_ns = 0
                metrics.begin_run()

            # ARMED = no inference; the coordinator gates RUNNING via B.
            if int(shared.safety_state.value) != int(SafetyState.RUNNING):
                time.sleep(_ARMED_IDLE_POLL_S)
                continue
            if epoch.started_monotonic_ns <= 0:
                raise RuntimeError("RUNNING state has no observation epoch")
            if metrics_run_generation is None:
                metrics_run_generation = run_generation

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
                metrics=metrics,
            )
            if observation is None:
                wait_for_observation()
                continue
            horizon = int(deployment.observation_horizon)
            if (
                observation.arm_history is None
                or observation.arm_history.values.shape[0] != horizon
            ):
                wait_for_observation(OBSERVATION_WAIT_ARM_HISTORY)
                continue  # no complete causal arm history yet — never infer
            if deployment.hand_enabled and (
                observation.hand_history is None
                or observation.hand_history.values.shape[0] != horizon
            ):
                wait_for_observation(OBSERVATION_WAIT_HAND_HISTORY)
                continue
            if (
                "point_cloud" in requested
                and len(observation.pointcloud_history) != horizon
            ):
                wait_for_observation(OBSERVATION_WAIT_POINTCLOUD_HISTORY)
                continue
            if "rgb" in requested and len(observation.rgb_history) != horizon:
                wait_for_observation(OBSERVATION_WAIT_RGB_HISTORY)
                continue
            if observation.logical_step_monotonic_ns <= last_logical_step_ns:
                wait_for_observation(OBSERVATION_WAIT_GRID_ADVANCE)
                continue
            metrics.increment(OBSERVATIONS_BUILT)
            observation_age_ms, observation_skew_ms = observation_timing_ms(observation)
            metrics.observe(OBSERVATION_AGE_MS, observation_age_ms)
            metrics.observe_timing(OBSERVATION_AGE_MS, observation_age_ms)
            metrics.observe(OBSERVATION_SKEW_MS, observation_skew_ms)
            metrics.observe_timing(OBSERVATION_SKEW_MS, observation_skew_ms)
            last_logical_step_ns = observation.logical_step_monotonic_ns

            started_ns = time.monotonic_ns()
            prediction = runtime.predict(observation)
            finished_ns = time.monotonic_ns()
            inference_ms = (finished_ns - started_ns) / 1e6
            metrics.observe(INFERENCE_MS, inference_ms)
            metrics.observe_timing(INFERENCE_MS, inference_ms)

            chunk = stamp_prediction_timing(
                prediction,
                logical_step_ns=observation.logical_step_monotonic_ns,
                step_dt_ns=step_dt_ns,
                inference_finished_ns=finished_ns,
                command_lead_ns=_duration_s_to_ns_ceil(
                    deployment.command_lead_s,
                    name="command_lead_s",
                ),
            )
            if chunk is None:
                metrics.increment(INFERENCE_FAILURES)
                elapsed = time.monotonic() - tick_start
                if period_s > elapsed:
                    time.sleep(period_s - elapsed)
                continue

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
            emit_inference_run_receipt("worker exit")
        finally:
            try:
                runtime.close()
            except Exception:
                logger.warning("inference: runtime.close raised", exc_info=True)
        logger.info("inference_loop: exited")

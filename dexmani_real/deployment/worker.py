"""Inference worker: observations -> flat predictions -> ``prediction_ring``.

The inference worker is the *only* process that touches the model. It reads
causal observations from the shared rings, runs
:meth:`~dexmani_real.deployment.contracts.PolicyRuntime.predict`, and publishes
the resulting :class:`~dexmani_real.deployment.contracts.Prediction` to the
latest-wins ``prediction_ring``. It never writes ``coupled_cmd_ring``, the
SDK, ``SafetyState``, or ``run_generation`` — model output is a proposal, not a
robot command.

``inference_loop`` is a plain ``*_loop(shared, config)`` function (not an
``mp.Process`` subclass); lifecycle and supervision stay in the runtime layer.

The worker loads the selected experiment through the public DexMani Policy API,
then wraps it with Real's NumPy observation/action adapter.
"""

from __future__ import annotations

import time

import numpy as np

from dexmani_real.config.defaults import PolicyParams
from dexmani_real.deployment.config import (
    FIXED_POLICY_RUNTIME_TARGET,
    FingertipAssemblerConfig,
    PolicyDeploymentConfig,
    PolicyWorkerConfig,
    policy_observation_fields,
)
from dexmani_real.deployment.contracts import PolicyRuntime, Prediction
from dexmani_real.deployment.metrics import PolicyStats, flush_every
from dexmani_real.deployment.observation import (
    FrameWindow,
    ObservationBatch,
    PointCloudFrame,
    PolicyObservation,
    RgbFrame,
)
from dexmani_real.deployment.timing import next_periodic_deadline_ns
from dexmani_real.ipc.channels import RuntimeChannels, new_frame
from dexmani_real.ipc.schema import PREDICTION_DTYPE, validate_point_cloud_array
from dexmani_real.planning.arm_fk import make_arm_fk
from dexmani_real.planning.fingertip import compute_fingertip_points_xarm_base
from dexmani_real.planning.hand_fk import HandKinematics
from dexmani_real.runtime.safety import SafetyState, read_run_state_snapshot
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Poll delay while required causal feedback is unavailable.
_NO_FEEDBACK_POLL_S = 0.005
# Poll interval while ARMED (no inference) — gentler than the feedback poll.
_ARMED_IDLE_POLL_S = 0.01
_SYNC_REQUEST_WAIT_S = 0.05


def _requested_observation_fields(policy_spec: object) -> set[str]:
    """Return source names directly from the validated ordered Policy fields."""
    return {field.name for field in policy_observation_fields(policy_spec)}


def _load_inference_runtime(config: PolicyWorkerConfig) -> PolicyRuntime:
    """Load Policy-owned model state and return the Real NumPy adapter."""
    # Deliberately imported inside the inference child: the parent must not
    # import torch/Policy or initialize CUDA, and model objects never cross spawn.
    from dexmani_policy.deployment import load_experiment

    from dexmani_real.integrations.dexmani_policy import DexManiPolicyRuntime

    loaded_policy = load_experiment(
        config.experiment,
        device=config.device,
        seed=config.seed,
    )
    try:
        return DexManiPolicyRuntime(loaded_policy, config.spec)
    except BaseException:
        loaded_policy.close()
        raise


def serialize_prediction(prediction: Prediction) -> np.ndarray:
    """Serialize one validated flat prediction without scheduling it."""
    if not isinstance(prediction, Prediction):
        raise TypeError("prediction must be a Prediction")
    frame = new_frame(PREDICTION_DTYPE)
    frame["run_generation"][0] = np.uint64(prediction.run_generation)
    frame["source_monotonic_ns"][0] = np.uint64(prediction.source_monotonic_ns)
    frame["logical_step_monotonic_ns"][0] = np.uint64(
        prediction.logical_step_monotonic_ns
    )
    frame["num_steps"][0] = np.uint32(prediction.num_steps)
    frame["action_dim"][0] = np.uint32(prediction.actions.shape[1])
    frame["actions"][0, : prediction.num_steps, : prediction.actions.shape[1]] = (
        prediction.actions
    )
    return frame


def publish_prediction(shared: RuntimeChannels, prediction: Prediction) -> bool:
    """Generation-fence and publish one flat prediction to the single-slot ring."""
    if not isinstance(prediction, Prediction):
        raise TypeError("prediction must be a Prediction")
    if int(shared.run_generation.value) != prediction.run_generation:
        return False
    shared.prediction_ring.write(serialize_prediction(prediction))
    return True


def _clear_sync_request_for_inactive_snapshot(
    shared: RuntimeChannels,
    *,
    observed_generation: int,
) -> bool:
    """Clear only while the lifecycle snapshot still names an inactive epoch.

    The safety transition and this check share ``motion_lock``. If a B request
    has already advanced the generation to RUNNING, its newly-set inference
    request cannot be cleared using an older ARMED snapshot.
    """
    with shared.motion_lock:
        if int(shared.run_generation.value) != int(observed_generation):
            return False
        if (
            int(shared.safety_state.value) == int(SafetyState.RUNNING)
            and not bool(shared.error_state.value)
            and not bool(shared.estop_request.value)
        ):
            return False
        shared.inference_request.clear()
        return True


def _consume_sync_request(
    shared: RuntimeChannels,
    *,
    observed_generation: int,
) -> int | None:
    """Consume a request only if it still belongs to the observed RUNNING epoch."""
    with shared.motion_lock:
        current_generation = int(shared.run_generation.value)
        if current_generation != int(observed_generation):
            return None
        if int(shared.safety_state.value) != int(SafetyState.RUNNING):
            return None
        if bool(shared.error_state.value) or bool(shared.estop_request.value):
            return None
        shared.inference_request.clear()
        return current_generation


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


def _read_tactile_provenance_history(
    ring,
    *,
    anchor_ns: int,
    history_len: int,
    max_age_ns: int,
    not_before_ns: int,
) -> FrameWindow | None:
    """Read only tactile provenance flags; never copy the full tactile tensor."""
    try:
        history = ring.get_last_k(min(int(history_len), ring.maxlen))
    except Exception:
        logger.warning("inference: tactile provenance read failed", exc_info=True)
        return None
    values, sequences, sources, publishes = [], [], [], []
    for data, ring_publish_ns, sequence in history:
        record = data[0]
        source_ns = int(record["source_monotonic_ns"])
        publish_ns = int(ring_publish_ns)
        if not (
            bool(record["fresh"])
            and bool(record["calibrated"])
            and int(record["unit_code"]) == 0
        ):
            continue
        if not (not_before_ns <= source_ns <= publish_ns <= anchor_ns):
            continue
        if anchor_ns - source_ns > max_age_ns:
            continue
        values.append(np.array([int(record["unit_code"])], dtype=np.uint8))
        sequences.append(int(sequence))
        sources.append(source_ns)
        publishes.append(publish_ns)
    if not values:
        return None
    return FrameWindow(
        values=np.stack(values),
        source_sequence=np.asarray(sequences, dtype=np.uint64),
        source_monotonic_ns=np.asarray(sources, dtype=np.uint64),
        publish_monotonic_ns=np.asarray(publishes, dtype=np.uint64),
        valid_mask=np.ones(len(values), dtype=np.uint8),
    )


def _select_control_grid_reference_ns(
    *, run_started_ns: int, anchor_ns: int, history_len: int, step_dt_ns: int
) -> tuple[np.ndarray, int]:
    """Return the last T completed episode-grid times, oldest first."""
    if history_len <= 0 or step_dt_ns <= 0 or anchor_ns < run_started_ns:
        return np.empty(0, dtype=np.uint64), 0
    latest_tick = (anchor_ns - run_started_ns) // step_dt_ns
    if latest_tick < history_len - 1:
        return np.empty(0, dtype=np.uint64), 0
    logical_step_ns = run_started_ns + latest_tick * step_dt_ns
    first_ns = logical_step_ns - (history_len - 1) * step_dt_ns
    return (
        np.arange(first_ns, logical_step_ns + 1, step_dt_ns, dtype=np.uint64),
        logical_step_ns,
    )


def _align_state_history_to_reference_ns(
    state_history: FrameWindow | None,
    reference_ns: np.ndarray,
    *,
    max_skew_ns: int,
) -> FrameWindow | None:
    """Choose newest source <= each reference, within the explicit skew bound."""
    if state_history is None or np.asarray(reference_ns).size == 0:
        return None
    sources = np.asarray(state_history.source_monotonic_ns, dtype=np.int64)
    valid = np.asarray(state_history.valid_mask, dtype=np.uint8) == 1
    selected: list[int] = []
    for value in np.asarray(reference_ns, dtype=np.int64):
        candidates = np.flatnonzero(
            valid & (sources <= value) & (value - sources <= max_skew_ns)
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
    camera_ring,
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
    payload = camera_ring.read_sequence(sequence, modalities=("rgb",))
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
            "inference: RGB frame shape %s does not match policy contract %s",
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
            shared.camera_ring,
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
    camera_ring,
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
            for header, ring_publish_ns, sequence in camera_ring.get_last_metadata(
                camera_ring.maxlen
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
            camera_ring,
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
    return _align_state_history_to_reference_ns(
        state_history,
        np.asarray(
            [frame.source_monotonic_ns for frame in camera_history], dtype=np.uint64
        ),
        max_skew_ns=max_skew_ns,
    )


def _build_observation(
    shared: RuntimeChannels,
    policy: PolicyParams,
    policy_spec: object,
    *,
    observation_id: int,
    run_generation: int,
    run_started_ns: int,
    anchor_ns: int,
    step_dt_ns: int,
) -> ObservationBatch | None:
    """Assemble requested causal modalities from the arm/hand rings.

    Policy modalities are projected to their concrete Real sensor fields.
    Every selected frame is additionally
    gated by its source/publish timestamps and modality-specific health flags.
    """
    horizon = int(getattr(policy_spec, "n_obs_steps"))
    max_age_ns = int(policy.max_input_age_s * 1e9)
    max_skew_ns = int(policy.max_observation_skew_s * 1e9)
    max_grid_lag_ns = int(policy.max_grid_lag_s * 1e9)
    history_span_ns = max(0, horizon - 1) * int(step_dt_ns)
    visual_history_max_age_ns = max_age_ns + history_span_ns + max_grid_lag_ns
    state_history_max_age_ns = visual_history_max_age_ns + max_skew_ns
    hand_history: FrameWindow | None = None
    hand_tactile_sum_history: FrameWindow | None = None
    hand_tactile_provenance_history: FrameWindow | None = None
    pointcloud: PointCloudFrame | None = None
    pointcloud_history: tuple[PointCloudFrame, ...] = ()
    requested = _requested_observation_fields(policy_spec)
    pointcloud_requested = "point_cloud" in requested
    rgb_requested = "rgb" in requested
    camera_requested = pointcloud_requested or rgb_requested
    rgb_history: tuple[RgbFrame, ...] = ()
    fields = {field.name: field for field in policy_observation_fields(policy_spec)}
    rgb_shape = tuple(fields["rgb"].shape) if rgb_requested else None
    state_history_len = shared.arm_state_ring.maxlen
    arm_history = _read_state_history(
        shared.arm_state_ring,
        history_len=state_history_len,
        anchor_ns=anchor_ns,
        values_field="qpos",
        required_true_fields=("state_valid",),
        max_age_ns=state_history_max_age_ns,
        not_before_ns=run_started_ns,
    )
    hand_state_requested = bool(
        requested & {"joint_state", "contact_force", "fingertip_points"}
    )
    tactile_requested = "contact_force" in requested
    if pointcloud_requested:
        all_pointclouds = _read_pointcloud_history(
            shared,
            anchor_ns=anchor_ns,
            max_age_ns=visual_history_max_age_ns,
            num_points=int(fields["point_cloud"].shape[0]),
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
                    shared.camera_ring,
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
        reference_ns, logical_step_ns = _select_control_grid_reference_ns(
            run_started_ns=run_started_ns,
            anchor_ns=anchor_ns,
            history_len=horizon,
            step_dt_ns=step_dt_ns,
        )
    if getattr(policy_spec, "requires_hand") is True:
        if hand_state_requested:
            hand_history = _read_state_history(
                shared.hand_state_ring,
                history_len=(shared.hand_state_ring.maxlen),
                anchor_ns=anchor_ns,
                values_field="qpos",
                required_true_fields=("state_valid",),
                required_false_fields=("qpos_stale",),
                max_age_ns=state_history_max_age_ns,
                not_before_ns=run_started_ns,
            )
            if "contact_force" in requested:
                hand_tactile_sum_history = _read_state_history(
                    shared.hand_state_ring,
                    history_len=shared.hand_state_ring.maxlen,
                    anchor_ns=anchor_ns,
                    values_field="tactile_sum",
                    required_true_fields=(
                        "state_valid",
                        "tactile_sum_valid",
                    ),
                    required_false_fields=("qpos_stale",),
                    max_age_ns=state_history_max_age_ns,
                    not_before_ns=run_started_ns,
                )
        if tactile_requested:
            hand_tactile_provenance_history = _read_tactile_provenance_history(
                shared.hand_tactile_ring,
                history_len=shared.hand_tactile_ring.maxlen,
                anchor_ns=anchor_ns,
                max_age_ns=state_history_max_age_ns,
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
        if hand_tactile_sum_history is not None:
            hand_tactile_sum_history = _align_state_history_to_camera_frames(
                hand_tactile_sum_history,
                reference_history,
                max_skew_ns=max_skew_ns,
            )
        if hand_tactile_provenance_history is not None:
            hand_tactile_provenance_history = _align_state_history_to_camera_frames(
                hand_tactile_provenance_history,
                reference_history,
                max_skew_ns=max_skew_ns,
            )
    elif not camera_requested and logical_step_ns > 0:
        arm_history = _align_state_history_to_reference_ns(
            arm_history, reference_ns, max_skew_ns=max_skew_ns
        )
        if hand_history is not None:
            hand_history = _align_state_history_to_reference_ns(
                hand_history, reference_ns, max_skew_ns=max_skew_ns
            )
        if hand_tactile_sum_history is not None:
            hand_tactile_sum_history = _align_state_history_to_reference_ns(
                hand_tactile_sum_history, reference_ns, max_skew_ns=max_skew_ns
            )
        if hand_tactile_provenance_history is not None:
            hand_tactile_provenance_history = _align_state_history_to_reference_ns(
                hand_tactile_provenance_history,
                reference_ns,
                max_skew_ns=max_skew_ns,
            )
    if pointcloud_requested:
        if pointcloud is None or logical_step_ns <= 0:
            return None
        latest_source_ns = int(pointcloud.source_monotonic_ns)
        if anchor_ns - latest_source_ns > max_age_ns:
            return None
        if rgb_requested and len(rgb_history) != horizon:
            return None
    elif rgb_requested:
        if len(rgb_history) != horizon or logical_step_ns <= 0:
            return None
        latest_source_ns = int(rgb_history[-1].source_monotonic_ns)
        if anchor_ns - latest_source_ns > max_age_ns:
            return None
    elif (
        arm_history is not None
        and arm_history.values.shape[0] == horizon
        and logical_step_ns > 0
    ):
        latest_source_ns = max(
            int(window.source_monotonic_ns[-1])
            for window in (
                arm_history,
                hand_history,
                hand_tactile_sum_history,
                hand_tactile_provenance_history,
            )
            if window is not None
        )
        if anchor_ns - latest_source_ns > max_age_ns:
            return None
    else:
        return None
    if arm_history is None or arm_history.values.shape[0] != horizon:
        return None
    if hand_state_requested and (
        hand_history is None or hand_history.values.shape[0] != horizon
    ):
        return None
    if tactile_requested:
        if hand_tactile_sum_history is None or hand_tactile_provenance_history is None:
            return None
        if not np.array_equal(
            hand_tactile_sum_history.source_monotonic_ns,
            hand_tactile_provenance_history.source_monotonic_ns,
        ):
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
        hand_tactile_sum_history=hand_tactile_sum_history,
        hand_tactile_provenance_history=hand_tactile_provenance_history,
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
        getattr(observation, "hand_tactile_sum_history", None),
        getattr(observation, "hand_tactile_provenance_history", None),
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


def _to_policy_observation(
    observation: ObservationBatch,
    policy_spec: object,
    *,
    fingertip_runtime: (
        tuple[object, HandKinematics, FingertipAssemblerConfig] | None
    ) = None,
) -> PolicyObservation:
    """Project typed ring readers into the exact public Policy array mapping."""
    field_names = tuple(field.name for field in policy_observation_fields(policy_spec))
    horizon = int(getattr(policy_spec, "n_obs_steps"))
    if observation.arm_history is None or observation.hand_history is None:
        raise ValueError("joint_state requires aligned arm and hand histories")
    arrays: dict[str, np.ndarray] = {}
    joint = np.concatenate(
        (observation.arm_history.values, observation.hand_history.values), axis=1
    )
    arrays["joint_state"] = np.ascontiguousarray(joint, dtype=np.float32)
    if "point_cloud" in field_names:
        arrays["point_cloud"] = np.ascontiguousarray(
            np.stack([frame.values for frame in observation.pointcloud_history]),
            dtype=np.float32,
        )
    if "rgb" in field_names:
        arrays["rgb"] = np.ascontiguousarray(
            np.stack([frame.values for frame in observation.rgb_history]),
            dtype=np.uint8,
        )
    if "contact_force" in field_names:
        if (
            observation.hand_tactile_sum_history is None
            or observation.hand_tactile_provenance_history is None
        ):
            raise ValueError("contact_force lacks calibrated tactile provenance")
        arrays["contact_force"] = np.ascontiguousarray(
            observation.hand_tactile_sum_history.values, dtype=np.float32
        )
    if "fingertip_points" in field_names:
        if fingertip_runtime is None:
            raise RuntimeError("fingertip_points requires local FK")
        arm_fk, hand_fk, config = fingertip_runtime
        arrays["fingertip_points"] = np.ascontiguousarray(
            np.stack(
                [
                    compute_fingertip_points_xarm_base(
                        observation.arm_history.values[index],
                        observation.hand_history.values[index],
                        arm_fk=arm_fk,
                        hand_fk=hand_fk,
                        handbase_position_eef_m=np.asarray(
                            config.handbase_position_eef_m
                        ),
                        handbase_quat_eef_wxyz=np.asarray(
                            config.handbase_quat_eef_wxyz
                        ),
                    )
                    for index in range(horizon)
                ]
            ),
            dtype=np.float32,
        )
    ordered = {name: arrays[name] for name in field_names}
    return PolicyObservation(
        observation_id=observation.observation_id,
        run_generation=observation.run_generation,
        anchor_monotonic_ns=observation.anchor_monotonic_ns,
        latest_source_monotonic_ns=observation.latest_source_monotonic_ns,
        logical_step_monotonic_ns=observation.logical_step_monotonic_ns,
        arrays=ordered,
    )


def inference_loop(
    shared: RuntimeChannels,
    policy: PolicyParams,
    config: PolicyWorkerConfig,
    deployment_config: PolicyDeploymentConfig | None = None,
    fingertip_config: FingertipAssemblerConfig | None = None,
) -> None:
    """Inference process entry point — produces proposals, never robot commands.

    Startup order: heartbeat early -> Policy-owned strict load -> mark ready.
    A load/import/instantiation failure raises out
    of this function and becomes a supervisor-observed process failure; there
    is no dummy safe mode. The main loop reads a fresh generation each tick and
    calls ``runtime.reset_episode`` when it changes.
    """
    if not isinstance(policy, PolicyParams):
        raise TypeError("inference_loop requires resolved runtime PolicyParams")
    if not isinstance(config, PolicyWorkerConfig):
        raise TypeError("inference_loop requires a PolicyWorkerConfig")
    deployment = deployment_config or PolicyDeploymentConfig()
    if not isinstance(deployment, PolicyDeploymentConfig):
        raise TypeError("inference_loop requires a PolicyDeploymentConfig")

    # Heartbeat before any lazy import so the supervisor never sees a dead gap.
    shared.set_heartbeat("inference", time.monotonic())
    stats = PolicyStats()

    runtime = _load_inference_runtime(config)
    try:
        fingertip_runtime = None
        if "fingertip_points" in _requested_observation_fields(config.spec):
            if not isinstance(fingertip_config, FingertipAssemblerConfig):
                raise TypeError("fingertip_points requires FingertipAssemblerConfig")
            hand_fk = HandKinematics(
                fingertip_config.hand_urdf_path,
                list(fingertip_config.fingertip_link_names),
            )
            if not hand_fk.is_ready():
                raise RuntimeError("fingertip FK startup failed")
            fingertip_runtime = (make_arm_fk(), hand_fk, fingertip_config)
        warmup_samples = 5
        timings_s = runtime.warmup(samples=warmup_samples)
        if len(timings_s) != warmup_samples:
            raise RuntimeError("policy runtime returned incomplete warmup timings")
        if any(not np.isfinite(value) or value < 0.0 for value in timings_s):
            raise RuntimeError("policy runtime returned invalid warmup timing")
        logger.info(
            "inference warmup: samples_ms=%s mode=%s",
            ",".join(f"{value * 1e3:.3f}" for value in timings_s),
            deployment.inference_mode,
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

    step_dt_ns = int(round(float(config.spec.control_dt_s) * 1e9))
    async_period_ns = int(config.spec.n_action_steps) * step_dt_ns

    observation_id = 0
    last_generation = -1
    last_logical_step_ns = 0
    sync_request_generation: int | None = None
    async_deadline_ns: int | None = None
    last_metrics_flush_ns = time.monotonic_ns()

    def wait_for_observation() -> None:
        nonlocal last_metrics_flush_ns
        last_metrics_flush_ns = flush_every(
            stats,
            last_ns=last_metrics_flush_ns,
            prefix="inference metrics",
            debug=True,
        )
        time.sleep(_NO_FEEDBACK_POLL_S)

    try:
        while shared.is_running.value:
            # Heartbeat every tick, including no-feedback and slow-inference paths.
            shared.set_heartbeat("inference", time.monotonic())

            run_snapshot = read_run_state_snapshot(shared)
            run_generation = run_snapshot.generation
            if run_generation != last_generation:
                runtime.reset_episode()
                last_generation = run_generation
                observation_id = 0  # new observation epoch for the new run
                last_logical_step_ns = 0
                sync_request_generation = None
                async_deadline_ns = None

            # ARMED = no inference; the policy executor gates RUNNING via B.
            if run_snapshot.state is not SafetyState.RUNNING:
                if deployment.inference_mode == "sync":
                    _clear_sync_request_for_inactive_snapshot(
                        shared,
                        observed_generation=run_generation,
                    )
                    sync_request_generation = None
                else:
                    async_deadline_ns = None
                time.sleep(_ARMED_IDLE_POLL_S)
                continue
            if bool(shared.error_state.value) or bool(shared.estop_request.value):
                if deployment.inference_mode == "sync":
                    _clear_sync_request_for_inactive_snapshot(
                        shared,
                        observed_generation=run_generation,
                    )
                    sync_request_generation = None
                else:
                    async_deadline_ns = None
                time.sleep(_ARMED_IDLE_POLL_S)
                continue
            if run_snapshot.started_monotonic_ns <= 0:
                raise RuntimeError("RUNNING state has no observation epoch")
            if deployment.inference_mode == "sync":
                if sync_request_generation is None:
                    if not shared.inference_request.wait(timeout=_SYNC_REQUEST_WAIT_S):
                        continue
                    shared.set_heartbeat("inference", time.monotonic())
                    request_generation = _consume_sync_request(
                        shared,
                        observed_generation=run_generation,
                    )
                    if request_generation is None:
                        continue
                    run_snapshot = read_run_state_snapshot(shared)
                    if (
                        run_snapshot.state is not SafetyState.RUNNING
                        or run_snapshot.generation != request_generation
                    ):
                        continue
                    if run_snapshot.started_monotonic_ns <= 0:
                        raise RuntimeError("RUNNING state has no observation epoch")
                    run_generation = run_snapshot.generation
                    sync_request_generation = run_generation
                elif sync_request_generation != run_generation:
                    sync_request_generation = None
                    continue
            else:
                now_ns = time.monotonic_ns()
                if async_deadline_ns is None:
                    async_deadline_ns = now_ns
                if now_ns < async_deadline_ns:
                    time.sleep(
                        min(
                            (async_deadline_ns - now_ns) / 1e9,
                            _SYNC_REQUEST_WAIT_S,
                        )
                    )
                    continue
            anchor_ns = time.monotonic_ns()
            observation_id += 1
            observation = _build_observation(
                shared,
                policy,
                config.spec,
                observation_id=observation_id,
                run_generation=run_generation,
                run_started_ns=run_snapshot.started_monotonic_ns,
                anchor_ns=anchor_ns,
                step_dt_ns=step_dt_ns,
            )
            if observation is None:
                wait_for_observation()
                continue
            if observation.logical_step_monotonic_ns <= last_logical_step_ns:
                wait_for_observation()
                continue
            observation_age_ms, observation_skew_ms = observation_timing_ms(observation)
            stats.observe_observation_age_ms(observation_age_ms)
            stats.observe_observation_skew_ms(observation_skew_ms)
            last_logical_step_ns = observation.logical_step_monotonic_ns

            started_ns = time.monotonic_ns()
            policy_observation = _to_policy_observation(
                observation,
                config.spec,
                fingertip_runtime=fingertip_runtime,
            )
            actions = runtime.predict(policy_observation)
            finished_ns = time.monotonic_ns()
            inference_ms = (finished_ns - started_ns) / 1e6
            stats.observe_inference_latency_ms(inference_ms)

            prediction = Prediction(
                run_generation=run_generation,
                source_monotonic_ns=observation.latest_source_monotonic_ns,
                logical_step_monotonic_ns=observation.logical_step_monotonic_ns,
                actions=actions,
            )
            if not publish_prediction(shared, prediction):
                logger.debug("inference: prediction dropped (generation advanced)")
            if deployment.inference_mode == "sync":
                sync_request_generation = None
            else:
                assert async_deadline_ns is not None
                async_deadline_ns = next_periodic_deadline_ns(
                    async_deadline_ns,
                    async_period_ns,
                    finished_ns,
                )

            last_metrics_flush_ns = flush_every(
                stats,
                last_ns=last_metrics_flush_ns,
                prefix="inference metrics",
                debug=True,
            )
    finally:
        try:
            runtime.close()
        except Exception:
            logger.warning("inference: runtime.close raised", exc_info=True)
        logger.info("inference_loop: exited")

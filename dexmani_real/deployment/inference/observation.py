"""Process-local observation windows for the deployment runtime.

These types never enter RuntimeChannels and therefore carry no
IPC dtype. They are the ``PolicyRuntime`` input contract.

Shared-memory readers take ownership copies. The builder owns temporal and
payload admission; these process-local containers only carry assembled values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dexmani_real.config.defaults import PolicyParams
from dexmani_real.deployment.config import FingertipAssemblerConfig
from dexmani_real.ipc.channels import RuntimeChannels
from dexmani_real.ipc.schema import (
    TACTILE_UNIT_CODE_XHAND_SDK_NATIVE,
    validate_point_cloud_array,
)
from dexmani_real.planning.kinematics.arm_fk import (
    compute_eef_pose_history_xarm_base,
    make_arm_fk,
)
from dexmani_real.planning.kinematics.fingertip import (
    compute_fingertip_history_xarm_base,
)
from dexmani_real.planning.kinematics.hand_fk import HandKinematics
from dexmani_real.sensor.camera.transforms import resize_rgb
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class PolicyObservation:
    """Narrow NumPy boundary passed to a Policy runtime.

    Mapping insertion order is the validated Policy modality order. Arrays are
    C-contiguous, writeable, inference-process-owned model inputs.
    """

    observation_id: int
    run_generation: int
    anchor_monotonic_ns: int
    latest_source_monotonic_ns: int
    logical_step_monotonic_ns: int
    arrays: Mapping[str, np.ndarray]


def _validate_finite(array: np.ndarray, *, name: str) -> None:
    if array.dtype.kind in "fc" and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")


@dataclass(frozen=True)
class FrameWindow:
    """Oldest-first window of frames from one ring + aligned per-frame metadata.

    ``values`` is the feature tensor with leading axis = number of frames ``T``
    (arm ``[T,7]``, hand ``[T,12]``, tactile-sum ``[T,5,3]``). The metadata
    arrays are all ``[T]`` and aligned to that same axis. ``valid_mask[i] == 0``
    marks a padding slot whose frame must be ignored by consumers.
    """

    values: np.ndarray
    source_sequence: np.ndarray
    source_monotonic_ns: np.ndarray
    publish_monotonic_ns: np.ndarray
    valid_mask: np.ndarray


@dataclass(frozen=True)
class PointCloudFrame:
    """One fixed-size xArm-base point cloud and its causal provenance."""

    values: np.ndarray  # [N, 6] float32
    source_camera_sequence: int
    source_monotonic_ns: int
    publish_monotonic_ns: int
    camera_generation: int


@dataclass(frozen=True)
class RgbFrame:
    """One raw RGB camera frame and its causal provenance."""

    values: np.ndarray  # [H, W, 3] uint8, RGB, [0, 255]
    source_camera_sequence: int
    source_monotonic_ns: int
    publish_monotonic_ns: int
    camera_generation: int


@dataclass(frozen=True)
class ObservationBatch:
    """One causal observation assembled from state and point-cloud rings.

    Process-local. ``arm_history``/``hand_history`` and
    ``hand_tactile_sum_history`` are sensor-value ``FrameWindow`` values.
    ``hand_tactile_force_history`` carries the full ``[T,5,120,3]`` tactile
    tensor and also serves as provenance when PolicySpec requests ``tactile_force``;
    otherwise ``hand_tactile_provenance_history`` contains only the unit-code proof
    aligned to tactile sums, so contact-only policies never copy the full
    tactile tensor.
    ``pointcloud`` is the latest causally valid ``PointCloudFrame``. Optional
    modalities are None when not requested.
    ``anchor_monotonic_ns`` is the causal cut: no frame published after the
    anchor may be included.
    """

    observation_id: int
    run_generation: int
    run_started_monotonic_ns: int
    anchor_monotonic_ns: int
    latest_source_monotonic_ns: int
    logical_step_monotonic_ns: int

    arm_history: FrameWindow | None = None
    hand_history: FrameWindow | None = None
    hand_tactile_sum_history: FrameWindow | None = None
    hand_tactile_provenance_history: FrameWindow | None = None
    hand_tactile_force_history: FrameWindow | None = None
    pointcloud: PointCloudFrame | None = None
    # Oldest-first causal window of recent point-cloud frames; ``pointcloud`` is
    # the latest (and last element) when non-empty.  ``point_cloud`` models use
    # this window for their per-step point-cloud history.
    pointcloud_history: tuple[PointCloudFrame, ...] = ()
    # Raw camera RGB frames on the same causal control grid as state and, when
    # requested jointly, the point-cloud history.
    rgb_history: tuple[RgbFrame, ...] = ()


def _requested_observation_fields(policy_spec: Any) -> set[str]:
    """Return source names directly from the validated ordered Policy fields."""
    return {field.name for field in policy_spec.observation_fields}


def build_fingertip_runtime(
    policy_spec: Any,
    fingertip_config: FingertipAssemblerConfig | None,
) -> tuple[object, HandKinematics | None, FingertipAssemblerConfig | None] | None:
    """Construct the local FK resources only when the observation requests them.

    ``eef_pose`` needs just the cached canonical ``ArmFK``; the hand FK and
    mount config are built only for ``fingertip_points``.  Both derive from
    the aligned arm qpos history, never from a second realtime EEF source.
    """
    requested = _requested_observation_fields(policy_spec)
    needs_hand_fk = "fingertip_points" in requested
    if not needs_hand_fk and "eef_pose" not in requested:
        return None
    if not needs_hand_fk:
        return make_arm_fk(), None, None
    if not isinstance(fingertip_config, FingertipAssemblerConfig):
        raise TypeError("fingertip_points requires FingertipAssemblerConfig")
    hand_fk = HandKinematics(
        fingertip_config.hand_urdf_path,
        list(fingertip_config.fingertip_link_names),
    )
    if not hand_fk.is_ready():
        raise RuntimeError("fingertip FK startup failed")
    return make_arm_fk(), hand_fk, fingertip_config


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
        if not (max(1, int(not_before_ns)) <= source_ns <= publish_ns <= anchor_ns):
            continue
        if max_age_ns is not None and anchor_ns - source_ns > max_age_ns:
            continue
        value = np.asarray(data[values_field][0], dtype=np.float64)
        if not np.all(np.isfinite(value)):
            continue
        values.append(value)
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
    """Read only tactile provenance flags; never copy the full tactile tensor.

    Contact-only policies gate on calibration, unit identity, causality, and
    age — not on dense-frame freshness, which aggregate ``tactile_sum``
    validity already covers through ``hand_state_ring``.
    """
    try:
        history = ring.get_last_k_fields(
            min(int(history_len), ring.maxlen),
            fields=("source_monotonic_ns", "calibrated", "unit_code"),
        )
    except Exception:
        logger.warning("inference: tactile provenance read failed", exc_info=True)
        return None
    values, sequences, sources, publishes = [], [], [], []
    for record, ring_publish_ns, sequence in history:
        source_ns = int(record["source_monotonic_ns"])
        publish_ns = int(ring_publish_ns)
        if not (
            bool(record["calibrated"])
            and int(record["unit_code"]) == TACTILE_UNIT_CODE_XHAND_SDK_NATIVE
        ):
            continue
        if not (max(1, not_before_ns) <= source_ns <= publish_ns <= anchor_ns):
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


def _read_tactile_force_history(
    ring,
    *,
    anchor_ns: int,
    history_len: int,
    max_age_ns: int,
    not_before_ns: int,
) -> FrameWindow | None:
    """Read the full ``[T,5,120,3]`` tactile history behind strict dense gates.

    Called only when the PolicySpec requests ``tactile_force``; the contact-only
    path keeps using ``_read_tactile_provenance_history`` and never copies the
    full tensor.  Unlike that provenance reader, this path additionally
    requires ``fresh`` (dense validity): it gates on fresh, calibrated,
    unit_code == 0, causal to the anchor, and within the age bound.
    """
    try:
        history = ring.get_last_k(min(int(history_len), ring.maxlen))
    except Exception:
        logger.warning("inference: tactile force read failed", exc_info=True)
        return None
    values, sequences, sources, publishes = [], [], [], []
    for data, ring_publish_ns, sequence in history:
        record = data[0]
        source_ns = int(record["source_monotonic_ns"])
        publish_ns = int(ring_publish_ns)
        if not (
            bool(record["fresh"])
            and bool(record["calibrated"])
            and int(record["unit_code"]) == TACTILE_UNIT_CODE_XHAND_SDK_NATIVE
        ):
            continue
        if not (max(1, not_before_ns) <= source_ns <= publish_ns <= anchor_ns):
            continue
        if anchor_ns - source_ns > max_age_ns:
            continue
        value = np.array(record["tactile_force"], dtype=np.float64)
        if not np.all(np.isfinite(value)):
            continue
        values.append(value)
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
) -> RgbFrame | None:
    """Copy one verified, causal raw RGB frame from the camera ring."""
    record = header[0]
    source_ns = int(record["source_monotonic_ns"])
    receive_ns = int(record["receive_monotonic_ns"])
    camera_publish_ns = int(record["publish_monotonic_ns"])
    camera_generation = int(record["camera_generation"])
    if not (
        sequence > 0
        and camera_generation > 0
        and int(record["camera_health"]) == 0
        and 0
        < source_ns
        <= receive_ns
        <= camera_publish_ns
        <= ring_publish_ns
        <= anchor_ns
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
    if (
        rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or min(rgb.shape[:2]) <= 0
    ):
        return None
    return RgbFrame(
        values=rgb,
        source_camera_sequence=sequence,
        source_monotonic_ns=source_ns,
        publish_monotonic_ns=camera_publish_ns,
        camera_generation=camera_generation,
    )


def _read_rgb_history(
    shared: RuntimeChannels,
    *,
    anchor_ns: int,
    max_age_ns: int,
    history_len: int,
    not_before_ns: int,
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
        )
        if frame is None or (
            frame.source_monotonic_ns != pointcloud.source_monotonic_ns
            or frame.camera_generation != pointcloud.camera_generation
        ):
            return ()
        frames.append(frame)
    return tuple(frames)


def _resize_rgb_history(
    frames: tuple[RgbFrame, ...], *, height: int, width: int
) -> tuple[RgbFrame, ...]:
    """Resize selected causal RGB frames into the Policy input shape."""
    return tuple(
        RgbFrame(
            values=resize_rgb(frame.values, height=height, width=width),
            source_camera_sequence=frame.source_camera_sequence,
            source_monotonic_ns=frame.source_monotonic_ns,
            publish_monotonic_ns=frame.publish_monotonic_ns,
            camera_generation=frame.camera_generation,
        )
        for frame in frames
    )


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
    previous_source_ns = 0
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
        if (
            frame.source_camera_sequence <= previous_sequence
            or frame.source_monotonic_ns <= previous_source_ns
        ):
            return (), 0
        selected.append(frame)
        previous_sequence = frame.source_camera_sequence
        previous_source_ns = frame.source_monotonic_ns
    return tuple(selected), logical_step_ns


def _build_observation(
    shared: RuntimeChannels,
    policy: PolicyParams,
    policy_spec: Any,
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
    reference_ns, logical_step_ns = _select_control_grid_reference_ns(
        run_started_ns=run_started_ns,
        anchor_ns=anchor_ns,
        history_len=horizon,
        step_dt_ns=step_dt_ns,
    )
    if logical_step_ns <= 0:
        return None
    max_age_ns = int(policy.max_input_age_s * 1e9)
    max_skew_ns = int(policy.max_observation_skew_s * 1e9)
    max_grid_lag_ns = int(policy.max_grid_lag_s * 1e9)
    history_span_ns = max(0, horizon - 1) * int(step_dt_ns)
    visual_history_max_age_ns = max_age_ns + history_span_ns + max_grid_lag_ns
    state_history_max_age_ns = visual_history_max_age_ns + max_skew_ns
    hand_history: FrameWindow | None = None
    hand_tactile_sum_history: FrameWindow | None = None
    hand_tactile_provenance_history: FrameWindow | None = None
    hand_tactile_force_history: FrameWindow | None = None
    pointcloud: PointCloudFrame | None = None
    pointcloud_history: tuple[PointCloudFrame, ...] = ()
    requested = _requested_observation_fields(policy_spec)
    pointcloud_requested = "point_cloud" in requested
    rgb_requested = "rgb" in requested
    rgb_history: tuple[RgbFrame, ...] = ()
    fields = {field.name: field for field in policy_spec.observation_fields}
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
    tactile_force_requested = "tactile_force" in requested
    if pointcloud_requested:
        all_pointclouds = _read_pointcloud_history(
            shared,
            anchor_ns=anchor_ns,
            max_age_ns=visual_history_max_age_ns,
            num_points=int(fields["point_cloud"].shape[0]),
            history_len=shared.pointcloud_ring.maxlen,
            not_before_ns=run_started_ns,
        )
        pointcloud_history, _ = _select_camera_control_grid(
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
                )
    elif rgb_requested:
        assert rgb_shape is not None
        all_rgb = _read_rgb_history(
            shared,
            anchor_ns=anchor_ns,
            max_age_ns=visual_history_max_age_ns,
            history_len=shared.camera_ring.maxlen,
            not_before_ns=run_started_ns,
        )
        selected_rgb, _ = _select_camera_control_grid(
            all_rgb,
            run_started_ns=run_started_ns,
            anchor_ns=anchor_ns,
            history_len=horizon,
            step_dt_ns=step_dt_ns,
            max_grid_lag_ns=max_grid_lag_ns,
        )
        rgb_history = selected_rgb
    if rgb_requested and len(rgb_history) == horizon:
        assert rgb_shape is not None
        rgb_history = _resize_rgb_history(
            rgb_history,
            height=int(rgb_shape[0]),
            width=int(rgb_shape[1]),
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
        if tactile_requested and not tactile_force_requested:
            hand_tactile_provenance_history = _read_tactile_provenance_history(
                shared.hand_tactile_ring,
                history_len=shared.hand_tactile_ring.maxlen,
                anchor_ns=anchor_ns,
                max_age_ns=state_history_max_age_ns,
                not_before_ns=run_started_ns,
            )
        if tactile_force_requested:
            hand_tactile_force_history = _read_tactile_force_history(
                shared.hand_tactile_ring,
                history_len=shared.hand_tactile_ring.maxlen,
                anchor_ns=anchor_ns,
                max_age_ns=state_history_max_age_ns,
                not_before_ns=run_started_ns,
            )
    # Every modality uses the policy control grid; camera exposure times do
    # not move the robot/contact observation back to an earlier raw instant.
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
    if hand_tactile_force_history is not None:
        hand_tactile_force_history = _align_state_history_to_reference_ns(
            hand_tactile_force_history,
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
                hand_tactile_force_history,
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
        provenance_history = (
            hand_tactile_force_history
            if tactile_force_requested
            else hand_tactile_provenance_history
        )
        if hand_tactile_sum_history is None or provenance_history is None:
            return None
        if not np.array_equal(
            hand_tactile_sum_history.source_monotonic_ns,
            provenance_history.source_monotonic_ns,
        ):
            return None
    if tactile_force_requested:
        if (
            hand_tactile_force_history is None
            or hand_tactile_force_history.values.shape[0] != horizon
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
        hand_tactile_force_history=hand_tactile_force_history,
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
        getattr(observation, "hand_tactile_force_history", None),
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
    policy_spec: Any,
    *,
    fingertip_runtime: (
        tuple[object, HandKinematics | None, FingertipAssemblerConfig | None] | None
    ) = None,
) -> PolicyObservation:
    """Project typed ring readers into the exact public Policy array mapping."""
    field_names = tuple(field.name for field in policy_spec.observation_fields)
    if observation.arm_history is None or observation.hand_history is None:
        raise ValueError("joint_state requires aligned arm and hand histories")
    arrays: dict[str, np.ndarray] = {}
    joint = np.concatenate(
        (observation.arm_history.values, observation.hand_history.values), axis=1
    )
    arrays["joint_state"] = np.ascontiguousarray(joint, dtype=np.float32)
    policy_joint_state = arrays["joint_state"]
    arm_qpos_policy = policy_joint_state[:, :7]
    hand_qpos_policy = policy_joint_state[:, 7:19]
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
        if observation.hand_tactile_sum_history is None or (
            observation.hand_tactile_force_history is None
            and observation.hand_tactile_provenance_history is None
        ):
            raise ValueError("contact_force lacks calibrated tactile provenance")
        arrays["contact_force"] = np.ascontiguousarray(
            observation.hand_tactile_sum_history.values, dtype=np.float32
        )
    if "tactile_force" in field_names:
        if observation.hand_tactile_force_history is None:
            raise ValueError("tactile_force lacks calibrated full-tactile history")
        arrays["tactile_force"] = np.ascontiguousarray(
            observation.hand_tactile_force_history.values, dtype=np.float32
        )
    if "eef_pose" in field_names or "fingertip_points" in field_names:
        if fingertip_runtime is None:
            raise RuntimeError("eef_pose/fingertip_points require local FK")
        arm_fk, hand_fk, config = fingertip_runtime
        eef_pose_history: np.ndarray | None = None
        if "eef_pose" in field_names:
            # One canonical Arm FK per aligned timestep; the same history is
            # reused below so fingertip_points never repeats the FK.
            eef_pose_history = compute_eef_pose_history_xarm_base(
                arm_qpos_policy, arm_fk=arm_fk
            )
            arrays["eef_pose"] = np.ascontiguousarray(
                eef_pose_history, dtype=np.float32
            )
        if "fingertip_points" in field_names:
            if hand_fk is None or config is None:
                raise RuntimeError("fingertip_points requires local hand FK")
            arrays["fingertip_points"] = np.ascontiguousarray(
                compute_fingertip_history_xarm_base(
                    arm_qpos_policy,
                    hand_qpos_policy,
                    hand_fk=hand_fk,
                    handbase_position_eef_m=np.asarray(config.handbase_position_eef_m),
                    handbase_quat_eef_wxyz=np.asarray(config.handbase_quat_eef_wxyz),
                    arm_fk=arm_fk,
                    eef_pose_history=eef_pose_history,
                ),
                dtype=np.float32,
            )
    ordered = {name: arrays[name] for name in field_names}
    for name, values in ordered.items():
        _validate_finite(values, name=f"PolicyObservation.{name}")
    return PolicyObservation(
        observation_id=observation.observation_id,
        run_generation=observation.run_generation,
        anchor_monotonic_ns=observation.anchor_monotonic_ns,
        latest_source_monotonic_ns=observation.latest_source_monotonic_ns,
        logical_step_monotonic_ns=observation.logical_step_monotonic_ns,
        arrays=ordered,
    )

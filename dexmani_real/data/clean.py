"""Pure one-source-to-one-artifact cleaning decisions for supported episodes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.data.contracts import (
    EpisodeAnnotation,
    EpisodeDecision,
    ProcessingConfig,
    build_source_segment_ends,
)
from dexmani_real.data.quality import assess_temporal_quality
from dexmani_real.recording.reader import EpisodeReader
from dexmani_real.recording.timeline import FillReason
from dexmani_real.sensor.camera_worker import CameraHealth

_FRAME_IK_FAIL = 2
_MAX_TRANSIENT_IK_HOLD_FRAMES = 4


def _as_bool(reader: EpisodeReader, name: str) -> np.ndarray:
    return np.asarray(reader.h5f[name][:], dtype=bool)


def _as_i64(reader: EpisodeReader, name: str) -> np.ndarray:
    return np.asarray(reader.h5f[name][:], dtype=np.int64)


def _as_f64(reader: EpisodeReader, name: str) -> np.ndarray:
    return np.asarray(reader.h5f[name][:], dtype=np.float64)


def align_tactile_sum_rows_to_references(
    contact_force: np.ndarray,
    hand_source_monotonic_ns: np.ndarray,
    tactile_source_monotonic_ns: np.ndarray,
    tactile_fresh: np.ndarray,
    tactile_calibrated: np.ndarray,
    tactile_unit_code: np.ndarray,
    reference_monotonic_ns: np.ndarray,
    *,
    max_observation_skew_s: float,
) -> np.ndarray:
    """Select the newest proven tactile-sum row causal to each reference.

    Candidate rows are restricted to rows already recorded at the target row,
    so a later persisted row can never repair an earlier observation. ``-1``
    marks references without a fresh, calibrated, unit-proven sample in skew.
    """
    contact = np.asarray(contact_force)
    hand_source_ns = np.asarray(hand_source_monotonic_ns, dtype=np.int64)
    source_ns = np.asarray(tactile_source_monotonic_ns, dtype=np.int64)
    fresh = np.asarray(tactile_fresh, dtype=bool)
    calibrated = np.asarray(tactile_calibrated, dtype=bool)
    unit_code = np.asarray(tactile_unit_code, dtype=np.int64)
    references = np.asarray(reference_monotonic_ns, dtype=np.int64)
    count = len(source_ns)
    if contact.shape != (count, 5, 3):
        raise ValueError("contact_force must have shape (frame_count, 5, 3)")
    if any(
        value.shape != (count,)
        for value in (
            hand_source_ns,
            source_ns,
            fresh,
            calibrated,
            unit_code,
            references,
        )
    ):
        raise ValueError("tactile provenance arrays must match frame_count")
    if not np.isfinite(max_observation_skew_s) or max_observation_skew_s <= 0.0:
        raise ValueError("max_observation_skew_s must be finite and positive")
    max_skew_ns = int(round(float(max_observation_skew_s) * 1e9))
    proven = (
        fresh
        & calibrated
        & (unit_code == 0)
        & (source_ns > 0)
        & (hand_source_ns == source_ns)
        & np.all(np.isfinite(contact), axis=(1, 2))
    )
    # Source clocks can reset or arrive out of order in malformed/partial raw
    # captures. Coordinate compression plus a Fenwick occupancy tree keeps the
    # prefix restriction exact without assuming monotonic input or scanning an
    # ever-growing prefix for every output row.
    source_coordinates = np.unique(source_ns[proven])
    coordinate_count = len(source_coordinates)
    occupied = np.zeros(coordinate_count, dtype=bool)
    latest_row = np.full(coordinate_count, -1, dtype=np.int64)
    occupancy_tree = np.zeros(coordinate_count + 1, dtype=np.int64)

    def _add_coordinate(coordinate: int) -> None:
        tree_index = coordinate + 1
        while tree_index <= coordinate_count:
            occupancy_tree[tree_index] += 1
            tree_index += tree_index & -tree_index

    def _prefix_count(end: int) -> int:
        result = 0
        tree_index = end
        while tree_index:
            result += int(occupancy_tree[tree_index])
            tree_index -= tree_index & -tree_index
        return result

    def _coordinate_for_rank(rank: int) -> int:
        coordinate = 0
        accumulated = 0
        step = 1 << (coordinate_count.bit_length() - 1)
        while step:
            candidate = coordinate + step
            if (
                candidate <= coordinate_count
                and accumulated + int(occupancy_tree[candidate]) < rank
            ):
                coordinate = candidate
                accumulated += int(occupancy_tree[candidate])
            step >>= 1
        return coordinate

    selected = np.full(count, -1, dtype=np.int64)
    for target_row, reference_ns in enumerate(references):
        if proven[target_row]:
            coordinate = int(np.searchsorted(source_coordinates, source_ns[target_row]))
            latest_row[coordinate] = target_row
            if not occupied[coordinate]:
                occupied[coordinate] = True
                _add_coordinate(coordinate)
        if reference_ns <= 0:
            continue
        upper_bound = int(
            np.searchsorted(source_coordinates, reference_ns, side="right")
        )
        available_count = _prefix_count(upper_bound)
        if available_count == 0:
            continue
        coordinate = _coordinate_for_rank(available_count)
        if reference_ns - source_coordinates[coordinate] <= max_skew_ns:
            selected[target_row] = latest_row[coordinate]
    return selected


def recompute_observation_skew_s(
    source_timestamps_ns: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Recompute aggregate source skew from raw timestamps and validity masks.

    The four source columns are arm, hand, VR, and camera.  A non-positive
    timestamp is the raw invalid-source sentinel and is ignored when its mask
    is false; rows with no valid source use the producer's legal ``0.0`` skew
    sentinel.  A finite timestamp under a true mask is required so malformed
    source metadata cannot pass visual cleaning.
    """
    timestamps = np.asarray(source_timestamps_ns)
    valid = np.asarray(valid_mask, dtype=bool)
    if timestamps.ndim != 2 or timestamps.shape[1] != 4:
        raise ValueError("source_timestamps_ns must have shape (frame_count, 4)")
    if valid.shape != timestamps.shape:
        raise ValueError("valid_mask must have the same shape as source_timestamps_ns")
    if not np.issubdtype(timestamps.dtype, np.number):
        raise ValueError("source_timestamps_ns must be numeric")
    finite = np.isfinite(timestamps)
    if np.any(valid & ~finite):
        raise ValueError("valid source timestamps must be finite")

    effective = valid & finite & (timestamps > 0)
    expected = np.zeros(timestamps.shape[0], dtype=np.float64)
    for row in np.flatnonzero(np.any(effective, axis=1)):
        source_times = timestamps[row, effective[row]]
        delta_ns: int | float
        if np.issubdtype(timestamps.dtype, np.integer):
            delta_ns = int(np.max(source_times)) - int(np.min(source_times))
        else:
            delta_ns = float(np.max(source_times)) - float(np.min(source_times))
        expected[row] = delta_ns / 1e9
    return expected


def observation_skew_valid_mask(
    recorded_observation_skew_s: np.ndarray,
    source_timestamps_ns: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_observation_skew_s: float,
) -> np.ndarray:
    """Validate visual aggregate skew against its raw source provenance.

    The aggregate is a deployment admission value: rows with the required
    arm/VR/camera sources must be finite, non-negative, bounded, and equal to
    the recomputed source span.  A row with no effective source may retain the
    producer's historical ``0.0`` or explicit ``NaN`` sentinel, but it remains
    invalid because the required-source mask is false.
    """
    recorded = np.asarray(recorded_observation_skew_s, dtype=np.float64)
    timestamps = np.asarray(source_timestamps_ns)
    valid = np.asarray(valid_mask, dtype=bool)
    if recorded.ndim != 1:
        raise ValueError("recorded_observation_skew_s must be 1-D")
    if timestamps.ndim != 2 or timestamps.shape[0] != recorded.shape[0]:
        raise ValueError("source_timestamps_ns must have shape (len(recorded), 4)")
    if valid.shape != timestamps.shape:
        raise ValueError("valid_mask must have the same shape as source_timestamps_ns")
    if not np.isfinite(max_observation_skew_s) or max_observation_skew_s <= 0.0:
        raise ValueError("max_observation_skew_s must be finite and positive")

    expected = recompute_observation_skew_s(timestamps, valid)
    finite_timestamps = np.isfinite(timestamps)
    source_metadata_valid = np.all(
        (~valid) | (finite_timestamps & (timestamps > 0.0)), axis=1
    )
    required_sources_valid = np.all(valid[:, [0, 2, 3]], axis=1)
    effective_sources = valid & finite_timestamps & (timestamps > 0.0)
    has_effective_source = np.any(effective_sources, axis=1)
    aggregate_finite_and_bounded = (
        np.isfinite(recorded) & (recorded >= 0.0) & (recorded <= max_observation_skew_s)
    )
    aggregate_consistent = aggregate_finite_and_bounded & np.isclose(
        recorded, expected, rtol=0.0, atol=1e-7
    )
    aggregate_consistent |= ~has_effective_source & (
        np.isnan(recorded) | np.isclose(recorded, 0.0, rtol=0.0, atol=1e-15)
    )
    return (
        required_sources_valid
        & source_metadata_valid
        & np.isfinite(expected)
        & aggregate_consistent
    )


def _finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(finite.size),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _range_mask(
    length: int, ranges: tuple[tuple[int, int], ...], *, default: bool
) -> np.ndarray:
    mask = np.full(length, default, dtype=bool)
    if ranges and default:
        for start, end in ranges:
            mask[max(0, start) : min(length, end)] = False
    elif ranges:
        for start, end in ranges:
            mask[max(0, start) : min(length, end)] = True
    return mask


def _true_ranges(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return half-open ranges for contiguous true runs."""

    values = np.asarray(mask, dtype=bool)
    padded = np.pad(values.astype(np.int8), (1, 1))
    edges = np.diff(padded)
    return tuple(
        (int(start), int(end))
        for start, end in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
        )
    )


def _transient_ik_hold_masks(
    held: np.ndarray, frame_status: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split IK fallback holds into brief pauses and persistent failures."""

    ik_hold = np.asarray(held, dtype=bool) & (
        np.asarray(frame_status, dtype=np.int64) == _FRAME_IK_FAIL
    )
    transient = np.zeros(ik_hold.shape, dtype=bool)
    persistent = np.zeros(ik_hold.shape, dtype=bool)
    for start, end in _true_ranges(ik_hold):
        target = (
            transient if end - start <= _MAX_TRANSIENT_IK_HOLD_FRAMES else persistent
        )
        target[start:end] = True
    return transient, persistent


def _empty_decision(
    path: Path,
    frame_count: int,
    config: ProcessingConfig,
    reason: str,
    *,
    hard_reason_counts: dict[str, int] | None = None,
) -> EpisodeDecision:
    return EpisodeDecision(
        source_path=path,
        source_frames=frame_count,
        profile=config.profile,
        selected_indices=np.empty(0, dtype=np.int64),
        keep_mask=np.zeros(frame_count, dtype=bool),
        drop_reason_bits=np.zeros(frame_count, dtype=np.uint64),
        drop_reason_names=(),
        hard_reason_counts=hard_reason_counts or {},
        boundary_counts={},
        selected_frames=0,
        quality={},
        rejected_reason=reason,
    )


def _inside(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    return np.all((values >= lower - tolerance) & (values <= upper + tolerance), axis=1)


def _rows_are_source_contiguous(
    arrays: Mapping[str, np.ndarray],
    previous_index: int,
    index: int,
    *,
    grid_dt_s: float,
    grid_dt_relative_tolerance: float,
) -> bool:
    """Return whether two retained rows can share one policy action stream."""
    if index != previous_index + 1:
        return False
    if (
        int(arrays["source_index"][index])
        != int(arrays["source_index"][previous_index]) + 1
    ):
        return False
    tolerance_s = max(1e-7, grid_dt_s * grid_dt_relative_tolerance)
    return bool(
        abs(
            float(arrays["timestamp"][index])
            - float(arrays["timestamp"][previous_index])
            - grid_dt_s
        )
        <= tolerance_s
    )


def _revalidate_camera_duplicates(
    arrays: Mapping[str, np.ndarray],
    camera_nominal: np.ndarray,
    camera_age_valid: np.ndarray,
    *,
    grid_dt_s: float,
    grid_dt_relative_tolerance: float,
) -> np.ndarray:
    """Recover false duplicate flags only from a trusted advancing predecessor."""

    revalidated = np.zeros(camera_nominal.shape, dtype=bool)
    trusted = np.asarray(camera_nominal, dtype=bool).copy()
    clock_reset = np.asarray(arrays["camera_clock_reset"], dtype=bool)
    for index in np.flatnonzero(arrays["camera_duplicate"]):
        if index == 0 or not trusted[index - 1]:
            continue
        previous = index - 1
        source_contiguous = _rows_are_source_contiguous(
            arrays,
            previous,
            int(index),
            grid_dt_s=grid_dt_s,
            grid_dt_relative_tolerance=grid_dt_relative_tolerance,
        )
        revalidated[index] = bool(
            source_contiguous
            and arrays["camera_generation"][index] > 0
            and arrays["camera_generation"][index]
            == arrays["camera_generation"][previous]
            and arrays["camera_depth_frame_number"][index]
            > arrays["camera_depth_frame_number"][previous]
            and arrays["camera_color_frame_number"][index]
            > arrays["camera_color_frame_number"][previous]
            and arrays["camera_source_monotonic_ns"][index]
            > arrays["camera_source_monotonic_ns"][previous]
            and np.isfinite(arrays["camera_depth_device_timestamp_s"][index])
            and np.isfinite(arrays["camera_color_device_timestamp_s"][index])
            and arrays["camera_depth_device_timestamp_s"][index]
            > arrays["camera_depth_device_timestamp_s"][previous]
            and arrays["camera_color_device_timestamp_s"][index]
            > arrays["camera_color_device_timestamp_s"][previous]
            and not clock_reset[index]
            and camera_age_valid[index]
        )
        trusted[index] = revalidated[index]
    return revalidated


def _deployment_action_limit_masks(
    arrays: Mapping[str, np.ndarray],
    candidate_mask: np.ndarray,
    config: ProcessingConfig,
    *,
    grid_dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Check the exact endpoint-delta contract enforced by deployment.

    At the start of every retained source-contiguous segment the policy executor
    compares a model endpoint against measured feedback. Later endpoints are
    compared against the previously published endpoint.  This mirrors that
    distinction so old teleop targets are never silently relabelled as
    deployment-safe data.
    """
    candidate = np.asarray(candidate_mask, dtype=bool)
    action_invalid = np.zeros(candidate.shape, dtype=bool)
    arm_invalid = np.zeros(candidate.shape, dtype=bool)
    hand_invalid = np.zeros(candidate.shape, dtype=bool)
    previous_index: int | None = None
    arm_limit = config.arm_max_delta_rad_per_tick
    hand_limit = float(config.hand_max_delta_rad_per_tick)
    endpoint_delta_tolerance = float(config.endpoint_delta_tolerance_rad)

    for index, candidate_row in enumerate(candidate):
        if not candidate_row:
            previous_index = None
            continue
        starts_segment = previous_index is None or not _rows_are_source_contiguous(
            arrays,
            previous_index,
            index,
            grid_dt_s=grid_dt_s,
            grid_dt_relative_tolerance=config.grid_dt_relative_tolerance,
        )
        if starts_segment:
            arm_reference = arrays["control_arm_qpos"][index]
            hand_reference = arrays["control_hand_qpos"][index]
        else:
            assert previous_index is not None
            arm_reference = arrays["action_arm"][previous_index]
            hand_reference = arrays["action_hand"][previous_index]

        if arm_limit is not None and np.any(
            np.abs(arrays["action_arm"][index] - arm_reference)
            > arm_limit + endpoint_delta_tolerance
        ):
            arm_invalid[index] = True
        if np.any(
            np.abs(arrays["action_hand"][index] - hand_reference)
            > hand_limit + endpoint_delta_tolerance
        ):
            hand_invalid[index] = True
        action_invalid[index] = arm_invalid[index] or hand_invalid[index]
        # An excluded endpoint starts a new candidate segment.  Its successor
        # must be safe from contemporaneous feedback, not from an action that
        # will not exist in the exported stream.
        previous_index = None if action_invalid[index] else index
    return action_invalid, arm_invalid, hand_invalid


def _quality_summary(
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
    *,
    grid_dt_s: float,
    tracking_error_warn_rad: float,
    horizon: int,
    segment_ends: np.ndarray,
) -> dict[str, Any]:
    action = np.concatenate((arrays["action_arm"], arrays["action_hand"]), axis=1)[
        selected
    ]
    segment_starts = (
        np.concatenate((np.asarray([0]), segment_ends[:-1]))
        if len(segment_ends)
        else np.empty(0, dtype=np.int64)
    )
    step_blocks = [
        np.diff(action[start:end], axis=0)
        for start, end in zip(segment_starts, segment_ends, strict=True)
        if end - start >= 2
    ]
    steps = (
        np.concatenate(step_blocks, axis=0)
        if step_blocks
        else np.empty((0, action.shape[1]), dtype=action.dtype)
    )
    max_step = (
        np.max(np.abs(steps), axis=1) if steps.size else np.empty(0, dtype=np.float64)
    )
    acceleration_blocks = [
        np.diff(np.diff(action[start:end], axis=0) / grid_dt_s, axis=0) / grid_dt_s
        for start, end in zip(segment_starts, segment_ends, strict=True)
        if end - start >= 3
    ]
    acceleration = (
        np.concatenate(acceleration_blocks, axis=0)
        if acceleration_blocks
        else np.empty((0, 19), dtype=np.float64)
    )
    summary: dict[str, Any] = {
        "tracking_error_rad": _finite_stats(arrays["tracking_error"][selected]),
        "high_tracking_error_count": int(
            np.count_nonzero(
                arrays["tracking_error"][selected] > tracking_error_warn_rad
            )
        ),
        "max_abs_action_step_rad": _finite_stats(max_step),
        "max_abs_action_velocity_rad_s": _finite_stats(max_step / grid_dt_s),
        "max_abs_action_acceleration_rad_s2": _finite_stats(
            np.max(np.abs(acceleration), axis=1) if acceleration.size else np.empty(0)
        ),
        "idle_step_ratio": float(np.mean(max_step <= 1e-4)) if max_step.size else 1.0,
        "full_window_count": int(
            sum(
                max(0, int(end - start) - horizon + 1)
                for start, end in zip(segment_starts, segment_ends, strict=True)
            )
        ),
    }
    if "camera_age_s" in arrays:
        summary["camera_age_s"] = _finite_stats(arrays["camera_age_s"][selected])
    if "pointcloud_valid_depth_ratio" in arrays:
        summary["pointcloud_valid_depth_ratio"] = _finite_stats(
            arrays["pointcloud_valid_depth_ratio"][selected]
        )
    return summary


def _source_gap_findings(
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
    reason_masks: Mapping[str, np.ndarray],
    temporal_excluded: np.ndarray,
    config: ProcessingConfig,
    *,
    grid_dt_s: float,
) -> tuple[dict[str, Any], ...]:
    """Audit every adjacency that differs from the original one-step grid.

    Every finding becomes an episode boundary during policy export.
    """

    findings: list[dict[str, Any]] = []
    if len(selected) < 2:
        return ()
    for left, right in zip(selected[:-1], selected[1:], strict=True):
        timestamp_delta_s = float(
            arrays["timestamp"][right] - arrays["timestamp"][left]
        )
        sample_delta = int(arrays["source_index"][right] - arrays["source_index"][left])
        row_delta = int(right - left)
        has_boundary = (
            row_delta != 1
            or sample_delta != 1
            or abs(timestamp_delta_s - grid_dt_s)
            > max(1e-7, grid_dt_s * config.grid_dt_relative_tolerance)
        )
        if not has_boundary:
            continue
        arm_delta = float(
            np.max(np.abs(arrays["action_arm"][right] - arrays["action_arm"][left]))
        )
        hand_delta = float(
            np.max(np.abs(arrays["action_hand"][right] - arrays["action_hand"][left]))
        )
        removed_slice = slice(int(left + 1), int(right))
        removed_reasons = [
            name
            for name, mask in reason_masks.items()
            if right > left + 1 and np.any(mask[removed_slice])
        ]
        if right > left + 1 and np.any(temporal_excluded[removed_slice]):
            removed_reasons.append("temporal_high_confidence")
        risky = bool(
            sample_delta != row_delta
            or abs(timestamp_delta_s - row_delta * grid_dt_s)
            > max(1e-7, grid_dt_s * config.grid_dt_relative_tolerance)
            or arm_delta > config.temporal_quality.abrupt_arm_step_rad
            or hand_delta > config.temporal_quality.abrupt_hand_step_rad
        )
        findings.append(
            {
                "source_row_before": int(left),
                "source_row_after": int(right),
                "removed_frame_count": max(0, row_delta - 1),
                "source_sample_delta": sample_delta,
                "source_timestamp_delta_s": timestamp_delta_s,
                "max_arm_action_delta_rad": arm_delta,
                "max_hand_action_delta_rad": hand_delta,
                "removed_reasons": sorted(set(removed_reasons)),
                "abrupt_or_irregular": risky,
            }
        )
    return tuple(findings)


def _reason_bits(
    reason_masks: Mapping[str, np.ndarray], temporal_excluded: np.ndarray
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode all source-row deletion reasons; kept rows remain exactly zero."""

    names = tuple(reason_masks) + ("temporal_high_confidence",)
    if len(names) > 64:
        raise ValueError("drop-reason contract exceeds uint64 capacity")
    length = len(temporal_excluded)
    bits = np.zeros(length, dtype=np.uint64)
    for bit, name in enumerate(names[:-1]):
        bits[np.asarray(reason_masks[name], dtype=bool)] |= np.uint64(1) << np.uint64(
            bit
        )
    bits[np.asarray(temporal_excluded, dtype=bool)] |= np.uint64(1) << np.uint64(
        len(names) - 1
    )
    return bits, names


def analyze_episode(
    reader: EpisodeReader,
    config: ProcessingConfig,
    annotation: EpisodeAnnotation | None = None,
    *,
    depth_valid_mask: np.ndarray | None = None,
    source_already_validated: bool = False,
) -> EpisodeDecision:
    """Return a deterministic compact-row decision without writing output."""

    if not source_already_validated:
        reader.require_valid(purpose="offline processing")
    annotation = annotation or EpisodeAnnotation()
    frame_count = int(reader.h5f["meta"].attrs["num_frames"])
    if not annotation.include:
        return _empty_decision(
            reader.h5_path,
            frame_count,
            config,
            "excluded by annotation",
            hard_reason_counts={"annotation_excluded_episode": frame_count},
        )
    if "action_arm_joint_sent" not in reader.h5f:
        return _empty_decision(
            reader.h5_path,
            frame_count,
            config,
            "action_arm_joint_sent is required; unsafe fallback is disabled",
            hard_reason_counts={"missing_arm_sent_stream": frame_count},
        )
    for label, ranges in (
        ("include_ranges", annotation.include_ranges),
        ("exclude_ranges", annotation.exclude_ranges),
    ):
        if any(end > frame_count for _, end in ranges):
            raise ValueError(f"{label} exceeds source frame count {frame_count}")

    arrays: dict[str, np.ndarray] = {
        "timestamp": _as_f64(reader, "timestamp"),
        "source_index": _as_i64(reader, "source_sample_index"),
        "fill_reason": _as_i64(reader, "fill_reason"),
        "sample_valid": _as_bool(reader, "flag_sample_valid"),
        "queued": _as_bool(reader, "flag_action_queued"),
        "held": _as_bool(reader, "flag_held"),
        "safety_reject": _as_bool(reader, "flag_safety_reject"),
        "frame_status": _as_i64(reader, "flag_frame_status"),
        "arm_connected": _as_bool(reader, "arm_connected"),
        "hand_connected": _as_bool(reader, "hand_connected"),
        "hand_stale": _as_bool(reader, "hand_qpos_stale"),
        "history_valid": np.asarray(
            reader.h5f["observation_history_valid_mask"][:], dtype=bool
        )[:, :, 0],
        "action_created": _as_i64(reader, "action_created_monotonic_ns"),
        "action_target": _as_i64(reader, "action_target_monotonic_ns"),
        "action_valid_until": _as_i64(reader, "action_valid_until_monotonic_ns"),
        # The first action in a retained segment is safety-gated against this
        # control-grid feedback, while visual policy state is camera-aligned.
        "control_arm_qpos": _as_f64(reader, "arm_qpos"),
        "control_hand_qpos": _as_f64(reader, "hand_qpos"),
        "action_arm": _as_f64(reader, "action_arm_joint_sent"),
        "action_hand": _as_f64(reader, "action_hand_joint"),
        "action_arm_ee": _as_f64(reader, "action_arm_ee"),
        "contact_force": _as_f64(reader, "hand_contact"),
        "fingertip_points": _as_f64(reader, "hand_fingertip"),
        "tracking_error": _as_f64(reader, "tracking_error"),
        "arm_last_cmd_seq": _as_i64(reader, "arm_last_cmd_seq"),
        "observation_anchor_monotonic_ns": _as_i64(
            reader, "observation_anchor_monotonic_ns"
        ),
        "arm_source_monotonic_ns": _as_i64(reader, "arm_source_monotonic_ns"),
        "hand_source_monotonic_ns": _as_i64(reader, "hand_source_monotonic_ns"),
    }
    visual_profile = config.profile.needs_rgb or config.profile.needs_pointcloud
    if visual_profile:
        # Raw v24 stores the state that a visual policy actually observes:
        # newest arm/hand feedback whose source time does not exceed the camera
        # source time.
        arrays["arm_qpos"] = _as_f64(reader, "policy_observation_arm_qpos")
        arrays["hand_qpos"] = _as_f64(reader, "policy_observation_hand_qpos")
        arrays["observation_source_monotonic_ns"] = np.column_stack(
            [
                _as_i64(reader, f"{name}_source_monotonic_ns")
                for name in ("arm", "hand", "vr", "camera")
            ]
        )
        arrays["camera_source_monotonic_ns"] = _as_i64(
            reader, "camera_source_monotonic_ns"
        )
    else:
        arrays["arm_qpos"] = arrays["control_arm_qpos"]
        arrays["hand_qpos"] = arrays["control_hand_qpos"]
    is_source = arrays["fill_reason"] == int(FillReason.SOURCE)
    timing_valid = (
        (arrays["action_created"] > 0)
        & (arrays["action_created"] <= arrays["action_target"])
        & (arrays["action_target"] <= arrays["action_valid_until"])
    )
    joint_numeric = np.concatenate(
        (
            arrays["control_arm_qpos"],
            arrays["control_hand_qpos"],
            arrays["arm_qpos"],
            arrays["hand_qpos"],
            arrays["action_arm"],
            arrays["action_hand"],
        ),
        axis=1,
    )
    tactile_reference_ns = (
        arrays["camera_source_monotonic_ns"]
        if visual_profile
        else arrays["observation_anchor_monotonic_ns"]
    )
    tactile_source_rows = align_tactile_sum_rows_to_references(
        arrays["contact_force"],
        arrays["hand_source_monotonic_ns"],
        _as_i64(reader, "tactile_source_monotonic_ns"),
        _as_bool(reader, "tactile_fresh"),
        _as_bool(reader, "tactile_calibrated"),
        _as_i64(reader, "tactile_unit_code"),
        tactile_reference_ns,
        max_observation_skew_s=config.max_observation_skew_s,
    )
    tactile_valid = tactile_source_rows >= 0
    tactile_forward_fill = tactile_valid & (
        tactile_source_rows != np.arange(frame_count, dtype=np.int64)
    )
    aligned_contact = np.full_like(arrays["contact_force"], np.nan)
    aligned_contact[tactile_valid] = arrays["contact_force"][
        tactile_source_rows[tactile_valid]
    ]
    arrays["contact_force"] = aligned_contact
    real_modalities_finite = np.all(
        np.isfinite(arrays["action_arm_ee"]), axis=1
    ) & np.all(np.isfinite(arrays["contact_force"]), axis=(1, 2))
    if not visual_profile:
        real_modalities_finite &= np.all(
            np.isfinite(arrays["fingertip_points"]), axis=(1, 2)
        )
    arm_lower = np.asarray(config.arm_joint_limit_lower_rad, dtype=np.float64)
    arm_upper = np.asarray(config.arm_joint_limit_upper_rad, dtype=np.float64)
    hand_state_lower = np.asarray(config.hand_state_limit_lower_rad, dtype=np.float64)
    hand_state_upper = np.asarray(config.hand_state_limit_upper_rad, dtype=np.float64)
    hand_action_lower = np.asarray(config.hand_action_limit_lower_rad, dtype=np.float64)
    hand_action_upper = np.asarray(config.hand_action_limit_upper_rad, dtype=np.float64)
    state_limits_valid = (
        _inside(
            arrays["control_arm_qpos"],
            arm_lower,
            arm_upper,
            config.joint_limit_tolerance_rad,
        )
        & _inside(
            arrays["control_hand_qpos"],
            hand_state_lower,
            hand_state_upper,
            config.hand_state_limit_tolerance_rad,
        )
        & _inside(
            arrays["arm_qpos"],
            arm_lower,
            arm_upper,
            config.joint_limit_tolerance_rad,
        )
        & _inside(
            arrays["hand_qpos"],
            hand_state_lower,
            hand_state_upper,
            config.hand_state_limit_tolerance_rad,
        )
    )
    action_mechanical_limits_valid = _inside(
        arrays["action_arm"], arm_lower, arm_upper, config.joint_limit_tolerance_rad
    ) & _inside(
        arrays["action_hand"],
        hand_action_lower,
        hand_action_upper,
        config.joint_limit_tolerance_rad,
    )
    transient_ik_hold, long_ik_failure_hold = _transient_ik_hold_masks(
        arrays["held"], arrays["frame_status"]
    )
    non_ik_frame_failure = (arrays["frame_status"] != 0) & ~(
        arrays["held"] & (arrays["frame_status"] == _FRAME_IK_FAIL)
    )
    reason_masks: dict[str, np.ndarray] = {
        "not_source_sample": ~(arrays["sample_valid"] & is_source),
        "action_not_queued": ~arrays["queued"],
        "safety_reject": arrays["safety_reject"],
        "frame_status_not_ok": non_ik_frame_failure,
        "long_ik_failure_hold": long_ik_failure_hold,
        "arm_source_invalid": ~(
            arrays["arm_connected"] & arrays["history_valid"][:, 0]
        ),
        "hand_source_invalid": ~(
            arrays["hand_connected"]
            & ~arrays["hand_stale"]
            & arrays["history_valid"][:, 1]
        ),
        "tactile_invalid": ~tactile_valid,
        "action_timing_invalid": ~timing_valid,
        "nonfinite_joint_or_action": ~np.all(np.isfinite(joint_numeric), axis=1),
        "nonfinite_real_modality": ~real_modalities_finite,
        "action_mechanical_limit_violation": ~action_mechanical_limits_valid,
    }
    audit_masks: dict[str, np.ndarray] = {
        "joint_state_limit_excursion": ~state_limits_valid,
        "transient_ik_hold": transient_ik_hold,
    }
    repair_masks: dict[str, np.ndarray] = {
        "tactile_forward_fill": tactile_forward_fill,
    }

    if not visual_profile:
        anchor_ns = arrays["observation_anchor_monotonic_ns"]
        arm_source_ns = arrays["arm_source_monotonic_ns"]
        hand_source_ns = arrays["hand_source_monotonic_ns"]
        max_skew_ns = int(round(config.max_observation_skew_s * 1e9))
        reason_masks["control_grid_observation_invalid"] = ~(
            (anchor_ns > 0)
            & (arm_source_ns > 0)
            & (hand_source_ns > 0)
            & (arm_source_ns <= anchor_ns)
            & (hand_source_ns <= anchor_ns)
            & (anchor_ns - arm_source_ns <= max_skew_ns)
            & (anchor_ns - hand_source_ns <= max_skew_ns)
        )

    if visual_profile:
        arrays.update(
            {
                "camera_age_s": _as_f64(reader, "camera_age_s"),
                "camera_frame_gap": _as_i64(reader, "camera_frame_gap"),
                "camera_duplicate": _as_bool(reader, "camera_duplicate"),
                "camera_health": _as_i64(reader, "camera_health"),
                "observation_valid": _as_bool(reader, "observation_valid"),
                "observation_skew_s": _as_f64(reader, "observation_skew_s"),
                "camera_generation": _as_i64(reader, "camera_generation"),
                "camera_depth_frame_number": _as_i64(
                    reader, "camera_depth_frame_number"
                ),
                "camera_color_frame_number": _as_i64(
                    reader, "camera_color_frame_number"
                ),
                "camera_source_monotonic_ns": arrays["camera_source_monotonic_ns"],
                "camera_clock_reset": _as_bool(reader, "camera_clock_reset"),
                "camera_depth_device_timestamp_s": _as_f64(
                    reader, "camera_depth_device_timestamp_s"
                ),
                "camera_color_device_timestamp_s": _as_f64(
                    reader, "camera_color_device_timestamp_s"
                ),
            }
        )
        camera_age_valid = (
            np.isfinite(arrays["camera_age_s"])
            & (arrays["camera_age_s"] >= 0.0)
            & (arrays["camera_age_s"] <= config.max_camera_age_s)
        )
        camera_clock_reset = arrays["camera_clock_reset"]
        camera_nominal = (
            _as_bool(reader, "flag_camera_fresh")
            & arrays["history_valid"][:, 3]
            & ~camera_clock_reset
            & (arrays["camera_health"] == int(CameraHealth.OK))
            & camera_age_valid
        )
        camera_duplicate_revalidated = _revalidate_camera_duplicates(
            arrays,
            camera_nominal,
            camera_age_valid,
            grid_dt_s=reader.timing.grid_dt_s,
            grid_dt_relative_tolerance=config.grid_dt_relative_tolerance,
        )
        camera_admitted = camera_nominal | camera_duplicate_revalidated
        reason_masks["camera_invalid"] = ~camera_admitted
        audit_masks["camera_duplicate_revalidated"] = camera_duplicate_revalidated
        effective_history_valid = arrays["history_valid"].copy()
        effective_history_valid[camera_duplicate_revalidated, 3] = True
        recorded_observation_skew_valid = observation_skew_valid_mask(
            arrays["observation_skew_s"],
            arrays["observation_source_monotonic_ns"],
            arrays["history_valid"],
            max_observation_skew_s=config.max_observation_skew_s,
        )
        revalidated_skew_s = recompute_observation_skew_s(
            arrays["observation_source_monotonic_ns"], effective_history_valid
        )
        source_metadata_valid = np.all(
            (~effective_history_valid)
            | (arrays["observation_source_monotonic_ns"] > 0),
            axis=1,
        )
        duplicate_observation_revalidated = (
            camera_duplicate_revalidated
            & np.all(effective_history_valid[:, [0, 2, 3]], axis=1)
            & source_metadata_valid
            & np.isfinite(revalidated_skew_s)
            & (revalidated_skew_s <= config.max_observation_skew_s)
        )
        reason_masks["observation_invalid"] = ~(
            (arrays["observation_valid"] & recorded_observation_skew_valid)
            | duplicate_observation_revalidated
        )
        policy_reference_ns = _as_i64(
            reader, "policy_observation_reference_monotonic_ns"
        )
        policy_arm_sequence = _as_i64(reader, "policy_observation_arm_source_sequence")
        policy_hand_sequence = _as_i64(
            reader, "policy_observation_hand_source_sequence"
        )
        policy_arm_source_ns = _as_i64(
            reader, "policy_observation_arm_source_monotonic_ns"
        )
        policy_hand_source_ns = _as_i64(
            reader, "policy_observation_hand_source_monotonic_ns"
        )
        policy_arm_publish_ns = _as_i64(
            reader, "policy_observation_arm_publish_monotonic_ns"
        )
        policy_hand_publish_ns = _as_i64(
            reader, "policy_observation_hand_publish_monotonic_ns"
        )
        policy_anchor_ns = _as_i64(reader, "observation_anchor_monotonic_ns")
        policy_skew_s = _as_f64(reader, "policy_observation_skew_s")
        expected_policy_skew_s = (
            policy_reference_ns
            - np.minimum(policy_arm_source_ns, policy_hand_source_ns)
        ) / 1e9
        policy_observation_valid = (
            _as_bool(reader, "policy_observation_valid")
            & (policy_reference_ns > 0)
            & (policy_reference_ns <= policy_anchor_ns)
            & (policy_reference_ns == _as_i64(reader, "camera_source_monotonic_ns"))
            & (policy_arm_sequence > 0)
            & (policy_hand_sequence > 0)
            & (policy_arm_source_ns > 0)
            & (policy_hand_source_ns > 0)
            & (policy_arm_source_ns <= policy_reference_ns)
            & (policy_hand_source_ns <= policy_reference_ns)
            & (policy_arm_source_ns <= policy_arm_publish_ns)
            & (policy_hand_source_ns <= policy_hand_publish_ns)
            & (policy_arm_publish_ns <= policy_anchor_ns)
            & (policy_hand_publish_ns <= policy_anchor_ns)
            & np.isfinite(policy_skew_s)
            & (policy_skew_s >= 0.0)
            & np.isclose(policy_skew_s, expected_policy_skew_s, rtol=0.0, atol=1e-9)
            & (policy_skew_s <= config.max_observation_skew_s)
        )
        reason_masks["policy_observation_invalid"] = ~policy_observation_valid
    if visual_profile:
        if depth_valid_mask is None:
            raise ValueError("RGB/pointcloud profile requires a depth_valid_mask")
        depth_valid = np.asarray(depth_valid_mask, dtype=bool)
        if depth_valid.shape != (frame_count,):
            raise ValueError(
                f"depth_valid_mask must have shape ({frame_count},), got {depth_valid.shape}"
            )
        reason_masks["depth_invalid"] = ~depth_valid

    include_mask = _range_mask(
        frame_count,
        annotation.include_ranges,
        default=not bool(annotation.include_ranges),
    )
    exclude_mask = _range_mask(frame_count, annotation.exclude_ranges, default=True)
    reason_masks["annotation_excluded_row"] = ~(include_mask & exclude_mask)

    timing = reader.timing
    pre_action_base_valid = np.ones(frame_count, dtype=bool)
    for mask in reason_masks.values():
        pre_action_base_valid &= ~mask

    dt = np.diff(arrays["timestamp"])
    timestamp_gap = np.abs(dt - timing.grid_dt_s) > max(
        1e-7, timing.grid_dt_s * config.grid_dt_relative_tolerance
    )
    source_gap = np.diff(arrays["source_index"]) != 1
    break_before = np.zeros(frame_count, dtype=bool)
    break_before[1:] = timestamp_gap | source_gap
    temporal_assessment = assess_temporal_quality(
        arrays,
        pre_action_base_valid,
        break_before,
        config.temporal_quality,
        tracking_error_warn_rad=config.tracking_error_warn_rad,
    )

    # Deployment endpoint deltas are audit telemetry.  Collection commands have
    # already crossed the live safety boundary; a large offline delta must not
    # relabel a finite, mechanically valid recorded action as corrupt data.
    candidate_mask = pre_action_base_valid & ~temporal_assessment.excluded_mask
    (
        deployment_action_invalid,
        deployment_arm_action_invalid,
        deployment_hand_action_invalid,
    ) = _deployment_action_limit_masks(
        arrays,
        candidate_mask,
        config,
        grid_dt_s=timing.grid_dt_s,
    )
    audit_masks["deployment_action_limit"] = deployment_action_invalid

    hard_invalid = np.zeros(frame_count, dtype=bool)
    for mask in reason_masks.values():
        hard_invalid |= mask
    base_valid = ~hard_invalid
    keep_mask = base_valid & ~temporal_assessment.excluded_mask
    selected = np.flatnonzero(keep_mask).astype(np.int64)
    bits, reason_names = _reason_bits(reason_masks, temporal_assessment.excluded_mask)
    source_gaps = _source_gap_findings(
        arrays,
        selected,
        reason_masks,
        temporal_assessment.excluded_mask,
        config,
        grid_dt_s=timing.grid_dt_s,
    )
    hard_invalid_reason_names = tuple(
        name for name in reason_masks if name != "annotation_excluded_row"
    )
    if np.any(temporal_assessment.excluded_mask):
        hard_invalid_reason_names += ("temporal_high_confidence",)
    warnings: list[str] = []
    if source_gaps:
        warnings.append(
            f"policy export will reject this episode: {len(source_gaps)} source "
            "discontinuity boundary(s)"
        )
    frame_ok = arrays["frame_status"] == 0
    if np.count_nonzero(frame_ok & ~_as_bool(reader, "flag_ik_ok")) > frame_count // 2:
        warnings.append("flag_ik_ok conflicts with mostly-OK frame_status")
    if (
        np.count_nonzero(frame_ok & ~_as_bool(reader, "flag_retarget_ok"))
        > frame_count // 2
    ):
        warnings.append("flag_retarget_ok conflicts with mostly-OK frame_status")

    segment_ends = build_source_segment_ends(selected, source_gaps)
    quality = _quality_summary(
        arrays,
        selected,
        grid_dt_s=timing.grid_dt_s,
        tracking_error_warn_rad=config.tracking_error_warn_rad,
        horizon=config.horizon,
        segment_ends=segment_ends,
    )
    quality["source_gap_count"] = len(source_gaps)
    quality["segment_count"] = len(segment_ends)
    quality["deployment_action_limits"] = {
        "arm_invalid_row_count": int(np.count_nonzero(deployment_arm_action_invalid)),
        "hand_invalid_row_count": int(np.count_nonzero(deployment_hand_action_invalid)),
        "invalid_row_count": int(np.count_nonzero(deployment_action_invalid)),
    }
    quality["audit_reason_counts"] = {
        name: int(np.count_nonzero(mask)) for name, mask in audit_masks.items()
    }
    quality["repair_reason_counts"] = {
        name: int(np.count_nonzero(mask)) for name, mask in repair_masks.items()
    }
    rejected_reason: str | None = None
    if quality["full_window_count"] < config.min_full_windows:
        rejected_reason = (
            f"source-contiguous segments provide {quality['full_window_count']} "
            f"full windows; requires {config.min_full_windows}"
        )
    return EpisodeDecision(
        source_path=reader.h5_path,
        source_frames=frame_count,
        profile=config.profile,
        selected_indices=selected,
        keep_mask=keep_mask,
        drop_reason_bits=bits,
        drop_reason_names=reason_names,
        hard_reason_counts={
            name: int(np.count_nonzero(mask)) for name, mask in reason_masks.items()
        },
        boundary_counts={
            "timestamp_discontinuity": int(np.count_nonzero(timestamp_gap)),
            "source_index_discontinuity": int(np.count_nonzero(source_gap)),
        },
        selected_frames=int(len(selected)),
        quality=quality,
        source_gap_findings=source_gaps,
        temporal_quality=temporal_assessment.to_dict(config.temporal_quality.policy),
        hard_invalid_reason_names=hard_invalid_reason_names,
        audit_reason_counts=quality["audit_reason_counts"],
        repair_reason_counts=quality["repair_reason_counts"],
        warnings=tuple(warnings),
        rejected_reason=rejected_reason,
    )

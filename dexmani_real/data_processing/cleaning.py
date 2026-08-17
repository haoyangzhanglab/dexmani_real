"""Pure decision logic for cleaning one schema-v16 episode."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dexmani_real.config.defaults import arm, hand
from dexmani_real.data_processing.contracts import EpisodeAnnotation, EpisodeDecision, ProcessingConfig, SegmentDecision
from dexmani_real.recording.episode_reader import EpisodeReader
from dexmani_real.recording.timestamp_buffer import FillReason


def _as_bool(reader: EpisodeReader, name: str) -> np.ndarray:
    return np.asarray(reader.h5f[name][:], dtype=bool)


def _as_i64(reader: EpisodeReader, name: str) -> np.ndarray:
    return np.asarray(reader.h5f[name][:], dtype=np.int64)


def _as_f64(reader: EpisodeReader, name: str) -> np.ndarray:
    return np.asarray(reader.h5f[name][:], dtype=np.float64)


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


def _range_mask(length: int, ranges: tuple[tuple[int, int], ...], *, default: bool) -> np.ndarray:
    mask = np.full(length, default, dtype=bool)
    if ranges and default:
        for start, end in ranges:
            mask[max(0, start) : min(length, end)] = False
    elif ranges:
        for start, end in ranges:
            mask[max(0, start) : min(length, end)] = True
    return mask


def _pointcloud_numeric_mask(reader: EpisodeReader, frame_count: int, *, chunk_size: int = 64) -> np.ndarray:
    dataset = reader.h5f["pointcloud"]
    valid = np.ones(frame_count, dtype=bool)
    for start in range(0, frame_count, chunk_size):
        end = min(frame_count, start + chunk_size)
        chunk = np.asarray(dataset[start:end], dtype=np.float32)
        finite = np.all(np.isfinite(chunk), axis=(1, 2))
        nonzero = np.any(np.linalg.norm(chunk[:, :, :3], axis=2) > 0.0, axis=1)
        color_range = np.all((chunk[:, :, 3:] >= 0.0) & (chunk[:, :, 3:] <= 1.0), axis=(1, 2))
        valid[start:end] = finite & nonzero & color_range
    return valid


def _segment_quality(
    arrays: Mapping[str, np.ndarray],
    start: int,
    end: int,
    *,
    grid_dt_s: float,
) -> dict[str, Any]:
    action = np.concatenate((arrays["action_arm"], arrays["action_hand"]), axis=1)[start:end]
    steps = np.diff(action, axis=0)
    max_step = np.max(np.abs(steps), axis=1) if steps.size else np.empty(0, dtype=np.float64)
    velocity = max_step / grid_dt_s
    acceleration = np.diff(steps / grid_dt_s, axis=0) / grid_dt_s if len(steps) >= 2 else np.empty((0, 19))
    jerk = np.diff(acceleration, axis=0) / grid_dt_s if len(acceleration) >= 2 else np.empty((0, 19))
    quality: dict[str, Any] = {
        "tracking_error_rad": _finite_stats(arrays["tracking_error"][start:end]),
        "max_abs_action_step_rad": _finite_stats(max_step),
        "max_abs_action_velocity_rad_s": _finite_stats(velocity),
        "max_abs_action_acceleration_rad_s2": _finite_stats(
            np.max(np.abs(acceleration), axis=1) if acceleration.size else np.empty(0)
        ),
        "max_abs_action_jerk_rad_s3": _finite_stats(np.max(np.abs(jerk), axis=1) if jerk.size else np.empty(0)),
        "idle_step_ratio": float(np.mean(max_step <= 1e-4)) if max_step.size else 1.0,
    }
    if "camera_age_s" in arrays:
        quality["camera_age_s"] = _finite_stats(arrays["camera_age_s"][start:end])
        quality["camera_frame_gap_count"] = int(np.count_nonzero(arrays["camera_frame_gap"][start:end] > 1))
        quality["camera_duplicate_count"] = int(np.count_nonzero(arrays["camera_duplicate"][start:end]))
    if "pointcloud_source_point_count" in arrays:
        quality["pointcloud_source_point_count"] = _finite_stats(arrays["pointcloud_source_point_count"][start:end])
        quality["pointcloud_padding_count"] = _finite_stats(arrays["pointcloud_padding_count"][start:end])
        quality["pointcloud_valid_depth_ratio"] = _finite_stats(arrays["pointcloud_valid_depth_ratio"][start:end])
    return quality


def analyze_episode(
    reader: EpisodeReader,
    config: ProcessingConfig,
    annotation: EpisodeAnnotation | None = None,
) -> EpisodeDecision:
    """Return a deterministic cleaning decision without writing output."""

    reader.require_valid(purpose="offline processing")
    annotation = annotation or EpisodeAnnotation()
    meta = reader.h5f["meta"]
    frame_count = int(meta.attrs["num_frames"])
    if not annotation.include:
        return EpisodeDecision(
            source_path=reader.h5_path,
            source_frames=frame_count,
            profile=config.profile,
            segments=(),
            hard_reason_counts={"annotation_excluded_episode": frame_count},
            boundary_counts={},
            dropped_short_segment_frames=0,
            selected_frames=0,
            quality={},
            rejected_reason="excluded by annotation",
        )
    if "action_arm_joint_sent" not in reader.h5f:
        return EpisodeDecision(
            source_path=reader.h5_path,
            source_frames=frame_count,
            profile=config.profile,
            segments=(),
            hard_reason_counts={"missing_arm_sent_stream": frame_count},
            boundary_counts={},
            dropped_short_segment_frames=0,
            selected_frames=0,
            quality={},
            rejected_reason="action_arm_joint_sent is required; unsafe fallback is disabled",
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
        "hand_error": _as_bool(reader, "hand_error_state"),
        "hand_stale": _as_bool(reader, "hand_qpos_stale"),
        "history_valid": np.asarray(reader.h5f["observation_history_valid_mask"][:], dtype=bool)[:, :, 0],
        "action_created": _as_i64(reader, "action_created_monotonic_ns"),
        "action_target": _as_i64(reader, "action_target_monotonic_ns"),
        "action_valid_until": _as_i64(reader, "action_valid_until_monotonic_ns"),
        "arm_qpos": _as_f64(reader, "arm_qpos"),
        "hand_qpos": _as_f64(reader, "hand_qpos"),
        "action_arm": _as_f64(reader, "action_arm_joint_sent"),
        "action_hand": _as_f64(reader, "action_hand_joint"),
        "tracking_error": _as_f64(reader, "tracking_error"),
    }

    is_source = arrays["fill_reason"] == int(FillReason.SOURCE)
    timing_valid = (
        (arrays["action_created"] > 0)
        & (arrays["action_created"] <= arrays["action_target"])
        & (arrays["action_target"] <= arrays["action_valid_until"])
    )
    numeric = np.concatenate(
        (
            arrays["arm_qpos"],
            arrays["hand_qpos"],
            arrays["action_arm"],
            arrays["action_hand"],
        ),
        axis=1,
    )
    finite = np.all(np.isfinite(numeric), axis=1)

    tolerance = config.joint_limit_tolerance_rad
    arm_lower = np.asarray(arm.joint_limit_lower, dtype=np.float64)
    arm_upper = np.asarray(arm.joint_limit_upper, dtype=np.float64)
    hand_state_lower = np.asarray(hand.mechanical_qpos_min_rad, dtype=np.float64)
    hand_state_upper = np.asarray(hand.mechanical_qpos_max_rad, dtype=np.float64)
    hand_action_lower = np.asarray(hand.qpos_min_rad, dtype=np.float64)
    hand_action_upper = np.asarray(hand.qpos_max_rad, dtype=np.float64)

    def _inside(
        values: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        value_tolerance: float = tolerance,
    ) -> np.ndarray:
        return np.all(
            (values >= lower - value_tolerance) & (values <= upper + value_tolerance),
            axis=1,
        )

    limits_valid = (
        _inside(arrays["arm_qpos"], arm_lower, arm_upper)
        & _inside(arrays["action_arm"], arm_lower, arm_upper)
        & _inside(
            arrays["hand_qpos"],
            hand_state_lower,
            hand_state_upper,
            value_tolerance=config.hand_state_limit_tolerance_rad,
        )
        & _inside(arrays["action_hand"], hand_action_lower, hand_action_upper)
    )

    reason_masks: dict[str, np.ndarray] = {
        "not_source_sample": ~(arrays["sample_valid"] & is_source),
        "action_not_queued": ~arrays["queued"],
        "held": arrays["held"],
        "safety_reject": arrays["safety_reject"],
        "frame_status_not_ok": arrays["frame_status"] != 0,
        "arm_source_invalid": ~(arrays["arm_connected"] & arrays["history_valid"][:, 0]),
        "hand_source_invalid": ~(
            arrays["hand_connected"] & ~arrays["hand_error"] & ~arrays["hand_stale"] & arrays["history_valid"][:, 1]
        ),
        "action_timing_invalid": ~timing_valid,
        "nonfinite_joint_or_action": ~finite,
        "joint_limit_violation": ~limits_valid,
    }

    if config.profile.needs_rgb or config.profile.needs_pointcloud:
        arrays.update(
            {
                "camera_age_s": _as_f64(reader, "camera_age_s"),
                "camera_frame_gap": _as_i64(reader, "camera_frame_gap"),
                "camera_duplicate": _as_bool(reader, "camera_duplicate"),
            }
        )
        camera_age_valid = (
            np.isfinite(arrays["camera_age_s"])
            & (arrays["camera_age_s"] >= 0.0)
            & (arrays["camera_age_s"] <= config.max_camera_age_s)
        )
        reason_masks["camera_invalid"] = ~(
            _as_bool(reader, "flag_camera_fresh")
            & arrays["history_valid"][:, 3]
            & ~_as_bool(reader, "camera_clock_reset")
            & camera_age_valid
        )

    if config.profile.needs_pointcloud:
        pointcloud_capacity = int(reader.h5f["pointcloud"].shape[1])
        arrays.update(
            {
                "pointcloud_source_point_count": _as_i64(reader, "pointcloud_source_point_count"),
                "pointcloud_padding_count": _as_i64(reader, "pointcloud_padding_count"),
                "pointcloud_valid_depth_ratio": _as_f64(reader, "pointcloud_valid_depth_ratio"),
            }
        )
        pointcloud_meta_valid = (
            _as_bool(reader, "flag_pointcloud_valid")
            & (arrays["pointcloud_source_point_count"] > 0)
            & (arrays["pointcloud_padding_count"] >= 0)
            & (arrays["pointcloud_padding_count"] <= pointcloud_capacity)
            & (
                arrays["pointcloud_padding_count"]
                == np.maximum(pointcloud_capacity - arrays["pointcloud_source_point_count"], 0)
            )
            & np.isfinite(arrays["pointcloud_valid_depth_ratio"])
            & (arrays["pointcloud_valid_depth_ratio"] >= 0.0)
            & (arrays["pointcloud_valid_depth_ratio"] <= 1.0)
        )
        reason_masks["pointcloud_invalid"] = ~(pointcloud_meta_valid & _pointcloud_numeric_mask(reader, frame_count))

    include_mask = _range_mask(
        frame_count,
        annotation.include_ranges,
        default=not bool(annotation.include_ranges),
    )
    exclude_mask = _range_mask(frame_count, annotation.exclude_ranges, default=True)
    annotation_selected = include_mask & exclude_mask
    reason_masks["annotation_excluded_row"] = ~annotation_selected

    hard_invalid = np.zeros(frame_count, dtype=bool)
    for mask in reason_masks.values():
        hard_invalid |= mask
    valid = ~hard_invalid

    timing = reader.timing
    dt = arrays["timestamp"][1:] - arrays["timestamp"][:-1]
    timestamp_gap = np.abs(dt - timing.grid_dt_s) > max(1e-7, timing.grid_dt_s * config.grid_dt_relative_tolerance)
    source_gap = np.diff(arrays["source_index"]) != 1
    break_before = np.zeros(frame_count, dtype=bool)
    break_before[1:] = timestamp_gap | source_gap

    candidate_ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(frame_count):
        if valid[index] and (start is None or not break_before[index]):
            if start is None:
                start = index
            continue
        if start is not None:
            candidate_ranges.append((start, index))
            start = None
        if valid[index]:
            start = index
    if start is not None:
        candidate_ranges.append((start, frame_count))

    segments: list[SegmentDecision] = []
    dropped_short = 0
    for range_start, range_end in candidate_ranges:
        length = range_end - range_start
        full_windows = max(0, length - config.horizon + 1)
        if full_windows < config.min_full_windows:
            dropped_short += length
            continue
        segments.append(
            SegmentDecision(
                start=range_start,
                end=range_end,
                full_window_count=full_windows,
                quality=_segment_quality(arrays, range_start, range_end, grid_dt_s=timing.grid_dt_s),
            )
        )

    warnings: list[str] = []
    ik_ok = _as_bool(reader, "flag_ik_ok")
    retarget_ok = _as_bool(reader, "flag_retarget_ok")
    frame_ok = arrays["frame_status"] == 0
    if np.count_nonzero(frame_ok & ~ik_ok) > frame_count // 2:
        warnings.append("flag_ik_ok conflicts with mostly-OK frame_status; historical recorder alias bug suspected")
    if np.count_nonzero(frame_ok & ~retarget_ok) > frame_count // 2:
        warnings.append(
            "flag_retarget_ok conflicts with mostly-OK frame_status; historical recorder alias bug suspected"
        )

    selected_frames = sum(segment.length for segment in segments)
    selected_indices = (
        np.concatenate([np.arange(segment.start, segment.end) for segment in segments])
        if segments
        else np.empty(0, dtype=np.int64)
    )
    overall_quality: dict[str, Any] = {
        "tracking_error_rad": _finite_stats(arrays["tracking_error"][selected_indices]),
        "high_tracking_error_count": int(
            np.count_nonzero(arrays["tracking_error"][selected_indices] > arm.tracking_error_warn_rad)
        ),
        "full_window_count": int(sum(segment.full_window_count for segment in segments)),
    }
    if "camera_age_s" in arrays:
        overall_quality["camera_age_s"] = _finite_stats(arrays["camera_age_s"][selected_indices])
    if "pointcloud_padding_count" in arrays:
        overall_quality["pointcloud_padding_count"] = _finite_stats(
            arrays["pointcloud_padding_count"][selected_indices]
        )

    rejected_reason = None if segments else "no contiguous segment satisfies the configured training window"
    return EpisodeDecision(
        source_path=reader.h5_path,
        source_frames=frame_count,
        profile=config.profile,
        segments=tuple(segments),
        hard_reason_counts={name: int(np.count_nonzero(mask)) for name, mask in reason_masks.items()},
        boundary_counts={
            "timestamp_discontinuity": int(np.count_nonzero(timestamp_gap)),
            "source_index_discontinuity": int(np.count_nonzero(source_gap)),
        },
        dropped_short_segment_frames=dropped_short,
        selected_frames=selected_frames,
        quality=overall_quality,
        warnings=tuple(warnings),
        rejected_reason=rejected_reason,
    )

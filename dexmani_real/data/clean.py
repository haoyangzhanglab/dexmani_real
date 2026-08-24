"""Pure one-source-to-one-artifact cleaning decisions for supported episodes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.data.contracts import (
    BridgePolicy,
    EpisodeAnnotation,
    EpisodeDecision,
    ProcessingConfig,
)
from dexmani_real.data.quality import assess_temporal_quality
from dexmani_real.recording.reader import EpisodeReader
from dexmani_real.recording.timeline import FillReason
from dexmani_real.sensor.camera_worker import CameraHealth


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


def _quality_summary(
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
    *,
    grid_dt_s: float,
    tracking_error_warn_rad: float,
    horizon: int,
) -> dict[str, Any]:
    action = np.concatenate((arrays["action_arm"], arrays["action_hand"]), axis=1)[
        selected
    ]
    steps = np.diff(action, axis=0)
    max_step = (
        np.max(np.abs(steps), axis=1) if steps.size else np.empty(0, dtype=np.float64)
    )
    acceleration = (
        np.diff(steps / grid_dt_s, axis=0) / grid_dt_s
        if len(steps) >= 2
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
        "full_window_count": max(0, int(len(selected)) - horizon + 1),
    }
    if "camera_age_s" in arrays:
        summary["camera_age_s"] = _finite_stats(arrays["camera_age_s"][selected])
    if "pointcloud_valid_depth_ratio" in arrays:
        summary["pointcloud_valid_depth_ratio"] = _finite_stats(
            arrays["pointcloud_valid_depth_ratio"][selected]
        )
    return summary


def _bridge_findings(
    arrays: Mapping[str, np.ndarray],
    selected: np.ndarray,
    reason_masks: Mapping[str, np.ndarray],
    temporal_excluded: np.ndarray,
    config: ProcessingConfig,
    *,
    grid_dt_s: float,
) -> tuple[dict[str, Any], ...]:
    """Audit every adjacency that differs from the original one-step grid.

    Findings never create output segments.  ``BridgePolicy`` decides whether
    the complete compacted episode is accepted or rejected.
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
                "risky": risky,
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
        "arm_qpos": _as_f64(reader, "arm_qpos"),
        "hand_qpos": _as_f64(reader, "hand_qpos"),
        "action_arm": _as_f64(reader, "action_arm_joint_sent"),
        "action_hand": _as_f64(reader, "action_hand_joint"),
        "action_arm_ee": _as_f64(reader, "action_arm_ee"),
        "contact_force": _as_f64(reader, "hand_contact"),
        "fingertip_points": _as_f64(reader, "hand_fingertip"),
        "tracking_error": _as_f64(reader, "tracking_error"),
        "arm_last_cmd_seq": _as_i64(reader, "arm_last_cmd_seq"),
    }
    is_source = arrays["fill_reason"] == int(FillReason.SOURCE)
    timing_valid = (
        (arrays["action_created"] > 0)
        & (arrays["action_created"] <= arrays["action_target"])
        & (arrays["action_target"] <= arrays["action_valid_until"])
    )
    joint_numeric = np.concatenate(
        (
            arrays["arm_qpos"],
            arrays["hand_qpos"],
            arrays["action_arm"],
            arrays["action_hand"],
        ),
        axis=1,
    )
    real_modalities_finite = (
        np.all(np.isfinite(arrays["action_arm_ee"]), axis=1)
        & np.all(np.isfinite(arrays["contact_force"]), axis=(1, 2))
        & np.all(np.isfinite(arrays["fingertip_points"]), axis=(1, 2))
    )
    arm_lower = np.asarray(config.arm_joint_limit_lower_rad, dtype=np.float64)
    arm_upper = np.asarray(config.arm_joint_limit_upper_rad, dtype=np.float64)
    hand_state_lower = np.asarray(config.hand_state_limit_lower_rad, dtype=np.float64)
    hand_state_upper = np.asarray(config.hand_state_limit_upper_rad, dtype=np.float64)
    hand_action_lower = np.asarray(config.hand_action_limit_lower_rad, dtype=np.float64)
    hand_action_upper = np.asarray(config.hand_action_limit_upper_rad, dtype=np.float64)
    limits_valid = (
        _inside(
            arrays["arm_qpos"], arm_lower, arm_upper, config.joint_limit_tolerance_rad
        )
        & _inside(
            arrays["action_arm"], arm_lower, arm_upper, config.joint_limit_tolerance_rad
        )
        & _inside(
            arrays["hand_qpos"],
            hand_state_lower,
            hand_state_upper,
            config.hand_state_limit_tolerance_rad,
        )
        & _inside(
            arrays["action_hand"],
            hand_action_lower,
            hand_action_upper,
            config.joint_limit_tolerance_rad,
        )
    )
    tactile_valid = (
        _as_bool(reader, "tactile_fresh")
        & _as_bool(reader, "tactile_calibrated")
        & (_as_i64(reader, "tactile_source_monotonic_ns") > 0)
    )
    reason_masks: dict[str, np.ndarray] = {
        "not_source_sample": ~(arrays["sample_valid"] & is_source),
        "action_not_queued": ~arrays["queued"],
        "held": arrays["held"],
        "safety_reject": arrays["safety_reject"],
        "frame_status_not_ok": arrays["frame_status"] != 0,
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
        "joint_limit_violation": ~limits_valid,
    }

    if config.profile.needs_rgb or config.profile.needs_pointcloud:
        arrays.update(
            {
                "camera_age_s": _as_f64(reader, "camera_age_s"),
                "camera_frame_gap": _as_i64(reader, "camera_frame_gap"),
                "camera_duplicate": _as_bool(reader, "camera_duplicate"),
                "camera_health": _as_i64(reader, "camera_health"),
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
            & (arrays["camera_health"] == int(CameraHealth.OK))
            & camera_age_valid
        )
    if config.profile.needs_rgb or config.profile.needs_pointcloud:
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
    hard_invalid = np.zeros(frame_count, dtype=bool)
    for mask in reason_masks.values():
        hard_invalid |= mask
    base_valid = ~hard_invalid

    timing = reader.timing
    dt = np.diff(arrays["timestamp"])
    timestamp_gap = np.abs(dt - timing.grid_dt_s) > max(
        1e-7, timing.grid_dt_s * config.grid_dt_relative_tolerance
    )
    source_gap = np.diff(arrays["source_index"]) != 1
    break_before = np.zeros(frame_count, dtype=bool)
    break_before[1:] = timestamp_gap | source_gap
    temporal_assessment = assess_temporal_quality(
        arrays,
        base_valid,
        break_before,
        config.temporal_quality,
        tracking_error_warn_rad=config.tracking_error_warn_rad,
    )
    keep_mask = base_valid & ~temporal_assessment.excluded_mask
    selected = np.flatnonzero(keep_mask).astype(np.int64)
    bits, reason_names = _reason_bits(reason_masks, temporal_assessment.excluded_mask)
    bridges = _bridge_findings(
        arrays,
        selected,
        reason_masks,
        temporal_assessment.excluded_mask,
        config,
        grid_dt_s=timing.grid_dt_s,
    )
    risky_bridges = [item for item in bridges if bool(item["risky"])]
    warnings: list[str] = []
    if risky_bridges:
        warnings.append(
            f"row compaction creates {len(risky_bridges)} risky transition(s)"
        )
    frame_ok = arrays["frame_status"] == 0
    if np.count_nonzero(frame_ok & ~_as_bool(reader, "flag_ik_ok")) > frame_count // 2:
        warnings.append("flag_ik_ok conflicts with mostly-OK frame_status")
    if (
        np.count_nonzero(frame_ok & ~_as_bool(reader, "flag_retarget_ok"))
        > frame_count // 2
    ):
        warnings.append("flag_retarget_ok conflicts with mostly-OK frame_status")

    rejected_reason: str | None = None
    if len(selected) < config.min_episode_frames:
        rejected_reason = (
            f"compact episode has {len(selected)} frames; requires "
            f"{config.min_episode_frames}"
        )
    elif risky_bridges and config.bridge_policy is BridgePolicy.REJECT:
        rejected_reason = (
            f"row compaction creates {len(risky_bridges)} risky transition(s); "
            "use bridge_policy=audit only after manual review"
        )

    quality = _quality_summary(
        arrays,
        selected,
        grid_dt_s=timing.grid_dt_s,
        tracking_error_warn_rad=config.tracking_error_warn_rad,
        horizon=config.horizon,
    )
    quality["bridge_count"] = len(bridges)
    quality["risky_bridge_count"] = len(risky_bridges)
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
        bridge_findings=bridges,
        temporal_quality=temporal_assessment.to_dict(config.temporal_quality.policy),
        warnings=tuple(warnings),
        rejected_reason=rejected_reason,
    )

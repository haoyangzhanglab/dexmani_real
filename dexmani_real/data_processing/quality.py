"""Pure source-range-local temporal quality assessment for offline processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dexmani_real.data_processing.contracts import QualityPolicy, TemporalQualityConfig


def contiguous_ranges(
    valid: np.ndarray, break_before: np.ndarray
) -> list[tuple[int, int]]:
    """Bound detector context without defining output episode segments."""

    valid_mask = np.asarray(valid, dtype=bool)
    boundaries = np.asarray(break_before, dtype=bool)
    if valid_mask.ndim != 1 or boundaries.shape != valid_mask.shape:
        raise ValueError("valid and break_before must be same-shape 1-D masks")

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index in range(len(valid_mask)):
        if valid_mask[index] and (start is None or not boundaries[index]):
            if start is None:
                start = index
            continue
        if start is not None:
            ranges.append((start, index))
            start = None
        if valid_mask[index]:
            start = index
    if start is not None:
        ranges.append((start, len(valid_mask)))
    return ranges


def _mask_ranges(mask: np.ndarray) -> list[list[int]]:
    values = np.asarray(mask, dtype=bool)
    if values.ndim != 1:
        raise ValueError("quality reason mask must be 1-D")
    ranges: list[list[int]] = []
    start: int | None = None
    for index, selected in enumerate(values):
        if selected and start is None:
            start = index
        elif not selected and start is not None:
            ranges.append([start, index])
            start = None
    if start is not None:
        ranges.append([start, len(values)])
    return ranges


def _finite_stats(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "p95": None, "p99": None, "max": None}
    return {
        "count": int(finite.size),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _mark_persistent_runs(
    destination: np.ndarray,
    condition: np.ndarray,
    ranges: list[tuple[int, int]],
    minimum_frames: int,
) -> None:
    for range_start, range_end in ranges:
        run_start: int | None = None
        for index in range(range_start, range_end):
            if condition[index] and run_start is None:
                run_start = index
            elif not condition[index] and run_start is not None:
                if index - run_start >= minimum_frames:
                    destination[run_start:index] = True
                run_start = None
        if run_start is not None and range_end - run_start >= minimum_frames:
            destination[run_start:range_end] = True


def _guard_within_ranges(
    mask: np.ndarray,
    ranges: list[tuple[int, int]],
    *,
    before: int,
    after: int,
) -> np.ndarray:
    guarded = np.zeros_like(mask, dtype=bool)
    for range_start, range_end in ranges:
        indices = np.flatnonzero(mask[range_start:range_end]) + range_start
        for index in indices:
            guarded[
                max(range_start, index - before) : min(range_end, index + after + 1)
            ] = True
    return guarded


@dataclass(frozen=True)
class TemporalQualityAssessment:
    """Auditable temporal findings and the policy-selected exclusion mask."""

    suspect_masks: Mapping[str, np.ndarray]
    high_confidence_masks: Mapping[str, np.ndarray]
    excluded_mask: np.ndarray
    score_statistics: Mapping[str, dict[str, float | int | None]]
    detectors_run: bool

    def to_dict(self, policy: QualityPolicy) -> dict[str, Any]:
        suspect_union = np.zeros_like(self.excluded_mask, dtype=bool)
        for mask in self.suspect_masks.values():
            suspect_union |= mask
        high_confidence_union = np.zeros_like(self.excluded_mask, dtype=bool)
        for mask in self.high_confidence_masks.values():
            high_confidence_union |= mask
        return {
            "policy": policy.value,
            "detectors_run": self.detectors_run,
            "suspect_reason_counts": {
                name: int(np.count_nonzero(mask))
                for name, mask in self.suspect_masks.items()
            },
            "suspect_ranges": {
                name: _mask_ranges(mask)
                for name, mask in self.suspect_masks.items()
                if np.any(mask)
            },
            "suspect_union_count": int(np.count_nonzero(suspect_union)),
            "high_confidence_reason_counts": {
                name: int(np.count_nonzero(mask))
                for name, mask in self.high_confidence_masks.items()
            },
            "high_confidence_ranges": {
                name: _mask_ranges(mask)
                for name, mask in self.high_confidence_masks.items()
                if np.any(mask)
            },
            "high_confidence_union_count": int(np.count_nonzero(high_confidence_union)),
            "excluded_count": int(np.count_nonzero(self.excluded_mask)),
            "excluded_ranges": _mask_ranges(self.excluded_mask),
            "score_statistics": dict(self.score_statistics),
        }


def assess_temporal_quality(
    arrays: Mapping[str, np.ndarray],
    base_valid: np.ndarray,
    break_before: np.ndarray,
    config: TemporalQualityConfig,
    *,
    tracking_error_warn_rad: float,
) -> TemporalQualityAssessment:
    """Detect anomalies locally; only STRICT excludes high-confidence rows."""

    base = np.asarray(base_valid, dtype=bool)
    boundaries = np.asarray(break_before, dtype=bool)
    if base.ndim != 1 or boundaries.shape != base.shape:
        raise ValueError("base_valid and break_before must be same-shape 1-D masks")
    frame_count = len(base)
    empty = np.zeros(frame_count, dtype=bool)
    if config.policy is QualityPolicy.HARD_ONLY:
        return TemporalQualityAssessment({}, {}, empty, {}, False)

    action_arm = np.asarray(arrays["action_arm"], dtype=np.float64)
    action_hand = np.asarray(arrays["action_hand"], dtype=np.float64)
    arm_qpos = np.asarray(arrays["arm_qpos"], dtype=np.float64)
    tracking_error = np.asarray(arrays["tracking_error"], dtype=np.float64)
    arm_last_cmd_seq = np.asarray(arrays["arm_last_cmd_seq"], dtype=np.int64)
    expected_shapes = {
        "action_arm": (frame_count, 7),
        "action_hand": (frame_count, 12),
        "arm_qpos": (frame_count, 7),
        "tracking_error": (frame_count,),
        "arm_last_cmd_seq": (frame_count,),
    }
    actual = {
        "action_arm": action_arm.shape,
        "action_hand": action_hand.shape,
        "arm_qpos": arm_qpos.shape,
        "tracking_error": tracking_error.shape,
        "arm_last_cmd_seq": arm_last_cmd_seq.shape,
    }
    for name, expected in expected_shapes.items():
        if actual[name] != expected:
            raise ValueError(f"{name} must have shape {expected}, got {actual[name]}")

    ranges = contiguous_ranges(base, boundaries)
    abrupt_arm = np.zeros(frame_count, dtype=bool)
    abrupt_hand = np.zeros(frame_count, dtype=bool)
    reversal_impulse = np.zeros(frame_count, dtype=bool)
    persistent_tracking = np.zeros(frame_count, dtype=bool)
    feedback_stall = np.zeros(frame_count, dtype=bool)
    command_apply_stall = np.zeros(frame_count, dtype=bool)
    arm_step_score = np.full(frame_count, np.nan, dtype=np.float64)
    hand_step_score = np.full(frame_count, np.nan, dtype=np.float64)
    reversal_score = np.full(frame_count, np.nan, dtype=np.float64)

    impulse_threshold = np.concatenate(
        (
            np.full(7, config.impulse_arm_min_rad, dtype=np.float64),
            np.full(12, config.impulse_hand_min_rad, dtype=np.float64),
        )
    )
    action = np.concatenate((action_arm, action_hand), axis=1)
    for range_start, range_end in ranges:
        if range_end - range_start >= 2:
            arm_steps = np.max(
                np.abs(np.diff(action_arm[range_start:range_end], axis=0)), axis=1
            )
            hand_steps = np.max(
                np.abs(np.diff(action_hand[range_start:range_end], axis=0)), axis=1
            )
            arm_step_score[range_start + 1 : range_end] = arm_steps
            hand_step_score[range_start + 1 : range_end] = hand_steps
            abrupt_arm[range_start + 1 : range_end] = (
                arm_steps > config.abrupt_arm_step_rad
            )
            abrupt_hand[range_start + 1 : range_end] = (
                hand_steps > config.abrupt_hand_step_rad
            )
        if range_end - range_start >= 3:
            previous = (
                action[range_start + 1 : range_end - 1]
                - action[range_start : range_end - 2]
            )
            following = (
                action[range_start + 2 : range_end]
                - action[range_start + 1 : range_end - 1]
            )
            previous_abs = np.abs(previous)
            following_abs = np.abs(following)
            smaller = np.minimum(previous_abs, following_abs)
            larger = np.maximum(previous_abs, following_abs)
            return_ratio = np.divide(
                smaller,
                larger,
                out=np.zeros_like(smaller),
                where=larger > 0.0,
            )
            per_joint = (
                (previous * following < 0.0)
                & (smaller >= impulse_threshold[None, :])
                & (return_ratio >= config.impulse_min_return_ratio)
            )
            centers = slice(range_start + 1, range_end - 1)
            reversal_impulse[centers] = np.any(per_joint, axis=1)
            reversal_score[centers] = np.max(
                np.where(previous * following < 0.0, smaller, 0.0), axis=1
            )

    tracking_condition = np.isfinite(tracking_error) & (
        tracking_error > tracking_error_warn_rad
    )
    _mark_persistent_runs(
        persistent_tracking,
        tracking_condition,
        ranges,
        config.tracking_persistence_frames,
    )

    window = config.stall_window_frames
    for range_start, range_end in ranges:
        for start in range(range_start, range_end - window):
            end = start + window
            command_delta = float(np.max(np.abs(action_arm[end] - action_arm[start])))
            if command_delta < config.stall_arm_command_delta_rad:
                continue
            state_delta = float(np.max(np.abs(arm_qpos[end] - arm_qpos[start])))
            tracking_window = tracking_error[start : end + 1]
            tracking_high = bool(
                np.all(np.isfinite(tracking_window))
                and np.median(tracking_window) > tracking_error_warn_rad
            )
            affected = slice(start + 1, end + 1)
            if state_delta <= config.stall_arm_state_delta_rad and tracking_high:
                feedback_stall[affected] = True
            applied_advance = int(arm_last_cmd_seq[end] - arm_last_cmd_seq[start])
            if 0 <= applied_advance <= config.stall_max_applied_command_advance:
                command_apply_stall[affected] = True

    suspect_masks = {
        "abrupt_arm_action_step": abrupt_arm,
        "abrupt_hand_action_step": abrupt_hand,
        "reversible_action_impulse": reversal_impulse,
        "persistent_arm_tracking_error": persistent_tracking,
        "arm_feedback_stall": feedback_stall,
        "arm_command_apply_stall": command_apply_stall,
    }
    high_confidence_masks = {
        "reversible_action_impulse": reversal_impulse,
        "arm_feedback_stall": feedback_stall,
        "arm_command_apply_stall": command_apply_stall,
    }
    high_confidence = np.zeros(frame_count, dtype=bool)
    for mask in high_confidence_masks.values():
        high_confidence |= mask
    excluded = (
        _guard_within_ranges(
            high_confidence,
            ranges,
            before=config.strict_guard_before_frames,
            after=config.strict_guard_after_frames,
        )
        if config.policy is QualityPolicy.STRICT
        else np.zeros(frame_count, dtype=bool)
    )
    return TemporalQualityAssessment(
        suspect_masks=suspect_masks,
        high_confidence_masks=high_confidence_masks,
        excluded_mask=excluded,
        score_statistics={
            "arm_action_step_rad": _finite_stats(arm_step_score),
            "hand_action_step_rad": _finite_stats(hand_step_score),
            "reversal_return_step_rad": _finite_stats(reversal_score),
            "tracking_error_rad": _finite_stats(tracking_error[base]),
        },
        detectors_run=True,
    )

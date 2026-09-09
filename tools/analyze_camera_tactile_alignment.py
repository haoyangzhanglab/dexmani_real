"""Read-only camera/tactile alignment diagnostic over raw episodes.

This tool quantifies the recording/processing reference mismatch behind the
``invalid_frames_report`` false positives, without modifying any episode.  It
reads raw ``data.h5`` directly (schema-version agnostic) so it can inspect both
historical v26 artifacts and future v27 artifacts alike.  No hardware SDK is
imported or constructed.

Reported groups:

* ``tactile`` — legacy persisted-row tactile selection offsets and lag.
* ``camera`` — ``flag_camera_fresh`` reuse, frame-number reuse, source/age deltas.
* ``coupling`` — tactile lag conditioned on camera fresh / not-new.
* ``episode_integrity`` — per-episode hard-invalid / segment admission under the
  current exporter gap-tolerance rule and the proposed strict whole-episode rule.
* ``proposed_camera_rule`` — historical camera rows that stop being hard-invalid
  once ``flag_camera_fresh`` is downgraded from a hard gate to audit telemetry.

Usage::

    python tools/analyze_camera_tactile_alignment.py \
        --input-root episodes/pick_place_toy \
        --write-json artifacts/camera_tactile_alignment_baseline.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np

from dexmani_real.dataset.clean import analyze_episode, select_tactile_rows_to_references
from dexmani_real.dataset.contracts import OutputProfile, ProcessingConfig

_NS_PER_SECOND = 1_000_000_000
_CURRENT_MAX_INTERIOR_GAP_ROWS = 2


def _discover_episode_dirs(input_root: str | Path) -> tuple[Path, ...]:
    root = Path(input_root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if (root / "data.h5").is_file():
        return (root,)
    episodes = tuple(
        sorted(
            child
            for child in root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
            and (child / "data.h5").is_file()
        )
    )
    if not episodes:
        raise FileNotFoundError(f"no episode directories found under {root}")
    return episodes


class _V25CompatView:
    """Synthesize ``tactile_sum_fresh`` for legacy v25 raw data.

    v25 collapsed aggregate/dense tactile validity into a single
    ``tactile_fresh`` column; the frozen v25→v26 migration copies
    ``tactile_sum_fresh`` conservatively from ``tactile_fresh``.  This read-only
    view mirrors that single-field migration so the canonical cleaner can run
    over v25 artifacts without touching them.
    """

    def __init__(self, data: h5py.File) -> None:
        self._data = data

    def __getitem__(self, name: str) -> Any:
        if name == "tactile_sum_fresh" and "tactile_sum_fresh" not in self._data:
            return self._data["tactile_fresh"]
        return self._data[name]

    def __contains__(self, name: object) -> bool:
        return name in self._data or name == "tactile_sum_fresh"

    def get(self, name: str, default: Any = None) -> Any:
        try:
            return self[name]
        except KeyError:
            return default


def _open_raw(episode_dir: Path) -> tuple[h5py.File, h5py.File, SimpleNamespace]:
    """Open ``data.h5`` and ``depth.h5`` without replaying schema admission."""
    data = h5py.File(episode_dir / "data.h5", "r")
    depth = h5py.File(episode_dir / "depth.h5", "r")
    control_hz = float(data["meta"].attrs["control_hz"])
    timing = SimpleNamespace(grid_dt_s=1.0 / control_hz)
    return data, depth, timing


def _reader_shim(
    episode_dir: Path, data: h5py.File, timing: SimpleNamespace
) -> SimpleNamespace:
    """Expose the same surface ``analyze_episode`` consumes, minus version gate."""
    view = data if "tactile_sum_fresh" in data else _V25CompatView(data)
    return SimpleNamespace(h5f=view, h5_path=episode_dir, timing=timing)


def _depth_valid_mask(depth: h5py.File, frame_count: int) -> np.ndarray:
    ds = depth["depth"]
    valid = np.zeros(frame_count, dtype=bool)
    if ds.ndim != 3:
        return valid
    for start in range(0, frame_count, 256):
        block = np.asarray(ds[start : start + 256], dtype=np.uint16)
        valid[start : start + len(block)] = np.any(block > 0, axis=(1, 2))
    return valid


def _finite_stats(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "p50": None, "p90": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(finite.size),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "p95": float(np.percentile(finite, 95)),
        "p99": float(np.percentile(finite, 99)),
        "max": float(np.max(finite)),
    }


def _bool(reader: SimpleNamespace, name: str) -> np.ndarray:
    return np.asarray(reader.h5f[name][:], dtype=bool)


def _i64(reader: SimpleNamespace, name: str) -> np.ndarray:
    return np.asarray(reader.h5f[name][:], dtype=np.int64)


def _f64(reader: SimpleNamespace, name: str) -> np.ndarray:
    return np.asarray(reader.h5f[name][:], dtype=np.float64)


def _current_rule_exportable(
    selected: np.ndarray,
    source_gap_findings: tuple[dict[str, Any], ...],
    grid_dt_s: float,
    tolerance_s: float,
) -> bool:
    """Mirror the current exporter gap-tolerance admission (leading trim OK)."""
    for finding in source_gap_findings:
        row_delta = int(finding["source_row_after"]) - int(finding["source_row_before"])
        if row_delta - 1 > _CURRENT_MAX_INTERIOR_GAP_ROWS:
            return False
        if int(finding["source_sample_delta"]) != row_delta:
            return False
        if abs(float(finding["source_timestamp_delta_s"]) - row_delta * grid_dt_s) > tolerance_s:
            return False
    return len(selected) > 0


def _strict_rule_exportable(selected: np.ndarray, source_frames: int) -> bool:
    """Proposed strict rule: no dropped source row at all."""
    return np.array_equal(selected, np.arange(source_frames, dtype=np.int64))


def _hard_invalid_mask(decision: Any) -> np.ndarray:
    mask = np.zeros(decision.source_frames, dtype=bool)
    for bit, name in enumerate(decision.drop_reason_names):
        if name in decision.hard_invalid_reason_names:
            mask |= (decision.drop_reason_bits & (np.uint64(1) << np.uint64(bit))) != 0
    return mask


def analyze_episode_record(
    episode_dir: Path,
    config: ProcessingConfig,
    *,
    data: h5py.File,
    depth: h5py.File,
    timing: SimpleNamespace,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute one episode's tactile/camera/integrity statistics read-only."""
    reader = _reader_shim(episode_dir, data, timing)
    frame_count = int(data["meta"].attrs["num_frames"])
    visual = config.profile.needs_rgb or config.profile.needs_pointcloud

    # ── tactile selection (legacy persisted-row selector, camera reference) ──
    hand_source_ns = _i64(reader, "hand_source_monotonic_ns")
    tactile_source_ns = _i64(reader, "tactile_source_monotonic_ns")
    tactile_pair_fresh = _bool(reader, "tactile_sum_fresh") & _bool(
        reader, "tactile_fresh"
    )
    tactile_calibrated = _bool(reader, "tactile_calibrated")
    tactile_unit_code = _i64(reader, "tactile_unit_code")
    camera_source_ns = _i64(reader, "camera_source_monotonic_ns")
    anchor_ns = _i64(reader, "observation_anchor_monotonic_ns")
    reference_ns = camera_source_ns if visual else anchor_ns
    tactile_rows = select_tactile_rows_to_references(
        hand_source_ns,
        tactile_source_ns,
        tactile_pair_fresh,
        tactile_calibrated,
        tactile_unit_code,
        reference_ns,
        max_observation_skew_s=config.max_observation_skew_s,
    )
    tactile_valid = tactile_rows >= 0
    frame0_deficit = bool(not tactile_valid[0]) if frame_count else False
    offsets = np.full(frame_count, 0, dtype=np.int64)
    lag_ns = np.full(frame_count, np.nan, dtype=np.float64)
    for row in range(frame_count):
        if tactile_rows[row] >= 0:
            offsets[row] = tactile_rows[row] - row
            lag_ns[row] = reference_ns[row] - tactile_source_ns[tactile_rows[row]]

    # ── camera statistics ──
    flag_camera_fresh = _bool(reader, "flag_camera_fresh") if visual else np.zeros(
        frame_count, dtype=bool
    )
    depth_frame_number = _i64(reader, "camera_depth_frame_number")
    color_frame_number = _i64(reader, "camera_color_frame_number")
    same_depth = int(
        np.count_nonzero(np.diff(depth_frame_number) == 0)
    ) if frame_count > 1 else 0
    same_color = int(
        np.count_nonzero(np.diff(color_frame_number) == 0)
    ) if frame_count > 1 else 0
    camera_source_delta_ms = (
        np.diff(camera_source_ns).astype(np.float64) / 1e6 if frame_count > 1 else np.empty(0)
    )
    camera_age_ms = (
        (anchor_ns - camera_source_ns).astype(np.float64) / 1e6
        if visual
        else np.empty(0)
    )

    # ── coupling ──
    fresh_mask = flag_camera_fresh & tactile_valid
    not_new_mask = (~flag_camera_fresh) & tactile_valid
    lag_when_fresh = lag_ns[fresh_mask] / 1e6
    lag_when_not_new = lag_ns[not_new_mask] / 1e6

    # ── episode integrity via the canonical cleaner ──
    decision = analyze_episode(
        reader,
        config,
        source_already_validated=True,
        depth_valid_mask=(
            _depth_valid_mask(depth, frame_count) if visual else None
        ),
    )
    hard_invalid = _hard_invalid_mask(decision)
    kept = decision.selected_indices
    leading_invalid = int(np.count_nonzero(hard_invalid[: kept[0]])) if kept.size else 0
    suffix_invalid = int(np.count_nonzero(hard_invalid[kept[-1] + 1 :])) if kept.size else 0
    internal_invalid = int(np.count_nonzero(hard_invalid)) - leading_invalid - suffix_invalid
    tolerance_s = max(1e-7, timing.grid_dt_s * config.grid_dt_relative_tolerance)
    current_exportable = _current_rule_exportable(
        kept, decision.source_gap_findings, timing.grid_dt_s, tolerance_s
    )
    strict_exportable = _strict_rule_exportable(kept, frame_count)

    # ── proposed camera rule simulation (historical raw has no camera_health) ──
    if visual:
        camera_age_s = (anchor_ns - camera_source_ns).astype(np.float64) / 1e9
        camera_old_hard = ~(
            flag_camera_fresh
            & (camera_source_ns > 0)
            & (camera_age_s >= 0.0)
            & (camera_age_s <= config.max_camera_age_s)
        )
        camera_proposed_hard = ~(
            (camera_source_ns > 0)
            & (camera_age_s >= 0.0)
            & (camera_age_s <= config.max_camera_age_s)
        )
        camera_reclassified = camera_old_hard & ~camera_proposed_hard
        old_hard_total = int(np.count_nonzero(camera_old_hard))
        proposed_hard_total = int(np.count_nonzero(camera_proposed_hard))
    else:
        camera_old_hard = np.zeros(frame_count, dtype=bool)
        camera_proposed_hard = np.zeros(frame_count, dtype=bool)
        camera_reclassified = np.zeros(frame_count, dtype=bool)
        old_hard_total = 0
        proposed_hard_total = 0

    # Whether reclassifying benign camera rows removes every interior boundary.
    internal_gap_disappears = False
    if camera_reclassified.any():
        simulated_hard = hard_invalid & ~camera_reclassified
        simulated_kept = np.flatnonzero(~simulated_hard).astype(np.int64)
        simulated_segments = len(
            _segment_ends(simulated_kept, timing.grid_dt_s, tolerance_s, data)
        )
        internal_gap_disappears = simulated_segments == 1

    summary = {
        "episode": episode_dir.name,
        "source_frames": frame_count,
        "tactile": {
            "frame0_causal_deficit": frame0_deficit,
            "same_row_selected": int(np.count_nonzero(offsets == 0)),
            "previous_row_selected": int(np.count_nonzero(offsets == -1)),
            "older_than_previous_row": int(np.count_nonzero(offsets < -1)),
            "offset_histogram": dict(
                sorted((int(k), int(v)) for k, v in Counter(offsets.tolist()).items())
            ),
        },
        "camera": {
            "flag_camera_fresh_false": int(np.count_nonzero(~flag_camera_fresh)),
            "same_depth_frame_number_as_previous": same_depth,
            "same_color_frame_number_as_previous": same_color,
        },
        "integrity": {
            "hard_invalid_frames": int(np.count_nonzero(hard_invalid)),
            "leading_invalid": leading_invalid,
            "internal_invalid": internal_invalid,
            "suffix_invalid": suffix_invalid,
            "segment_count": int(len(decision.segment_ends)),
            "exportable_current_rule": bool(current_exportable),
            "exportable_strict_rule": bool(strict_exportable),
        },
        "camera_rule": {
            "hard_rows_old_rule": old_hard_total,
            "hard_rows_proposed_rule": proposed_hard_total,
            "reclassified_rows": int(np.count_nonzero(camera_reclassified)),
            "internal_gap_disappears": bool(internal_gap_disappears),
        },
    }
    raw = {
        "lag_ms": lag_ns[tactile_valid] / 1e6,
        "source_delta_ms": camera_source_delta_ms,
        "age_ms": camera_age_ms,
        "lag_fresh_ms": lag_when_fresh,
        "lag_not_new_ms": lag_when_not_new,
    }
    return summary, raw


def _segment_ends(
    selected: np.ndarray,
    grid_dt_s: float,
    tolerance_s: float,
    data: h5py.File,
) -> np.ndarray:
    """Recompute compact-array segment ends for a simulated selection."""
    if selected.size == 0:
        return np.empty(0, dtype=np.int64)
    samples = np.asarray(data["source_sample_index"][selected], dtype=np.int64)
    timestamps = np.asarray(data["timestamp"][selected], dtype=np.float64)
    discontinuity = (
        (np.diff(selected) != 1)
        | (np.diff(samples) != 1)
        | (np.abs(np.diff(timestamps) - grid_dt_s) > tolerance_s)
    )
    return np.concatenate(
        (np.flatnonzero(discontinuity).astype(np.int64) + 1, [selected.size])
    ).astype(np.int64)


def _concat(arrays: list[np.ndarray]) -> np.ndarray:
    nonempty = [np.asarray(a, dtype=np.float64) for a in arrays if a.size]
    return np.concatenate(nonempty) if nonempty else np.empty(0, dtype=np.float64)


def run(input_root: str | Path, *, profile: str = "pointcloud") -> dict[str, Any]:
    config = ProcessingConfig(profile=OutputProfile(profile))
    episode_dirs = _discover_episode_dirs(input_root)
    records: list[dict[str, Any]] = []
    raw_arrays: list[dict[str, np.ndarray]] = []
    for episode_dir in episode_dirs:
        data, depth, timing = _open_raw(episode_dir)
        try:
            summary, raw = analyze_episode_record(
                episode_dir,
                config,
                data=data,
                depth=depth,
                timing=timing,
            )
            records.append(summary)
            raw_arrays.append(raw)
        finally:
            depth.close()
            data.close()

    frame0_deficit = sum(int(r["tactile"]["frame0_causal_deficit"]) for r in records)
    same_row = sum(r["tactile"]["same_row_selected"] for r in records)
    previous_row = sum(r["tactile"]["previous_row_selected"] for r in records)
    older_row = sum(r["tactile"]["older_than_previous_row"] for r in records)
    lag_ms = _finite_stats(_concat([r["lag_ms"] for r in raw_arrays]))
    fresh_false = sum(r["camera"]["flag_camera_fresh_false"] for r in records)
    same_depth = sum(r["camera"]["same_depth_frame_number_as_previous"] for r in records)
    same_color = sum(r["camera"]["same_color_frame_number_as_previous"] for r in records)
    source_delta = _finite_stats(_concat([r["source_delta_ms"] for r in raw_arrays]))
    age_ms = _finite_stats(_concat([r["age_ms"] for r in raw_arrays]))
    lag_fresh = _finite_stats(_concat([r["lag_fresh_ms"] for r in raw_arrays]))
    lag_not_new = _finite_stats(_concat([r["lag_not_new_ms"] for r in raw_arrays]))

    camera_old = sum(r["camera_rule"]["hard_rows_old_rule"] for r in records)
    camera_proposed = sum(r["camera_rule"]["hard_rows_proposed_rule"] for r in records)
    camera_reclassified = sum(r["camera_rule"]["reclassified_rows"] for r in records)
    internal_gap_disappears = sum(
        int(r["camera_rule"]["internal_gap_disappears"]) for r in records
    )
    exportable_current = sum(
        int(r["integrity"]["exportable_current_rule"]) for r in records
    )
    exportable_strict = sum(int(r["integrity"]["exportable_strict_rule"]) for r in records)
    episodes_with_hard_invalid = sum(
        int(r["integrity"]["hard_invalid_frames"] > 0) for r in records
    )

    return {
        "schema_name": "dexmani-real-camera-tactile-alignment-baseline",
        "schema_version": 1,
        "input_root": str(Path(input_root).resolve()),
        "profile": profile,
        "episode_count": len(records),
        "tactile": {
            "frame0_causal_deficit_count": frame0_deficit,
            "same_row_selected_count": same_row,
            "previous_row_selected_count": previous_row,
            "older_than_previous_row_count": older_row,
            "camera_minus_tactile_ms": lag_ms,
        },
        "camera": {
            "flag_camera_fresh_false_count": fresh_false,
            "same_depth_frame_number_as_previous_count": same_depth,
            "same_color_frame_number_as_previous_count": same_color,
            "camera_source_delta_ms": source_delta,
            "camera_age_ms": age_ms,
        },
        "coupling": {
            "tactile_lag_ms_when_camera_fresh": lag_fresh,
            "tactile_lag_ms_when_camera_not_new": lag_not_new,
        },
        "episode_integrity": {
            "episodes_with_hard_invalid": episodes_with_hard_invalid,
            "exportable_current_rule": exportable_current,
            "exportable_strict_rule": exportable_strict,
        },
        "proposed_camera_rule": {
            "camera_hard_rows_old_rule": camera_old,
            "camera_hard_rows_proposed_rule": camera_proposed,
            "camera_rows_reclassified_to_audit": camera_reclassified,
            "episodes_losing_internal_gap": internal_gap_disappears,
        },
        "episodes": records,
    }


def _print_summary(result: dict[str, Any]) -> None:
    print(f"episodes: {result['episode_count']}  profile: {result['profile']}")
    print(f"input: {result['input_root']}")
    print("\n[tactile]")
    print(f"  frame0 causal deficit episodes : {result['tactile']['frame0_causal_deficit_count']}")
    print(f"  same-row / prev-row / older    : "
          f"{result['tactile']['same_row_selected_count']} / "
          f"{result['tactile']['previous_row_selected_count']} / "
          f"{result['tactile']['older_than_previous_row_count']}")
    lag = result["tactile"]["camera_minus_tactile_ms"]
    print(f"  camera - tactile lag (ms)      : p50={lag['p50']} p95={lag['p95']} "
          f"p99={lag['p99']} max={lag['max']}")
    print("\n[camera]")
    print(f"  flag_camera_fresh=false rows   : {result['camera']['flag_camera_fresh_false_count']}")
    print(f"  same depth / color frame reuse : "
          f"{result['camera']['same_depth_frame_number_as_previous_count']} / "
          f"{result['camera']['same_color_frame_number_as_previous_count']}")
    print(f"  source delta ms                : p50={result['camera']['camera_source_delta_ms']['p50']} "
          f"max={result['camera']['camera_source_delta_ms']['max']}")
    print(f"  age ms                         : p50={result['camera']['camera_age_ms']['p50']} "
          f"max={result['camera']['camera_age_ms']['max']}")
    print("\n[coupling]")
    print(f"  lag ms | camera fresh          : p50={result['coupling']['tactile_lag_ms_when_camera_fresh']['p50']}")
    print(f"  lag ms | camera not-new        : p50={result['coupling']['tactile_lag_ms_when_camera_not_new']['p50']}")
    print("\n[episode integrity]")
    print(f"  episodes with hard-invalid     : {result['episode_integrity']['episodes_with_hard_invalid']}")
    print(f"  exportable current / strict    : "
          f"{result['episode_integrity']['exportable_current_rule']} / "
          f"{result['episode_integrity']['exportable_strict_rule']}")
    print("\n[proposed camera rule]")
    print(f"  camera hard rows old / proposed: "
          f"{result['proposed_camera_rule']['camera_hard_rows_old_rule']} / "
          f"{result['proposed_camera_rule']['camera_hard_rows_proposed_rule']}")
    print(f"  camera rows reclassified       : {result['proposed_camera_rule']['camera_rows_reclassified_to_audit']}")
    print(f"  episodes losing internal gap   : {result['proposed_camera_rule']['episodes_losing_internal_gap']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="episodes/pick_place_toy")
    parser.add_argument("--profile", default="pointcloud",
                        choices=["joint", "rgb", "pointcloud", "rgb_pc"])
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args(argv)

    result = run(args.input_root, profile=args.profile)
    _print_summary(result)
    if args.write_json:
        out_path = Path(args.write_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

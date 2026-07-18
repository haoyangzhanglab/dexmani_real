"""DataValidator — automated quality checks for teleop episodes.

8 validation check categories run on HDF5 episode files before Zarr export:
  1. no_nan_obs     — observations contain no NaN values
  2. no_nan_action  — actions contain no NaN values
  3. non_zero_variance — each dimension has variance > epsilon
  4. camera_fresh   — camera frames are non-all-zero (if camera data present)
  5. min_frames     — episode has >= min_frames frames (default 50 ≙ 1s @50Hz)
  6. no_duplicate_frames  — no consecutive identical frames (stuck sensor)
  7. timestamp_monotonic  — timestamps non-decreasing (duplicates = forward-fill)
  8. camera_stall   — <=10% of frames with stale (frozen) camera data

The actual total varies per episode depending on which data streams are present.

Ref: data collection loop design — Phase 3 (offline tools).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

__all__ = ["DataValidator", "ValidationReport", "ValidationCheck"]


@dataclass
class ValidationCheck:
    """Result of a single validation check."""
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    """Aggregate validation report for one or more episodes."""
    episode_path: str = ""
    total_checks: int = 0
    passed_checks: int = 0
    checks: list[ValidationCheck] = field(default_factory=list)
    episode_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.passed_checks == self.total_checks and self.total_checks > 0


class DataValidator:
    """Validates HDF5 teleop episodes before Zarr export.

    Usage:
        validator = DataValidator()
        report = validator.validate("episode_000.h5")
        if report.is_valid:
            # proceed to export
    """

    def __init__(
        self,
        min_frames: int = 50,
        variance_epsilon: float = 1e-8,
        duplicate_epsilon: float = 1e-4,
    ) -> None:
        self.min_frames = min_frames
        self.variance_epsilon = variance_epsilon
        self.duplicate_epsilon = duplicate_epsilon

    def validate(self, h5_path: str | Path) -> ValidationReport:
        """Run all 8 checks on a single episode file.

        Returns a ValidationReport with pass/fail per check.
        """
        h5_path = Path(h5_path)
        report = ValidationReport(episode_path=str(h5_path))
        checks: list[ValidationCheck] = []

        try:
            with h5py.File(str(h5_path), "r") as f:
                meta = dict(f["meta"].attrs) if "meta" in f else {}
                report.episode_metadata = {
                    k: v for k, v in meta.items()
                    if not isinstance(v, (np.ndarray, bytes))
                }

                # ── 1. No NaN in observations ──
                checks.append(self._check_no_nan(f, "arm_qpos", "no_nan_obs"))
                checks.append(self._check_no_nan(f, "arm_ee", "no_nan_obs"))
                checks.append(self._check_no_nan(f, "hand_qpos", "no_nan_obs"))

                # ── 2. No NaN in actions ──
                checks.append(self._check_no_nan(f, "action_arm_joint", "no_nan_action"))
                checks.append(self._check_no_nan(f, "action_arm_ee", "no_nan_action"))
                checks.append(self._check_no_nan(f, "action_hand_joint", "no_nan_action"))

                # ── 3. Non-zero variance ──
                for key in ("arm_qpos", "arm_ee", "hand_qpos",
                            "action_arm_joint", "action_hand_joint"):
                    if key in f:
                        checks.append(self._check_variance(f, key))

                # ── 4. Camera freshness ──
                checks.append(self._check_camera(f))

                # ── 5. Minimum frames ──
                checks.append(self._check_min_frames(f))

                # ── 6. No consecutive duplicate frames ──
                checks.append(self._check_duplicate_frames(f))

                # ── 7. Timestamp monotonicity ──
                checks.append(self._check_timestamp_monotonicity(f))

                # ── 8. Camera stall (frozen frames, schema v6+) ──
                checks.append(self._check_camera_stall(f))

        except (OSError, KeyError) as e:
            checks.append(ValidationCheck(
                name="file_open", passed=False,
                detail=f"Cannot open/read file: {e}",
            ))

        report.checks = checks
        report.total_checks = len(checks)
        report.passed_checks = sum(1 for c in checks if c.passed)
        return report

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_no_nan(
        self, f: h5py.File, key: str, check_name: str,
    ) -> ValidationCheck:
        if key not in f:
            return ValidationCheck(name=check_name, passed=True,
                                   detail=f"{key} not present (skipped).")
        data = np.asarray(f[key][:], dtype=np.float64)
        has_nan = np.any(~np.isfinite(data))
        return ValidationCheck(
            name=check_name,
            passed=not has_nan,
            detail=f"{key}: {'OK' if not has_nan else 'CONTAINS NaN/Inf'}"
        )

    def _check_variance(
        self, f: h5py.File, key: str,
    ) -> ValidationCheck:
        data = np.asarray(f[key][:], dtype=np.float64)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        var = np.var(data, axis=0)
        zero_var = np.sum(var < self.variance_epsilon)
        check_name = f"non_zero_variance/{key}"
        return ValidationCheck(
            name="non_zero_variance",
            passed=zero_var == 0,
            detail=f"{key}: {zero_var}/{var.shape[0]} dims with zero variance"
            if zero_var > 0 else f"{key}: OK"
        )

    def _check_camera(self, f: h5py.File) -> ValidationCheck:
        if "rgb" not in f:
            return ValidationCheck(
                name="camera_fresh", passed=True,
                detail="No camera data (skipped).",
            )
        rgb = np.asarray(f["rgb"][:], dtype=np.uint8)
        # Check first 10 frames: if any have non-zero pixels, camera is OK
        sample = rgb[:min(10, rgb.shape[0])]
        all_zero = all(np.count_nonzero(frame) == 0 for frame in sample)
        return ValidationCheck(
            name="camera_fresh",
            passed=not all_zero,
            detail="Camera frames OK" if not all_zero
            else "All camera frames are zero (camera failure).",
        )

    def _check_min_frames(self, f: h5py.File) -> ValidationCheck:
        n_frames = f["arm_qpos"].shape[0] if "arm_qpos" in f else 0
        ok = n_frames >= self.min_frames
        return ValidationCheck(
            name="min_frames",
            passed=ok,
            detail=f"{n_frames} frames (min={self.min_frames})"
        )

    def _check_duplicate_frames(self, f: h5py.File) -> ValidationCheck:
        """Check for consecutive identical frames (indicates stuck sensor)."""
        if "arm_qpos" not in f or "action_arm_joint" not in f:
            return ValidationCheck(
                name="no_duplicate_frames", passed=True,
                detail="No obs/action data (skipped).",
            )
        obs = np.asarray(f["arm_qpos"][:], dtype=np.float64)
        act = np.asarray(f["action_arm_joint"][:], dtype=np.float64)
        if len(obs) < 2:
            return ValidationCheck(
                name="no_duplicate_frames", passed=True,
                detail="Too few frames for duplicate check.",
            )
        obs_diff = np.sum(np.abs(np.diff(obs, axis=0)), axis=1)
        act_diff = np.sum(np.abs(np.diff(act, axis=0)), axis=1)
        n_dup_obs = int(np.sum(obs_diff < self.duplicate_epsilon))
        n_dup_act = int(np.sum(act_diff < self.duplicate_epsilon))
        total_dup = max(n_dup_obs, n_dup_act)
        ok = total_dup == 0
        return ValidationCheck(
            name="no_duplicate_frames",
            passed=ok,
            detail=f"{total_dup} duplicate frames"
            if not ok else "No duplicate frames."
        )

    def _check_timestamp_monotonicity(self, f: h5py.File) -> ValidationCheck:
        """Check that timestamps never regress (non-decreasing).

        Duplicates are allowed: forward-filled grid slots (overrun ticks) share
        the previous raw timestamp by design — only a backwards jump is a fault.
        """
        if "timestamp" not in f:
            return ValidationCheck(
                name="timestamp_monotonic", passed=True,
                detail="No timestamp data (skipped).",
            )
        ts = np.asarray(f["timestamp"][:], dtype=np.float64)
        if len(ts) < 2:
            return ValidationCheck(
                name="timestamp_monotonic", passed=True,
                detail="Too few timestamps for check.",
            )
        diffs = np.diff(ts)
        n_regressions = int(np.sum(diffs < 0))
        n_duplicates = int(np.sum(diffs == 0))
        ok = n_regressions == 0
        return ValidationCheck(
            name="timestamp_monotonic",
            passed=ok,
            detail=f"{n_regressions} backwards timestamps"
            if not ok else f"Timestamps non-decreasing ({n_duplicates} forward-filled duplicates)."
        )

    def _check_camera_stall(self, f: h5py.File) -> ValidationCheck:
        """Check /flag_camera_fresh (schema v6+): frozen/forward-filled camera data."""
        if "flag_camera_fresh" not in f or "rgb" not in f:
            return ValidationCheck(
                name="camera_stall", passed=True,
                detail="No freshness flag / camera data (skipped).",
            )
        fresh = np.asarray(f["flag_camera_fresh"][:], dtype=bool)
        stale_frac = 1.0 - float(fresh.mean()) if fresh.size else 0.0
        ok = stale_frac <= 0.10
        return ValidationCheck(
            name="camera_stall",
            passed=ok,
            detail=f"{stale_frac:.1%} of frames have stale camera data"
            + ("" if ok else " (>10% — camera stalled mid-episode)"),
        )

    # ------------------------------------------------------------------
    # Batch validation
    # ------------------------------------------------------------------

    def validate_directory(
        self, data_dir: str | Path,
    ) -> list[ValidationReport]:
        """Validate all episode_*.h5 files in a directory."""
        data_dir = Path(data_dir)
        h5_paths = sorted(data_dir.glob("episode_*.h5"))
        reports = []
        for h5_path in h5_paths:
            report = self.validate(h5_path)
            reports.append(report)
            status = "PASS" if report.is_valid else "FAIL"
            print(f"  [{status}] {h5_path.name}: "
                  f"{report.passed_checks}/{report.total_checks} checks")
        return reports

    @staticmethod
    def save_reports(
        reports: list[ValidationReport],
        output_path: str | Path,
    ) -> None:
        """Save validation reports as JSON."""
        output = []
        for r in reports:
            output.append({
                "episode_path": r.episode_path,
                "is_valid": r.is_valid,
                "passed_checks": r.passed_checks,
                "total_checks": r.total_checks,
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail}
                    for c in r.checks
                ],
                "metadata": r.episode_metadata,
            })
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)

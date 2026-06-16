"""DataValidator — validate HDF5 episode structure and content."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.recording.quality_flags import ALL_GOOD_MASK


@dataclass
class ValidationReport:
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [f"ValidationReport({status})"]
        for e in self.errors:
            lines.append(f"  [ERROR] {e}")
        for w in self.warnings:
            lines.append(f"  [WARN]  {w}")
        return "\n".join(lines)


class DataValidator:
    """Validates an HDF5 episode file for correctness and consistency."""

    # Expected joint ranges for sanity checks
    ARM_JOINT_RANGE = (-3.15, 3.15)       # rad, xArm7
    HAND_JOINT_RANGE = (-0.5, 2.5)         # rad, XHand
    HAND_CURRENT_MAX = 1000.0               # mA
    HAND_TEMP_MAX = 80.0                    # °C

    def __init__(self) -> None:
        pass

    def validate(self, episode_path: str) -> ValidationReport:
        path = Path(episode_path)
        if not path.exists():
            return ValidationReport(
                passed=False, errors=[f"File not found: {episode_path}"]
            )

        errors: list[str] = []
        warnings: list[str] = []

        try:
            with h5py.File(str(path), "r") as f:
                self._check_structure(f, errors)
                if errors:
                    return ValidationReport(passed=False, errors=errors, warnings=warnings)

                self._check_nan_inf(f, errors)
                self._check_shapes(f, errors)
                self._check_timestamps(f, warnings)
                self._check_joint_ranges(f, warnings)
                self._check_current_temperature(f, warnings)
                self._check_quality_flags(f, warnings)
        except Exception as e:
            errors.append(f"Failed to open/read HDF5: {e}")
            return ValidationReport(passed=False, errors=errors, warnings=warnings)

        return ValidationReport(
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def _check_structure(self, f: h5py.File, errors: list[str]) -> None:
        required_groups = ["obs", "action", "vr", "meta"]
        for g in required_groups:
            if g not in f:
                errors.append(f"Missing required group: /{g}")

        required_datasets = [
            "obs/arm_qpos", "obs/hand_qpos",
            "action/arm_qpos", "action/hand_qpos",
            "vr/wrist_pos", "vr/landmarks",
            "quality_flags",
        ]
        for ds in required_datasets:
            if ds not in f:
                errors.append(f"Missing required dataset: /{ds}")

    def _check_nan_inf(self, f: h5py.File, errors: list[str]) -> None:
        datasets_to_check = [
            "obs/arm_qpos", "obs/arm_qvel", "obs/arm_tau",
            "obs/eef_pos", "obs/eef_quat",
            "obs/hand_qpos", "obs/hand_current",
            "action/arm_qpos", "action/hand_qpos",
            "vr/wrist_pos", "vr/wrist_quat",
        ]
        for key in datasets_to_check:
            if key not in f:
                continue
            data = np.asarray(f[key])
            if np.any(np.isnan(data)):
                errors.append(f"NaN values in /{key}")
            if np.any(np.isinf(data)):
                errors.append(f"Inf values in /{key}")

    def _check_shapes(self, f: h5py.File, errors: list[str]) -> None:
        expected = {
            "obs/arm_qpos": 7,
            "obs/arm_qvel": 7,
            "obs/arm_tau": 7,
            "obs/eef_pos": 3,
            "obs/eef_quat": 4,
            "obs/hand_qpos": 12,
            "obs/hand_current": 12,
            "action/arm_qpos": 7,
            "action/hand_qpos": 12,
            "vr/wrist_pos": 3,
            "vr/wrist_quat": 4,
            "vr/landmarks": (21, 3),
        }

        # Determine T
        if "quality_flags" in f:
            t = np.asarray(f["quality_flags"]).shape[0]
        else:
            return

        for key, expected_dim in expected.items():
            if key not in f:
                continue
            shape = np.asarray(f[key]).shape
            if shape[0] != t:
                errors.append(
                    f"Shape mismatch for /{key}: dim0={shape[0]}, expected {t}"
                )
            if isinstance(expected_dim, tuple):
                if shape[1:] != expected_dim:
                    errors.append(
                        f"Shape mismatch for /{key}: dims={shape[1:]}, expected {expected_dim}"
                    )
            else:
                if shape[-1] != expected_dim:
                    errors.append(
                        f"Shape mismatch for /{key}: last_dim={shape[-1]}, expected {expected_dim}"
                    )

    def _check_timestamps(self, f: h5py.File, warnings: list[str]) -> None:
        # Check for non-monotonic sequence in VR source_ts if available
        pass

    def _check_joint_ranges(self, f: h5py.File, warnings: list[str]) -> None:
        for key, (lo, hi) in [
            ("obs/arm_qpos", self.ARM_JOINT_RANGE),
            ("action/arm_qpos", self.ARM_JOINT_RANGE),
        ]:
            if key not in f:
                continue
            data = np.asarray(f[key])
            below = data < lo
            above = data > hi
            n_below = int(np.any(below, axis=1).sum())
            n_above = int(np.any(above, axis=1).sum())
            if n_below > 0:
                warnings.append(
                    f"/{key}: {n_below} frames below {lo} rad"
                )
            if n_above > 0:
                warnings.append(
                    f"/{key}: {n_above} frames above {hi} rad"
                )

        for key, (lo, hi) in [
            ("obs/hand_qpos", self.HAND_JOINT_RANGE),
            ("action/hand_qpos", self.HAND_JOINT_RANGE),
        ]:
            if key not in f:
                continue
            data = np.asarray(f[key])
            below = data < lo
            above = data > hi
            n_below = int(np.any(below, axis=1).sum())
            n_above = int(np.any(above, axis=1).sum())
            if n_below > 0:
                warnings.append(
                    f"/{key}: {n_below} frames below {lo} rad"
                )
            if n_above > 0:
                warnings.append(
                    f"/{key}: {n_above} frames above {hi} rad"
                )

    def _check_current_temperature(self, f: h5py.File, warnings: list[str]) -> None:
        if "obs/hand_current" in f:
            cur = np.asarray(f["obs/hand_current"])
            if np.any(cur > self.HAND_CURRENT_MAX):
                n = int(np.any(cur > self.HAND_CURRENT_MAX, axis=1).sum())
                warnings.append(
                    f"/obs/hand_current: {n} frames exceed {self.HAND_CURRENT_MAX} mA"
                )

        if "obs/hand_temperature" in f:
            temp = np.asarray(f["obs/hand_temperature"])
            if np.any(temp > self.HAND_TEMP_MAX):
                n = int(np.any(temp > self.HAND_TEMP_MAX, axis=1).sum())
                warnings.append(
                    f"/obs/hand_temperature: {n} frames exceed {self.HAND_TEMP_MAX} °C"
                )

    def _check_quality_flags(self, f: h5py.File, warnings: list[str]) -> None:
        if "quality_flags" not in f:
            return
        qf = np.asarray(f["quality_flags"], dtype=np.uint16).ravel()
        n_total = qf.shape[0]
        n_good = int(((qf & np.uint16(ALL_GOOD_MASK)) == np.uint16(ALL_GOOD_MASK)).sum())
        ratio = n_good / n_total if n_total > 0 else 0.0

        if ratio < 0.5:
            warnings.append(
                f"Only {n_good}/{n_total} frames ({ratio:.1%}) have all quality flags set"
            )

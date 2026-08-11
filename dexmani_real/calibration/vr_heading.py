"""Robust VR heading estimation and transactional calibration persistence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np

_QUATERNION_NORM_EPS = 1e-12
_NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class HeadingQualityGate:
    """Acceptance thresholds for a stationary heading calibration."""

    min_inliers: int = 30
    min_inlier_ratio: float = 0.80
    min_resultant_length: float = 0.996
    max_std_deg: float = 5.0
    max_deviation_deg: float = 12.0
    robust_sigma: float = 3.5
    robust_outlier_floor_deg: float = 3.0
    robust_outlier_cap_deg: float = 45.0
    mean_resultant_epsilon: float = 1e-6
    horizontal_norm_epsilon: float = 1e-6
    excellent_min_inlier_ratio: float = 0.95
    excellent_min_resultant_length: float = 0.999
    excellent_max_std_deg: float = 2.0
    excellent_max_deviation_deg: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.min_inliers, bool) or not isinstance(self.min_inliers, Integral) or self.min_inliers <= 0:
            raise ValueError("min_inliers must be a positive integer")
        ratios = (self.min_inlier_ratio, self.excellent_min_inlier_ratio)
        resultants = (
            self.min_resultant_length,
            self.excellent_min_resultant_length,
            self.mean_resultant_epsilon,
        )
        if any(not np.isfinite(value) or not 0.0 < value <= 1.0 for value in ratios + resultants):
            raise ValueError("ratio and resultant thresholds must be finite and in (0, 1]")
        positive = (
            self.max_std_deg,
            self.max_deviation_deg,
            self.robust_sigma,
            self.robust_outlier_floor_deg,
            self.robust_outlier_cap_deg,
            self.horizontal_norm_epsilon,
            self.excellent_max_std_deg,
            self.excellent_max_deviation_deg,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("angular, robust, and projection thresholds must be finite and positive")
        if self.robust_outlier_floor_deg > self.robust_outlier_cap_deg:
            raise ValueError("robust_outlier_floor_deg must not exceed robust_outlier_cap_deg")
        if self.max_std_deg > self.max_deviation_deg:
            raise ValueError("max_std_deg must not exceed max_deviation_deg")
        if self.excellent_min_inlier_ratio < self.min_inlier_ratio:
            raise ValueError("excellent_min_inlier_ratio must be at least min_inlier_ratio")
        if self.excellent_min_resultant_length < self.min_resultant_length:
            raise ValueError("excellent_min_resultant_length must be at least min_resultant_length")
        if self.excellent_max_std_deg > self.max_std_deg:
            raise ValueError("excellent_max_std_deg must not exceed max_std_deg")
        if self.excellent_max_deviation_deg > self.max_deviation_deg:
            raise ValueError("excellent_max_deviation_deg must not exceed max_deviation_deg")


DEFAULT_HEADING_QUALITY_GATE = HeadingQualityGate()


@dataclass(frozen=True)
class HeadingEstimate:
    """Heading estimate plus every metric used by the acceptance decision."""

    gate: HeadingQualityGate
    total_count: int
    valid_count: int
    inlier_count: int
    inlier_ratio: float
    resultant_length: float | None
    std_deg: float | None
    max_deviation_deg: float | None
    theta_rad: float | None
    mean_forward_xy: np.ndarray | None
    rotation_vr_to_robot: np.ndarray | None
    inlier_mask: np.ndarray
    grade: str
    accepted: bool
    reasons: tuple[str, ...]

    @property
    def theta_deg(self) -> float | None:
        return None if self.theta_rad is None else float(np.rad2deg(self.theta_rad))

    @property
    def quality_text(self) -> str:
        if self.resultant_length is None or self.std_deg is None or self.max_deviation_deg is None:
            return self.grade
        return (
            f"{self.grade} (R={self.resultant_length:.4f}, "
            f"σ={self.std_deg:.1f}°, max={self.max_deviation_deg:.1f}°)"
        )


def forward_from_quat_wxyz(quat_wxyz: object) -> np.ndarray:
    """Return local FLU +X expressed in the parent frame."""
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    if quat.shape != (4,) or not np.all(np.isfinite(quat)):
        raise ValueError("quat_wxyz must be a finite array with shape (4,)")
    scale = float(np.max(np.abs(quat)))
    if scale < _QUATERNION_NORM_EPS:
        raise ValueError("quat_wxyz norm must be finite and nonzero")
    scaled_quat = quat / scale
    w, x, y, z = scaled_quat / np.linalg.norm(scaled_quat)
    return np.array(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ],
        dtype=np.float64,
    )


def reference_sample_from_vr_frame(
    data: np.ndarray,
    ring_sequence: int,
    reference: str,
) -> tuple[int, int, np.ndarray]:
    """Extract the source identity, receive time, and quaternion for one reference."""
    if reference not in {"head", "wrist"}:
        raise ValueError("reference must be 'head' or 'wrist'")
    if data.shape != (1,) or data.dtype.names is None:
        raise ValueError("VR frame must be a one-record structured array")
    record = data[0]
    if reference == "head":
        sequence = int(record["head_sequence_id"])
        timestamp_ns = int(record["head_recv_ts_ns"])
        quaternion = record["head_quat_wxyz"]
    else:
        sequence = int(ring_sequence)
        timestamp_ns = int(record["local_recv_ns"])
        quaternion = record["wrist_quat_wxyz"]
    return sequence, timestamp_ns, np.array(quaternion, dtype=np.float64, copy=True)


def timestamp_is_fresh(timestamp_ns: int, now_ns: int, max_age_s: float) -> bool:
    """Return whether a host-monotonic source time is causal and recent."""
    if not np.isfinite(max_age_s) or max_age_s <= 0.0:
        raise ValueError("max_age_s must be finite and positive")
    age_ns = int(now_ns) - int(timestamp_ns)
    return int(timestamp_ns) > 0 and 0 <= age_ns <= int(max_age_s * _NS_PER_SECOND)


def _wrapped_residuals(angles_rad: np.ndarray, center_rad: float) -> np.ndarray:
    return np.angle(np.exp(1j * (angles_rad - center_rad)))


def _planar_angles(samples: np.ndarray, gate: HeadingQualityGate) -> tuple[np.ndarray, np.ndarray]:
    finite_mask = np.all(np.isfinite(samples), axis=1)
    horizontal_norm = np.zeros(samples.shape[0], dtype=np.float64)
    horizontal_norm[finite_mask] = np.linalg.norm(samples[finite_mask, :2], axis=1)
    valid_mask = finite_mask & np.isfinite(horizontal_norm) & (horizontal_norm >= gate.horizontal_norm_epsilon)
    unit_xy = samples[valid_mask, :2] / horizontal_norm[valid_mask, None]
    return valid_mask, np.arctan2(unit_xy[:, 1], unit_xy[:, 0])


def _robust_inliers(
    angles_rad: np.ndarray,
    gate: HeadingQualityGate,
) -> tuple[np.ndarray | None, float | None, str | None]:
    seed_vector = complex(np.mean(np.exp(1j * angles_rad)))
    seed_resultant = float(abs(seed_vector))
    if not np.isfinite(seed_resultant) or seed_resultant < gate.mean_resultant_epsilon:
        return (
            None,
            seed_resultant if np.isfinite(seed_resultant) else None,
            ("circular mean is degenerate (opposed or uniformly distributed headings)"),
        )

    seed_rad = float(np.angle(seed_vector))
    absolute_residuals = np.abs(_wrapped_residuals(angles_rad, seed_rad))
    residual_median = float(np.median(absolute_residuals))
    residual_mad = float(np.median(np.abs(absolute_residuals - residual_median)))
    robust_scale = 1.4826 * residual_mad
    limit_deg = np.clip(
        np.rad2deg(residual_median + gate.robust_sigma * robust_scale),
        gate.robust_outlier_floor_deg,
        gate.robust_outlier_cap_deg,
    )
    return absolute_residuals <= np.deg2rad(limit_deg), seed_resultant, None


def _poor_estimate(
    *,
    gate: HeadingQualityGate,
    total_count: int,
    valid_count: int,
    inlier_mask: np.ndarray,
    reasons: tuple[str, ...],
    resultant_length: float | None = None,
    std_deg: float | None = None,
    max_deviation_deg: float | None = None,
) -> HeadingEstimate:
    inlier_count = int(np.count_nonzero(inlier_mask))
    return HeadingEstimate(
        gate=gate,
        total_count=total_count,
        valid_count=valid_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / total_count if total_count else 0.0,
        resultant_length=resultant_length,
        std_deg=std_deg,
        max_deviation_deg=max_deviation_deg,
        theta_rad=None,
        mean_forward_xy=None,
        rotation_vr_to_robot=None,
        inlier_mask=inlier_mask,
        grade="POOR",
        accepted=False,
        reasons=reasons,
    )


def _quality_reasons(
    gate: HeadingQualityGate,
    *,
    inlier_count: int,
    inlier_ratio: float,
    resultant_length: float,
    std_deg: float,
    max_deviation_deg: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if inlier_count < gate.min_inliers:
        reasons.append(f"inliers {inlier_count} < {gate.min_inliers}")
    if inlier_ratio < gate.min_inlier_ratio:
        reasons.append(f"inlier ratio {inlier_ratio:.3f} < {gate.min_inlier_ratio:.3f}")
    if resultant_length < gate.min_resultant_length:
        reasons.append(f"resultant length {resultant_length:.4f} < {gate.min_resultant_length:.4f}")
    if std_deg > gate.max_std_deg:
        reasons.append(f"angular std {std_deg:.2f}° > {gate.max_std_deg:.2f}°")
    if max_deviation_deg > gate.max_deviation_deg:
        reasons.append(f"max deviation {max_deviation_deg:.2f}° > {gate.max_deviation_deg:.2f}°")
    return tuple(reasons)


def _heading_rotation(theta_rad: float) -> tuple[np.ndarray, np.ndarray]:
    mean_forward_xy = np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.float64)
    cos_theta, sin_theta = (float(value) for value in mean_forward_xy)
    rotation_vr_to_robot = np.array(
        [
            [cos_theta, sin_theta, 0.0],
            [-sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return mean_forward_xy, rotation_vr_to_robot


def estimate_heading(
    forwards: object,
    gate: HeadingQualityGate = DEFAULT_HEADING_QUALITY_GATE,
) -> HeadingEstimate:
    """Estimate planar heading from 3D forward vectors using circular statistics.

    Non-finite and near-vertical vectors are invalid observations. They remain
    in the denominator of the inlier-ratio gate so missing tracking cannot make
    a calibration appear artificially precise.
    """
    samples = np.asarray(forwards, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("forwards must have shape (N, 3)")

    total_count = int(samples.shape[0])
    full_inlier_mask = np.zeros(total_count, dtype=bool)
    valid_mask, angles_rad = _planar_angles(samples, gate)
    valid_count = int(np.count_nonzero(valid_mask))
    if valid_count == 0:
        return _poor_estimate(
            gate=gate,
            total_count=total_count,
            valid_count=0,
            inlier_mask=full_inlier_mask,
            reasons=("no finite forward vector has a usable horizontal projection",),
        )

    valid_inlier_mask, seed_resultant, filtering_error = _robust_inliers(angles_rad, gate)
    if valid_inlier_mask is None:
        return _poor_estimate(
            gate=gate,
            total_count=total_count,
            valid_count=valid_count,
            inlier_mask=full_inlier_mask,
            reasons=(filtering_error or "robust heading filtering failed",),
            resultant_length=seed_resultant,
        )

    full_inlier_mask[valid_mask] = valid_inlier_mask
    inlier_angles = angles_rad[valid_inlier_mask]
    if inlier_angles.size == 0:
        return _poor_estimate(
            gate=gate,
            total_count=total_count,
            valid_count=valid_count,
            inlier_mask=full_inlier_mask,
            reasons=("robust filtering left no heading inliers",),
        )

    mean_vector = complex(np.mean(np.exp(1j * inlier_angles)))
    resultant_length = float(abs(mean_vector))
    if not np.isfinite(resultant_length) or resultant_length < gate.mean_resultant_epsilon:
        return _poor_estimate(
            gate=gate,
            total_count=total_count,
            valid_count=valid_count,
            inlier_mask=full_inlier_mask,
            reasons=("inlier circular mean is degenerate",),
            resultant_length=resultant_length if np.isfinite(resultant_length) else None,
        )

    theta_rad = float(np.angle(mean_vector))
    residuals_rad = _wrapped_residuals(inlier_angles, theta_rad)
    std_deg = float(np.rad2deg(np.sqrt(np.mean(np.square(residuals_rad)))))
    max_deviation_deg = float(np.rad2deg(np.max(np.abs(residuals_rad))))
    inlier_count = int(inlier_angles.size)
    inlier_ratio = inlier_count / total_count

    reasons = _quality_reasons(
        gate,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        resultant_length=resultant_length,
        std_deg=std_deg,
        max_deviation_deg=max_deviation_deg,
    )
    accepted = not reasons
    excellent = (
        accepted
        and inlier_ratio >= gate.excellent_min_inlier_ratio
        and resultant_length >= gate.excellent_min_resultant_length
        and std_deg <= gate.excellent_max_std_deg
        and max_deviation_deg <= gate.excellent_max_deviation_deg
    )
    mean_forward_xy, rotation_vr_to_robot = _heading_rotation(theta_rad)
    return HeadingEstimate(
        gate=gate,
        total_count=total_count,
        valid_count=valid_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_ratio,
        resultant_length=resultant_length,
        std_deg=std_deg,
        max_deviation_deg=max_deviation_deg,
        theta_rad=theta_rad,
        mean_forward_xy=mean_forward_xy,
        rotation_vr_to_robot=rotation_vr_to_robot,
        inlier_mask=full_inlier_mask,
        grade="excellent" if excellent else "good" if accepted else "POOR",
        accepted=accepted,
        reasons=reasons,
    )


def build_heading_config(estimate: HeadingEstimate, reference: str) -> dict[str, Any]:
    """Build the backward-compatible JSON payload for an accepted estimate."""
    if reference not in {"head", "wrist"}:
        raise ValueError("reference must be 'head' or 'wrist'")
    if (
        not estimate.accepted
        or estimate.theta_deg is None
        or estimate.mean_forward_xy is None
        or estimate.rotation_vr_to_robot is None
        or estimate.resultant_length is None
        or estimate.std_deg is None
        or estimate.max_deviation_deg is None
    ):
        raise ValueError("refusing to serialize a rejected heading estimate")
    values = np.concatenate(
        (
            estimate.mean_forward_xy,
            estimate.rotation_vr_to_robot.reshape(-1),
            np.array(
                [
                    estimate.theta_deg,
                    estimate.resultant_length,
                    estimate.std_deg,
                    estimate.max_deviation_deg,
                ]
            ),
        )
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("heading estimate contains non-finite values")

    return {
        "description": "Fixed VR-to-robot heading transform",
        "T_vr_to_robot": estimate.rotation_vr_to_robot.tolist(),
        "theta_deg": estimate.theta_deg,
        "convention": "R_z(-theta) maps VR FLU forward -> robot base +X",
        "ref": reference,
        "quality": estimate.quality_text,
        "quality_grade": estimate.grade,
        "quality_metrics": {
            "inlier_ratio": estimate.inlier_ratio,
            "resultant_length": estimate.resultant_length,
            "std_deg": estimate.std_deg,
            "max_deviation_deg": estimate.max_deviation_deg,
            "thresholds": asdict(estimate.gate),
        },
        "frames": estimate.total_count,
        "valid_frames": estimate.valid_count,
        "inlier_frames": estimate.inlier_count,
        "mean_forward_xy": estimate.mean_forward_xy.tolist(),
    }


def write_json_atomic_with_backup(path: str | Path, payload: Mapping[str, Any]) -> Path | None:
    """Atomically replace *path*, preserving the previous file as a backup."""
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    backup_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)

        if destination.exists():
            backup_path = destination.with_name(f"{destination.name}.bak.{time.time_ns()}.{os.getpid()}")
            shutil.copy2(destination, backup_path)
        os.replace(temporary_path, destination)
        temporary_path = None
        return backup_path
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

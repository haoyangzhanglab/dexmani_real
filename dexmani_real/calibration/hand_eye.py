"""Pure eye-to-hand calibration, quality checks, and atomic persistence."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

HAND_EYE_METHODS: dict[str, int] = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass(frozen=True)
class HandEyeQualityLimits:
    """Acceptance thresholds for one camera-calibration experiment."""

    min_samples: int = 10
    max_position_rms_mm: float = 5.0
    max_position_error_mm: float = 10.0
    max_rotation_rms_deg: float = 3.0
    max_rotation_error_deg: float = 6.0
    min_translation_span_m: float = 0.05
    min_rotation_span_deg: float = 20.0
    min_rotation_axis_ratio: float = 0.1

    def __post_init__(self) -> None:
        values = (
            self.max_position_rms_mm,
            self.max_position_error_mm,
            self.max_rotation_rms_deg,
            self.max_rotation_error_deg,
            self.min_translation_span_m,
            self.min_rotation_span_deg,
            self.min_rotation_axis_ratio,
        )
        if self.min_samples < 3 or not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("hand-eye quality thresholds must be finite and positive")
        if not 0.0 < self.min_rotation_axis_ratio <= 1.0:
            raise ValueError("min_rotation_axis_ratio must be in (0, 1]")


@dataclass(frozen=True)
class MotionExcitation:
    translation_span_m: float
    rotation_span_deg: float
    rotation_axis_ratio: float

    def __post_init__(self) -> None:
        values = (self.translation_span_m, self.rotation_span_deg, self.rotation_axis_ratio)
        if not all(np.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("motion-excitation metrics must be finite and non-negative")
        if self.rotation_axis_ratio > 1.0:
            raise ValueError("rotation_axis_ratio must not exceed 1")


@dataclass(frozen=True)
class HandEyeQuality:
    position_rms_mm: float
    position_max_mm: float
    rotation_rms_deg: float
    rotation_max_deg: float
    accepted: bool
    rejection_reasons: tuple[str, ...]


@dataclass(frozen=True)
class HandEyeCandidate:
    method: str
    transform_base_camera: np.ndarray
    position_errors_mm: np.ndarray
    rotation_errors_deg: np.ndarray
    quality: HandEyeQuality
    score: float


@dataclass(frozen=True)
class HandEyeSelection:
    best: HandEyeCandidate
    candidates: tuple[HandEyeCandidate, ...]
    failures: tuple[tuple[str, str], ...]
    excitation: MotionExcitation


def _finite_vectors(values: list[np.ndarray], name: str, *, minimum: int) -> list[np.ndarray]:
    if len(values) < minimum:
        raise ValueError(f"{name} needs at least {minimum} samples")
    result = [np.asarray(value, dtype=np.float64) for value in values]
    if any(value.shape != (3,) for value in result):
        raise ValueError(f"{name} samples must have shape (3,)")
    if not all(np.all(np.isfinite(value)) for value in result):
        raise ValueError(f"{name} samples must be finite")
    return result


def _validate_rotation_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite (3, 3) matrix")
    if not np.allclose(value.T @ value, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(value), 1.0, atol=1e-5):
        raise ValueError(f"{name} is not a proper rotation matrix")
    return value


def validate_transform(transform: np.ndarray, name: str = "transform") -> np.ndarray:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite (4, 4) matrix")
    if not np.allclose(value[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous bottom row")
    _validate_rotation_matrix(value[:3, :3], f"{name} rotation")
    return value


def measure_motion_excitation(
    ee_positions_base_m: list[np.ndarray],
    ee_rpy_base_rad: list[np.ndarray],
) -> MotionExcitation:
    positions = np.stack(_finite_vectors(ee_positions_base_m, "EEF position", minimum=3))
    rpy = np.stack(_finite_vectors(ee_rpy_base_rad, "EEF orientation", minimum=3))
    translation_span_m = float(np.max(np.linalg.norm(positions[:, None] - positions[None, :], axis=2)))

    rotations = Rotation.from_euler("xyz", rpy)
    relative = rotations[0].inv() * rotations
    rotation_span_deg = float(np.rad2deg(np.max(relative.magnitude())))
    rotvec = relative.as_rotvec()
    angles = np.linalg.norm(rotvec, axis=1)
    axes = rotvec[angles > np.deg2rad(1.0)] / angles[angles > np.deg2rad(1.0), None]
    if len(axes) < 2:
        axis_ratio = 0.0
    else:
        singular_values = np.linalg.svd(axes, compute_uv=False)
        axis_ratio = float(singular_values[1] / max(singular_values[0], 1e-12))
    return MotionExcitation(translation_span_m, rotation_span_deg, axis_ratio)


def calibrate_eye_to_hand(
    ee_positions_base_m: list[np.ndarray],
    ee_rpy_base_rad: list[np.ndarray],
    marker_rvecs_camera: list[np.ndarray],
    marker_positions_camera_m: list[np.ndarray],
    *,
    method: int = cv2.CALIB_HAND_EYE_PARK,
) -> np.ndarray:
    """Return ``T_base_camera`` for a fixed camera and an EEF-mounted marker."""
    sample_count = len(ee_positions_base_m)
    positions = _finite_vectors(ee_positions_base_m, "EEF position", minimum=3)
    orientations = _finite_vectors(ee_rpy_base_rad, "EEF orientation", minimum=3)
    marker_rvecs = _finite_vectors(marker_rvecs_camera, "marker rotation", minimum=3)
    marker_positions = _finite_vectors(marker_positions_camera_m, "marker position", minimum=3)
    if not all(len(values) == sample_count for values in (orientations, marker_rvecs, marker_positions)):
        raise ValueError("hand-eye sample lists must have equal lengths")

    rotations_ee_base = [Rotation.from_euler("xyz", value).as_matrix() for value in orientations]
    rotations_base_ee = [matrix.T for matrix in rotations_ee_base]
    positions_base_ee = [(-matrix.T @ value).reshape(3, 1) for matrix, value in zip(rotations_ee_base, positions)]
    rotations_marker_camera = [cv2.Rodrigues(value)[0] for value in marker_rvecs]
    positions_marker_camera = [value.reshape(3, 1) for value in marker_positions]

    rotation_camera_base, position_camera_base = cv2.calibrateHandEye(
        rotations_base_ee,
        positions_base_ee,
        rotations_marker_camera,
        positions_marker_camera,
        method=method,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_camera_base
    transform[:3, 3] = np.asarray(position_camera_base, dtype=np.float64).reshape(3)
    return validate_transform(transform, "T_base_camera")


def compute_marker_consistency(
    transform_base_camera: np.ndarray,
    ee_positions_base_m: list[np.ndarray],
    ee_rpy_base_rad: list[np.ndarray],
    marker_rvecs_camera: list[np.ndarray],
    marker_positions_camera_m: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-sample position and rotation closure errors."""
    transform_base_camera = validate_transform(transform_base_camera, "T_base_camera")
    sample_count = len(ee_positions_base_m)
    positions = _finite_vectors(ee_positions_base_m, "EEF position", minimum=3)
    orientations = _finite_vectors(ee_rpy_base_rad, "EEF orientation", minimum=3)
    marker_rvecs = _finite_vectors(marker_rvecs_camera, "marker rotation", minimum=3)
    marker_positions = _finite_vectors(marker_positions_camera_m, "marker position", minimum=3)
    if not all(len(values) == sample_count for values in (orientations, marker_rvecs, marker_positions)):
        raise ValueError("hand-eye sample lists must have equal lengths")

    closures: list[np.ndarray] = []
    for ee_position, ee_rpy, marker_rvec, marker_position in zip(
        positions, orientations, marker_rvecs, marker_positions
    ):
        transform_base_ee = np.eye(4, dtype=np.float64)
        transform_base_ee[:3, :3] = Rotation.from_euler("xyz", ee_rpy).as_matrix()
        transform_base_ee[:3, 3] = ee_position
        transform_camera_marker = np.eye(4, dtype=np.float64)
        transform_camera_marker[:3, :3] = cv2.Rodrigues(marker_rvec)[0]
        transform_camera_marker[:3, 3] = marker_position
        closure = np.linalg.inv(transform_base_ee) @ transform_base_camera @ transform_camera_marker
        closures.append(validate_transform(closure, "T_ee_marker"))

    closure_positions = np.stack([value[:3, 3] for value in closures])
    mean_position = np.mean(closure_positions, axis=0)
    position_errors_mm = np.linalg.norm(closure_positions - mean_position, axis=1) * 1000.0
    closure_rotations = Rotation.from_matrix(np.stack([value[:3, :3] for value in closures]))
    mean_rotation = closure_rotations.mean()
    rotation_errors_deg = np.rad2deg((mean_rotation.inv() * closure_rotations).magnitude())
    if not np.all(np.isfinite(position_errors_mm)) or not np.all(np.isfinite(rotation_errors_deg)):
        raise ValueError("hand-eye closure residuals are non-finite")
    return position_errors_mm, rotation_errors_deg


def assess_quality(
    position_errors_mm: np.ndarray,
    rotation_errors_deg: np.ndarray,
    excitation: MotionExcitation,
    limits: HandEyeQualityLimits,
) -> HandEyeQuality:
    position = np.asarray(position_errors_mm, dtype=np.float64)
    rotation = np.asarray(rotation_errors_deg, dtype=np.float64)
    if position.ndim != 1 or rotation.shape != position.shape or len(position) < limits.min_samples:
        raise ValueError("quality residuals must be equal-length vectors with enough samples")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(rotation)):
        raise ValueError("quality residuals must be finite")
    if np.any(position < 0.0) or np.any(rotation < 0.0):
        raise ValueError("quality residuals must be non-negative")

    position_rms = float(np.sqrt(np.mean(np.square(position))))
    position_max = float(np.max(position))
    rotation_rms = float(np.sqrt(np.mean(np.square(rotation))))
    rotation_max = float(np.max(rotation))
    reasons: list[str] = []
    checks = (
        (position_rms > limits.max_position_rms_mm, f"position RMS {position_rms:.1f}mm"),
        (position_max > limits.max_position_error_mm, f"position max {position_max:.1f}mm"),
        (rotation_rms > limits.max_rotation_rms_deg, f"rotation RMS {rotation_rms:.1f}deg"),
        (rotation_max > limits.max_rotation_error_deg, f"rotation max {rotation_max:.1f}deg"),
        (
            excitation.translation_span_m < limits.min_translation_span_m,
            f"translation span {excitation.translation_span_m:.3f}m",
        ),
        (
            excitation.rotation_span_deg < limits.min_rotation_span_deg,
            f"rotation span {excitation.rotation_span_deg:.1f}deg",
        ),
        (
            excitation.rotation_axis_ratio < limits.min_rotation_axis_ratio,
            f"rotation axis ratio {excitation.rotation_axis_ratio:.2f}",
        ),
    )
    reasons.extend(message for failed, message in checks if failed)
    return HandEyeQuality(position_rms, position_max, rotation_rms, rotation_max, not reasons, tuple(reasons))


def select_hand_eye_calibration(
    ee_positions_base_m: list[np.ndarray],
    ee_rpy_base_rad: list[np.ndarray],
    marker_rvecs_camera: list[np.ndarray],
    marker_positions_camera_m: list[np.ndarray],
    *,
    limits: HandEyeQualityLimits = HandEyeQualityLimits(),
) -> HandEyeSelection:
    excitation = measure_motion_excitation(ee_positions_base_m, ee_rpy_base_rad)
    candidates: list[HandEyeCandidate] = []
    failures: list[tuple[str, str]] = []
    for name, method in HAND_EYE_METHODS.items():
        try:
            transform = calibrate_eye_to_hand(
                ee_positions_base_m,
                ee_rpy_base_rad,
                marker_rvecs_camera,
                marker_positions_camera_m,
                method=method,
            )
            position_errors, rotation_errors = compute_marker_consistency(
                transform,
                ee_positions_base_m,
                ee_rpy_base_rad,
                marker_rvecs_camera,
                marker_positions_camera_m,
            )
            quality = assess_quality(position_errors, rotation_errors, excitation, limits)
            score = (
                quality.position_rms_mm / limits.max_position_rms_mm
                + quality.position_max_mm / limits.max_position_error_mm
                + quality.rotation_rms_deg / limits.max_rotation_rms_deg
                + quality.rotation_max_deg / limits.max_rotation_error_deg
            )
            candidates.append(HandEyeCandidate(name, transform, position_errors, rotation_errors, quality, score))
        except (cv2.error, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            failures.append((name, str(exc)))
    if not candidates:
        raise RuntimeError("all hand-eye solvers failed")
    accepted = [candidate for candidate in candidates if candidate.quality.accepted]
    best = min(accepted or candidates, key=lambda candidate: candidate.score)
    return HandEyeSelection(best, tuple(candidates), tuple(failures), excitation)


def update_camera_calibration(
    json_path: Path,
    *,
    serial: str,
    transform_world_camera: np.ndarray,
    capture_metadata: dict[str, Any] | None = None,
) -> tuple[str, Path | None]:
    """Atomically update one serial-numbered camera entry and preserve a backup."""
    transform = validate_transform(transform_world_camera, "T_world_camera")
    if not serial:
        raise ValueError("camera serial must be non-empty")
    rotation = Rotation.from_matrix(transform[:3, :3])
    quat_wxyz = np.roll(rotation.as_quat(), 1)
    position = transform[:3, 3]
    values = np.concatenate((position, quat_wxyz))
    if not np.all(np.isfinite(values)):
        raise ValueError("camera pose must be finite")

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if not isinstance(existing, dict):
            raise ValueError("camera calibration root must be an object")
    else:
        existing = {}

    camera_name = next(
        (name for name, value in existing.items() if isinstance(value, dict) and value.get("serial") == serial),
        None,
    )
    if camera_name is None:
        index = 0
        while f"camera_{index}" in existing:
            index += 1
        camera_name = f"camera_{index}"
    entry: dict[str, Any] = {
        "serial": serial,
        "type": "eye_to_hand",
        "pose": {
            "position": [round(float(value), 6) for value in position],
            "orientation": [round(float(value), 6) for value in quat_wxyz],
        },
    }
    if capture_metadata is not None:
        entry["calibration_capture"] = capture_metadata
    existing[camera_name] = entry

    json_path.parent.mkdir(parents=True, exist_ok=True)
    backup: Path | None = None
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=json_path.parent,
            prefix=f".{json_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            json.dump(existing, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if json_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = json_path.with_suffix(f"{json_path.suffix}.bak.{timestamp}")
            shutil.copy2(json_path, backup)
        os.replace(temp_path, json_path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    return camera_name, backup


__all__ = [
    "HAND_EYE_METHODS",
    "HandEyeCandidate",
    "HandEyeQuality",
    "HandEyeQualityLimits",
    "HandEyeSelection",
    "MotionExcitation",
    "assess_quality",
    "calibrate_eye_to_hand",
    "compute_marker_consistency",
    "measure_motion_excitation",
    "select_hand_eye_calibration",
    "update_camera_calibration",
    "validate_transform",
]

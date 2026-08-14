"""Validated VR-heading calibration contract used before teleop startup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

VR_TRANSFORM_SCHEMA_VERSION = 1
VR_TRANSFORM_CONVENTION = "R_z(-theta) maps VR FLU forward → robot base +X"
VR_TRANSFORM_MIN_FRAMES = 30
_ROTATION_ATOL = 1e-6


@dataclass(frozen=True)
class VRTransformQuality:
    grade: str
    std_deg: float
    max_deviation_deg: float
    frames: int


@dataclass(frozen=True)
class VRTransformCalibration:
    transform: np.ndarray
    theta_deg: float
    reference: str
    quality: VRTransformQuality


def validate_rotation_matrix(value: Any, *, name: str = "rotation") -> np.ndarray:
    """Return a copied SO(3) matrix or raise with a boundary-specific error."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    orthogonality_error = float(
        np.linalg.norm(matrix.T @ matrix - np.eye(3), ord="fro")
    )
    determinant = float(np.linalg.det(matrix))
    if orthogonality_error > _ROTATION_ATOL or abs(determinant - 1.0) > _ROTATION_ATOL:
        raise ValueError(
            f"{name} must be a proper SO(3) rotation "
            f"(orthogonality_error={orthogonality_error:.3g}, det={determinant:.9g})"
        )
    return matrix.copy()


def _quality_from_payload(payload: Any) -> VRTransformQuality:
    if not isinstance(payload, dict):
        raise ValueError("VR calibration quality must be a structured object")
    raw_grade = payload.get("grade")
    grade = raw_grade.lower() if isinstance(raw_grade, str) else ""
    if grade not in {"excellent", "good", "poor"}:
        raise ValueError(
            "VR calibration quality.grade must be excellent, good, or poor"
        )
    try:
        raw_std_deg = payload["std_deg"]
        raw_max_deviation_deg = payload["max_deviation_deg"]
        raw_frames = payload["frames"]
        if (
            not isinstance(raw_std_deg, (int, float))
            or isinstance(raw_std_deg, bool)
            or not isinstance(raw_max_deviation_deg, (int, float))
            or isinstance(raw_max_deviation_deg, bool)
        ):
            raise TypeError("quality metrics must be JSON numbers")
        std_deg = float(raw_std_deg)
        max_deviation_deg = float(raw_max_deviation_deg)
        frames = int(raw_frames)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("VR calibration quality fields are malformed") from exc
    if (
        not np.isfinite(std_deg)
        or not np.isfinite(max_deviation_deg)
        or std_deg < 0.0
        or max_deviation_deg < std_deg
        or max_deviation_deg > 180.0
        or not isinstance(raw_frames, int)
        or isinstance(raw_frames, bool)
        or frames < VR_TRANSFORM_MIN_FRAMES
    ):
        raise ValueError("VR calibration quality metrics are invalid")
    expected_grade = (
        "excellent" if std_deg < 2.0 else "good" if std_deg < 5.0 else "poor"
    )
    if grade != expected_grade:
        raise ValueError(
            f"VR calibration quality grade {grade!r} disagrees with std_deg={std_deg:.3f}"
        )
    return VRTransformQuality(
        grade=grade,
        std_deg=std_deg,
        max_deviation_deg=max_deviation_deg,
        frames=frames,
    )


def load_vr_transform(
    path: str | Path, *, reject_poor: bool = True
) -> VRTransformCalibration:
    """Load schema-v1 calibration and reject unsafe or ambiguous transforms."""
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"VR transform config not found: {calibration_path}")
    with calibration_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    schema_version = (
        payload.get("schema_version") if isinstance(payload, dict) else None
    )
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version != VR_TRANSFORM_SCHEMA_VERSION
    ):
        raise ValueError(
            "VR transform must use schema_version=1; migrate the calibration file or recalibrate"
        )
    if payload.get("convention") != VR_TRANSFORM_CONVENTION:
        raise ValueError("VR transform convention is missing or unsupported")
    reference = str(payload.get("ref", ""))
    if reference not in {"head", "wrist"}:
        raise ValueError("VR transform ref must be 'head' or 'wrist'")
    raw_theta_deg = payload.get("theta_deg")
    if not isinstance(raw_theta_deg, (int, float)) or isinstance(raw_theta_deg, bool):
        raise ValueError("VR transform theta_deg must be a JSON number")
    theta_deg = float(raw_theta_deg)
    if not np.isfinite(theta_deg) or not -180.0 <= theta_deg <= 180.0:
        raise ValueError("VR transform theta_deg must be finite and within [-180, 180]")
    transform = validate_rotation_matrix(
        payload.get("T_vr_to_robot"), name="T_vr_to_robot"
    )
    theta_rad = float(np.deg2rad(theta_deg))
    expected_transform = np.array(
        [
            [np.cos(theta_rad), np.sin(theta_rad), 0.0],
            [-np.sin(theta_rad), np.cos(theta_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not np.allclose(transform, expected_transform, atol=_ROTATION_ATOL, rtol=0.0):
        raise ValueError(
            "T_vr_to_robot disagrees with theta_deg and the declared convention"
        )
    quality = _quality_from_payload(payload.get("quality"))
    if reject_poor and quality.grade == "poor":
        raise ValueError(
            f"VR calibration quality is poor (std={quality.std_deg:.2f}°); recalibration is required"
        )
    return VRTransformCalibration(
        transform=transform,
        theta_deg=theta_deg,
        reference=reference,
        quality=quality,
    )


__all__ = [
    "VR_TRANSFORM_CONVENTION",
    "VR_TRANSFORM_MIN_FRAMES",
    "VR_TRANSFORM_SCHEMA_VERSION",
    "VRTransformCalibration",
    "VRTransformQuality",
    "load_vr_transform",
    "validate_rotation_matrix",
]

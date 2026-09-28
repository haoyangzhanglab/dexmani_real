"""Validated VR-heading calibration contract used before teleop startup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.utils.geometry import validate_rotation_matrix

# R_z(-theta) maps VR FLU forward to robot base +X. Recalibrate if this convention changes.
VR_TRANSFORM_MIN_FRAMES = 30


@dataclass(frozen=True)
class VRTransformQuality:
    std_deg: float
    max_deviation_deg: float
    frames: int

    @property
    def grade(self) -> str:
        return "excellent" if self.std_deg < 2.0 else "good" if self.std_deg < 5.0 else "poor"


@dataclass(frozen=True)
class VRTransformCalibration:
    transform: np.ndarray
    theta_deg: float
    reference: str
    quality: VRTransformQuality


def _quality_from_payload(payload: Any) -> VRTransformQuality:
    if not isinstance(payload, dict):
        raise ValueError("VR calibration quality must be a structured object")
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
    return VRTransformQuality(
        std_deg=std_deg,
        max_deviation_deg=max_deviation_deg,
        frames=frames,
    )


def load_vr_transform(path: str | Path, *, reject_poor: bool = True) -> VRTransformCalibration:
    """Load current calibration measurements and derive the runtime transform."""
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise FileNotFoundError(f"VR transform config not found: {calibration_path}")
    with calibration_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    return parse_vr_transform(payload, reject_poor=reject_poor)


def parse_vr_transform(payload: Any, *, reject_poor: bool = True) -> VRTransformCalibration:
    """Validate current heading measurements and derive runtime rotation/quality."""
    if not isinstance(payload, dict):
        raise ValueError("VR calibration must be an object")
    reference = str(payload.get("ref", ""))
    if reference not in {"head", "wrist"}:
        raise ValueError("VR transform ref must be 'head' or 'wrist'")
    raw_theta_deg = payload.get("theta_deg")
    if not isinstance(raw_theta_deg, (int, float)) or isinstance(raw_theta_deg, bool):
        raise ValueError("VR transform theta_deg must be a JSON number")
    theta_deg = float(raw_theta_deg)
    if not np.isfinite(theta_deg) or not -180.0 <= theta_deg <= 180.0:
        raise ValueError("VR transform theta_deg must be finite and within [-180, 180]")
    theta_rad = float(np.deg2rad(theta_deg))
    transform = np.array(
        [
            [np.cos(theta_rad), np.sin(theta_rad), 0.0],
            [-np.sin(theta_rad), np.cos(theta_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    transform = validate_rotation_matrix(transform, name="VR heading rotation")
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
    "VR_TRANSFORM_MIN_FRAMES",
    "VRTransformCalibration",
    "VRTransformQuality",
    "load_vr_transform",
    "parse_vr_transform",
]

"""Current VR-heading measurements and startup quality gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


def quality_from_payload(payload: Any) -> VRTransformQuality:
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


def load_vr_transform(path: str | Path) -> VRTransformCalibration:
    """Load a heading calibration suitable for starting teleoperation."""
    with Path(path).open(encoding="utf-8") as stream:
        calibration = parse_vr_transform(json.load(stream))
    if calibration.quality.grade == "poor":
        raise ValueError(
            f"VR calibration quality is poor (std={calibration.quality.std_deg:.2f}°); "
            "recalibration is required"
        )
    return calibration


def parse_vr_transform(payload: Any) -> VRTransformCalibration:
    """Validate heading measurements and derive their rotation and measured quality."""
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
    quality = quality_from_payload(payload.get("quality"))
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

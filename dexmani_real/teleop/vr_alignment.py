"""Current VR-heading measurements and startup quality gate."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# R_z(-theta) maps VR FLU forward to robot base +X. Recalibrate if this convention changes.
VR_TRANSFORM_MIN_FRAMES = 30
MIN_FORWARD_NORM = 1e-6


def circular_mean(
    forwards: np.ndarray, *, outlier_sigma: float = 3.0
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return the circular mean heading and an inlier mask."""
    fwd_2d = forwards[:, :2]
    norms = np.linalg.norm(fwd_2d, axis=1)
    valid = norms >= MIN_FORWARD_NORM
    fwd_2d = fwd_2d[valid]
    norms = norms[valid]
    if fwd_2d.shape[0] == 0:
        raise ValueError("all VR forward samples are near-vertical")
    fwd_unit = fwd_2d / norms[:, None]

    mean_0 = np.mean(fwd_unit, axis=0)
    mean_0_norm = float(np.linalg.norm(mean_0))
    if mean_0_norm < MIN_FORWARD_NORM:
        raise ValueError("VR forward samples have no stable mean heading")
    mean_0 /= mean_0_norm

    angles = np.arctan2(fwd_unit[:, 1], fwd_unit[:, 0])
    center = np.arctan2(mean_0[1], mean_0[0])
    delta = np.angle(np.exp(1j * (angles - center)))
    dists = np.abs(delta)
    threshold = max(1e-12, outlier_sigma * float(np.sqrt(np.mean(delta**2))))
    inlier = dists <= threshold

    if not inlier.any():
        raise ValueError("no VR heading inliers")
    fwd_inlier = fwd_unit[inlier]
    mean_fwd = np.mean(fwd_inlier, axis=0)
    mean_fwd_norm = float(np.linalg.norm(mean_fwd))
    if mean_fwd_norm < MIN_FORWARD_NORM:
        raise ValueError("VR heading inliers have no stable mean direction")
    mean_fwd /= mean_fwd_norm
    theta = float(np.arctan2(mean_fwd[1], mean_fwd[0]))

    full_inlier = np.zeros(len(forwards), dtype=bool)
    full_inlier[valid] = inlier
    return theta, mean_fwd, full_inlier


def quality_measurements(
    forwards: np.ndarray,
    theta_mean: float,
    inlier: np.ndarray,
) -> dict[str, float]:
    """Measure heading scatter using the valid inlier mask returned by circular_mean."""
    fwd_2d = forwards[inlier, :2]
    thetas = np.arctan2(fwd_2d[:, 1], fwd_2d[:, 0])
    dtheta = np.angle(np.exp(1j * (thetas - theta_mean)))
    std_deg = float(np.rad2deg(np.std(dtheta)))
    max_dev = float(np.rad2deg(float(np.max(np.abs(dtheta)))))

    return {
        "std_deg": std_deg,
        "max_deviation_deg": max_dev,
    }


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
    std_deg = payload.get("std_deg")
    max_deviation_deg = payload.get("max_deviation_deg")
    frames = payload.get("frames")
    if (
        not isinstance(std_deg, (int, float))
        or isinstance(std_deg, bool)
        or not isinstance(max_deviation_deg, (int, float))
        or isinstance(max_deviation_deg, bool)
        or not 0.0 <= std_deg <= max_deviation_deg <= 180.0
        or type(frames) is not int
        or frames < VR_TRANSFORM_MIN_FRAMES
    ):
        raise ValueError("VR calibration quality metrics are invalid")
    return VRTransformQuality(
        std_deg=float(std_deg),
        max_deviation_deg=float(max_deviation_deg),
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
    reference = payload.get("ref")
    if reference not in ("head", "wrist"):
        raise ValueError("VR transform ref must be 'head' or 'wrist'")
    theta_deg = payload.get("theta_deg")
    if (
        not isinstance(theta_deg, (int, float))
        or isinstance(theta_deg, bool)
        or not -180.0 <= theta_deg <= 180.0
    ):
        raise ValueError("VR transform theta_deg must be finite and within [-180, 180]")
    theta_deg = float(theta_deg)
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

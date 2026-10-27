"""VR 朝向估计、质量检查与基座对齐。"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MIN_HEADING_SAMPLES = 30
MIN_FORWARD_NORM = 1e-6


@dataclass(frozen=True)
class VRCalibration:
    theta_deg: float
    reference: str
    std_deg: float
    max_deviation_deg: float
    frames: int

    def __post_init__(self):
        if self.reference not in ("head", "wrist"):
            raise ValueError("VR reference must be 'head' or 'wrist'")
        if not -180 <= self.theta_deg <= 180:
            raise ValueError("VR heading must be finite and within [-180, 180]")
        if not 0 <= self.std_deg <= self.max_deviation_deg <= 180:
            raise ValueError("invalid VR heading scatter")
        if not isinstance(self.frames, int) or self.frames < MIN_HEADING_SAMPLES:
            raise ValueError(f"need at least {MIN_HEADING_SAMPLES} VR heading samples")

    @property
    def grade(self):
        return "excellent" if self.std_deg < 2 else "good" if self.std_deg < 5 else "poor"

    @property
    def transform(self):
        # R_z(-theta) 将 VR FLU 前向映射到机器人基座 +X。
        theta = np.deg2rad(self.theta_deg)
        cosine, sine = np.cos(theta), np.sin(theta)
        return np.array([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])


def estimate_heading(forwards: np.ndarray, reference: str) -> VRCalibration:
    directions = forwards[:, :2]
    norms = np.linalg.norm(directions, axis=1)
    valid = np.isfinite(directions).all(axis=1) & (norms >= MIN_FORWARD_NORM)
    directions = directions[valid] / norms[valid, None]
    if len(directions) < MIN_HEADING_SAMPLES:
        raise ValueError("not enough horizontal VR heading samples")
    mean = directions.mean(axis=0)
    if np.linalg.norm(mean) < MIN_FORWARD_NORM:
        raise ValueError("VR samples have no stable mean heading")

    angles = np.arctan2(directions[:, 1], directions[:, 0])
    center = np.arctan2(mean[1], mean[0])
    delta = np.angle(np.exp(1j * (angles - center)))
    inliers = np.abs(delta) <= max(1e-12, 3 * np.sqrt(np.mean(delta**2)))
    if np.count_nonzero(inliers) < MIN_HEADING_SAMPLES:
        raise ValueError("not enough VR heading inliers")
    mean = directions[inliers].mean(axis=0)
    if np.linalg.norm(mean) < MIN_FORWARD_NORM:
        raise ValueError("VR inliers have no stable mean heading")
    theta = np.arctan2(mean[1], mean[0])
    deviation = np.rad2deg(np.angle(np.exp(1j * (angles[inliers] - theta))))
    return VRCalibration(
        float(np.rad2deg(theta)),
        reference,
        float(deviation.std()),
        float(np.abs(deviation).max()),
        int(inliers.sum()),
    )


def load_vr_transform(path: str | Path) -> VRCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    quality = payload["quality"]
    calibration = VRCalibration(
        float(payload["theta_deg"]),
        payload["ref"],
        float(quality["std_deg"]),
        float(quality["max_deviation_deg"]),
        quality["frames"],
    )
    if calibration.grade == "poor":
        raise ValueError(
            f"VR heading scatter is {calibration.std_deg:.2f}°; recalibration required"
        )
    return calibration

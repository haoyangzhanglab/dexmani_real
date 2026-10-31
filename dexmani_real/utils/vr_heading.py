"""VR 朝向估计、质量检查与基座对齐。"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dexmani_real.config.calibration_data import VR_TRANSFORM_PATH
from dexmani_real.geometry.pose import forward_from_quat_wxyz
from dexmani_real.sensor.vr import _rotation_to_flu
from dexmani_real.utils.io import write_json

MIN_HEADING_SAMPLES = 30
MIN_FORWARD_NORM = 1e-6
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HeadingCalibration:
    theta_deg: float
    reference: str
    std_deg: float
    max_deviation_deg: float
    frames: int

    @property
    def grade(self):
        return "excellent" if self.std_deg < 2 else "good" if self.std_deg < 5 else "poor"


def estimate_heading(forwards: np.ndarray, reference: str) -> HeadingCalibration:
    if reference not in ("head", "wrist"):
        raise ValueError("VR reference must be 'head' or 'wrist'")
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
    return HeadingCalibration(
        float(np.rad2deg(theta)),
        reference,
        float(deviation.std()),
        float(np.abs(deviation).max()),
        int(inliers.sum()),
    )


def load_vr_transform(path: str | Path) -> np.ndarray:
    """从有效角度构造运行变换，历史标定报告不参与运行准入。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    theta_deg = float(payload["theta_deg"])
    if not -180 <= theta_deg <= 180:
        raise ValueError("VR heading must be finite and within [-180, 180]")
    # R_z(-theta) 将 VR FLU 前向映射到机器人基座 +X。
    theta = np.deg2rad(theta_deg)
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.array([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])


def save_vr_transform(calibration: HeadingCalibration, path: str | Path = VR_TRANSFORM_PATH) -> Path:
    if calibration.grade == "poor":
        raise ValueError("VR heading quality is poor; collect a steadier sample")
    return write_json({
        "theta_deg": calibration.theta_deg,
        "ref": calibration.reference,
        "quality": {
            "std_deg": calibration.std_deg,
            "max_deviation_deg": calibration.max_deviation_deg,
            "frames": calibration.frames,
        },
        "calibrated_at_utc": datetime.now(timezone.utc).isoformat(),
    }, path)


def read_heading(event, reference):
    from hand_tracking_sdk import HandFrame, HandSide, HeadFrame

    if reference == "head" and isinstance(event, HeadFrame):
        sequence, pose = int(event.recv_ts_ns), event.head
        if sequence <= 0:
            return None
    elif reference == "wrist" and isinstance(event, HandFrame) and event.side == HandSide.RIGHT:
        sequence, pose = int(event.sequence_id), event.wrist
    else:
        return None
    try:
        return sequence, _rotation_to_flu(pose)
    except ValueError as exc:
        logger.warning("Invalid VR %s heading rejected: %s", reference, exc)
        return None


def collect_headings(events, reference, duration_s):
    deadline = time.monotonic() + duration_s
    forwards = []
    duplicates = 0
    previous = None
    last_print = 0.0
    for event in events:
        if time.monotonic() >= deadline:
            break
        sample = read_heading(event, reference)
        if sample is not None:
            sequence, quaternion = sample
            if sequence == previous:
                duplicates += 1
            else:
                previous = sequence
                forwards.append(forward_from_quat_wxyz(quaternion))
        now = time.monotonic()
        if now - last_print >= 1.0:
            print(f"    collected {len(forwards)} frames...")
            last_print = now
    else:
        raise RuntimeError("VR receiver stopped during collection")
    return np.asarray(forwards, dtype=np.float64).reshape(-1, 3), duplicates

"""VR 朝向估计、质量检查与基座对齐。"""

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dexmani_real.config.calibration_data import VR_TRANSFORM_PATH
from dexmani_real.geometry.pose import forward_from_quat_wxyz
from dexmani_real.ipc.sensors import read_vr_frame
from dexmani_real.utils.io import write_json

MIN_HEADING_SAMPLES = 30
MIN_FORWARD_NORM = 1e-6


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

    @property
    def transform(self):
        # R_z(-theta) 将 VR FLU 前向映射到机器人基座 +X。
        theta = np.deg2rad(self.theta_deg)
        cosine, sine = np.cos(theta), np.sin(theta)
        return np.array([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])


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


def load_vr_transform(path: str | Path) -> HeadingCalibration:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    quality = payload["quality"]
    calibration = HeadingCalibration(
        float(payload["theta_deg"]),
        payload["ref"],
        float(quality["std_deg"]),
        float(quality["max_deviation_deg"]),
        quality["frames"],
    )
    if calibration.reference not in ("head", "wrist"):
        raise ValueError("VR reference must be 'head' or 'wrist'")
    if not -180 <= calibration.theta_deg <= 180:
        raise ValueError("VR heading must be finite and within [-180, 180]")
    if not 0 <= calibration.std_deg <= calibration.max_deviation_deg <= 180:
        raise ValueError("invalid VR heading scatter")
    if not isinstance(calibration.frames, int) or calibration.frames < MIN_HEADING_SAMPLES:
        raise ValueError(f"need at least {MIN_HEADING_SAMPLES} VR heading samples")
    if calibration.grade == "poor":
        raise ValueError(
            f"VR heading scatter is {calibration.std_deg:.2f}°; recalibration required"
        )
    return calibration


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


def read_heading(channels, reference):
    frame = read_vr_frame(channels)
    if frame is None:
        return None
    # 头部姿态会随手部帧重复发布，必须使用头部自身的接收时间去重。
    if reference == "head":
        stamp = int(frame["head_recv_ts_ns"])
        return (stamp, frame["head_quat_wxyz"]) if stamp > 0 else None
    return int(frame["sequence_id"]), frame["wrist_quat_wxyz"]


def collect_headings(channels, sensors, reference, duration_s):
    deadline = time.monotonic() + duration_s
    forwards = []
    duplicates = 0
    previous = None
    last_print = 0.0
    while time.monotonic() < deadline:
        if not sensors.check():
            raise RuntimeError("VR receiver exited during collection")
        sample = read_heading(channels, reference)
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
        time.sleep(0.01)
    return np.asarray(forwards, dtype=np.float64).reshape(-1, 3), duplicates

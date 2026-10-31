"""Raw 首帧选择使用的 RGB-D、相机几何和来源时钟。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import RectificationMaps, validate_distortion, validate_rgbd


@dataclass(frozen=True)
class Frame:
    """Aligned color-camera RGB-D in metres; source identity and clock describe the recorded timing."""

    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    distortion_model: str
    distortion_coeffs: tuple[float, float, float, float, float]
    source_id: str
    timestamp_ns: int
    clock: str
    row_index: int | None = None


def load_frame(path: str | Path) -> Frame:
    """Read one source-grid NPZ without inferring calibration, units or timing."""
    path = Path(path).expanduser().resolve()
    required = {
        "rgb",
        "depth_m",
        "K",
        "distortion_model",
        "distortion_coeffs",
        "source_id",
        "timestamp_ns",
        "clock",
    }
    with np.load(path, allow_pickle=False) as saved:
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
        text = {}
        for key in ("distortion_model", "source_id", "clock"):
            value = saved[key]
            if value.shape != () or value.dtype.kind != "U" or not str(value.item()):
                raise ValueError(f"{path.name}: {key} must be a nonempty scalar Unicode string")
            text[key] = str(value.item())
        integer = {}
        for key in ("timestamp_ns", "row_index"):
            if key not in saved.files:
                continue
            value = saved[key]
            if value.shape != () or value.dtype.kind not in {"i", "u"} or int(value.item()) < 0:
                raise ValueError(f"{path.name}: {key} must be a nonnegative scalar integer")
            integer[key] = int(value.item())
        coefficients = saved["distortion_coeffs"]
        if coefficients.shape != (5,) or coefficients.dtype.kind not in {"f", "i", "u"}:
            raise ValueError(f"{path.name}: distortion_coeffs must contain five numbers")
        k1, k2, p1, p2, k3 = map(float, coefficients)
        depth_m = saved["depth_m"]
        if depth_m.dtype != np.float32:
            raise ValueError("aligned depth must be finite nonnegative float32 [H,W] metres")
        rgb, depth_m, K = validate_rgbd(saved["rgb"], depth_m, saved["K"])
        distortion = validate_distortion(text["distortion_model"], (k1, k2, p1, p2, k3))
        return Frame(
            rgb=rgb,
            depth_m=depth_m,
            K=K,
            distortion_coeffs=distortion,
            row_index=integer.get("row_index"),
            timestamp_ns=integer["timestamp_ns"],
            **text,
        )


def save_frame(path, frame):
    """Save the original aligned RGB-D grid and its source identity and timing."""
    payload = dict(
        rgb=frame.rgb,
        depth_m=frame.depth_m,
        K=frame.K,
        distortion_model=frame.distortion_model,
        distortion_coeffs=frame.distortion_coeffs,
        source_id=frame.source_id,
        timestamp_ns=frame.timestamp_ns,
        clock=frame.clock,
    )
    if frame.row_index is not None:
        payload["row_index"] = frame.row_index
    np.savez_compressed(path, **payload)


def rectify_frame(frame, mask=None, *, maps=None):
    """Return aligned RGB, measured depth, K and optional mask on the pinhole grid."""
    if maps is None:
        maps = RectificationMaps(frame.depth_m.shape, frame.K, frame.distortion_coeffs)
    rgb, depth_m, mask = maps.apply(frame.rgb, frame.depth_m, mask)
    return rgb, depth_m, frame.K, mask

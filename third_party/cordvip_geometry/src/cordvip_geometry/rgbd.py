"""米制 RGB-D、相机几何与可选来源说明。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import validate_distortion, validate_intrinsics
from .io import write_npz


@dataclass(frozen=True)
class RGBDFrame:
    """color-camera 网格的米制 RGB-D，附可选来源说明。"""

    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    distortion_model: str
    distortion_coeffs: tuple[float, float, float, float, float]
    source_id: str | None = None
    timestamp_ns: int | None = None
    clock: str | None = None
    row_index: int | None = None

    def __post_init__(self):
        rgb, depth_m = np.asarray(self.rgb), np.asarray(self.depth_m)
        if depth_m.dtype.kind != "f":
            raise ValueError("depth_m must contain floating-point metres")
        with np.errstate(over="ignore", invalid="ignore"):
            depth_m = depth_m.astype(np.float32, copy=False)
        if (
            depth_m.ndim != 2
            or not all(depth_m.shape)
            or not np.isfinite(depth_m).all()
            or np.any(depth_m < 0)
        ):
            raise ValueError(
                "depth_m must be nonempty finite nonnegative float32 [H,W] metres"
            )
        if rgb.dtype != np.uint8 or rgb.shape != (*depth_m.shape, 3):
            raise ValueError("RGB must be uint8 [H,W,3] aligned with depth_m")
        for name in ("source_id", "clock"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
        value = self.timestamp_ns
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, np.integer))
            ):
                raise ValueError("timestamp_ns must be an integer")
            object.__setattr__(self, "timestamp_ns", int(value))
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth_m)
        object.__setattr__(self, "K", validate_intrinsics(self.K))
        object.__setattr__(
            self,
            "distortion_coeffs",
            validate_distortion(
                self.distortion_model,
                self.distortion_coeffs,
            ),
        )

    @classmethod
    def from_camera(cls, camera, camera_row):
        """复用 CameraSnapshot 标定，将运行时相机行转为米制帧。"""
        if camera.depth_scale_m is None:
            raise ValueError("RGB-D geometry requires a camera depth scale")
        rgb = np.asarray(camera_row["rgb"])
        depth = np.asarray(camera_row["depth"])
        shape = (camera.color.height, camera.color.width)
        if depth.dtype != np.uint16 or depth.shape != shape:
            raise ValueError(
                "camera depth must be uint16 [H,W] matching color intrinsics"
            )
        return cls(
            rgb=rgb,
            depth_m=np.multiply(depth, camera.depth_scale_m, dtype=np.float32),
            K=camera.color.matrix(),
            distortion_model=camera.color.distortion_model,
            distortion_coeffs=camera.color.distortion_coeffs,
            timestamp_ns=camera_row.get("timestamp_ns"),
            clock="host_monotonic_oldest_rgbd_receive",
        )


def load_frame(path: str | Path) -> RGBDFrame:
    path = Path(path).expanduser().resolve()
    with np.load(path, allow_pickle=False) as saved:
        return RGBDFrame(
            rgb=saved["rgb"],
            depth_m=saved["depth_m"],
            K=saved["K"],
            distortion_model=saved["distortion_model"].item(),
            distortion_coeffs=saved["distortion_coeffs"],
            source_id=saved["source_id"].item() if "source_id" in saved.files else None,
            timestamp_ns=saved["timestamp_ns"].item() if "timestamp_ns" in saved.files else None,
            clock=saved["clock"].item() if "clock" in saved.files else None,
            row_index=saved["row_index"].item() if "row_index" in saved.files else None,
        )


def save_frame(path, frame):
    payload = {
        "rgb": frame.rgb,
        "depth_m": frame.depth_m,
        "K": frame.K,
        "distortion_model": frame.distortion_model,
        "distortion_coeffs": frame.distortion_coeffs,
    }
    for name in ("source_id", "timestamp_ns", "clock", "row_index"):
        value = getattr(frame, name)
        if value is not None:
            payload[name] = value
    write_npz(path, payload)

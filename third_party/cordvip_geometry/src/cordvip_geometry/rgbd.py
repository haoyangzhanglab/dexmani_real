"""Aligned RGB-D frames and offline NPZ inputs, independent of hardware packages."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class Frame:
    """Aligned color-camera data; zero depth is invalid, all distances are metres.

    ``source_id`` identifies the camera frame when that evidence exists. Raw
    episodes without camera identities use an explicitly labelled row identity.
    ``clock`` distinguishes source receive time from observation-relative time;
    neither is a camera exposure timestamp. Repeated camera frames may appear in
    multiple Raw rows with the same source ID and source timestamp.
    """

    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    distortion_model: str
    distortion_coeffs: tuple[float, float, float, float, float]
    source_id: str
    timestamp_ns: int
    clock: str
    row_index: int | None = None

    def __post_init__(self) -> None:
        if (
            self.rgb.dtype != np.uint8
            or self.rgb.ndim != 3
            or self.rgb.shape[2] != 3
            or min(self.rgb.shape[:2]) <= 0
        ):
            raise ValueError("RGB must be nonempty uint8 [H,W,3]")
        if (
            self.depth_m.dtype != np.float32
            or self.depth_m.shape != self.rgb.shape[:2]
            or not np.isfinite(self.depth_m).all()
            or np.any(self.depth_m < 0)
        ):
            raise ValueError("aligned depth must be finite nonnegative float32 [H,W] metres")
        if (
            self.K.shape != (3, 3)
            or not np.isfinite(self.K).all()
            or self.K[0, 0] <= 0
            or self.K[1, 1] <= 0
            or not np.allclose(self.K[2], (0, 0, 1), rtol=0, atol=1e-9)
        ):
            raise ValueError("K must be a finite pinhole matrix with positive focal lengths")
        if (
            len(self.distortion_coeffs) != 5
            or not np.isfinite(self.distortion_coeffs).all()
            or not self.distortion_model
        ):
            raise ValueError("camera distortion requires its model and five finite coefficients")
        if not self.source_id or not self.clock or self.timestamp_ns < 0:
            raise ValueError("frame requires source identity, clock, and a nonnegative timestamp")


def iter_rgbd_directory(path: str | Path) -> Iterator[Frame]:
    """Read lexically ordered ``000000.npz`` frames without video/SDK imports.

    Each file contains the Frame fields, with ``row_index`` optional. Text fields
    are scalar Unicode arrays and timestamps are scalar integers. No calibration,
    depth units, frame identity, or clock metadata are inferred from filenames.
    """
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"RGB-D input must be a directory: {directory}")
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise ValueError(f"RGB-D directory contains no NPZ frames: {directory}")
    if any(re.fullmatch(r"[0-9]{6}\.npz", file.name) is None for file in paths):
        raise ValueError("RGB-D frame filenames must contain six digits, for example 000000.npz")
    required = {
        "rgb", "depth_m", "K", "distortion_model", "distortion_coeffs",
        "source_id", "timestamp_ns", "clock",
    }
    for frame_path in paths:
        with np.load(frame_path, allow_pickle=False) as saved:
            missing = required - set(saved.files)
            if missing:
                raise ValueError(f"{frame_path.name}: missing fields {sorted(missing)}")
            text = {}
            for key in ("distortion_model", "source_id", "clock"):
                value = saved[key]
                if value.shape != () or value.dtype.kind != "U" or not str(value.item()):
                    raise ValueError(f"{frame_path.name}: {key} must be a nonempty scalar Unicode string")
                text[key] = str(value.item())
            integer = {}
            for key in ("timestamp_ns", "row_index"):
                if key not in saved.files:
                    continue
                value = saved[key]
                if value.shape != () or value.dtype.kind not in {"i", "u"} or int(value.item()) < 0:
                    raise ValueError(f"{frame_path.name}: {key} must be a nonnegative scalar integer")
                integer[key] = int(value.item())
            coefficients = saved["distortion_coeffs"]
            if coefficients.shape != (5,) or coefficients.dtype.kind not in {"f", "i", "u"}:
                raise ValueError(f"{frame_path.name}: distortion_coeffs must contain five numbers")
            yield Frame(
                rgb=saved["rgb"],
                depth_m=saved["depth_m"],
                K=saved["K"],
                distortion_coeffs=tuple(float(value) for value in coefficients),
                row_index=integer.get("row_index"),
                timestamp_ns=integer["timestamp_ns"],
                **text,
            )

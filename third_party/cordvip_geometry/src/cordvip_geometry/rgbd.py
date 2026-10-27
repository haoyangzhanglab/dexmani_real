"""Aligned RGB-D frames and offline NPZ inputs, independent of hardware packages."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .geometry import validate_rgbd


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

    def __post_init__(self) -> None:
        if self.depth_m.dtype != np.float32:
            raise ValueError("aligned depth must be finite nonnegative float32 [H,W] metres")
        _, _, intrinsics = validate_rgbd(self.rgb, self.depth_m, self.K)
        intrinsics.setflags(write=False)
        object.__setattr__(self, "K", intrinsics)
        if (
            len(self.distortion_coeffs) != 5
            or not np.isfinite(self.distortion_coeffs).all()
            or not self.distortion_model
        ):
            raise ValueError("camera distortion requires its model and five finite coefficients")
        if not self.source_id or not self.clock or self.timestamp_ns < 0:
            raise ValueError("frame requires source identity, clock, and a nonnegative timestamp")


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
        return Frame(
            rgb=saved["rgb"],
            depth_m=saved["depth_m"],
            K=saved["K"],
            distortion_coeffs=tuple(float(value) for value in coefficients),
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
    from .geometry import rectify_rgbd

    model = frame.distortion_model.removeprefix("distortion.")
    coefficients = np.asarray(frame.distortion_coeffs)
    if model not in {"none", "brown_conrady"}:
        raise ValueError(f"Unsupported distortion model: {frame.distortion_model}")
    if model == "none" and np.any(coefficients != 0):
        raise ValueError("none distortion requires zero coefficients")
    return rectify_rgbd(frame.rgb, frame.depth_m, frame.K, coefficients, mask=mask, maps=maps)


def iter_rgbd_directory(path: str | Path) -> Iterator[Frame]:
    """Read lexically ordered ``000000.npz`` frames; ``row_index`` stays optional."""
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"RGB-D input must be a directory: {directory}")
    paths = sorted(directory.glob("*.npz"))
    if not paths:
        raise ValueError(f"RGB-D directory contains no NPZ frames: {directory}")
    if any(re.fullmatch(r"[0-9]{6}\.npz", file.name) is None for file in paths):
        raise ValueError("RGB-D frame filenames must contain six digits, for example 000000.npz")
    for frame_path in paths:
        yield load_frame(frame_path)

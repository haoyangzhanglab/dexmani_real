"""Load saved aligned RGB-D geometry for Raw consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from dexmani_real.recording.storage.reader import (
    EpisodeReader,
    positive_float_attr,
    positive_int_attr,
    text_attr,
)
from dexmani_real.sensor.camera.geometry import CameraIntrinsics
from dexmani_real.utils.geometry import validate_rigid_transform


@dataclass(frozen=True)
class RawEpisodeCameraModel:
    """Color-grid intrinsics and depth scale saved with aligned Raw RGB-D."""

    intrinsics: CameraIntrinsics
    depth_scale_m: float


def color_intrinsics_from_meta(meta: Any) -> CameraIntrinsics:
    matrix = np.asarray(meta["camera_color_intrinsics"], dtype=np.float64).reshape(3, 3)
    coefficients = tuple(meta["camera_color_distortion_coeffs"])
    color = CameraIntrinsics(
        width=positive_int_attr(meta, "camera_color_width"),
        height=positive_int_attr(meta, "camera_color_height"),
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        ppx=float(matrix[0, 2]),
        ppy=float(matrix[1, 2]),
        distortion_model=text_attr(meta, "camera_color_distortion_model"),
        distortion_coeffs=cast(tuple[float, float, float, float, float], coefficients),
    )
    # Do not silently discard skew, invalid off-diagonal entries or a malformed last row.
    if not np.allclose(matrix, color.matrix(), rtol=0.0, atol=1e-9):
        raise ValueError("camera_color_intrinsics must be a canonical pinhole matrix")
    return color


def load_raw_episode_camera_model(reader: EpisodeReader) -> RawEpisodeCameraModel:
    """Load the aligned color-grid camera model from Raw physical metadata."""
    meta = reader.meta
    if text_attr(meta, "camera_payload_mode") != "depth_to_color_aligned_rgbd":
        raise ValueError("raw camera payload is not depth-to-color aligned RGB-D")
    return RawEpisodeCameraModel(
        intrinsics=color_intrinsics_from_meta(meta),
        depth_scale_m=positive_float_attr(meta, "depth_scale"),
    )


def load_raw_episode_base_from_color(reader: EpisodeReader) -> np.ndarray:
    """Return Raw's static aligned-color camera to xArm-base transform."""
    meta = reader.meta
    transform = validate_rigid_transform(
        np.asarray(meta["camera_T_xarm_base_from_color"]).reshape(4, 4),
        label="camera_T_xarm_base_from_color",
    )
    transform.setflags(write=False)
    return transform

"""Reconstruct xArm-base clouds from Raw RGB-D arrays and camera metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.recording.storage.reader import EpisodeReader
from dexmani_real.sensor.camera.geometry import (
    CameraIntrinsics,
    RGBDGeometry,
    validate_aligned_depth_distortion,
)
from dexmani_real.sensor.pointcloud import build_point_cloud
from dexmani_real.utils.geometry import validate_rigid_transform


@dataclass(frozen=True)
class RawEpisodeCameraModel:
    """Aligned RGB-D geometry and depth scale from one raw episode."""

    geometry: RGBDGeometry
    depth_scale_m: float


def _color_intrinsics_from_meta(meta: Any) -> CameraIntrinsics:
    matrix = np.asarray(meta["camera_color_intrinsics"], dtype=np.float64).reshape(3, 3)
    coefficients = tuple(float(value) for value in meta["camera_color_distortion_coeffs"])
    color = CameraIntrinsics(
        width=int(meta["camera_color_width"]),
        height=int(meta["camera_color_height"]),
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        ppx=float(matrix[0, 2]),
        ppy=float(matrix[1, 2]),
        distortion_model=str(meta["camera_color_distortion_model"]),
        distortion_coeffs=cast(tuple[float, float, float, float, float], coefficients),
    )
    # Do not silently discard skew, invalid off-diagonal entries or a malformed last row.
    if not np.allclose(matrix, color.matrix(), rtol=0.0, atol=1e-9):
        raise ValueError("camera_color_intrinsics must be a canonical pinhole matrix")
    return color


def load_raw_episode_camera_model(reader: EpisodeReader) -> RawEpisodeCameraModel:
    """Load the aligned color-grid camera model from Raw physical metadata."""
    meta = reader.meta
    if str(meta.get("camera_payload_mode", "")) != "depth_to_color_aligned_rgbd":
        raise ValueError("raw camera payload is not depth-to-color aligned RGB-D")
    color = _color_intrinsics_from_meta(meta)
    validate_aligned_depth_distortion(color.distortion_model)
    geometry = RGBDGeometry(
        depth=color,
        color=color,
        T_color_from_depth=np.eye(4, dtype=np.float64),
    )
    depth_scale_m = float(meta["depth_scale"])
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    return RawEpisodeCameraModel(
        geometry=geometry,
        depth_scale_m=depth_scale_m,
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


@dataclass(frozen=True)
class RawEpisodePointCloudDeriver:
    """Reconstruct point clouds with one raw episode's camera geometry."""

    camera: RawEpisodeCameraModel
    T_xarm_base_from_color: np.ndarray
    pointcloud: PointCloudConfig
    table_plane_abcd: tuple[float, float, float, float] | None

    def __post_init__(self) -> None:
        self.pointcloud.validate()
        transform = validate_rigid_transform(
            self.T_xarm_base_from_color, label="T_xarm_base_from_color"
        )
        transform.setflags(write=False)
        object.__setattr__(self, "T_xarm_base_from_color", transform)
        if not self.pointcloud.remove_table:
            object.__setattr__(self, "table_plane_abcd", None)
        elif self.table_plane_abcd is None:
            raise ValueError("remove_table=True requires an explicit table_plane_abcd")
        else:
            plane = tuple(float(value) for value in self.table_plane_abcd)
            if len(plane) != 4 or not np.isfinite(plane).all() or plane[2] <= 0:
                raise ValueError("table plane must be finite with an upward normal")
            norm = np.linalg.norm(plane[:3])
            if norm <= 0.0 or plane[2] / norm <= 0.0:
                raise ValueError("table plane must have an upward non-zero normal")
            object.__setattr__(self, "table_plane_abcd", plane)

    def derive(self, depth: np.ndarray, rgb: np.ndarray) -> np.ndarray | None:
        """Derive ``float32[N,6]`` from same-row aligned depth/RGB arrays without modifying them."""
        return build_point_cloud(
            depth_raw=np.asarray(depth, dtype=np.uint16),
            color=np.asarray(rgb, dtype=np.uint8),
            depth_scale_m=self.camera.depth_scale_m,
            geometry=self.camera.geometry,
            T_xarm_base_from_color=self.T_xarm_base_from_color,
            table_plane_abcd=self.table_plane_abcd,
            config=self.pointcloud,
        )

"""Raw-v22 camera metadata to canonical xArm-base point-cloud inputs.

This module owns the persisted-data boundary shared by offline processing and
raw episode visualization. It performs no hardware IO and does not own or
close the borrowed :class:`EpisodeReader`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.recording.reader import EpisodeReader
from dexmani_real.sensor.camera_geometry import CameraIntrinsics, RGBDGeometry
from dexmani_real.sensor.pointcloud import build_point_cloud


@dataclass(frozen=True)
class RawEpisodeCameraModel:
    """Validated aligned RGB-D geometry and depth scale from one raw episode."""

    geometry: RGBDGeometry
    depth_scale_m: float


def validate_rigid_transform(transform: np.ndarray, *, label: str) -> np.ndarray:
    """Return a validated finite rigid homogeneous 4x4 transform."""
    value = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = value[:3, :3]
    if (
        not np.all(np.isfinite(value))
        or not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8, rtol=0.0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5, rtol=0.0)
    ):
        raise ValueError(f"{label} must be a finite rigid homogeneous transform")
    return value


def _intrinsics_from_meta(meta: Any, *, stream: str) -> CameraIntrinsics:
    prefix = f"camera_{stream}"
    matrix = np.asarray(meta[f"{prefix}_intrinsics"], dtype=np.float64).reshape(3, 3)
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-9):
        raise ValueError(f"{prefix}_intrinsics must be a canonical pinhole matrix")
    coefficients = tuple(float(value) for value in meta[f"{prefix}_distortion_coeffs"])
    return CameraIntrinsics(
        width=int(meta[f"{prefix}_width"]),
        height=int(meta[f"{prefix}_height"]),
        fx=float(matrix[0, 0]),
        fy=float(matrix[1, 1]),
        ppx=float(matrix[0, 2]),
        ppy=float(matrix[1, 2]),
        distortion_model=str(meta[f"{prefix}_distortion_model"]),
        distortion_coeffs=cast(tuple[float, float, float, float, float], coefficients),
    )


def load_raw_episode_camera_model(reader: EpisodeReader) -> RawEpisodeCameraModel:
    """Load the depth-to-color aligned camera model persisted by raw v22."""
    if reader.schema_version != 22:
        raise ValueError(
            "aligned RGB-D point-cloud inputs require raw schema v22, "
            f"got v{reader.schema_version}"
        )
    meta = reader.h5f["meta"].attrs
    if str(meta.get("camera_payload_mode", "")) != "depth_to_color_aligned_rgbd":
        raise ValueError("raw v22 camera payload is not depth-to-color aligned RGB-D")
    native_geometry = RGBDGeometry(
        depth=_intrinsics_from_meta(meta, stream="depth"),
        color=_intrinsics_from_meta(meta, stream="color"),
        T_color_from_depth=np.asarray(
            meta["camera_T_color_from_depth"], dtype=np.float64
        ).reshape(4, 4),
    )
    depth_scale_m = float(meta["depth_scale"])
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth_scale must be finite and positive")
    return RawEpisodeCameraModel(
        geometry=native_geometry.aligned_depth_to_color(),
        depth_scale_m=depth_scale_m,
    )


def load_raw_episode_base_from_color(reader: EpisodeReader) -> np.ndarray:
    """Return the static color-camera to xArm-base transform for eye-to-hand."""
    meta = reader.h5f["meta"].attrs
    camera_type = str(meta.get("camera_type", ""))
    if camera_type == "eye_in_hand":
        raise ValueError(
            "raw-v22 xArm-base camera geometry for eye_in_hand requires arm pose "
            "evaluated at color/depth exposure times, which is not persisted"
        )
    if camera_type != "eye_to_hand":
        raise ValueError(f"unsupported camera_type {camera_type!r}")

    transform = validate_rigid_transform(
        np.asarray(meta["camera_T_xarm_base_from_color"]),
        label="camera_T_xarm_base_from_color",
    )
    transform = transform.copy()
    transform.setflags(write=False)
    return transform


@dataclass(frozen=True)
class RawEpisodePointCloudDeriver:
    """Derive canonical point clouds while the caller owns the episode reader."""

    reader: EpisodeReader
    camera: RawEpisodeCameraModel
    T_xarm_base_from_color: np.ndarray
    pointcloud: PointCloudConfig
    table_plane_abcd: tuple[float, float, float, float] | None

    def derive(self, source_index: int, rgb: np.ndarray) -> np.ndarray | None:
        """Derive one ``float32[N,6]`` xArm-base cloud from a raw grid row."""
        depth_raw = np.asarray(self.reader.h5f["depth"][source_index], dtype=np.uint16)
        return build_point_cloud(
            depth_raw=depth_raw,
            color=np.asarray(rgb, dtype=np.uint8),
            depth_scale_m=self.camera.depth_scale_m,
            geometry=self.camera.geometry,
            T_xarm_base_from_color=self.T_xarm_base_from_color,
            table_plane_abcd=self.table_plane_abcd,
            config=self.pointcloud,
        )

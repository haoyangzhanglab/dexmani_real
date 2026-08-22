"""Pure native RGB-D to xArm-base point-cloud construction.

The module owns only deterministic numerical geometry.  It does not import a
camera SDK, Open3D, shared memory, calibration files, or recording code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import (  # type: ignore[import-untyped]
    convolve,
    maximum_filter,
    minimum_filter,
)

from dexmani_real.sensor.camera_geometry import CameraIntrinsics, RGBDGeometry

POINT_CLOUD_COLOR_SOURCE = "native_color_projection"
POINT_CLOUD_SAMPLING = (
    "voxel_first_representative_then_deterministic_uniform_or_cyclic_pad"
)

__all__ = [
    "POINT_CLOUD_COLOR_SOURCE",
    "POINT_CLOUD_SAMPLING",
    "PointCloudConfig",
    "build_point_cloud",
]


@dataclass(frozen=True)
class PointCloudConfig:
    """Fixed production parameters for the native point-cloud path."""

    num_points: int = 1024
    depth_min_m: float = 0.30
    depth_max_m: float = 1.50
    edge_jump_m: float = 0.030
    edge_surface_band_m: float = 0.008
    table_clearance_m: float = 0.008
    workspace: tuple[float, float, float, float, float, float] = (
        0.25,
        -0.60,
        0.005,
        0.85,
        0.60,
        0.80,
    )
    voxel_size_m: float = 0.005

    def __post_init__(self) -> None:
        if isinstance(self.num_points, bool) or int(self.num_points) <= 0:
            raise ValueError("num_points must be a positive integer")
        object.__setattr__(self, "num_points", int(self.num_points))
        values = (
            self.depth_min_m,
            self.depth_max_m,
            self.edge_jump_m,
            self.edge_surface_band_m,
            self.table_clearance_m,
            self.voxel_size_m,
            *self.workspace,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("point-cloud configuration values must be finite")
        if not 0.0 < self.depth_min_m < self.depth_max_m:
            raise ValueError("depth range must satisfy 0 < depth_min_m < depth_max_m")
        if self.edge_jump_m <= 0.0 or self.edge_surface_band_m < 0.0:
            raise ValueError("edge thresholds must be non-negative with positive jump")
        if self.table_clearance_m < 0.0:
            raise ValueError("table_clearance_m must be non-negative")
        if self.voxel_size_m <= 0.0:
            raise ValueError("voxel_size_m must be positive")
        lower = self.workspace[:3]
        upper = self.workspace[3:]
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise ValueError(
                "workspace lower bounds must be strictly below upper bounds"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable processing-policy representation."""
        return {
            "num_points": self.num_points,
            "depth_min_m": self.depth_min_m,
            "depth_max_m": self.depth_max_m,
            "edge_jump_m": self.edge_jump_m,
            "edge_surface_band_m": self.edge_surface_band_m,
            "table_clearance_m": self.table_clearance_m,
            "workspace": list(self.workspace),
            "voxel_size_m": self.voxel_size_m,
        }


def _depth_valid_mask(
    depth_raw: np.ndarray, *, depth_scale_m: float, config: PointCloudConfig
) -> tuple[np.ndarray, np.ndarray]:
    if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
        raise ValueError("depth_raw must be a native uint16 [H,W] frame")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be finite and positive")
    depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale_m)
    valid = (
        (depth_raw > 0)
        & np.isfinite(depth_m)
        & (depth_m >= config.depth_min_m)
        & (depth_m <= config.depth_max_m)
    )
    return depth_m, valid


def _reject_flying_depth(
    depth_m: np.ndarray, valid: np.ndarray, config: PointCloudConfig
) -> np.ndarray:
    """Reject only intermediate depths at local discontinuities.

    Foreground/background endpoint samples remain eligible. The production
    path deliberately does not combine this rule with median, speckle,
    temporal, cluster, or statistical-outlier filters.
    """
    if depth_m.shape != valid.shape:
        raise ValueError("depth and valid mask shapes must match")
    if not np.any(valid):
        return valid

    local_min = minimum_filter(
        np.where(valid, depth_m, np.inf), size=3, mode="constant", cval=np.inf
    )
    local_max = maximum_filter(
        np.where(valid, depth_m, -np.inf), size=3, mode="constant", cval=-np.inf
    )
    local_valid_count = convolve(
        valid.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
    )
    endpoint_distance = np.minimum(depth_m - local_min, local_max - depth_m)
    flying = (
        valid
        & (local_valid_count >= 3)
        & ((local_max - local_min) > config.edge_jump_m)
        & (endpoint_distance > config.edge_surface_band_m)
    )
    return valid & ~flying


def _deproject_depth(
    depth_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized deprojection for the measured L515 native depth NONE model."""
    if intrinsics.distortion_model not in {"distortion.none", "none"}:
        raise ValueError(
            "native depth deprojection only supports the measured NONE distortion "
            f"model, got {intrinsics.distortion_model!r}"
        )
    if (
        depth_m.shape != (intrinsics.height, intrinsics.width)
        or valid.shape != depth_m.shape
    ):
        raise ValueError("depth frame shape does not match depth intrinsics")
    rows, columns = np.nonzero(valid)
    z = depth_m[rows, columns]
    x = (
        (columns.astype(np.float32) - np.float32(intrinsics.ppx))
        * z
        / np.float32(intrinsics.fx)
    )
    y = (
        (rows.astype(np.float32) - np.float32(intrinsics.ppy))
        * z
        / np.float32(intrinsics.fy)
    )
    return np.column_stack((x, y, z)).astype(np.float32), rows, columns


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
    ):
        raise ValueError("transform must be a finite homogeneous 4x4 matrix")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    return (points @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)


def _crop_points(
    points_base: np.ndarray,
    *,
    table_plane_abcd: tuple[float, float, float, float] | None,
    table_clearance_m: float,
    workspace: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    if not np.isfinite(table_clearance_m):
        raise ValueError("table_clearance_m must be finite")
    keep = np.ones(points_base.shape[0], dtype=bool)
    if table_plane_abcd is not None:
        plane = np.asarray(table_plane_abcd, dtype=np.float64)
        if plane.shape != (4,) or not np.all(np.isfinite(plane)):
            raise ValueError("table_plane_abcd must contain four finite values")
        norm = np.linalg.norm(plane[:3])
        if norm <= 0.0:
            raise ValueError("table plane normal must be non-zero")
        plane /= norm
        keep &= points_base @ plane[:3] + plane[3] >= table_clearance_m
    lower = np.asarray(workspace[:3], dtype=np.float32)
    upper = np.asarray(workspace[3:], dtype=np.float32)
    keep &= np.all((points_base >= lower) & (points_base <= upper), axis=1)
    return keep


def _voxel_representatives(points_base: np.ndarray, voxel_size_m: float) -> np.ndarray:
    if points_base.ndim != 2 or points_base.shape[1] != 3:
        raise ValueError("points_base must have shape [N,3]")
    if points_base.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    keys = np.floor(points_base / np.float32(voxel_size_m)).astype(np.int64)
    # Group exact 3-int rows through one packed scalar key. This preserves the
    # same first-sample-per-voxel result as ``unique(..., axis=0)`` while
    # avoiding NumPy's substantially slower generic multi-axis sort path.
    packed_dtype = np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
    packed_keys = np.ascontiguousarray(keys).view(packed_dtype).reshape(-1)
    _, first_indices = np.unique(packed_keys, return_index=True)
    return np.sort(first_indices.astype(np.int64))


def _project_color(
    points_color: np.ndarray, intrinsics: CameraIntrinsics
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project color-camera points with the observed Brown-Conrady model."""
    if points_color.ndim != 2 or points_color.shape[1] != 3:
        raise ValueError("points_color must have shape [N,3]")
    z = points_color[:, 2]
    valid = np.isfinite(points_color).all(axis=1) & (z > 0.0)
    x = np.zeros_like(z, dtype=np.float32)
    y = np.zeros_like(z, dtype=np.float32)
    x[valid] = points_color[valid, 0] / z[valid]
    y[valid] = points_color[valid, 1] / z[valid]
    if intrinsics.distortion_model in {"distortion.none", "none"}:
        xd, yd = x, y
    elif intrinsics.distortion_model in {"distortion.brown_conrady", "brown_conrady"}:
        k1, k2, p1, p2, k3 = (
            np.float32(value) for value in intrinsics.distortion_coeffs
        )
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    else:
        raise ValueError(
            "color projection has no validated path for distortion model "
            f"{intrinsics.distortion_model!r}"
        )
    columns = np.rint(
        xd * np.float32(intrinsics.fx) + np.float32(intrinsics.ppx)
    ).astype(np.int64)
    rows = np.rint(yd * np.float32(intrinsics.fy) + np.float32(intrinsics.ppy)).astype(
        np.int64
    )
    valid &= (
        (columns >= 0)
        & (columns < intrinsics.width)
        & (rows >= 0)
        & (rows < intrinsics.height)
    )
    return rows, columns, valid


def _color_visibility(
    rows: np.ndarray,
    columns: np.ndarray,
    points_color: np.ndarray,
    valid_projection: np.ndarray,
) -> np.ndarray:
    """Keep nearest depth-derived candidate at each color texel (0 m tolerance)."""
    visible = np.zeros(valid_projection.shape, dtype=bool)
    if not np.any(valid_projection):
        return visible
    # Width is not passed because pixel IDs need only unique ordered pairs here.
    coordinates = np.column_stack((rows[valid_projection], columns[valid_projection]))
    _, inverse = np.unique(coordinates, axis=0, return_inverse=True)
    depth = points_color[valid_projection, 2]
    nearest = np.full(int(inverse.max()) + 1, np.inf, dtype=np.float32)
    np.minimum.at(nearest, inverse, depth)
    visible_indices = np.flatnonzero(valid_projection)
    visible[visible_indices] = depth <= nearest[inverse]
    return visible


def _fixed_size_sample(cloud: np.ndarray, num_points: int) -> np.ndarray:
    if cloud.ndim != 2 or cloud.shape[1] != 6:
        raise ValueError("cloud must have shape [N,6]")
    count = cloud.shape[0]
    if count == 0:
        raise ValueError("cannot sample an empty cloud")
    if count == num_points:
        return np.ascontiguousarray(cloud, dtype=np.float32)
    if count > num_points:
        indices = (np.arange(num_points, dtype=np.int64) * count) // num_points
    else:
        indices = np.arange(num_points, dtype=np.int64) % count
    return np.ascontiguousarray(cloud[indices], dtype=np.float32)


def build_point_cloud(
    *,
    depth_raw: np.ndarray,
    color: np.ndarray,
    depth_scale_m: float,
    geometry: RGBDGeometry,
    T_xarm_base_from_depth: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float] | None,
    config: PointCloudConfig,
) -> np.ndarray | None:
    """Build native depth/color ``float32[num_points,6]`` in xArm-base frame."""
    if color.dtype != np.uint8 or color.shape != (
        geometry.color.height,
        geometry.color.width,
        3,
    ):
        raise ValueError("color must be native uint8 [H,W,3] matching color intrinsics")
    depth_m, valid = _depth_valid_mask(
        depth_raw, depth_scale_m=depth_scale_m, config=config
    )
    trusted = _reject_flying_depth(depth_m, valid, config)
    if not np.any(trusted):
        return None
    points_depth, _rows, _columns = _deproject_depth(depth_m, trusted, geometry.depth)
    points_base = _transform_points(points_depth, T_xarm_base_from_depth)
    crop = _crop_points(
        points_base,
        table_plane_abcd=table_plane_abcd,
        table_clearance_m=config.table_clearance_m,
        workspace=config.workspace,
    )
    if not np.any(crop):
        return None
    points_depth = points_depth[crop]
    points_base = points_base[crop]
    representatives = _voxel_representatives(points_base, config.voxel_size_m)
    if representatives.size == 0:
        return None
    points_depth = points_depth[representatives]
    points_base = points_base[representatives]
    points_color = _transform_points(points_depth, geometry.T_color_from_depth)
    color_rows, color_columns, projection_valid = _project_color(
        points_color, geometry.color
    )
    visible = _color_visibility(
        color_rows, color_columns, points_color, projection_valid
    )
    if not np.any(visible):
        return None
    colors = (
        color[color_rows[visible], color_columns[visible]].astype(np.float32) / 255.0
    )
    cloud = np.column_stack((points_base[visible], colors)).astype(np.float32)
    return _fixed_size_sample(cloud, config.num_points)

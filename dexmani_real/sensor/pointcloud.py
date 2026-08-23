"""Pure native RGB-D to xArm-base point-cloud construction.

The module owns only deterministic numerical geometry.  It does not import a
camera SDK, Open3D, shared memory, calibration files, or recording code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import (  # type: ignore[import-untyped]
    convolve,
    maximum_filter,
    minimum_filter,
)
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.sensor.camera_geometry import CameraIntrinsics, RGBDGeometry

POINT_CLOUD_POLICY_ID = "native_support_mean_voxel_radius_v1"
POINT_CLOUD_COLOR_SOURCE = "native_color_projection"
POINT_CLOUD_SAMPLING = "deterministic_spatial_hash_or_cyclic_pad"
POINT_CLOUD_TRANSFORM = (
    "depth_gate_and_local_support;native_depth_deprojection;"
    "table_plane_crop_in_depth_frame;xarm_base_transform;workspace_crop;"
    "mean_voxel;spatial_candidate_cap;radius_outlier;color_projection_and_visibility;"
    "spatial_hash_or_cyclic_pad"
)

__all__ = [
    "POINT_CLOUD_COLOR_SOURCE",
    "POINT_CLOUD_POLICY_ID",
    "POINT_CLOUD_SAMPLING",
    "POINT_CLOUD_TRANSFORM",
    "PointCloudBuildStats",
    "PointCloudConfig",
    "build_raw_point_cloud",
    "build_point_cloud",
    "build_point_cloud_with_stats",
    "depth_points_in_base",
]


@dataclass(frozen=True)
class PointCloudBuildStats:
    """Cheap stage counts for diagnostics without copying intermediate clouds."""

    depth_valid_points: int = 0
    depth_trusted_points: int = 0
    cropped_points: int = 0
    voxel_points: int = 0
    outlier_candidate_points: int = 0
    outlier_inlier_points: int = 0
    color_visible_points: int = 0
    unique_output_points: int = 0
    padded_output_points: int = 0
    failure_stage: str | None = None


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


def _validate_color(color: np.ndarray, geometry: RGBDGeometry) -> None:
    if color.dtype != np.uint8 or color.shape != (
        geometry.color.height,
        geometry.color.width,
        3,
    ):
        raise ValueError("color must be native uint8 [H,W,3] matching color intrinsics")


def _reject_flying_depth(
    depth_m: np.ndarray, valid: np.ndarray, config: PointCloudConfig
) -> np.ndarray:
    """Reject intermediate edge depths and locally unsupported measurements."""
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
    trusted = valid & ~flying
    if config.depth_support_min_neighbors == 0:
        return trusted

    padded_depth = np.pad(depth_m, 1, mode="constant", constant_values=np.nan)
    padded_valid = np.pad(trusted, 1, mode="constant", constant_values=False)
    height, width = depth_m.shape
    support_count = np.zeros(depth_m.shape, dtype=np.uint8)
    for row_offset in range(3):
        for column_offset in range(3):
            if row_offset == 1 and column_offset == 1:
                continue
            neighbor_depth = padded_depth[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            neighbor_valid = padded_valid[
                row_offset : row_offset + height,
                column_offset : column_offset + width,
            ]
            support_count += neighbor_valid & (
                np.abs(neighbor_depth - depth_m) <= config.edge_surface_band_m
            )
    return trusted & (support_count >= config.depth_support_min_neighbors)


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


def _validated_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    rotation = matrix[:3, :3] if matrix.shape == (4, 4) else np.empty((0, 0))
    if (
        matrix.shape != (4, 4)
        or not np.all(np.isfinite(matrix))
        or not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6)
    ):
        raise ValueError("transform must be a finite rigid homogeneous 4x4 matrix")
    return matrix


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    matrix = _validated_transform(transform)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N,3]")
    return (points @ matrix[:3, :3].T + matrix[:3, 3]).astype(np.float32)


def _table_keep_mask(
    points_depth: np.ndarray,
    *,
    depth_rows: np.ndarray,
    depth_columns: np.ndarray,
    depth_shape: tuple[int, int],
    T_xarm_base_from_depth: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float] | None,
    table_clearance_m: float,
    table_support_min_pixels: int,
) -> np.ndarray:
    if not np.isfinite(table_clearance_m):
        raise ValueError("table_clearance_m must be finite")
    if table_plane_abcd is None:
        return np.ones(points_depth.shape[0], dtype=bool)
    plane = np.asarray(table_plane_abcd, dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise ValueError("table_plane_abcd must contain four finite values")
    norm = np.linalg.norm(plane[:3])
    if norm <= 0.0 or plane[2] / norm <= 0.0:
        raise ValueError("table plane must have an upward non-zero normal")
    plane /= norm

    # Transform the plane into the depth frame. This removes the large table
    # support before transforming every surviving point into xArm-base.
    plane_depth = _validated_transform(T_xarm_base_from_depth).T @ plane
    signed_height = points_depth @ plane_depth[:3] + plane_depth[3]
    below_table = signed_height < -table_clearance_m
    near_table = np.abs(signed_height) <= table_clearance_m

    # A calibrated plane may be a few millimetres imperfect, but a real object
    # boundary can have the same height. Remove only locally supported plane
    # pixels and leave isolated candidates to the later 3-D outlier decision.
    near_table_image = np.zeros(depth_shape, dtype=np.uint8)
    near_table_image[depth_rows[near_table], depth_columns[near_table]] = 1
    support = convolve(
        near_table_image, np.ones((3, 3), dtype=np.uint8), mode="constant"
    )
    supported_table = near_table & (
        support[depth_rows, depth_columns] >= table_support_min_pixels
    )
    return ~(below_table | supported_table)


def _workspace_keep_mask(
    points_base: np.ndarray,
    workspace: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    lower = np.asarray(workspace[:3], dtype=np.float32)
    upper = np.asarray(workspace[3:], dtype=np.float32)
    return np.all((points_base >= lower) & (points_base <= upper), axis=1)


def _voxel_means(
    points_depth: np.ndarray,
    points_base: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_base.ndim != 2 or points_base.shape[1] != 3:
        raise ValueError("points_base must have shape [N,3]")
    if points_base.shape[0] == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        return empty_points, empty_points.copy(), np.empty((0, 3), dtype=np.int64)
    keys = np.floor(points_base / np.float32(voxel_size_m)).astype(np.int64)
    # Group exact 3-int rows through one packed scalar key. Mean aggregation is
    # independent of depth-image traversal order and avoids first-pixel bias.
    packed_dtype = np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))
    packed_keys = np.ascontiguousarray(keys).view(packed_dtype).reshape(-1)
    _, first_indices, inverse = np.unique(
        packed_keys, return_index=True, return_inverse=True
    )
    counts = np.bincount(inverse).astype(np.float64)

    def aggregate(points: np.ndarray) -> np.ndarray:
        columns = [
            np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)
        ]
        return np.column_stack(columns).astype(np.float32)

    return aggregate(points_depth), aggregate(points_base), keys[first_indices]


def _radius_inlier_mask(
    query_points_base: np.ndarray,
    *,
    reference_points_base: np.ndarray,
    radius_m: float,
    min_neighbors: int,
) -> np.ndarray:
    """Return a conservative radius-neighbor mask after voxel reduction."""
    if query_points_base.shape[0] == 0:
        return np.empty(0, dtype=bool)
    if min_neighbors == 0:
        return np.ones(query_points_base.shape[0], dtype=bool)
    neighbor_counts = np.asarray(
        cKDTree(reference_points_base).query_ball_point(
            query_points_base, r=radius_m, return_length=True
        )
    )
    # query_ball_point includes the query point itself.
    return neighbor_counts >= min_neighbors + 1


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


def _spatial_hash(voxel_keys: np.ndarray) -> np.ndarray:
    unsigned = np.ascontiguousarray(voxel_keys, dtype=np.int64).view(np.uint64)
    hashed = (
        unsigned[:, 0] * np.uint64(0x9E3779B185EBCA87)
        ^ unsigned[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)
        ^ unsigned[:, 2] * np.uint64(0x165667B19E3779F9)
    )
    hashed ^= hashed >> np.uint64(30)
    hashed *= np.uint64(0xBF58476D1CE4E5B9)
    hashed ^= hashed >> np.uint64(27)
    hashed *= np.uint64(0x94D049BB133111EB)
    return hashed ^ (hashed >> np.uint64(31))


def _spatial_candidate_indices(
    voxel_keys: np.ndarray, max_candidates: int
) -> np.ndarray:
    count = voxel_keys.shape[0]
    if count <= max_candidates:
        return np.arange(count, dtype=np.int64)
    hashes = _spatial_hash(voxel_keys)
    selected = np.argpartition(hashes, max_candidates - 1)[:max_candidates]
    return selected[np.argsort(hashes[selected], kind="stable")]


def _fixed_size_sample(
    cloud: np.ndarray, voxel_keys: np.ndarray, num_points: int
) -> np.ndarray:
    if cloud.ndim != 2 or cloud.shape[1] != 6:
        raise ValueError("cloud must have shape [N,6]")
    count = cloud.shape[0]
    if count == 0:
        raise ValueError("cannot sample an empty cloud")
    if voxel_keys.shape != (count, 3):
        raise ValueError("voxel_keys must have shape [N,3] matching cloud")
    spatial_order = np.argsort(_spatial_hash(voxel_keys), kind="stable")
    if count > num_points:
        indices = spatial_order[:num_points]
    else:
        indices = spatial_order[np.arange(num_points, dtype=np.int64) % count]
    return np.ascontiguousarray(cloud[indices], dtype=np.float32)


def depth_points_in_base(
    *,
    depth_raw: np.ndarray,
    depth_scale_m: float,
    depth_intrinsics: CameraIntrinsics,
    T_xarm_base_from_depth: np.ndarray,
    config: PointCloudConfig,
) -> np.ndarray:
    """Return supported, depth-gated points for offline table calibration."""
    if not isinstance(config, PointCloudConfig):
        raise TypeError("config must be a PointCloudConfig")
    depth_m, valid = _depth_valid_mask(
        depth_raw, depth_scale_m=depth_scale_m, config=config
    )
    trusted = _reject_flying_depth(depth_m, valid, config)
    if not np.any(trusted):
        return np.empty((0, 3), dtype=np.float32)
    points_depth, _rows, _columns = _deproject_depth(depth_m, trusted, depth_intrinsics)
    return _transform_points(points_depth, T_xarm_base_from_depth)


def build_raw_point_cloud(
    *,
    depth_raw: np.ndarray,
    color: np.ndarray,
    depth_scale_m: float,
    geometry: RGBDGeometry,
    T_xarm_base_from_depth: np.ndarray,
) -> np.ndarray | None:
    """Build the diagnostic full measured cloud without production filtering.

    Every finite non-zero native depth sample is retained. RGB is projected
    from the native color stream; samples outside its field of view are gray.
    This function is intentionally not used by recording or policy workers.
    """
    _validate_color(color, geometry)
    if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
        raise ValueError("depth_raw must be a native uint16 [H,W] frame")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be finite and positive")
    depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale_m)
    valid = (depth_raw > 0) & np.isfinite(depth_m)
    if not np.any(valid):
        return None
    points_depth, _rows, _columns = _deproject_depth(depth_m, valid, geometry.depth)
    points_base = _transform_points(points_depth, T_xarm_base_from_depth)
    points_color = _transform_points(points_depth, geometry.T_color_from_depth)
    color_rows, color_columns, projection_valid = _project_color(
        points_color, geometry.color
    )
    colors = np.full(points_base.shape, 0.5, dtype=np.float32)
    colors[projection_valid] = (
        color[
            color_rows[projection_valid],
            color_columns[projection_valid],
        ].astype(np.float32)
        / 255.0
    )
    return np.ascontiguousarray(
        np.column_stack((points_base, colors)), dtype=np.float32
    )


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
    cloud, _stats = build_point_cloud_with_stats(
        depth_raw=depth_raw,
        color=color,
        depth_scale_m=depth_scale_m,
        geometry=geometry,
        T_xarm_base_from_depth=T_xarm_base_from_depth,
        table_plane_abcd=table_plane_abcd,
        config=config,
    )
    return cloud


def build_point_cloud_with_stats(
    *,
    depth_raw: np.ndarray,
    color: np.ndarray,
    depth_scale_m: float,
    geometry: RGBDGeometry,
    T_xarm_base_from_depth: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float] | None,
    config: PointCloudConfig,
) -> tuple[np.ndarray | None, PointCloudBuildStats]:
    """Build a cloud and return allocation-light stage diagnostics."""
    if not isinstance(config, PointCloudConfig):
        raise TypeError("config must be a PointCloudConfig")
    _validate_color(color, geometry)
    depth_m, valid = _depth_valid_mask(
        depth_raw, depth_scale_m=depth_scale_m, config=config
    )
    valid_count = int(np.count_nonzero(valid))
    trusted = _reject_flying_depth(depth_m, valid, config)
    trusted_count = int(np.count_nonzero(trusted))
    if not np.any(trusted):
        return None, PointCloudBuildStats(
            depth_valid_points=valid_count,
            failure_stage="depth_support",
        )
    points_depth, rows, columns = _deproject_depth(depth_m, trusted, geometry.depth)
    if table_plane_abcd is not None:
        table_keep = _table_keep_mask(
            points_depth,
            depth_rows=rows,
            depth_columns=columns,
            depth_shape=(int(depth_m.shape[0]), int(depth_m.shape[1])),
            T_xarm_base_from_depth=T_xarm_base_from_depth,
            table_plane_abcd=table_plane_abcd,
            table_clearance_m=config.table_clearance_m,
            table_support_min_pixels=config.table_support_min_pixels,
        )
        if not np.any(table_keep):
            return None, PointCloudBuildStats(
                depth_valid_points=valid_count,
                depth_trusted_points=trusted_count,
                failure_stage="table_workspace_crop",
            )
        points_depth = points_depth[table_keep]
    points_base = _transform_points(points_depth, T_xarm_base_from_depth)
    workspace_keep = _workspace_keep_mask(points_base, config.workspace)
    cropped_count = int(np.count_nonzero(workspace_keep))
    if not np.any(workspace_keep):
        return None, PointCloudBuildStats(
            depth_valid_points=valid_count,
            depth_trusted_points=trusted_count,
            failure_stage="table_workspace_crop",
        )
    points_depth = points_depth[workspace_keep]
    points_base = points_base[workspace_keep]
    points_depth, points_base, voxel_keys = _voxel_means(
        points_depth, points_base, config.voxel_size_m
    )
    voxel_count = points_base.shape[0]
    if voxel_count == 0:
        return None, PointCloudBuildStats(
            depth_valid_points=valid_count,
            depth_trusted_points=trusted_count,
            cropped_points=cropped_count,
            failure_stage="voxelization",
        )
    candidate_indices = _spatial_candidate_indices(
        voxel_keys, config.num_points * config.outlier_candidate_multiplier
    )
    candidate_count = int(candidate_indices.size)
    reference_points_base = points_base
    points_depth = points_depth[candidate_indices]
    points_base = points_base[candidate_indices]
    voxel_keys = voxel_keys[candidate_indices]
    inlier = _radius_inlier_mask(
        points_base,
        reference_points_base=reference_points_base,
        radius_m=config.outlier_radius_m,
        min_neighbors=config.outlier_min_neighbors,
    )
    inlier_count = int(np.count_nonzero(inlier))
    if not np.any(inlier):
        return None, PointCloudBuildStats(
            depth_valid_points=valid_count,
            depth_trusted_points=trusted_count,
            cropped_points=cropped_count,
            voxel_points=voxel_count,
            outlier_candidate_points=candidate_count,
            failure_stage="radius_outlier",
        )
    points_depth = points_depth[inlier]
    points_base = points_base[inlier]
    voxel_keys = voxel_keys[inlier]
    points_color = _transform_points(points_depth, geometry.T_color_from_depth)
    color_rows, color_columns, projection_valid = _project_color(
        points_color, geometry.color
    )
    visible = _color_visibility(
        color_rows, color_columns, points_color, projection_valid
    )
    visible_count = int(np.count_nonzero(visible))
    if not np.any(visible):
        return None, PointCloudBuildStats(
            depth_valid_points=valid_count,
            depth_trusted_points=trusted_count,
            cropped_points=cropped_count,
            voxel_points=voxel_count,
            outlier_candidate_points=candidate_count,
            outlier_inlier_points=inlier_count,
            failure_stage="color_visibility",
        )
    colors = (
        color[color_rows[visible], color_columns[visible]].astype(np.float32) / 255.0
    )
    cloud = np.column_stack((points_base[visible], colors)).astype(np.float32)
    result = _fixed_size_sample(cloud, voxel_keys[visible], config.num_points)
    return result, PointCloudBuildStats(
        depth_valid_points=valid_count,
        depth_trusted_points=trusted_count,
        cropped_points=cropped_count,
        voxel_points=voxel_count,
        outlier_candidate_points=candidate_count,
        outlier_inlier_points=inlier_count,
        color_visible_points=visible_count,
        unique_output_points=min(visible_count, config.num_points),
        padded_output_points=max(0, config.num_points - visible_count),
    )

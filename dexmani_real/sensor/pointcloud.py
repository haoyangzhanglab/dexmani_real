"""Pure depth-to-color aligned RGB-D to xArm-base point-cloud construction.

The module owns only deterministic numerical geometry.  It does not import a
camera SDK, Open3D, shared memory, calibration files, or recording code.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, cast

import cv2
import numpy as np
from scipy.ndimage import (
    label as label_connected_components,  # type: ignore[import-untyped]
)
from scipy.sparse import coo_matrix  # type: ignore[import-untyped]
from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.sensor.camera_geometry import CameraIntrinsics, RGBDGeometry

_KERNEL_3X3 = np.ones((3, 3), dtype=np.uint8)

POINT_CLOUD_POLICY_ID = "depth_to_color_orthogonal_edge_table_voxel_radius_graph_v9"
POINT_CLOUD_COLOR_SOURCE = "mean_rgb_of_aligned_depth_pixels_per_voxel"
POINT_CLOUD_SAMPLING = "deterministic_coarse_voxel_stratified_hash_or_cyclic_pad"
POINT_CLOUD_TRANSFORM = (
    "depth_gate_and_cardinal_edge_support;depth_to_color_deprojection;"
    "table_plane_height_hysteresis_crop_in_color_frame_before_deprojection;"
    "xarm_base_transform;workspace_crop;mean_voxel_xyz_and_rgb;"
    "single_radius_graph_density_and_component_outlier;spatial_candidate_cap;"
    "coarse_voxel_stratified_hash_or_cyclic_pad"
)

__all__ = [
    "POINT_CLOUD_COLOR_SOURCE",
    "POINT_CLOUD_POLICY_ID",
    "POINT_CLOUD_SAMPLING",
    "POINT_CLOUD_TRANSFORM",
    "PointCloudBuildTimings",
    "PointCloudBuildStats",
    "PointCloudConfig",
    "build_raw_point_cloud",
    "build_point_cloud",
    "build_point_cloud_with_stats",
    "aligned_depth_points_in_base",
]


@dataclass(frozen=True)
class PointCloudBuildTimings:
    """Per-frame elapsed times for the deterministic production stages."""

    depth_filter_ms: float = 0.0
    table_crop_ms: float = 0.0
    deprojection_ms: float = 0.0
    base_workspace_ms: float = 0.0
    voxelization_ms: float = 0.0
    spatial_outlier_filter_ms: float = 0.0
    color_sampling_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True)
class PointCloudBuildStats:
    """Stage counts and elapsed times without retaining intermediate clouds."""

    depth_valid_points: int = 0
    depth_trusted_points: int = 0
    table_rejected_points: int = 0
    workspace_rejected_points: int = 0
    cropped_points: int = 0
    voxel_points: int = 0
    radius_density_points: int = 0
    spatial_inlier_points: int = 0
    candidate_points: int = 0
    failure_stage: str | None = None
    timings: PointCloudBuildTimings = field(default_factory=PointCloudBuildTimings)


def _depth_valid_mask(
    depth_raw: np.ndarray, *, depth_scale_m: float, config: PointCloudConfig
) -> tuple[np.ndarray, np.ndarray]:
    if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
        raise ValueError("depth_raw must be a uint16 [H,W] frame")
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
        raise ValueError("color must be uint8 [H,W,3] matching color intrinsics")


def _local_3x3_count(mask: np.ndarray) -> np.ndarray:
    """Count true pixels in each clipped 3x3 neighborhood."""
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a boolean [H,W] array")
    return cv2.boxFilter(
        mask.astype(np.uint8),
        ddepth=cv2.CV_8U,
        ksize=(3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )


def _reject_flying_depth(
    depth_m: np.ndarray, valid: np.ndarray, config: PointCloudConfig
) -> np.ndarray:
    """Reject intermediate edge depths and locally unsupported measurements."""
    if depth_m.shape != valid.shape:
        raise ValueError("depth and valid mask shapes must match")
    if not np.any(valid):
        return valid

    local_min = cast(
        np.ndarray,
        cv2.erode(
            np.where(valid, depth_m, np.float32(np.inf)),
            _KERNEL_3X3,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=(float("inf"),),
        ),
    )
    local_max = cast(
        np.ndarray,
        cv2.dilate(
            np.where(valid, depth_m, np.float32(-np.inf)),
            _KERNEL_3X3,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=(float("-inf"),),
        ),
    )
    local_valid_count = _local_3x3_count(valid)
    endpoint_distance = np.minimum(depth_m - local_min, local_max - depth_m)
    depth_discontinuity = (
        valid
        & (local_valid_count >= 3)
        & ((local_max - local_min) > config.edge_jump_m)
    )
    flying = depth_discontinuity & (endpoint_distance > config.edge_surface_band_m)
    trusted = valid & ~flying
    if (
        config.depth_support_min_neighbors == 0
        and config.edge_support_min_neighbors == 0
    ):
        return trusted

    padded_depth = np.pad(depth_m, 1, mode="constant", constant_values=np.nan)
    padded_valid = np.pad(trusted, 1, mode="constant", constant_values=False)
    height, width = depth_m.shape
    support_count = np.zeros(depth_m.shape, dtype=np.uint8)
    cardinally_supported = np.ones(depth_m.shape, dtype=bool)
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
            same_surface = neighbor_valid & (
                np.abs(neighbor_depth - depth_m) <= config.edge_surface_band_m
            )
            support_count += same_surface
            if (row_offset == 1) != (column_offset == 1):
                cardinally_supported &= same_surface
    required_support = np.where(
        depth_discontinuity,
        config.edge_support_min_neighbors,
        config.depth_support_min_neighbors,
    )
    # Erode only the unreliable one-pixel layer at a depth discontinuity.
    # Resolved objects keep their non-discontinuity interiors; one- and
    # two-pixel structures at an edge are intentionally treated as unreliable.
    return (
        trusted
        & (support_count >= required_support)
        & (~depth_discontinuity | cardinally_supported)
    )


def _undistorted_depth_coordinates(
    rows: np.ndarray,
    columns: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized camera rays for color-grid depth pixels."""
    xd = (columns.astype(np.float32) - np.float32(intrinsics.ppx)) / np.float32(
        intrinsics.fx
    )
    yd = (rows.astype(np.float32) - np.float32(intrinsics.ppy)) / np.float32(
        intrinsics.fy
    )
    if intrinsics.distortion_model in {"distortion.none", "none"}:
        x, y = xd, yd
    elif intrinsics.distortion_model in {"distortion.brown_conrady", "brown_conrady"}:
        # Invert the Brown-Conrady projection with the same fixed-point scheme
        # used by librealsense for deprojection.
        k1, k2, p1, p2, k3 = (
            np.float32(value) for value in intrinsics.distortion_coeffs
        )
        x, y = xd.copy(), yd.copy()
        for _ in range(10):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (xd - delta_x) / radial
            y = (yd - delta_y) / radial
    else:
        raise ValueError(
            "depth deprojection has no validated path for distortion model "
            f"{intrinsics.distortion_model!r}"
        )
    return x, y


@lru_cache(maxsize=4)
def _depth_rays(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Cache one deprojection ray per static aligned depth pixel."""
    rows, columns = np.indices((intrinsics.height, intrinsics.width), dtype=np.int64)
    x, y = _undistorted_depth_coordinates(
        rows.reshape(-1), columns.reshape(-1), intrinsics
    )
    rays = np.column_stack((x, y, np.ones(x.size, dtype=np.float32))).reshape(
        intrinsics.height, intrinsics.width, 3
    )
    rays.setflags(write=False)
    return rays


@lru_cache(maxsize=4)
def _ray_plane_factors(
    intrinsics: CameraIntrinsics,
    plane_normal_color: tuple[float, float, float],
) -> np.ndarray:
    """Cache the static ray/plane-normal dot product for one calibration."""
    factors = _depth_rays(intrinsics) @ np.asarray(plane_normal_color, dtype=np.float64)
    factors.setflags(write=False)
    return factors


def _deproject_depth(
    depth_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized deprojection for depth-to-color aligned depth."""
    if (
        depth_m.shape != (intrinsics.height, intrinsics.width)
        or valid.shape != depth_m.shape
    ):
        raise ValueError("depth frame shape does not match depth intrinsics")
    rows, columns = np.nonzero(valid)
    z = depth_m[rows, columns]
    rays = _depth_rays(intrinsics)[rows, columns]
    return np.ascontiguousarray(rays * z[:, None], dtype=np.float32), rows, columns


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


def _table_height_hysteresis_keep_mask(
    valid: np.ndarray,
    *,
    rows: np.ndarray,
    columns: np.ndarray,
    signed_height_m: np.ndarray,
    core_height_m: float,
    object_seed_height_m: float,
    object_seed_min_pixels: int,
) -> np.ndarray:
    """Preserve low object surfaces only when connected to reliable high seeds."""
    if valid.ndim != 2 or valid.dtype != np.bool_:
        raise ValueError("valid must be a boolean [H,W] mask")
    if rows.shape != columns.shape or rows.shape != signed_height_m.shape:
        raise ValueError("rows, columns, and signed heights must have equal shape")

    # The >core mask is normally sparse over a tabletop. One 8-connected label
    # pass is linear in image size and avoids iterative morphology. Requiring
    # several high seed pixels prevents an isolated depth spike from preserving
    # an otherwise table-like residual component.
    above_core = signed_height_m > core_height_m
    above_core_image = np.zeros(valid.shape, dtype=bool)
    above_core_image[rows[above_core], columns[above_core]] = True
    component_labels, component_count = label_connected_components(
        above_core_image,
        structure=_KERNEL_3X3,
    )
    if component_count == 0:
        return np.zeros(valid.shape, dtype=bool)

    object_seeds = signed_height_m >= object_seed_height_m
    valid_labels = component_labels[rows, columns]
    seed_labels = valid_labels[object_seeds]
    seed_counts = np.bincount(seed_labels, minlength=component_count + 1)
    keep_component = seed_counts >= object_seed_min_pixels
    keep_component[0] = False

    keep = np.zeros(valid.shape, dtype=bool)
    keep[rows, columns] = keep_component[valid_labels]
    return keep


def _table_keep_mask(
    depth_m: np.ndarray,
    valid: np.ndarray,
    *,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float] | None,
    table_core_height_m: float,
    table_object_seed_height_m: float,
    table_object_seed_min_pixels: int,
) -> np.ndarray:
    """Keep non-table valid pixels before allocating 3-D point arrays."""
    if (
        depth_m.shape != (intrinsics.height, intrinsics.width)
        or valid.shape != depth_m.shape
    ):
        raise ValueError("depth and valid mask must match depth intrinsics")
    if not np.isfinite(table_core_height_m) or not np.isfinite(
        table_object_seed_height_m
    ):
        raise ValueError("table crop heights must be finite")
    if table_core_height_m < 0.0 or table_object_seed_height_m <= table_core_height_m:
        raise ValueError("table crop heights must satisfy 0 <= core < object seed")
    if table_object_seed_min_pixels < 1:
        raise ValueError("table_object_seed_min_pixels must be positive")
    if table_plane_abcd is None:
        return valid.copy()
    plane = np.asarray(table_plane_abcd, dtype=np.float64)
    if plane.shape != (4,) or not np.all(np.isfinite(plane)):
        raise ValueError("table_plane_abcd must contain four finite values")
    norm = np.linalg.norm(plane[:3])
    if norm <= 0.0 or plane[2] / norm <= 0.0:
        raise ValueError("table plane must have an upward non-zero normal")
    plane /= norm

    # Transform the plane into the aligned color-camera frame. Cached rays make
    # ``z * dot(plane_normal, ray) + offset`` equivalent to evaluating the
    # plane after deprojection, without allocating table points in 3-D.
    plane_color = _validated_transform(T_xarm_base_from_color).T @ plane
    rows, columns = np.nonzero(valid)
    plane_normal_color = tuple(float(value) for value in plane_color[:3])
    ray_plane_factors = _ray_plane_factors(intrinsics, plane_normal_color)
    signed_height = (
        depth_m[rows, columns] * ray_plane_factors[rows, columns] + plane_color[3]
    )
    return _table_height_hysteresis_keep_mask(
        valid,
        rows=rows,
        columns=columns,
        signed_height_m=signed_height,
        core_height_m=table_core_height_m,
        object_seed_height_m=table_object_seed_height_m,
        object_seed_min_pixels=table_object_seed_min_pixels,
    )


def _workspace_keep_mask(
    points_base: np.ndarray,
    workspace: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    lower = np.asarray(workspace[:3], dtype=np.float32)
    upper = np.asarray(workspace[3:], dtype=np.float32)
    return np.all((points_base >= lower) & (points_base <= upper), axis=1)


def _packed_grid_keys(keys: np.ndarray) -> np.ndarray:
    """Pack integer XYZ rows into collision-free scalar group identifiers."""
    if keys.ndim != 2 or keys.shape[1] != 3 or keys.shape[0] == 0:
        raise ValueError("grid keys must have non-empty shape [N,3]")
    minimum = np.min(keys, axis=0)
    shifted = keys - minimum
    extents = np.max(shifted, axis=0) + 1
    extent_yz = int(extents[1]) * int(extents[2])
    volume = int(extents[0]) * extent_yz
    if volume > np.iinfo(np.int64).max:
        raise OverflowError("grid key volume exceeds int64")
    return shifted[:, 0] * extent_yz + shifted[:, 1] * int(extents[2]) + shifted[:, 2]


def _voxel_means(
    points_base: np.ndarray,
    colors: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_base.ndim != 2 or points_base.shape[1] != 3:
        raise ValueError("points_base must have shape [N,3]")
    if colors.shape != points_base.shape:
        raise ValueError("colors must have shape [N,3] matching points_base")
    if points_base.shape[0] == 0:
        empty_points = np.empty((0, 3), dtype=np.float32)
        return (
            empty_points.copy(),
            empty_points.copy(),
            np.empty((0, 3), dtype=np.int64),
        )
    keys = np.floor(points_base / np.float32(voxel_size_m)).astype(np.int64)
    # Mean aggregation is independent of depth-image traversal order and avoids
    # first-pixel bias. Bounded workspace keys use a fast collision-free int64.
    packed_keys = _packed_grid_keys(keys)
    _, first_indices, inverse = np.unique(
        packed_keys, return_index=True, return_inverse=True
    )
    counts = np.bincount(inverse).astype(np.float64)

    def aggregate(points: np.ndarray) -> np.ndarray:
        columns = [
            np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)
        ]
        return np.column_stack(columns).astype(np.float32)

    return (
        aggregate(points_base),
        aggregate(colors),
        keys[first_indices],
    )


def _radius_component_keep_masks(
    points_base: np.ndarray,
    *,
    radius_m: float,
    min_neighbors: int,
    min_component_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use one radius graph for connected-island and local-density filtering."""
    if points_base.ndim != 2 or points_base.shape[1] != 3:
        raise ValueError("points_base must have shape [N,3]")
    count = points_base.shape[0]
    if count == 0:
        empty = np.empty(0, dtype=bool)
        return empty, empty.copy()
    if min_neighbors == 0 and min_component_points <= 1:
        keep = np.ones(count, dtype=bool)
        return keep, keep.copy()

    # One undirected radius graph provides exact neighbor counts and physical
    # connected components.
    pairs = cKDTree(
        points_base,
        compact_nodes=False,
        balanced_tree=False,
    ).query_pairs(
        radius_m,
        output_type="ndarray",
    )
    neighbor_counts = np.bincount(pairs.reshape(-1), minlength=count)
    density_keep = neighbor_counts >= min_neighbors
    retained_pairs = pairs[density_keep[pairs[:, 0]] & density_keep[pairs[:, 1]]]
    graph = coo_matrix(
        (
            np.ones(retained_pairs.shape[0], dtype=np.uint8),
            (retained_pairs[:, 0], retained_pairs[:, 1]),
        ),
        shape=(count, count),
    )
    _component_count, labels = connected_components(
        graph, directed=False, return_labels=True
    )
    component_sizes = np.bincount(labels, minlength=count)
    component_keep = density_keep & (component_sizes[labels] >= min_component_points)
    return density_keep, component_keep


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


def _coarse_stratified_sample_indices(
    voxel_keys: np.ndarray,
    num_points: int,
    coarse_voxel_stride: int,
) -> np.ndarray:
    """Select broad spatial coverage, then fill remaining slots by fine hash."""
    count = voxel_keys.shape[0]
    if voxel_keys.shape != (count, 3):
        raise ValueError("voxel_keys must have shape [N,3]")
    if num_points <= 0 or coarse_voxel_stride <= 0:
        raise ValueError("sample count and coarse voxel stride must be positive")
    if count < num_points:
        raise ValueError("stratified sampling requires at least num_points candidates")

    fine_order = np.argsort(_spatial_hash(voxel_keys), kind="stable")
    coarse_keys = np.floor_divide(voxel_keys, coarse_voxel_stride)
    ordered_coarse_keys = coarse_keys[fine_order]
    packed_keys = _packed_grid_keys(ordered_coarse_keys)
    _, first_in_fine_order = np.unique(packed_keys, return_index=True)
    representatives = fine_order[first_in_fine_order]
    representatives = representatives[
        np.argsort(_spatial_hash(coarse_keys[representatives]), kind="stable")
    ]
    if representatives.size >= num_points:
        return representatives[:num_points]

    selected = np.zeros(count, dtype=bool)
    selected[representatives] = True
    fill = fine_order[~selected[fine_order]][: num_points - representatives.size]
    return np.concatenate((representatives, fill))


def _fixed_size_sample(
    cloud: np.ndarray,
    voxel_keys: np.ndarray,
    num_points: int,
    coarse_voxel_stride: int,
) -> np.ndarray:
    if cloud.ndim != 2 or cloud.shape[1] != 6:
        raise ValueError("cloud must have shape [N,6]")
    count = cloud.shape[0]
    if count == 0:
        raise ValueError("cannot sample an empty cloud")
    if voxel_keys.shape != (count, 3):
        raise ValueError("voxel_keys must have shape [N,3] matching cloud")
    if count > num_points:
        indices = _coarse_stratified_sample_indices(
            voxel_keys,
            num_points,
            coarse_voxel_stride,
        )
    else:
        spatial_order = np.argsort(_spatial_hash(voxel_keys), kind="stable")
        indices = spatial_order[np.arange(num_points, dtype=np.int64) % count]
    return np.ascontiguousarray(cloud[indices], dtype=np.float32)


def aligned_depth_points_in_base(
    *,
    depth_raw: np.ndarray,
    depth_scale_m: float,
    aligned_depth_intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
) -> np.ndarray:
    """Return aligned supported points in xArm-base for table calibration."""
    if not isinstance(config, PointCloudConfig):
        raise TypeError("config must be a PointCloudConfig")
    depth_m, valid = _depth_valid_mask(
        depth_raw, depth_scale_m=depth_scale_m, config=config
    )
    trusted = _reject_flying_depth(depth_m, valid, config)
    if not np.any(trusted):
        return np.empty((0, 3), dtype=np.float32)
    points_depth, _rows, _columns = _deproject_depth(
        depth_m, trusted, aligned_depth_intrinsics
    )
    return _transform_points(points_depth, T_xarm_base_from_color)


def build_raw_point_cloud(
    *,
    depth_raw: np.ndarray,
    color: np.ndarray,
    depth_scale_m: float,
    geometry: RGBDGeometry,
    T_xarm_base_from_color: np.ndarray,
) -> np.ndarray | None:
    """Build the diagnostic full aligned cloud without production filtering.

    Every finite non-zero depth-to-color sample is retained. RGB is read from
    the matching color pixel, with the aligned geometry's identity transform.
    This function is intentionally not used by recording or policy workers.
    """
    _validate_color(color, geometry)
    if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
        raise ValueError("depth_raw must be a uint16 [H,W] frame")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be finite and positive")
    depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale_m)
    valid = (depth_raw > 0) & np.isfinite(depth_m)
    if not np.any(valid):
        return None
    points_depth, rows, columns = _deproject_depth(depth_m, valid, geometry.depth)
    points_base = _transform_points(points_depth, T_xarm_base_from_color)
    colors = color[rows, columns].astype(np.float32) / 255.0
    return np.ascontiguousarray(
        np.column_stack((points_base, colors)), dtype=np.float32
    )


def build_point_cloud(
    *,
    depth_raw: np.ndarray,
    color: np.ndarray,
    depth_scale_m: float,
    geometry: RGBDGeometry,
    T_xarm_base_from_color: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float] | None,
    config: PointCloudConfig,
) -> np.ndarray | None:
    """Build aligned depth/color ``float32[num_points,6]`` in xArm-base frame."""
    cloud, _stats = build_point_cloud_with_stats(
        depth_raw=depth_raw,
        color=color,
        depth_scale_m=depth_scale_m,
        geometry=geometry,
        T_xarm_base_from_color=T_xarm_base_from_color,
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
    T_xarm_base_from_color: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float] | None,
    config: PointCloudConfig,
) -> tuple[np.ndarray | None, PointCloudBuildStats]:
    """Build a cloud and return allocation-light stage diagnostics."""
    if not isinstance(config, PointCloudConfig):
        raise TypeError("config must be a PointCloudConfig")
    _validate_color(color, geometry)
    started_ns = time.perf_counter_ns()
    elapsed_ms = {
        "depth_filter_ms": 0.0,
        "table_crop_ms": 0.0,
        "deprojection_ms": 0.0,
        "base_workspace_ms": 0.0,
        "voxelization_ms": 0.0,
        "spatial_outlier_filter_ms": 0.0,
        "color_sampling_ms": 0.0,
    }

    def stats(**kwargs: Any) -> PointCloudBuildStats:
        return PointCloudBuildStats(
            **kwargs,
            timings=PointCloudBuildTimings(
                **elapsed_ms,
                total_ms=(time.perf_counter_ns() - started_ns) / 1e6,
            ),
        )

    stage_started_ns = time.perf_counter_ns()
    depth_m, valid = _depth_valid_mask(
        depth_raw, depth_scale_m=depth_scale_m, config=config
    )
    valid_count = int(np.count_nonzero(valid))
    trusted = _reject_flying_depth(depth_m, valid, config)
    trusted_count = int(np.count_nonzero(trusted))
    elapsed_ms["depth_filter_ms"] = (time.perf_counter_ns() - stage_started_ns) / 1e6
    if trusted_count == 0:
        return None, stats(
            depth_valid_points=valid_count,
            failure_stage="depth_support",
        )

    stage_started_ns = time.perf_counter_ns()
    if table_plane_abcd is not None:
        table_keep = _table_keep_mask(
            depth_m,
            trusted,
            intrinsics=geometry.depth,
            T_xarm_base_from_color=T_xarm_base_from_color,
            table_plane_abcd=table_plane_abcd,
            table_core_height_m=config.table_core_height_m,
            table_object_seed_height_m=config.table_object_seed_height_m,
            table_object_seed_min_pixels=config.table_object_seed_min_pixels,
        )
        table_kept_count = int(np.count_nonzero(table_keep))
        table_rejected_count = trusted_count - table_kept_count
        if table_kept_count == 0:
            elapsed_ms["table_crop_ms"] = (
                time.perf_counter_ns() - stage_started_ns
            ) / 1e6
            return None, stats(
                depth_valid_points=valid_count,
                depth_trusted_points=trusted_count,
                table_rejected_points=table_rejected_count,
                failure_stage="table_crop",
            )
    else:
        table_rejected_count = 0
        table_keep = trusted
    elapsed_ms["table_crop_ms"] = (time.perf_counter_ns() - stage_started_ns) / 1e6

    stage_started_ns = time.perf_counter_ns()
    points_depth, rows, columns = _deproject_depth(depth_m, table_keep, geometry.depth)
    colors = color[rows, columns].astype(np.float32) / 255.0
    elapsed_ms["deprojection_ms"] = (time.perf_counter_ns() - stage_started_ns) / 1e6

    stage_started_ns = time.perf_counter_ns()
    points_base = _transform_points(points_depth, T_xarm_base_from_color)
    workspace_keep = _workspace_keep_mask(points_base, config.workspace)
    cropped_count = int(np.count_nonzero(workspace_keep))
    workspace_rejected_count = int(workspace_keep.size - cropped_count)
    if cropped_count == 0:
        elapsed_ms["base_workspace_ms"] = (
            time.perf_counter_ns() - stage_started_ns
        ) / 1e6
        return None, stats(
            depth_valid_points=valid_count,
            depth_trusted_points=trusted_count,
            table_rejected_points=table_rejected_count,
            workspace_rejected_points=workspace_rejected_count,
            failure_stage="workspace_crop",
        )
    if cropped_count != workspace_keep.size:
        points_base = points_base[workspace_keep]
        colors = colors[workspace_keep]
    elapsed_ms["base_workspace_ms"] = (time.perf_counter_ns() - stage_started_ns) / 1e6

    stage_started_ns = time.perf_counter_ns()
    points_base, colors, voxel_keys = _voxel_means(
        points_base, colors, config.voxel_size_m
    )
    elapsed_ms["voxelization_ms"] = (time.perf_counter_ns() - stage_started_ns) / 1e6
    voxel_count = points_base.shape[0]
    if voxel_count == 0:
        return None, stats(
            depth_valid_points=valid_count,
            depth_trusted_points=trusted_count,
            table_rejected_points=table_rejected_count,
            workspace_rejected_points=workspace_rejected_count,
            cropped_points=cropped_count,
            failure_stage="voxelization",
        )

    stage_started_ns = time.perf_counter_ns()
    density_keep, inlier = _radius_component_keep_masks(
        points_base,
        radius_m=config.outlier_radius_m,
        min_neighbors=config.outlier_min_neighbors,
        min_component_points=config.outlier_min_component_points,
    )
    density_count = int(np.count_nonzero(density_keep))
    if density_count == 0:
        elapsed_ms["spatial_outlier_filter_ms"] = (
            time.perf_counter_ns() - stage_started_ns
        ) / 1e6
        return None, stats(
            depth_valid_points=valid_count,
            depth_trusted_points=trusted_count,
            table_rejected_points=table_rejected_count,
            workspace_rejected_points=workspace_rejected_count,
            cropped_points=cropped_count,
            voxel_points=voxel_count,
            failure_stage="radius_density",
        )
    inlier_count = int(np.count_nonzero(inlier))
    if inlier_count == 0:
        elapsed_ms["spatial_outlier_filter_ms"] = (
            time.perf_counter_ns() - stage_started_ns
        ) / 1e6
        return None, stats(
            depth_valid_points=valid_count,
            depth_trusted_points=trusted_count,
            table_rejected_points=table_rejected_count,
            workspace_rejected_points=workspace_rejected_count,
            cropped_points=cropped_count,
            voxel_points=voxel_count,
            radius_density_points=density_count,
            failure_stage="radius_components",
        )
    points_base = points_base[inlier]
    colors = colors[inlier]
    voxel_keys = voxel_keys[inlier]
    candidate_indices = _spatial_candidate_indices(
        voxel_keys, config.num_points * config.outlier_candidate_multiplier
    )
    candidate_count = int(candidate_indices.size)
    points_base = points_base[candidate_indices]
    colors = colors[candidate_indices]
    voxel_keys = voxel_keys[candidate_indices]
    elapsed_ms["spatial_outlier_filter_ms"] = (
        time.perf_counter_ns() - stage_started_ns
    ) / 1e6

    stage_started_ns = time.perf_counter_ns()
    cloud = np.column_stack((points_base, colors)).astype(np.float32)
    result = _fixed_size_sample(
        cloud,
        voxel_keys,
        config.num_points,
        config.sampling_coarse_voxel_stride,
    )
    elapsed_ms["color_sampling_ms"] = (time.perf_counter_ns() - stage_started_ns) / 1e6
    return result, stats(
        depth_valid_points=valid_count,
        depth_trusted_points=trusted_count,
        table_rejected_points=table_rejected_count,
        workspace_rejected_points=workspace_rejected_count,
        cropped_points=cropped_count,
        voxel_points=voxel_count,
        radius_density_points=density_count,
        spatial_inlier_points=inlier_count,
        candidate_points=candidate_count,
    )

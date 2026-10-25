"""Deterministic aligned RGB-D geometry in the xArm base frame, without device or file I/O."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from fpsample import bucket_fps_kdline_sampling
from numpy.typing import NDArray
from scipy.sparse import coo_matrix  # type: ignore[import-untyped]
from scipy.sparse.csgraph import connected_components  # type: ignore[import-untyped]
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from dexmani_real.geometry.camera import (
    CameraIntrinsics,
    validate_aligned_depth_distortion,
)

if TYPE_CHECKING:
    from dexmani_real.perception.pointcloud.config import PointCloudConfig

NEIGHBORHOOD_KERNEL = np.ones((3, 3), dtype=np.uint8)

__all__ = [
    "build_raw_point_cloud",
    "build_point_cloud",
    "aligned_depth_points_in_base",
]


def depth_meters(
    depth_raw: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale_m: float,
) -> np.ndarray:
    if depth_raw.shape != (intrinsics.height, intrinsics.width) or depth_raw.dtype != np.uint16:
        raise ValueError("depth_raw must be uint16 [H,W] matching aligned color intrinsics")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be finite and positive")
    return depth_raw.astype(np.float32) * np.float32(depth_scale_m)


def depth_valid_mask(
    depth_raw: np.ndarray,
    *,
    intrinsics: CameraIntrinsics,
    depth_scale_m: float,
    config: PointCloudConfig,
) -> tuple[np.ndarray, np.ndarray]:
    depth_m = depth_meters(depth_raw, intrinsics, depth_scale_m)
    valid = (
        (depth_raw > 0)
        & np.isfinite(depth_m)
        & (depth_m >= config.depth_min_m)
        & (depth_m <= config.depth_max_m)
    )
    return depth_m, valid


def validate_color(color: np.ndarray, intrinsics: CameraIntrinsics) -> None:
    if color.dtype != np.uint8 or color.shape != (
        intrinsics.height,
        intrinsics.width,
        3,
    ):
        raise ValueError("color must be uint8 [H,W,3] matching color intrinsics")


def reject_flying_depth(
    depth_m: np.ndarray, valid: np.ndarray, config: PointCloudConfig
) -> np.ndarray:
    """Reject intermediate edge depths and locally unsupported measurements."""
    if not np.any(valid):
        return valid

    local_min = cast(
        np.ndarray,
        cv2.erode(
            np.where(valid, depth_m, np.float32(np.inf)),
            NEIGHBORHOOD_KERNEL,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=(float("inf"),),
        ),
    )
    local_max = cast(
        np.ndarray,
        cv2.dilate(
            np.where(valid, depth_m, np.float32(-np.inf)),
            NEIGHBORHOOD_KERNEL,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=(float("-inf"),),
        ),
    )
    local_valid_count = cast(
        NDArray[np.uint8],
        cv2.boxFilter(
            valid.astype(np.uint8),
            ddepth=cv2.CV_8U,
            ksize=(3, 3),
            normalize=False,
            borderType=cv2.BORDER_CONSTANT,
        ),
    )
    endpoint_distance = np.minimum(depth_m - local_min, local_max - depth_m)
    depth_discontinuity = (
        valid & (local_valid_count >= 3) & ((local_max - local_min) > config.edge_jump_m)
    )
    flying = depth_discontinuity & (endpoint_distance > config.edge_surface_band_m)
    trusted = valid & ~flying
    if config.depth_support_min_neighbors == 0 and config.edge_support_min_neighbors == 0:
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
    # Keep resolved interiors; reject unreliable one-/two-pixel structures at depth edges.
    return (
        trusted
        & (support_count >= required_support)
        & (~depth_discontinuity | cardinally_supported)
    )


def undistorted_depth_coordinates(
    rows: np.ndarray,
    columns: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized camera rays for color-grid depth pixels."""
    validate_aligned_depth_distortion(intrinsics.distortion_model)
    xd = (columns.astype(np.float32) - np.float32(intrinsics.ppx)) / np.float32(intrinsics.fx)
    yd = (rows.astype(np.float32) - np.float32(intrinsics.ppy)) / np.float32(intrinsics.fy)
    if intrinsics.distortion_model in {"distortion.none", "none"}:
        x, y = xd, yd
    else:
        # Invert Brown-Conrady using librealsense's fixed-point deprojection scheme.
        k1, k2, p1, p2, k3 = (np.float32(value) for value in intrinsics.distortion_coeffs)
        x, y = xd.copy(), yd.copy()
        for _ in range(10):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (xd - delta_x) / radial
            y = (yd - delta_y) / radial
    return x, y


@lru_cache(maxsize=4)
def depth_rays(intrinsics: CameraIntrinsics) -> np.ndarray:
    """Cache one deprojection ray per static aligned depth pixel."""
    rows, columns = np.indices((intrinsics.height, intrinsics.width), dtype=np.int64)
    x, y = undistorted_depth_coordinates(rows.reshape(-1), columns.reshape(-1), intrinsics)
    rays = np.column_stack((x, y, np.ones(x.size, dtype=np.float32))).reshape(
        intrinsics.height, intrinsics.width, 3
    )
    rays.setflags(write=False)
    return rays


@lru_cache(maxsize=4)
def compute_ray_plane_factors(
    intrinsics: CameraIntrinsics,
    plane_normal_color: tuple[float, float, float],
) -> np.ndarray:
    """Cache the static ray/plane-normal dot product for one calibration."""
    factors = depth_rays(intrinsics) @ np.asarray(plane_normal_color, dtype=np.float64)
    factors.setflags(write=False)
    return factors


def deproject_depth(
    depth_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized deprojection for depth-to-color aligned depth."""
    rows, columns = np.nonzero(valid)
    z = depth_m[rows, columns]
    rays = depth_rays(intrinsics)[rows, columns]
    return np.ascontiguousarray(rays * z[:, None], dtype=np.float32), rows, columns


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def table_height_hysteresis_keep_mask(
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
    seed_image = np.zeros(valid.shape, dtype=bool)
    object_seeds = signed_height_m >= object_seed_height_m
    seed_image[rows[object_seeds], columns[object_seeds]] = True
    seed_count, seed_labels = cv2.connectedComponents(seed_image.astype(np.uint8), connectivity=8)
    seed_sizes = np.bincount(seed_labels[seed_image], minlength=seed_count)
    reliable_seed = seed_sizes >= object_seed_min_pixels
    reliable_seed[0] = False
    if not np.any(reliable_seed):
        return np.zeros(valid.shape, dtype=bool)

    above_core = signed_height_m > core_height_m
    above_core_image = np.zeros(valid.shape, dtype=bool)
    above_core_image[rows[above_core], columns[above_core]] = True
    component_count, component_labels = cv2.connectedComponents(
        above_core_image.astype(np.uint8),
        connectivity=8,
    )
    keep_component = np.zeros(component_count, dtype=bool)
    keep_component[component_labels[reliable_seed[seed_labels]]] = True
    keep_component[0] = False
    return keep_component[component_labels]


def table_keep_mask(
    depth_m: np.ndarray,
    valid: np.ndarray,
    *,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float],
    table_core_height_m: float,
    table_object_seed_height_m: float,
    table_object_seed_min_pixels: int,
) -> np.ndarray:
    """Keep non-table valid pixels before allocating 3-D point arrays."""
    plane = np.asarray(table_plane_abcd, dtype=np.float64)
    norm = np.linalg.norm(plane[:3])
    plane /= norm

    # Evaluate the base-frame table plane on color-camera rays before allocating 3-D points.
    plane_color = T_xarm_base_from_color.T @ plane
    rows, columns = np.nonzero(valid)
    plane_normal_color = tuple(float(value) for value in plane_color[:3])
    ray_plane_factors = compute_ray_plane_factors(intrinsics, plane_normal_color)
    signed_height = depth_m[rows, columns] * ray_plane_factors[rows, columns] + plane_color[3]
    return table_height_hysteresis_keep_mask(
        valid,
        rows=rows,
        columns=columns,
        signed_height_m=signed_height,
        core_height_m=table_core_height_m,
        object_seed_height_m=table_object_seed_height_m,
        object_seed_min_pixels=table_object_seed_min_pixels,
    )


def workspace_keep_mask(
    points_base: np.ndarray,
    workspace: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    lower = np.asarray(workspace[:3], dtype=np.float32)
    upper = np.asarray(workspace[3:], dtype=np.float32)
    return np.all((points_base >= lower) & (points_base <= upper), axis=1)


def packed_grid_keys(keys: np.ndarray) -> np.ndarray:
    """Pack integer XYZ rows into collision-free scalar group identifiers."""
    minimum = np.min(keys, axis=0)
    shifted = keys - minimum
    extents = np.max(shifted, axis=0) + 1
    extent_yz = int(extents[1]) * int(extents[2])
    volume = int(extents[0]) * extent_yz
    if volume > np.iinfo(np.int64).max:
        raise OverflowError("grid key volume exceeds int64")
    return shifted[:, 0] * extent_yz + shifted[:, 1] * int(extents[2]) + shifted[:, 2]


def voxel_means(
    points_base: np.ndarray,
    colors: np.ndarray,
    voxel_size_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = np.floor(points_base / np.float32(voxel_size_m)).astype(np.int64)
    # Voxel means avoid first-pixel bias; bounded grid keys fit collision-free int64 packing.
    packed_keys = packed_grid_keys(keys)
    _, first_indices, inverse = np.unique(packed_keys, return_index=True, return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)

    def aggregate(points: np.ndarray) -> np.ndarray:
        columns = [np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)]
        return np.column_stack(columns).astype(np.float32)

    return (
        aggregate(points_base),
        aggregate(colors),
        keys[first_indices],
    )


def radius_component_keep_mask(
    points_base: np.ndarray,
    *,
    radius_m: float,
    min_neighbors: int,
    min_component_points: int,
) -> np.ndarray:
    """Use one radius graph for connected-island and local-density filtering."""
    count = points_base.shape[0]
    if min_neighbors == 0 and min_component_points <= 1:
        return np.ones(count, dtype=bool)

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
    if min_component_points <= 1 or not np.any(density_keep):
        return density_keep

    retained_pairs = pairs[density_keep[pairs[:, 0]] & density_keep[pairs[:, 1]]]
    graph = coo_matrix(
        (
            np.ones(retained_pairs.shape[0], dtype=np.uint8),
            (retained_pairs[:, 0], retained_pairs[:, 1]),
        ),
        shape=(count, count),
    )
    _, labels = connected_components(graph, directed=False, return_labels=True)
    component_sizes = np.bincount(labels, minlength=count)
    return density_keep & (component_sizes[labels] >= min_component_points)


def spatial_hash(voxel_keys: np.ndarray) -> np.ndarray:
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


def spatial_candidate_indices(voxel_keys: np.ndarray, max_candidates: int) -> np.ndarray:
    count = voxel_keys.shape[0]
    if count <= max_candidates:
        return np.arange(count, dtype=np.int64)
    hashes = spatial_hash(voxel_keys)
    selected = np.argpartition(hashes, max_candidates - 1)[:max_candidates]
    return selected[np.argsort(hashes[selected], kind="stable")]


def fixed_size_sample_indices(
    points_base: np.ndarray,
    voxel_keys: np.ndarray,
    num_points: int,
) -> np.ndarray:
    """QuickFPS over voxel means; cycle observed points only when fewer than requested."""
    count = voxel_keys.shape[0]
    if count > num_points:
        # fpsample 0.3.3 uses the post-partition tree position as its deterministic seed.
        return bucket_fps_kdline_sampling(
            points_base,
            num_points,
            h=min(5, count.bit_length() - 1),
            start_idx=0,
        )
    spatial_order = np.argsort(spatial_hash(voxel_keys), kind="stable")
    return spatial_order[np.arange(num_points, dtype=np.int64) % count]


def aligned_depth_points_in_base(
    *,
    depth_raw: np.ndarray,
    depth_scale_m: float,
    aligned_depth_intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    config: PointCloudConfig,
) -> np.ndarray:
    """Return table-calibration points using configuration/extrinsics validated at setup."""
    depth_m, valid = depth_valid_mask(
        depth_raw, intrinsics=aligned_depth_intrinsics, depth_scale_m=depth_scale_m, config=config
    )
    trusted = reject_flying_depth(depth_m, valid, config)
    if not np.any(trusted):
        return np.empty((0, 3), dtype=np.float32)
    points_color, _, _ = deproject_depth(depth_m, trusted, aligned_depth_intrinsics)
    return transform_points(points_color, T_xarm_base_from_color)


def build_raw_point_cloud(
    *,
    depth_raw: np.ndarray,
    color: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
) -> np.ndarray | None:
    """Return diagnostic points using extrinsics validated when loading calibration."""
    validate_color(color, intrinsics)
    depth_m = depth_meters(depth_raw, intrinsics, depth_scale_m)
    valid = (depth_raw > 0) & np.isfinite(depth_m)
    if not np.any(valid):
        return None
    points_color, rows, columns = deproject_depth(depth_m, valid, intrinsics)
    points_base = transform_points(points_color, T_xarm_base_from_color)
    colors = color[rows, columns].astype(np.float32) / 255.0
    return np.ascontiguousarray(np.column_stack((points_base, colors)), dtype=np.float32)


def build_point_cloud(
    *,
    depth_raw: np.ndarray,
    color: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
    table_plane_abcd: tuple[float, float, float, float] | None,
    config: PointCloudConfig,
) -> np.ndarray | None:
    """Build ``float32[num_points,6]`` with static inputs validated at session/episode setup."""
    if not config.remove_table:
        table_plane_abcd = None
    elif table_plane_abcd is None:
        raise ValueError("remove_table=True requires a current calibrated table plane")

    validate_color(color, intrinsics)
    depth_m, valid = depth_valid_mask(
        depth_raw, intrinsics=intrinsics, depth_scale_m=depth_scale_m, config=config
    )
    trusted = reject_flying_depth(depth_m, valid, config)
    if not np.any(trusted):
        return None

    if table_plane_abcd is not None:
        table_keep = table_keep_mask(
            depth_m,
            trusted,
            intrinsics=intrinsics,
            T_xarm_base_from_color=T_xarm_base_from_color,
            table_plane_abcd=table_plane_abcd,
            table_core_height_m=config.table_core_height_m,
            table_object_seed_height_m=config.table_object_seed_height_m,
            table_object_seed_min_pixels=config.table_object_seed_min_pixels,
        )
        if not np.any(table_keep):
            return None
    else:
        table_keep = trusted

    points_color, rows, columns = deproject_depth(depth_m, table_keep, intrinsics)
    colors = color[rows, columns].astype(np.float32) / 255.0

    points_base = transform_points(points_color, T_xarm_base_from_color)
    workspace_keep = workspace_keep_mask(points_base, config.workspace)
    cropped_count = int(np.count_nonzero(workspace_keep))
    if cropped_count == 0:
        return None
    if cropped_count != workspace_keep.size:
        points_base = points_base[workspace_keep]
        colors = colors[workspace_keep]

    points_base, colors, voxel_keys = voxel_means(points_base, colors, config.voxel_size_m)
    inlier = radius_component_keep_mask(
        points_base,
        radius_m=config.outlier_radius_m,
        min_neighbors=config.outlier_min_neighbors,
        min_component_points=config.outlier_min_component_points,
    )
    retained = np.flatnonzero(inlier)
    if retained.size == 0:
        return None
    # This limits final sampling work, not the earlier radius/component search.
    sampling_candidate_limit = config.num_points * config.outlier_candidate_multiplier
    candidates = spatial_candidate_indices(voxel_keys[retained], sampling_candidate_limit)
    retained = retained[candidates]
    sampled = fixed_size_sample_indices(
        points_base[retained],
        voxel_keys[retained],
        config.num_points,
    )
    selected = retained[sampled]
    return np.column_stack((points_base[selected], colors[selected]))

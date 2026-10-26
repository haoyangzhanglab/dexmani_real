"""点集的体素聚合、连通域过滤和确定性采样；不依赖深度图或设备。"""

from __future__ import annotations

import numpy as np
from fpsample import bucket_fps_kdline_sampling
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


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

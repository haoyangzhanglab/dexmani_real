"""对齐 RGB-D 到 xArm 基座系的 float32[N,6] XYZRGB；无有效点时返回 (0,6)。"""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.geometry.camera import CameraIntrinsics, deproject_depth, depth_rays
from dexmani_real.utils.table_plane import normalize_table_plane

DEPTH_RANGE_M = (0.30, 1.50)
NEIGHBORHOOD_KERNEL = np.ones((3, 3), dtype=np.uint8)
EDGE_JUMP_M = 0.030
EDGE_SURFACE_BAND_M = 0.008
DEPTH_MIN_SUPPORT = 2
EDGE_MIN_SUPPORT = 5
VOXEL_SIZE_M = 0.005
OUTLIER_RADIUS_M = 0.012
OUTLIER_MIN_NEIGHBORS = 6
OUTLIER_MIN_COMPONENT_POINTS = 10


def filter_depth(depth_m: np.ndarray) -> np.ndarray:
    import cv2

    valid = (depth_m >= DEPTH_RANGE_M[0]) & (depth_m <= DEPTH_RANGE_M[1])
    local_min: np.ndarray = cv2.erode(
        np.where(valid, depth_m, np.float32(np.inf)),
        NEIGHBORHOOD_KERNEL,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=(float("inf"),),
    )
    local_max: np.ndarray = cv2.dilate(
        np.where(valid, depth_m, np.float32(-np.inf)),
        NEIGHBORHOOD_KERNEL,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=(float("-inf"),),
    )
    # 深度断层中，远离前后表面的中间值视为飞点。
    local_count = cast(NDArray[np.uint8], cv2.boxFilter(
        valid.astype(np.uint8),
        ddepth=cv2.CV_8U,
        ksize=(3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    ))
    edge = valid & (local_count >= 3) & ((local_max - local_min) > EDGE_JUMP_M)
    # 断层判定完成后，复用局部极值数组存放到前后表面的距离。
    np.subtract(depth_m, local_min, out=local_min)
    np.subtract(local_max, depth_m, out=local_max)
    distance = np.minimum(local_min, local_max, out=local_min)
    trusted = valid & ~(edge & (distance > EDGE_SURFACE_BAND_M))

    # 正邻接与对角邻接分别计数，再合并为八邻域支持数。
    support_count = np.zeros(depth_m.shape, dtype=np.uint8)
    cardinal_count = np.zeros(depth_m.shape, dtype=np.uint8)
    for first, second, counts in (
        (np.s_[:, :-1], np.s_[:, 1:], cardinal_count),
        (np.s_[:-1, :], np.s_[1:, :], cardinal_count),
        (np.s_[:-1, :-1], np.s_[1:, 1:], support_count),
        (np.s_[:-1, 1:], np.s_[1:, :-1], support_count),
    ):
        same_surface = trusted[first] & trusted[second] & (
            np.abs(depth_m[first] - depth_m[second]) <= EDGE_SURFACE_BAND_M
        )
        counts[first] += same_surface
        counts[second] += same_surface
    support_count += cardinal_count
    required_support = np.where(edge, np.uint8(EDGE_MIN_SUPPORT), np.uint8(DEPTH_MIN_SUPPORT))
    return trusted & (support_count >= required_support) & (~edge | (cardinal_count == 4))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def voxel_downsample(points: np.ndarray, colors: np.ndarray):
    # 每个坐标轴连续存放，减少逐轴归约的跨步访存。
    keys = np.floor(points.T / np.float32(VOXEL_SIZE_M)).astype(np.int64, order="C")
    keys -= keys.min(axis=1)[:, None]
    packed = np.ravel_multi_index(keys, keys.max(axis=1) + 1)
    _, inverse = np.unique(packed, return_inverse=True)
    counts = np.bincount(inverse)

    def mean(values):
        return np.column_stack(
            [np.bincount(inverse, weights=values[:, axis]) / counts for axis in range(3)]
        ).astype(np.float32)

    return mean(points), mean(colors)


def radius_inliers(points: np.ndarray) -> np.ndarray:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(points, compact_nodes=False, balanced_tree=False).query_pairs(
        OUTLIER_RADIUS_M,
        output_type="ndarray",
    )
    density_keep = np.bincount(pairs.ravel(), minlength=len(points)) >= OUTLIER_MIN_NEIGHBORS
    # 密度检查会移除邻居并留下孤立残片；只在保留下来的点之间求连通分量。
    pairs = pairs[density_keep[pairs[:, 0]] & density_keep[pairs[:, 1]]]
    graph = coo_matrix(
        (np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
        shape=(len(points), len(points)),
    )
    _, labels = connected_components(graph, directed=False, return_labels=True)
    component_sizes = np.bincount(labels)
    return density_keep & (component_sizes[labels] >= OUTLIER_MIN_COMPONENT_POINTS)


def sample_indices(points: np.ndarray, num_points: int) -> np.ndarray:
    """对非空点集 FPS；稀疏时循环重复已有点以满足形状，不增加几何信息。"""
    if len(points) > num_points:
        from fpsample import bucket_fps_kdline_sampling

        return bucket_fps_kdline_sampling(
            points,
            num_points,
            h=min(5, len(points).bit_length() - 1),
            start_idx=0,
        )
    return np.arange(num_points) % len(points)


def aligned_depth_points_in_base(
    *,
    depth_raw: np.ndarray,
    depth_scale_m: float,
    aligned_depth_intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
) -> np.ndarray:
    """桌面标定使用深度过滤后的点，不执行桌面和工作空间裁剪。"""
    depth_m = depth_raw.astype(np.float32)
    depth_m *= np.float32(depth_scale_m)
    points, _, _ = deproject_depth(
        depth_m,
        filter_depth(depth_m),
        aligned_depth_intrinsics,
    )
    return transform_points(points, T_xarm_base_from_color)


def build_raw_point_cloud(
    *,
    depth_raw: np.ndarray,
    color: np.ndarray,
    depth_scale_m: float,
    intrinsics: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray,
) -> np.ndarray:
    depth_m = depth_raw.astype(np.float32)
    depth_m *= np.float32(depth_scale_m)
    points, rows, columns = deproject_depth(depth_m, depth_raw > 0, intrinsics)
    xyz = transform_points(points, T_xarm_base_from_color)
    rgb = color[rows, columns].astype(np.float32)
    rgb /= 255.0
    return np.column_stack((xyz, rgb))


class PointCloudProcessor:
    def __init__(
        self,
        *,
        depth_scale_m: float | None,
        intrinsics: CameraIntrinsics,
        T_xarm_base_from_color: np.ndarray | None,
        table_plane_abcd: tuple[float, float, float, float] | None,
        config: PointCloudConfig,
    ):
        config.validate()
        if T_xarm_base_from_color is None:
            raise ValueError("pointcloud requires camera extrinsics")
        if depth_scale_m is None:
            raise ValueError("pointcloud requires a camera depth scale")
        if config.remove_table and table_plane_abcd is None:
            raise ValueError("remove_table=True requires a calibrated table plane")

        self.config = config
        self.depth_scale = np.float32(depth_scale_m)
        self.lower, self.upper = np.asarray(config.workspace, dtype=np.float32).reshape(2, 3)
        rays = depth_rays(intrinsics)
        transform = T_xarm_base_from_color
        self.rays_base = (rays @ transform[:3, :3].T).astype(np.float32)
        self.translation = transform[:3, 3].astype(np.float32)
        if config.remove_table:
            plane_color = transform.T @ normalize_table_plane(table_plane_abcd)
            self.table_height_per_depth = rays @ plane_color[:3]
            self.table_offset = plane_color[3]

    def process(self, depth_raw: np.ndarray, color: np.ndarray) -> np.ndarray:
        depth_m = depth_raw.astype(np.float32)
        depth_m *= self.depth_scale
        valid = filter_depth(depth_m)
        if self.config.remove_table:
            # 先沿射线裁掉桌面，避免为桌面像素分配三维点。
            height = depth_m * self.table_height_per_depth
            height += self.table_offset
            valid &= height > self.config.table_margin_m
        for axis in range(3):
            coordinate = self.rays_base[..., axis] * depth_m
            coordinate += self.translation[axis]
            valid &= (coordinate >= self.lower[axis]) & (coordinate <= self.upper[axis])
        pixels = np.flatnonzero(valid)
        if not len(pixels):
            return np.empty((0, 6), dtype=np.float32)
        points = self.rays_base.reshape(-1, 3)[pixels]
        points *= depth_m.ravel()[pixels, None]
        points += self.translation
        colors = color.reshape(-1, 3)[pixels].astype(np.float32)
        colors /= 255.0
        points, colors = voxel_downsample(points, colors)
        keep = radius_inliers(points)
        points, colors = points[keep], colors[keep]
        if len(points) == 0:
            return np.empty((0, 6), dtype=np.float32)

        selected = sample_indices(points, self.config.num_points)
        return np.column_stack((points[selected], colors[selected]))


def processing_metadata(config, table_plane_abcd):
    """记录实际配置和固定滤波常量；与处理实现共用权威值。"""
    from dataclasses import asdict

    metadata = {
        "config": {**asdict(config), "workspace": list(config.workspace)},
        "depth_range_m": list(DEPTH_RANGE_M),
        "edge_jump_m": EDGE_JUMP_M,
        "edge_surface_band_m": EDGE_SURFACE_BAND_M,
        "depth_min_support": DEPTH_MIN_SUPPORT,
        "edge_min_support": EDGE_MIN_SUPPORT,
        "voxel_size_m": VOXEL_SIZE_M,
        "outlier_radius_m": OUTLIER_RADIUS_M,
        "outlier_min_neighbors": OUTLIER_MIN_NEIGHBORS,
        "outlier_min_component_points": OUTLIER_MIN_COMPONENT_POINTS,
    }
    if config.remove_table:
        metadata["table_plane_abcd"] = list(table_plane_abcd)
    return metadata

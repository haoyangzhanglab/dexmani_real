"""对齐 RGB-D 到 xArm 基座系的 float32[N,6] XYZRGB 点云。"""

import cv2
import numpy as np
from fpsample import bucket_fps_kdline_sampling
from scipy.spatial import cKDTree

from dexmani_real.geometry.camera import CameraIntrinsics, deproject_depth, depth_rays
from dexmani_real.perception.config import PointCloudConfig

NEIGHBORHOOD_KERNEL = np.ones((3, 3), dtype=np.uint8)


def filter_depth(depth_m: np.ndarray, config: PointCloudConfig) -> np.ndarray:
    valid = (depth_m >= config.depth_min_m) & (depth_m <= config.depth_max_m)
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
    edge = (local_max - local_min) > config.edge_jump_m
    distance = np.minimum(depth_m - local_min, local_max - depth_m)
    return valid & ~(edge & (distance > config.edge_surface_band_m))


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    return (points @ transform[:3, :3].T + transform[:3, 3]).astype(np.float32)


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size_m: float):
    keys = np.floor(points / np.float32(voxel_size_m)).astype(np.int64)
    shifted = keys - keys.min(axis=0)
    extents = shifted.max(axis=0) + 1
    stride = int(extents[1]) * int(extents[2])
    if int(extents[0]) * stride > np.iinfo(np.int64).max:
        raise ValueError("voxel grid exceeds int64; increase voxel_size_m")
    packed = shifted[:, 0] * stride + shifted[:, 1] * int(extents[2]) + shifted[:, 2]
    _, inverse = np.unique(packed, return_inverse=True)
    counts = np.bincount(inverse)

    def mean(values):
        return np.column_stack(
            [np.bincount(inverse, weights=values[:, axis]) / counts for axis in range(3)]
        ).astype(np.float32)

    return mean(points), mean(colors)


def radius_inliers(points: np.ndarray, radius_m: float, min_neighbors: int) -> np.ndarray:
    if min_neighbors == 0:
        return np.ones(len(points), dtype=bool)
    pairs = cKDTree(points, compact_nodes=False, balanced_tree=False).query_pairs(
        radius_m,
        output_type="ndarray",
    )
    return np.bincount(pairs.ravel(), minlength=len(points)) >= min_neighbors


def sample_indices(points: np.ndarray, num_points: int) -> np.ndarray:
    if len(points) > num_points:
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
    config: PointCloudConfig,
) -> np.ndarray:
    """桌面标定使用深度过滤后的点，不执行桌面和工作空间裁剪。"""
    depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale_m)
    points, _, _ = deproject_depth(
        depth_m,
        filter_depth(depth_m, config),
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
) -> np.ndarray | None:
    depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale_m)
    points, rows, columns = deproject_depth(depth_m, depth_raw > 0, intrinsics)
    if len(points) == 0:
        return None
    xyz = transform_points(points, T_xarm_base_from_color)
    rgb = color[rows, columns].astype(np.float32) / 255.0
    return np.column_stack((xyz, rgb))


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
    if config.remove_table and table_plane_abcd is None:
        raise ValueError("remove_table=True requires a calibrated table plane")

    depth_m = depth_raw.astype(np.float32) * np.float32(depth_scale_m)
    valid = filter_depth(depth_m, config)
    if config.remove_table:
        plane = np.asarray(table_plane_abcd, dtype=np.float64)
        plane_color = T_xarm_base_from_color.T @ (plane / np.linalg.norm(plane[:3]))
        # 在相机射线上计算桌面高度，避免为桌面像素分配三维点。
        height = depth_m * (depth_rays(intrinsics) @ plane_color[:3]) + plane_color[3]
        valid &= height > config.table_margin_m
    points, rows, columns = deproject_depth(depth_m, valid, intrinsics)
    points = transform_points(points, T_xarm_base_from_color)
    lower, upper = np.asarray(config.workspace, dtype=np.float32).reshape(2, 3)
    keep = np.all((points >= lower) & (points <= upper), axis=1)
    if not keep.any():
        return None

    points = points[keep]
    colors = color[rows[keep], columns[keep]].astype(np.float32) / 255.0
    points, colors = voxel_downsample(points, colors, config.voxel_size_m)
    keep = radius_inliers(points, config.outlier_radius_m, config.outlier_min_neighbors)
    points, colors = points[keep], colors[keep]
    if len(points) == 0:
        return None

    selected = sample_indices(points, config.num_points)
    return np.column_stack((points[selected], colors[selected]))

"""Deterministic numerical transforms for the Sim-label HDF5 view."""

from __future__ import annotations

import numpy as np


def resize_rgb(frame: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Resize one RGB frame using the fixed downsampling contract."""

    import cv2

    value = np.asarray(frame)
    if value.ndim != 3 or value.shape[2] != 3 or value.dtype != np.uint8:
        raise ValueError(f"RGB frame must be uint8 [H,W,3], got {value.shape} {value.dtype}")
    if height <= 0 or width <= 0:
        raise ValueError("target RGB height and width must be positive")
    if value.shape[:2] == (height, width):
        return np.ascontiguousarray(value)
    interpolation = cv2.INTER_AREA if height <= value.shape[0] and width <= value.shape[1] else cv2.INTER_LINEAR
    resized = cv2.resize(value, (width, height), interpolation=interpolation)
    return np.ascontiguousarray(resized, dtype=np.uint8)


def resize_camera_intrinsic(
    camera_k: np.ndarray,
    *,
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    """Scale a row-major pinhole K for a resize with no crop."""

    value = np.asarray(camera_k, dtype=np.float64)
    if value.shape == (9,):
        value = value.reshape(3, 3)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError("camera_K must be a finite 3x3 matrix or length-9 vector")
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError("camera intrinsic source and target dimensions must be positive")
    if not np.allclose(value[2], np.array([0.0, 0.0, 1.0]), rtol=0.0, atol=1e-8):
        raise ValueError("camera_K must have the canonical pinhole last row [0,0,1]")
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    result = value.copy()
    result[0, 0] *= scale_x
    result[0, 2] *= scale_x
    result[1, 1] *= scale_y
    result[1, 2] *= scale_y
    return result.astype(np.float32).reshape(9)


def _fps_numpy(points_xyz: np.ndarray, count: int) -> np.ndarray:
    """Deterministic CPU fallback when Open3D is unavailable."""

    selected = np.empty(count, dtype=np.int64)
    center = np.mean(points_xyz, axis=0)
    selected[0] = int(np.argmax(np.sum((points_xyz - center) ** 2, axis=1)))
    min_distance = np.sum((points_xyz - points_xyz[selected[0]]) ** 2, axis=1)
    for index in range(1, count):
        selected[index] = int(np.argmax(min_distance))
        distance = np.sum((points_xyz - points_xyz[selected[index]]) ** 2, axis=1)
        np.minimum(min_distance, distance, out=min_distance)
    return selected


def resize_point_cloud(
    point_cloud: np.ndarray,
    *,
    source_point_count: int,
    target_point_count: int,
) -> np.ndarray:
    """Create a deterministic fixed-size XYZRGB cloud without frame changes.

    Real's producer places all unique filtered points first when it must pad a
    sparse cloud.  ``source_point_count`` therefore lets this transform ignore
    producer duplicates before doing FPS or deterministic cyclic padding.
    """

    value = np.asarray(point_cloud, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 6:
        raise ValueError(f"point cloud must have shape [M,6], got {value.shape}")
    if target_point_count <= 0 or source_point_count <= 0:
        raise ValueError("source and target point counts must be positive")
    if not np.all(np.isfinite(value)):
        raise ValueError("point cloud contains non-finite values")
    if np.any(value[:, 3:] < 0.0) or np.any(value[:, 3:] > 1.0):
        raise ValueError("point cloud RGB must be in [0,1]")
    unique_count = min(int(source_point_count), value.shape[0])
    source = np.ascontiguousarray(value[:unique_count])
    if not np.any(np.linalg.norm(source[:, :3], axis=1) > 0.0):
        raise ValueError("point cloud has no non-zero XYZ point")
    if unique_count < target_point_count:
        indices = np.resize(np.arange(unique_count, dtype=np.int64), target_point_count)
        return np.ascontiguousarray(source[indices], dtype=np.float32)
    if unique_count == target_point_count:
        return source.copy()

    try:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(source[:, :3].astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(source[:, 3:].astype(np.float64))
        sampled = cloud.farthest_point_down_sample(target_point_count)
        result = np.concatenate((np.asarray(sampled.points), np.asarray(sampled.colors)), axis=1)
        if result.shape != (target_point_count, 6):
            raise ValueError(f"Open3D FPS returned unexpected shape {result.shape}")
        return np.ascontiguousarray(result, dtype=np.float32)
    except ImportError:
        indices = _fps_numpy(source[:, :3].astype(np.float64), target_point_count)
        return np.ascontiguousarray(source[indices], dtype=np.float32)

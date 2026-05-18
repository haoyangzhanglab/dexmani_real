import cv2
import torch
import numpy as np
from typing import Any, Optional, Sequence, Tuple, Union


ArrayLike = Union[np.ndarray, torch.Tensor]


def parse_resolution(text: str) -> Tuple[int, int]:
    """Parse a resolution string like '640x480' into (width, height)."""
    width_text, height_text = text.lower().split("x")
    return int(width_text), int(height_text)


def to_numpy(value: Any) -> Any:
    """Convert tensor-like value to numpy while keeping None unchanged."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def intrinsics_to_matrix(intrinsics: Any, dtype: np.dtype = np.float32) -> np.ndarray:
    """Convert pyrealsense2 intrinsics to a 3x3 camera matrix."""
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )


def intrinsics_to_dict(intrinsics: Any) -> dict:
    """Convert pyrealsense2 intrinsics to a small dict."""
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "cx": float(intrinsics.ppx),
        "cy": float(intrinsics.ppy),
        "model": str(intrinsics.model),
        "coeffs": [float(value) for value in intrinsics.coeffs],
    }


def check_intrinsics(intrinsics: ArrayLike) -> np.ndarray:
    """Validate and return camera intrinsics as a 3x3 float32 numpy array."""
    intrinsics_array = np.asarray(intrinsics, dtype=np.float32)
    if intrinsics_array.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics_array.shape}.")
    return intrinsics_array


def check_transform(T_out_camera: Optional[ArrayLike]) -> Optional[np.ndarray]:
    """Validate a 4x4 transform matrix while keeping None unchanged."""
    if T_out_camera is None:
        return None
    transform_array = np.asarray(T_out_camera, dtype=np.float32)
    if transform_array.shape != (4, 4):
        raise ValueError(f"T_out_camera must have shape (4, 4), got {transform_array.shape}.")
    return transform_array


def check_workspace(workspace: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float, float, float, float]]:
    """Validate workspace [x_min, x_max, y_min, y_max, z_min, z_max]."""
    if workspace is None:
        return None
    if len(workspace) != 6:
        raise ValueError("workspace must be [x_min, x_max, y_min, y_max, z_min, z_max].")
    x_min, x_max, y_min, y_max, z_min, z_max = [float(value) for value in workspace]
    if not (x_min < x_max and y_min < y_max and z_min < z_max):
        raise ValueError(f"invalid workspace: {workspace}.")
    return x_min, x_max, y_min, y_max, z_min, z_max


def depth_meters_to_mm(depth_meters: ArrayLike) -> np.ndarray:
    """Convert depth in meters to uint16 millimeters."""
    depth_array = np.asarray(to_numpy(depth_meters), dtype=np.float32)
    depth_mm = np.rint(depth_array * 1000.0)
    depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth_mm, 0.0, np.iinfo(np.uint16).max)
    return depth_mm.astype(np.uint16)


def depth_mm_to_meters(depth_mm: ArrayLike, device: str = "cpu") -> torch.Tensor:
    """Convert uint16 millimeter depth to float32 meters."""
    depth_tensor = torch.as_tensor(depth_mm, dtype=torch.float32, device=device)
    return depth_tensor * 0.001


def make_rays(height: int, width: int, intrinsics: ArrayLike, device: str = "cpu") -> torch.Tensor:
    """Create per-pixel camera rays with shape (height, width, 3).

    Each ray equals K^-1 @ [u, v, 1]. Multiplying rays by depth in meters
    gives 3D points in the current depth image coordinate frame.
    """
    intrinsics_tensor = torch.as_tensor(check_intrinsics(intrinsics), dtype=torch.float32, device=device)
    pixel_u, pixel_v = torch.meshgrid(
        torch.arange(width, device=device, dtype=torch.float32),
        torch.arange(height, device=device, dtype=torch.float32),
        indexing="xy",
    )
    pixel_homo = torch.stack((pixel_u, pixel_v, torch.ones_like(pixel_u)), dim=-1)
    return pixel_homo @ torch.linalg.inv(intrinsics_tensor).T


def depth_to_points(depth_mm: ArrayLike, rays: ArrayLike, device: str = "cpu") -> torch.Tensor:
    """Back-project a uint16 millimeter depth image to dense 3D points in meters."""
    depth_meters = depth_mm_to_meters(depth_mm, device=device)
    rays_tensor = torch.as_tensor(rays, dtype=torch.float32, device=device)
    if depth_meters.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {tuple(depth_meters.shape)}.")
    if rays_tensor.shape[:2] != depth_meters.shape[:2] or rays_tensor.shape[-1] != 3:
        raise ValueError(
            f"rays must have shape (H, W, 3) matching depth, got rays={tuple(rays_tensor.shape)}, "
            f"depth={tuple(depth_meters.shape)}."
        )
    return (rays_tensor * depth_meters[..., None]).reshape(-1, 3)


def image_to_colors(color: Optional[ArrayLike], depth_shape: Optional[Tuple[int, int]] = None, device: str = "cpu") -> Optional[torch.Tensor]:
    """Flatten an RGB image to normalized float32 per-point colors in [0, 1]."""
    if color is None:
        return None
    color_tensor = torch.as_tensor(color, dtype=torch.float32, device=device)
    if color_tensor.ndim != 3 or color_tensor.shape[-1] != 3:
        raise ValueError(f"color must have shape (H, W, 3), got {tuple(color_tensor.shape)}.")
    if depth_shape is not None and tuple(color_tensor.shape[:2]) != tuple(depth_shape):
        raise ValueError(
            f"color shape {tuple(color_tensor.shape[:2])} must match depth shape {tuple(depth_shape)}. "
            "Use align_to='depth' or align_to='color' before generating colored point cloud."
        )
    if color_tensor.max() > 1.0:
        color_tensor = color_tensor / 255.0
    return color_tensor.reshape(-1, 3).clamp(0.0, 1.0)


def filter_depth(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    min_depth: Optional[float] = 0.05,
    max_depth: Optional[float] = 2.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Remove invalid points and points outside the camera-frame depth range in meters."""
    keep = torch.isfinite(points).all(dim=1)
    depth_z = points[:, 2]
    if min_depth is not None:
        keep = keep & (depth_z > float(min_depth))
    if max_depth is not None:
        keep = keep & (depth_z < float(max_depth))
    points = points[keep]
    colors = colors[keep] if colors is not None else None
    return points, colors


def transform_points(points: torch.Tensor, T_out_camera: Optional[ArrayLike] = None) -> torch.Tensor:
    """Transform points by T_out_camera; keep camera frame when None."""
    if T_out_camera is None:
        return points
    transform_tensor = torch.as_tensor(check_transform(T_out_camera), dtype=torch.float32, device=points.device)
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_homo = torch.cat((points, ones), dim=1)
    return (points_homo @ transform_tensor.T)[:, :3]


def crop_workspace(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    workspace: Optional[Sequence[float]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Crop points by workspace in the current point coordinate frame, in meters."""
    checked_workspace = check_workspace(workspace)
    if checked_workspace is None:
        return points, colors
    x_min, x_max, y_min, y_max, z_min, z_max = checked_workspace
    keep = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    keep = keep & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    keep = keep & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    points = points[keep]
    colors = colors[keep] if colors is not None else None
    return points, colors


def sample_points(
    points: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    npoints: Optional[int] = 1024,
    sampling: str = "random",
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Sample a point cloud and keep colors aligned with points."""
    if sampling not in ("none", "random", "fps", "first"):
        raise ValueError("sampling must be one of: 'none', 'random', 'fps', 'first'.")
    if sampling == "none" or npoints is None:
        return points, colors
    if npoints <= 0:
        raise ValueError("npoints must be positive, or None when returning all points.")

    point_count = int(points.shape[0])
    if point_count == 0:
        raise ValueError("No valid points after depth filter/crop. Check depth range, workspace, and T_out_camera.")

    if point_count < npoints:
        if sampling == "first":
            repeat_count = int(np.ceil(npoints / point_count))
            sample_index = torch.arange(point_count, device=points.device).repeat(repeat_count)[:npoints]
        else:
            base_index = torch.randperm(point_count, device=points.device)
            extra_index = torch.randint(0, point_count, (npoints - point_count,), device=points.device)
            sample_index = torch.cat((base_index, extra_index), dim=0)
    elif sampling == "first":
        sample_index = torch.arange(npoints, device=points.device)
    elif sampling == "random":
        sample_index = torch.randperm(point_count, device=points.device)[:npoints]
    else:
        try:
            import pytorch3d.ops as torch3d_ops

            sample_index = torch3d_ops.sample_farthest_points(points[None], K=npoints)[1][0]
        except ImportError:
            sample_index = torch.randperm(point_count, device=points.device)[:npoints]

    points = points[sample_index]
    colors = colors[sample_index] if colors is not None else None
    return points, colors


def pack_pointcloud(points: torch.Tensor, colors: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pack XYZ and normalized RGB into one float32 tensor with shape (N, 6)."""
    points = points.to(dtype=torch.float32)
    if colors is None:
        colors = torch.zeros_like(points)
    else:
        colors = colors.to(dtype=torch.float32).clamp(0.0, 1.0)
    return torch.cat((points, colors), dim=1).to(dtype=torch.float32)


def rgbd_to_pointcloud(
    depth: ArrayLike,
    intrinsics: Optional[ArrayLike] = None,
    color: Optional[ArrayLike] = None,
    *,
    rays: Optional[ArrayLike] = None,
    T_out_camera: Optional[ArrayLike] = None,
    workspace: Optional[Sequence[float]] = None,
    bound: Optional[Sequence[float]] = None,
    npoints: Optional[int] = 1024,
    min_depth: Optional[float] = 0.05,
    max_depth: Optional[float] = 2.0,
    sampling: str = "random",
    device: str = "cpu",
    return_tensor: bool = True,
) -> Union[torch.Tensor, np.ndarray]:
    """Convert aligned RGB-D to packed XYZRGB point cloud.

    Args:
        depth: uint16 depth image in millimeters.
        color: RGB uint8 image. Point colors are normalized to float32 [0, 1].

    Returns:
        pointcloud: shape (N, 6), float32.
            columns 0:3 are XYZ in meters.
            columns 3:6 are RGB in [0, 1].
    """
    if workspace is not None and bound is not None:
        raise ValueError("Use only one of workspace or bound, not both.")
    if workspace is None:
        workspace = bound

    depth_tensor = torch.as_tensor(depth, device=device)
    if depth_tensor.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {tuple(depth_tensor.shape)}.")
    height, width = int(depth_tensor.shape[0]), int(depth_tensor.shape[1])

    if rays is None:
        if intrinsics is None:
            raise ValueError("intrinsics must be provided when rays is None.")
        rays = make_rays(height, width, intrinsics, device=device)

    points = depth_to_points(depth_tensor, rays, device=device)
    colors = image_to_colors(color, depth_shape=(height, width), device=device) if color is not None else None

    points, colors = filter_depth(points, colors, min_depth=min_depth, max_depth=max_depth)
    points = transform_points(points, T_out_camera=T_out_camera)
    points, colors = crop_workspace(points, colors, workspace=workspace)
    points, colors = sample_points(points, colors, npoints=npoints, sampling=sampling)
    pointcloud = pack_pointcloud(points, colors)

    if return_tensor:
        return pointcloud
    return pointcloud.detach().cpu().numpy().astype(np.float32)


def depth_valid_ratio(depth: ArrayLike, min_depth: Optional[float] = 0.05, max_depth: Optional[float] = 2.0) -> float:
    """Compute ratio of finite uint16 depth pixels within [min_depth, max_depth] meters."""
    depth_meters = np.asarray(to_numpy(depth), dtype=np.float32) * 0.001
    valid = np.isfinite(depth_meters) & (depth_meters > 0.0)
    if min_depth is not None:
        valid = valid & (depth_meters > float(min_depth))
    if max_depth is not None:
        valid = valid & (depth_meters < float(max_depth))
    if depth_meters.size == 0:
        return 0.0
    return float(valid.mean())


def stack_points_colors(pointcloud: Any, colors: Any = None) -> np.ndarray:
    """Return numpy xyzrgb point cloud for visualization.

    Kept for backward-friendly demo usage. If colors is provided, points/colors are concatenated.
    In the standardized API, pass the packed pointcloud directly.
    """
    points_array = to_numpy(pointcloud).astype(np.float32)
    if colors is None:
        return points_array
    colors_array = to_numpy(colors).astype(np.float32)
    if colors_array.max(initial=0.0) > 1.0:
        colors_array = colors_array / 255.0
    return np.concatenate([points_array, colors_array], axis=1).astype(np.float32)


def pack_obs(
    *,
    color: Optional[np.ndarray] = None,
    depth: Optional[np.ndarray] = None,
    timestamp: Optional[float] = None,
    host_time: Optional[float] = None,
    pointcloud: Any = None,
    intrinsics: Optional[np.ndarray] = None,
    intrinsics_info: Optional[dict] = None,
    depth_scale: Optional[float] = None,
    serial: Optional[str] = None,
    frame_id: Optional[int] = None,
    align_to: Optional[str] = None,
    pointcloud_frame: Optional[str] = None,
    workspace: Optional[Sequence[float]] = None,
    npoints: Optional[int] = None,
    sampling: Optional[str] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    valid_ratio: Optional[float] = None,
    mode: str = "full",
) -> dict:
    """Pack RGB-D, packed point cloud, and metadata into a plain dict."""
    obs = {
        "timestamp": timestamp,
        "host_time": host_time,
        "intrinsics": intrinsics,
        "intrinsics_info": intrinsics_info,
        "depth_scale": depth_scale,
        "meta": {
            "serial": serial,
            "frame_id": frame_id,
            "align_to": align_to,
            "depth_unit": "mm",
            "depth_dtype": "uint16",
            "pointcloud_format": "xyzrgb",
            "pointcloud_xyz_unit": "m",
            "pointcloud_rgb_range": [0.0, 1.0],
            "pointcloud_dtype": "float32",
            "pointcloud_frame": pointcloud_frame,
            "workspace": list(workspace) if workspace is not None else None,
            "workspace_unit": "m" if workspace is not None else None,
            "npoints": npoints,
            "sampling": sampling,
            "min_depth": min_depth,
            "max_depth": max_depth,
            "min_depth_unit": "m" if min_depth is not None else None,
            "max_depth_unit": "m" if max_depth is not None else None,
            "depth_valid_ratio": valid_ratio,
        },
    }

    if mode in ("rgbd", "full"):
        obs["rgb"] = color
        obs["depth"] = depth

    if mode in ("pointcloud", "full"):
        obs["pointcloud"] = pointcloud
        if pointcloud is not None:
            obs["meta"]["point_count"] = int(pointcloud.shape[0])

    return obs


def make_depth_vis(depth: ArrayLike, min_depth: float = 0.05, max_depth: float = 1.5) -> np.ndarray:
    """Create an OpenCV BGR visualization image for uint16 millimeter depth."""
    depth_meters = np.asarray(to_numpy(depth), dtype=np.float32) * 0.001
    valid = (depth_meters > 0.0) & np.isfinite(depth_meters)
    depth_clip = np.clip(depth_meters, min_depth, max_depth)
    depth_norm = (depth_clip - min_depth) / max(max_depth - min_depth, 1e-6)
    depth_uint8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    depth_vis[~valid] = 0
    return depth_vis


def blend_overlay(image_bgr: np.ndarray, overlay_bgr: Optional[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """Blend a BGR overlay image onto a BGR image for demo visualization."""
    if overlay_bgr is None:
        return image_bgr
    overlay_resized = cv2.resize(overlay_bgr, (image_bgr.shape[1], image_bgr.shape[0]))
    return cv2.addWeighted(image_bgr, 1.0 - float(alpha), overlay_resized, float(alpha), 0.0)


def vis_point_cloud(pointcloud: np.ndarray, voxel_size: Optional[float] = None, point_size: float = 5.0) -> None:
    """Visualize xyz or xyzrgb point cloud with Open3D.

    xyzrgb point clouds should use normalized RGB in [0, 1]. Values in [0, 255]
    are also accepted for convenience and will be normalized for visualization.
    """
    import open3d as o3d

    pointcloud_array = np.asarray(pointcloud)
    if pointcloud_array.ndim != 2 or pointcloud_array.shape[1] not in (3, 6):
        raise ValueError("pointcloud must have shape (N, 3) or (N, 6).")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_array[:, :3].astype(np.float64))

    if pointcloud_array.shape[1] == 6:
        color_array = pointcloud_array[:, 3:].astype(np.float64)
        if color_array.max(initial=0.0) > 1.0:
            color_array = color_array / 255.0
        pcd.colors = o3d.utility.Vector3dVector(color_array)

    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(float(voxel_size))

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="point cloud viewer")
    vis.add_geometry(pcd)
    vis.add_geometry(frame)
    vis.get_render_option().point_size = float(point_size)
    vis.run()
    vis.destroy_window()
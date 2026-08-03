"""Point cloud utilities — intrinsics helpers, depth-to-pointcloud pipeline, FPS sampling, voxel downsampling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence, Union

import numpy as np
import torch

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]
SamplingMode = Literal["none", "random", "fps", "first"]


@dataclass(frozen=True)
class PointCloudConfig:
    npoints: int | None = 1024
    min_depth: float | None = 0.05
    max_depth: float | None = 1.5
    sampling: SamplingMode = "random"
    voxel_size: float | None = None
    workspace: tuple[float, float, float, float, float, float] | None = None
    device: str = "cpu"
    return_tensor: bool = True

    def __post_init__(self) -> None:
        if self.sampling not in ("none", "random", "fps", "first"):
            raise ValueError("sampling must be one of: 'none', 'random', 'fps', 'first'.")
        if self.voxel_size is not None and self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive or None.")
        if self.workspace is not None:
            object.__setattr__(self, "workspace", check_workspace(self.workspace))  # bypass frozen


def to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def intrinsics_to_matrix(intrinsics: Any, dtype: np.typing.DTypeLike = np.float32) -> np.ndarray:
    return np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.ppx],
            [0.0, intrinsics.fy, intrinsics.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )


def intrinsics_to_dict(intrinsics: Any) -> dict:
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


def intrinsics_to_vector(K: ArrayLike) -> np.ndarray:
    K = check_intrinsics(K)
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float32)


def check_intrinsics(K: ArrayLike) -> np.ndarray:
    K = np.asarray(K, dtype=np.float32)
    if K.shape != (3, 3):
        raise ValueError(f"K must have shape (3, 3), got {K.shape}.")
    return K


def check_transform(T_out_camera: ArrayLike | None) -> np.ndarray | None:
    if T_out_camera is None:
        return None
    T = np.asarray(T_out_camera, dtype=np.float32)
    if T.shape != (4, 4):
        raise ValueError(f"T_out_camera must have shape (4, 4), got {T.shape}.")
    return T


def check_workspace(workspace: Sequence[float] | None) -> tuple[float, float, float, float, float, float] | None:
    if workspace is None:
        return None
    if len(workspace) != 6:
        raise ValueError("workspace must be [x_min, y_min, z_min, x_max, y_max, z_max].")
    x_min, y_min, z_min, x_max, y_max, z_max = [float(value) for value in workspace]
    if not (x_min < x_max and y_min < y_max and z_min < z_max):
        raise ValueError(f"invalid workspace: {workspace}.")
    return x_min, y_min, z_min, x_max, y_max, z_max


def depth_to_meters(depth: ArrayLike, depth_scale: float | None = None) -> np.ndarray:
    """Convert depth to float32 meters. Integer input × depth_scale (default 0.001); float input passthrough."""
    depth_array = to_numpy(depth)
    if np.issubdtype(depth_array.dtype, np.integer):
        scale = 0.001 if depth_scale is None else float(depth_scale)
        return depth_array.astype(np.float32) * scale
    return depth_array.astype(np.float32)


def make_rays(height: int, width: int, K: ArrayLike, device: str = "cpu") -> torch.Tensor:
    K_tensor = torch.as_tensor(check_intrinsics(K), dtype=torch.float32, device=device)
    u, v = torch.meshgrid(
        torch.arange(width, dtype=torch.float32, device=device),
        torch.arange(height, dtype=torch.float32, device=device),
        indexing="xy",
    )
    pixels = torch.stack((u, v, torch.ones_like(u)), dim=-1)
    return pixels @ torch.linalg.inv(K_tensor).T


def depth_to_xyz(
    depth: ArrayLike, K: ArrayLike | None = None, *, rays: ArrayLike | None = None, device: str = "cpu"
) -> torch.Tensor:
    depth_tensor = torch.as_tensor(depth_to_meters(depth), dtype=torch.float32, device=device)
    if depth_tensor.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {tuple(depth_tensor.shape)}.")

    if rays is None:
        if K is None:
            raise ValueError("K must be provided when rays is None.")
        height, width = int(depth_tensor.shape[0]), int(depth_tensor.shape[1])
        rays = make_rays(height, width, K, device=device)

    rays_tensor = torch.as_tensor(rays, dtype=torch.float32, device=device)
    if rays_tensor.shape[:2] != depth_tensor.shape or rays_tensor.shape[-1] != 3:
        raise ValueError(
            f"rays shape {tuple(rays_tensor.shape)} does not match depth shape {tuple(depth_tensor.shape)}."
        )
    return (rays_tensor * depth_tensor[..., None]).reshape(-1, 3)


def image_to_colors(rgb: ArrayLike | None, image_shape: tuple[int, int], device: str = "cpu") -> torch.Tensor | None:
    if rgb is None:
        return None
    rgb_array = np.ascontiguousarray(to_numpy(rgb))
    color = torch.as_tensor(rgb_array, dtype=torch.float32, device=device)
    if color.ndim != 3 or color.shape[-1] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {tuple(color.shape)}.")
    if tuple(color.shape[:2]) != tuple(image_shape):
        raise ValueError(
            f"rgb shape {tuple(color.shape[:2])} must match depth shape {image_shape}; "
            "align depth/color before generating colored point clouds."
        )
    if color.numel() > 0 and color.max() > 1.0:
        color = color / 255.0
    return color.reshape(-1, 3).clamp(0.0, 1.0)


def filter_points_by_depth(
    points: torch.Tensor,
    colors: torch.Tensor | None = None,
    min_depth: float | None = 0.05,
    max_depth: float | None = 1.5,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    keep = torch.isfinite(points).all(dim=1)
    keep = keep & (points[:, 2] > 0.0)
    if min_depth is not None:
        keep = keep & (points[:, 2] >= float(min_depth))
    if max_depth is not None:
        keep = keep & (points[:, 2] <= float(max_depth))
    return points[keep], colors[keep] if colors is not None else None


def transform_points(points: torch.Tensor, T_out_camera: ArrayLike | None = None) -> torch.Tensor:
    if T_out_camera is None:
        return points
    T = torch.as_tensor(check_transform(T_out_camera), dtype=torch.float32, device=points.device)
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat((points, ones), dim=1)
    return (points_h @ T.T)[:, :3]


def crop_points(
    points: torch.Tensor,
    colors: torch.Tensor | None = None,
    workspace: Sequence[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if workspace is None:
        return points, colors
    x_min, y_min, z_min, x_max, y_max, z_max = workspace
    keep = (points[:, 0] >= x_min) & (points[:, 0] <= x_max)
    keep = keep & (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
    keep = keep & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[keep], colors[keep] if colors is not None else None


def voxel_down_sample(
    points: torch.Tensor,
    colors: torch.Tensor | None = None,
    voxel_size: float = 0.005,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Voxel-average downsampling via Open3D."""
    if points.shape[0] == 0 or voxel_size <= 0:
        return points, colors
    return _voxel_down_sample_o3d(points, colors, voxel_size)


def _voxel_down_sample_o3d(
    points: torch.Tensor,
    colors: torch.Tensor | None,
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy().astype(np.float64))
    if colors is not None:
        c = colors.detach().cpu().numpy().astype(np.float64)
        if c.size and c.max() > 1.0:
            c /= 255.0
        pcd.colors = o3d.utility.Vector3dVector(c)

    pcd = pcd.voxel_down_sample(voxel_size)

    new_points = torch.as_tensor(np.asarray(pcd.points), dtype=points.dtype, device=points.device)
    new_colors = None
    if colors is not None and pcd.has_colors():
        new_colors = torch.as_tensor(np.asarray(pcd.colors), dtype=colors.dtype, device=colors.device)

    return new_points, new_colors


def sample_points(
    points: torch.Tensor,
    colors: torch.Tensor | None = None,
    npoints: int | None = 1024,
    sampling: SamplingMode = "random",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if sampling not in ("none", "random", "fps", "first"):
        raise ValueError("sampling must be one of: 'none', 'random', 'fps', 'first'.")
    if sampling == "none" or npoints is None:
        return points, colors
    if npoints <= 0:
        raise ValueError("npoints must be positive or None.")

    count = int(points.shape[0])
    if count == 0:
        raise ValueError("No valid points after depth filtering/cropping.")

    if sampling == "first":
        index = torch.arange(min(count, npoints), device=points.device)
        if count < npoints:
            index = index.repeat(int(np.ceil(npoints / count)))[:npoints]
    elif sampling == "fps" and count >= npoints:
        try:
            import pytorch3d.ops as torch3d_ops

            if torch.cuda.is_available() and points.device.type == "cpu":
                try:
                    pts_gpu = points.cuda()
                    index = torch3d_ops.sample_farthest_points(pts_gpu[None], K=npoints)[1][0]
                    index = index.cpu()
                except RuntimeError:
                    index = torch3d_ops.sample_farthest_points(points[None], K=npoints)[1][0]
            else:
                try:
                    index = torch3d_ops.sample_farthest_points(points[None], K=npoints)[1][0]
                except RuntimeError:
                    pts_cpu = points.cpu()
                    index = torch3d_ops.sample_farthest_points(pts_cpu[None], K=npoints)[1][0]
        except ImportError:
            logger.warning("pytorch3d not available — falling back to random sampling instead of FPS")
            index = torch.randperm(count, device=points.device)[:npoints]
    elif count >= npoints:
        index = torch.randperm(count, device=points.device)[:npoints]
    else:
        base = torch.randperm(count, device=points.device)
        extra = torch.randint(0, count, (npoints - count,), device=points.device)
        index = torch.cat((base, extra), dim=0)

    return points[index], colors[index] if colors is not None else None


def pack_xyzrgb(points: torch.Tensor, colors: torch.Tensor | None = None) -> torch.Tensor:
    points = points.to(dtype=torch.float32)
    if colors is None:
        colors = torch.zeros_like(points)
    else:
        colors = colors.to(dtype=torch.float32).clamp(0.0, 1.0)
    return torch.cat((points, colors), dim=1)


def rgbd_to_pointcloud(
    depth: ArrayLike,
    K: ArrayLike | None = None,
    rgb: ArrayLike | None = None,
    *,
    rays: ArrayLike | None = None,
    config: PointCloudConfig | None = None,
    T_out_camera: ArrayLike | None = None,
    workspace: Sequence[float] | None = None,
    npoints: int | None = None,
    min_depth: float | None = None,
    max_depth: float | None = None,
    sampling: SamplingMode | None = None,
    voxel_size: float | None = None,
    device: str | None = None,
    return_tensor: bool | None = None,
    depth_scale: float | None = None,
) -> Union[torch.Tensor, np.ndarray]:
    if config is None:
        config = PointCloudConfig()

    overrides: dict[str, Any] = {}
    if workspace is not None:
        overrides["workspace"] = check_workspace(workspace)
    if npoints is not None:
        overrides["npoints"] = npoints
    if min_depth is not None:
        overrides["min_depth"] = min_depth
    if max_depth is not None:
        overrides["max_depth"] = max_depth
    if sampling is not None:
        overrides["sampling"] = sampling
    if voxel_size is not None:
        overrides["voxel_size"] = voxel_size
    if device is not None:
        overrides["device"] = device
    if return_tensor is not None:
        overrides["return_tensor"] = return_tensor
    if overrides:
        config = replace(config, **overrides)

    depth_m = depth_to_meters(depth, depth_scale=depth_scale)
    if depth_m.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {depth_m.shape}.")
    height, width = int(depth_m.shape[0]), int(depth_m.shape[1])

    points = depth_to_xyz(depth_m, K, rays=rays, device=config.device)
    colors = image_to_colors(rgb, (height, width), device=config.device) if rgb is not None else None

    points, colors = filter_points_by_depth(points, colors, config.min_depth, config.max_depth)
    points = transform_points(points, T_out_camera)
    points, colors = crop_points(points, colors, config.workspace)
    if config.voxel_size is not None:
        points, colors = voxel_down_sample(points, colors, config.voxel_size)
    points, colors = sample_points(points, colors, config.npoints, config.sampling)

    pointcloud = pack_xyzrgb(points, colors)
    if config.return_tensor:
        return pointcloud
    return pointcloud.detach().cpu().numpy().astype(np.float32)


def depth_valid_ratio(depth: ArrayLike, min_depth: float | None = 0.05, max_depth: float | None = 1.5) -> float:
    depth_m = depth_to_meters(depth)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if min_depth is not None:
        valid = valid & (depth_m >= float(min_depth))
    if max_depth is not None:
        valid = valid & (depth_m <= float(max_depth))
    return float(valid.mean()) if depth_m.size else 0.0


def make_depth_vis(depth: ArrayLike, min_depth: float = 0.05, max_depth: float = 1.5) -> np.ndarray:
    import cv2

    depth_m = depth_to_meters(depth)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_clip = np.clip(depth_m, min_depth, max_depth)
    depth_norm = (depth_clip - min_depth) / max(max_depth - min_depth, 1e-6)
    depth_uint8 = (255.0 * (1.0 - depth_norm)).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    depth_vis[~valid] = 0
    return depth_vis

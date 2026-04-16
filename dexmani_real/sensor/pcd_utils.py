import torch
import numpy as np
import open3d as o3d
import pytorch3d.ops as torch3d_ops
from typing import Optional, Dict, Tuple


def depth_to_pointcloud(
    depth: np.ndarray,
    intr: np.ndarray,
    color: Optional[np.ndarray] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    H, W = depth.shape
    K_inv = torch.from_numpy(intr).float().inverse()

    uu, vv = torch.meshgrid(
        torch.arange(W).float(),
        torch.arange(H).float(),
        indexing="xy",
    )
    pixel_coords = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1)

    camera_dirs = pixel_coords @ K_inv.T
    points = camera_dirs * torch.from_numpy(depth).float().unsqueeze(-1)
    points = points.reshape(-1, 3)

    colors = None
    if color is not None:
        colors = torch.from_numpy(color).float().reshape(-1, 3)

    return points, colors


def transform_to_world(
    points: torch.Tensor,
    colors: Optional[torch.Tensor],
    transform: Optional[np.ndarray],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    if transform is None:
        return points, colors
    T = torch.from_numpy(transform).float().to(points.device)
    ones = torch.ones_like(points[:, :1])
    pts_homo = torch.cat([points, ones], dim=1) 
    pts_world = (T @ pts_homo.T).T[:, :3]
    return pts_world, colors


def apply_valid_mask(
    points: torch.Tensor,
    colors: Optional[torch.Tensor],
    min_depth: float = 0.1,
    max_depth: float = 5.0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    z = points[:, 2]
    valid = torch.isfinite(points).all(dim=1) & (z > min_depth) & (z < max_depth)
    return points[valid], colors[valid] if colors is not None else None


def workspace_crop(
    points: torch.Tensor,
    colors: Optional[torch.Tensor],
    bound: list[float],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = bound
    mask = (
        (points[:, 0] > x_lo) & (points[:, 0] < x_hi)
        & (points[:, 1] > y_lo) & (points[:, 1] < y_hi)
        & (points[:, 2] > z_lo) & (points[:, 2] < z_hi)
    )
    return points[mask], colors[mask] if colors is not None else None


def fps_downsample(
    points: torch.Tensor,
    colors: Optional[torch.Tensor],
    npoints: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    N = points.shape[0]
    if N <= npoints:
        idx = torch.randperm(N, device=points.device)
        idx = torch.cat([idx, idx[: npoints - N]])
        return points[idx], colors[idx] if colors is not None else None

    _, sample_idx = torch3d_ops.sample_farthest_points(points.unsqueeze(0), K=npoints)
    sample_idx = sample_idx.squeeze(0)
    return points[sample_idx], colors[sample_idx] if colors is not None else None


def get_pointcloud(
    depth: np.ndarray,
    intr: np.ndarray,
    color: np.ndarray,
    *,
    bound: Optional[list[float]] = None,
    npoints: int = 1024,
    min_depth: float = 0.1,
    max_depth: float = 5.0,
    device: str = "cpu",
    transform: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:

    points, colors = depth_to_pointcloud(depth, intr, color)
    points = points.to(device)
    if colors is not None:
        colors = colors.to(device)

    points, colors = apply_valid_mask(points, colors, min_depth, max_depth)

    points, colors = transform_to_world(points, colors, transform)

    if bound is not None:
        points, colors = workspace_crop(points, colors, bound)

    points, colors = fps_downsample(points, colors, npoints)

    return {
        "pos": points.detach().cpu().numpy().astype(np.float32),
        "color": colors.detach().cpu().numpy().astype(np.uint8) if colors is not None else None,
    }


def vis_point_cloud(points: np.ndarray, voxel_size: Optional[float] = None) -> None:
    
    assert points.ndim == 2 and points.shape[1] in [3, 6], "points should be (N, 3) or (N, 6)"

    point_size = 5
    frame_size = 0.1
    window_name = "pcd viewer"
    is_centered = False

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] == 6:
        colors = points[:, 3:]
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel_size:
        pcd = pcd.voxel_down_sample(voxel_size)

    if is_centered:
        pcd.translate(-pcd.get_center())

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    vis.add_geometry(frame)
    opt = vis.get_render_option()
    opt.point_size = point_size
    vis.run()
    vis.destroy_window()
import numpy as np
import torch


def make_rays(height, width, intr, device="cpu"):
    if intr is None:
        raise ValueError("intr must be provided when rays is None.")

    u, v = torch.meshgrid(
        torch.arange(width, device=device, dtype=torch.float32),
        torch.arange(height, device=device, dtype=torch.float32),
        indexing="xy",
    )
    uv1 = torch.stack((u, v, torch.ones_like(u)), dim=-1)
    intr = torch.as_tensor(intr, dtype=torch.float32, device=device)
    return uv1 @ torch.linalg.inv(intr).T


def depth_to_points(depth, rays, device="cpu"):
    depth = torch.as_tensor(depth, dtype=torch.float32, device=device)
    rays = torch.as_tensor(rays, dtype=torch.float32, device=device)
    if rays.shape[:2] != depth.shape[:2]:
        raise ValueError(f"rays shape {tuple(rays.shape[:2])} must match depth shape {tuple(depth.shape[:2])}.")
    return (rays * depth[..., None]).reshape(-1, 3)


def image_to_colors(color, device="cpu"):
    return torch.as_tensor(color, dtype=torch.uint8, device=device).reshape(-1, 3)


def mask_depth(points, colors=None, min_depth=0.1, max_depth=5.0):
    z = points[:, 2]
    keep = torch.isfinite(points).all(dim=1)
    keep &= z > min_depth
    keep &= z < max_depth
    points = points[keep]
    colors = colors[keep] if colors is not None else None
    return points, colors


def transform_points(points, transform=None):
    if transform is None:
        return points
    transform = torch.as_tensor(transform, dtype=torch.float32, device=points.device)
    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat((points, ones), dim=1)
    return (points_h @ transform.T)[:, :3]


def crop_points(points, colors=None, bound=None):
    if bound is None:
        return points, colors
    x0, x1, y0, y1, z0, z1 = bound
    keep = (points[:, 0] > x0) & (points[:, 0] < x1)
    keep &= (points[:, 1] > y0) & (points[:, 1] < y1)
    keep &= (points[:, 2] > z0) & (points[:, 2] < z1)
    points = points[keep]
    colors = colors[keep] if colors is not None else None
    return points, colors


def sample_points(points, colors=None, npoints=1024):
    if npoints is None:
        return points, colors
    if npoints <= 0:
        raise ValueError("npoints must be positive or None.")

    n = points.shape[0]
    if n == 0:
        raise ValueError("No valid points after depth mask/crop. Check depth range, crop bound, and camera pose.")

    if n < npoints:
        base = torch.randperm(n, device=points.device)
        extra = torch.randint(0, n, (npoints - n,), device=points.device)
        idx = torch.cat((base, extra), dim=0)
    else:
        try:
            import pytorch3d.ops as torch3d_ops

            idx = torch3d_ops.sample_farthest_points(points[None], K=npoints)[1][0]
        except ImportError:
            idx = torch.randperm(n, device=points.device)[:npoints]

    points = points[idx]
    colors = colors[idx] if colors is not None else None
    return points, colors


def rgbd_to_pointcloud(
    depth,
    intr=None,
    color=None,
    *,
    rays=None,
    bound=None,
    npoints=1024,
    min_depth=0.1,
    max_depth=5.0,
    transform=None,
    device="cpu",
    return_tensor=True,
):
    depth_shape = depth.shape[:2]
    if color is not None and color.shape[:2] != depth_shape:
        raise ValueError(f"color shape {color.shape[:2]} must match depth shape {depth_shape}.")

    if rays is None:
        rays = make_rays(depth_shape[0], depth_shape[1], intr, device=device)

    points = depth_to_points(depth, rays, device=device)
    colors = image_to_colors(color, device=device) if color is not None else None

    points, colors = mask_depth(points, colors, min_depth=min_depth, max_depth=max_depth)
    points = transform_points(points, transform=transform)
    points, colors = crop_points(points, colors, bound=bound)
    points, colors = sample_points(points, colors, npoints=npoints)

    if return_tensor:
        return points, colors

    points = points.cpu().numpy().astype(np.float32)
    colors = colors.cpu().numpy() if colors is not None else None
    return points, colors


def vis_point_cloud(points, voxel_size=None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    if points.shape[1] == 6:
        colors = points[:, 3:]
        if colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    if voxel_size is not None:
        pcd = pcd.voxel_down_sample(voxel_size)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="pcd viewer")
    vis.add_geometry(pcd)
    vis.add_geometry(frame)
    vis.get_render_option().point_size = 5
    vis.run()
    vis.destroy_window()

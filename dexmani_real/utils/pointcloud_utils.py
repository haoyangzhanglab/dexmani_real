"""Point cloud utilities — depth-to-pointcloud, FPS sampling, voxel downsampling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence, Union

import numpy as np
import numba as _numba
import torch


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
        # object.__setattr__ bypasses frozen=True in __post_init__ to normalize
        # the workspace tuple after construction.
        if self.sampling not in ("none", "random", "fps", "first"):
            raise ValueError("sampling must be one of: 'none', 'random', 'fps', 'first'.")
        if self.voxel_size is not None and self.voxel_size <= 0:
            raise ValueError("voxel_size must be positive or None.")
        if self.workspace is not None:
            object.__setattr__(self, "workspace", check_workspace(self.workspace))


def to_numpy(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def intrinsics_to_matrix(intrinsics: Any, dtype: np.dtype = np.float32) -> np.ndarray:
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


def depth_to_meters(depth: ArrayLike) -> np.ndarray:
    depth_array = to_numpy(depth)
    if np.issubdtype(depth_array.dtype, np.integer):
        return depth_array.astype(np.float32) * 0.001
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


def depth_to_xyz(depth: ArrayLike, K: ArrayLike | None = None, *, rays: ArrayLike | None = None, device: str = "cpu") -> torch.Tensor:
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
        raise ValueError(f"rays shape {tuple(rays_tensor.shape)} does not match depth shape {tuple(depth_tensor.shape)}.")
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
    """Voxel downsampling — average points within each voxel. Returns (points, colors).

    Should be called before sample_points so that subsequent FPS selects from
    spatially uniform voxel centers. Prefers open3d C++ (~1ms) over torch (~300ms/250k points).
    """
    if points.shape[0] == 0 or voxel_size <= 0:
        return points, colors

    # Prefer open3d (100x+ faster than torch scatter on CPU)
    try:
        return _voxel_down_sample_o3d(points, colors, voxel_size)
    except ImportError:
        return _voxel_down_sample_torch(points, colors, voxel_size)


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


def _voxel_down_sample_torch(
    points: torch.Tensor,
    colors: torch.Tensor | None,
    voxel_size: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    voxel_ijk = (points / voxel_size).floor().long()
    unique_voxels, inverse = voxel_ijk.unique(dim=0, return_inverse=True)
    n_voxels = unique_voxels.shape[0]

    new_points = torch.zeros(n_voxels, 3, device=points.device, dtype=torch.float32)
    new_points.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, 3), points.float())
    count = torch.zeros(n_voxels, device=points.device, dtype=torch.float32)
    count.scatter_add_(0, inverse, torch.ones(points.shape[0], device=points.device, dtype=torch.float32))
    new_points /= count.unsqueeze(-1)
    new_points = new_points.to(points.dtype)

    new_colors = None
    if colors is not None:
        new_colors = torch.zeros(n_voxels, 3, device=colors.device, dtype=torch.float32)
        new_colors.scatter_add_(0, inverse.unsqueeze(-1).expand(-1, 3), colors.float())
        new_colors /= count.unsqueeze(-1)
        new_colors = new_colors.to(colors.dtype)

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

            # Try GPU path first (if available and points are on CPU).
            # Fall back to CPU if the pytorch3d extension lacks GPU support.
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
                    # pytorch3d lacks GPU support — move to CPU and retry
                    pts_cpu = points.cpu()
                    index = torch3d_ops.sample_farthest_points(pts_cpu[None], K=npoints)[1][0]
        except ImportError:
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
) -> Union[torch.Tensor, np.ndarray]:
    if config is None:
        config = PointCloudConfig()

    if workspace is not None:
        config = replace(config, workspace=check_workspace(workspace))
    if npoints is not None:
        config = replace(config, npoints=npoints)
    if min_depth is not None:
        config = replace(config, min_depth=min_depth)
    if max_depth is not None:
        config = replace(config, max_depth=max_depth)
    if sampling is not None:
        config = replace(config, sampling=sampling)
    if voxel_size is not None:
        config = replace(config, voxel_size=voxel_size)
    if device is not None:
        config = replace(config, device=device)
    if return_tensor is not None:
        config = replace(config, return_tensor=return_tensor)

    depth_m = depth_to_meters(depth)
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


# ---------------------------------------------------------------------------
# Flying-pixel removal kernel (numba JIT)
# ---------------------------------------------------------------------------


@_numba.njit(cache=True, nogil=True)
def _remove_flying_pixels_kernel(
    filtered: np.ndarray,
    depth: np.ndarray,
    Gx: np.ndarray,
    Gy: np.ndarray,
    edge_ys: np.ndarray,
    edge_xs: np.ndarray,
    noise_threshold: float,
    margin: float,
    sample_radius: int,
    min_valid_samples: int,
    pad: int,
    gap: int,
    beta: float,
    left_buf: np.ndarray,
    right_buf: np.ndarray,
) -> None:
    """Numba-JIT kernel: modify ``filtered`` in-place, setting flying pixels to 0."""
    H, W = depth.shape
    max_sr = sample_radius

    for idx in range(len(edge_ys)):
        y = edge_ys[idx]
        x = edge_xs[idx]

        if x < pad or x >= W - pad or y < pad or y >= H - pad:
            continue

        gx = Gx[y, x]
        gy = Gy[y, x]
        norm = (gx * gx + gy * gy) ** 0.5
        if norm == 0.0:
            continue
        dx = gx / norm
        dy = gy / norm

        center = depth[y, x]
        if center <= 0.0:
            continue

        # Effective thresholds. beta > 0 → depth-adaptive: interpret noise_threshold
        # and margin as dimensionless multipliers of the local lateral spacing
        # (center * beta), so one setting covers the whole depth range. beta == 0 →
        # absolute meters (original behaviour).
        if beta > 0.0:
            eta_c = center * beta
            noise_thr = noise_threshold * eta_c
            margin_c = margin * eta_c
            if margin_c < 0.004:  # near-field floor (m) so margin never vanishes
                margin_c = 0.004
        else:
            noise_thr = noise_threshold
            margin_c = margin

        n_left = 0
        n_right = 0

        # Sample from k = gap + 1 so the near-edge ramp body (1..gap px) is skipped
        # and never contaminates each side's clean-surface statistics.
        for k in range(gap + 1, max_sr + 1):
            # --- negative direction (foreground side) ---
            sx = x - k * dx
            sy = y - k * dy
            x0 = int(np.floor(sx))
            y0 = int(np.floor(sy))
            x1 = x0 + 1
            y1 = y0 + 1
            if x0 >= 0 and x1 < W and y0 >= 0 and y1 < H:
                fx = sx - x0
                fy = sy - y0
                q00 = depth[y0, x0]
                q10 = depth[y0, x1]
                q01 = depth[y1, x0]
                q11 = depth[y1, x1]
                if q00 > 0.0 and q10 > 0.0 and q01 > 0.0 and q11 > 0.0:
                    d = (
                        q00 * (1.0 - fx) * (1.0 - fy)
                        + q10 * fx * (1.0 - fy)
                        + q01 * (1.0 - fx) * fy
                        + q11 * fx * fy
                    )
                    if d > 0.0 and n_left < max_sr:
                        left_buf[n_left] = d
                        n_left += 1

            # --- positive direction (background side) ---
            sx = x + k * dx
            sy = y + k * dy
            x0 = int(np.floor(sx))
            y0 = int(np.floor(sy))
            x1 = x0 + 1
            y1 = y0 + 1
            if x0 >= 0 and x1 < W and y0 >= 0 and y1 < H:
                fx = sx - x0
                fy = sy - y0
                q00 = depth[y0, x0]
                q10 = depth[y0, x1]
                q01 = depth[y1, x0]
                q11 = depth[y1, x1]
                if q00 > 0.0 and q10 > 0.0 and q01 > 0.0 and q11 > 0.0:
                    d = (
                        q00 * (1.0 - fx) * (1.0 - fy)
                        + q10 * fx * (1.0 - fy)
                        + q01 * (1.0 - fx) * fy
                        + q11 * fx * fy
                    )
                    if d > 0.0 and n_right < max_sr:
                        right_buf[n_right] = d
                        n_right += 1

        # --- guard 1: enough samples on both sides ---
        if n_left < min_valid_samples or n_right < min_valid_samples:
            continue

        # --- guard 2: each side is a clean surface ---
        # left stats
        mean_l = 0.0
        for i in range(n_left):
            mean_l += left_buf[i]
        mean_l /= n_left
        var_l = 0.0
        for i in range(n_left):
            diff = left_buf[i] - mean_l
            var_l += diff * diff
        std_l = (var_l / n_left) ** 0.5
        if std_l > noise_thr:
            continue

        # right stats
        mean_r = 0.0
        for i in range(n_right):
            mean_r += right_buf[i]
        mean_r /= n_right
        var_r = 0.0
        for i in range(n_right):
            diff = right_buf[i] - mean_r
            var_r += diff * diff
        std_r = (var_r / n_right) ** 0.5
        if std_r > noise_thr:
            continue

        # --- guard 3: sides genuinely differ ---
        diff_sr = mean_l - mean_r
        if diff_sr < 0.0:
            diff_sr = -diff_sr
        if diff_sr < margin_c * 2.0:
            continue

        # --- flying-pixel test ---
        dist_l = center - mean_l
        if dist_l < 0.0:
            dist_l = -dist_l
        dist_r = center - mean_r
        if dist_r < 0.0:
            dist_r = -dist_r

        if dist_l > margin_c and dist_r > margin_c:
            filtered[y, x] = 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def remove_flying_pixels_at_edges(
    depth: np.ndarray,
    edge_threshold: float = 0.02,
    noise_threshold: float = 0.005,
    margin: float = 0.01,
    sample_radius: int = 5,
    min_valid_samples: int = 2,
    gap: int = 0,
    beta: float = 0.0,
    edge_gate_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Remove ToF/LiDAR flying pixels at depth edges by gradient-direction consistency.

    Flying pixels (mixed pixels) occur when a finite-size laser spot straddles a
    depth discontinuity. The returned signal blends foreground and background,
    producing a "ramp" of 1-3 pixels bridging the two surfaces in the depth map.

    This function detects these ramp pixels by sampling along the depth gradient
    direction on both sides: if both sides are "clean surfaces" (low variance)
    and the center pixel depth falls **between** them (neither foreground nor
    background), it is deleted (set to 0).

    Design constraints:
    - **Never introduces false depth** — only deletes (sets to 0), never fills or
      interpolates. A deleted pixel is indistinguishable from an original
      invalid pixel.
    - **Minimises false deletion** — three conservative guards must all pass
      before a pixel is removed; any uncertainty → keep.

    Args:
        depth: (H, W) float32 depth map in **meters**. 0 = invalid pixel.
        edge_threshold: Minimum depth gradient magnitude (meters) to consider a
            pixel as an edge candidate. Default 0.02 (2 cm).
        noise_threshold: Maximum standard deviation (meters) allowed on either
            side of the edge for the surfaces to be considered "clean".
            Default 0.005 (5 mm).
        margin: Minimum distance (meters) the center pixel must have from BOTH
            side means to be classified as flying. Larger → more conservative.
            Default 0.01 (1 cm).
        sample_radius: Number of pixels to sample along the gradient direction
            on each side. Default 5.
        min_valid_samples: Minimum number of valid depth samples required on
            each side. Default 2.
        gap: Sampling dead-zone (pixels). Samples are taken from k = gap + 1
            along the gradient direction, so the 1..gap px ramp body adjacent to
            the edge does not contaminate each side's clean-surface statistics.
            Default 0 (sample from k = 1, original behaviour).
        beta: Per-pixel angular resolution (rad/pixel), typically 1 / fx. When
            > 0, enables depth-adaptive mode: edge_threshold, noise_threshold and
            margin are interpreted as dimensionless multipliers of the local
            lateral spacing (depth * beta) instead of absolute meters, so one
            setting covers the whole depth range. Default 0.0 (absolute meters,
            original behaviour).
        edge_gate_mask: Optional (H, W) bool array. When given, an edge candidate
            is kept only where the mask is True (logical AND with the depth-gradient
            candidates) — e.g. a color/IR edge band, so deletion needs a second
            independent cue. Default None (no gating).

    Returns:
        filtered: (H, W) float32 depth map, same shape as input. Flying pixels
            are set to 0; all other pixels are unchanged.

    Performance:
        ~1-3 ms for 640×480 (numba JIT, real-time ready at 50+ Hz).
    """
    import cv2

    H, W = depth.shape
    filtered = depth.copy()

    # ── Fast path: no valid depth ──────────────────────────────────────
    valid_mask = depth > 0.0
    if valid_mask.sum() < min_valid_samples * 2:
        return filtered

    # ── 1. Depth gradient via Sobel ────────────────────────────────────
    depth_f32 = np.where(valid_mask, depth.astype(np.float32), 0.0)
    Gx = cv2.Sobel(depth_f32, cv2.CV_32F, 1, 0, ksize=3)
    Gy = cv2.Sobel(depth_f32, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(Gx**2 + Gy**2)

    # Discard pixels where the 3×3 Sobel window touches an invalid pixel.
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbour_count = cv2.filter2D(valid_mask.astype(np.uint8), -1, kernel)
    grad_mag[neighbour_count != 9] = 0.0

    # ── 2. Edge candidate pixels ───────────────────────────────────────
    if beta > 0.0:
        # Depth-adaptive gradient threshold: η(d) = edge_threshold · d · beta.
        edge_candidates = grad_mag > (edge_threshold * depth_f32 * beta)
    else:
        edge_candidates = grad_mag > edge_threshold
    if edge_gate_mask is not None:
        edge_candidates &= edge_gate_mask
    edge_ys, edge_xs = np.where(edge_candidates)

    if len(edge_ys) == 0:
        return filtered

    pad = sample_radius + 1

    # Pre-allocate reusable per-candidate buffers so the hot loop never
    # calls malloc.
    left_buf = np.empty(sample_radius, dtype=np.float64)
    right_buf = np.empty(sample_radius, dtype=np.float64)

    # ── 3. JIT-compiled per-candidate loop ─────────────────────────────
    _remove_flying_pixels_kernel(
        filtered,
        depth_f32,
        Gx,
        Gy,
        edge_ys.astype(np.int64, copy=False),
        edge_xs.astype(np.int64, copy=False),
        float(noise_threshold),
        float(margin),
        int(sample_radius),
        int(min_valid_samples),
        int(pad),
        int(gap),
        float(beta),
        left_buf,
        right_buf,
    )

    return filtered


def vis_point_cloud(pointcloud: ArrayLike, voxel_size: float | None = None, point_size: float = 5.0) -> None:
    import open3d as o3d

    pointcloud_array = to_numpy(pointcloud).astype(np.float32)
    if pointcloud_array.ndim != 2 or pointcloud_array.shape[1] not in (3, 6):
        raise ValueError("pointcloud must have shape (N, 3) or (N, 6).")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud_array[:, :3].astype(np.float64))

    if pointcloud_array.shape[1] == 6:
        colors = pointcloud_array[:, 3:].astype(np.float64)
        if colors.size and colors.max() > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

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
"""Point cloud utilities — depth validity masking, depth-to-pointcloud, FPS sampling, voxel downsampling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Sequence, Union

import numpy as np
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


@dataclass(frozen=True)
class DepthEdgeConfig:
    """Depth-discontinuity band zeroing (native depth domain, before alignment).

    Per-pixel threshold: T_edge(z) = clip(n_sigma * sigma_z(z), t_min, t_max),
    where sigma_z(z) = sigma_poly[0] + sigma_poly[1]*z + ... (meters, low order
    first) is the planar depth std at distance z — calibrate it from per-pixel
    TEMPORAL std over a static plane. Defaults reproduce the 8-12 mm
    experimental window at 0.5-1.0 m and must be calibrated for real use.
    t_max must stay below the minimum object height the system must keep
    (~0.8x is a reasonable margin); None disables the clamp.
    """

    sigma_poly: tuple[float, ...] = (0.0010, 0.0012)  # sigma_z(z) = c0 + c1*z + ... (meters)
    n_sigma: float = 5.0
    t_min: float = 0.008  # threshold floor (m)
    t_max: float | None = None  # threshold ceiling (m), ~0.8x min object height
    dilate_px: int = 1  # edge-band dilation radius (band width ~= 2 + 2*dilate_px)

    def __post_init__(self) -> None:
        if len(self.sigma_poly) == 0:
            raise ValueError("sigma_poly must have at least one coefficient.")
        if self.t_min <= 0:
            raise ValueError("t_min must be positive.")
        if self.t_max is not None and self.t_max < self.t_min:
            raise ValueError("t_max must be >= t_min.")
        if self.dilate_px < 0:
            raise ValueError("dilate_px must be >= 0.")


def build_edge_threshold_lut(depth_scale: float, config: DepthEdgeConfig) -> np.ndarray:
    """Precompute a (65536,) uint16 LUT: raw depth value -> T_edge in raw units.

    Built once per connect (depth_scale = depth_units); per frame the
    z-dependent threshold is then a single uint16 gather with no float math.
    """
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive.")
    z = np.arange(65536, dtype=np.float64) * float(depth_scale)
    sigma = np.zeros_like(z)
    for coeff in reversed(config.sigma_poly):  # Horner
        sigma = sigma * z + float(coeff)
    t = np.maximum(config.n_sigma * sigma, config.t_min)
    if config.t_max is not None:
        t = np.minimum(t, config.t_max)
    return np.clip(np.rint(t / depth_scale), 1, 65535).astype(np.uint16)


def compute_depth_edge_mask(depth_raw: ArrayLike, t_lut: np.ndarray, dilate_px: int = 1) -> np.ndarray:
    """Depth-discontinuity band mask (True = zero this pixel), native depth domain.

    Exact 8-neighbour max jump via morphology, entirely in uint16 raw units:
        jump(p) = max_{q in N8(p), q valid} |z(p) - z(q)|
                = max( dilate3(z)(p) - z(p),  z(p) - erode3(z_sub)(p) )
    Invalid pixels (raw == 0) are excluded from every neighbourhood — 0 is the
    neutral element for the max side, and they are substituted with 65535 on the
    min side — so hole boundaries are NOT flagged as edges. For a valid center
    the 3x3 window contains the center itself, hence dilate >= z >= erode and
    the uint16 differences cannot underflow; invalid centers may wrap but are
    removed by `& valid` BEFORE dilation, so garbage never propagates.

    Call on RAW depth with confidence/IR-rejected pixels already zeroed, and
    BEFORE any alignment — resampling mixes depth/RGB/occlusion boundaries.
    """
    import cv2

    depth_array = to_numpy(depth_raw)
    if depth_array.ndim != 2 or depth_array.dtype != np.uint16:
        raise ValueError(f"depth_raw must be (H, W) uint16, got {depth_array.shape} {depth_array.dtype}.")
    t_lut = np.asarray(t_lut)
    if t_lut.shape != (65536,) or t_lut.dtype != np.uint16:
        raise ValueError("t_lut must be the (65536,) uint16 array from build_edge_threshold_lut().")

    valid = depth_array != 0
    z_min_src = np.where(valid, depth_array, np.uint16(65535))
    kernel3 = np.ones((3, 3), dtype=np.uint8)
    local_max = cv2.dilate(depth_array, kernel3)
    local_min = cv2.erode(z_min_src, kernel3)
    jump = np.maximum(local_max - depth_array, depth_array - local_min)

    edge = (jump > t_lut[depth_array]) & valid
    if dilate_px > 0 and edge.any():
        size = 2 * dilate_px + 1
        edge = cv2.dilate(edge.astype(np.uint8), np.ones((size, size), dtype=np.uint8)).astype(bool)
    return edge


@dataclass(frozen=True)
class DepthValidityConfig:
    """Image-domain depth validity thresholds (confidence / IR gating, L515-oriented).

    Thresholds are sensor-specific — defaults are conservative starting points for
    the L515 (8-bit IR). Set a field to None to disable that sub-check.
    """

    confidence_min: int | None = 2  # keep pixels with confidence >= this
    ir_min: int | None = 2  # reject extremely low IR return (weak echo)
    ir_saturation: int | None = 250  # reject saturated IR (overexposure / specular)
    saturation_dilate_px: int = 3  # dilate saturation mask to kill the specular halo
    edge: DepthEdgeConfig | None = None  # depth-discontinuity band zeroing (None = off)

    def __post_init__(self) -> None:
        if self.saturation_dilate_px < 0:
            raise ValueError("saturation_dilate_px must be >= 0.")


def compute_depth_valid_mask(
    depth: ArrayLike,
    confidence: ArrayLike | None = None,
    ir: ArrayLike | None = None,
    config: DepthValidityConfig | None = None,
) -> np.ndarray:
    """Per-pixel (H, W) bool validity mask: depth > 0, confidence gate, IR gate.

    The IR gate rejects extremely low return (ir < ir_min) and saturation
    (ir >= ir_saturation) — the saturation mask is dilated by saturation_dilate_px
    so the corrupted halo around specular highlights is removed too. IR saturation
    and specular reflection produce dense, mixed-direction depth spikes that 3-D
    outlier removal cannot catch, hence this image-space mask.

    `confidence`/`ir` must be pixel-registered with `depth`. On L515 they are
    registered with the RAW depth frame; when streaming with
    align_mode="depth_to_color", apply this mask to the raw depth (invalid -> 0)
    BEFORE rs.align — zeroed pixels do not project through alignment.
    """
    if config is None:
        config = DepthValidityConfig()

    depth_array = to_numpy(depth)
    if depth_array.ndim != 2:
        raise ValueError(f"depth must have shape (H, W), got {depth_array.shape}.")
    valid = depth_array > 0

    if confidence is not None and config.confidence_min is not None:
        conf = to_numpy(confidence)
        if conf.shape != depth_array.shape:
            raise ValueError(
                f"confidence shape {conf.shape} must match depth shape {depth_array.shape}; "
                "confidence is registered to the raw depth frame — mask before alignment."
            )
        valid &= conf >= config.confidence_min

    if ir is not None:
        ir_array = to_numpy(ir)
        if ir_array.shape != depth_array.shape:
            raise ValueError(
                f"ir shape {ir_array.shape} must match depth shape {depth_array.shape}; "
                "IR is registered to the raw depth frame — mask before alignment."
            )
        if config.ir_min is not None:
            valid &= ir_array >= config.ir_min
        if config.ir_saturation is not None:
            saturated = ir_array >= config.ir_saturation
            if config.saturation_dilate_px > 0 and saturated.any():
                import cv2

                kernel_size = 2 * config.saturation_dilate_px + 1
                kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                saturated = cv2.dilate(saturated.astype(np.uint8), kernel).astype(bool)
            valid &= ~saturated

    return valid


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
    confidence: ArrayLike | None = None,
    ir: ArrayLike | None = None,
    validity: DepthValidityConfig | None = None,
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

    if confidence is not None or ir is not None:
        # Image-domain validity gate (confidence + IR) — invalid pixels -> 0,
        # dropped by filter_points_by_depth after back-projection.
        valid = compute_depth_valid_mask(depth_m, confidence=confidence, ir=ir, config=validity)
        depth_m = np.where(valid, depth_m, np.float32(0.0))

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
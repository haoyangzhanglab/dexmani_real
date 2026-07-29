"""Fixed-size world-frame point cloud from aligned RGB-D — production pipeline.

Driver-precomputed rays deprojection -> camera-frame depth gate
-> world transform -> single workspace crop -> 5 mm voxel -> radius outlier removal
-> fixed-size sample.

Runs inside the CameraProcess child at 30 Hz; the same (num_points, 6) float32
output is recorded to HDF5 (/pointcloud) and consumed by the policy loop, so
training data and deployment observations are byte-identical by construction.

Module-level imports are numpy-only (open3d / torch imported lazily inside
methods, same pattern as pointcloud_utils._voxel_down_sample_o3d) so importing
the config in the parent process stays lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = ["PointCloudProcessor", "PointCloudProcessorConfig"]

import numpy as np


@dataclass(frozen=True)
class PointCloudProcessorConfig:
    """Parameters of the depth->pointcloud pipeline (validated 2026-07-15)."""

    num_points: int = 2048
    # Camera-frame z gate: 0.3 m near (min-Z + margin), 1.5 m far (workspace corner)
    # workspace corner under the current cameras.json extrinsics + ~0.15 m margin.
    depth_min_m: float = 0.3
    depth_max_m: float = 1.5
    # World-frame axis-aligned crop, (x_min, y_min, z_min, x_max, y_max, z_max)
    # — same ordering as pointcloud_utils.check_workspace. z_min = 0.005 removes
    # the desk surface itself.
    workspace: tuple[float, float, float, float, float, float] = (0.0, -0.6, 0.005, 0.8, 0.6, 0.8)
    voxel_size: float = 0.005
    # Radius outlier removal: drop points with fewer than min_points neighbours
    # within radius (2× voxel size). O(N) radius search — same speed as the
    # original DBSCAN but without the clustering/bincount overhead.
    radius_outlier_min_points: int = 3
    radius_outlier_radius: float = 0.01
    # The open3d pipeline (voxel → radius → FPS) is O(N); limiting input to
    # a fixed ceiling keeps point-cloud latency predictable regardless of scene
    # complexity. 8 000 is enough for a dense 5 mm grid across the workspace.
    max_open3d_input: int = 8000
    # "o3d": open3d CPU farthest_point_down_sample (~6.6 ms @ 5.3k -> 2048,
    # colors preserved — verified on open3d 0.19). "pytorch3d": GPU
    # sample_farthest_points (~9 ms, B=1 underutilizes the GPU); only safe in the
    # forked CameraProcess child if the parent never initialized CUDA pre-fork.
    fps_backend: Literal["o3d", "pytorch3d"] = "o3d"

    def to_meta_dict(self) -> dict:
        """h5py-safe snapshot for /meta persistence (pc_* prefixed)."""
        return {
            "pc_num_points": int(self.num_points),
            "pc_depth_min_m": float(self.depth_min_m),
            "pc_depth_max_m": float(self.depth_max_m),
            "pc_workspace": [float(v) for v in self.workspace],
            "pc_voxel_size": float(self.voxel_size),
            "pc_radius_outlier_min_points": int(self.radius_outlier_min_points),
            "pc_radius_outlier_radius": float(self.radius_outlier_radius),
            "pc_fps_backend": str(self.fps_backend),
        }


class PointCloudProcessor:
    """Stateless-per-frame processor; extrinsics and RNG precomputed once."""

    _timing_log_every: int = 30

    def __init__(self, T_world_camera: np.ndarray, config: PointCloudProcessorConfig | None = None) -> None:
        T = np.asarray(T_world_camera, dtype=np.float64)
        if T.shape != (4, 4) or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError(f"T_world_camera must be a (4,4) homogeneous transform, got shape {T.shape}.")
        self.config = config or PointCloudProcessorConfig()
        self._R = T[:3, :3]
        self._t = T[:3, 3]
        self._rng = np.random.default_rng()
        # Internal timing accumulators (ms)
        self._t_numpy = 0.0
        self._t_voxel = 0.0
        self._t_radius = 0.0
        self._t_fps = 0.0
        self._t_in_n = 0
        self._t_voxel_n = 0
        self._t_radius_n = 0
        self._t_n = 0

    def process(self, depth_m: np.ndarray, rgb: np.ndarray, rays: np.ndarray) -> np.ndarray | None:
        """(H,W) float32 meters + (H,W,3) uint8 RGB + (H,W,3) float32 unit rays
        -> (num_points, 6) float32 [xyz world-frame, rgb in 0..1], or None if no
        points survive the gates/filters (caller decides the fallback).
        """
        cfg = self.config
        if rays.shape[:2] != depth_m.shape or rgb.shape[:2] != depth_m.shape:
            raise ValueError(f"Shape mismatch: depth {depth_m.shape}, rgb {rgb.shape[:2]}, rays {rays.shape[:2]}.")

        import time as _time

        _tn0 = _time.monotonic()

        # Gate on raw depth BEFORE deprojection — skip ~270k invalid pixels
        # instead of deprojecting all 307k then throwing most away.
        z_flat = depth_m.ravel()
        mask = np.isfinite(z_flat) & (z_flat > cfg.depth_min_m) & (z_flat < cfg.depth_max_m)
        if not np.any(mask):
            return None
        # Deproject only valid pixels (rays precomputed by the driver).
        pts_cam = (rays.reshape(-1, 3)[mask] * z_flat[mask, None]).astype(np.float64)
        # Convert colors only for gate survivors (o3d float64 path).
        cols = rgb.reshape(-1, 3)[mask].astype(np.float64) / 255.0

        # World transform + single workspace crop (no RANSAC in production).
        pts = pts_cam @ self._R.T + self._t
        x_min, y_min, z_min, x_max, y_max, z_max = cfg.workspace
        crop = (
            (pts[:, 0] >= x_min)
            & (pts[:, 0] <= x_max)
            & (pts[:, 1] >= y_min)
            & (pts[:, 1] <= y_max)
            & (pts[:, 2] >= z_min)
            & (pts[:, 2] <= z_max)
        )
        pts = pts[crop]
        if pts.shape[0] == 0:
            return None
        cols = cols[crop]

        # Cap open3d input so the pipeline stays fast regardless of scene
        # complexity.  Uniform random subsample preserves spatial coverage
        # without bias; the subsequent voxel grid + FPS are the definitive
        # quality gates.
        if pts.shape[0] > cfg.max_open3d_input:
            idx = self._rng.choice(pts.shape[0], cfg.max_open3d_input, replace=False)
            pts = pts[idx]
            cols = cols[idx]

        import open3d as o3d  # lazy: keep parent-process imports light

        n_in = pts.shape[0]

        _t0 = _time.monotonic()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(cols)
        pcd = pcd.voxel_down_sample(voxel_size=cfg.voxel_size)
        if len(pcd.points) == 0:
            return None
        n_voxel = len(pcd.points)

        _t1 = _time.monotonic()

        # Radius outlier removal: drop specks (0-1 neighbours in the 5 mm grid).
        pcd, _ = pcd.remove_radius_outlier(nb_points=cfg.radius_outlier_min_points, radius=cfg.radius_outlier_radius)
        if len(pcd.points) == 0:
            return None
        n_radius = len(pcd.points)

        _t2 = _time.monotonic()

        # Fixed-size output: FPS when enough points, random duplicate pad otherwise.
        n = len(pcd.points)
        if n >= cfg.num_points:
            if cfg.fps_backend == "pytorch3d":
                pts_out, cols_out = self._fps_pytorch3d(pcd)
            else:
                pcd = pcd.farthest_point_down_sample(cfg.num_points)
                pts_out = np.asarray(pcd.points)
                cols_out = np.asarray(pcd.colors)
        else:
            pad = self._rng.integers(0, n, cfg.num_points - n)
            idx = np.concatenate([np.arange(n), pad])
            pts_out = np.asarray(pcd.points)[idx]
            cols_out = np.asarray(pcd.colors)[idx]

        _t3 = _time.monotonic()

        # Periodic timing log
        self._t_numpy += (_t0 - _tn0) * 1000
        self._t_voxel += (_t1 - _t0) * 1000
        self._t_radius += (_t2 - _t1) * 1000
        self._t_fps += (_t3 - _t2) * 1000
        self._t_in_n += n_in
        self._t_voxel_n += n_voxel
        self._t_radius_n += n_radius
        self._t_n += 1
        if self._t_n >= self._timing_log_every:
            from dexmani_real.utils.log import get_logger

            _log = get_logger(__name__)
            _log.debug(
                "PointCloudProcessor [%d frames]: numpy=%.1fms in=%.0fk pts "
                "voxel=%.1fms(%.0fk→%.0fk) radius_outlier=%.1fms(%.0fk→%.0fk) "
                "fps=%.1fms(%.0fk→%d)",
                self._t_n,
                self._t_numpy / self._t_n,
                self._t_in_n / self._t_n / 1000,
                self._t_voxel / self._t_n,
                self._t_in_n / self._t_n / 1000,
                self._t_voxel_n / self._t_n / 1000,
                self._t_radius / self._t_n,
                self._t_voxel_n / self._t_n / 1000,
                self._t_radius_n / self._t_n / 1000,
                self._t_fps / self._t_n,
                self._t_radius_n / self._t_n / 1000,
                cfg.num_points,
            )
            self._t_numpy = self._t_voxel = self._t_radius = self._t_fps = 0.0
            self._t_in_n = self._t_voxel_n = self._t_radius_n = 0
            self._t_n = 0

        return np.ascontiguousarray(np.concatenate([pts_out, cols_out], axis=1), dtype=np.float32)

    def _fps_pytorch3d(self, pcd) -> tuple[np.ndarray, np.ndarray]:
        import torch  # lazy: only when the pytorch3d backend is selected
        from pytorch3d.ops import sample_farthest_points

        pts = np.asarray(pcd.points, dtype=np.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pts_t = torch.from_numpy(pts)[None].to(device)
        _, idx_t = sample_farthest_points(pts_t, K=self.config.num_points)
        idx = idx_t[0].cpu().numpy()
        return pts[idx].astype(np.float64), np.asarray(pcd.colors)[idx]

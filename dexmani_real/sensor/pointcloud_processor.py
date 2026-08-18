"""Fixed-size world-frame point cloud from aligned RGB-D.

Driver-precomputed rays deprojection → camera-frame depth gate → depth edge
filter (LoG gradient) → speckle filter → world transform → desk-plane removal
→ workspace crop → 5 mm voxel → DBSCAN two-in-one outlier filter (noise-point
removal + small-cluster cull) → deterministic FPS or ordered padding.

Used by the offline processing pipeline to derive a fixed-size view from the
Real v17 depth sidecar. Runtime recording stores depth and point-cloud quality
metadata; it does not write a ``/pointcloud`` dataset.

Module-level imports are numpy-only (open3d / torch imported lazily inside
methods, same pattern as pointcloud_utils._voxel_down_sample_o3d) so importing
the config in the parent process stays lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

__all__ = ["PointCloudProcessor", "PointCloudProcessorConfig"]

import numpy as np


@dataclass(frozen=True)
class PointCloudProcessorConfig:
    """Parameters of the depth-to-point-cloud pipeline."""

    num_points: int = 2048
    # Camera-frame depth gate.
    depth_min_m: float = 0.3
    depth_max_m: float = 1.5
    # World-frame crop: (x_min, y_min, z_min, x_max, y_max, z_max).
    workspace: tuple[float, float, float, float, float, float] = (0.25, -0.6, 0.005, 0.85, 0.6, 0.8)
    voxel_size: float = 0.005
    # Optional radius outlier filter; 0 disables it.
    radius_outlier_min_points: int = 0
    radius_outlier_radius: float = 0.01
    # Optional statistical outlier filter; 0 neighbours disables it.
    stat_outlier_nb_neighbors: int = 0
    stat_outlier_std_ratio: float = 1.0
    # DBSCAN noise and small-cluster filter; size 0 disables cluster culling.
    dbscan_eps: float = 0.015
    dbscan_min_points: int = 5
    dbscan_min_cluster_size: int = 25
    # Farthest-point sampling backend.
    fps_backend: Literal["o3d", "pytorch3d"] = "o3d"
    # 0 disables the dense-cloud strided pre-pass.
    hybrid_fps_threshold: int = 0
    # Depth discontinuity filter; threshold <= 0 disables it.
    depth_edge_threshold_m: float = 0.030
    depth_edge_dilate_px: int = 1
    # Relative component of the depth-edge threshold; 0 disables it.
    depth_edge_relative_ratio: float = 0.02

    # Optional desk plane: ax+by+cz+d=0 in world frame.
    desk_plane: tuple[float, float, float, float] | None = None
    desk_clearance_m: float = 0.008
    # Auto-loaded when desk_plane is None and the file exists.
    desk_plane_path: str = "dexmani_real/config/desk_plane.json"

    # Apply a 3x3 median filter before deprojection.
    depth_median_enabled: bool = True

    # Remove valid-depth components smaller than this size; 0 disables it.
    speckle_min_pixels: int = 5

    def to_meta_dict(self) -> dict:
        """h5py-safe snapshot for /meta persistence (pc_* prefixed)."""
        d: dict = {
            "pc_num_points": int(self.num_points),
            "pc_depth_min_m": float(self.depth_min_m),
            "pc_depth_max_m": float(self.depth_max_m),
            "pc_workspace": [float(v) for v in self.workspace],
            "pc_voxel_size": float(self.voxel_size),
            "pc_radius_outlier_min_points": int(self.radius_outlier_min_points),
            "pc_radius_outlier_radius": float(self.radius_outlier_radius),
            "pc_stat_outlier_nb_neighbors": int(self.stat_outlier_nb_neighbors),
            "pc_stat_outlier_std_ratio": float(self.stat_outlier_std_ratio),
            "pc_dbscan_eps": float(self.dbscan_eps),
            "pc_dbscan_min_points": int(self.dbscan_min_points),
            "pc_dbscan_min_cluster_size": int(self.dbscan_min_cluster_size),
            "pc_fps_backend": str(self.fps_backend),
            "pc_hybrid_fps_threshold": int(self.hybrid_fps_threshold),
            "pc_depth_edge_threshold_m": float(self.depth_edge_threshold_m),
            "pc_depth_edge_dilate_px": int(self.depth_edge_dilate_px),
            "pc_depth_edge_relative_ratio": float(self.depth_edge_relative_ratio),
            "pc_depth_median_enabled": bool(self.depth_median_enabled),
            "pc_speckle_min_pixels": int(self.speckle_min_pixels),
        }
        if self.desk_plane is not None:
            d["pc_desk_plane"] = [float(v) for v in self.desk_plane]
            d["pc_desk_clearance_m"] = float(self.desk_clearance_m)
        d["pc_desk_plane_path"] = str(self.desk_plane_path)
        return d

    @classmethod
    def from_meta_dict(cls, meta: Mapping[str, Any]) -> "PointCloudProcessorConfig":
        """Reconstruct a config from a persisted ``to_meta_dict`` snapshot."""

        def _float(key: str, default: float) -> float:
            value = meta.get(key, default)
            return float(value) if value is not None else default

        def _int(key: str, default: int) -> int:
            return int(meta.get(key, default))

        def _tuple(key: str, default: tuple[float, ...]) -> tuple[float, ...]:
            value = meta.get(key, default)
            return tuple(float(v) for v in value) if value is not None else default

        desk_plane = meta.get("pc_desk_plane")
        if desk_plane is not None:
            desk_plane = tuple(float(v) for v in desk_plane)

        return cls(
            num_points=_int("pc_num_points", 2048),
            depth_min_m=_float("pc_depth_min_m", 0.3),
            depth_max_m=_float("pc_depth_max_m", 1.5),
            workspace=_tuple("pc_workspace", (0.25, -0.6, 0.005, 0.85, 0.6, 0.8)),
            voxel_size=_float("pc_voxel_size", 0.005),
            radius_outlier_min_points=_int("pc_radius_outlier_min_points", 0),
            radius_outlier_radius=_float("pc_radius_outlier_radius", 0.01),
            stat_outlier_nb_neighbors=_int("pc_stat_outlier_nb_neighbors", 0),
            stat_outlier_std_ratio=_float("pc_stat_outlier_std_ratio", 1.0),
            dbscan_eps=_float("pc_dbscan_eps", 0.015),
            dbscan_min_points=_int("pc_dbscan_min_points", 5),
            dbscan_min_cluster_size=_int("pc_dbscan_min_cluster_size", 25),
            fps_backend=str(meta.get("pc_fps_backend", "o3d")),
            hybrid_fps_threshold=_int("pc_hybrid_fps_threshold", 0),
            depth_edge_threshold_m=_float("pc_depth_edge_threshold_m", 0.030),
            depth_edge_dilate_px=_int("pc_depth_edge_dilate_px", 1),
            depth_edge_relative_ratio=_float("pc_depth_edge_relative_ratio", 0.02),
            depth_median_enabled=bool(meta.get("pc_depth_median_enabled", True)),
            speckle_min_pixels=_int("pc_speckle_min_pixels", 5),
            desk_plane=desk_plane,
            desk_clearance_m=_float("pc_desk_clearance_m", 0.008),
            desk_plane_path=str(meta.get("pc_desk_plane_path", "dexmani_real/config/desk_plane.json")),
        )


class PointCloudProcessor:
    """Stateless-per-frame processor; extrinsics precomputed once."""

    _timing_log_every: int = 16

    def __init__(
        self,
        T_world_camera: np.ndarray,
        config: PointCloudProcessorConfig | None = None,
    ) -> None:
        T = np.asarray(T_world_camera, dtype=np.float64)
        if T.shape != (4, 4) or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError(f"T_world_camera must be a (4,4) homogeneous transform, got shape {T.shape}.")
        self.config = config or PointCloudProcessorConfig()
        self._R = T[:3, :3]
        self._t = T[:3, 3]
        # Auto-load desk plane from persisted JSON if not explicitly provided.
        self._desk_plane: tuple[float, float, float, float] | None = self.config.desk_plane
        if self._desk_plane is None and self.config.desk_plane_path:
            self._desk_plane = PointCloudProcessor._try_load_desk_plane(self.config.desk_plane_path)
        # Internal timing accumulators (ms)
        self._t_numpy = 0.0
        self._t_voxel = 0.0
        self._t_radius = 0.0
        self._t_stat = 0.0
        self._t_dbscan = 0.0
        self._t_fps = 0.0
        self._t_in_n = 0
        self._t_voxel_n = 0
        self._t_radius_n = 0
        self._t_n = 0
        self.last_source_point_count = 0
        self.last_valid_depth_ratio = 0.0
        self.last_padding_count = 0

    @staticmethod
    def apply_depth_median(depth_m: np.ndarray, enabled: bool) -> np.ndarray:
        """Apply the 3x3 median filter (L515 salt-and-pepper denoise) to ``depth_m``.

        ``enabled=False`` returns ``depth_m`` unchanged.  ``enabled=True`` zeroes
        invalid pixels (NaN or <= 0) before a single 3x3 medianBlur, then restores
        NaN so they are excluded by the subsequent depth gate (depth_min_m=0.3 > 0).
        Edge-preserving: unlike Gaussian, median does not soften depth edges.
        """
        if not enabled:
            return depth_m
        import cv2

        _invalid = ~(np.isfinite(depth_m) & (depth_m > 0))
        _work = depth_m.copy()
        _work[_invalid] = 0.0
        _work = cv2.medianBlur(_work, 3)
        _work[_invalid] = np.nan
        return _work

    def process(self, depth_m: np.ndarray, rgb: np.ndarray, rays: np.ndarray) -> np.ndarray | None:
        """(H,W) float32 meters + (H,W,3) uint8 RGB + (H,W,3) float32 unit rays
        -> (num_points, 6) float32 [xyz world-frame, rgb in 0..1], or None if no
        points survive the gates/filters (caller decides the fallback).
        """
        cfg = self.config
        if rays.shape[:2] != depth_m.shape or rgb.shape[:2] != depth_m.shape:
            raise ValueError(f"Shape mismatch: depth {depth_m.shape}, rgb {rgb.shape[:2]}, rays {rays.shape[:2]}.")
        self.last_source_point_count = 0
        self.last_valid_depth_ratio = float(np.count_nonzero(np.isfinite(depth_m) & (depth_m > 0)) / depth_m.size)
        self.last_padding_count = 0

        import time as _time

        _tn0 = _time.monotonic()

        # Median filter before the depth gate.
        depth_m = self.apply_depth_median(depth_m, cfg.depth_median_enabled)

        # Gate raw depth before deprojection.
        z_flat = depth_m.ravel()
        mask = np.isfinite(z_flat) & (z_flat > cfg.depth_min_m) & (z_flat < cfg.depth_max_m)
        if not np.any(mask):
            return None

        # Remove pixels near depth discontinuities.
        if cfg.depth_edge_threshold_m > 0:
            import cv2

            # Use a smoothed Laplacian to detect depth edges.
            depth_blur = cv2.GaussianBlur(depth_m, (3, 3), sigmaX=0.8)
            laplacian = cv2.Laplacian(depth_blur, cv2.CV_32F, ksize=3)
            edge_mag = np.abs(laplacian)
            if cfg.depth_edge_relative_ratio > 0:
                _thresh = np.maximum(cfg.depth_edge_threshold_m, depth_m * cfg.depth_edge_relative_ratio)
                edge_2d = edge_mag > _thresh
            else:
                edge_2d = edge_mag > cfg.depth_edge_threshold_m  # NaN > thresh → False
            if cfg.depth_edge_dilate_px > 0:
                k = 2 * cfg.depth_edge_dilate_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (k, k))
                edge_2d = cv2.dilate(edge_2d.astype(np.uint8), kernel).astype(bool)
            mask = mask & ~edge_2d.ravel()

        if not np.any(mask):
            return None

        # Remove small connected components from the valid-depth mask.
        if cfg.speckle_min_pixels > 0:
            import cv2

            _mask_2d = mask.reshape(depth_m.shape).astype(np.uint8)
            _, labels, stats, _ = cv2.connectedComponentsWithStats(_mask_2d, connectivity=8)
            # label 0 is the background (invalid pixels) — skip it.
            for _label in range(1, len(stats)):
                if stats[_label, cv2.CC_STAT_AREA] < cfg.speckle_min_pixels:
                    _mask_2d[labels == _label] = 0
            mask = _mask_2d.ravel().astype(bool)
            if not np.any(mask):
                return None

        # Deproject only valid pixels (rays precomputed by the driver).
        pts_cam = (rays.reshape(-1, 3)[mask] * z_flat[mask, None]).astype(np.float64)
        # Convert colors only for gate survivors (o3d float64 path).
        cols = rgb.reshape(-1, 3)[mask].astype(np.float64) / 255.0

        # Transform to world coordinates and apply the workspace crop.
        pts = pts_cam @ self._R.T + self._t

        # Desk-plane removal.
        # Pre-calibrated plane in world frame: points whose signed distance
        # to the desk is below desk_clearance_m are removed.  Auto-loaded
        # from desk_plane_path at init when not explicitly configured.
        # When no plane is available, falls back to workspace z_min.
        if self._desk_plane is not None:
            a, b, c, d_plane = self._desk_plane
            _dist = a * pts[:, 0] + b * pts[:, 1] + c * pts[:, 2] + d_plane
            _above_desk = _dist >= cfg.desk_clearance_m
            pts = pts[_above_desk]
            if pts.shape[0] == 0:
                return None
            cols = cols[_above_desk]

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

        # Remove DBSCAN noise and clusters below the configured size.
        if cfg.dbscan_min_cluster_size > 0:
            labels = np.asarray(
                pcd.cluster_dbscan(eps=cfg.dbscan_eps, min_points=cfg.dbscan_min_points, print_progress=False)
            )
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            small_labels = set(unique_labels[counts < cfg.dbscan_min_cluster_size])
            keep = np.array([(lbl >= 0 and lbl not in small_labels) for lbl in labels])
            pcd = pcd.select_by_index(np.where(keep)[0])
            if len(pcd.points) == 0:
                return None

        _t_dbscan_end = _time.monotonic()

        # Optional radius outlier removal.
        if cfg.radius_outlier_min_points > 0:
            pcd, _ = pcd.remove_radius_outlier(
                nb_points=cfg.radius_outlier_min_points, radius=cfg.radius_outlier_radius
            )
            if len(pcd.points) == 0:
                return None
        n_radius = len(pcd.points)

        _t2 = _time.monotonic()

        # Optional statistical outlier removal.
        if cfg.stat_outlier_nb_neighbors > 0 and len(pcd.points) > cfg.stat_outlier_nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=cfg.stat_outlier_nb_neighbors, std_ratio=cfg.stat_outlier_std_ratio
            )
            if len(pcd.points) == 0:
                return None

        _t_stat_end = _time.monotonic()

        # Select or deterministically pad to the configured point count.
        n = len(pcd.points)
        self.last_source_point_count = int(n)
        self.last_padding_count = max(0, int(cfg.num_points - n))
        if n >= cfg.num_points:
            if cfg.hybrid_fps_threshold > 0 and n > cfg.hybrid_fps_threshold:
                idx = np.linspace(0, n - 1, cfg.num_points).round().astype(np.int64)
                pts_out = np.asarray(pcd.points)[idx]
                cols_out = np.asarray(pcd.colors)[idx]
            elif cfg.fps_backend == "pytorch3d":
                pts_out, cols_out = self._fps_pytorch3d(pcd)
            else:
                pcd = pcd.farthest_point_down_sample(cfg.num_points)
                pts_out = np.asarray(pcd.points)
                cols_out = np.asarray(pcd.colors)
        else:
            idx = np.resize(np.arange(n, dtype=np.int64), cfg.num_points)
            pts_out = np.asarray(pcd.points)[idx]
            cols_out = np.asarray(pcd.colors)[idx]

        _t3 = _time.monotonic()

        self._t_numpy += (_t0 - _tn0) * 1000
        self._t_voxel += (_t1 - _t0) * 1000
        self._t_dbscan += (_t_dbscan_end - _t1) * 1000
        self._t_radius += (_t2 - _t_dbscan_end) * 1000
        self._t_stat += (_t_stat_end - _t2) * 1000
        self._t_fps += (_t3 - _t_stat_end) * 1000
        self._t_in_n += n_in
        self._t_voxel_n += n_voxel
        self._t_radius_n += n_radius
        self._t_n += 1
        if self._t_n >= self._timing_log_every:
            from dexmani_real.utils.log import get_logger

            _log = get_logger(__name__)
            _log.debug(
                "PointCloudProcessor [%d frames]: numpy=%.1fms in=%.0fk pts "
                "voxel=%.1fms(%.0fk→%.0fk) dbscan=%.1fms radius_outlier=%.1fms(%.0fk→%.0fk) "
                "stat_outlier=%.1fms fps=%.1fms(%.0fk→%d)",
                self._t_n,
                self._t_numpy / self._t_n,
                self._t_in_n / self._t_n / 1000,
                self._t_voxel / self._t_n,
                self._t_in_n / self._t_n / 1000,
                self._t_voxel_n / self._t_n / 1000,
                self._t_dbscan / self._t_n,
                self._t_radius / self._t_n,
                self._t_voxel_n / self._t_n / 1000,
                self._t_radius_n / self._t_n / 1000,
                self._t_stat / self._t_n,
                self._t_fps / self._t_n,
                self._t_radius_n / self._t_n / 1000,
                cfg.num_points,
            )
            self._t_numpy = self._t_voxel = self._t_dbscan = self._t_radius = self._t_stat = self._t_fps = 0.0
            self._t_in_n = self._t_voxel_n = self._t_radius_n = 0
            self._t_n = 0

        return np.ascontiguousarray(np.concatenate([pts_out, cols_out], axis=1), dtype=np.float32)

    @staticmethod
    def calibrate_desk_plane(
        depth_m: np.ndarray,
        rgb: np.ndarray,
        rays: np.ndarray,
        T_world_camera: np.ndarray,
        *,
        depth_min_m: float = 0.3,
        depth_max_m: float = 1.5,
        ransac_distance_threshold: float = 0.01,
        ransac_n: int = 3,
        ransac_iterations: int = 1000,
    ) -> tuple[float, float, float, float]:
        """Fit the desk plane from a single RGB-D frame (one-shot calibration).

        Returns ``(a, b, c, d)`` where ``a*x + b*y + c*z + d = 0`` in the
        world frame, with the normal oriented upward (c > 0).  Pass the
        result to ``PointCloudProcessorConfig.desk_plane``.

        The caller should capture a frame showing an empty desk (no objects).
        """
        import open3d as o3d

        T = np.asarray(T_world_camera, dtype=np.float64)
        if T.shape != (4, 4) or not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0]):
            raise ValueError(f"T_world_camera must be a (4,4) homogeneous transform, got shape {T.shape}.")
        R, t = T[:3, :3], T[:3, 3]

        # Depth gate + deproject (same pattern as process(), desk-only scene).
        z_flat = depth_m.ravel()
        mask = np.isfinite(z_flat) & (z_flat > depth_min_m) & (z_flat < depth_max_m)
        if not np.any(mask):
            raise RuntimeError("No valid depth pixels for desk plane calibration.")

        pts_cam = (rays.reshape(-1, 3)[mask] * z_flat[mask, None]).astype(np.float64)
        cols = rgb.reshape(-1, 3)[mask].astype(np.float64) / 255.0

        # World transform.
        pts = pts_cam @ R.T + t

        # Coarse workspace crop to isolate the desk region.
        # Keep a generous z band — the RANSAC plane fit will find the desk.
        crop = (
            (pts[:, 0] >= -0.2)
            & (pts[:, 0] <= 1.0)
            & (pts[:, 1] >= -0.8)
            & (pts[:, 1] <= 0.8)
            & (pts[:, 2] >= -0.2)
            & (pts[:, 2] <= 0.5)
        )
        pts = pts[crop]
        cols = cols[crop]
        if pts.shape[0] < 100:
            raise RuntimeError(f"Too few desk points ({pts.shape[0]}) for plane fitting.")

        # RANSAC plane fit.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=ransac_distance_threshold,
            ransac_n=ransac_n,
            num_iterations=ransac_iterations,
        )
        a, b, c, d = [float(v) for v in plane_model]

        # Normalize normal to unit length.
        norm = np.sqrt(a * a + b * b + c * c)
        if norm < 1e-12:
            raise RuntimeError("Degenerate desk plane (zero-length normal).")
        a, b, c, d = a / norm, b / norm, c / norm, d / norm

        # Orient normal upward (world Z+).
        if c < 0:
            a, b, c, d = -a, -b, -c, -d

        return (a, b, c, d)

    @staticmethod
    def save_desk_plane(plane: tuple[float, float, float, float], path: str) -> None:
        """Persist a desk plane equation to a JSON file.

        Args:
            plane: ``(a, b, c, d)`` — normalised, upward-pointing normal.
            path: JSON file path (relative paths resolved from the calling
                  process's working directory).
        """
        # Keep the parent-process import path lightweight.
        from dexmani_real.recording.transaction import atomic_json_dump

        a, b, c, d = [float(v) for v in plane]
        atomic_json_dump({"a": a, "b": b, "c": c, "d": d}, path)
        from dexmani_real.utils.log import get_logger

        _log = get_logger(__name__)
        _log.info("Desk plane saved to %s: a=%.4f b=%.4f c=%.4f d=%.4f", path, a, b, c, d)

    @staticmethod
    def load_desk_plane(path: str) -> tuple[float, float, float, float]:
        """Load a desk plane equation from a JSON file.

        Returns ``(a, b, c, d)``.
        Raises ``FileNotFoundError`` if the file doesn't exist,
        ``KeyError`` / ``json.JSONDecodeError`` on malformed content.
        """
        import json

        with open(path) as f:
            data = json.load(f)
        a, b, c, d = float(data["a"]), float(data["b"]), float(data["c"]), float(data["d"])
        # Validate: normal should have unit length and point upward.
        norm = np.sqrt(a * a + b * b + c * c)
        if not np.isclose(norm, 1.0, atol=1e-4):
            from dexmani_real.utils.log import get_logger

            _log = get_logger(__name__)
            _log.warning("Desk plane normal in %s has norm=%.6f (expected 1.0) — re-normalizing", path, norm)
            a, b, c, d = a / norm, b / norm, c / norm, d / norm
        if c < 0:
            from dexmani_real.utils.log import get_logger

            _log = get_logger(__name__)
            _log.warning("Desk plane in %s has downward normal — flipping", path)
            a, b, c, d = -a, -b, -c, -d
        return (a, b, c, d)

    @staticmethod
    def _try_load_desk_plane(rel_path: str) -> tuple[float, float, float, float] | None:
        """Try to load a desk plane from *rel_path* (relative to repo root).

        Returns the plane tuple or ``None`` if the file doesn't exist.
        """
        import os
        from pathlib import Path

        _repo_root = Path(__file__).resolve().parents[2]
        _abs = str(_repo_root / rel_path)
        if not os.path.isfile(_abs):
            return None
        try:
            return PointCloudProcessor.load_desk_plane(_abs)
        except (KeyError, ValueError, OSError) as e:
            from dexmani_real.utils.log import get_logger

            _log = get_logger(__name__)
            _log.warning("Failed to load desk plane from %s: %s — using workspace z_min fallback", _abs, e)
            return None

    def _fps_pytorch3d(self, pcd) -> tuple[np.ndarray, np.ndarray]:
        import torch  # lazy: only when the pytorch3d backend is selected
        from pytorch3d.ops import sample_farthest_points

        pts = np.asarray(pcd.points, dtype=np.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pts_t = torch.from_numpy(pts)[None].to(device)
        _, idx_t = sample_farthest_points(pts_t, K=self.config.num_points)
        idx = idx_t[0].cpu().numpy()
        return pts[idx].astype(np.float64), np.asarray(pcd.colors)[idx]

"""Derive fixed-size world-frame point clouds from recorded RGB-D frames.

OpenCV, Open3D, Torch, and PyTorch3D stay lazy so metadata-only processes do
not pay their import cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from dexmani_real.utils.log import get_logger

__all__ = ["PointCloudProcessor", "PointCloudProcessorConfig"]

logger = get_logger(__name__)


@dataclass(frozen=True)
class PointCloudProcessorConfig:
    """Parameters of the depth-to-point-cloud pipeline."""

    num_points: int = 2048
    depth_min_m: float = 0.3
    depth_max_m: float = 1.5
    workspace: tuple[float, float, float, float, float, float] = (
        0.25,
        -0.6,
        0.005,
        0.85,
        0.6,
        0.8,
    )
    voxel_size: float = 0.005

    radius_outlier_min_points: int = 0
    radius_outlier_radius: float = 0.01
    stat_outlier_nb_neighbors: int = 0
    stat_outlier_std_ratio: float = 1.0
    dbscan_eps: float = 0.015
    dbscan_min_points: int = 5
    dbscan_min_cluster_size: int = 25

    fps_backend: Literal["o3d", "pytorch3d"] = "o3d"
    hybrid_fps_threshold: int = 0

    depth_edge_threshold_m: float = 0.030
    depth_edge_dilate_px: int = 1
    depth_edge_relative_ratio: float = 0.02
    depth_median_enabled: bool = True
    speckle_min_pixels: int = 5

    # ax + by + cz + d = 0 in the world frame.
    desk_plane: tuple[float, float, float, float] | None = None
    desk_clearance_m: float = 0.008
    desk_plane_path: str = "dexmani_real/config/desk_plane.json"

    def to_meta_dict(self) -> dict[str, Any]:
        """Return the ``pc_*`` snapshot persisted with recorded episodes."""
        values = asdict(self)
        if values["desk_plane"] is None:
            del values["desk_plane"]
            del values["desk_clearance_m"]
        return {
            f"pc_{key}": list(value) if isinstance(value, tuple) else value
            for key, value in values.items()
        }

    @classmethod
    def from_meta_dict(cls, meta: Mapping[str, Any]) -> PointCloudProcessorConfig:
        """Reconstruct a config from a persisted ``to_meta_dict`` snapshot."""
        defaults = asdict(cls())
        values = {
            key: meta.get(f"pc_{key}", default) for key, default in defaults.items()
        }
        values["workspace"] = tuple(float(value) for value in values["workspace"])
        desk_plane = meta.get("pc_desk_plane")
        values["desk_plane"] = (
            None if desk_plane is None else tuple(float(value) for value in desk_plane)
        )
        return cls(**values)


class PointCloudProcessor:
    """Stateless-per-frame RGB-D processor with precomputed extrinsics."""

    def __init__(
        self,
        T_world_camera: np.ndarray,
        config: PointCloudProcessorConfig | None = None,
    ) -> None:
        transform = np.asarray(T_world_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.allclose(
            transform[3], (0.0, 0.0, 0.0, 1.0)
        ):
            raise ValueError(
                f"T_world_camera must be a homogeneous (4,4) transform, got {transform.shape}"
            )

        self.config = config or PointCloudProcessorConfig()
        self._R = transform[:3, :3]
        self._t = transform[:3, 3]
        self._workspace_lower = np.asarray(self.config.workspace[:3])
        self._workspace_upper = np.asarray(self.config.workspace[3:])
        self._desk_plane = self.config.desk_plane
        if self._desk_plane is None and self.config.desk_plane_path:
            self._desk_plane = self._try_load_desk_plane(self.config.desk_plane_path)

        self.last_source_point_count = 0
        self.last_valid_depth_ratio = 0.0
        self.last_padding_count = 0

    @property
    def desk_plane(self) -> tuple[float, float, float, float] | None:
        """Effective desk plane (configured or auto-loaded), else None."""
        return self._desk_plane

    @staticmethod
    def apply_depth_median(depth_m: np.ndarray, enabled: bool) -> np.ndarray:
        """Apply an edge-preserving 3x3 median filter to valid depth pixels."""
        if not enabled:
            return depth_m

        import cv2

        invalid = ~np.isfinite(depth_m) | (depth_m <= 0)
        filtered = depth_m.copy()
        filtered[invalid] = 0.0
        filtered = cv2.medianBlur(filtered, 3)
        filtered[invalid] = np.nan
        return filtered

    def process(
        self, depth_m: np.ndarray, rgb: np.ndarray, rays: np.ndarray
    ) -> np.ndarray | None:
        """Convert float32 depth/rays and uint8 RGB into fixed-size float32 XYZRGB."""
        if (
            depth_m.ndim != 2
            or rgb.shape != (*depth_m.shape, 3)
            or rays.shape != (*depth_m.shape, 3)
        ):
            raise ValueError(
                f"Expected depth [H,W], rgb/rays [H,W,3]; got {depth_m.shape}, {rgb.shape}, {rays.shape}"
            )

        self.last_source_point_count = 0
        self.last_padding_count = 0
        valid_depth = np.isfinite(depth_m) & (depth_m > 0)
        self.last_valid_depth_ratio = float(
            np.count_nonzero(valid_depth) / depth_m.size
        )

        depth_m = self.apply_depth_median(depth_m, self.config.depth_median_enabled)
        mask = self._depth_mask(depth_m)
        if not np.any(mask):
            return None

        points, colors = self._deproject_and_crop(depth_m, rgb, rays, mask)
        if not len(points):
            return None

        cloud = self._filter_cloud(points, colors)
        if cloud is None:
            return None
        return self._resize_cloud(cloud)

    def _depth_mask(self, depth_m: np.ndarray) -> np.ndarray:
        cfg = self.config
        mask = (
            np.isfinite(depth_m)
            & (depth_m > cfg.depth_min_m)
            & (depth_m < cfg.depth_max_m)
        )

        if cfg.depth_edge_threshold_m > 0:
            import cv2

            smoothed = cv2.GaussianBlur(depth_m, (3, 3), sigmaX=0.8)
            edge_magnitude = np.abs(cv2.Laplacian(smoothed, cv2.CV_32F, ksize=3))
            if cfg.depth_edge_relative_ratio > 0:
                edges = edge_magnitude > np.maximum(
                    cfg.depth_edge_threshold_m,
                    depth_m * cfg.depth_edge_relative_ratio,
                )
            else:
                edges = edge_magnitude > cfg.depth_edge_threshold_m
            if cfg.depth_edge_dilate_px > 0:
                size = 2 * cfg.depth_edge_dilate_px + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (size, size))
                edges = cv2.dilate(edges.astype(np.uint8), kernel).astype(bool)
            mask &= ~edges

        if cfg.speckle_min_pixels > 0 and np.any(mask):
            import cv2

            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), connectivity=8
            )
            areas = stats[:, cv2.CC_STAT_AREA]
            keep_label = areas >= cfg.speckle_min_pixels
            keep_label[0] = False
            mask = keep_label[labels] if count > 1 else np.zeros_like(mask)

        return mask

    def _deproject_and_crop(
        self,
        depth_m: np.ndarray,
        rgb: np.ndarray,
        rays: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = (rays[mask] * depth_m[mask, None]).astype(np.float64)
        points = points @ self._R.T + self._t
        colors = rgb[mask].astype(np.float64) / 255.0

        if self._desk_plane is not None:
            normal = np.asarray(self._desk_plane[:3])
            above_desk = (
                points @ normal + self._desk_plane[3] >= self.config.desk_clearance_m
            )
            points, colors = points[above_desk], colors[above_desk]

        inside = np.all(
            (points >= self._workspace_lower) & (points <= self._workspace_upper),
            axis=1,
        )
        return points[inside], colors[inside]

    def _filter_cloud(self, points: np.ndarray, colors: np.ndarray):
        import open3d as o3d

        cfg = self.config
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        cloud = cloud.voxel_down_sample(cfg.voxel_size)
        if not cloud.has_points():
            return None

        if cfg.dbscan_min_cluster_size > 0:
            labels = np.asarray(
                cloud.cluster_dbscan(
                    cfg.dbscan_eps, cfg.dbscan_min_points, print_progress=False
                ),
                dtype=np.int32,
            )
            clustered = labels >= 0
            if not np.any(clustered):
                return None
            sizes = np.bincount(labels[clustered])
            keep = clustered & (
                sizes[np.maximum(labels, 0)] >= cfg.dbscan_min_cluster_size
            )
            cloud = cloud.select_by_index(np.flatnonzero(keep))

        if cfg.radius_outlier_min_points > 0 and cloud.has_points():
            cloud, _ = cloud.remove_radius_outlier(
                cfg.radius_outlier_min_points, cfg.radius_outlier_radius
            )

        if (
            cfg.stat_outlier_nb_neighbors > 0
            and len(cloud.points) > cfg.stat_outlier_nb_neighbors
        ):
            cloud, _ = cloud.remove_statistical_outlier(
                cfg.stat_outlier_nb_neighbors,
                cfg.stat_outlier_std_ratio,
            )

        return cloud if cloud.has_points() else None

    def _resize_cloud(self, cloud) -> np.ndarray:
        cfg = self.config
        count = len(cloud.points)
        self.last_source_point_count = count
        self.last_padding_count = max(0, cfg.num_points - count)

        if count < cfg.num_points:
            indices = np.resize(np.arange(count), cfg.num_points)
            points = np.asarray(cloud.points)[indices]
            colors = np.asarray(cloud.colors)[indices]
        elif cfg.hybrid_fps_threshold > 0 and count > cfg.hybrid_fps_threshold:
            indices = np.linspace(0, count - 1, cfg.num_points).round().astype(np.int64)
            points = np.asarray(cloud.points)[indices]
            colors = np.asarray(cloud.colors)[indices]
        elif cfg.fps_backend == "pytorch3d":
            points, colors = self._fps_pytorch3d(cloud)
        else:
            sampled = cloud.farthest_point_down_sample(cfg.num_points)
            points, colors = np.asarray(sampled.points), np.asarray(sampled.colors)

        return np.ascontiguousarray(np.column_stack((points, colors)), dtype=np.float32)

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
        """Fit an upward-facing world-frame desk plane from one empty-desk frame."""
        del rgb  # Kept in the public signature for compatibility with the diagnostic tool.

        import open3d as o3d

        transform = np.asarray(T_world_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.allclose(
            transform[3], (0.0, 0.0, 0.0, 1.0)
        ):
            raise ValueError(
                f"T_world_camera must be a homogeneous (4,4) transform, got {transform.shape}"
            )

        mask = np.isfinite(depth_m) & (depth_m > depth_min_m) & (depth_m < depth_max_m)
        if not np.any(mask):
            raise RuntimeError("No valid depth pixels for desk plane calibration")

        flat_mask = mask.ravel()
        points = (
            rays.reshape(-1, 3)[flat_mask] * depth_m.ravel()[flat_mask, None]
        ).astype(np.float64)
        points = points @ transform[:3, :3].T + transform[:3, 3]
        lower = np.array((-0.2, -0.8, -0.2))
        upper = np.array((1.0, 0.8, 0.5))
        points = points[np.all((points >= lower) & (points <= upper), axis=1)]
        if len(points) < 100:
            raise RuntimeError(f"Too few desk points ({len(points)}) for plane fitting")

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        plane, _ = cloud.segment_plane(
            distance_threshold=ransac_distance_threshold,
            ransac_n=ransac_n,
            num_iterations=ransac_iterations,
        )
        plane = np.asarray(plane, dtype=np.float64)
        norm = np.linalg.norm(plane[:3])
        if norm == 0:
            raise RuntimeError("Desk plane has a zero-length normal")
        plane /= norm
        if plane[2] < 0:
            plane *= -1
        return (float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))

    @staticmethod
    def save_desk_plane(plane: tuple[float, float, float, float], path: str) -> None:
        """Persist a desk plane equation atomically."""
        from dexmani_real.recording.transaction import atomic_json_dump

        a, b, c, d = (float(value) for value in plane)
        atomic_json_dump({"a": a, "b": b, "c": c, "d": d}, path)
        logger.info(
            "Desk plane saved to %s: a=%.4f b=%.4f c=%.4f d=%.4f", path, a, b, c, d
        )

    @staticmethod
    def load_desk_plane(path: str) -> tuple[float, float, float, float]:
        """Load, normalize, and orient a desk plane equation."""
        import json

        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        plane = np.array([data[key] for key in ("a", "b", "c", "d")], dtype=np.float64)
        norm = np.linalg.norm(plane[:3])
        if norm == 0:
            raise ValueError(f"Desk plane in {path} has a zero-length normal")
        plane /= norm
        if plane[2] < 0:
            plane *= -1
        return (float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3]))

    @staticmethod
    def _try_load_desk_plane(path: str) -> tuple[float, float, float, float] | None:
        plane_path = Path(path)
        if not plane_path.is_absolute():
            plane_path = Path(__file__).resolve().parents[2] / plane_path
        if not plane_path.is_file():
            return None
        try:
            return PointCloudProcessor.load_desk_plane(str(plane_path))
        except (KeyError, TypeError, ValueError, OSError):
            logger.warning("Ignoring invalid desk plane %s", plane_path, exc_info=True)
            return None

    def _fps_pytorch3d(self, cloud) -> tuple[np.ndarray, np.ndarray]:
        import torch
        from pytorch3d.ops import sample_farthest_points

        points = np.asarray(cloud.points, dtype=np.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        points_tensor = torch.from_numpy(points)[None].to(device)
        _, indices_tensor = sample_farthest_points(
            points_tensor, K=self.config.num_points
        )
        indices = indices_tensor[0].cpu().numpy()
        return points[indices].astype(np.float64), np.asarray(cloud.colors)[indices]

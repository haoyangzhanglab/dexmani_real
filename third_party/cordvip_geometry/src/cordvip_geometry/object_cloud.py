"""Separate visible model geometry from same-frame, depth-supported object points.

This is a geometric filter, not semantic segmentation or a pose-confidence model.
An unrelated surface within the configured depth tolerance can still pass it.
"""

from dataclasses import dataclass

import numpy as np

from .geometry import validate_pose, validate_rgbd


@dataclass
class ObjectCloudResult:
    """Camera-frame meters and RGB uint8, with image-aligned sampled masks.

    Image arrays retain the input (H, W) shape. With pixel_stride > 1, untested
    pixels are zero/False; they must not be interpreted as a dense segmentation.
    Missing measured depths never become points, including on the model back side.
    """

    model_depth_m: np.ndarray
    model_mask: np.ndarray
    observed_mask: np.ndarray
    observed_points_camera: np.ndarray
    observed_colors: np.ndarray
    diagnostics: dict


class ObjectCloudProjector:
    """Cache one meter-scale mesh and intersect camera rays in its local frame.

    ``project`` requires undistorted pinhole RGB-D aligned in the same camera
    frame, with depth representing camera Z in meters (zero denotes missing).
    The caller supplies ``T_camera_object`` for exactly that RGB-D frame.
    """

    def __init__(self, mesh, *, pixel_stride=2, depth_tolerance_m=0.015, nthreads=1):
        for name, value in (("pixel_stride", pixel_stride), ("nthreads", nthreads)):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        depth_tolerance_m = float(depth_tolerance_m)
        if not np.isfinite(depth_tolerance_m) or depth_tolerance_m <= 0:
            raise ValueError("depth_tolerance_m must be finite and positive")
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if (np.iscomplexobj(vertices) or vertices.ndim != 2 or vertices.shape[1] != 3
                or not len(vertices) or not np.isfinite(vertices).all()):
            raise ValueError("Mesh vertices must be finite real (N, 3) coordinates in meters")
        if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
                or not np.issubdtype(faces.dtype, np.integer)
                or np.any(faces < 0) or np.any(faces >= len(vertices))
                or np.any(faces > np.iinfo(np.uint32).max)):
            raise ValueError("Mesh faces must be valid integer triangle indices")
        with np.errstate(over="ignore", invalid="ignore"):
            vertices = np.asarray(vertices, dtype=np.float32)
        if not np.isfinite(vertices).all():
            raise ValueError("Mesh vertices exceed the finite float32 range")
        triangles = vertices[faces].astype(np.float64)
        crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        if not np.any(np.linalg.norm(crosses, axis=1) > 0):
            raise ValueError("Mesh has no nondegenerate triangles")

        # Open3D is optional and imported only when this explicit feature starts.
        import open3d as o3d

        self.pixel_stride = int(pixel_stride)
        self.depth_tolerance_m = depth_tolerance_m
        self._nthreads = int(nthreads)
        self._tensor = o3d.core.Tensor
        self._scene = o3d.t.geometry.RaycastingScene(nthreads=self._nthreads)
        self._scene.add_triangles(
            self._tensor(vertices), self._tensor(faces.astype(np.uint32)),
        )
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        self._bounds_corners = np.array([
            (x, y, z) for x in (low[0], high[0])
            for y in (low[1], high[1]) for z in (low[2], high[2])
        ], dtype=np.float64)
        self._image_shape = None
        self._intrinsics = None
        self._directions_camera = None

    def _camera_rays(self, shape, intrinsics):
        if self._image_shape != shape or not np.array_equal(self._intrinsics, intrinsics):
            v, u = np.mgrid[0:shape[0]:self.pixel_stride, 0:shape[1]:self.pixel_stride]
            self._directions_camera = np.stack((
                (u - intrinsics[0, 2]) / intrinsics[0, 0],
                (v - intrinsics[1, 2]) / intrinsics[1, 1],
                np.ones_like(u),
            ), axis=-1)
            self._image_shape = shape
            self._intrinsics = intrinsics.copy()
        return self._directions_camera

    def _ray_roi(self, directions, intrinsics, pose):
        """Conservatively bound the mesh projection on the global sampling grid."""
        height, width = directions.shape[:2]
        full = (slice(0, height), slice(0, width))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            corners = self._bounds_corners @ pose[:3, :3].T + pose[:3, 3]
            # A box crossing the camera plane has no finite projected bound.
            # Keep the full raycast there; clipping geometry is not necessary.
            if not np.isfinite(corners).all() or np.any(corners[:, 2] <= 1e-6):
                return full
            pixels = corners[:, :2] / corners[:, 2, None]
            pixels *= (intrinsics[0, 0], intrinsics[1, 1])
            pixels += (intrinsics[0, 2], intrinsics[1, 2])
        if not np.isfinite(pixels).all():
            return full
        # With positive Z, perspective extrema of a convex box occur at its
        # corners. One extra sample guards float32 ray/mesh rounding. Divide
        # before rounding so cropped rays stay anchored to the original grid.
        lower = np.floor(pixels.min(axis=0) / self.pixel_stride) - 1
        upper = np.ceil(pixels.max(axis=0) / self.pixel_stride) + 2
        lower = np.clip(lower, (0, 0), (width, height)).astype(int)
        upper = np.clip(upper, (0, 0), (width, height)).astype(int)
        return slice(lower[1], upper[1]), slice(lower[0], upper[0])

    def project(self, rgb, depth_m, intrinsics, T_camera_object):
        """Select measured points agreeing with the model's first visible surface.

        ``depth_coverage`` is valid measured depth / projected samples;
        ``matched_fraction`` is depth-consistent samples / projected samples.
        The median residual covers all projected samples with measured depth,
        including occluders and background disagreements. Undefined ratios and
        residuals are None, not a claim of perfect agreement or zero error.
        These diagnostics describe geometric consistency, not tracking confidence.
        """
        rgb, depth_m, intrinsics, _ = validate_rgbd(rgb, depth_m, intrinsics)
        pose = validate_pose(T_camera_object)
        directions = self._camera_rays(depth_m.shape, intrinsics)
        roi = self._ray_roi(directions, intrinsics, pose)
        roi_directions = directions[roi]
        rotation, translation = pose[:3, :3], pose[:3, 3]
        rays = np.empty((*roi_directions.shape[:2], 6), dtype=np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            rays[..., :3] = -translation @ rotation
            rays[..., 3:] = roi_directions @ rotation
        if not np.isfinite(rays).all():
            raise ValueError("Camera rays or pose exceed the finite float32 range")
        # Open3D returns the parameter in origin + t_hit * direction. Because
        # camera rays have Z=1 and the transform is rigid, t_hit is camera Z in
        # meters, not Euclidean range. Do not normalize these ray directions.
        hit = np.full(directions.shape[:2], np.inf, dtype=np.float32)
        if rays.size:
            hit[roi] = self._scene.cast_rays(self._tensor(rays), nthreads=self._nthreads)["t_hit"].numpy()
        projected = np.isfinite(hit) & (hit > 0)
        sampled_depth = depth_m[::self.pixel_stride, ::self.pixel_stride]
        valid = projected & (sampled_depth > 0)
        residual = np.zeros_like(hit)
        residual[valid] = sampled_depth[valid] - hit[valid]
        matched = valid & (np.abs(residual) <= self.depth_tolerance_m)

        model_depth = np.zeros_like(depth_m)
        model_depth[::self.pixel_stride, ::self.pixel_stride] = np.where(projected, hit, 0)
        model_mask = model_depth > 0
        observed_mask = np.zeros(depth_m.shape, dtype=bool)
        observed_mask[::self.pixel_stride, ::self.pixel_stride] = matched
        with np.errstate(over="ignore", invalid="ignore"):
            points = (directions[matched] * sampled_depth[matched, None]).astype(np.float32)
        if not np.isfinite(points).all():
            raise ValueError("Observed points exceed the finite float32 range")
        colors = rgb[::self.pixel_stride, ::self.pixel_stride][matched].copy()
        projected_count = int(np.count_nonzero(projected))
        valid_count = int(np.count_nonzero(valid))
        matched_count = int(np.count_nonzero(matched))
        diagnostics = {
            "pixel_stride": self.pixel_stride,
            "depth_tolerance_m": self.depth_tolerance_m,
            "projected_pixel_count": projected_count,
            "valid_depth_pixel_count": valid_count,
            "matched_pixel_count": matched_count,
            "depth_coverage": valid_count / projected_count if projected_count else None,
            "matched_fraction": matched_count / projected_count if projected_count else None,
            "median_abs_depth_error_m": float(np.median(np.abs(residual[valid]))) if valid_count else None,
            "occluded_pixel_count": int(np.count_nonzero(valid & (residual < -self.depth_tolerance_m))),
            "background_pixel_count": int(np.count_nonzero(valid & (residual > self.depth_tolerance_m))),
        }
        return ObjectCloudResult(model_depth, model_mask, observed_mask, points, colors, diagnostics)

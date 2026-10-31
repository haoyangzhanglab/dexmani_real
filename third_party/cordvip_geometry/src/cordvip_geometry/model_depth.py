"""Render sampled model Z for comparison with independently measured scene depth."""

import numpy as np

RAYCAST_THREADS = 1


def camera_rays(shape, intrinsics, pixel_stride):
    """One scene's read-only stride grid; Z=1 preserves camera-depth ray parameters."""
    v, u = np.mgrid[0 : shape[0] : pixel_stride, 0 : shape[1] : pixel_stride]
    directions = np.stack(
        (
            (u - intrinsics[0, 2]) / intrinsics[0, 0],
            (v - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(u),
        ),
        axis=-1,
    )
    directions.setflags(write=False)
    return directions


class ModelDepthRenderer:
    """Cache a metre-scale mesh; render camera Z on the global stride grid, with zero for missing hits."""

    def __init__(self, mesh, *, pixel_stride=2):
        if isinstance(pixel_stride, bool) or not isinstance(pixel_stride, (int, np.integer)) or pixel_stride <= 0:
            raise ValueError("pixel_stride must be a positive integer")
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.faces)
        if (
            np.iscomplexobj(vertices)
            or vertices.ndim != 2
            or vertices.shape[1] != 3
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("Mesh vertices must be finite real (N, 3) coordinates in meters")
        if (
            faces.ndim != 2
            or faces.shape[1] != 3
            or not len(faces)
            or not np.issubdtype(faces.dtype, np.integer)
            or np.any(faces < 0)
            or np.any(faces >= len(vertices))
            or np.any(faces > np.iinfo(np.uint32).max)
        ):
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
        self._tensor = o3d.core.Tensor
        self._scene = o3d.t.geometry.RaycastingScene(nthreads=RAYCAST_THREADS)
        self._scene.add_triangles(
            self._tensor(vertices),
            self._tensor(faces.astype(np.uint32)),
        )
        low, high = vertices.min(axis=0), vertices.max(axis=0)
        self._bounds_corners = np.array(
            [
                (x, y, z)
                for x in (low[0], high[0])
                for y in (low[1], high[1])
                for z in (low[2], high[2])
            ],
            dtype=np.float64,
        )
    def ray_roi(self, directions, intrinsics, pose):
        """Conservatively bound the mesh projection on the global sampling grid."""
        height, width = directions.shape[:2]
        full = (slice(0, height), slice(0, width))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            corners = self._bounds_corners @ pose[:3, :3].T + pose[:3, 3]
            # Boxes crossing the camera plane require the full raycast; their projected bound is unbounded.
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

    def render(self, image_shape, intrinsics, T_camera_object, *, directions):
        """Render the first positive model intersection for this frame's camera/pose."""
        pose = T_camera_object
        roi = self.ray_roi(directions, intrinsics, pose)
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
            hit[roi] = self._scene.cast_rays(self._tensor(rays), nthreads=RAYCAST_THREADS)[
                "t_hit"
            ].numpy()
        projected = np.isfinite(hit) & (hit > 0)
        model_depth = np.zeros(image_shape, dtype=np.float32)
        model_depth[:: self.pixel_stride, :: self.pixel_stride] = np.where(projected, hit, 0)
        return model_depth

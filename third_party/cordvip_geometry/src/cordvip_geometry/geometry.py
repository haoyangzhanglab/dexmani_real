"""Metric geometry shared by the standalone object preparation and tracking demo."""

from pathlib import Path

import numpy as np


def validate_points(points):
    points = np.asarray(points)
    if np.iscomplexobj(points):
        raise ValueError("Point coordinates must be real")
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite (N, 3) array")
    return points


def float32_points(points):
    with np.errstate(over="ignore", invalid="ignore"):
        points = validate_points(points).astype(np.float32)
    if not np.isfinite(points).all():
        raise ValueError("Point coordinates exceed the finite float32 range")
    return points


def validate_normals(value, points):
    normals = float32_points(value)
    if normals.shape != points.shape or not np.allclose(
        np.linalg.norm(normals, axis=1), 1.0, rtol=0, atol=1e-4
    ):
        raise ValueError("Normals must be unit vectors corresponding to every model point")
    return normals


def project_points(points, K):
    points = np.asarray(points)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-5)
    pixels = np.full((len(points), 2), np.nan)
    projected = points[valid] @ np.asarray(K).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def validate_pose(pose):
    """Validate an SE(3) transform without projecting or repairing its rotation."""
    if np.iscomplexobj(pose):
        raise ValueError("Pose must be real")
    pose = np.array(pose, dtype=np.float64, copy=True)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Pose must be a finite (4, 4) array")
    if not np.allclose(pose[3], [0, 0, 0, 1], rtol=0, atol=1e-6):
        raise ValueError("Pose has an invalid homogeneous row")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-4):
        raise ValueError("Pose rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-4):
        raise ValueError("Pose rotation must have determinant +1")
    return pose


def transform_points(points, pose):
    """Apply T_destination_source to row-vector XYZ points in the source frame."""
    points = validate_points(points)
    pose = validate_pose(pose)
    return float32_points(points @ pose[:3, :3].T + pose[:3, 3])


def save_points(path, points):
    """Save a fixed sampling asset without overwriting an existing file."""
    points = float32_points(points)
    if not len(points):
        raise ValueError("A sampling asset must contain at least one point")
    with Path(path).open("xb") as stream:
        np.save(stream, points, allow_pickle=False)


def load_points(path):
    points = float32_points(np.load(path, allow_pickle=False))
    if not len(points):
        raise ValueError("A sampling asset must contain at least one point")
    return points


def load_metric_mesh(path, *, scale_to_meters=1.0):
    """Load one triangle mesh, preserving its origin and scaling all axes equally."""
    scale_to_meters = float(scale_to_meters)
    if not np.isfinite(scale_to_meters) or scale_to_meters <= 0:
        raise ValueError("scale_to_meters must be finite and positive")
    import trimesh

    mesh = trimesh.load(str(Path(path)), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected one triangle mesh; export a merged mesh first")
    vertices = validate_points(mesh.vertices)
    if not len(vertices) or not len(mesh.faces):
        raise ValueError("Mesh must contain vertices and triangular faces")
    if np.asarray(mesh.faces).ndim != 2 or np.asarray(mesh.faces).shape[1] != 3:
        raise ValueError("Mesh faces must be triangles")
    mesh = mesh.copy()
    mesh.apply_scale(scale_to_meters)
    validate_points(mesh.vertices)
    return mesh


def mesh_scale_from_extent(mesh, axis, extent_m):
    """The caller must identify the mesh axis corresponding to the measured size."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    extent_m = float(extent_m)
    if not np.isfinite(extent_m) or extent_m <= 0:
        raise ValueError("extent_m must be finite and positive")
    vertices = validate_points(mesh.vertices)
    if not len(vertices):
        raise ValueError("Mesh contains no vertices")
    original_extent = np.ptp(vertices[:, axis])
    if original_extent <= 0:
        raise ValueError("Mesh has zero extent along the selected axis")
    return extent_m / original_extent


def sample_mesh_surface(mesh, num_points=1024, seed=0):
    """Return fixed surface points and mesh-face normals, not KNN contact normals."""
    if not isinstance(num_points, (int, np.integer)) or num_points <= 0:
        raise ValueError("num_points must be a positive integer")
    vertices = validate_points(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError("Mesh faces must be an integer (F, 3) array")
    if not len(faces) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("Mesh has no faces or contains invalid vertex indices")
    triangles = vertices[faces]
    crosses = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    twice_area = np.linalg.norm(crosses, axis=1)
    total = twice_area.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Mesh has no finite, nonzero surface area")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(faces), num_points, p=twice_area / total)
    selected = triangles[indices]
    # sqrt(u) gives uniform area density within each selected triangle.
    uv = rng.random((num_points, 2))
    root_u = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root_u, root_u * (1 - uv[:, 1]), root_u * uv[:, 1]))
    points = np.sum(selected * weights[:, :, None], axis=1)
    normals = crosses[indices] / twice_area[indices, None]
    return float32_points(points), float32_points(normals)


def validate_intrinsics(intrinsics):
    if np.iscomplexobj(intrinsics):
        raise ValueError("Intrinsics must be real")
    intrinsics = np.array(intrinsics, dtype=np.float64, copy=True)
    if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError("Intrinsics must be a finite (3, 3) matrix")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("Pinhole focal lengths must be positive")
    if not np.allclose(intrinsics[2], [0, 0, 1], rtol=0, atol=1e-9):
        raise ValueError("Intrinsics have an invalid homogeneous row")
    if abs(intrinsics[0, 1]) > 1e-9 or abs(intrinsics[1, 0]) > 1e-9:
        raise ValueError("Only zero-skew pinhole intrinsics are supported")
    return intrinsics


def depth_and_mask(depth_m, mask):
    depth_m = np.asarray(depth_m)
    if np.iscomplexobj(depth_m):
        raise ValueError("Depth must be real")
    if (
        depth_m.ndim != 2
        or not all(depth_m.shape)
        or not np.isfinite(depth_m).all()
        or np.any(depth_m < 0)
    ):
        raise ValueError(
            "Depth must be finite nonnegative (H, W), in meters; zero denotes missing depth"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        depth_m = np.asarray(depth_m, dtype=np.float32)
    if not np.isfinite(depth_m).all():
        raise ValueError("Depth exceeds the finite float32 range")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != depth_m.shape or not np.isfinite(mask).all():
            raise ValueError("Mask must be finite and have the same (H, W) shape as depth")
        mask = mask.astype(bool)
    return depth_m, mask


def validate_rgbd(rgb, depth_m, intrinsics, mask=None):
    depth_m, mask = depth_and_mask(depth_m, mask)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*depth_m.shape, 3):
        raise ValueError("RGB must be uint8 (H, W, 3), aligned with depth")
    return rgb, depth_m, validate_intrinsics(intrinsics), mask


class RectificationMaps:
    """One session's last image geometry; bounded to one pair of OpenCV maps."""

    def __init__(self):
        self.key = self.maps = None

    def get(self, shape, intrinsics, distortion):
        import cv2

        key = (
            tuple(shape),
            np.asarray(intrinsics, dtype=np.float64).tobytes(),
            np.asarray(distortion, dtype=np.float64).tobytes(),
        )
        if key != self.key:
            height, width = shape
            maps = cv2.initUndistortRectifyMap(
                intrinsics,
                distortion,
                None,
                intrinsics,
                (width, height),
                cv2.CV_32FC1,
            )
            self.key, self.maps = key, maps
        return self.maps


def rectify_rgbd(rgb, depth_m, intrinsics, distortion, mask=None, *, maps=None):
    """Rectify forward Brown-Conrady RGB-D; nearest-neighbor depth/mask sampling preserves boundaries."""
    rgb, depth_m, intrinsics, mask = validate_rgbd(rgb, depth_m, intrinsics, mask)
    if np.iscomplexobj(distortion):
        raise ValueError("Distortion coefficients must be real")
    distortion = np.asarray(distortion, dtype=np.float64)
    if distortion.shape != (5,) or not np.isfinite(distortion).all():
        raise ValueError("Brown-Conrady distortion must be a finite five-element vector")
    if not np.any(distortion):
        return rgb.copy(), depth_m.copy(), intrinsics, None if mask is None else mask.copy()
    import cv2

    height, width = depth_m.shape
    if maps is None:
        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsics,
            distortion,
            None,
            intrinsics,
            (width, height),
            cv2.CV_32FC1,
        )
    else:
        map_x, map_y = maps.get((height, width), intrinsics, distortion)
    rgb = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    depth_m = cv2.remap(depth_m, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    if mask is not None:
        mask = cv2.remap(
            mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        ).astype(bool)
    return rgb, depth_m, intrinsics, mask


def deproject_depth(depth_m, intrinsics, mask=None):
    """Return camera-frame XYZ meters, in row-major order of (depth > 0) & mask."""
    depth_m, mask = depth_and_mask(depth_m, mask)
    intrinsics = validate_intrinsics(intrinsics)
    valid = depth_m > 0
    if mask is not None:
        valid &= mask
    v, u = np.nonzero(valid)
    z = depth_m[v, u]
    x = (u - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (v - intrinsics[1, 2]) * z / intrinsics[1, 1]
    return float32_points(np.column_stack((x, y, z)))

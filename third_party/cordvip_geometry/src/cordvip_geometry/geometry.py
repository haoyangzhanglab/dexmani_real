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
    """Convert newly computed geometry, retaining the post-conversion finite check."""
    with np.errstate(over="ignore", invalid="ignore"):
        points = np.array(points, dtype=np.float32, copy=True)
    if not np.isfinite(points).all():
        raise ValueError("Point coordinates exceed the finite float32 range")
    return points


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
    """Transform admitted XYZ/SE(3) inputs; the new float32 result can still overflow."""
    return float32_points(points @ pose[:3, :3].T + pose[:3, 3])


def save_points(path, points):
    """Save a fixed sampling asset without overwriting an existing file."""
    points = float32_points(validate_points(points))
    if not len(points):
        raise ValueError("A sampling asset must contain at least one point")
    with Path(path).open("xb") as stream:
        np.save(stream, points, allow_pickle=False)


def load_points(path):
    points = float32_points(validate_points(np.load(path, allow_pickle=False)))
    if not len(points):
        raise ValueError("A sampling asset must contain at least one point")
    return points


def load_metric_mesh(path):
    """Load one triangle mesh in its stored units, preserving origin and orientation."""
    import trimesh

    mesh = trimesh.load(str(Path(path)), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("Expected one triangle mesh; export a merged mesh first")
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Mesh must contain vertices and triangular faces")
    if np.asarray(mesh.faces).ndim != 2 or np.asarray(mesh.faces).shape[1] != 3:
        raise ValueError("Mesh faces must be triangles")
    validate_points(mesh.vertices)
    return mesh


def mesh_scale_from_extent(mesh, axis, extent_m):
    """The caller must identify the mesh axis corresponding to the measured size."""
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1 or 2")
    extent_m = float(extent_m)
    if not np.isfinite(extent_m) or extent_m <= 0:
        raise ValueError("extent_m must be finite and positive")
    vertices = np.asarray(mesh.vertices)
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


def validate_rgbd(rgb, depth_m, intrinsics):
    """Admit external RGB-D and own the camera matrix shared by downstream geometry."""
    depth_m = np.asarray(depth_m)
    if np.iscomplexobj(depth_m):
        raise ValueError("Depth must be real")
    if (
        depth_m.ndim != 2
        or not all(depth_m.shape)
        or np.any(depth_m < 0)
    ):
        raise ValueError(
            "Depth must be finite nonnegative (H, W), in meters; zero denotes missing depth"
        )
    with np.errstate(over="ignore", invalid="ignore"):
        depth_m = np.asarray(depth_m, dtype=np.float32)
    if not np.isfinite(depth_m).all():
        raise ValueError("Depth exceeds the finite float32 range")
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8 or rgb.shape != (*depth_m.shape, 3):
        raise ValueError("RGB must be uint8 (H, W, 3), aligned with depth")
    return rgb, depth_m, validate_intrinsics(intrinsics)


def validate_intrinsics(intrinsics):
    """Own one finite pinhole matrix at the input boundary."""
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
    intrinsics.setflags(write=False)
    return intrinsics


def validate_distortion(model, coefficients):
    """Admit the recorded forward distortion model once per camera."""
    model = model.removeprefix("distortion.")
    if model not in {"none", "brown_conrady"}:
        raise ValueError(f"Unsupported distortion model: {model}")
    if np.iscomplexobj(coefficients):
        raise ValueError("Distortion coefficients must be real")
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError("Brown-Conrady distortion must be a finite five-element vector")
    if model == "none" and np.any(coefficients != 0):
        raise ValueError("none distortion requires zero coefficients")
    return tuple(coefficients)


class RectificationMaps:
    """One admitted fixed camera's maps; no per-frame camera cache or stream switching."""

    def __init__(self, shape, intrinsics, distortion):
        self.maps = None
        if np.any(distortion):
            import cv2

            height, width = shape
            self.maps = cv2.initUndistortRectifyMap(
                intrinsics,
                np.asarray(distortion),
                None,
                intrinsics,
                (width, height),
                cv2.CV_32FC1,
            )

    def apply(self, rgb, depth_m, mask=None):
        """Rectify arrays; a separately supplied selection mask remains an input boundary."""
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != depth_m.shape or not np.isfinite(mask).all():
                raise ValueError("Mask must be finite and have the same (H, W) shape as depth")
            mask = mask.astype(bool)
        if self.maps is None:
            return rgb.copy(), depth_m.copy(), None if mask is None else mask.copy()
        import cv2

        map_x, map_y = self.maps
        rgb = cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        depth_m = cv2.remap(depth_m, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        if mask is not None:
            mask = cv2.remap(
                mask.astype(np.uint8), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
            ).astype(bool)
        return rgb, depth_m, mask

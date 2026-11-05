"""网格采样、刚体变换与相机几何。"""

from pathlib import Path

import numpy as np

MAX_MESH_FACES = 20_000


def simplify_mesh(mesh):
    if len(mesh.faces) <= MAX_MESH_FACES:
        return mesh
    import open3d as o3d
    import trimesh

    visual = mesh.visual.to_color() if mesh.visual.kind == "texture" else mesh.visual
    reduced = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(mesh.vertices),
        o3d.utility.Vector3iVector(mesh.faces),
    )
    reduced.vertex_colors = o3d.utility.Vector3dVector(
        visual.vertex_colors[:, :3] / 255.0
    )
    reduced = reduced.simplify_quadric_decimation(MAX_MESH_FACES)
    return trimesh.Trimesh(
        vertices=np.asarray(reduced.vertices),
        faces=np.asarray(reduced.triangles),
        vertex_colors=np.rint(np.asarray(reduced.vertex_colors) * 255).astype(np.uint8),
        process=False,
    )


def validate_points(points):
    points = np.asarray(points)
    if np.iscomplexobj(points):
        raise ValueError("Point coordinates must be real")
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Points must be a finite (N, 3) array")
    return points


def float32_points(points):
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
    """检查 SE(3)，不投影或修补旋转。"""
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
    return (points @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


def save_points(path, points):
    with Path(path).open("xb") as stream:
        np.save(stream, points, allow_pickle=False)


def load_points(path):
    points = float32_points(validate_points(np.load(path, allow_pickle=False)))
    if not len(points):
        raise ValueError("A sampling asset must contain at least one point")
    return points


def load_mesh(path):
    """按文件单位加载三角网格，保留原点和朝向。"""
    import trimesh

    mesh = trimesh.load(str(Path(path)), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Expected one triangle mesh; export a merged mesh first")
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Mesh must contain vertices and triangular faces")
    if np.asarray(mesh.faces).ndim != 2 or np.asarray(mesh.faces).shape[1] != 3:
        raise ValueError("Mesh faces must be triangles")
    validate_points(mesh.vertices)
    if np.any(mesh.faces < 0) or np.any(mesh.faces >= len(mesh.vertices)):
        raise ValueError("Mesh has invalid vertex indices")
    return mesh


def mesh_scale_from_extent(mesh, axis, extent_m):
    vertices = np.asarray(mesh.vertices)
    original_extent = float(np.ptp(vertices[:, axis]))
    if original_extent <= 0:
        raise ValueError("Mesh has zero extent along the selected axis")
    return extent_m / original_extent


def sample_mesh_surface(mesh, num_points=1024, seed=0):
    """采样固定表面点与三角面法向，法向不代表真实接触。"""
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    twice_area = np.linalg.norm(crosses, axis=1)
    total = twice_area.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Mesh has no finite, nonzero surface area")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(faces), num_points, p=twice_area / total)
    selected = triangles[indices]
    # sqrt(u) 保证三角面内均匀采样。
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


def validate_distortion(model, coefficients):
    model = model.removeprefix("distortion.")
    if model not in {"none", "brown_conrady"}:
        raise ValueError(f"Unsupported distortion model: {model}")
    if np.iscomplexobj(coefficients):
        raise ValueError("Distortion coefficients must be real")
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError(
            "Brown-Conrady distortion must be a finite five-element vector"
        )
    if model == "none" and np.any(coefficients != 0):
        raise ValueError("none distortion requires zero coefficients")
    return tuple(coefficients)


class RectificationMaps:
    """固定相机的去畸变映射，初始化一次。"""

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

    def apply(self, rgb, depth_m):
        if self.maps is None:
            return rgb, depth_m
        import cv2

        map_x, map_y = self.maps
        rgb = cv2.remap(
            rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        depth_m = cv2.remap(
            depth_m, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
        return rgb, depth_m

    def apply_mask(self, mask):
        if self.maps is None:
            return mask
        import cv2

        map_x, map_y = self.maps
        return cv2.remap(
            mask.astype(np.uint8),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        ).astype(bool)

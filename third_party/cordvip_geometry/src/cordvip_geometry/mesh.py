"""网格读取、尺度调整、表面采样与射线深度。"""

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


def load_mesh(path):
    """按文件单位加载三角网格，保留原点和朝向。"""
    import trimesh

    mesh = trimesh.load(str(path), process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Expected one triangle mesh; export a merged mesh first")
    if not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("Mesh must contain vertices and triangular faces")
    if mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3:
        raise ValueError("Mesh faces must be triangles")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Mesh contains nonfinite vertices")
    if np.any(mesh.faces < 0) or np.any(mesh.faces >= len(mesh.vertices)):
        raise ValueError("Mesh has invalid vertex indices")
    return mesh


def sample_mesh_surface(mesh, num_points=1024, seed=0):
    """按面积采样固定表面 XYZ。"""
    areas = mesh.area_faces
    total = areas.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Mesh has no finite, nonzero surface area")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(mesh.faces), num_points, p=areas / total)
    triangles = mesh.vertices[mesh.faces[indices]]
    # sqrt(u) 保证三角面内均匀采样。
    uv = rng.random((num_points, 2))
    root_u = np.sqrt(uv[:, 0])
    weights = np.column_stack((1 - root_u, root_u * (1 - uv[:, 1]), root_u * uv[:, 1]))
    points = np.sum(triangles * weights[:, :, None], axis=1)
    return points.astype(np.float32)


def scale_mesh(mesh, *, scale_to_meters=None, size_axis=None, size_meters=None):
    """按实测尺度缩放副本，保留物体原点和朝向。"""
    if (size_axis is None) != (size_meters is None):
        raise ValueError("size_axis 与 size_meters 必须同时提供")
    if (scale_to_meters is None) == (size_axis is None):
        raise ValueError("提供缩放因子或实测轴尺寸，二选一")
    if size_axis is not None:
        if not np.isfinite(size_meters) or size_meters <= 0:
            raise ValueError("实测尺寸必须为有限正数")
        extent = float(np.ptp(mesh.vertices[:, "xyz".index(size_axis)]))
        if extent <= 0:
            raise ValueError("Mesh has zero extent along the selected axis")
        scale_to_meters = size_meters / extent
    if not np.isfinite(scale_to_meters) or scale_to_meters <= 0:
        raise ValueError("缩放因子必须为有限正数")
    result = mesh.copy()
    result.apply_scale(scale_to_meters)
    return result


class MeshRaycaster:
    def __init__(self, mesh):
        import open3d as o3d

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        self._tensor = o3d.core.Tensor
        self._scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self._scene.add_triangles(
            self._tensor(vertices),
            self._tensor(np.asarray(mesh.faces, dtype=np.uint32)),
        )

    def render(self, directions, pose):
        rotation, translation = pose[:3, :3], pose[:3, 3]
        rays = np.empty((len(directions), 6), dtype=np.float32)
        rays[:, :3] = -translation @ rotation
        rays[:, 3:] = directions @ rotation
        # 不归一化方向；刚体变换后的 t_hit 仍是相机 Z，而非欧氏距离。
        hits = self._scene.cast_rays(self._tensor(rays), nthreads=1)["t_hit"].numpy()
        return np.where(np.isfinite(hits) & (hits > 0), hits, 0)

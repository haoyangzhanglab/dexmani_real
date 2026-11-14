"""网格读取、米制尺度调整与射线深度。"""

import logging

import numpy as np

MAX_MESH_FACES = 20_000
logger = logging.getLogger(__name__)


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
    reduced.vertex_colors = o3d.utility.Vector3dVector(visual.vertex_colors[:, :3] / 255.0)
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
    areas = mesh.area_faces
    total = areas.sum()
    if not np.isfinite(areas).all() or not np.isfinite(total) or total <= 0:
        raise ValueError(f"Mesh has no finite, nonzero surface area: {path}")
    degenerate = int(np.count_nonzero(~mesh.nondegenerate_faces()))
    if degenerate:
        # 少量退化面不影响剩余表面；不在加载时改写资产或改变三角面顺序。
        logger.warning("%s: %d/%d degenerate faces", path, degenerate, len(mesh.faces))
    return mesh


def validate_scale(*, scale_to_meters=None, size_axis=None, size_meters=None):
    """检查缩放参数；任务入口在加载模型前复用此检查。"""
    if (size_axis is None) != (size_meters is None):
        raise ValueError("size_axis 与 size_meters 必须同时提供")
    if (scale_to_meters is None) == (size_axis is None):
        raise ValueError("必须提供缩放因子或完整实测尺寸，二选一")
    if size_axis is not None and size_axis not in ("x", "y", "z"):
        raise ValueError("实测轴必须为 x、y 或 z")
    if any(
        value is not None and (not np.isfinite(value) or value <= 0)
        for value in (scale_to_meters, size_meters)
    ):
        raise ValueError("尺度必须为有限正数")


def scale_mesh(mesh, *, scale_to_meters=None, size_axis=None, size_meters=None):
    """按实测尺度缩放副本，保留物体原点和朝向。"""
    validate_scale(
        scale_to_meters=scale_to_meters,
        size_axis=size_axis,
        size_meters=size_meters,
    )
    if size_axis is not None:
        extent = float(np.ptp(mesh.vertices[:, "xyz".index(size_axis)]))
        if not np.isfinite(extent) or extent <= 0:
            raise ValueError("Mesh has no finite, positive extent along the selected axis")
        scale_to_meters = size_meters / extent
    result = mesh.copy()
    result.apply_scale(scale_to_meters)
    if not np.isfinite(result.vertices).all() or not np.isfinite(result.area) or result.area <= 0:
        raise ValueError("Scaled mesh has no finite, nonzero surface area")
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

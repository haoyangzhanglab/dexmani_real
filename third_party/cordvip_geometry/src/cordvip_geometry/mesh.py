"""网格读取、米制尺度调整与射线深度。"""

import logging
from itertools import product
from numbers import Real

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
    simplified = trimesh.Trimesh(
        vertices=np.asarray(reduced.vertices),
        faces=np.asarray(reduced.triangles),
        vertex_colors=np.rint(np.asarray(reduced.vertex_colors) * 255).astype(np.uint8),
        process=False,
    )
    if not 0 < simplified.area < np.inf:
        raise ValueError("Mesh simplification produced no finite, nonzero surface area")
    return simplified


def load_mesh(path, *, simplify=False):
    """按文件单位加载三角网格，可选简化并检查实际使用的表面。"""
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
    if not 0 < mesh.area < np.inf:
        raise ValueError(f"Mesh has no finite, nonzero surface area: {path}")
    if simplify:
        try:
            mesh = simplify_mesh(mesh)
        except ValueError as error:
            raise ValueError(f"{path}: {error}") from error
    degenerate = int(np.count_nonzero(~mesh.nondegenerate_faces()))
    if degenerate:
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
        value is not None and (
            isinstance(value, bool) or not isinstance(value, Real) or not 0 < float(value) < np.inf
        )
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
    if not np.isfinite(result.vertices).all() or not 0 < result.area < np.inf:
        raise ValueError("Scaled mesh has no finite, nonzero surface area")
    return result


class MeshRaycaster:
    def __init__(self, mesh):
        import open3d as o3d

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        self._bounds_corners = np.array(list(product(*mesh.bounds.T)))
        self._tensor = o3d.core.Tensor
        self._scene = o3d.t.geometry.RaycastingScene(nthreads=1)
        self._scene.add_triangles(
            self._tensor(vertices),
            self._tensor(np.asarray(mesh.faces, dtype=np.uint32)),
        )

    def render(self, directions, pose):
        """返回射线网格对应的相机深度。"""
        rotation, translation = pose[:3, :3], pose[:3, 3]
        rays = np.empty((*directions.shape[:-1], 6), dtype=np.float32)
        rays[..., :3] = -translation @ rotation
        rays[..., 3:] = directions @ rotation
        # 不归一化方向；刚体变换后的 t_hit 仍是相机 Z，而非欧氏距离。
        hits = self._scene.cast_rays(self._tensor(rays), nthreads=1)["t_hit"].numpy()
        return np.where(np.isfinite(hits) & (hits > 0), hits, 0)

    def render_roi(self, directions, pose, intrinsics, pixel_stride, margin_pixels):
        """仅渲染 CAD 投影覆盖的射线，返回完整采样网格。"""
        rotation, translation = pose[:3, :3], pose[:3, 3]
        # 求逆与 render 的射线变换对应，包括带有浮点误差的旋转矩阵。
        corners = np.linalg.solve(rotation.T, self._bounds_corners.T).T + translation
        x, y, z = corners.T
        shape = directions.shape[:2]
        behind = z < 0.001
        if behind.any():
            if behind.all():
                return np.zeros(shape, dtype=np.float32)
            return self.render(directions, pose)

        u = intrinsics[0, 0] * x / z + intrinsics[0, 2]
        v = intrinsics[1, 1] * y / z + intrinsics[1, 2]
        col_start = max(0, int(np.ceil((u.min() - margin_pixels) / pixel_stride)))
        col_stop = min(
            shape[1], int(np.floor((u.max() + margin_pixels) / pixel_stride)) + 1
        )
        row_start = max(0, int(np.ceil((v.min() - margin_pixels) / pixel_stride)))
        row_stop = min(
            shape[0], int(np.floor((v.max() + margin_pixels) / pixel_stride)) + 1
        )
        rendered = np.zeros(shape, dtype=np.float32)
        if row_start < row_stop and col_start < col_stop:
            roi = np.s_[row_start:row_stop, col_start:col_stop]
            rendered[roi] = self.render(directions[roi], pose)
        return rendered

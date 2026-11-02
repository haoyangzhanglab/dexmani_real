"""在语义实测像素上查询模型深度。"""

import numpy as np

RAYCAST_THREADS = 1


def camera_rays(shape, intrinsics, pixel_stride):
    """固定采样网格；Z=1 使射线参数等于相机深度。"""
    v, u = np.mgrid[0 : shape[0] : pixel_stride, 0 : shape[1] : pixel_stride]
    directions = np.stack(
        (
            (u - intrinsics[0, 2]) / intrinsics[0, 0],
            (v - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(u),
        ),
        axis=-1,
    )
    return directions


class ModelDepthRenderer:
    def __init__(self, mesh, *, pixel_stride=2):
        import open3d as o3d

        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        self.pixel_stride = pixel_stride
        self._tensor = o3d.core.Tensor
        self._scene = o3d.t.geometry.RaycastingScene(nthreads=RAYCAST_THREADS)
        self._scene.add_triangles(
            self._tensor(vertices),
            self._tensor(np.asarray(mesh.faces, dtype=np.uint32)),
        )

    def render(self, observed_pixel_indices, pose, *, directions):
        if not len(observed_pixel_indices):
            return np.empty(0, dtype=np.float32)
        rows, cols = (observed_pixel_indices // self.pixel_stride).T
        rotation, translation = pose[:3, :3], pose[:3, 3]
        rays = np.empty((len(rows), 6), dtype=np.float32)
        rays[:, :3] = -translation @ rotation
        rays[:, 3:] = directions[rows, cols] @ rotation
        # 不归一化方向；刚体变换后的 t_hit 仍是相机 Z，而非欧氏距离。
        hits = self._scene.cast_rays(self._tensor(rays), nthreads=RAYCAST_THREADS)[
            "t_hit"
        ].numpy()
        return np.where(np.isfinite(hits) & (hits > 0), hits, 0)

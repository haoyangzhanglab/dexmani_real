"""Open3D 点云窗口；点和静态几何均位于 xArm 基座系。"""

from collections.abc import Sequence

import numpy as np


class PointCloudViewer:
    def __init__(
        self, *, workspace: Sequence[float], base_from_color: np.ndarray,
        keys: Sequence[str] = (), title: str = "Point Cloud",
    ):
        import open3d as o3d

        self.o3d = o3d
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.events: list[str] = []
        self.has_points = False
        try:
            if not self.vis.create_window(title, width=960, height=720):
                raise RuntimeError("Open3D could not create a window")
            self.cloud = o3d.geometry.PointCloud()
            self.vis.add_geometry(self.cloud, reset_bounding_box=False)
            lower, upper = np.asarray(workspace).reshape(2, 3)
            box = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
                o3d.geometry.AxisAlignedBoundingBox(lower, upper),
            )
            box.paint_uniform_color([0, 1, 0])
            self.vis.add_geometry(box)
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2))
            camera = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15)
            camera.transform(base_from_color)
            self.vis.add_geometry(camera)
            options = self.vis.get_render_option()
            options.point_size = 3.0
            options.background_color = np.ones(3)
            for key in {*keys, "q", "esc"}:
                code = 256 if key == "esc" else ord(key.upper())
                self.vis.register_key_action_callback(code, self.key_callback(key))
        except BaseException:
            self.close()
            raise

    def key_callback(self, key: str):
        def callback(vis, action, mods):
            if action == 1:
                self.events.append(key)
            return False
        return callback

    def update(self, points: np.ndarray | None) -> None:
        if points is None:
            points = np.empty((0, 6), dtype=np.float32)
        self.cloud.points = self.o3d.utility.Vector3dVector(points[:, :3])
        self.cloud.colors = self.o3d.utility.Vector3dVector(points[:, 3:6])
        self.vis.update_geometry(self.cloud)
        if len(points) and not self.has_points:
            self.vis.reset_view_point(True)
            self.has_points = True

    def poll_events(self) -> list[str]:
        if self.vis is None:
            return ["closed"]
        if not self.vis.poll_events():
            self.close()
            return ["closed"]
        self.vis.update_renderer()
        events, self.events = self.events, []
        return events

    def close(self) -> None:
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None

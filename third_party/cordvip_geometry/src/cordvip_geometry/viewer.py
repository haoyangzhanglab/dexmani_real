"""已保存场景结果的 RGB 投影与可选 Open3D 点云视图。"""

from __future__ import annotations

import logging

import numpy as np

from .geometry import transform_points

logger = logging.getLogger(__name__)


def scene_display(frame, object_ids, colors) -> np.ndarray:
    import cv2

    from .geometry import project_points

    rgb = frame['rgb']
    overlay = rgb.copy()
    for index, name in enumerate(object_ids):
        prefix = name + '__'
        color = tuple(int(value) for value in colors[name])
        mask = frame[prefix + 'semantic_mask']
        if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
            raise ValueError(f'{name}: semantic mask must match the saved RGB grid')
        overlay[mask] = (overlay[mask] * .62 + np.asarray(color) * .38).astype(np.uint8)
        pose_valid = bool(frame[prefix + 'pose_valid'])
        if pose_valid:
            pose = frame[prefix + 'T_camera_object']
            axes = np.array([[0, 0, 0], [.035, 0, 0], [0, .035, 0], [0, 0, .035]])
            uv, good = project_points(axes @ pose[:3, :3].T + pose[:3, 3], frame['K'])
            if good.all() and np.abs(uv).max() < 1e6:
                uv = np.rint(uv).astype(int)
                for axis, axis_color in enumerate(((255, 70, 70), (70, 255, 70), (70, 70, 255)), 1):
                    cv2.line(overlay, tuple(uv[0]), tuple(uv[axis]), axis_color, 2)
        state = str(frame[prefix + 'state'])
        error = str(frame[prefix + 'error'])
        label = f'{name}: {state}' + (f' | {error}' if error else '')
        cv2.putText(overlay, label, (10, 24 + index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
    return np.ascontiguousarray(overlay[..., ::-1])


def scene_clouds(
    frame, object_ids, colors, *, model_templates: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """None 选择实测点，模板字典选择由有效位姿重建的模型点。"""
    points, point_colors = [], []
    for name in object_ids:
        if model_templates is not None:
            cloud = (
                transform_points(model_templates[name], frame[name + '__T_camera_object'])
                if name in model_templates and bool(frame[name + '__pose_valid'])
                else np.empty((0, 3), np.float32)
            )
        else:
            cloud = frame[name + '__observed_points_camera']
        points.append(cloud)
        point_colors.append(np.tile(np.asarray(colors[name]) / 255., (len(cloud), 1)))
    if 'hand_points_camera' in frame:
        hand = frame['hand_points_camera']
        points.append(hand)
        point_colors.append(np.tile([1., .55, .14], (len(hand), 1)))
    cloud = np.concatenate(points) if points else np.empty((0, 3), np.float32)
    cloud_colors = np.concatenate(point_colors) if point_colors else np.empty((0, 3), np.float32)
    return cloud, cloud_colors


class DemoViewer:
    def __init__(self, *, show_3d=True):
        import cv2

        self.title = 'Offline scene | n/space next | v observed/model | q quit'
        self.vis = None
        self._has_points = False
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        try:
            if show_3d:
                import open3d as o3d

                self.o3d = o3d
                self.vis = o3d.visualization.Visualizer()
                if not self.vis.create_window('Scene in color-camera frame', 960, 720):
                    raise RuntimeError('Open3D could not create a window; use --no-3d or --headless')
                self.cloud = o3d.geometry.PointCloud()
                self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1))
                self.vis.add_geometry(self.cloud)
                self.vis.get_render_option().point_size = 3.
        except BaseException:
            try:
                self.close()
            except BaseException:
                logger.exception("scene viewer cleanup also failed")
            raise

    def update(self, panel: np.ndarray) -> None:
        """新行到达时更新 RGB 内容。"""
        import cv2

        cv2.imshow(self.title, panel)

    def update_cloud(self, points: np.ndarray, colors: np.ndarray) -> None:
        """仅在点云内容变化时提交 Open3D 几何。"""
        if self.vis is not None:
            self.cloud.points = self.o3d.utility.Vector3dVector(points)
            self.cloud.colors = self.o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.cloud)
            if not self._has_points and len(points):
                self.vis.reset_view_point(True)
                self._has_points = True

    def poll(self) -> int:
        """维持窗口响应和必要渲染，不重建显示内容。"""
        import cv2

        if self.vis is not None:
            if not self.vis.poll_events():
                return ord('q')
            self.vis.update_renderer()
        if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            return ord('q')
        return cv2.waitKey(20) & 0xff

    def close(self):
        import cv2

        failure = None
        if self.vis is not None:
            try:
                self.vis.destroy_window()
            except BaseException as error:
                failure = error
        try:
            cv2.destroyWindow(self.title)
        except BaseException as error:
            if failure is None:
                failure = error
            else:
                logger.exception("scene RGB window close also failed")
        if failure is not None:
            raise failure

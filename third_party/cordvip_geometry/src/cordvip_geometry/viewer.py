"""已保存场景结果的 RGB 投影与可选 Open3D 点云视图。"""

from __future__ import annotations

import numpy as np


def scene_display(frame, object_ids, colors, *, show_model=False, include_clouds=False):
    import cv2

    from .geometry import project_points

    rgb = frame['rgb']
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) == 0:
        raise ValueError('scene result RGB must be nonempty uint8 [H,W,3]')
    overlay = rgb.copy()
    points, point_colors = [], []
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
        cv2.putText(overlay, f'{name}: {state}', (10, 24 + index * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        if include_clouds:
            cloud = frame[prefix + ('model_points_camera' if show_model else 'observed_points_camera')]
            points.append(cloud)
            point_colors.append(np.tile(np.asarray(color) / 255., (len(cloud), 1)))
    if include_clouds and 'hand_points_camera' in frame:
        hand = frame['hand_points_camera']
        points.append(hand)
        point_colors.append(np.tile([1., .55, .14], (len(hand), 1)))
    cloud = np.concatenate(points) if points else np.empty((0, 3), np.float32)
    cloud_colors = np.concatenate(point_colors) if point_colors else np.empty((0, 3), np.float32)
    return np.ascontiguousarray(overlay[..., ::-1]), cloud, cloud_colors


class DemoViewer:
    def __init__(self, *, show_3d=True):
        import cv2

        self.title = 'Offline scene | n/space next | v observed/model | q quit'
        self.vis = None
        self._has_points = False
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if show_3d:
            import open3d as o3d

            self.o3d = o3d
            self.vis = o3d.visualization.Visualizer()
            if not self.vis.create_window('Scene in color-camera frame', 960, 720):
                self.vis = None
                cv2.destroyWindow(self.title)
                raise RuntimeError('Open3D could not create a window; use --no-3d or --headless')
            self.cloud = o3d.geometry.PointCloud()
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1))
            self.vis.add_geometry(self.cloud)
            self.vis.get_render_option().point_size = 3.

    def update(self, panel, points, colors):
        import cv2

        cv2.imshow(self.title, panel)
        if self.vis is not None:
            self.cloud.points = self.o3d.utility.Vector3dVector(points)
            self.cloud.colors = self.o3d.utility.Vector3dVector(colors)
            self.vis.update_geometry(self.cloud)
            if not self._has_points and len(points):
                self.vis.reset_view_point(True)
                self._has_points = True
            if not self.vis.poll_events():
                return ord('q')
            self.vis.update_renderer()
        if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            return ord('q')
        return cv2.waitKey(20) & 0xff

    def close(self):
        import cv2

        if self.vis is not None:
            self.vis.destroy_window()
        cv2.destroyWindow(self.title)

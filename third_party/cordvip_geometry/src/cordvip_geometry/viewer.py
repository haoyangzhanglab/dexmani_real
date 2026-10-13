"""RGB projections and an optional camera-frame Open3D view."""

from __future__ import annotations

import numpy as np


def project_points(points, K):
    points = np.asarray(points)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 1e-5)
    pixels = np.full((len(points), 2), np.nan)
    projected = points[valid] @ np.asarray(K).T
    pixels[valid] = projected[:, :2] / projected[:, 2:3]
    return pixels, valid


def make_panel(rgb, depth_m, K, *, model_points=None, pose=None, bounds_camera=None, lines=()):
    import cv2

    overlay = np.ascontiguousarray(rgb[..., ::-1]).copy()
    height, width = depth_m.shape
    if model_points is not None:
        pixels, valid = project_points(model_points, K)
        valid &= ((pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                  & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
        for x, y in np.rint(pixels[valid]).astype(int):
            cv2.circle(overlay, (int(x), int(y)), 1, (255, 200, 0), -1)
    if bounds_camera is not None:
        pixels, valid = project_points(bounds_camera, K)
        for start in range(8):
            for bit in (1, 2, 4):
                end = start ^ bit
                if end > start and valid[start] and valid[end] and np.abs(pixels[[start, end]]).max() < 1e6:
                    p1, p2 = np.rint(pixels[[start, end]]).astype(int)
                    cv2.line(overlay, tuple(p1), tuple(p2), (255, 200, 0), 1)
    if pose is not None:
        # Axes belong to the mesh's object frame; they are not a robot frame.
        axes = np.array([[0, 0, 0], [.04, 0, 0], [0, .04, 0], [0, 0, .04]])
        camera_axes = axes @ pose[:3, :3].T + pose[:3, 3]
        pixels, valid = project_points(camera_axes, K)
        if valid.all() and np.abs(pixels).max() < 1e6:
            pixels = np.rint(pixels).astype(int)
            for index, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0)), 1):
                cv2.line(overlay, tuple(pixels[0]), tuple(pixels[index]), color, 2)
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    # Fixed 0--2 m display range affects colors only, never tracking inputs.
    depth_u8 = (np.clip(np.where(valid_depth, depth_m, 0) / 2, 0, 1) * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_vis[~valid_depth] = 0
    panel = np.hstack((overlay, depth_vis))
    text_lines = ["RGB / model projection | depth (0--2 m)", *lines]
    for index, line in enumerate(text_lines):
        origin = (10, 22 + index * 22)
        cv2.putText(panel, str(line), origin, cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 0), 3)
        cv2.putText(panel, str(line), origin, cv2.FONT_HERSHEY_SIMPLEX, .48, (255, 255, 255), 1)
    return panel


def select_mask(rgb):
    """Draw on the frozen rectified frame; Enter accepts, Esc cancels."""
    import cv2

    title = "Object mask: click polygon; right click undo; Enter accept; Esc cancel"
    points = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while True:
            canvas = np.ascontiguousarray(rgb[..., ::-1]).copy()
            if len(points) >= 2:
                cv2.polylines(canvas, [np.array(points, dtype=np.int32)], len(points) >= 3,
                              (0, 255, 255), 2)
            for point in points:
                cv2.circle(canvas, point, 3, (0, 255, 255), -1)
            cv2.imshow(title, canvas)
            key = cv2.waitKey(20) & 0xff
            if key in (10, 13) and len(points) >= 3:
                mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 1)
                if np.count_nonzero(mask) >= 16:
                    return mask.astype(bool)
            if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise KeyboardInterrupt("Mask selection cancelled")
    finally:
        cv2.destroyWindow(title)


class DemoViewer:
    def __init__(self, *, show_3d=True):
        import cv2

        self.title = "CordViP object example | space pause | n step | r register | s snapshot | q quit"
        self.layer = 0
        self.vis = self.observed = self.model = self.axis = None
        self._has_points = False
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if show_3d:
            import open3d as o3d

            self.o3d = o3d
            self.vis = o3d.visualization.Visualizer()
            if not self.vis.create_window("Camera frame: gray observed / cyan model", 960, 720):
                self.vis = None
                cv2.destroyWindow(self.title)
                raise RuntimeError("Open3D could not create a window; use --no-3d or --headless")
            self.observed = o3d.geometry.PointCloud()
            self.model = o3d.geometry.PointCloud()
            self.axis = o3d.geometry.LineSet()
            self.axis.points = o3d.utility.Vector3dVector(np.zeros((4, 3)))
            self.axis.lines = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
            self.axis.colors = o3d.utility.Vector3dVector(np.eye(3))
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=.1))
            self.vis.add_geometry(self.observed)
            self.vis.add_geometry(self.model)
            self.vis.add_geometry(self.axis)
            options = self.vis.get_render_option()
            options.background_color = np.array([.08, .08, .08])
            options.point_size = 3.0

    def update(self, panel, observed_points, model_points=None, *, pose=None, paused=False):
        import cv2

        if paused:
            panel = panel.copy()
            cv2.putText(panel, "PAUSED: space continue / n next frame", (10, panel.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 255), 2)
        cv2.imshow(self.title, panel)
        if self.vis is not None:
            empty = np.empty((0, 3))
            observed = observed_points if self.layer != 2 else empty
            model = model_points if model_points is not None and self.layer != 1 else empty
            self.observed.points = self.o3d.utility.Vector3dVector(observed)
            self.observed.paint_uniform_color([.55, .55, .55])
            self.model.points = self.o3d.utility.Vector3dVector(model)
            self.model.paint_uniform_color([0, .8, 1])
            self.vis.update_geometry(self.observed)
            self.vis.update_geometry(self.model)
            if pose is not None:
                local_axes = np.array([[0, 0, 0], [.04, 0, 0], [0, .04, 0], [0, 0, .04]])
                self.axis.points = self.o3d.utility.Vector3dVector(local_axes @ pose[:3, :3].T + pose[:3, 3])
                self.vis.update_geometry(self.axis)
            if not self._has_points and len(observed_points):
                self.vis.reset_view_point(True)
                self._has_points = True
            if not self.vis.poll_events():
                return ord("q")
            self.vis.update_renderer()
        if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            return ord("q")
        key = cv2.waitKey(1) & 0xff
        if key == ord("v"):
            self.layer = (self.layer + 1) % 3
        return key

    def screenshot(self, path):
        if self.vis is not None:
            self.vis.capture_screen_image(str(path), do_render=True)

    def close(self):
        import cv2

        if self.vis is not None:
            self.vis.destroy_window()
        cv2.destroyAllWindows()

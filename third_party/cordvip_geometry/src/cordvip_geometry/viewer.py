"""RGB projections and an optional camera-frame Open3D view."""

from __future__ import annotations

import numpy as np


def make_panel(rgb, depth_m, *, lines=()):
    import cv2

    overlay = np.ascontiguousarray(rgb[..., ::-1]).copy()
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    # Fixed 0--2 m display range affects colors only, never tracking inputs.
    depth_u8 = (np.clip(np.where(valid_depth, depth_m, 0) / 2, 0, 1) * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    depth_vis[~valid_depth] = 0
    panel = np.hstack((overlay, depth_vis))
    text_lines = ["RGB | measured depth (0--2 m)", *lines]
    for index, line in enumerate(text_lines):
        origin = (10, 22 + index * 22)
        cv2.putText(panel, str(line), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
        cv2.putText(panel, str(line), origin, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    return panel


class DemoViewer:
    def __init__(self, *, show_3d=True):
        import cv2

        self.title = "CordViP object example | space pause | n step | s snapshot | q quit"
        self.vis = self.observed = None
        self._has_points = False
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        if show_3d:
            import open3d as o3d

            self.o3d = o3d
            self.vis = o3d.visualization.Visualizer()
            if not self.vis.create_window("Camera frame: observed depth points", 960, 720):
                self.vis = None
                cv2.destroyWindow(self.title)
                raise RuntimeError("Open3D could not create a window; use --no-3d or --headless")
            self.observed = o3d.geometry.PointCloud()
            self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
            self.vis.add_geometry(self.observed)
            options = self.vis.get_render_option()
            options.background_color = np.array([0.08, 0.08, 0.08])
            options.point_size = 3.0

    def update(self, panel, observed_points, *, paused=False):
        import cv2

        if paused:
            panel = panel.copy()
            cv2.putText(
                panel,
                "PAUSED: space continue / n next frame",
                (10, panel.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
        cv2.imshow(self.title, panel)
        if self.vis is not None:
            self.observed.points = self.o3d.utility.Vector3dVector(observed_points)
            self.observed.paint_uniform_color([0.55, 0.55, 0.55])
            self.vis.update_geometry(self.observed)
            if not self._has_points and len(observed_points):
                self.vis.reset_view_point(True)
                self._has_points = True
            if not self.vis.poll_events():
                return ord("q")
            self.vis.update_renderer()
        if cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            return ord("q")
        return cv2.waitKey(1) & 0xFF

    def screenshot(self, path):
        if self.vis is not None:
            self.vis.capture_screen_image(str(path), do_render=True)

    def close(self):
        import cv2

        if self.vis is not None:
            self.vis.destroy_window()
        cv2.destroyAllWindows()


def scene_display(result, colors, hand=None, *, include_clouds):
    import cv2
    import numpy as np

    from .geometry import project_points

    overlay = result["rgb"].copy()
    observed, models, observed_colors, model_colors = [], [], [], []
    for name, item in result["objects"].items():
        color = colors[name]
        mask = item["cloud"].semantic_mask
        overlay[mask] = (overlay[mask] * 0.62 + np.asarray(color) * 0.38).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, contours, -1, color, 1)
        if include_clouds:
            points = item["cloud"].observed_points_camera
            observed.append(points)
            observed_colors.append(np.tile(np.asarray(color) / 255.0, (len(points), 1)))
            points = item["model_points_camera"]
            models.append(points)
            model_colors.append(np.tile(np.asarray(color) / 255.0, (len(points), 1)))
        if item["pose_valid"]:
            p = item["pose"]
            axes = np.array([[0, 0, 0], [0.035, 0, 0], [0, 0.035, 0], [0, 0, 0.035]])
            uv, good = project_points(axes @ p[:3, :3].T + p[:3, 3], result["K"])
            if good.all() and np.abs(uv).max() < 1e6:
                uv = np.rint(uv).astype(int)
                for i, axis_color in enumerate(((255, 70, 70), (70, 255, 70), (70, 70, 255)), 1):
                    cv2.line(overlay, tuple(uv[0]), tuple(uv[i]), axis_color, 2)
                cv2.putText(overlay, name, tuple(uv[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    if hand is not None:
        uv, good = project_points(hand["points_camera"], result["K"])
        h, w = overlay.shape[:2]
        good &= (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        for u, v in np.rint(uv[good]).astype(int):
            cv2.circle(overlay, (int(u), int(v)), 1, (255, 140, 35), -1)
    clouds = dict(
        observed_points=np.concatenate(observed),
        model_points=np.concatenate(models),
        observed_colors=np.concatenate(observed_colors),
        model_colors=np.concatenate(model_colors),
    ) if include_clouds else {}
    return overlay, clouds

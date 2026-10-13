"""Nonblocking source-frame comparison and hand/object point-cloud display."""

from pathlib import Path

import numpy as np


class LiveViewer:
    """Display RGB inputs; all 3-D geometry is in the color-camera frame, meters.

    Selection freezes only the displayed initialization image. The application
    remains responsible for associating its mask with that exact RGB-D frame.
    """

    _WIDTH = 640
    _HEIGHT = 480
    _HEADER = 52
    _FOOTER = 210

    def __init__(self, show_3d=True):
        import cv2

        self.cv2 = cv2
        self.title = "Live object pose | camera and processed source frame"
        self._selection_rgb = None
        self._polygon = []
        self._manual_selection = True
        self._candidates = []
        self._candidate_index = None
        self._selection_rect = None
        self._panel = None
        self._pending_key = 255
        self.layer = 0
        self.vis = None
        self._view_initialized = False
        self._closed = False
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, 2 * self._WIDTH, self._HEADER + self._HEIGHT + self._FOOTER)
        cv2.setMouseCallback(self.title, self._on_mouse)
        if show_3d:
            try:
                import open3d as o3d

                self.o3d = o3d
                self.vis = o3d.visualization.VisualizerWithKeyCallback()
                if not self.vis.create_window(
                    "Camera frame | observed geometry / object models / hand", 960, 720
                ):
                    raise RuntimeError("Open3D could not create a window; use --no-3d")
                self.observed = o3d.geometry.PointCloud()
                self.model = o3d.geometry.PointCloud()
                self.hand = o3d.geometry.PointCloud()
                self.axis = o3d.geometry.LineSet()
                self.vis.add_geometry(self.model, reset_bounding_box=False)
                self.vis.add_geometry(self.observed, reset_bounding_box=False)
                self.vis.add_geometry(self.hand, reset_bounding_box=False)
                self.vis.add_geometry(self.axis, reset_bounding_box=False)
                options = self.vis.get_render_option()
                options.background_color = np.array([.06, .07, .08])
                options.point_size = 3.0
                for source, target in ((82, ord("r")), (73, ord("i")), (80, ord("p")),
                                       (83, ord("s")), (86, ord("v")), (32, 32),
                                       (81, ord("q")), (257, 13), (256, 27),
                                       (91, ord("[")), (93, ord("]")),
                                       *((key, key) for key in range(49, 58))):
                    self.vis.register_key_callback(source, self._forward_key(target))
            except Exception:
                self.close()
                raise

    @property
    def selection_active(self):
        return self._selection_rgb is not None

    @staticmethod
    def _rgb(image):
        image = np.asarray(image)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or not all(image.shape[:2]):
            raise ValueError("Viewer images must be nonempty uint8 RGB (H, W, 3)")
        return image

    def begin_selection(self, rgb, *, manual=True):
        self._selection_rgb = self._rgb(rgb).copy()
        self._polygon = []
        self._selection_rect = None
        self._manual_selection = manual
        self._candidates = []
        self._candidate_index = None

    def set_candidates(self, candidates):
        if not self.selection_active:
            raise RuntimeError('Freeze the source image before showing candidates')
        for candidate in candidates:
            if np.asarray(candidate['mask']).shape != self._selection_rgb.shape[:2]:
                raise ValueError('Candidate mask must match the frozen source image')
        self._manual_selection = False
        self._candidates = candidates
        self._candidate_index = 0 if candidates else None

    def use_polygon(self):
        if not self.selection_active:
            raise RuntimeError('Freeze the source image before editing a polygon')
        self._manual_selection = True
        self._polygon = []
        self._candidates = []
        self._candidate_index = None

    @property
    def selected_candidate_index(self):
        return self._candidate_index

    def choose_candidate(self, key):
        if not self.selection_active or not self._candidates:
            return False
        if ord('1') <= key <= ord('9'):
            index = key - ord('1')
            if index >= len(self._candidates):
                return False
        elif key in (ord('['), ord(']')):
            index = (self._candidate_index + (1 if key == ord(']') else -1)) % len(self._candidates)
        else:
            return False
        self._candidate_index = index
        return True

    def cancel_selection(self):
        self._selection_rgb = None
        self._polygon = []
        self._selection_rect = None
        self._candidates = []
        self._candidate_index = None

    def selected_mask(self):
        if not self.selection_active:
            return None
        if not self._manual_selection:
            return self._candidates[self._candidate_index]['mask'].copy() if self._candidate_index is not None else None
        if len(self._polygon) < 3:
            return None
        mask = np.zeros(self._selection_rgb.shape[:2], dtype=np.uint8)
        self.cv2.fillPoly(mask, [np.asarray(self._polygon, dtype=np.int32)], 1)
        return mask.astype(bool) if np.count_nonzero(mask) >= 16 else None

    def _on_mouse(self, event, x, y, flags, param):
        if not self.selection_active or self._selection_rect is None:
            return
        left, top, width, height = self._selection_rect
        if not (left <= x < left + width and top <= y < top + height):
            return
        if event == self.cv2.EVENT_LBUTTONDOWN:
            source_h, source_w = self._selection_rgb.shape[:2]
            u = min(source_w - 1, int((x - left) * source_w / width))
            v = min(source_h - 1, int((y - top) * source_h / height))
            if self._manual_selection:
                self._polygon.append((u, v))
            else:
                for index, candidate in enumerate(self._candidates):
                    if candidate['mask'][v, u]:
                        self._candidate_index = index
                        break
        elif self._manual_selection and event == self.cv2.EVENT_RBUTTONDOWN and self._polygon:
            self._polygon.pop()

    def _forward_key(self, key):
        def callback(vis):
            self._pending_key = key
            return False
        return callback

    def _text(self, canvas, text, origin, color=(220, 220, 220), scale=.48, max_width=None):
        text = str(text)
        if max_width is not None:
            original = text
            while text and self.cv2.getTextSize(text, self.cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > max_width:
                text = text[:-1]
            if text != original:
                text = text[:-3] + "..."
        self.cv2.putText(canvas, text, origin, self.cv2.FONT_HERSHEY_SIMPLEX,
                         scale, color, 1, self.cv2.LINE_AA)

    def _place_image(self, canvas, image, panel_index):
        if image is None:
            self._text(canvas, "Waiting for frame", (panel_index * self._WIDTH + 225, self._HEADER + 235))
            return None
        image = self._rgb(image)
        h, w = image.shape[:2]
        ratio = min(self._WIDTH / w, self._HEIGHT / h)
        width, height = max(1, round(w * ratio)), max(1, round(h * ratio))
        left = panel_index * self._WIDTH + (self._WIDTH - width) // 2
        top = self._HEADER + (self._HEIGHT - height) // 2
        resized = self.cv2.resize(image[..., ::-1], (width, height), interpolation=self.cv2.INTER_AREA)
        canvas[top:top + height, left:left + width] = resized
        return left, top, width, height

    def _draw_selection(self, canvas):
        if not self._polygon or self._selection_rect is None:
            return
        left, top, width, height = self._selection_rect
        source_h, source_w = self._selection_rgb.shape[:2]
        polygon = np.asarray(self._polygon, dtype=np.float64)
        polygon *= [width / source_w, height / source_h]
        polygon += [left, top]
        polygon = np.rint(polygon).astype(np.int32)
        if len(polygon) >= 2:
            self.cv2.polylines(canvas, [polygon], len(polygon) >= 3, (0, 230, 255), 2)
        for point in polygon:
            self.cv2.circle(canvas, tuple(point), 3, (0, 230, 255), -1)

    def _candidate_image(self):
        image = self._selection_rgb.copy()
        for index, candidate in enumerate(self._candidates):
            selected = index == self._candidate_index
            color = np.array([255, 220, 30] if selected else [70, 190, 220])
            mask = candidate['mask']
            image[mask] = (image[mask] * .6 + color * .4).astype(np.uint8)
            x1, y1, x2, y2 = np.rint(candidate['box_xyxy']).astype(int)
            self.cv2.rectangle(image, (x1, y1), (x2, y2), tuple(int(v) for v in color), 2 if selected else 1)
            label = f"{index + 1}: {candidate['label']} {candidate['score']:.2f}"
            self._text(image, label, (max(2, x1), max(16, y1 - 5)),
                       tuple(int(v) for v in color), .48)
        return image

    @staticmethod
    def _points(points):
        if points is None:
            return np.empty((0, 3))
        points = np.asarray(points)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Viewer points must be finite (N, 3), in camera-frame meters")
        return points

    @staticmethod
    def _colors(colors, points):
        if colors is None:
            return None
        colors = np.asarray(colors)
        if (colors.shape != points.shape or not np.issubdtype(colors.dtype, np.floating)
                or not np.isfinite(colors).all() or np.any(colors < 0) or np.any(colors > 1)):
            raise ValueError("Viewer colors must be floating RGB (N, 3) in [0, 1], aligned with points")
        return colors

    def _update_3d(self, observed_points, model_points, pose, valid, hand_points=None,
                   *, observed_colors=None, model_colors=None):
        observed = self._points(observed_points)
        model = self._points(model_points)
        observed_colors = self._colors(observed_colors, observed)
        model_colors = self._colors(model_colors, model)
        empty = np.empty((0, 3))
        self.observed.points = self.o3d.utility.Vector3dVector(observed if self.layer != 2 else empty)
        if observed_colors is None:
            self.observed.paint_uniform_color([.15, .95, .30])
        else:
            colors = observed_colors if valid else np.full(observed.shape, .4)
            self.observed.colors = self.o3d.utility.Vector3dVector(colors if self.layer != 2 else empty)
        self.model.points = self.o3d.utility.Vector3dVector(model if self.layer != 1 else empty)
        if model_colors is None:
            self.model.paint_uniform_color([0, .8, 1] if valid else [.4, .4, .4])
        else:
            colors = model_colors if valid else np.full(model.shape, .4)
            self.model.colors = self.o3d.utility.Vector3dVector(colors if self.layer != 1 else empty)
        self.vis.update_geometry(self.observed)
        self.vis.update_geometry(self.model)
        self.hand.points = self.o3d.utility.Vector3dVector(self._points(hand_points))
        self.hand.paint_uniform_color([1., .55, .1] if valid else [.4, .4, .4])
        self.vis.update_geometry(self.hand)
        if pose is not None:
            pose = np.asarray(pose)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("Viewer pose must be a finite (4, 4) matrix")
            local = np.array([[0, 0, 0], [.04, 0, 0], [0, .04, 0], [0, 0, .04]])
            self.axis.points = self.o3d.utility.Vector3dVector(local @ pose[:3, :3].T + pose[:3, 3])
            self.axis.lines = self.o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
            self.axis.colors = self.o3d.utility.Vector3dVector(np.eye(3) if valid else np.full((3, 3), .4))
        else:
            self.axis.points = self.o3d.utility.Vector3dVector(empty)
            self.axis.lines = self.o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
            self.axis.colors = self.o3d.utility.Vector3dVector(empty)
        self.vis.update_geometry(self.axis)
        if not self._view_initialized and (len(observed) or len(model)):
            self.vis.reset_view_point(True)
            # Match the RGB camera's +Z forward, +Y down convention on first
            # display. Keep this view fixed afterwards so motion remains visible.
            points = np.concatenate((observed, model, self._points(hand_points)))
            control = self.vis.get_view_control()
            control.set_lookat((points.min(axis=0) + points.max(axis=0)) / 2)
            control.set_front([0, 0, -1])
            control.set_up([0, -1, 0])
            control.set_zoom(1.)
            self._view_initialized = True
        if not self.vis.poll_events():
            return False
        self.vis.update_renderer()
        return True

    def update(self, live_rgb, result_rgb, observed_points=None, model_points=None, pose=None,
               *, lines=(), result_valid=False, hand_points=None, observed_colors=None,
               model_colors=None, status_text=None, help_text=None):
        """Draw one UI tick and return a key (255 for none); never block selection.

        Optional per-point RGB colors use float [0, 1] and become grey when the
        result is invalid. Text overrides let a caller describe mesh-free or
        multi-object results without claiming a single object pose is available.
        """
        if self._closed:
            return ord("q")
        canvas = np.full((self._HEADER + self._HEIGHT + self._FOOTER, 2 * self._WIDTH, 3), 22, np.uint8)
        left_label = "FROZEN INITIALIZATION FRAME" if self.selection_active else "LATEST CAMERA FRAME"
        self._text(canvas, left_label, (12, 22), (0, 220, 255) if self.selection_active else (235, 235, 235), .55)
        self._text(canvas, "LAST PROCESSED SOURCE FRAME", (self._WIDTH + 12, 22), scale=.55)
        if self.selection_active:
            selection_help = ("Left click: polygon | right click: undo | Enter: register | Esc: cancel"
                              if self._manual_selection else
                              "1-9 / [ ] / click: candidate | Enter: confirm | p: polygon | Esc: cancel")
            self._text(canvas, selection_help, (12, 43), scale=.43)
        if status_text is None:
            status_text = "Pose available" if result_valid else "No valid pose / retained result only"
        self._text(canvas, status_text,
                   (self._WIDTH + 12, 43), (100, 220, 100) if result_valid else (80, 170, 240), .43)
        left_image = self._selection_rgb if self.selection_active else live_rgb
        if self.selection_active and self._candidates:
            left_image = self._candidate_image()
        self._selection_rect = self._place_image(canvas, left_image, 0) if self.selection_active else None
        if not self.selection_active:
            self._place_image(canvas, left_image, 0)
        self._place_image(canvas, result_rgb, 1)
        self._draw_selection(canvas)
        self.cv2.line(canvas, (self._WIDTH, 0), (self._WIDTH, self._HEADER + self._HEIGHT), (80, 80, 80), 1)
        footer_y = self._HEADER + self._HEIGHT + 21
        for index, line in enumerate(list(lines)[:8]):
            self._text(canvas, line, (12, footer_y + 20 * index), max_width=2 * self._WIDTH - 24)
        mode = ("overlay", "measured object", "model")[self.layer]
        if help_text is None:
            help_text = f"i/r: select/re-register | p: polygon | space: pause | v: 3D {mode} | s: save | q: quit"
        self._text(canvas, help_text,
                   (12, canvas.shape[0] - 14), (190, 200, 210))
        self._panel = canvas
        self.cv2.imshow(self.title, canvas)
        if self.vis is not None and not self._update_3d(
            observed_points, model_points, pose, result_valid, hand_points,
            observed_colors=observed_colors, model_colors=model_colors,
        ):
            return ord("q")
        key = self.cv2.waitKey(1) & 0xff
        if self.cv2.getWindowProperty(self.title, self.cv2.WND_PROP_VISIBLE) < 1:
            return ord("q")
        if key == 255:
            key = self._pending_key
        self._pending_key = 255
        if key == ord("v"):
            self.layer = (self.layer + 1) % 3
        return key

    def screenshot(self, path):
        """Save the displayed source/status panel and, if enabled, a *_3d image."""
        if self._panel is None:
            raise RuntimeError("Display a frame before taking a screenshot")
        path = Path(path)
        if not self.cv2.imwrite(str(path), self._panel):
            raise OSError(f"Could not write screenshot: {path}")
        if self.vis is not None:
            self.vis.capture_screen_image(str(path.with_name(f"{path.stem}_3d{path.suffix}")), do_render=True)

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.vis is not None:
            self.vis.destroy_window()
            self.vis = None
        self.cv2.destroyWindow(self.title)

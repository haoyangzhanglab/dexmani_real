"""模拟 OpenCV/Open3D 窗口，验证事件和资源生命周期。"""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.sensor.realsense import RGBDFrame
from dexmani_real.utils.visualization.image import ImageViewer
from dexmani_real.utils.visualization.pointcloud import PointCloudViewer
from scripts.tools.view_pointcloud import calibrate_table_plane
from scripts.tools.view_realsense import RealSenseViewer


class ImageEventTests(unittest.TestCase):
    def test_no_key_and_control_characters(self):
        with patch("cv2.namedWindow"), patch("cv2.getWindowProperty", return_value=1), \
                patch("cv2.waitKey", side_effect=[-1, ord("P"), ord("S"), 13, 27]):
            viewer = ImageViewer("offline")
            self.assertEqual([viewer.poll_events() for _ in range(5)], [[], ["p"], ["s"], ["enter"], ["esc"]])

    def test_closed_window_releases_once(self):
        with patch("cv2.namedWindow"), patch("cv2.getWindowProperty", return_value=0), \
                patch("cv2.waitKey", return_value=-1), patch("cv2.destroyWindow") as destroy:
            viewer = ImageViewer("offline")
            self.assertEqual(viewer.poll_events(), ["closed"])
            viewer.close()
            destroy.assert_called_once()


class CloudEventTests(unittest.TestCase):
    def test_closed_event_already_releases_window(self):
        viewer = PointCloudViewer.__new__(PointCloudViewer)
        vis = Mock()
        vis.poll_events.return_value = False
        viewer.vis = vis
        self.assertEqual(viewer.poll_events(), ["closed"])
        viewer.close()
        vis.destroy_window.assert_called_once()

    def test_toggle_callback_ignores_release_and_auto_repeat(self):
        viewer = PointCloudViewer.__new__(PointCloudViewer)
        viewer.events = []
        callback = viewer.key_callback("p")
        for action in (0, 1, 2):
            callback(None, action, 0)
        self.assertEqual(viewer.events, ["p"])


class TableCalibrationTests(unittest.TestCase):
    def test_five_distinct_depth_frames_and_explicit_publication(self):
        for answer in ("n", "y"):
            with self.subTest(answer=answer):
                camera = Mock(depth_scale_m=0.001)
                camera.color_intrinsics = object()
                frames = [None, *[SimpleNamespace(depth_frame_number=i, depth=np.zeros((2, 3)))
                                 for i in (1, 1, 2, 3, 4, 5)]]
                camera.read.side_effect = frames
                points = np.tile([0.3, 0.0, 0.4], (3, 1))
                fit = SimpleNamespace(plane_abcd=(0., 0., 1., 0.), inlier_points=15,
                                      evaluated_points=15, inlier_ratio=1., rms_residual_m=0., tilt_deg=0.)
                with patch("builtins.input", side_effect=["", answer]), patch("builtins.print"), \
                        patch("dexmani_real.perception.pointcloud.aligned_depth_points_in_base", return_value=points) as deproject, \
                        patch("dexmani_real.geometry.plane.fit_table_plane", return_value=fit) as fit_plane, \
                        patch("dexmani_real.utils.table_plane.save_table_plane") as save:
                    result = calibrate_table_plane(camera=camera, T_xarm_base_from_color=np.eye(4),
                                                   config=PointCloudConfig(), plane_path=Path("/tmp/offline-plane.json"))
                self.assertEqual(result, fit.plane_abcd)
                self.assertEqual(deproject.call_count, 5)
                self.assertIs(deproject.call_args.kwargs["aligned_depth_intrinsics"], camera.color_intrinsics)
                self.assertEqual(fit_plane.call_args.args[0].shape, (15, 3))
                self.assertEqual(save.call_count, int(answer == "y"))


class PreviewLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.viewer = RealSenseViewer(SimpleNamespace(cloud=None))
        frame = RGBDFrame(np.zeros((2, 3, 3), dtype=np.uint8), np.ones((2, 3), dtype=np.uint16), 1, 1, 1)
        self.viewer.camera = Mock(depth_scale_m=0.001, read=Mock(return_value=frame))
        self.viewer.pointcloud_config = Mock(workspace=(0, 0, 0, 1, 1, 1))
        self.viewer.ensure_cloud_transform = Mock(return_value=np.eye(4))
        self.processor = Mock(process=Mock(return_value=np.zeros((1, 6))))
        self.viewer.ensure_processor = Mock(return_value=self.processor)

    def test_toggle_switch_close_and_reopen_share_one_camera(self):
        image = Mock()
        image.poll_events.side_effect = [["p"], ["s"], [], ["p"], ["q"]]
        first = Mock(poll_events=Mock(side_effect=[[], ["closed"]]))
        second = Mock(poll_events=Mock(return_value=[]))
        with patch("dexmani_real.utils.visualization.image.ImageViewer", return_value=image), \
                patch("dexmani_real.utils.visualization.pointcloud.PointCloudViewer", side_effect=[first, second]), \
                patch("dexmani_real.perception.pointcloud.build_raw_point_cloud", return_value=np.zeros((1, 6))) as raw:
            self.viewer.capture()
        self.processor.process.assert_called_once()
        self.assertEqual(raw.call_count, 2)
        self.assertEqual(self.viewer.camera.read.call_count, 4)
        first.close.assert_not_called()  # closed 的释放责任属于 PointCloudViewer.poll_events。
        second.close.assert_called_once()
        image.close.assert_called_once()

    def test_rgbd_only_needs_no_extrinsics_or_pointcloud_processor(self):
        image = Mock(poll_events=Mock(side_effect=[[], ["q"]]))
        with patch("dexmani_real.utils.visualization.image.ImageViewer", return_value=image):
            self.viewer.capture()
        self.viewer.ensure_cloud_transform.assert_not_called()
        self.viewer.ensure_processor.assert_not_called()

    def test_primary_capture_error_survives_window_cleanup_failure(self):
        self.viewer.args.cloud = "raw"
        self.viewer.camera.read.side_effect = ValueError("capture failed")
        image = Mock(poll_events=Mock(return_value=[]))
        cloud = Mock(close=Mock(side_effect=RuntimeError("close failed")))
        with patch("dexmani_real.utils.visualization.image.ImageViewer", return_value=image), \
                patch("dexmani_real.utils.visualization.pointcloud.PointCloudViewer", return_value=cloud), \
                self.assertLogs("scripts.tools.view_realsense", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "capture failed"):
                self.viewer.capture()
        cloud.close.assert_called_once()
        image.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()

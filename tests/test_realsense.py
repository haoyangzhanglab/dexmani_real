"""只模拟 SDK 帧，不连接相机。"""

import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.sensor import CameraConfig
from dexmani_real.sensor.realsense import RealSenseCamera, run_camera


class RealSenseFrameTests(unittest.TestCase):
    def setUp(self):
        self.camera = RealSenseCamera(CameraConfig(width=3, height=2))
        self.camera.color_intrinsics = SimpleNamespace(width=3, height=2)

    def queue_frame(self, depth, color):
        frames = Mock()
        frames.get_color_frame.return_value.get_data.return_value = color
        self.camera.aligner = Mock()
        self.camera.aligner.process.return_value.get_depth_frame.return_value.get_data.return_value = depth
        now = time.monotonic_ns()
        self.stamps = (now - 2_000, now - 1_000)
        self.camera.pending_frame = frames, (7, 9), self.stamps

    def test_read_preserves_source_and_owns_rgb_and_depth(self):
        depth = np.arange(6, dtype=np.uint16).reshape(2, 3)
        color = np.tile(np.array([10, 20, 30], dtype=np.uint8), (2, 3, 1))
        self.queue_frame(depth, color)
        frame = self.camera.read()
        self.assertEqual(frame.timestamp_ns, min(self.stamps))
        self.assertEqual((frame.depth_frame_number, frame.color_frame_number), (7, 9))
        np.testing.assert_array_equal(frame.rgb[0, 0], [30, 20, 10])
        self.assertTrue(frame.rgb.flags.c_contiguous)
        self.assertTrue(frame.depth.flags.c_contiguous)
        depth[:] = 0
        color[:] = 0
        self.assertEqual(frame.depth[1, 2], 5)
        self.assertEqual(frame.rgb[0, 0, 0], 30)

    def test_wrong_depth_type_or_grid_is_rejected_before_conversion(self):
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        for depth in (np.zeros((2, 3), dtype=np.float32), np.zeros((3, 2), dtype=np.uint16)):
            with self.subTest(shape=depth.shape, dtype=depth.dtype):
                self.queue_frame(depth, color)
                with self.assertRaisesRegex(ValueError, "aligned camera depth"):
                    self.camera.read()

    def test_wrong_color_type_or_grid_is_rejected(self):
        depth = np.zeros((2, 3), dtype=np.uint16)
        for color in (np.zeros((2, 3, 3), dtype=np.float32), np.zeros((2, 3), dtype=np.uint8)):
            with self.subTest(shape=color.shape, dtype=color.dtype):
                self.queue_frame(depth, color)
                with self.assertRaisesRegex(ValueError, "camera color"):
                    self.camera.read()

    def test_new_color_does_not_refresh_old_depth_source(self):
        camera = self.camera
        camera.last_identity = (7, 9)
        camera.received_ns = [time.monotonic_ns() - 10_000, time.monotonic_ns() - 5_000]
        depth_stamp = camera.received_ns[0]
        frames = Mock()
        frames.get_depth_frame.return_value.get_frame_number.return_value = 7
        frames.get_color_frame.return_value.get_frame_number.return_value = 10
        queued = Mock()
        queued.as_frameset.return_value = frames
        camera.receive_frame(queued)
        self.assertEqual(camera.pending_frame[1], (7, 10))
        self.assertEqual(camera.pending_frame[2][0], depth_stamp)
        self.assertGreater(camera.pending_frame[2][1], depth_stamp)

    def test_callback_failure_is_reported_by_reader(self):
        queued = Mock()
        queued.as_frameset.side_effect = RuntimeError("invalid SDK frameset")
        self.camera.receive_frame(queued)
        with self.assertRaisesRegex(RuntimeError, "invalid SDK frameset"):
            self.camera.read()

    def test_unexpected_camera_profile_is_rejected_and_disconnected(self):
        device = Mock()
        device.get_info.side_effect = {"serial": "offline", "name": "D435", "product": "D400"}.__getitem__
        device.first_depth_sensor.return_value.supports.return_value = False
        device.first_color_sensor.return_value.supports.return_value = False
        device.first_depth_sensor.return_value.get_depth_scale.return_value = 0.001
        profile = Mock()
        profile.get_device.return_value = device
        profile.get_stream.return_value.as_video_stream_profile.return_value.get_intrinsics.return_value = SimpleNamespace(
            width=4, height=2, fx=1., fy=1., ppx=0., ppy=0., model="none", coeffs=(0.,) * 5,
        )
        pipeline = Mock(start=Mock(return_value=profile))
        sdk = SimpleNamespace(
            context=Mock(return_value=Mock(query_devices=Mock(return_value=[device]))),
            config=Mock(), pipeline=Mock(return_value=pipeline),
            camera_info=SimpleNamespace(serial_number="serial", name="name", product_line="product"),
            stream=SimpleNamespace(depth="depth", color="color"),
            format=SimpleNamespace(z16="z16", bgr8="bgr8"),
            option=SimpleNamespace(emitter_enabled="emitter", visual_preset="preset", auto_exposure_priority="exposure"),
        )
        original_sdk = sys.modules.get("pyrealsense2")
        sys.modules["pyrealsense2"] = sdk
        try:
            with self.assertRaisesRegex(ValueError, "color profile differs"):
                self.camera.connect()
        finally:
            if original_sdk is None:
                sys.modules.pop("pyrealsense2", None)
            else:
                sys.modules["pyrealsense2"] = original_sdk
        pipeline.stop.assert_called_once()
        self.assertIsNone(self.camera.pipeline)

    def test_worker_startup_failure_still_attempts_device_cleanup(self):
        failure, cleanup_error = ValueError("startup failed"), RuntimeError("disconnect failed")
        camera = Mock(connect=Mock(side_effect=failure), disconnect=Mock(side_effect=cleanup_error))
        with patch("dexmani_real.sensor.realsense.RealSenseCamera", return_value=camera), \
                self.assertLogs("dexmani_real.sensor.realsense", level="ERROR"):
            with self.assertRaisesRegex(ValueError, "startup failed") as captured:
                run_camera(None, CameraConfig())
        self.assertIs(captured.exception, failure)
        self.assertEqual(failure.cleanup_errors, [("camera_disconnect", cleanup_error)])
        camera.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()

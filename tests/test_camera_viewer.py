"""单相机预览的两种点云、曝光恢复与来源快照；不连接设备。"""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

from dexmani_real.config.real import RealConfig
from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.sensor.realsense import RGBDFrame
from scripts.tools import view_realsense as viewer


class CameraViewerTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            config=None, camera_calibration=None, cloud="raw", calibrate_table=False,
            save_dir=None, benchmark_frames=0,
        )
        self.runtime = RealConfig()
        self.runtime = replace(self.runtime, camera=replace(self.runtime.camera, width=16, height=16))
        self.intrinsics = CameraIntrinsics(16, 16, 16., 16., 8., 8., "none", (0.,) * 5)
        self.frame = RGBDFrame(
            np.zeros((16, 16, 3), np.uint8), np.full((16, 16), 500, np.uint16), 10, 11, 123456789,
        )
        self.camera = Mock(
            active_serial="fake-camera", color_intrinsics=self.intrinsics, depth_scale_m=.001,
        )
        self.camera.get_device_info.return_value = dict(name="fake", serial="fake-camera", firmware="test")
        self.camera.read.return_value = self.frame
        self.image = Mock()
        self.image.poll_events.side_effect = [[], ["s"], ["q"]]
        self.cloud = Mock()
        self.cloud.poll_events.return_value = []
        self.processed = np.ones((4, 6), np.float32)
        self.raw = np.zeros((8, 6), np.float32)
        self.processor = Mock()
        self.processor.process.return_value = self.processed

    def run_viewer(self, *, restore_error=None):
        def set_priority(camera, value):
            if value == 1. and restore_error is not None:
                raise restore_error
            return value

        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(viewer, "load_config", return_value=self.runtime))
            stack.enter_context(patch.object(viewer, "configure_logging"))
            camera_factory = stack.enter_context(patch("dexmani_real.sensor.realsense.RealSenseCamera", return_value=self.camera))
            stack.enter_context(patch.object(viewer, "get_ae_priority", return_value=1.))
            exposure = stack.enter_context(patch.object(viewer, "set_ae_priority", side_effect=set_priority))
            self.exposure = exposure
            stack.enter_context(patch.object(viewer, "ImageViewer", return_value=self.image))
            stack.enter_context(patch.object(viewer, "PointCloudViewer", return_value=self.cloud))
            stack.enter_context(patch.object(viewer, "load_camera_extrinsics", return_value=np.eye(4)))
            stack.enter_context(patch.object(viewer, "load_table_plane", return_value=(0., 0., 1., 0.)))
            processor_factory = stack.enter_context(patch.object(viewer, "PointCloudProcessor", return_value=self.processor))
            raw = stack.enter_context(patch.object(viewer, "build_raw_point_cloud", return_value=self.raw))
            benchmark = stack.enter_context(patch.object(viewer, "benchmark_production_pipeline"))
            calibration = stack.enter_context(patch.object(viewer, "calibrate_table_plane", return_value=(0., 0., 1., -.02)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            self.assertEqual(viewer.RealSenseViewer(self.args).run(), 0)
            camera_factory.assert_called_once()
        return exposure, processor_factory, raw, benchmark, calibration

    def test_one_camera_connection_serves_both_cloud_views_and_restores_exposure(self):
        exposure, factory, raw, benchmark, calibration = self.run_viewer()
        self.camera.connect.assert_called_once()
        self.camera.disconnect.assert_called_once()
        exposure.assert_has_calls([call(self.camera, 0.), call(self.camera, 1.)])
        raw.assert_called_once()
        self.processor.process.assert_called_once()
        updates = [entry.args[0] for entry in self.cloud.update.call_args_list]
        self.assertTrue(any(item is self.raw for item in updates))
        self.assertTrue(any(item is self.processed for item in updates))
        benchmark.assert_not_called()
        calibration.assert_not_called()
        self.image.close.assert_called_once()
        self.cloud.close.assert_called_once()

    def test_bad_rgb_remains_the_primary_error_and_camera_is_closed(self):
        self.frame = replace(self.frame, rgb=self.frame.rgb.astype(np.float32))
        self.camera.read.return_value = self.frame
        with self.assertLogs(viewer.logger, level="ERROR"):
            with self.assertRaisesRegex(ValueError, "camera RGB"):
                self.run_viewer(restore_error=OSError("restore failed"))
        self.camera.disconnect.assert_called_once()
        self.exposure.assert_has_calls([call(self.camera, 0.), call(self.camera, 1.)])
        self.image.close.assert_called_once()
        self.image.show.assert_not_called()

    def test_explicit_stages_reuse_connection_and_save_original_source_times(self):
        with tempfile.TemporaryDirectory() as directory:
            self.args.save_dir = Path(directory)
            self.args.calibrate_table = True
            self.args.benchmark_frames = 3
            exposure, factory, raw, benchmark, calibration = self.run_viewer()
            calibration.assert_called_once()
            benchmark.assert_called_once_with(camera=self.camera, processor=self.processor, frame_count=3)
            self.assertEqual(factory.call_args.kwargs["table_plane_abcd"], (0., 0., 1., -.02))
            self.camera.connect.assert_called_once()
            self.camera.disconnect.assert_called_once()
            snapshot = next(Path(directory).iterdir())
            metadata = json.loads((snapshot / "metadata.json").read_text())
            self.assertEqual(metadata["source_timestamp_ns"], 123456789)
            self.assertEqual(metadata["depth_frame_number"], 10)
            self.assertEqual(metadata["color_frame_number"], 11)
            self.assertEqual(metadata["table_plane_source"], "calibrated_this_run")
            np.testing.assert_array_equal(np.load(snapshot / "rgb.npy"), self.frame.rgb)
            np.testing.assert_array_equal(np.load(snapshot / "processed_point_cloud.npy"), self.processed)


if __name__ == "__main__":
    unittest.main()

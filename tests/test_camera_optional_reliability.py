"""相机来源时间、选中外参与诊断收尾的无设备回归。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.sensor import CameraConfig
from dexmani_real.sensor import realsense
from dexmani_real.utils.camera_calibrate.extrinsics import (
    load_camera_extrinsics,
    save_camera_calibration,
)
from scripts.tools import view_pointcloud


class Frames:
    def __init__(self, depth_number, color_number):
        self.depth = SimpleNamespace(
            get_frame_number=lambda: depth_number,
            get_data=lambda: np.ones((2, 2), dtype=np.uint16),
        )
        self.color = SimpleNamespace(
            get_frame_number=lambda: color_number,
            get_data=lambda: np.zeros((2, 2, 3), dtype=np.uint8),
        )

    def as_frameset(self):
        return self

    def get_depth_frame(self):
        return self.depth

    def get_color_frame(self):
        return self.color

    def keep(self):
        pass


class CameraReliabilityTests(unittest.TestCase):
    def test_callback_time_survives_queue_wait_and_rgb_only_update(self):
        camera = realsense.RealSenseCamera(CameraConfig())
        camera.aligner = SimpleNamespace(process=lambda frames: frames)
        with patch.object(realsense.time, "monotonic_ns", return_value=1_000_000_000):
            camera.receive_frame(Frames(1, 1))
        with patch.object(realsense.time, "monotonic_ns", return_value=1_500_000_000):
            self.assertEqual(camera.read().timestamp_ns, 1_000_000_000)
        with patch.object(realsense.time, "monotonic_ns", return_value=1_800_000_000):
            camera.receive_frame(Frames(1, 2))
        with patch.object(realsense.time, "monotonic_ns", return_value=1_900_000_000):
            frame = camera.read()
        self.assertEqual(frame.color_frame_number, 2)
        self.assertEqual(frame.timestamp_ns, 1_000_000_000)
        with patch.object(realsense.time, "monotonic_ns", return_value=3_100_000_000):
            camera.receive_frame(Frames(1, 3))
            with self.assertRaisesRegex(TimeoutError, "depth"):
                camera.read()

    def test_callback_failure_reaches_owner(self):
        camera = realsense.RealSenseCamera(CameraConfig())
        camera.receive_frame(Frames(2, 2))
        camera.receive_frame(Frames(1, 3))
        with self.assertRaisesRegex(RuntimeError, "moved backward"):
            camera.read()

    def test_selected_calibration_does_not_validate_unrelated_entry(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            unrelated = {"T_base_from_camera": "broken", "note": "preserve"}
            path.write_text(json.dumps({
                "good": {"T_base_from_camera": np.eye(4).tolist()},
                "other": unrelated,
            }))
            np.testing.assert_array_equal(load_camera_extrinsics("good", path), np.eye(4))
            with self.assertRaises(ValueError):
                load_camera_extrinsics("other", path)
            save_camera_calibration(np.eye(4), "good", path)
            self.assertEqual(json.loads(path.read_text())["other"], unrelated)
            path.write_text("{broken")
            with self.assertRaises(json.JSONDecodeError):
                load_camera_extrinsics("good", path)

    def test_depth_sampling_and_benchmark_ignore_rgb_only_frames(self):
        frames = [
            SimpleNamespace(depth_frame_number=n, depth=np.ones((2, 2)), rgb=None)
            for n in (1, 1, 2)
        ]
        camera = SimpleNamespace(read=Mock(side_effect=frames), depth_scale_m=0.001)
        processor = SimpleNamespace(process=Mock(return_value=np.ones((1, 6))))
        view_pointcloud.benchmark_production_pipeline(
            camera=camera, processor=processor, frame_count=2,
        )
        self.assertEqual(camera.read.call_count, 3)
        self.assertEqual(processor.process.call_count, 2)

        camera.read = Mock(side_effect=frames)
        fit = SimpleNamespace(
            plane_abcd=(0, 0, 1, 0), inlier_points=2, evaluated_points=2,
            inlier_ratio=1, rms_residual_m=0, tilt_deg=0,
        )
        with (
            patch("builtins.input", side_effect=("", "n")),
            patch.object(view_pointcloud, "aligned_depth_points_in_base",
                         return_value=np.zeros((1, 3))),
            patch.object(view_pointcloud, "fit_table_plane", return_value=fit) as fit_plane,
        ):
            view_pointcloud.calibrate_table_plane(
                camera=camera, intrinsics=None, T_xarm_base_from_color=np.eye(4),
                config=SimpleNamespace(workspace=(-1, -1, -1, 1, 1, 1)),
                plane_path=Path("unused.json"), frame_count=2,
            )
        self.assertEqual(camera.read.call_count, 3)
        self.assertEqual(fit_plane.call_args.args[0].shape, (2, 3))

    def test_diagnostic_restores_exposure_on_normal_and_failed_capture(self):
        runtime = SimpleNamespace(
            camera=SimpleNamespace(validate=lambda: None),
            pointcloud=SimpleNamespace(remove_table=False),
        )
        args = SimpleNamespace(save_dir=None, config=None, camera_calibration=None)
        for failed, restore_failed in ((False, False), (True, False), (True, True), (False, True)):
            with self.subTest(failed=failed, restore_failed=restore_failed):
                priority = [1.0]
                changes = []

                def set_option(option, value):
                    priority[0] = value
                    changes.append(value)
                    if restore_failed and value == 1.0:
                        raise RuntimeError("restore failed")

                camera = SimpleNamespace(
                    connect=Mock(), disconnect=Mock(),
                    color_sensor=SimpleNamespace(
                        supports=lambda option: True,
                        get_option=lambda option: priority[0], set_option=set_option,
                    ),
                )
                preview = SimpleNamespace(
                    show=Mock(), poll_events=lambda: ["q"], close=Mock(),
                )
                with (
                    patch("builtins.input", return_value="n"),
                    patch.object(view_pointcloud, "configure_logging"),
                    patch.object(view_pointcloud, "load_config", return_value=runtime),
                    patch.object(realsense, "RealSenseCamera", return_value=camera),
                    patch.object(view_pointcloud, "print_device_info", return_value={}),
                    patch.object(view_pointcloud, "print_depth_stats"),
                    patch.object(view_pointcloud, "rgbd_panel"),
                    patch.object(view_pointcloud, "ImageViewer", return_value=preview),
                    patch.object(view_pointcloud, "capture_frame",
                                 side_effect=ValueError("capture failed") if failed else None,
                                 return_value=(None, None, None)),
                ):
                    if failed:
                        with self.assertRaisesRegex(ValueError, "capture failed"):
                            view_pointcloud.run_pointcloud_diagnostic(args)
                    elif restore_failed:
                        with self.assertRaisesRegex(RuntimeError, "restore failed"):
                            view_pointcloud.run_pointcloud_diagnostic(args)
                    else:
                        self.assertEqual(view_pointcloud.run_pointcloud_diagnostic(args), 0)
                self.assertEqual(changes, [0.0, 1.0])
                camera.disconnect.assert_called_once()


if __name__ == "__main__":
    unittest.main()

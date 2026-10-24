"""合成显示 API、诊断业务和无设备 CLI；不启动真实采集或输入监听。"""

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.data.recorder import AsyncEpisodeRecorder
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.sensor.camera.geometry import CameraIntrinsics
from dexmani_real.visualization import (
    log_camera_geometry, log_points, log_rgbd, log_series, log_workspace, viewer_blueprint,
)
from tests.offline_fixtures import make_row, recording_metadata


@unittest.skipUnless(importlib.util.find_spec("rerun"), "requires visualization extra")
class VisualizationTests(unittest.TestCase):
    def test_episode_reads_table_only_when_cloud_can_be_computed(self):
        import rerun as rr
        from scripts import visualize_episode as viewer

        runtime = ExperimentConfig()
        for extrinsics, enabled, remove_table in (
            (False, True, True), (True, False, True), (True, True, False), (True, True, True),
        ):
            with self.subTest(extrinsics=extrinsics, enabled=enabled, remove_table=remove_table):
                metadata = dict(
                    recording_metadata(), camera_payload_mode="depth_to_color_aligned_rgbd",
                    camera_color_intrinsics=np.array([[8, 0, 4], [0, 8, 4], [0, 0, 1]]),
                    camera_color_distortion_model="none", camera_color_distortion_coeffs=np.zeros(5),
                )
                if extrinsics:
                    metadata["camera_T_xarm_base_from_color"] = np.eye(4)
                with tempfile.TemporaryDirectory() as directory:
                    recorder = AsyncEpisodeRecorder(directory, control_hz=10)
                    recorder.start_episode(
                        task_label="offline", collection_source="teleop", metadata=metadata,
                    )
                    row = make_row()
                    row.arm["qpos"] = row.hand["qpos"] = np.nan
                    recorder.add_frame(row)
                    saved = recorder.save_episode("operator", stop_result="confirmed")
                    rr.init(
                        "offline-episode", recording_id=f"{extrinsics}-{enabled}-{remove_table}",
                        spawn=False, strict=True,
                    )
                    memory = rr.memory_recording()
                    with patch.object(rr, "init"), patch.object(
                        viewer, "resolve_table_plane", return_value=(0, 0, 1, 0),
                    ) as table, patch.object(viewer, "build_point_cloud", return_value=None) as cloud:
                        visualizer = viewer.EpisodeVisualizer(
                            str(saved),
                            handbase_position_eef_m=runtime.hand.T_eef_handbase_pos_xyz,
                            handbase_quat_eef_wxyz=runtime.hand.T_eef_handbase_quat_wxyz,
                            fingertip_link_names=runtime.hand.fingertip_link_names,
                            table_config=runtime.environment.table,
                            pointcloud_config=(
                                replace(runtime.pointcloud, remove_table=remove_table)
                                if enabled else None
                            ),
                        )
                        try:
                            visualizer.log_step(0)
                            self.assertEqual(table.call_count, int(extrinsics and enabled and remove_table))
                            self.assertEqual(cloud.call_count, int(extrinsics and enabled))
                            self.assertGreater(len(memory.drain_as_bytes()), 0)
                        finally:
                            visualizer.close()

    def test_sdk_serialization_rgb_depth_transform_missing_data_and_layout(self):
        import rerun as rr

        rr.init("dexmani-offline-unit", spawn=False, strict=True)
        memory = rr.memory_recording()
        self.addCleanup(rr.disconnect)
        rgb = np.array([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [0, 0, 0]]], dtype=np.uint8)
        depth = np.full((2, 2), 500, dtype=np.uint16)
        intrinsics = CameraIntrinsics(2, 2, 2, 2, 1, 1, "none", (0, 0, 0, 0, 0))
        transform = np.eye(4)
        transform[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
        transform[:3, 3] = [0.1, 0.2, 0.3]
        with patch.object(rr, "log", wraps=rr.log) as log:
            log_camera_geometry(intrinsics)
            self.assertFalse(any(isinstance(call.args[1], rr.Transform3D) for call in log.call_args_list))
            log.reset_mock()
            log_camera_geometry(intrinsics, transform)
            geometry = log.call_args.args[1]
            np.testing.assert_allclose(geometry.translation.as_arrow_array().to_pylist(), [[0.1, 0.2, 0.3]])
            np.testing.assert_allclose(
                geometry.mat3x3.as_arrow_array().to_pylist(), [transform[:3, :3].T.ravel()],
            )
            log_rgbd(rgb, depth, depth_scale_m=0.002, depth_range_m=(0.3, 2.5))
            images = {call.args[0]: call.args[1] for call in log.call_args_list}
            np.testing.assert_array_equal(images["camera/color/rgb"].buffer.as_arrow_array().to_pylist()[0], rgb.ravel())
            self.assertEqual(images["depth/image"].meter.as_arrow_array().to_pylist(), [500.0])
            self.assertEqual(images["depth/image"].depth_range.as_arrow_array().to_pylist(), [[150., 1250.]])
            cloud = np.array([[0.4, 0.1, 0.3, 1, 0, 0]], dtype=np.float32)
            log_points("pcd", cloud)
            points = log.call_args.args[1]
            np.testing.assert_allclose(points.positions.as_arrow_array().to_pylist(), cloud[:, :3])
            self.assertEqual(points.colors.as_arrow_array().to_pylist(), [0xff0000ff])
            for missing in (None, np.empty((0, 6)), np.full((1, 3), np.nan)):
                log_points("pcd", missing)
                self.assertIsInstance(log.call_args.args[1], rr.Clear)
            with self.assertRaises(ValueError):
                log_points("pcd", np.zeros((2, 4)))
            log_points("eef", np.array([[0.4, 0.1, 0.3]]), colors=[255, 255, 255], radii=0.02)
            log_series("state/q", [0.1, 0.2], labels=["q0", "q1"])
            log_workspace((-1, -1, -1, 1, 1, 1))
            with self.assertRaises(ValueError):
                log_rgbd(rgb, depth, depth_scale_m=0)
        for camera_in_base in (False, True):
            blueprint = viewer_blueprint([("State", ["state/q"])], camera_in_base=camera_in_base)
            with patch.object(rr, "log", wraps=rr.log) as log:
                rr.send_blueprint(blueprint)
            self.assertIsNotNone(blueprint)
        self.assertGreater(len(memory.drain_as_bytes()), 1000)

    def test_camera_exposure_original_restored_on_capture_failure(self):
        # 在 sys.modules 替换前初始化本地扩展，避免 mock 回滚触发 PyO3 二次加载。
        importlib.import_module("rerun")
        from scripts.tools import view_realsense as viewer

        for original in (None, 1.0):
            with self.subTest(original=original):
                camera = Mock()
                camera_type = Mock(return_value=camera)
                camera_type.list_cameras.return_value = [dict(name="fake", serial="fake", firmware="0", product_line="fake")]
                args = SimpleNamespace(config=None, camera_calibration=None)
                with patch.dict(sys.modules, {"dexmani_real.sensor.camera.realsense": SimpleNamespace(RealSenseCamera=camera_type)}), patch.object(
                    viewer, "get_ae_priority", return_value=original,
                ), patch.object(viewer, "set_ae_priority", side_effect=lambda camera, value: value) as setting, patch.object(
                    viewer.RealSenseExample, "capture", side_effect=RuntimeError("capture failed"),
                ), patch.object(viewer, "configure_logging"), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(viewer.RealSenseExample(args).run(), 1)
                self.assertEqual([call.args[1] for call in setting.call_args_list], [] if original is None else [0.0, 1.0])
                camera.disconnect.assert_called_once()
                camera_type.assert_called_once_with(ExperimentConfig().camera, auto_exposure_priority=None)

    def test_live_pointcloud_is_demand_driven_and_missing_geometry_disables_it(self):
        from scripts.tools import view_realsense as viewer
        import rerun as rr

        for key, mode in (
            ("p", "processed"), ("s", "raw"), ("c", "disabled"), ("p", "missing_geometry"),
        ):
            with self.subTest(mode=mode):
                owner = viewer.RealSenseExample(SimpleNamespace())
                owner.rr = rr
                owner.runtime = ExperimentConfig()
                owner.production = owner.runtime.pointcloud
                frame = SimpleNamespace(rgb=np.zeros((2, 2, 3), dtype=np.uint8),
                                        depth_aligned_to_color_raw=np.ones((2, 2), dtype=np.uint16),
                                        timestamp_ns=1234)
                owner.camera = Mock()
                owner.camera.get_color_intrinsics.return_value = CameraIntrinsics(2, 2, 2, 2, 1, 1, "none", (0,) * 5)
                owner.camera.get_depth_scale.return_value = 0.001
                owner.camera.read.return_value = frame
                owner.load_pointcloud_geometry = Mock(return_value=(np.eye(4), None))
                if mode == "missing_geometry":
                    owner.load_pointcloud_geometry.side_effect = ValueError("missing calibration")
                keyboard = Mock(healthy=True, estop_latched=False)
                keyboard.pop_event.side_effect = (["s", "p", None, "q"] if mode == "raw" else [key, None, "q"])
                cloud = np.zeros((1, 6), dtype=np.float32)
                with patch.object(viewer, "KeyboardInput", return_value=keyboard), patch.object(rr, "init"), patch.object(
                    rr, "send_blueprint",
                ), patch.object(rr, "disconnect"), patch.object(viewer, "build_point_cloud", return_value=cloud) as processed, patch.object(
                    viewer, "build_raw_point_cloud", return_value=cloud,
                ) as raw, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(owner.capture(allow_exposure_changes=False), "user_exit")
                self.assertEqual(processed.call_count, int(mode == "processed"))
                self.assertEqual(raw.call_count, int(mode == "raw"))
                keyboard.stop.assert_called_once()

    def test_table_publication_still_requires_confirmation(self):
        from scripts.tools import view_pointcloud as viewer

        fit = SimpleNamespace(plane_abcd=(0, 0, 1, 0), inlier_points=100, evaluated_points=100,
                              inlier_ratio=1, rms_residual_m=0, tilt_deg=0)
        camera = Mock()
        camera.read.return_value = SimpleNamespace(depth_aligned_to_color_raw=np.ones((2, 2), dtype=np.uint16))
        config = ExperimentConfig().pointcloud
        center = (np.asarray(config.workspace[:3]) + np.asarray(config.workspace[3:])) / 2
        for confirmation in ("n", "y"):
            with patch("builtins.input", side_effect=["", confirmation]), patch.object(viewer, "fit_table_plane", return_value=fit), patch.object(
                viewer, "aligned_depth_points_in_base", return_value=np.tile(center, (100, 1)),
            ), patch.object(viewer, "publish_table_plane") as publish, contextlib.redirect_stdout(io.StringIO()):
                plane = viewer.calibrate_table_plane(
                    camera=camera, intrinsics=None, T_xarm_base_from_color=np.eye(4),
                    config=config, plane_path=Path("/tmp/not-written-table.json"), frame_count=5,
                )
            self.assertEqual(plane, fit.plane_abcd)
            self.assertEqual(publish.call_count, int(confirmation == "y"))


class InputAndCliTests(unittest.TestCase):
    def test_diagnostic_events_do_not_change_default_operator_keys(self):
        fake_keyboard = SimpleNamespace(Key=SimpleNamespace(esc="esc", up="up", down="down", left="left",
                                                            right="right", space="space", enter="enter", backspace="backspace"))
        key = SimpleNamespace(char="s")
        for diagnostic in (False, True):
            keys = KeyboardInput(capture_commands=not diagnostic, capture_raw_events=diagnostic,
                                 raw_event_keys=frozenset({"s", "a"}))
            keys.callbacks_active = True
            press, release = keys.callbacks(fake_keyboard)
            press(key)
            press(key)
            if diagnostic:
                self.assertEqual(keys.pop_event(), "s")
                self.assertIsNone(keys.pop_event())
                self.assertEqual(keys.poll(timeout=0), [])
            else:
                self.assertEqual(keys.poll(timeout=0), [OperatorCommand.STOP])
                self.assertIsNone(keys.pop_event())
            release(key)
            self.assertFalse(keys.is_pressed("s"))

    def test_help_and_print_config_do_not_import_device_sdks_or_rerun(self):
        root = Path(__file__).resolve().parents[1]
        entries = [("scripts/" + name, "--help") for name in (
            "collect_teleop.py", "keyboard_teleop.py", "calibrate_camera.py", "run_policy.py",
            "replay_episode.py", "tools/view_realsense.py", "tools/view_pointcloud.py", "visualize_episode.py",
        )] + [("scripts/collect_teleop.py", "--print-config"), ("scripts/run_policy.py", "--print-config")]
        code = '''import importlib.abc, runpy, sys
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pyrealsense2', 'xarm', 'xhand', 'hand_tracking_sdk', 'pynput', 'rerun', 'dexmani_policy'}:
            raise AssertionError('unexpected optional/device import: ' + fullname)
sys.meta_path.insert(0, Deny())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        for entry in entries:
            with self.subTest(entry=entry):
                result = subprocess.run([sys.executable, "-c", code, *entry], cwd=root, capture_output=True, text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stderr)

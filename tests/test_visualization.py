"""显示恢复的离线回归；不连接相机或机械臂。"""

import json
import unittest
from concurrent.futures import CancelledError
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np

from dexmani_real.calibration.camera.aruco import ARUCO_DICT, MARKER_ID, ArucoSampler
from dexmani_real.calibration.camera.calibrate import CameraCalibrator
from dexmani_real.config import load_config
from dexmani_real.geometry.camera import CameraIntrinsics
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.runtime.home import HomeResult
from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState
from dexmani_real.utils.visualization.image import (
    colorize_depth,
    draw_aruco,
    rgbd_panel,
)
from scripts.tools import view_pointcloud, view_realsense
from scripts.visualize_episode import EpisodeVisualizer

INTRINSICS = CameraIntrinsics(4, 3, 3., 3., 2., 1., 'none', (0.,) * 5)
CLOUD = np.ones((2, 6), dtype=np.float32)
FRAME = SimpleNamespace(
    rgb=np.zeros((3, 4, 3), dtype=np.uint8),
    depth=np.full((3, 4), 700, dtype=np.uint16),
)


class ImageTests(unittest.TestCase):
    def test_depth_and_rgb_are_not_modified(self):
        depth = np.array([[0., np.nan, np.inf, .3, 1.5]])
        before = depth.copy()
        gray = colorize_depth(depth, (.3, 1.5), colormap='gray')
        np.testing.assert_array_equal(gray[0, :3], 0)
        np.testing.assert_array_equal(gray[0, 3], 255)
        np.testing.assert_array_equal(gray[0, 4], 0)
        np.testing.assert_array_equal(depth, before)
        jet = colorize_depth(depth, (.3, 1.5))
        self.assertGreater(jet[0, 3, 2], jet[0, 3, 0])
        self.assertGreater(jet[0, 4, 0], jet[0, 4, 2])
        rgb = np.full((1, 5, 3), [255, 0, 0], dtype=np.uint8)
        panel = rgbd_panel(rgb, depth, depth_range_m=(.3, 1.5))
        np.testing.assert_array_equal(panel[:, :5], rgb[..., ::-1])
        np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])

    def test_aruco_result_and_drawing(self):
        k = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
        sampler = ArucoSampler(k, np.zeros(5))
        for marker_id in (None, MARKER_ID, MARKER_ID + 1):
            with self.subTest(marker_id=marker_id):
                rgb = np.full((480, 640, 3), 255, dtype=np.uint8)
                if marker_id is not None:
                    marker = cv2.aruco.generateImageMarker(
                        cv2.aruco.getPredefinedDictionary(ARUCO_DICT), marker_id, 180,
                    )
                    rgb[150:330, 230:410] = marker[..., None]
                original = rgb.copy()
                result = sampler.detect(rgb)
                np.testing.assert_array_equal(rgb, original)
                self.assertEqual(result.pose is not None, marker_id == MARKER_ID)
                self.assertEqual(result.ids is not None, marker_id is not None)
                canvas = rgb.copy()
                draw_aruco(canvas, result.corners, result.ids, result.pose,
                           intrinsics=k, distortion=np.zeros(5), axis_length=.05)
                self.assertEqual(np.array_equal(canvas, original), marker_id is None)


class LivePreviewTests(unittest.TestCase):
    def test_freeze_switch_reset_and_empty_cloud(self):
        runtime = load_config()
        viewer = view_realsense.RealSenseViewer(SimpleNamespace(camera_calibration=None))
        viewer.pointcloud_config = runtime.pointcloud
        viewer.table = runtime.environment.table
        viewer.camera = SimpleNamespace(color_intrinsics=INTRINSICS, depth_scale_m=.001,
                                        active_serial='test', read=Mock(return_value=FRAME))
        # F 冻结时不重算；S 解冻，P/R 清缓存，重开后复用处理器。
        events = [['p'], ['f'], ['d'], ['s'], ['s'], ['p'], ['p'], ['r'], ['p'], ['q']]
        image = Mock()
        image.poll_events.side_effect = events
        windows = []

        def new_window(**kwargs):
            window = Mock()
            window.poll_events.return_value = []
            windows.append(window)
            return window

        processor = Mock()
        processor.process.side_effect = [CLOUD, None, CLOUD, CLOUD]
        with patch.object(view_realsense, 'load_camera_extrinsics', return_value={'test': np.eye(4)}) as extrinsics, \
             patch.object(view_realsense, 'load_table_plane', return_value=(0., 0., 1., 0.)) as plane, \
             patch.object(view_realsense, 'ImageViewer', return_value=image), \
             patch.object(view_realsense, 'PointCloudViewer', side_effect=new_window), \
             patch.object(view_realsense, 'PointCloudProcessor', return_value=processor) as factory, \
             patch.object(view_realsense, 'build_raw_point_cloud', return_value=CLOUD) as raw:
            viewer.capture(allow_exposure_changes=False)
        self.assertEqual(factory.call_count, 1)
        extrinsics.assert_called_once()
        plane.assert_called_once()
        self.assertEqual(processor.process.call_count, 4)
        self.assertEqual(raw.call_count, 1)
        self.assertEqual(viewer.camera.read.call_count, 9)
        self.assertEqual(len(windows), 3)
        self.assertTrue(all(window.close.called for window in windows))
        self.assertTrue(any(call.args[0] is None for call in windows[0].update.call_args_list))
        image.close.assert_called_once()

    def test_no_frame_still_services_close(self):
        runtime = load_config()
        viewer = view_realsense.RealSenseViewer(SimpleNamespace(camera_calibration=None))
        viewer.pointcloud_config = runtime.pointcloud
        viewer.table = runtime.environment.table
        viewer.camera = SimpleNamespace(color_intrinsics=INTRINSICS, depth_scale_m=.001,
                                        active_serial='test', read=Mock(return_value=None))
        image = Mock()
        image.poll_events.side_effect = [[], ['closed']]
        with patch.object(view_realsense, 'ImageViewer', return_value=image):
            viewer.capture(allow_exposure_changes=False)
        viewer.camera.read.assert_called_once()
        image.close.assert_called_once()

    def test_exposure_restored_on_preview_failure(self):
        camera = Mock()
        camera.get_device_info.return_value = {'serial': 'test'}
        viewer = view_realsense.RealSenseViewer(SimpleNamespace(config=None))
        viewer.capture = Mock(side_effect=RuntimeError('preview failed'))
        with patch('dexmani_real.sensor.realsense.RealSenseCamera', return_value=camera), \
             patch.object(view_realsense, 'configure_logging'), \
             patch.object(view_realsense, 'list_cameras', return_value=[{'serial': 'test'}]), \
             patch.object(view_realsense, 'get_ae_priority', return_value=1.), \
             patch.object(view_realsense, 'set_ae_priority', side_effect=[0., 1.]) as set_priority:
            self.assertEqual(viewer.run(), 1)
        self.assertEqual([call.args[1] for call in set_priority.call_args_list], [0., 1.])
        camera.disconnect.assert_called_once()


class PointCloudWindowTests(unittest.TestCase):
    def test_geometry_empty_results_viewpoint_and_local_keys(self):
        import open3d as o3d

        from dexmani_real.utils.visualization.pointcloud import PointCloudViewer

        native = Mock()
        native.create_window.return_value = True
        native.poll_events.return_value = True
        with patch.object(o3d.visualization, 'VisualizerWithKeyCallback', return_value=native):
            viewer = PointCloudViewer(workspace=(0., -.5, 0., .8, .5, .8),
                                      base_from_color=np.eye(4), keys='psfrcda')
        self.assertEqual(native.add_geometry.call_count, 4)
        viewer.update(CLOUD)
        np.testing.assert_array_equal(np.asarray(viewer.cloud.points), CLOUD[:, :3])
        viewer.update(None)
        self.assertEqual(len(viewer.cloud.points), 0)
        viewer.update(CLOUD)
        native.reset_view_point.assert_called_once_with(True)
        callback = viewer.key_callback('p')
        callback(native, 2, 0)
        self.assertEqual(viewer.poll_events(), [])
        callback(native, 1, 0)
        self.assertEqual(viewer.poll_events(), ['p'])
        native.poll_events.return_value = False
        self.assertEqual(viewer.poll_events(), ['closed'])
        viewer.close()
        native.destroy_window.assert_called_once()

    def test_window_creation_failure_is_reported_and_cleaned(self):
        import open3d as o3d

        from dexmani_real.utils.visualization.pointcloud import PointCloudViewer

        native = Mock()
        native.create_window.return_value = False
        with patch.object(o3d.visualization, 'VisualizerWithKeyCallback', return_value=native):
            with self.assertRaisesRegex(RuntimeError, 'could not create'):
                PointCloudViewer(workspace=(0., -.5, 0., .8, .5, .8), base_from_color=np.eye(4))
        native.destroy_window.assert_called_once()


class CalibrationTests(unittest.TestCase):
    def test_session_failure_keeps_cause_and_closes_resources(self):
        for error, reason in (
            (RobotError('arm feedback stale'), RunEndReason.HARDWARE_FAULT),
            (ValueError('invalid target'), RunEndReason.POLICY_FAILURE),
        ):
            with self.subTest(reason=reason):
                calibrator = CameraCalibrator(load_config())
                calibrator.prepare = Mock()
                calibrator.runtime_state = SessionState()
                calibrator.keys = Mock(healthy=True)
                calibrator.robot = Mock()
                calibrator.sensors = Mock()
                calibrator.sensors.shutdown.return_value = True

                def fail_motion(state=calibrator.runtime_state, error=error):
                    state.stop_signal.clear()
                    state.begin_motion()
                    raise error

                calibrator.run_control_loop = fail_motion
                with (
                    patch('dexmani_real.calibration.camera.calibrate.snapshot_camera_parameters'),
                    patch('dexmani_real.calibration.camera.calibrate.ImageViewer'),
                    patch('builtins.print'),
                    self.assertLogs('dexmani_real.calibration.camera.calibrate', level='ERROR') as logs,
                ):
                    self.assertEqual(calibrator.run(), 1)
                self.assertIn(str(error), logs.output[0])
                self.assertEqual(calibrator.runtime_state.run_ended_reason, reason)
                self.assertEqual(calibrator.runtime_state.safety_state, SafetyState.FAULT)
                self.assertTrue(calibrator.runtime_state.stop_signal.is_set())
                calibrator.robot.close.assert_called_once()
                calibrator.keys.stop.assert_called_once()
                calibrator.sensors.shutdown.assert_called_once()
                calibrator.preview.close.assert_called_once()

    def test_home_reports_rejection_and_cancellation_reasons(self):
        for interrupted, reason, message in (
            (False, 'collision: test_obstacle', 'HOME failed'),
            (True, 'home interrupted during planning', 'HOME cancelled'),
        ):
            with self.subTest(interrupted=interrupted):
                calibrator = CameraCalibrator(load_config())
                calibrator.runtime_state = SessionState()
                calibrator.robot = Mock()
                calibrator.jog = Mock()
                calibrator.keys = Mock()
                calibrator.rate = Mock()
                calibrator.home_planner = Mock()
                with (
                    patch('dexmani_real.calibration.camera.calibrate.home_arm',
                          return_value=HomeResult(False, reason, interrupted)),
                    self.assertLogs('dexmani_real.calibration.camera.calibrate', level='INFO') as logs,
                ):
                    calibrator.run_home()
                self.assertIn('Returning home...', logs.output[0])
                self.assertIn(f'{message}: {reason}', logs.output[1])
                calibrator.jog.stop.assert_called_once()
                calibrator.keys.discard_events.assert_called_once_with('r')
                calibrator.rate.reset.assert_called_once()

    def test_quit_during_startup_keeps_calibration_exit_status(self):
        for saved in (False, True):
            with self.subTest(saved=saved):
                calibrator = CameraCalibrator(load_config())
                calibrator.prepare = Mock()
                calibrator.runtime_state = SessionState()
                calibrator.calibration_saved = saved
                calibrator.keys = Mock(healthy=True)
                calibrator.sensors = Mock()
                calibrator.sensors.shutdown.return_value = True
                calibrator.robot = Mock()

                def cancel(state=calibrator.runtime_state):
                    state.request_quit()
                    raise OperationCancelled('startup cancelled')

                calibrator.robot.connect.side_effect = cancel
                with (
                    patch('builtins.print'),
                    self.assertLogs('dexmani_real.calibration.camera.calibrate', level='INFO') as logs,
                ):
                    self.assertEqual(calibrator.run(), 0 if saved else 2)
                self.assertIn('startup cancelled', logs.output[0])
                self.assertFalse(calibrator.runtime_state.error_state)
                self.assertEqual(calibrator.runtime_state.safety_state, SafetyState.DISARMED)
                calibrator.robot.close.assert_called_once()
                calibrator.sensors.start.assert_not_called()

    def test_close_during_capture_cancels_before_read_or_append(self):
        calibrator = CameraCalibrator(load_config())
        calibrator.preview = Mock()
        calibrator.preview.poll_events.return_value = ['closed']
        calibrator.runtime_state = SessionState()
        calibrator.keys = Mock()
        calibrator.keys.is_pressed.return_value = False
        calibrator.robot = Mock()
        with self.assertRaises(CancelledError):
            calibrator.read_capture_feedback()
        self.assertTrue(calibrator.runtime_state.quit_requested)
        calibrator.robot.read_state.assert_not_called()
        self.assertEqual(calibrator.samples, [])

    def test_window_close_cancels_home_and_latches_once(self):
        from dexmani_real.runtime.home import home_arm

        runtime = load_config()
        calibrator = CameraCalibrator(runtime)
        calibrator.preview = Mock()
        calibrator.preview.poll_events.return_value = ["closed"]
        calibrator.runtime_state = SessionState()
        calibrator.runtime_state.arm()
        robot = Mock(arm_config=runtime.arm, hand_config=runtime.hand, stop_required=False)
        planner = Mock()
        result = home_arm(calibrator.runtime_state, robot, planner,
                          cancel_requested=calibrator.cancel_requested)
        self.assertTrue(result.interrupted)
        robot.service_feedback.assert_not_called()
        planner.plan.assert_not_called()
        generation = calibrator.runtime_state.run_id
        self.assertTrue(calibrator.cancel_requested())
        self.assertEqual(calibrator.runtime_state.run_id, generation)
        calibrator.preview.poll_events.assert_called_once()

    def test_window_close_at_publication_keeps_existing_calibration(self):
        from dexmani_real.calibration.camera.hand_eye import HandEyeResult

        with TemporaryDirectory() as directory:
            target = Path(directory) / "cameras.json"
            original = json.dumps({"test": {"T_base_from_camera": np.eye(4).tolist()}})
            target.write_text(original)
            calibrator = CameraCalibrator(load_config(), output_path=target)
            calibrator.preview = Mock()
            calibrator.preview.poll_events.side_effect = [[], [], [], ["closed"]]
            calibrator.runtime_state = SessionState()
            calibrator.keys = Mock()
            calibrator.keys.is_pressed.return_value = False
            calibrator.jog = Mock()
            calibrator.robot = Mock()
            calibrator.camera = SimpleNamespace(serial="test", color=INTRINSICS)
            calibrator.samples = [(np.eye(4), np.eye(4))] * 10
            calibrator.show_preview = Mock()
            result = HandEyeResult(np.eye(4), "TEST", np.zeros(10), np.zeros(10))
            with patch('dexmani_real.calibration.camera.calibrate.solve_hand_eye', return_value=result):
                calibrator.solve_and_save()
            self.assertTrue(calibrator.runtime_state.quit_requested)
            self.assertFalse(calibrator.calibration_saved)
            self.assertEqual(target.read_text(), original)
            self.assertEqual(list(Path(directory).iterdir()), [target])

    def test_stale_preview_updates_status_without_new_image(self):
        calibrator = CameraCalibrator(load_config())
        calibrator.preview = Mock()
        calibrator.preview.poll_events.return_value = []
        calibrator.runtime_state = SessionState()
        calibrator.keys = Mock()
        calibrator.keys.is_pressed.return_value = False
        calibrator.sensors = Mock()
        calibrator.display_image = np.zeros((480, 640, 3), dtype=np.uint8)
        calibrator.preview_timestamp_ns = 1
        calibrator.marker_detected = True
        calibrator.status = "Sample removed; ENTER=recompute"
        with patch('dexmani_real.calibration.camera.calibrate.read_camera_frame', return_value=None), \
             patch('dexmani_real.calibration.camera.calibrate.draw_text') as draw:
            calibrator.show_preview()
        self.assertEqual(draw.call_args_list[0].args[1], ["MARKER STALE"])
        self.assertIn(calibrator.status, draw.call_args_list[1].args[1][0])
        calibrator.preview.show.assert_called_once()
        np.testing.assert_array_equal(calibrator.display_image, 0)

    def test_sample_change_invalidates_result_and_status(self):
        calibrator = CameraCalibrator(load_config())
        calibrator.cancel_requested = Mock(return_value=False)
        calibrator.robot = Mock()
        calibrator.samples = [(np.eye(4), np.eye(4))]
        calibrator.result = Mock(worst_sample=0)
        calibrator.status = 'SAVED'
        calibrator.handle_sample_events(['x'])
        self.assertEqual(calibrator.samples, [])
        self.assertIsNone(calibrator.result)
        self.assertIn('recompute', calibrator.status)


class DiagnosticTests(unittest.TestCase):
    def test_static_view_starts_after_disconnect(self):
        runtime = load_config(overrides={'pointcloud': {'remove_table': False}})
        camera = Mock(color_intrinsics=INTRINSICS, depth_scale_m=.001)
        preview = Mock()
        preview.poll_events.return_value = ['enter']
        cloud_view = Mock()
        cloud_view.poll_events.return_value = ['closed']

        def open_cloud(**kwargs):
            camera.disconnect.assert_called_once()
            return cloud_view

        with ExitStack() as stack:
            patches = {
                'configure_logging': Mock(),
                'load_config': Mock(return_value=runtime),
                'load_camera_extrinsics': Mock(return_value={'test': np.eye(4)}),
                'print_device_info': Mock(return_value={'serial': 'test'}),
                'capture_frame': Mock(return_value=(FRAME.rgb, FRAME.depth, FRAME.depth * .001)),
                'load_extrinsics': Mock(return_value=np.eye(4)),
                'ImageViewer': Mock(return_value=preview),
                'PointCloudViewer': Mock(side_effect=open_cloud),
                'PointCloudProcessor': Mock(return_value=Mock(process=Mock(return_value=CLOUD))),
                'benchmark_production_pipeline': Mock(),
            }
            for name, value in patches.items():
                stack.enter_context(patch.object(view_pointcloud, name, value))
            stack.enter_context(patch('builtins.input', return_value='n'))
            stack.enter_context(patch('dexmani_real.sensor.realsense.RealSenseCamera', return_value=camera))
            self.assertEqual(view_pointcloud.run_pointcloud_diagnostic(
                SimpleNamespace(save_dir=None, config=None, camera_calibration=None)), 0)
        cloud_view.update.assert_called_once_with(CLOUD)
        cloud_view.close.assert_called_once()


class EpisodeTests(unittest.TestCase):
    def test_cleanup_errors_propagate_and_disconnect_is_always_attempted(self):
        viewer = EpisodeVisualizer.__new__(EpisodeVisualizer)
        viewer.reader = Mock()
        viewer.rr = Mock()
        viewer.reader.close.side_effect = OSError("reader close failed")
        with self.assertRaisesRegex(OSError, "reader close failed"):
            viewer.close()
        viewer.rr.disconnect.assert_called_once()
        viewer.reader.close.side_effect = None
        viewer.rr.disconnect.side_effect = RuntimeError("disconnect failed")
        with self.assertRaisesRegex(RuntimeError, "disconnect failed"):
            viewer.close()

    def test_invalid_fk_is_explicitly_cleared_by_caller(self):
        viewer = EpisodeVisualizer.__new__(EpisodeVisualizer)
        viewer.rr = Mock()
        viewer.timestamps_s = np.array([0.])
        viewer.state = {"hand_fingertip": np.full((1, 5, 3), np.nan),
                        "arm_ee": np.full((1, 9), np.nan)}
        viewer.log_camera = Mock(return_value=(FRAME.rgb, FRAME.depth))
        viewer.log_pointcloud = Mock()
        with patch('scripts.visualize_episode.log_points') as log:
            viewer.log_step(0)
        self.assertEqual([call.args for call in log.call_args_list],
                         [("fingertips", None), ("eef", None)])

    def test_curves_share_definition_and_explicit_timelines(self):
        import rerun as rr

        from dexmani_real.utils.visualization.rerun import log_series

        reader = MagicMock(num_frames=3)
        reader.timestamps_s = np.array([0., .033, .067])
        reader.iter_rgb_frames.return_value = iter([FRAME.rgb] * 3)
        reader.__getitem__.side_effect = lambda key: np.stack([FRAME.depth] * 3)
        reader.read_camera.return_value = SimpleNamespace(
            T_xarm_base_from_color=None, color=INTRINSICS, depth_scale_m=.001,
        )
        arrays = {key: np.arange(3 * n).reshape(3, n).astype(float) for key, n in (
            ('arm_qpos', 7), ('hand_qpos', 12), ('action_arm_joint_target', 7),
            ('action_hand_joint_target', 12), ('hand_contact_mag', 5),
            ('hand_force_thumb', 3), ('hand_force_index', 3),
        )}
        arrays['arm_qpos'][1, 0] = np.nan
        arrays['arm_ee'] = np.zeros((3, 9))
        arrays['hand_fingertip'] = np.zeros((3, 5, 3))
        with patch('scripts.visualize_episode.RawReader', return_value=reader), \
             patch.object(EpisodeVisualizer, 'preload_state', return_value=arrays), \
             patch.object(rr, 'init'), patch.object(rr, 'send_blueprint') as blueprint, \
             patch.object(rr, 'log'), patch.object(rr, 'send_columns') as send:
            visualizer = EpisodeVisualizer('example', fingertip_link_names=(), table_config=load_config().environment.table)
        self.assertEqual(len(visualizer.series), 7)
        self.assertEqual(send.call_count, 49)
        paths = [call.args[0] for call in send.call_args_list]
        self.assertEqual(len(set(paths)), 49)
        self.assertIn('state/hand_force_index/2', paths)
        self.assertEqual(list(dict.fromkeys(row[0] for row in visualizer.series)),
                         ['State', 'Action', 'Contact Force'])
        blueprint.assert_called_once()
        by_path = {call.args[0]: call.kwargs for call in send.call_args_list}
        for _, path, values, labels in visualizer.series:
            for index in range(len(labels)):
                call = by_path[f'{path}/{index}']
                step, stamp = call['indexes']
                self.assertEqual((step.timeline, stamp.timeline), ('step', 'time'))
                np.testing.assert_array_equal(step.times, np.arange(3))
                np.testing.assert_array_equal(stamp.times, reader.timestamps_s)
                # Scalar.columns 同时包含 indicator；数值列位于末尾。
                column = list(call['columns'])[-1].as_arrow_array().flatten().to_numpy()
                np.testing.assert_array_equal(column, values[:, index])
        rr.init('visualization-regression', spawn=False, strict=True)
        memory = rr.memory_recording()
        log_series('probe', arrays['arm_qpos'], labels=[str(i) for i in range(7)],
                   steps=np.arange(3), timestamps_s=reader.timestamps_s)
        visualizer.log_recording_data()
        for step in range(3):
            visualizer.log_step(step)
        self.assertGreater(memory.num_msgs(), 0)
        self.assertEqual(visualizer.rgb_index, 2)
        rr.disconnect()


if __name__ == '__main__':
    unittest.main()

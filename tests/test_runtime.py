import json
import threading
import unittest
from contextlib import ExitStack
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config import load_experiment_config
from dexmani_real.data.raw import camera_metadata
from dexmani_real.geometry.pose import Pose
from dexmani_real.ipc.sensors import snapshot_camera_parameters
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.robot.xarm7.config import ArmConfig
from dexmani_real.robot.xhand.config import HandConfig
from dexmani_real.runtime import home, keyboard, lifecycle
from dexmani_real.runtime.jog import (
    CartesianJog,
    jog_delta,
    jog_motion_requested,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState


@dataclass(frozen=True)
class KeyCode:
    char: str | None


KEY_BACKEND = SimpleNamespace(
    KeyCode=KeyCode,
    Key=Enum("Key", "esc up down left right space enter backspace"),
)


def armed_state():
    state = SessionState()
    state.arm()
    state.stop_signal.clear()
    return state


def active_keyboard(**callbacks):
    keys = keyboard.Keyboard(**callbacks)
    keys.listener = Mock()
    keys.listener.is_alive.return_value = True
    keys.active = True
    return keys


class KeyboardTests(unittest.TestCase):
    def setUp(self):
        self.state = armed_state()
        self.keys = active_keyboard(
            estop_callback=self.state.request_emergency_stop,
            bindings={'s': self.state.request_stop, 'q': self.state.request_quit},
        )
        self.press, self.release = self.keys.callbacks(KEY_BACKEND)
        self.addCleanup(self.keys.stop)

    def test_discrete_keys_fire_once_until_release(self):
        for name in ("space", "enter", "backspace", "x", "b", "r", "d"):
            key = KEY_BACKEND.Key.__members__.get(name, KeyCode(name))
            self.press(key)
            self.press(key)
            self.assertEqual(self.keys.poll(0), [name])
            self.release(key)
            self.press(key)
            self.assertEqual(self.keys.poll(0), [name])
            self.release(key)
        self.assertFalse(self.state.stop_request)

    def test_stop_discard_and_quit_revoke_before_event_is_visible(self):
        for name in ("s", "d", "q"):
            state = armed_state()
            state.begin_motion()
            observed = []

            def stop():
                state.request_stop()
                observed.append((tuple(self.keys.events), state.can_move(run_id=state.run_id)))

            self.keys.bindings["s"] = self.keys.bindings["d"] = self.keys.bindings["q"] = stop
            self.press(KeyCode(name))
            self.assertEqual(observed, [((), False)])
            self.assertEqual(self.keys.poll(0), [name])
            self.release(KeyCode(name))

    def test_callback_failure_latches_estop_without_publishing_success(self):
        self.keys.bindings["s"] = Mock(side_effect=RuntimeError("stop failed"))
        with self.assertLogs(keyboard.logger, level="ERROR") as logs:
            self.press(KeyCode("s"))
        self.assertIn("stop failed", "\n".join(logs.output))
        self.assertEqual(self.keys.poll(0), ["esc"])
        self.assertTrue(self.state.emergency_signal.is_set())

    def test_release_and_repress_invalidate_old_proposal(self):
        self.press(KeyCode("w"))
        _, generation = self.keys.snapshot()
        self.press(KeyCode("w"))
        self.assertEqual(self.keys.snapshot()[1], generation)
        self.release(KeyCode("w"))
        self.press(KeyCode("w"))
        commit = Mock()
        self.assertIsNone(self.keys.commit_if_current(generation, commit))
        commit.assert_not_called()
        self.assertTrue(self.keys.is_pressed("w"))

    def test_listener_death_clears_held_keys_and_latches_once(self):
        self.press(KeyCode("w"))
        self.keys.poll(0)
        self.keys.listener.is_alive.return_value = False
        with self.assertLogs(keyboard.logger, level="ERROR"):
            self.assertEqual(self.keys.poll(0), ["esc"])
        self.assertEqual(self.keys.poll(0), [])
        self.assertEqual(self.keys.snapshot()[0], frozenset())
        self.assertTrue(self.state.emergency_signal.is_set())

    def test_discard_keeps_terminal_events(self):
        for name in ("b", "h", "t", "s", "q"):
            self.press(KeyCode(name))
        self.keys.discard_events("b", "h", "t")
        self.assertEqual(self.keys.poll(0), ["s", "q"])

    def test_stop_disables_late_callbacks_and_restores_terminal_on_error(self):
        self.keys.saved_termios = ["saved"]
        self.keys.listener.stop.side_effect = lambda: self.press(KeyCode("q"))
        self.keys.listener.join.side_effect = OSError("listener failed")
        with patch.object(keyboard, "restore_terminal_echo") as restore:
            with self.assertRaisesRegex(OSError, "listener failed"):
                self.keys.stop()
            restore.assert_called_once_with(["saved"])
        self.assertFalse(self.state.quit_requested)
        self.assertEqual(self.keys.poll(0), [])
        self.keys.stop()
        self.keys.listener.stop.assert_called_once()

    def test_start_is_single_use_and_startup_failure_restores_terminal(self):
        for failure in (None, RuntimeError("backend unavailable")):
            with self.subTest(failure=failure):
                listener = Mock()
                listener.start.side_effect = failure
                listener.is_alive.return_value = True
                backend = SimpleNamespace(**vars(KEY_BACKEND), Listener=Mock(return_value=listener))
                keys = keyboard.Keyboard()
                with (
                    patch.dict("sys.modules", {"pynput": SimpleNamespace(keyboard=backend)}),
                    patch.object(keyboard, "suppress_terminal_echo", return_value=["saved"]),
                    patch.object(keyboard, "restore_terminal_echo") as restore,
                ):
                    if failure:
                        with self.assertRaisesRegex(RuntimeError, "backend unavailable"):
                            keys.start()
                    else:
                        keys.start()
                        self.assertTrue(keys.healthy)
                        keys.stop()
                    restore.assert_called_once_with(["saved"])
                    with self.assertRaisesRegex(RuntimeError, "single-use"):
                        keys.start()

    def test_startup_wait_is_bounded(self):
        release = threading.Event()
        listener = Mock()
        listener.wait.side_effect = lambda: release.wait(1)
        backend = SimpleNamespace(**vars(KEY_BACKEND), Listener=Mock(return_value=listener))
        keys = keyboard.Keyboard()
        try:
            with (
                patch.dict("sys.modules", {"pynput": SimpleNamespace(keyboard=backend)}),
                patch.object(keyboard, "suppress_terminal_echo"),
                patch.object(keyboard, "restore_terminal_echo"),
                patch.object(keyboard, "LISTENER_START_TIMEOUT_S", 0.001),
            ):
                with self.assertRaisesRegex(RuntimeError, "startup timed out"):
                    keys.start()
                self.assertFalse(keys.active)
                listener.stop.assert_called_once()
        finally:
            release.set()

    def test_callback_setup_failure_restores_terminal(self):
        keys = keyboard.Keyboard()
        with (
            patch.dict("sys.modules", {"pynput": SimpleNamespace(keyboard=KEY_BACKEND)}),
            patch.object(keyboard, "suppress_terminal_echo", return_value=["saved"]),
            patch.object(keyboard, "restore_terminal_echo") as restore,
            patch.object(keys, "callbacks", side_effect=RuntimeError("invalid backend")),
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid backend"):
                keys.start()
            restore.assert_called_once_with(["saved"])
            self.assertFalse(keys.active)

    def test_terminal_restore_failure_marks_shutdown_failed(self):
        self.keys.saved_termios = ["saved"]
        with (
            patch.object(keyboard, "restore_terminal_echo", side_effect=OSError("terminal closed")),
            self.assertLogs(lifecycle.logger, level="ERROR") as logs,
        ):
            self.assertFalse(lifecycle.close_runtime(self.state, keyboard=self.keys))
        self.assertIn("terminal closed", "\n".join(logs.output))
        self.assertEqual(self.state.safety_state, SafetyState.FAULT)


class CommandRoutingTests(unittest.TestCase):
    def test_keyboard_home_tap_is_not_lost_between_control_ticks(self):
        from dexmani_real.teleop.keyboard import KeyboardTeleop

        session = object.__new__(KeyboardTeleop)
        session.runtime_state = armed_state()
        session.keys = active_keyboard()
        self.addCleanup(session.keys.stop)
        session.keys.handle_key("r", True)
        session.keys.handle_key("r", False)
        session.rate = Mock()
        session.run_home = Mock(side_effect=session.runtime_state.request_quit)
        session.motion_tick = Mock()
        self.assertTrue(session.run_control_loop())
        session.run_home.assert_called_once()
        session.motion_tick.assert_not_called()

    def test_teleop_terminal_keys_take_priority(self):
        from dexmani_real.teleop.loop import TeleopLoop

        runner = object.__new__(TeleopLoop)
        runner.handle_operator_command = Mock(return_value=False)
        for terminal in ("esc", "q", "s", "d"):
            runner.handle_operator_command.reset_mock()
            runner.handle_commands(["b", "h", terminal], 1)
            self.assertEqual(runner.handle_operator_command.call_args.args, (terminal,))

    def test_teleop_stop_and_discard_keep_recorded_reason(self):
        from dexmani_real.teleop.loop import TeleopLoop

        runner = object.__new__(TeleopLoop)
        runner.runtime_state = armed_state()
        runner.end_episode = Mock()
        for key, reason, save in (("s", "stop", True), ("d", "discard", False)):
            runner.handle_operator_command(key)
            runner.end_episode.assert_called_with(save, reason)

    def test_policy_stop_batch_blocks_begin_and_home(self):
        from dexmani_real.deployment.session import PolicySession

        session = object.__new__(PolicySession)
        session.request_start = Mock()
        session.run_home = Mock()
        with self.assertLogs("dexmani_real.deployment.session", level="WARNING"):
            session.handle_command_batch(["b", "h", "s"])
        session.request_start.assert_not_called()
        session.run_home.assert_not_called()


class ReplayInputTests(unittest.TestCase):
    def setUp(self):
        from dexmani_real.replay import run as session

        self.session = session
        self.runtime = load_experiment_config()
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.output = stack.enter_context(TemporaryDirectory())
        self.state = armed_state()
        self.keys = active_keyboard(
            estop_callback=self.state.request_emergency_stop,
            bindings={'s': self.state.request_stop, 'q': self.state.request_quit},
        )
        self.keys.start = Mock()
        self.addCleanup(self.keys.stop)
        self.robot = Mock(stop_required=False)
        self.home = Mock(return_value=home.HomeResult(True))
        replacements = {
            "configure_logging": Mock(),
            "validate_robot_config": Mock(),
            "validate_control_rate": Mock(),
            "validate_trajectory": Mock(),
            "load_table_plane": Mock(),
            "compute_metrics": Mock(return_value=dict(
                matching_frames=0, original_frames=0, arm_joint=None, hand_joint=None,
                eef_position=None, eef_orientation=None,
            )),
            "SessionState": Mock(return_value=self.state),
            "Keyboard": Mock(return_value=self.keys),
            "Robot": Mock(return_value=self.robot),
            "HomePlanner": Mock(),
            "home_robot": self.home,
            "JointReplay": Mock(),
        }
        replacements["JointReplay"].return_value.run.return_value = session.ReplayResult(
            session.ReplayStatus.COMPLETED,
        )
        self.evaluate = replacements["compute_metrics"]
        stack.enter_context(patch.multiple(session, **replacements))
        stack.enter_context(patch("builtins.print"))

    def press(self, key):
        self.keys.handle_key(key, True)
        self.keys.handle_key(key, False)

    def run_batches(self, batches, expected_status=None):
        batches = iter(batches)

        def poll(timeout):
            for key in next(batches):
                self.press(key)
            return keyboard.Keyboard.poll(self.keys, 0)

        self.keys.poll = poll
        outcome = self.session.run_replay(
            SimpleNamespace(fps=30, episode_path="episode"),
            self.runtime,
            output_dir=self.output,
        )
        self.assertEqual(outcome.status, expected_status or self.session.ReplayStatus.COMPLETED, outcome.reason)
        self.robot.close.assert_called_once()

    def test_invalid_execution_age_fails_before_device_startup(self):
        for age in (0, -1, float("nan"), float("inf")):
            with self.subTest(age=age), self.assertRaisesRegex(ValueError, "max_decision_age_s"):
                self.session.run_replay(
                    SimpleNamespace(fps=30, episode_path="episode"),
                    replace(
                        self.runtime,
                        execution=replace(self.runtime.execution, max_decision_age_s=age),
                    ),
                    output_dir=self.output,
                )
        self.session.Robot.assert_not_called()
        self.assertEqual(list(Path(self.output).iterdir()), [])

    def test_startup_preserves_hand_pose_until_replay(self):
        self.run_batches([["q"]])
        self.robot.send_hand_home.assert_not_called()
        self.home.assert_not_called()
        self.session.JointReplay.return_value.run.assert_called_once()

    def test_startup_failure_still_saves_empty_data_and_failure(self):
        self.robot.connect.side_effect = RuntimeError("connect failed")
        with self.assertLogs(self.session.logger, "ERROR"):
            self.run_batches([], self.session.ReplayStatus.FAULT)
        result = json.loads((Path(self.output) / "session_result.json").read_text())
        self.assertIn("connect failed", result["reason"])
        self.assertIsNone(result["trajectory_status"])
        with np.load(Path(self.output) / "replay_data.npz") as saved:
            self.assertEqual(saved["arm_qpos"].shape, (0, 7))

    def test_quit_during_connection_is_clean_and_retains_empty_data(self):
        def quit():
            self.press("q")
            raise self.session.OperationCancelled("connection cancelled")

        self.robot.connect.side_effect = quit
        self.run_batches([], self.session.ReplayStatus.USER_QUIT)
        self.session.JointReplay.assert_not_called()
        result = json.loads((Path(self.output) / "session_result.json").read_text())
        self.assertTrue(result["shutdown_clean"])
        self.assertEqual(result["replay_rows"], 0)

    def test_final_publication_failure_propagates_after_cleanup(self):
        write = self.session.write_json

        def fail_final(payload, path, **kwargs):
            if path.name == "session_result.json" and payload["status"] != "incomplete":
                self.robot.close.assert_called_once()
                raise OSError("result disk failure")
            return write(payload, path, **kwargs)

        with patch.object(self.session, "write_json", side_effect=fail_final):
            with self.assertRaisesRegex(OSError, "result disk failure"):
                self.run_batches([["q"]])

    def test_stop_blocks_same_batch_home_and_allows_new_request_after_acknowledgment(self):
        self.run_batches([["s", "h"], [], ["h"], ["q"]])
        self.home.assert_called_once()

    def test_home_discards_requests_accumulated_during_motion(self):
        def execute(*args, **kwargs):
            self.press("h")
            return home.HomeResult(True)

        self.home.side_effect = execute
        self.run_batches([["h"], [], ["q"]])
        self.home.assert_called_once()

    def test_stop_during_home_is_an_interruption_and_allows_another_home(self):
        def interrupt(*args, **kwargs):
            self.press("s")
            self.home.side_effect = None
            return home.HomeResult(False, "home interrupted", interrupted=True)

        self.home.side_effect = interrupt
        self.run_batches([["h"], [], ["h"], ["q"]])
        self.assertEqual(self.home.call_count, 2)
        result = json.loads((Path(self.output) / "session_result.json").read_text())
        self.assertIn(dict(ok=False, interrupted=True, reason="home interrupted"), result["home_results"])

    def test_home_failure_does_not_erase_completed_trajectory(self):
        self.home.return_value = home.HomeResult(False, "home failed")
        self.run_batches([["h"]], self.session.ReplayStatus.FAULT)
        result = json.loads((Path(self.output) / "session_result.json").read_text())
        self.assertEqual(result["trajectory_status"], "completed")
        self.assertEqual(result["status"], "fault")
        self.assertIn("home failed", result["reason"])

    def test_evaluation_failure_is_recorded_after_cleanup(self):
        def fail(*args, **kwargs):
            self.robot.close.assert_called_once()
            raise RuntimeError("metrics failed")

        self.evaluate.side_effect = fail
        with self.assertLogs(self.session.logger, "ERROR"):
            self.run_batches([["q"]], self.session.ReplayStatus.FAULT)
        result = json.loads((Path(self.output) / "session_result.json").read_text())
        self.assertEqual(result["trajectory_status"], "completed")
        self.assertIn("metrics failed", result["reason"])
        self.assertTrue((Path(self.output) / "replay_data.npz").is_file())
        self.assertFalse((Path(self.output) / "metrics.json").exists())

    def test_shutdown_failure_cannot_report_success(self):
        self.robot.close.side_effect = RuntimeError("close failed")
        with self.assertLogs("dexmani_real.runtime.lifecycle", "ERROR"):
            self.run_batches([["q"]], self.session.ReplayStatus.FAULT)
        result = json.loads((Path(self.output) / "session_result.json").read_text())
        self.assertFalse(result["shutdown_clean"])


class CameraSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(serial=None, width=640, height=480)
        self.channels = SimpleNamespace(
            camera_ready=Mock(is_set=Mock(return_value=True)),
            camera_serial=SimpleNamespace(value=b"active"),
            camera_intrinsics=SimpleNamespace(value=json.dumps({
                "width": 640, "height": 480, "fx": 600, "fy": 600, "ppx": 320, "ppy": 240,
                "distortion_model": "none", "distortion_coeffs": [0] * 5,
            }).encode()),
            camera_depth_scale=SimpleNamespace(value=0.001),
        )

    def test_existing_calibration_must_contain_active_serial(self):
        for required in (False, True):
            with self.subTest(required=required), self.assertRaisesRegex(KeyError, "active"):
                snapshot_camera_parameters(
                    self.channels, self.config, {"other": np.eye(4)}, require_extrinsics=required,
                )

    def test_absent_calibration_is_allowed_only_for_optional_consumers(self):
        camera = snapshot_camera_parameters(self.channels, self.config, None)
        self.assertIsNone(camera.T_xarm_base_from_color)
        self.assertNotIn("camera_T_xarm_base_from_color", camera_metadata(camera))
        with self.assertRaisesRegex(ValueError, "requires current extrinsics"):
            snapshot_camera_parameters(self.channels, self.config, None, require_extrinsics=True)

    def test_snapshot_owns_transform_and_records_matching_serial(self):
        transform = np.eye(4)
        camera = snapshot_camera_parameters(self.channels, self.config, {"active": transform})
        transform[:3, 3] = 1
        np.testing.assert_array_equal(camera.T_xarm_base_from_color, np.eye(4))
        self.assertFalse(camera.T_xarm_base_from_color.flags.writeable)
        self.assertEqual(camera_metadata(camera)["camera_serial"], "active")
        self.assertEqual(
            camera_metadata(camera)["camera_T_xarm_base_from_color"], np.eye(4).ravel().tolist(),
        )


class PreviewInputTests(unittest.TestCase):
    def setUp(self):
        from scripts.tools import view_realsense as preview

        self.preview = preview
        self.runtime = load_experiment_config()
        self.viewer = preview.RealSenseViewer(SimpleNamespace(config=None, camera_calibration=None))
        self.viewer.rr = Mock()
        self.viewer.pointcloud_config = self.runtime.pointcloud
        self.viewer.table = self.runtime.environment.table
        self.viewer.calibration = None
        self.viewer.table_plane_abcd = None
        self.viewer.camera = Mock(active_serial="camera")
        self.viewer.camera.depth_scale_m = 0.001
        self.viewer.camera.read.return_value = SimpleNamespace(
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth=np.ones((2, 2), dtype=np.uint16),
            timestamp_ns=1,
        )
        self.keys = Mock(healthy=True, estop_latched=False)
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name in (
            "log_camera_geometry", "log_rgbd", "log_points", "log_series",
            "viewer_blueprint", "validate_aligned_depth_distortion",
        ):
            stack.enter_context(patch.object(preview, name))
        stack.enter_context(patch.object(preview, "Keyboard", return_value=self.keys))
        stack.enter_context(patch("builtins.print"))
        self.raw = stack.enter_context(patch.object(preview, "build_raw_point_cloud", return_value=None))
        self.processor = stack.enter_context(patch.object(preview, "PointCloudProcessor"))
        self.processor.return_value.process.return_value = None

    def test_geometry_is_cached_and_raw_does_not_require_table(self):
        transform, plane = np.eye(4), (0.0, 0.0, 1.0, 0.0)
        with (
            patch.object(self.preview, "load_camera_extrinsics") as calibration,
            patch.object(self.preview, "load_table_plane", return_value=plane) as resolve,
        ):
            calibration.return_value = {"camera": transform}
            np.testing.assert_array_equal(self.viewer.load_pointcloud_geometry("raw", "camera")[0], transform)
            resolve.assert_not_called()
            self.assertEqual(self.viewer.load_pointcloud_geometry("processed", "camera")[1], plane)
            self.assertEqual(self.viewer.load_pointcloud_geometry("processed", "camera")[1], plane)
            resolve.assert_called_once_with(self.runtime.environment.table)
            calibration.assert_called_once_with(None)

    def test_cloud_toggle_mode_and_reset_share_geometry_state(self):
        self.viewer.load_pointcloud_geometry = Mock(return_value=(np.eye(4), None))
        self.keys.poll.side_effect = [["p"], ["s"], ["p"], ["p"], ["r"], ["q"]]
        self.assertIsNone(self.viewer.capture(allow_exposure_changes=False))
        self.assertEqual(self.raw.call_count, 2)
        self.processor.return_value.process.assert_called_once()
        self.processor.assert_called_once()
        self.assertEqual(
            [call.args for call in self.viewer.load_pointcloud_geometry.call_args_list],
            [("processed", "camera"), ("raw", "camera"), ("raw", "camera")],
        )
        self.keys.stop.assert_called_once()
        self.viewer.rr.disconnect.assert_called_once()

    def test_capture_errors_propagate_and_close_viewer(self):
        self.viewer.load_pointcloud_geometry = Mock(return_value=(np.eye(4), None))
        for operation, keys, error in (
            (self.viewer.load_pointcloud_geometry, ["p"], OSError("calibration failed")),
            (self.viewer.camera.read, [], RuntimeError("camera failed")),
            (self.raw, ["p", "s"], ValueError("raw cloud failed")),
            (self.processor.return_value.process, ["p"], ValueError("processed cloud failed")),
        ):
            with self.subTest(error=error):
                self.keys.poll.side_effect = [keys, ["q"]]
                self.keys.stop.reset_mock()
                self.viewer.rr.disconnect.reset_mock()
                operation.side_effect = error
                try:
                    with self.assertRaises(type(error)) as raised:
                        self.viewer.capture(allow_exposure_changes=False)
                finally:
                    operation.side_effect = None
                self.assertIs(raised.exception, error)
                self.keys.stop.assert_called_once()
                self.viewer.rr.disconnect.assert_called_once()

    def test_run_restores_exposure_and_reports_failures(self):
        camera = self.viewer.camera
        for error, restored, status in (
            (None, 1.0, 0),
            (RuntimeError("capture failed"), 1.0, 1),
            (ValueError("invalid geometry"), 1.0, None),
            (None, None, 1),
            (None, OSError("restore failed"), 1),
        ):
            with (
                self.subTest(error=error, restored=restored),
                patch.dict("sys.modules", {"rerun": self.viewer.rr}),
                patch("dexmani_real.sensor.realsense.RealSenseCamera", return_value=camera),
                patch.object(self.preview, "list_cameras", return_value=[{}]),
                patch.object(self.preview, "configure_logging"),
                patch.object(self.preview, "get_ae_priority", return_value=1.0),
                patch.object(self.preview, "set_ae_priority", side_effect=[0.0, restored]) as priority,
                patch.object(self.viewer, "capture", side_effect=error),
            ):
                camera.reset_mock()
                if isinstance(error, ValueError):
                    with self.assertRaises(ValueError) as raised:
                        self.viewer.run()
                    self.assertIs(raised.exception, error)
                else:
                    self.assertEqual(self.viewer.run(), status)
                self.assertEqual([call.args for call in priority.call_args_list], [(camera, 0.0), (camera, 1.0)])
                camera.disconnect.assert_called_once()

    def test_exposure_read_failure_is_not_treated_as_unsupported(self):
        sensor = self.viewer.camera.color_sensor
        sensor.supports.return_value = True
        sensor.get_option.side_effect = RuntimeError("sensor read failed")
        sdk = SimpleNamespace(option=SimpleNamespace(auto_exposure_priority=1))
        with patch.dict("sys.modules", {"pyrealsense2": sdk}):
            with self.assertRaisesRegex(RuntimeError, "sensor read failed"):
                self.preview.get_ae_priority(self.viewer.camera)
            sensor.supports.return_value = False
            self.assertIsNone(self.preview.get_ae_priority(self.viewer.camera))


class JogTests(unittest.TestCase):
    def setUp(self):
        self.state = armed_state()
        self.keys = active_keyboard()
        self.keys.handle_key("w", True)
        self.generation = self.keys.snapshot()[1]
        self.addCleanup(self.keys.stop)
        self.jog = CartesianJog(
            None, SimpleNamespace(workspace_command_margin_m=.01),
            (-np.ones(3) * .99, np.ones(3) * .99),
        )
        self.robot = Mock()
        self.pose = Pose(p=np.zeros(3), q=np.array([1., 0., 0., 0.]))

    def submit(self):
        return self.jog.submit(
            self.pose, np.zeros(7), self.keys, self.generation, self.state, self.robot,
            self.state.run_id, valid_until_ns=10**20, execute_until_ns=10**20,
        )

    def test_opposing_and_diagonal_keys(self):
        self.assertFalse(jog_motion_requested({"w", "s"}))
        self.assertFalse(jog_motion_requested({"w", "q"}))
        self.assertTrue(jog_motion_requested({"w", "s", "left"}))
        dx, rotation = jog_delta({"w", "d", "i", "j"}, .008, .03)
        np.testing.assert_allclose(dx, [.008 / np.sqrt(2), .008 / np.sqrt(2), 0])
        self.assertAlmostEqual(np.linalg.norm(rotation), .03)

    def test_success_reuses_owned_command_and_copies_pose(self):
        self.assertTrue(self.submit())
        command = self.robot.send_action.call_args.args[0]
        self.assertIs(self.jog.command_qpos, command.arm_qpos)
        self.assertFalse(self.jog.command_qpos.flags.writeable)
        self.assertFalse(np.shares_memory(self.jog.command_pose.p, self.pose.p))

    def test_stale_input_never_submits(self):
        self.keys.handle_key("w", False)
        self.assertFalse(self.submit())
        self.robot.send_action.assert_not_called()

    def test_release_or_stop_during_submission_does_not_commit_reference(self):
        for cancel in (lambda: self.keys.handle_key("w", False), lambda: self.state.request_stop()):
            with self.subTest(cancel=cancel):
                self.state = armed_state()
                self.keys.handle_key("w", True)
                self.generation = self.keys.snapshot()[1]
                old_id = self.state.run_id
                self.robot.send_action.side_effect = lambda *args, **kwargs: cancel()
                with self.assertRaises(OperationCancelled):
                    self.submit()
                self.assertIsNone(self.jog.command_qpos)
                self.assertFalse(self.state.can_move(run_id=old_id))

    def test_stop_acknowledgement_cannot_clear_a_newer_request(self):
        self.state.begin_motion()
        self.state.request_stop()
        stopped_id = self.state.run_id
        self.state.request_stop()
        self.state.acknowledge_stop(stopped_id)
        self.assertTrue(self.state.stop_request)
        self.state.revoke_motion(SafetyState.FAULT)
        self.assertEqual(self.state.run_ended_reason, RunEndReason.OPERATOR)

    def test_expiry_during_submission_rejects_reference_update(self):
        clock = Clock()
        deadline = clock.ns + 1_000_000
        old_id = self.state.run_id
        self.robot.send_action.side_effect = lambda *args, **kwargs: clock.sleep(.01)
        with patch("dexmani_real.runtime.jog.time.monotonic_ns", clock.monotonic_ns):
            with self.assertRaises(TimeoutError):
                self.jog.submit(
                    self.pose, np.zeros(7), self.keys, self.generation,
                    self.state, self.robot, old_id,
                    valid_until_ns=deadline, execute_until_ns=deadline,
                )
        self.assertIsNone(self.jog.command_qpos)
        self.assertFalse(self.state.can_move(run_id=old_id))


class JogStepTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        clock_patch = patch("dexmani_real.runtime.jog.time.monotonic_ns", self.clock.monotonic_ns)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.state = armed_state()
        self.keys = active_keyboard()
        self.addCleanup(self.keys.stop)
        self.keys.handle_key("w", True)
        self.runtime = load_experiment_config(cli_overrides={"hand.enabled": False})
        self.solver = Mock()
        self.solver.compute_eef_pose_base.return_value = Pose(
            p=np.zeros(3), q=np.array([1., 0., 0., 0.]),
        )
        self.solver.solve.return_value = SimpleNamespace(qpos=np.ones(7) * .01)
        self.jog = CartesianJog(
            self.solver, self.runtime.keyboard_teleop, (-np.ones(3), np.ones(3)),
        )
        self.arm = np.zeros(1, dtype=[("timestamp_ns", "u8"), ("qpos", "f8", (7,))])
        self.arm["timestamp_ns"] = self.clock.ns - 10_000_000
        self.robot = Mock(stop_required=False)
        self.robot.read_state.return_value = SimpleNamespace(arm=self.arm, hand=None)

        def prepare(*args, **kwargs):
            self.clock.sleep(.1)
            self.arm["timestamp_ns"] = self.clock.ns - 10_000_000
            self.state.stop_signal.clear()
            self.robot.stop_required = True

        def stop():
            self.state.stop_signal.set()
            self.robot.stop_required = False

        self.robot.prepare_streaming.side_effect = prepare
        self.robot.stop.side_effect = stop

    def step(self, source="observation"):
        return self.jog.step(
            self.robot, self.keys, self.state, self.runtime, deadline_source=source,
        )

    def test_preparation_refreshes_once_and_preserves_execution_clock(self):
        for source in ("observation", "arm"):
            with self.subTest(source=source):
                result = self.step(source)
                self.assertTrue(result.committed)
                self.robot.read_state.assert_called_once()
                self.robot.prepare_streaming.assert_called_once()
                origin = (self.clock.ns if source == "observation"
                          else int(self.arm["timestamp_ns"][0]))
                self.assertEqual(
                    self.robot.send_action.call_args.kwargs["execute_until_ns"],
                    origin + int(self.runtime.keyboard_teleop.target_max_age_s * 1e9),
                )
                self.robot.read_state.reset_mock()
        self.keys.handle_key("w", False)
        result = self.step()
        self.assertFalse(result.committed)
        self.assertIsNotNone(result.observation)
        self.robot.read_state.assert_called_once()
        self.robot.stop.assert_called_once()
        self.assertIsNone(self.jog.command_pose)

    def test_release_during_prepare_stops_before_ik_or_submission(self):
        self.robot.prepare_streaming.side_effect = lambda *a, **k: self.keys.handle_key("w", False)
        result = self.step()
        self.assertFalse(result.committed)
        self.robot.stop.assert_called_once()
        self.robot.read_state.assert_not_called()
        self.solver.solve.assert_not_called()
        self.robot.send_action.assert_not_called()

    def test_rejected_target_waits_for_release(self):
        self.solver.solve.return_value = SimpleNamespace(qpos=None, reason="collision")
        self.assertFalse(self.step().committed)
        self.assertTrue(self.jog.blocked_until_release)
        self.robot.stop.assert_called_once()
        self.assertIsNone(self.jog.command_pose)
        self.step()
        self.solver.solve.assert_called_once()
        self.robot.stop.assert_called_once()
        self.robot.send_action.assert_not_called()
        self.keys.handle_key("w", False)
        self.step()
        self.assertFalse(self.jog.blocked_until_release)

    def test_input_change_during_ik_stops_in_the_same_tick(self):
        def solve(*args, **kwargs):
            self.keys.handle_key("w", False)
            return SimpleNamespace(qpos=np.ones(7) * .01)

        self.solver.solve.side_effect = solve
        result = self.step()
        self.assertFalse(result.committed)
        self.robot.send_action.assert_not_called()
        self.robot.stop.assert_called_once()
        self.assertIsNone(self.jog.command_pose)
        self.assertEqual(self.state.safety_state, SafetyState.ARMED)

    def test_stop_failure_after_rejection_is_not_treated_as_submission_timeout(self):
        self.solver.solve.return_value = SimpleNamespace(qpos=None, reason="collision")
        self.robot.stop.side_effect = TimeoutError("stop unresolved")
        with self.assertRaisesRegex(TimeoutError, "stop unresolved"):
            self.step()
        self.robot.stop.assert_called_once()
        self.assertIsNone(self.jog.command_pose)

    def test_stale_feedback_stops_without_solving(self):
        self.state.begin_motion()
        self.robot.stop_required = True
        self.arm["timestamp_ns"] = 1
        result = self.step()
        self.assertIn("arm_feedback_stale", result.issues)
        self.robot.stop.assert_called_once()
        self.solver.solve.assert_not_called()
        self.robot.send_action.assert_not_called()

    def test_expired_submission_stops_without_committing_reference(self):
        self.robot.send_action.side_effect = TimeoutError("expired")
        result = self.step()
        self.assertFalse(result.committed)
        self.assertIsNone(self.jog.command_pose)
        self.robot.stop.assert_called_once()

    def test_pause_clears_reference_even_when_stop_fails(self):
        self.jog.command_pose = self.solver.compute_eef_pose_base.return_value
        self.robot.stop.side_effect = OSError("stop failed")
        with self.assertRaisesRegex(OSError, "stop failed"):
            self.jog.pause(self.robot, self.state)
        self.assertTrue(self.jog.blocked_until_release)
        self.assertIsNone(self.jog.command_pose)

    def test_unexpected_feedback_or_solver_error_propagates(self):
        for operation in (self.robot.read_state, self.solver.solve):
            with self.subTest(operation=operation):
                operation.side_effect = TimeoutError("unexpected")
                with self.assertRaisesRegex(TimeoutError, "unexpected"):
                    self.step()
                operation.side_effect = None
        self.robot.send_action.assert_not_called()


class Clock:
    def __init__(self):
        self.ns = 1_000_000_000

    def monotonic_ns(self):
        return self.ns

    def monotonic(self):
        return self.ns / 1e9

    def sleep(self, seconds):
        self.ns += int(seconds * 1e9)


class HomeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        for name in ("monotonic_ns", "monotonic", "sleep"):
            patcher = patch.object(home.time, name, getattr(self.clock, name))
            patcher.start()
            self.addCleanup(patcher.stop)
        self.state = armed_state()
        self.robot = SimpleNamespace(
            arm_config=ArmConfig(),
            hand_config=replace(HandConfig(), home_timeout_s=.08),
            stop_required=False,
        )
        self.robot.stop = Mock(side_effect=lambda: setattr(self.robot, "stop_required", False))
        self.robot.send_hand_home = Mock(
            side_effect=lambda *args, **kwargs: setattr(self.robot, "stop_required", True)
        )
        self.robot.read_state = Mock(side_effect=self.feedback)

        def move_arm(*args, **kwargs):
            self.robot.stop_required = True
            return True

        self.robot.home_arm = Mock(side_effect=move_arm)
        self.planner = Mock()
        self.path = self.planner.plan
        self.path.return_value = np.zeros((2, 7))

    def feedback(self):
        self.clock.sleep(.001)
        arm = np.zeros(1, dtype=[("timestamp_ns", "u8"), ("qpos", "f8", (7,)), ("qvel", "f8", (7,))])
        hand = np.zeros(1, dtype=[("timestamp_ns", "u8"), ("qpos", "f8", (12,))])
        arm["timestamp_ns"] = hand["timestamp_ns"] = self.clock.ns
        hand["qpos"] = np.deg2rad(self.robot.hand_config.home_qpos_deg)
        return SimpleNamespace(arm=arm, hand=hand)

    def test_combined_home_stops_each_completed_operation_once(self):
        result = home.home_robot(
            self.state, self.robot, self.planner, cancel_requested=lambda: False, service=Mock(),
        )
        self.assertTrue(result.ok)
        self.assertEqual(self.robot.stop.call_count, 2)
        self.robot.home_arm.assert_called_once()
        self.assertGreaterEqual(self.robot.read_state.call_count, 4)

    def test_repeated_hand_sample_cannot_establish_convergence(self):
        cached = self.feedback().hand
        self.robot.read_state.side_effect = None
        self.robot.read_state.return_value = SimpleNamespace(hand=cached)
        result = home.home_hand(self.state, self.robot, cancel_requested=lambda: False)
        self.assertFalse(result.ok)
        self.assertIn("timed out", result.reason)
        self.robot.stop.assert_called_once()

    def test_cancellation_after_hand_submission_stops_motion(self):
        result = home.home_hand(
            self.state, self.robot,
            cancel_requested=lambda: self.robot.send_hand_home.called,
        )
        self.assertTrue(result.interrupted)
        self.robot.stop.assert_called_once()

    def test_disabled_hand_and_rejected_path_do_not_send(self):
        self.robot.hand_config = replace(self.robot.hand_config, enabled=False)
        self.path.side_effect = home.HomePathRejected("collision")
        result = home.home_robot(
            self.state, self.robot, self.planner, cancel_requested=lambda: False, service=Mock(),
        )
        self.assertIn("collision", result.reason)
        self.robot.send_hand_home.assert_not_called()
        self.robot.home_arm.assert_not_called()
        self.robot.stop.assert_not_called()

    def test_stationary_feedback_timeout_does_not_plan_or_send(self):
        patcher = patch.object(home, "HOME_CONVERGENCE_TIMEOUT_S", .03)
        patcher.start()
        self.addCleanup(patcher.stop)
        old_feedback = self.feedback()
        self.robot.read_state.side_effect = None
        self.robot.read_state.return_value = old_feedback
        result = home.home_arm(
            self.state, self.robot, self.planner, cancel_requested=lambda: False, service=Mock(),
        )
        self.assertFalse(result.ok)
        self.path.assert_not_called()
        self.robot.home_arm.assert_not_called()

    def test_primary_failure_survives_cleanup_failure(self):
        primary = RuntimeError("SDK failure")

        def fail(*args, **kwargs):
            self.robot.stop_required = True
            raise primary

        self.robot.home_arm.side_effect = fail
        self.robot.stop.side_effect = OSError("stop failure")
        with self.assertLogs(home.logger, level="ERROR"):
            with self.assertRaises(RuntimeError) as raised:
                home.home_arm(
                    self.state, self.robot, self.planner,
                    cancel_requested=lambda: False, service=Mock(),
                )
        self.assertIs(raised.exception, primary)

    def test_stop_failure_cannot_report_home_success(self):
        self.robot.stop.side_effect = OSError("stop failure")
        with self.assertRaisesRegex(OSError, "stop failure"):
            home.home_hand(self.state, self.robot, cancel_requested=lambda: False)

    def test_arm_execution_timeout_stops_motion(self):
        def move(waypoints, target, run_id, abort, **kwargs):
            self.robot.stop_required = True
            self.clock.sleep(100)
            self.assertGreaterEqual(self.clock.ns, kwargs["deadline_ns"])
            self.assertFalse(abort())
            raise TimeoutError("HOME execution deadline expired")

        self.robot.home_arm.side_effect = move
        result = home.home_arm(
            self.state, self.robot, self.planner, cancel_requested=lambda: False, service=Mock(),
        )
        self.assertFalse(result.ok)
        self.assertFalse(result.interrupted)
        self.assertIn("arm home timed out", result.reason)
        self.assertIn("HOME execution deadline expired", result.reason)
        self.robot.stop.assert_called_once()

    def test_arm_cancellation_remains_distinct_from_elapsed_deadline(self):
        def cancel(*args, **kwargs):
            self.robot.stop_required = True
            self.clock.sleep(100)
            raise OperationCancelled("operator cancelled HOME")

        self.robot.home_arm.side_effect = cancel
        result = home.home_arm(
            self.state, self.robot, self.planner, cancel_requested=lambda: False, service=Mock(),
        )
        self.assertFalse(result.ok)
        self.assertTrue(result.interrupted)
        self.assertEqual(result.reason, "operator cancelled HOME")
        self.robot.stop.assert_called_once()

    def test_keyboard_interrupt_latches_estop_and_stops(self):
        def interrupted(*args, **kwargs):
            self.robot.stop_required = True
            raise KeyboardInterrupt

        self.robot.send_hand_home.side_effect = interrupted
        with self.assertRaises(KeyboardInterrupt):
            home.home_hand(self.state, self.robot, cancel_requested=lambda: False)
        self.assertTrue(self.state.emergency_signal.is_set())
        self.robot.stop.assert_called_once()


class FakeProcess:
    def __init__(self, name="camera", exit_mode="graceful"):
        self.name, self.exit_mode = name, exit_mode
        self.alive = False
        self.exitcode = None
        self.joins = []

    def start(self):
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        self.joins.append(timeout)
        if self.exit_mode == "graceful":
            self.alive, self.exitcode = False, 0

    def terminate(self):
        if self.exit_mode == "terminate":
            self.alive, self.exitcode = False, -15

    def kill(self):
        if self.exit_mode == "kill":
            self.alive, self.exitcode = False, -9


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.channels = SimpleNamespace(is_running=SimpleNamespace(value=True))
        self.sensors = lifecycle.SensorWorkers(self.channels)
        self.ready = threading.Event()
        self.ready.set()

    def start(self, process, **kwargs):
        self.sensors.start(process, ready=self.ready, may_start=lambda: True, **kwargs)

    def test_start_failure_does_not_take_ownership(self):
        process = Mock()
        process.start.side_effect = OSError("spawn failed")
        with self.assertRaisesRegex(OSError, "spawn failed"):
            self.start(process)
        self.assertEqual(self.sensors.started, [])

    def test_readiness_timeout_retains_process_for_cleanup(self):
        process = FakeProcess()
        clock = Clock()
        ready = Mock()
        ready.wait.side_effect = lambda timeout: clock.sleep(timeout)
        with patch.object(lifecycle.time, "monotonic", clock.monotonic):
            with self.assertRaisesRegex(RuntimeError, "failed to become ready"):
                self.sensors.start(process, ready=ready, timeout_s=.1, may_start=lambda: True)
        self.assertEqual(len(self.sensors.started), 1)
        self.assertTrue(self.sensors.shutdown())

    def test_worker_death_clears_ready(self):
        process = FakeProcess()
        self.start(process, timeout_s=1)
        process.alive, process.exitcode = False, 1
        with self.assertLogs(lifecycle.logger, level="ERROR"):
            self.assertFalse(self.sensors.check())
        self.assertFalse(self.ready.is_set())

    def test_forced_shutdown_is_not_clean_and_uses_remaining_budget(self):
        process = FakeProcess(exit_mode="kill")
        self.start(process)
        with patch.object(lifecycle.time, "monotonic", return_value=10):
            with self.assertLogs(lifecycle.logger, level="WARNING"):
                self.assertFalse(self.sensors.shutdown(deadline=9))
        self.assertEqual(process.joins, [0., 1., 1.])
        self.assertFalse(self.channels.is_running.value)
        self.assertFalse(self.ready.is_set())
        self.assertEqual(self.sensors.started, [])

    def test_unresolved_worker_does_not_prevent_other_cleanup(self):
        stuck, normal = FakeProcess(exit_mode="stuck"), FakeProcess("vr")
        self.start(stuck)
        self.start(normal)
        with self.assertLogs(lifecycle.logger, level="ERROR"):
            with self.assertRaisesRegex(RuntimeError, "shutdown unresolved"):
                self.sensors.shutdown(deadline=0)
        self.assertFalse(normal.alive)
        self.assertEqual(self.sensors.started, [(stuck, self.ready)])
        stuck.exit_mode = "graceful"
        self.assertTrue(self.sensors.shutdown())
        self.assertEqual(len(normal.joins), 1)
        self.assertEqual(self.sensors.started, [])

    def test_close_revokes_first_and_continues_after_failure(self):
        state = armed_state()
        state.begin_motion()
        run_id = state.run_id
        calls = []

        def close_robot():
            self.assertFalse(state.can_move(run_id=run_id))
            calls.append("robot")
            raise OSError("close failure")

        robot = SimpleNamespace(close=close_robot)
        keys = SimpleNamespace(stop=lambda: calls.append("keyboard"))
        sensors = SimpleNamespace(shutdown=Mock(side_effect=lambda **kwargs: calls.append("sensors") or True))
        with self.assertLogs(lifecycle.logger, level="ERROR"):
            self.assertFalse(lifecycle.close_runtime(state, robot, keyboard=keys, sensors=sensors, deadline=42))
        self.assertEqual(calls, ["robot", "keyboard", "sensors"])
        sensors.shutdown.assert_called_once_with(deadline=42)
        self.assertEqual(state.safety_state, SafetyState.FAULT)

    def test_removed_timeout_config_is_rejected(self):
        with self.assertRaises(TypeError):
            load_experiment_config(cli_overrides={"safety.camera_ready_timeout_s": 1})


if __name__ == "__main__":
    unittest.main()

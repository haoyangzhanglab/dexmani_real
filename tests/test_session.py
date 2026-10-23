import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import CancelledError
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.runtime.observation import ObservationNeeds
from fakes import FakeClock, make_episode, make_session


class AssemblyTests(unittest.TestCase):
    def test_policy_starts_only_requested_sensor_workers(self):
        from dexmani_real.deployment.session import PolicySession

        for fields, raw, expected in (
            (("joint_state",), False, []),
            (("finger_tactile_force",), False, []),
            (("taxel_tactile_force",), False, []),
            (("camera_intrinsic",), False, ["camera"]),
            (("camera_extrinsic",), False, ["camera"]),
            (("point_cloud",), False, ["camera", "pointcloud"]),
            (("rgb",), False, ["camera"]),
            (("rgb", "point_cloud"), False, ["camera", "pointcloud"]),
            (("joint_state",), True, ["camera"]),
        ):
            with self.subTest(fields=fields, raw=raw), ExitStack() as stack:
                doubles = {}
                for name in ("SensorChannels", "SensorProcesses", "DexManiRobot", "ActionMapper",
                             "build_home_planner", "build_observation_kinematics", "build_observation_camera",
                             "CameraExtrinsics", "load_optional_camera_extrinsics", "AsyncEpisodeRecorder"):
                    doubles[name] = stack.enter_context(patch("dexmani_real.deployment.session." + name))
                stack.enter_context(patch("dexmani_real.deployment.inference.InferenceWorker"))
                process_context = Mock()
                stack.enter_context(patch("dexmani_real.deployment.session.mp.get_context", return_value=process_context))
                info = SimpleNamespace(observation_fields=fields, n_obs_steps=1, hand_history_steps=None,
                                       control_dt_s=.05, horizon=8, action_mode="joint")
                session = PolicySession(ExperimentConfig(), info,
                                         needs=ObservationNeeds.for_policy(fields, record_raw=raw),
                                         recording_dir=Path("/tmp/fake-recording") if raw else None)
                session.runtime_state = session.sensors = session.robot = None
                session.results = None
                session.warmup_policy = lambda: None
                session.prepare_resources()
                session.keyboard = Mock(healthy=True, estop_latched=False)
                session.start_devices()
                self.assertEqual([call.kwargs["name"] for call in process_context.Process.call_args_list], expected)
                self.assertEqual(doubles["DexManiRobot"].call_args.kwargs["required_tactile"], session.needs.required_tactile)
                self.assertEqual(doubles["SensorChannels"].called, bool(expected))

    def test_teleop_recording_and_no_record_sensor_needs(self):
        from dexmani_real.teleop.session import TeleopSession

        for recording in (False, True):
            with self.subTest(recording=recording), ExitStack() as stack:
                runtime = ExperimentConfig()
                runtime = replace(runtime, control=replace(runtime.control, recording_enabled=recording))
                names = ("configure_logging", "load_vr_transform", "load_optional_camera_extrinsics",
                         "ActionMapper", "VRWristMapper", "TeleopController", "build_home_planner",
                         "SensorChannels", "SensorProcesses", "DexManiRobot", "AsyncEpisodeRecorder", "TeleopRunner")
                doubles = {name: stack.enter_context(patch("dexmani_real.teleop.session." + name)) for name in names}
                stack.enter_context(patch("dexmani_real.teleop.session.resolve_runtime_table", side_effect=lambda cfg: cfg))
                stack.enter_context(patch("dexmani_real.teleop.session.shutdown_local_runtime", return_value=True))
                process_context = Mock()
                stack.enter_context(patch("dexmani_real.teleop.session.mp.get_context", return_value=process_context))
                session = TeleopSession(runtime)
                session.build_hand_retargeter = lambda: None
                doubles["TeleopRunner"].return_value.run.side_effect = lambda: session.runtime_state.request_quit()
                self.assertEqual(session.run(), 0)
                needs = doubles["TeleopRunner"].call_args.kwargs["needs"]
                self.assertTrue(needs.vr)
                self.assertEqual(needs.camera, "rgbd" if recording else "none")
                self.assertEqual(needs.required_tactile, ("aggregate", "dense"))
                self.assertEqual(doubles["SensorProcesses"].return_value.start.call_count, int(recording))

    def test_replay_has_no_sensor_process_or_vr_startup(self):
        from dexmani_real.replay.session import ReplaySession
        from dexmani_real.replay.replayer import ReplayOutcome, ReplayStatus
        from dexmani_real.robot.home import HomeResult
        from dexmani_real.runtime.operator_input import OperatorCommand

        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            for name in ("configure_logging", "verify_replay_preflight", "build_home_planner",
                         "DexManiRobot", "KeyboardInput", "EpisodeResults", "evaluate_replay"):
                stack.enter_context(patch("dexmani_real.replay.session." + name))
            stack.enter_context(patch("dexmani_real.config.experiment.resolve_runtime_table", side_effect=lambda cfg: cfg))
            stack.enter_context(patch("dexmani_real.planning.kinematics.arm_fk.make_arm_fk"))
            stack.enter_context(patch("dexmani_real.replay.session.home_hand", return_value=HomeResult(True)))
            stack.enter_context(patch("dexmani_real.replay.session.shutdown_local_runtime", return_value=True))
            replayer = stack.enter_context(patch("dexmani_real.replay.session.TrajectoryReplayer"))
            replayer.return_value.run.return_value = ReplayOutcome(ReplayStatus.USER_QUIT, reason="test")
            process_start = stack.enter_context(patch("multiprocessing.process.BaseProcess.start", side_effect=AssertionError("unexpected process")))
            session = ReplaySession(SimpleNamespace(fps=20), ExperimentConfig(), output_dir=Path(directory))
            # Feed Q through the fake listener when replay offers its post-run HOME prompt.
            from dexmani_real.replay.session import KeyboardInput
            KeyboardInput.return_value.poll.return_value = [OperatorCommand.QUIT]
            session.run()
            replayer.return_value.run.assert_called_once()
            process_start.assert_not_called()


class SessionShutdownTests(unittest.TestCase):
    def test_recorder_close_failure_does_not_skip_devices_keyboard_or_model(self):
        session = make_session(make_episode(FakeClock()))
        session.recorder = Mock()
        session.recorder.close.side_effect = RuntimeError("recorder close failure")
        session.model = Mock()
        session.model.close_result.return_value = {"status": "pending", "reason": "busy"}
        session.results = Mock()
        with patch("dexmani_real.deployment.session.shutdown_local_runtime", return_value=True) as shutdown:
            self.assertEqual(session.close(clean=True, failure=None), 1)
        session.model.close.assert_called_once()
        session.recorder.close.assert_called_once()
        shutdown.assert_called_once()
        self.assertIs(shutdown.call_args.kwargs["keyboard"], session.keyboard)
        self.assertEqual(session.results.finish_session.call_args.kwargs["model_close"]["status"], "pending")
        self.assertFalse(session.results.finish_session.call_args.kwargs["shutdown_clean"])

    def test_independent_runtime_resources_close_after_robot_failure(self):
        from dexmani_real.runtime.processes import shutdown_local_runtime
        from dexmani_real.runtime.state import RuntimeState
        robot, keyboard, sensors = Mock(), Mock(), Mock()
        robot.close.side_effect = RuntimeError("device close failed")
        sensors.shutdown.return_value = True
        self.assertFalse(shutdown_local_runtime(RuntimeState(), robot, sensors=sensors, keyboard=keyboard))
        keyboard.stop.assert_called_once()
        sensors.shutdown.assert_called_once()

    def test_listener_loss_is_an_immediate_software_fence(self):
        session = make_session(make_episode(FakeClock()))
        epoch, _ = session.runtime_state.begin_motion()
        session.keyboard.healthy = False
        self.assertFalse(session.check_services())
        self.assertTrue(session.runtime_state.estop_request)
        self.assertFalse(session.runtime_state.command_may_cross_sdk(run_id=epoch))


class StartupCancellationTests(unittest.TestCase):
    def test_entry_listeners_cancel_before_any_device_factory(self):
        from dexmani_real.calibration.camera.session import CameraCalibrationSession
        from dexmani_real.deployment.session import PolicySession
        from dexmani_real.robot.robot import DexManiRobot
        from dexmani_real.runtime.state import RuntimeState
        from dexmani_real.teleop.keyboard_session import KeyboardTeleopSession
        from dexmani_real.teleop.session import TeleopSession

        for entry in ("keyboard", "calibration", "policy", "teleop"):
            with self.subTest(entry=entry), ExitStack() as stack:
                runtime, state = ExperimentConfig(), RuntimeState()
                listener = Mock(healthy=True, estop_latched=False)
                listener.start.side_effect = state.request_quit if entry in ("keyboard", "calibration") else state.request_stop
                arm = stack.enter_context(patch(
                    "dexmani_real.robot.xarm7.executor.ArmExecutor",
                    side_effect=AssertionError("device factory after startup cancellation"),
                ))
                process_start = stack.enter_context(patch(
                    "multiprocessing.process.BaseProcess.start", side_effect=AssertionError("real process start"),
                ))
                stack.enter_context(patch(
                    "dexmani_real.runtime.operator_input.KeyboardInput.start",
                    side_effect=AssertionError("real listener start"),
                ))
                if entry == "keyboard":
                    for name in ("make_online_ik_solver", "TargetChecker", "build_home_planner"):
                        stack.enter_context(patch("dexmani_real.teleop.keyboard_session." + name))
                    session = KeyboardTeleopSession(runtime, no_hand=False)
                    session.runtime_state, session.keys = state, listener
                    session.jog_config = runtime.keyboard_teleop
                    session.robot = DexManiRobot(state, runtime)
                    with self.assertRaises(CancelledError):
                        session.start()
                elif entry == "calibration":
                    stack.enter_context(patch("dexmani_real.calibration.camera.session.KeyboardInput", return_value=listener))
                    stack.enter_context(patch("dexmani_real.calibration.camera.session.shutdown_local_runtime", return_value=True))
                    session = CameraCalibrationSession(runtime, hand_geometry="absent")
                    session.prepare = lambda: None
                    session.runtime_state, session.sensors, session.ctx = state, Mock(), Mock()
                    session.keys, session.window_created = None, False
                    session.robot = DexManiRobot(state, runtime, check_services=session.check_services)
                    with self.assertLogs("dexmani_real.calibration.camera.session", level="ERROR"):
                        self.assertEqual(session.run(), 1)
                    session.ctx.Process.assert_not_called()
                elif entry == "policy":
                    session = PolicySession(runtime, SimpleNamespace(), needs=ObservationNeeds.for_policy(
                        ("joint_state",), record_raw=False,
                    ))
                    session.runtime_state, session.keyboard, session.sensors = state, listener, Mock()
                    session.robot = DexManiRobot(state, runtime)
                    with self.assertRaises(CancelledError):
                        session.start_devices()
                else:
                    runtime = replace(runtime, control=replace(runtime.control, recording_enabled=False))
                    for name in ("configure_logging", "load_vr_transform", "ActionMapper", "VRWristMapper",
                                 "TeleopController", "build_home_planner", "SensorChannels", "SensorProcesses"):
                        stack.enter_context(patch("dexmani_real.teleop.session." + name))
                    stack.enter_context(patch("dexmani_real.teleop.session.resolve_runtime_table", side_effect=lambda cfg: cfg))
                    stack.enter_context(patch("dexmani_real.teleop.session.shutdown_local_runtime", return_value=True))
                    stack.enter_context(patch("dexmani_real.teleop.session.RuntimeState", return_value=state))
                    runner = Mock(keyboard=listener)
                    stack.enter_context(patch("dexmani_real.teleop.session.TeleopRunner", return_value=runner))
                    stack.enter_context(patch("dexmani_real.teleop.session.mp.get_context", return_value=Mock()))
                    session = TeleopSession(runtime)
                    session.build_hand_retargeter = lambda: None
                    with self.assertLogs("dexmani_real.teleop.session", level="ERROR"):
                        self.assertEqual(session.run(), 1)
                    runner.run.assert_not_called()
                listener.start.assert_called_once()
                arm.assert_not_called()
                process_start.assert_not_called()
                self.assertTrue(state.stop_request)
                self.assertFalse(state.requests_allow_motion())
                self.assertIsNone(state.begin_motion())

    def test_arm_initialization_stop_timeout_and_feedback_fault_fence_authority(self):
        from dexmani_real.robot.xarm7.executor import ArmExecutor
        from dexmani_real.robot.xarm7.protocol import OperationCancelled
        from dexmani_real.runtime.state import RuntimeState, SafetyState

        for cause, error in (("stop", OperationCancelled), ("timeout", TimeoutError), ("feedback", RuntimeError)):
            with self.subTest(cause=cause), ExitStack() as stack:
                state = RuntimeState()
                executor = ArmExecutor.__new__(ArmExecutor)
                executor.cfg = SimpleNamespace(homing=SimpleNamespace(convergence_timeout_s=.01), executor_timeout_s=.01)
                executor.runtime_state, executor.factory, executor.process = state, None, None
                executor.io = SimpleNamespace(
                    authority=SimpleNamespace(value=0), emergency=threading.Event(), ready=Mock(),
                    fault=SimpleNamespace(value=b""), stopped=threading.Event(), stop=threading.Event(),
                    closing=threading.Event(),
                )
                executor.ctx, executor.child_pipe = Mock(), Mock()
                executor.ctx.Process.return_value.is_alive.return_value = True
                stack.enter_context(patch.dict(sys.modules, {"ruckig": SimpleNamespace(Ruckig=Mock())}))
                stack.enter_context(patch("multiprocessing.process.BaseProcess.start", side_effect=AssertionError("real process start")))

                def wait(**kwargs):
                    if cause == "stop":
                        state.request_stop()
                        self.assertEqual(executor.io.authority.value, 0)
                    elif cause == "feedback":
                        executor.io.fault.value = b"feedback fault"
                    return False

                executor.io.ready.wait.side_effect = wait
                if cause == "timeout":
                    stack.enter_context(patch("dexmani_real.robot.xarm7.executor.time.monotonic", side_effect=[0., 1., 1.]))
                with self.assertRaises(error):
                    executor.connect()
                self.assertEqual(executor.io.authority.value, 0)
                self.assertTrue(executor.io.stop.is_set())
                self.assertTrue(executor.io.closing.is_set())
                self.assertIsNone(state.begin_motion())
                if cause == "stop":
                    self.assertTrue(state.stop_request)
                elif cause == "feedback":
                    self.assertEqual(state.safety_state, SafetyState.FAULT)

    def test_sensor_start_cancellation_and_timeout_retain_started_ownership(self):
        from dexmani_real.runtime.processes import SensorProcesses
        from dexmani_real.runtime.state import RuntimeState

        for cause in ("stop_before_start", "quit_at_ready", "timeout"):
            with self.subTest(cause=cause), ExitStack() as stack:
                state, process, ready = RuntimeState(), Mock(), Mock()
                process.is_alive.return_value = True
                sensors = SensorProcesses(state)
                if cause == "stop_before_start":
                    state.request_stop()
                elif cause == "quit_at_ready":
                    def became_ready(**kwargs):
                        state.request_quit()
                        return True
                    ready.wait.side_effect = became_ready
                else:
                    stack.enter_context(patch("dexmani_real.runtime.processes.time.monotonic", side_effect=[0., 1.]))
                stack.enter_context(patch("multiprocessing.process.BaseProcess.start", side_effect=AssertionError("real process start")))
                with self.assertRaises(RuntimeError):
                    sensors.start(process, ready=ready, timeout_s=.5)
                if cause == "stop_before_start":
                    process.start.assert_not_called()
                    self.assertEqual(sensors.started, [])
                else:
                    process.start.assert_called_once()
                    self.assertEqual(sensors.started, [(process, ready)])
                if cause != "timeout":
                    self.assertTrue(state.stop_request)
                    next_process = Mock()
                    with self.assertRaises(RuntimeError):
                        sensors.start(next_process, ready=Mock())
                    next_process.start.assert_not_called()

    def test_quit_at_vr_readiness_is_consumed_before_startup_succeeds(self):
        from dexmani_real.runtime.operator_input import OperatorCommand
        from dexmani_real.runtime.state import RunEndReason, RuntimeState, SafetyState
        from dexmani_real.teleop.runner import TeleopRunner

        runtime, state, ready = ExperimentConfig(), RuntimeState(), Mock()
        state.arm()
        epoch = state.run_id
        with patch("dexmani_real.teleop.runner.AudioFeedback"), patch("dexmani_real.teleop.runner.KeyboardInput"):
            runner = TeleopRunner(
                state, runtime, Mock(), task_label="test", needs=ObservationNeeds(camera="none", vr=True),
                sensors=SimpleNamespace(channels=SimpleNamespace(vr_ready=ready)), controller=Mock(),
                home_planner=None, vr_process=Mock(), recorder=None,
            )
        runner.keyboard.healthy = True
        runner.keyboard.poll.side_effect = [[], [OperatorCommand.QUIT]]

        def became_ready(**kwargs):
            state.request_stop(reason=RunEndReason.QUIT)
            self.assertFalse(state.command_may_cross_sdk(run_id=epoch, required_safety_state=SafetyState.ARMED))
            return True

        ready.wait.side_effect = became_ready
        self.assertFalse(runner.wait_for_vr())
        self.assertTrue(state.quit_requested)
        self.assertIsNone(state.begin_motion())
        runner.robot.stop.assert_called_once()

    def test_robot_partial_start_and_close_failures_still_close_both_devices(self):
        from dexmani_real.robot.robot import DexManiRobot
        from dexmani_real.runtime.state import RuntimeState

        for cause in ("hand_connect", "arm_close"):
            with self.subTest(cause=cause):
                state = RuntimeState()
                arm, hand = Mock(is_connected=True), Mock(is_connected=True)
                robot = DexManiRobot(state, ExperimentConfig(), hand_factory=lambda cfg: hand)
                if cause == "hand_connect":
                    hand.connect.side_effect = RuntimeError("hand startup failed")
                    with patch("dexmani_real.robot.xarm7.executor.ArmExecutor", return_value=arm), self.assertRaisesRegex(RuntimeError, "hand startup failed"):
                        robot.connect()
                else:
                    robot.owner, robot.arm, robot.hand = threading.get_ident(), arm, hand
                    arm.close.side_effect = RuntimeError("arm close unresolved")
                    with self.assertRaises(RuntimeError):
                        robot.close()
                    self.assertTrue(robot.stop_required)
                arm.close.assert_called_once()
                hand.disconnect.assert_called_once()
                self.assertTrue(state.error_state)
                self.assertFalse(state.requests_allow_motion())

    def test_sensor_shutdown_failures_retain_handles_and_do_not_skip_other_workers(self):
        from dexmani_real.runtime.processes import SensorProcesses, shutdown_local_runtime
        from dexmani_real.runtime.state import RuntimeState, SafetyState

        for cause in ("join_failure", "unconfirmed_exit"):
            with self.subTest(cause=cause):
                state = RuntimeState()
                first, second, first_ready, second_ready = Mock(), Mock(), Mock(), Mock()
                second.is_alive.return_value, second.exitcode = False, 0
                if cause == "join_failure":
                    first.join.side_effect = RuntimeError("join failed")
                else:
                    first.is_alive.return_value, first.exitcode = True, None
                sensors = SensorProcesses(state)
                sensors.started = [(first, first_ready), (second, second_ready)]
                with self.assertLogs("dexmani_real.runtime.processes", level="ERROR"):
                    self.assertFalse(shutdown_local_runtime(state, sensors=sensors, timeout_s=0.))
                second.join.assert_called_once()
                second_ready.clear.assert_called_once()
                self.assertEqual(sensors.started, [(first, first_ready), (second, second_ready)])
                self.assertEqual(state.safety_state, SafetyState.FAULT)
                self.assertTrue(state.error_state)
                if cause == "unconfirmed_exit":
                    first.terminate.assert_called_once()
                    first.kill.assert_called_once()


class OfflineEntryTests(unittest.TestCase):
    def test_help_and_declared_config_never_start_devices_or_load_models(self):
        root = Path(__file__).resolve().parents[1]
        guard = r'''
import importlib.abc
import multiprocessing.process
import runpy
import sys
from unittest.mock import patch

class BlockDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'xarm', 'xhand', 'xhand_sdk', 'pyrealsense2', 'pynput',
            'dexmani_policy', 'torch', 'pinocchio', 'mplib',
        }:
            raise AssertionError('unexpected device/model import: ' + fullname)

sys.meta_path.insert(0, BlockDeviceImports())
script = sys.argv[1]
sys.argv = sys.argv[1:]
with patch('multiprocessing.process.BaseProcess.start', side_effect=AssertionError('process start')), \
     patch('dexmani_real.robot.robot.DexManiRobot.connect', side_effect=AssertionError('robot connect')), \
     patch('dexmani_real.runtime.operator_input.KeyboardInput.start', side_effect=AssertionError('listener start')):
    runpy.run_path(script, run_name='__main__')
'''
        cases = [(name, "--help") for name in (
            "collect_teleop", "run_policy", "replay_episode", "calibrate_camera",
            "calibrate_vr_heading", "keyboard_teleop",
        )] + [("collect_teleop", "--print-config"), ("run_policy", "--print-config")]
        for name, flag in cases:
            with self.subTest(script=name, flag=flag):
                completed = subprocess.run(
                    [sys.executable, "-c", guard, str(root / "scripts" / (name + ".py")), flag],
                    cwd=root, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertTrue(completed.stdout)

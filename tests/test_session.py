import tempfile
import unittest
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.runtime.observation import ObservationNeeds


class AssemblyTests(unittest.TestCase):
    def test_policy_starts_only_requested_sensor_workers(self):
        from dexmani_real.deployment.session import _PolicySession

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
                session = _PolicySession(ExperimentConfig(), info,
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

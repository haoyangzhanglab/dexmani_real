"""Capture 所有权、初始化回滚和真实线程上的晚结果回收。"""

import tempfile
import threading
import unittest
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np

from dexmani_real.config.experiment import ExperimentConfig
from dexmani_real.data.recorder import AsyncEpisodeRecorder, RecordingError
from dexmani_real.deployment.episode import EpisodeEnd, PolicyEpisode
from dexmani_real.deployment.inference import InferenceWorker
from dexmani_real.deployment.session import PolicySession
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.state import RuntimeState
from dexmani_real.teleop.runner import TeleopRunner
from tests.offline_fixtures import make_row, recording_metadata


def make_runner(directory):
    state = RuntimeState()
    state.arm()
    robot = SimpleNamespace(
        stop=Mock(), stop_required=False, reset_hand_feedback=Mock(), prepare_streaming=Mock(),
        arm=SimpleNamespace(update_health=Mock()), hand=None,
    )
    runner = TeleopRunner(
        state, ExperimentConfig(), robot, task_label="offline", control_needs=ObservationNeeds(), record_needs=ObservationNeeds(),
        sensors=SimpleNamespace(channels=None), controller=Mock(), home_planner=None,
        vr_process=None, recording_dir=directory, keyboard=Mock(), audio=Mock(),
    )
    robot.prepare_streaming.side_effect = lambda run_id, **kwargs: state.stop_signal.clear()
    runner.admit_observation = Mock(side_effect=lambda row: [] if row is not None else ["unavailable"])
    return runner


def make_episode(state, model, *, recorder=None, results=None):
    return PolicyEpisode(
        state, ExperimentConfig(),
        SimpleNamespace(control_dt_s=0.1, n_obs_steps=1, hand_history_steps=None, task_name="offline"),
        preparing_epoch=state.run_id, robot=SimpleNamespace(stop_required=False, stop=Mock(), hand=None),
        control_needs=ObservationNeeds(), record_needs=ObservationNeeds(), sensors=SimpleNamespace(channels=None), action_mapper=None,
        model_runtime=model, max_running_s=None, query_ids=count(1), capture_ids=count(1),
        recorder=recorder, results=results,
    )


class CaptureOwnerTests(unittest.TestCase):
    def test_runner_creates_two_recorders_and_no_record_stays_disabled(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "dexmani_real.teleop.runner.read_observation", return_value=make_row(),
        ), patch("dexmani_real.teleop.runner.control_deadline_ns", return_value=10**18), patch(
            "dexmani_real.teleop.runner.snapshot_recording_metadata", side_effect=lambda *args, **kwargs: recording_metadata(),
        ) as metadata:
            runner = make_runner(directory)
            recorders = []
            for _ in range(2):
                runner.start_segment(runner.runtime_state.run_id)
                recorders.append(runner.recorder)
                runner.pause_control("operator")
                runner.end_segment(True, "operator")
                self.assertIsNone(runner.recorder)
            self.assertIsNot(*recorders)
            self.assertEqual(len(list(Path(directory).glob("episode_*"))), 2)
            self.assertTrue(all(not r.writer_pending for r in recorders))
            runner = make_runner(None)
            metadata.reset_mock()
            runner.start_segment(runner.runtime_state.run_id)
            self.assertIsNone(runner.recorder)
            metadata.assert_not_called()
            runner.close()

    def test_preparation_cancel_before_and_after_start(self):
        for after_start in (False, True):
            with self.subTest(after_start=after_start), tempfile.TemporaryDirectory() as directory, patch(
                "dexmani_real.teleop.runner.read_observation", return_value=make_row() if after_start else None,
            ), patch("dexmani_real.teleop.runner.control_deadline_ns", return_value=10**18), patch(
                "dexmani_real.teleop.runner.snapshot_recording_metadata", return_value=recording_metadata(),
            ):
                runner = make_runner(directory)
                runner.robot.prepare_streaming.side_effect = lambda epoch, **kwargs: runner.runtime_state.request_stop()
                runner.start_segment(runner.runtime_state.run_id)
                self.assertIsNone(runner.recorder)
                paths = list(Path(directory).glob("episode_*"))
                self.assertEqual(len(paths), int(after_start))
                if after_start:
                    with h5py.File(paths[0] / "data.h5") as data:
                        self.assertEqual(data["meta"].attrs["num_frames"], 0)
                        self.assertEqual(data["meta"].attrs["technical_status"], "failed")

    def test_policy_start_id_precedes_motion_and_survives_cancel(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "dexmani_real.deployment.episode.snapshot_recording_metadata", return_value=recording_metadata(),
        ):
            state = RuntimeState()
            state.arm()
            recorder = AsyncEpisodeRecorder(directory, control_hz=10)
            episode = make_episode(state, SimpleNamespace(future=None), recorder=recorder)
            episode.start_observation = Mock(return_value=make_row())
            episode.robot.prepare_streaming = Mock(side_effect=lambda epoch, **kwargs: state.request_stop())
            episode.admit()
            self.assertTrue(recorder.start_attempted)
            self.assertIsNone(episode.termination.run_id)
            self.assertEqual(next(episode.capture_ids), 2)
            self.assertEqual(recorder.published_path.name, "episode_001")
            with h5py.File(recorder.published_path / "data.h5") as data:
                self.assertEqual(data["meta"].attrs["num_frames"], 0)
            termination = episode.termination
            episode.finish("different")
            self.assertIs(episode.termination, termination)

    def test_policy_never_replaces_failed_or_live_writer(self):
        for failure in (False, True):
            session = PolicySession(
                ExperimentConfig(), SimpleNamespace(),
                control_needs=ObservationNeeds(), record_needs=ObservationNeeds(),
            )
            session.runtime_state = RuntimeState()
            session.runtime_state.arm()
            session.start_epoch = session.runtime_state.run_id
            session.recorder = SimpleNamespace(
                check_error=Mock(side_effect=RecordingError("failed") if failure else None),
                is_recording=False, writer_pending=True,
            )
            old = session.recorder
            with patch("dexmani_real.deployment.session.AsyncEpisodeRecorder") as constructor:
                with self.assertRaises(RecordingError):
                    session.advance_episode()
                constructor.assert_not_called()
            self.assertIs(session.recorder, old)
            self.assertEqual(session.completed, 0)

    def test_only_running_episodes_increment_round_count(self):
        session = PolicySession(ExperimentConfig(), SimpleNamespace(), control_needs=ObservationNeeds(), record_needs=ObservationNeeds())
        session.runtime_state = RuntimeState()
        session.model = SimpleNamespace(future=None)
        session.keyboard = Mock()
        for run_id, expected in ((None, 0), (3, 1)):
            episode = SimpleNamespace(termination=None)
            episode.step = lambda: setattr(episode, "termination", EpisodeEnd(run_id, "quit", "quit", 0))
            session.episode = episode
            session.advance_episode()
            self.assertEqual(session.completed, expected)
            self.assertIsNone(session.episode)


class CleanupTests(unittest.TestCase):
    def test_partial_teleop_initialization_releases_acquired_resources(self):
        from contextlib import ExitStack
        from dexmani_real.teleop.session import TeleopSession

        robot, keyboard, audio, sensors = Mock(), Mock(), Mock(), Mock()
        sensors.shutdown.return_value = True
        runtime = ExperimentConfig()
        with ExitStack() as stack:
            replacements = {
                "resolve_runtime_table": lambda value: value,
                "load_vr_transform": Mock(return_value=SimpleNamespace(transform=np.eye(3))),
                "ActionMapper.for_mode": Mock(), "build_home_planner": Mock(),
                "load_optional_camera_extrinsics": Mock(return_value=None),
                "SensorChannels": Mock(), "SensorProcesses": Mock(return_value=sensors),
                "DexManiRobot": Mock(return_value=robot), "KeyboardInput": Mock(return_value=keyboard),
                "AudioFeedback": Mock(return_value=audio),
                "TeleopRunner": Mock(side_effect=RuntimeError("partial initialization")),
                "configure_logging": Mock(),
            }
            for name, value in replacements.items():
                stack.enter_context(patch("dexmani_real.teleop.session." + name, value))
            stack.enter_context(patch.object(TeleopSession, "build_hand_retargeter", return_value=None))
            stack.enter_context(
                patch(
                    "dexmani_real.teleop.controller.make_arm_fk",
                    return_value=Mock(),
                )
            )
            with self.assertLogs("dexmani_real.teleop.session", level="ERROR"):
                self.assertEqual(TeleopSession(runtime).run(), 1)
        audio.close.assert_called_once()
        keyboard.stop.assert_called_once()
        keyboard.start.assert_not_called()
        robot.connect.assert_not_called()
        robot.close.assert_called_once()
        sensors.shutdown.assert_called_once()

    def test_multiple_cleanup_errors_are_retained_and_remaining_closes_attempted(self):
        session = PolicySession(ExperimentConfig(), SimpleNamespace(), control_needs=ObservationNeeds(), record_needs=ObservationNeeds())
        session.runtime_state = RuntimeState()
        session.robot = SimpleNamespace(stop_required=False, close=Mock(side_effect=OSError("robot close")))
        session.keyboard = SimpleNamespace(stop=Mock(side_effect=OSError("keyboard close")))
        session.sensors = SimpleNamespace(shutdown=Mock(side_effect=OSError("sensor close")))
        session.recorder = SimpleNamespace(finish_request=None, close=Mock(side_effect=OSError("recorder close")))
        session.model = SimpleNamespace(close=Mock(side_effect=OSError("model close")),
                                        close_result=Mock(return_value=dict(status="pending")))
        session.results = Mock()
        with self.assertLogs("dexmani_real.runtime.processes", level="ERROR") as logs:
            self.assertEqual(session.close(clean=False, failure=RuntimeError("startup")), 1)
        self.assertTrue(any("robot close" in line for line in logs.output))
        self.assertTrue(any("keyboard close" in line for line in logs.output))
        self.assertTrue(any("sensor close" in line for line in logs.output))
        for close in (session.robot.close, session.keyboard.stop, session.sensors.shutdown,
                      session.recorder.close, session.model.close, session.model.close_result):
            close.assert_called_once()
        stages = [item["stage"] for item in session.results.record_errors.call_args.args[0]]
        self.assertEqual(stages, ["model_close_submit", "recorder_close"])
        self.assertEqual(session.results.finish_session.call_args.kwargs["model_close"]["status"], "pending")


class InferenceOwnershipTests(unittest.TestCase):
    def worker(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def predict(observation):
            calls.append(("predict", threading.get_ident()))
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test gate")
            raise ValueError("late model error")

        def close():
            calls.append(("close", threading.get_ident()))

        model = SimpleNamespace(warmup=lambda: None, predict=predict, close=close)
        worker = InferenceWorker(lambda: model, build_observation=lambda rows: rows)
        worker.submit("load")
        worker.future.result(timeout=1)
        worker.poll()
        self.addCleanup(lambda: worker.close_result(2))
        self.addCleanup(worker.close)
        self.addCleanup(release.set)
        return worker, entered, release, calls

    def test_worker_close_serializes_late_task_and_retains_pending_or_failed(self):
        worker, entered, release, calls = self.worker()
        identity = dict(run_id=7, query_id=2)
        worker.submit("predict", [], query_context=identity)
        self.assertTrue(entered.wait(1))
        close_future = worker.close()
        self.assertIs(worker.close(), close_future)
        self.assertEqual(worker.close_result()["status"], "pending")
        release.set()
        result = worker.close_result(2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["retired_task"]["query_context"], identity)
        self.assertEqual(result["retired_task"]["exception_type"], "ValueError")
        self.assertEqual([name for name, _ in calls], ["predict", "close"])
        self.assertEqual(calls[0][1], calls[1][1])

    def test_ended_episode_owns_query_until_late_error_is_reclaimed(self):
        worker, entered, release, _ = self.worker()
        state = RuntimeState()
        state.arm()
        state.stop_signal.clear()
        epoch, stamp = state.begin_motion()
        results = Mock()
        episode = make_episode(state, worker, results=results)
        episode.run_id, episode.started_ns = epoch, stamp
        episode.query = SimpleNamespace(run_id=epoch, query_id=3)
        worker.submit("predict", [], query_context=dict(run_id=epoch, query_id=3))
        self.assertTrue(entered.wait(1))
        episode.finish("operator")
        termination = episode.termination
        self.assertIsNotNone(episode.query)
        release.set()
        worker.future.result(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "retired model predict failed"):
            episode.poll_model()
        self.assertIsNone(episode.query)
        self.assertIs(episode.termination, termination)
        self.assertTrue(state.error_state)
        self.assertEqual(results.record_errors.call_args.args[0][0]["query_context"], dict(run_id=epoch, query_id=3))


class PolicyBoundaryTests(unittest.TestCase):
    def running_episode(self, *, recorder=None):
        state = RuntimeState()
        state.arm()
        state.stop_signal.clear()
        run_id, started_ns = state.begin_motion()
        episode = make_episode(state, SimpleNamespace(future=None), recorder=recorder)
        episode.run_id, episode.started_ns = run_id, started_ns
        episode.robot.arm = SimpleNamespace(update_health=Mock())
        episode.robot.send_action = Mock()
        return episode

    def test_record_only_sources_fail_before_submission_without_changing_control(self):
        from dexmani_real.deployment.episode import failure_reason
        from dexmani_real.deployment.observation import policy_needs
        from dexmani_real.runtime.observation import control_deadline_ns
        from dexmani_real.runtime.state import RunEndReason

        for missing in ("camera", "tactile_dense"):
            row = make_row()
            if missing == "camera":
                row.camera = None
            else:
                row.hand["tactile_dense_timestamp_ns"] = 0
                row.hand["tactile_dense"] = np.nan
            episode = self.running_episode()
            episode.control_needs = policy_needs(("joint_state",))
            episode.record_needs = ObservationNeeds(camera="rgbd", required_tactile=("aggregate", "dense"))
            episode.needs = episode.control_needs.union(episode.record_needs)
            with patch("dexmani_real.deployment.episode.read_observation", return_value=row) as read, patch(
                "time.monotonic_ns", return_value=row.observation_timestamp_ns,
            ):
                self.assertIs(episode.read_observation(), row)
                deadline = control_deadline_ns(row, episode.runtime, episode.control_needs)
                episode.recorder = Mock()
                with self.assertRaises(RecordingError) as error:
                    episode.read_observation()
                self.assertEqual(failure_reason(error.exception), RunEndReason.RECORDING_FAILURE)
                self.assertEqual(read.call_count, 2)  # One sample for each of the two ticks.
                self.assertEqual(control_deadline_ns(row, episode.runtime, episode.control_needs), deadline)
            self.assertEqual(episode.observation_read_issues, [])
            episode.robot.send_action.assert_not_called()
            episode.robot.arm.update_health.assert_called_once_with(health_until_ns=deadline)

    def test_action_context_only_updates_after_normal_current_submission(self):
        from collections import deque
        from dexmani_real.robot.commands import RobotCommand, RobotError

        for failed in (True, False):
            row = make_row()
            episode = self.running_episode()
            episode.plan = SimpleNamespace(expires_at_ns=2_000_000_000, segment_end=2)
            episode.next_step_ns = 1_050_000_000
            episode.previous_arm = np.zeros(7)
            episode.hand_history = deque([np.zeros(12)], maxlen=3)
            command = RobotCommand(episode.run_id, np.ones(7), np.ones(12))
            episode.command_for_slot = Mock(return_value=(command, None))
            episode.pending_frame = (row, None)
            if failed:
                episode.robot.send_action.side_effect = RobotError("partial send")
            with patch("time.monotonic_ns", return_value=1_000_000_000):
                if failed:
                    with self.assertRaises(RobotError):
                        episode.execute_slot(row, 0)
                    self.assertIs(episode.pending_frame[1], command)
                    np.testing.assert_array_equal(episode.previous_arm, 0)
                    self.assertEqual(len(episode.hand_history), 1)
                else:
                    episode.execute_slot(row, 0)
                    np.testing.assert_array_equal(episode.previous_arm, command.arm_qpos)
                    np.testing.assert_array_equal(episode.hand_history[-1], command.hand_qpos)
                    self.assertEqual(len(episode.hand_history), 2)
                    self.assertIsNone(episode.pending_frame)

    def test_pre_submission_cancellation_preserves_wait_and_stop_error_keeps_capture_result(self):
        from dexmani_real.robot.commands import RobotCommand
        from dexmani_real.runtime.state import RunEndReason

        with tempfile.TemporaryDirectory() as directory:
            recorder = AsyncEpisodeRecorder(directory, control_hz=10)
            recorder.start_episode(task_label="offline", collection_source="policy_rollout", metadata=recording_metadata())
            episode = self.running_episode(recorder=recorder)
            episode.pending_frame = (make_row(), None)
            episode.plan = SimpleNamespace(expires_at_ns=2_000_000_000)
            command = RobotCommand(episode.run_id, np.ones(7), np.ones(12))
            episode.command_for_slot = Mock(return_value=(command, None))
            episode.robot.stop.side_effect = OSError("cleanup stop failed")
            episode.runtime_state.request_stop(reason=RunEndReason.OPERATOR)
            with self.assertRaisesRegex(OSError, "cleanup stop failed"):
                episode.execute_slot(make_row(), 0)
            episode.robot.send_action.assert_not_called()
            with h5py.File(recorder.published_path / "data.h5") as data:
                self.assertTrue(np.isnan(data["action_arm_joint_target"][:]).all())
                self.assertEqual(data["meta"].attrs["technical_status"], "normal")
                self.assertEqual(data["meta"].attrs["termination_reason"], "operator")
                self.assertIn("cleanup stop failed", data["meta/termination_details"][()].decode())

    def test_retired_success_cannot_update_plan_and_reset_runs_on_model_thread(self):
        from dexmani_real.deployment.inference import ModelResult

        episode = self.running_episode()
        old_id = episode.run_id
        episode.query = SimpleNamespace(run_id=old_id, query_id=9, result=None)
        query = episode.query
        result = ModelResult(np.ones((2, 19)), None, 1, 3, dict(run_id=old_id, query_id=9), 2)
        episode.model.poll = Mock(return_value=("predict", result))
        episode.model.future = object()
        episode.finish("operator")
        episode.poll_model()
        self.assertIsNone(query.result)
        self.assertIsNone(episode.plan)
        self.assertIsNone(episode.query)

        entered, release = threading.Event(), threading.Event()
        calls = []

        def predict(value):
            calls.append(("predict", threading.get_ident()))
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test gate")
            return value

        def reset_cache():
            calls.append(("cache_reset", threading.get_ident()))

        model = SimpleNamespace(
            warmup=lambda: None, predict=predict,
            reset_episode=lambda: calls.append(("model_reset", threading.get_ident())),
            close=lambda: None,
        )
        worker = InferenceWorker(lambda: model, build_observation=lambda value: value, reset_observation=reset_cache)
        try:
            worker.submit("load")
            worker.future.result(1)
            worker.poll()
            worker.submit("predict", [])
            self.assertTrue(entered.wait(1))
            with self.assertRaises(RuntimeError):
                worker.submit("reset_episode")
            self.assertEqual([name for name, _ in calls], ["predict"])
            release.set()
            worker.future.result(1)
            worker.poll()
            worker.submit("reset_episode")
            worker.future.result(1)
            worker.poll()
        finally:
            release.set()
            worker.close()
            worker.close_result(2)
        self.assertEqual([name for name, _ in calls], ["predict", "cache_reset", "model_reset"])
        self.assertEqual(len({thread for _, thread in calls}), 1)
        self.assertNotEqual(calls[0][1], threading.get_ident())

    def test_process_failures_keep_control_and_recording_responsibility(self):
        from dexmani_real.robot.commands import RobotError

        for dynamic_control in (False, True):
            session = PolicySession(
                ExperimentConfig(), SimpleNamespace(),
                control_needs=ObservationNeeds(camera="rgb" if dynamic_control else "none"),
                record_needs=ObservationNeeds(camera="rgbd"),
            )
            session.keyboard = SimpleNamespace(estop_latched=False, healthy=True)
            session.sensors = SimpleNamespace(check=lambda *, names: "camera" not in names)
            with self.assertRaises(RobotError if dynamic_control else RecordingError):
                session.check_services()
        session.control_needs = ObservationNeeds(camera_static=True)
        session.record_needs = ObservationNeeds()
        self.assertTrue(session.check_services())

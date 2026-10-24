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
    return TeleopRunner(
        state, ExperimentConfig(), robot, task_label="offline", needs=ObservationNeeds(),
        sensors=SimpleNamespace(channels=None), controller=Mock(), home_planner=None,
        vr_process=None, recording_dir=directory, keyboard=Mock(), audio=Mock(),
    )


def make_episode(state, model, *, recorder=None, results=None):
    return PolicyEpisode(
        state, ExperimentConfig(),
        SimpleNamespace(control_dt_s=0.1, n_obs_steps=1, hand_history_steps=None, task_name="offline"),
        preparing_epoch=state.run_id, robot=SimpleNamespace(stop_required=False, stop=Mock(), hand=None),
        needs=ObservationNeeds(), sensors=SimpleNamespace(channels=None), action_mapper=None,
        model_runtime=model, max_running_s=None, query_ids=count(1), capture_ids=count(1),
        recorder=recorder, results=results,
    )


class CaptureOwnerTests(unittest.TestCase):
    def test_runner_creates_two_recorders_and_no_record_stays_disabled(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "dexmani_real.teleop.runner.read_observation", return_value=make_row(),
        ), patch("dexmani_real.teleop.runner.upstream_deadline_ns", return_value=10**18), patch(
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
            ), patch("dexmani_real.teleop.runner.upstream_deadline_ns", return_value=10**18), patch(
                "dexmani_real.teleop.runner.snapshot_recording_metadata", return_value=recording_metadata(),
            ):
                runner = make_runner(directory)
                runner.robot.prepare_streaming.side_effect = lambda epoch: runner.runtime_state.request_stop()
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
            episode.robot.prepare_streaming = Mock(side_effect=lambda epoch: state.request_stop())
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
            session = PolicySession(ExperimentConfig(), SimpleNamespace(), needs=ObservationNeeds())
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
        session = PolicySession(ExperimentConfig(), SimpleNamespace(), needs=ObservationNeeds())
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
            with self.assertLogs("dexmani_real.teleop.session", level="ERROR"):
                self.assertEqual(TeleopSession(runtime).run(), 1)
        audio.close.assert_called_once()
        keyboard.stop.assert_called_once()
        keyboard.start.assert_not_called()
        robot.connect.assert_not_called()
        robot.close.assert_called_once()
        sensors.shutdown.assert_called_once()

    def test_multiple_cleanup_errors_are_retained_and_remaining_closes_attempted(self):
        session = PolicySession(ExperimentConfig(), SimpleNamespace(), needs=ObservationNeeds())
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

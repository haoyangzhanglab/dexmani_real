import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from fakes import FakeClock, FakeRecorder, camera_metadata, make_runner, observation, start_runner
from dexmani_real.data.recorder import RecordingError
from dexmani_real.data.results import EpisodeResults
from dexmani_real.deployment.schedule import ActionPlan
from dexmani_real.robot.commands import DispatchError, DispatchResult, DispatchStatus
from dexmani_real.robot.home import HomeResult
from dexmani_real.runtime.operator_input import OperatorCommand as Key


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        for context in (
            patch("time.monotonic_ns", self.clock.monotonic_ns),
            patch("dexmani_real.deployment.runner.snapshot_recording_metadata", return_value=camera_metadata()),
        ):
            context.start()
            self.addCleanup(context.stop)

    def predictions(self, count=8):
        return np.arange(count * 19, dtype=np.float64).reshape(count, 19) / 1000

    def test_sync_wait_has_no_action_and_resume_records_actual_clock(self):
        recorder = FakeRecorder()
        runner = make_runner(self.clock, recorder=recorder)
        start_runner(runner)
        self.clock.advance(.05)
        runner.step()
        self.assertEqual(runner.robot.sent, [])
        self.assertTrue(all(item[1] is None for item in recorder.frames))
        runner.model.complete(self.predictions())
        self.clock.advance(.01)
        resumed = self.clock.now
        runner.step()
        self.assertEqual(len(runner.robot.sent), 1)
        self.assertEqual(recorder.frames[-1][3]["control_ns"], resumed)
        self.assertEqual(runner.next_step_ns, resumed + runner.dt_ns)
        self.assertEqual(recorder.frames[-1][3]["prediction_index"], 0)
        self.clock.advance(.05)
        runner.step()
        self.clock.advance(.05)
        runner.step()
        self.assertEqual(len(runner.robot.sent), 2)
        self.assertIsNone(recorder.frames[-1][1])

    def test_async_early_and_late_handoff_keep_actual_indices(self):
        for delay, expected_index in ((1, 1), (3, 3)):
            with self.subTest(delay=delay):
                runner = make_runner(self.clock, mode="async")
                start_runner(runner)
                runner.model.complete(self.predictions())
                self.clock.advance(.05)
                runner.step()  # bootstrap action, slot 1
                self.clock.advance(.05)
                runner.step()  # prefetch at slot 2, handoff at 3
                query_slot = runner.query.slot
                for _ in range(delay - 1):
                    self.clock.advance(.05)
                    runner.step()  # old unexpired tail
                runner.model.complete(self.predictions() + 1)
                runner.step()  # completion alone cannot send before the next due slot
                count = len(runner.robot.sent)
                self.clock.advance(.05)
                runner.step()
                self.assertEqual(len(runner.robot.sent), count + 1)
                self.assertEqual(runner.slot_timing["prediction_index"], expected_index)
                self.assertEqual(runner.plan.anchor_slot, query_slot)
                np.testing.assert_array_equal(runner.robot.sent[-1][0].arm_qpos,
                                              (self.predictions() + 1)[expected_index, :7])

    def test_exhaustion_and_missed_slot_never_repeat_actions(self):
        for missed in (False, True):
            with self.subTest(missed=missed):
                runner = make_runner(self.clock, mode="async")
                start_runner(runner)
                runner.model.complete(self.predictions(2))
                self.clock.advance(.05)
                runner.step()
                self.clock.advance(.05)
                runner.step()
                count = len(runner.robot.sent)
                self.clock.advance(.15 if missed else .05)
                runner.step()
                self.assertIsNone(runner.run_id)
                self.assertEqual(len(runner.robot.sent), count)

    def test_rtc_frozen_command_is_not_remapped(self):
        runner = make_runner(self.clock, mode="rtc")
        start_runner(runner)
        runner.model.complete(self.predictions())
        self.clock.advance(.05)
        runner.step()
        self.clock.advance(.05)
        runner.step()
        frozen = runner.plan.frozen[2]
        self.assertIs(runner.robot.sent[-1][0], frozen)
        prefix = runner.model.calls[-1][2]["rtc_prefix"]
        np.testing.assert_array_equal(prefix[0], np.concatenate((frozen.arm_qpos, frozen.hand_qpos)))
        self.assertEqual(runner.action_mapper.maps, 2)

    def test_cancelled_reset_cannot_start_next_run(self):
        runner = make_runner(self.clock)
        runner.request_start()
        runner.step()
        runner.request_stop()
        runner.step()
        self.assertIsNone(runner.run_id)
        runner.model.complete()
        runner.step()
        self.assertIsNone(runner.run_id)
        self.assertIsNone(runner.preparing_epoch)
        self.assertEqual(runner.robot.sent, [])
        runner.step()  # consume confirmed stop before a fresh B
        runner.request_start()
        runner.step()
        self.assertIsNone(runner.run_id)
        self.assertEqual([call[0] for call in runner.model.calls], ["reset_episode", "reset_episode"])

    def test_retired_prediction_must_be_reclaimed_before_new_reset(self):
        runner = make_runner(self.clock)
        start_runner(runner)
        old_query = runner.query
        runner.request_stop()
        runner.step()
        self.assertIs(runner.query, old_query)
        runner.request_start()
        runner.step()
        self.assertEqual([call[0] for call in runner.model.calls], ["reset_episode", "predict"])
        runner.model.complete(error=RuntimeError("retired prediction failed"))
        runner.step()
        self.assertIsNone(runner.query)
        self.assertEqual(runner.robot.sent, [])
        runner.request_start()
        runner.step()
        self.assertEqual(runner.model.calls[-1][0], "reset_episode")

    def test_writer_start_failure_keeps_attempt_and_does_not_count_run(self):
        with tempfile.TemporaryDirectory() as directory:
            results = EpisodeResults(directory, "policy")
            recorder = FakeRecorder()
            recorder.start_error = RecordingError("writer not ready")
            recorder.finish_error = RecordingError("writer not ready")
            runner = make_runner(self.clock, recorder=recorder, results=results)
            runner.request_start()
            runner.step()
            runner.model.complete()
            with self.assertRaises(RecordingError):
                runner.step()
            self.assertEqual(runner.completed, 0)
            self.assertEqual(runner.recording_attempts, 1)
            self.assertEqual(len(recorder.finishes), 1)
            result = results.session["episodes"][0]
            self.assertFalse(result["started"])
            self.assertEqual(result["termination_reason"], "preparation_failed")
            self.assertEqual(result["raw"]["status"], "failed")
            self.assertEqual(result["raw"]["staging_path"], str(recorder.staging_path))

    def test_zero_row_cancelled_capture_and_next_attempt_have_distinct_names(self):
        recorder = FakeRecorder()
        runner = make_runner(self.clock, recorder=recorder)
        real_start = recorder.start_episode
        def cancelled_start(**kwargs):
            real_start(**kwargs)
            runner.request_stop()
        recorder.start_episode = cancelled_start
        start_runner(runner)
        self.assertEqual(len(recorder.frames), 0)
        self.assertEqual(recorder.finishes[0]["reason"], "start_cancelled")
        self.assertEqual(runner.completed, 0)
        runner.step()
        recorder.start_episode = real_start
        start_runner(runner)
        runner.request_stop()
        runner.step()
        self.assertEqual(runner.completed, 1)
        self.assertEqual([item["episode_name"] for item in recorder.starts], ["episode_001", "episode_002"])

    def test_partial_dispatch_is_recorded_once_before_failure(self):
        recorder = FakeRecorder()
        runner = make_runner(self.clock, recorder=recorder)
        start_runner(runner)
        runner.model.complete(self.predictions())
        receipt = DispatchResult(DispatchStatus.SUBMITTED, DispatchStatus.UNKNOWN, self.clock.now, (runner.run_id, 1))
        runner.robot.dispatch_error = DispatchError("hand failed", receipt)
        self.clock.advance(.05)
        with self.assertRaises(DispatchError):
            runner.step()
        self.assertEqual(len(recorder.frames), 2)
        self.assertIs(recorder.frames[-1][2], receipt)
        self.assertEqual(len(recorder.finishes), 1)
        self.assertEqual(runner.completed, 1)
        runner.finish_episode("again")
        self.assertEqual(len(recorder.finishes), 1)

    def test_queue_failure_is_not_retried(self):
        recorder = FakeRecorder()
        runner = make_runner(self.clock, recorder=recorder)
        start_runner(runner)
        recorder.add_frame = Mock(side_effect=RecordingError("queue full"))
        self.clock.advance(.05)
        with self.assertRaises(RecordingError):
            runner.step()
        recorder.add_frame.assert_called_once()
        self.assertEqual(len(recorder.finishes), 1)

    def test_stop_failure_still_finalizes_with_unresolved_evidence(self):
        recorder = FakeRecorder()
        runner = make_runner(self.clock, recorder=recorder)
        start_runner(runner)
        runner.robot.stop_required = True
        runner.robot.stop_error = RuntimeError("stop unresolved")
        runner.request_stop()
        with self.assertRaises(RuntimeError):
            runner.step()
        self.assertFalse(recorder.finishes[0]["arm_execution"]["robot_stopped"])
        self.assertEqual(len(recorder.finishes), 1)

    def test_stop_quit_suppress_begin_home_and_tactile_in_entire_batch(self):
        for terminal in (Key.STOP, Key.QUIT):
            for batch in ([Key.BEGIN, Key.HOME, terminal], [terminal, Key.BEGIN, Key.HOME]):
                with self.subTest(batch=batch):
                    runner = make_runner(self.clock)
                    runner.run_home = Mock()
                    runner.handle_command_batch(batch)
                    self.assertFalse(runner.start_requested)
                    runner.run_home.assert_not_called()

    def test_home_drains_stale_preparation_but_keeps_stop(self):
        runner = make_runner(self.clock)
        def home(*args, **kwargs):
            runner.keyboard.signals.extend((Key.HOME, Key.BEGIN, Key.RESET_TACTILE, Key.STOP))
            return HomeResult(True)
        with patch("dexmani_real.deployment.runner.home_robot", side_effect=home):
            runner.handle_command_batch([Key.HOME, Key.BEGIN])
        self.assertFalse(runner.start_requested)
        self.assertEqual(runner.keyboard.signals, [Key.STOP])

import tempfile
import unittest
from concurrent.futures import CancelledError
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from fakes import (
    FakeClock,
    FakeRecorder,
    camera_metadata,
    file_contents,
    make_episode,
    make_session,
    observation,
    start_episode,
    start_session_episode,
)
from dexmani_real.data.recorder import AsyncEpisodeRecorder, RecordingError
from dexmani_real.data.results import EpisodeResults
from dexmani_real.deployment.schedule import ActionPlan
from dexmani_real.robot.commands import DispatchError, DispatchResult, DispatchStatus
from dexmani_real.robot.home import HomeResult
from dexmani_real.runtime.operator_input import OperatorCommand as Key
from dexmani_real.runtime.state import RunEndReason, SafetyState


class EpisodeTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        for context in (
            patch("time.monotonic_ns", self.clock.monotonic_ns),
            patch("dexmani_real.deployment.episode.snapshot_recording_metadata", return_value=camera_metadata()),
            patch("dexmani_real.deployment.episode.read_observation",
                  side_effect=lambda channels, runtime, robot, **kw: observation(
                      self.clock, runtime, camera=kw["needs"].camera == "rgbd")),
        ):
            context.start()
            self.addCleanup(context.stop)

    def predictions(self, count=8):
        return np.arange(count * 19, dtype=np.float64).reshape(count, 19) / 1000

    def test_sync_wait_has_no_action_and_resume_records_actual_clock(self):
        recorder = FakeRecorder()
        episode = make_episode(self.clock, recorder=recorder)
        start_episode(episode)
        self.clock.advance(.05)
        episode.step()
        self.assertEqual(episode.robot.sent, [])
        self.assertTrue(all(item[1] is None for item in recorder.frames))
        episode.model.complete(self.predictions())
        self.clock.advance(.01)
        resumed = self.clock.now
        episode.step()
        self.assertEqual(len(episode.robot.sent), 1)
        self.assertEqual(recorder.frames[-1][3]["control_ns"], resumed)
        self.assertEqual(episode.next_step_ns, resumed + episode.dt_ns)
        self.assertEqual(recorder.frames[-1][3]["prediction_index"], 0)
        self.clock.advance(.05)
        episode.step()
        self.clock.advance(.05)
        episode.step()
        self.assertEqual(len(episode.robot.sent), 2)
        self.assertIsNone(recorder.frames[-1][1])

    def test_async_early_and_late_handoff_keep_actual_indices(self):
        for delay, expected_index in ((1, 1), (3, 3)):
            with self.subTest(delay=delay):
                episode = make_episode(self.clock, mode="async")
                start_episode(episode)
                episode.model.complete(self.predictions())
                self.clock.advance(.05)
                episode.step()  # bootstrap action, slot 1
                self.clock.advance(.05)
                episode.step()  # prefetch at slot 2, handoff at 3
                query_slot = episode.query.slot
                for _ in range(delay - 1):
                    self.clock.advance(.05)
                    episode.step()  # old unexpired tail
                episode.model.complete(self.predictions() + 1)
                episode.step()  # completion alone cannot send before the next due slot
                count = len(episode.robot.sent)
                self.clock.advance(.05)
                episode.step()
                self.assertEqual(len(episode.robot.sent), count + 1)
                self.assertEqual(episode.slot_timing["prediction_index"], expected_index)
                self.assertEqual(episode.plan.anchor_slot, query_slot)
                np.testing.assert_array_equal(episode.robot.sent[-1][0].arm_qpos,
                                              (self.predictions() + 1)[expected_index, :7])

    def test_exhaustion_and_missed_slot_never_repeat_actions(self):
        for missed in (False, True):
            with self.subTest(missed=missed):
                episode = make_episode(self.clock, mode="async")
                start_episode(episode)
                episode.model.complete(self.predictions(2))
                self.clock.advance(.05)
                episode.step()
                self.clock.advance(.05)
                episode.step()
                count = len(episode.robot.sent)
                self.clock.advance(.15 if missed else .05)
                episode.step()
                self.assertIsNone(episode.run_id)
                self.assertEqual(len(episode.robot.sent), count)

    def test_rtc_frozen_command_is_not_remapped(self):
        episode = make_episode(self.clock, mode="rtc")
        start_episode(episode)
        episode.model.complete(self.predictions())
        self.clock.advance(.05)
        episode.step()
        self.clock.advance(.05)
        episode.step()
        frozen = episode.plan.frozen[2]
        self.assertIs(episode.robot.sent[-1][0], frozen)
        prefix = episode.model.calls[-1][2]["rtc_prefix"]
        np.testing.assert_array_equal(prefix[0], np.concatenate((frozen.arm_qpos, frozen.hand_qpos)))
        self.assertEqual(episode.action_mapper.maps, 2)

    def test_rtc_checkpoint_cancels_without_operator_reentry(self):
        episode = make_episode(self.clock, mode="rtc")
        start_episode(episode)
        session = make_session(episode)
        session.keyboard.signals = [Key.HOME, Key.BEGIN]
        session.run_home = Mock(side_effect=AssertionError("HOME during prefix"))
        episode.plan = ActionPlan(self.predictions(), 0, 2, self.clock.now + 1_000_000_000, {}, 1)
        episode.robot.check = episode.runtime_state.request_stop
        prefix, commands, issue = episode.prepare_prefix(observation(self.clock), 1)
        self.assertEqual(issue.detail, "prefix_authority_revoked")
        self.assertIsNone(prefix)
        self.assertIsNone(commands)
        session.run_home.assert_not_called()
        self.assertEqual(session.keyboard.signals, [Key.HOME, Key.BEGIN])

    def test_cancelled_reset_cannot_start_next_run(self):
        session = make_session(make_episode(self.clock))
        session.request_start()
        session.advance_episode()
        old_episode = session.episode
        session.request_stop()
        session.advance_episode()
        self.assertIsNone(old_episode.run_id)
        self.assertIsNotNone(old_episode.termination)
        self.assertEqual(old_episode.termination.reason, "start_cancelled")
        self.assertFalse(session.model.future.done())
        self.assertFalse(session.request_start())
        session.model.complete()
        session.advance_episode()
        self.assertIsNone(session.episode)
        self.assertIsNone(old_episode.preparing_epoch)
        self.assertEqual(session.robot.sent, [])
        self.assertEqual(session.completed, 0)
        self.assertTrue(session.request_start())
        session.advance_episode()
        self.assertIsNot(session.episode, old_episode)
        self.assertIsNone(session.episode.run_id)
        self.assertEqual([call[0] for call in session.model.calls], ["reset_episode", "reset_episode"])

    def test_retired_prediction_must_be_reclaimed_before_new_reset(self):
        session = make_session(make_episode(self.clock))
        start_session_episode(session)
        old_episode, old_query = session.episode, session.episode.query
        session.request_stop()
        session.advance_episode()
        self.assertIs(session.episode, old_episode)
        self.assertIs(old_episode.query, old_query)
        self.assertFalse(session.request_start())
        self.assertEqual([call[0] for call in session.model.calls], ["reset_episode", "predict"])
        session.model.complete(self.predictions())
        self.assertFalse(session.request_start())  # done is not reclaimed
        session.advance_episode()
        self.assertIsNone(session.episode)
        self.assertIsNone(old_episode.query)
        self.assertEqual(session.robot.sent, [])
        self.assertEqual(session.completed, 1)
        self.assertTrue(session.request_start())
        session.advance_episode()
        self.assertEqual(session.model.calls[-1][0], "reset_episode")

    def test_retired_model_errors_preserve_end_and_prevent_reuse(self):
        for operation in ("reset_episode", "predict"):
            for error_type in (CancelledError, RuntimeError):
                with self.subTest(operation=operation, error_type=error_type), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    results = EpisodeResults(root / "results", "policy")
                    recorder = AsyncEpisodeRecorder(root / "raw", control_hz=20)
                    self.addCleanup(recorder.close)
                    session = make_session(make_episode(self.clock, recorder=recorder, results=results))
                    session.request_start()
                    session.advance_episode()
                    if operation == "predict":
                        session.model.complete()
                        session.advance_episode()
                    old_episode = session.episode
                    context = dict(session.model.context)
                    session.request_stop()
                    session.advance_episode()
                    ended = old_episode.termination
                    self.assertIsNotNone(ended)
                    expected = "operator" if operation == "predict" else "start_cancelled"
                    self.assertEqual(ended.reason, expected)
                    episodes = deepcopy(results.session["episodes"])
                    before = file_contents(root / "raw")
                    error = error_type("retired model failed")
                    session.model.complete(error=error)
                    with self.assertRaisesRegex(RuntimeError, "retired model") as caught:
                        session.advance_episode()
                    self.assertIs(caught.exception.__cause__, error)
                    self.assertIs(old_episode.termination, ended)
                    self.assertIsNone(old_episode.query)
                    self.assertIsNone(old_episode.preparing_epoch)
                    self.assertIsNone(session.episode)
                    self.assertTrue(session.runtime_state.error_state)
                    self.assertNotEqual(session.runtime_state.safety_state, SafetyState.FAULT)
                    self.assertFalse(session.request_start())
                    self.assertEqual(session.robot.sent, [])
                    self.assertEqual(session.completed, int(operation == "predict"))
                    old_episode.finish("second cleanup")
                    session.advance_episode(shutdown=True)
                    self.assertEqual(results.session["episodes"], episodes)
                    self.assertEqual(len(results.session["errors"]), 1)
                    detail = results.session["errors"][0]
                    self.assertEqual(detail["operation"], operation)
                    self.assertEqual(detail["query_context"], context)
                    self.assertEqual(detail["exception_type"], error_type.__name__)
                    self.assertEqual(detail["message"], str(error))
                    self.assertEqual(file_contents(root / "raw"), before)
                    if operation == "predict":
                        self.assertEqual(len(before), 2)
                        self.assertEqual(session.runtime_state.run_ended_reason, RunEndReason.OPERATOR)

    def test_reset_cancellation_during_poll_preserves_cancellation(self):
        for request in ("request_stop", "request_quit", "request_emergency_stop"):
            for error in (None, CancelledError("reset failed"), RuntimeError("reset failed")):
                with self.subTest(request=request, error=error):
                    episode = make_episode(self.clock)
                    episode.results = Mock()
                    episode.step()
                    episode.model.complete(error=error)
                    poll = episode.model.poll
                    def stop_during_poll():
                        getattr(episode.runtime_state, request)()
                        return poll()
                    episode.model.poll = stop_during_poll
                    if error is None:
                        episode.step()
                    else:
                        with self.assertRaisesRegex(RuntimeError, "retired model") as caught:
                            episode.step()
                        self.assertIs(caught.exception.__cause__, error)
                    self.assertEqual(episode.termination.reason, "start_cancelled")
                    self.assertEqual(episode.robot.sent, [])
                    episode.results.finish_episode.assert_called_once()

    def test_retired_model_failure_ends_session_as_policy_failure(self):
        session = make_session(make_episode(self.clock, recorder=FakeRecorder()))
        session.sensors = SimpleNamespace(channels=None, request_shutdown=Mock())
        start_session_episode(session)
        old_episode = session.episode
        session.request_stop()
        session.advance_episode()
        ended = old_episode.termination
        session.model.complete(error=RuntimeError("predict failed"))
        with patch("dexmani_real.deployment.session.time.sleep", side_effect=lambda delay: session.request_quit()):
            with self.assertRaisesRegex(RuntimeError, "retired model predict"):
                session.run_control_loop()
        session.sensors.request_shutdown.assert_called_once_with(reason=RunEndReason.POLICY_FAILURE)
        self.assertTrue(session.runtime_state.error_state)
        self.assertIs(old_episode.termination, ended)
        self.assertEqual(len(session.recorder.finishes), 1)

    def test_polled_model_error_survives_cancelled_cleanup_failure(self):
        for operation in ("reset_episode", "predict"):
            with self.subTest(operation=operation):
                episode = make_episode(self.clock, recorder=FakeRecorder(), results=Mock())
                if operation == "predict":
                    start_episode(episode)
                else:
                    episode.step()
                model_error = RuntimeError("model task failed")
                episode.model.complete(error=model_error)
                poll = episode.model.poll
                stop_error = RuntimeError("stop unconfirmed")
                def stop_during_poll():
                    episode.runtime_state.request_stop()
                    episode.robot.stop_required = True
                    episode.robot.stop_error = stop_error
                    return poll()
                episode.model.poll = stop_during_poll
                with self.assertRaisesRegex(RuntimeError, "stop unconfirmed") as caught:
                    episode.step()
                self.assertIs(caught.exception, stop_error)
                self.assertIsNone(episode.model.future)
                self.assertTrue(episode.runtime_state.error_state)
                episode.results.record_errors.assert_called_once()
                detail, = episode.results.record_errors.call_args.args[0]
                self.assertEqual(detail["operation"], operation)
                self.assertEqual(detail["message"], str(model_error))
                errors = episode.results.finish_episode.call_args.kwargs["errors"]
                self.assertEqual(errors[0]["stage"], "stop")
                self.assertEqual(errors[0]["message"], str(stop_error))
                episode.results.finish_episode.assert_called_once()

    def test_writer_start_failure_keeps_attempt_and_does_not_count_run(self):
        with tempfile.TemporaryDirectory() as directory:
            results = EpisodeResults(directory, "policy")
            recorder = FakeRecorder()
            recorder.start_error = RecordingError("writer not ready")
            recorder.finish_error = RecordingError("writer not ready")
            episode = make_episode(self.clock, recorder=recorder, results=results)
            episode.step()
            episode.model.complete()
            with self.assertRaises(RecordingError):
                episode.step()
            self.assertIsNone(episode.termination.run_id)
            self.assertEqual(len(recorder.starts), 1)
            self.assertEqual(len(recorder.finishes), 1)
            result = results.session["episodes"][0]
            self.assertFalse(result["started"])
            self.assertEqual(result["termination_reason"], "preparation_failed")
            self.assertEqual(result["raw"]["status"], "failed")
            self.assertEqual(result["raw"]["staging_path"], str(recorder.staging_path))

    def test_zero_row_cancelled_capture_and_next_attempt_have_distinct_names(self):
        recorder = FakeRecorder()
        session = make_session(make_episode(self.clock, recorder=recorder))
        real_start = recorder.start_episode
        def cancelled_start(**kwargs):
            real_start(**kwargs)
            session.request_stop()
        recorder.start_episode = cancelled_start
        start_session_episode(session)
        self.assertEqual(len(recorder.frames), 0)
        self.assertEqual(recorder.finishes[0]["reason"], "start_cancelled")
        self.assertEqual(session.completed, 0)
        self.assertIsNone(session.episode)
        recorder.start_episode = real_start
        start_session_episode(session)
        session.request_stop()
        session.advance_episode()
        self.assertEqual(session.completed, 1)
        self.assertEqual([item["episode_name"] for item in recorder.starts], ["episode_001", "episode_002"])

    def test_capture_summary_distinguishes_publication_from_row_count(self):
        for saved, status in ((Path("/tmp/fake-published"), "published"), (None, "not_published")):
            with self.subTest(status=status):
                recorder = FakeRecorder()
                recorder.save_episode = Mock(return_value=saved)
                episode = make_episode(self.clock, recorder=recorder)
                episode.results = Mock()
                episode.prepare()
                episode.model.complete()
                episode.poll_model()
                episode.runtime_state.request_stop()
                episode.step()
                raw = episode.results.finish_episode.call_args.kwargs["raw"]
                self.assertEqual(raw["status"], status)
                self.assertEqual(raw["row_count"], 0)
                self.assertEqual(raw["path"], str(saved) if saved is not None else None)
                timing = recorder.save_episode.call_args.kwargs["timing"]
                self.assertEqual(timing["run_id"], episode.termination.run_id)
                self.assertEqual(timing["ended_ns"], self.clock.now)
                episode.finish("again")
                recorder.save_episode.assert_called_once()

    def test_partial_dispatch_is_recorded_once_before_failure(self):
        recorder = FakeRecorder()
        episode = make_episode(self.clock, recorder=recorder)
        start_episode(episode)
        episode.model.complete(self.predictions())
        receipt = DispatchResult(DispatchStatus.SUBMITTED, DispatchStatus.UNKNOWN, self.clock.now, (episode.run_id, 1))
        episode.robot.dispatch_error = DispatchError("hand failed", receipt)
        self.clock.advance(.05)
        with self.assertRaises(DispatchError):
            episode.step()
        self.assertEqual(len(recorder.frames), 2)
        self.assertIs(recorder.frames[-1][2], receipt)
        self.assertEqual(len(recorder.finishes), 1)
        self.assertIsNotNone(episode.termination.run_id)
        episode.finish("again")
        self.assertEqual(len(recorder.finishes), 1)

    def test_queue_failure_is_not_retried(self):
        recorder = FakeRecorder()
        episode = make_episode(self.clock, recorder=recorder)
        start_episode(episode)
        recorder.add_frame = Mock(side_effect=RecordingError("queue full"))
        self.clock.advance(.05)
        with self.assertRaises(RecordingError):
            episode.step()
        recorder.add_frame.assert_called_once()
        self.assertEqual(len(recorder.finishes), 1)

    def test_program_error_after_record_does_not_duplicate_control_row(self):
        recorder = FakeRecorder()
        episode = make_episode(self.clock, recorder=recorder)
        start_episode(episode)
        episode.model.complete(self.predictions())
        real_record = episode.record_pending
        def recorded_then_failed():
            real_record()
            raise ValueError("program failure after enqueue")
        episode.record_pending = recorded_then_failed
        self.clock.advance(.05)
        with self.assertRaisesRegex(ValueError, "program failure after enqueue"):
            episode.step()
        self.assertEqual(len(recorder.frames), 2)
        self.assertIsNotNone(recorder.frames[-1][2])
        self.assertEqual(len(recorder.finishes), 1)

    def test_program_error_after_dispatch_keeps_receipt_and_traceback(self):
        recorder = FakeRecorder()
        episode = make_episode(self.clock, recorder=recorder)
        start_episode(episode)
        episode.model.complete(self.predictions())
        error = ValueError("evaluation bug")
        episode.results = Mock()
        episode.results.record_dispatch.side_effect = error
        self.clock.advance(.05)
        with self.assertRaises(ValueError) as caught:
            episode.step()
        self.assertIs(caught.exception, error)
        self.assertEqual(len(recorder.frames), 2)
        self.assertEqual(recorder.frames[-1][2].arm, DispatchStatus.SUBMITTED)
        self.assertEqual(recorder.frames[-1][3]["prediction_index"], 0)
        self.assertEqual(len(recorder.finishes), 1)

    def test_reset_identity_mismatch_never_grants_motion(self):
        episode = make_episode(self.clock)
        episode.step()
        episode.model.context = {"preparing_epoch": episode.preparing_epoch - 1}
        episode.model.complete()
        with self.assertRaisesRegex(RuntimeError, "reset/preparation ownership mismatch"):
            episode.step()
        self.assertIsNone(episode.termination.run_id)
        self.assertEqual(episode.termination.reason, "preparation_failed")
        self.assertEqual(episode.robot.sent, [])

    def test_reset_error_is_not_silently_treated_as_operator_cancellation(self):
        for error_type in (CancelledError, RuntimeError):
            with self.subTest(error_type=error_type):
                episode = make_episode(self.clock)
                episode.step()
                error = error_type("model reset failed")
                episode.model.complete(error=error)
                with self.assertRaises(error_type) as caught:
                    episode.step()
                self.assertIs(caught.exception, error)
                self.assertEqual(episode.termination.reason, "preparation_failed")

    def test_session_counts_once_even_when_finalization_raises(self):
        recorder = FakeRecorder()
        session = make_session(make_episode(self.clock, recorder=recorder))
        start_session_episode(session)
        recorder.finish_error = RecordingError("publish failed")
        session.request_stop()
        with self.assertRaises(RecordingError):
            session.advance_episode()
        session.advance_episode(shutdown=True)
        self.assertEqual(session.completed, 1)
        self.assertEqual(len(recorder.finishes), 1)

    def test_episode_limit_quits_without_waiting_for_retired_predict(self):
        session = make_session(make_episode(self.clock))
        session.num_episodes = 1
        start_session_episode(session)
        session.request_stop()
        session.advance_episode()
        self.assertEqual(session.completed, 1)
        self.assertTrue(session.runtime_state.quit_requested)
        self.assertFalse(session.model.future.done())
        self.assertIsNotNone(session.episode)

    def test_session_loop_runs_two_attempts_with_unique_queries_and_captures(self):
        recorder = FakeRecorder()
        session = make_session(make_episode(self.clock, recorder=recorder))
        session.num_episodes = 2
        events = iter(([Key.BEGIN], [], [], [Key.STOP], [Key.BEGIN], [], [], [Key.STOP]))
        def poll(timeout=0):
            batch = next(events)
            if Key.STOP in batch:
                session.request_stop()
            return batch
        def advance_clock(delay):
            self.clock.advance(.05)
            if session.model.future is not None and not session.model.future.done():
                session.model.complete(self.predictions() if session.model.operation == "predict" else None)
        session.keyboard.poll = poll
        with patch("dexmani_real.deployment.session.time.sleep", side_effect=advance_clock):
            session.run_control_loop()
        self.assertEqual(session.completed, 2)
        self.assertEqual(len(session.robot.sent), 2)
        self.assertEqual(len(recorder.finishes), 2)
        contexts = [call[3] for call in session.model.calls if call[0] == "predict"]
        self.assertEqual([context["query_id"] for context in contexts], [1, 2])
        self.assertLess(contexts[0]["run_id"], contexts[1]["run_id"])
        self.assertEqual([item["episode_name"] for item in recorder.starts], ["episode_001", "episode_002"])
        self.assertIsNone(session.episode)

    def test_quit_during_reset_finishes_without_waiting_or_granting_motion(self):
        session = make_session(make_episode(self.clock))
        events = iter(([Key.BEGIN], [Key.QUIT]))
        def poll(timeout=0):
            batch = next(events)
            if Key.QUIT in batch:
                session.request_quit()
            return batch
        session.keyboard.poll = poll
        with patch("dexmani_real.deployment.session.time.sleep"):
            session.run_control_loop()
        self.assertEqual(session.completed, 0)
        self.assertEqual(session.episode.termination.reason, "start_cancelled")
        self.assertFalse(session.model.future.done())
        self.assertEqual(session.robot.sent, [])

    def test_stop_failure_still_finalizes_with_unresolved_evidence(self):
        recorder = FakeRecorder()
        episode = make_episode(self.clock, recorder=recorder)
        start_episode(episode)
        episode.robot.stop_required = True
        episode.robot.stop_error = RuntimeError("stop unresolved")
        episode.runtime_state.request_stop()
        with self.assertRaises(RuntimeError):
            episode.step()
        self.assertFalse(recorder.finishes[0]["arm_execution"]["robot_stopped"])
        self.assertEqual(len(recorder.finishes), 1)

    def test_cancelled_preparation_with_stop_fault_is_failure(self):
        recorder = FakeRecorder()
        episode = make_episode(self.clock, recorder=recorder)
        real_start = recorder.start_episode
        def cancelled_start(**kwargs):
            real_start(**kwargs)
            episode.robot.stop_required = True
            episode.robot.stop_error = RuntimeError("preparation stop failed")
            episode.runtime_state.request_stop()
        recorder.start_episode = cancelled_start
        with self.assertRaisesRegex(RuntimeError, "preparation stop failed"):
            start_episode(episode)
        self.assertEqual(episode.termination.reason, "preparation_failed")
        self.assertEqual(recorder.finishes[0]["reason"], "preparation_failed")
        self.assertFalse(recorder.finishes[0]["arm_execution"]["robot_stopped"])

    def test_stop_quit_suppress_begin_home_and_tactile_in_entire_batch(self):
        for terminal in (Key.STOP, Key.QUIT):
            for batch in ([Key.BEGIN, Key.HOME, terminal], [terminal, Key.BEGIN, Key.HOME]):
                with self.subTest(batch=batch):
                    episode = make_episode(self.clock)
                    session = make_session(episode)
                    session.run_home = Mock()
                    session.handle_command_batch(batch)
                    self.assertIsNone(session.start_epoch)
                    session.run_home.assert_not_called()

    def test_home_drains_stale_preparation_but_keeps_stop(self):
        episode = make_episode(self.clock)
        session = make_session(episode)
        def home(*args, **kwargs):
            session.keyboard.signals.extend((Key.HOME, Key.BEGIN, Key.RESET_TACTILE, Key.STOP))
            return HomeResult(True)
        with patch("dexmani_real.deployment.session.home_robot", side_effect=home):
            session.handle_command_batch([Key.HOME, Key.BEGIN])
        self.assertIsNone(session.start_epoch)
        self.assertEqual(session.keyboard.signals, [Key.STOP])

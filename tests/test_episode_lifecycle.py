import threading
import time
import unittest
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.deployment.rollout import EpisodeResult, Rollout
from dexmani_real.deployment.schedule import ActionChunk
from dexmani_real.deployment.session import PolicySession
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.state import RunEndReason, SessionState


def make_episode(*, model=None, recorder=None):
    state = SessionState()
    state.arm()
    state.stop_signal.clear()  # 对应设备 prepare_streaming 成功后的状态。
    epoch, _ = state.begin_motion()
    robot = SimpleNamespace(
        arm=SimpleNamespace(check_error=Mock()), stop_required=True,
        service_idle=Mock(), service_feedback=Mock(), stop=Mock(),
    )
    model = model or SimpleNamespace(has_pending_task=False, discard_pending=Mock())
    episode = Rollout(
        state, SimpleNamespace(execution=ExecutionConfig()),
        SimpleNamespace(control_dt_s=.01, n_obs_steps=1, hand_history_steps=None),
        preparing_epoch=epoch, robot=robot, observations=SimpleNamespace(needs=ObservationNeeds()),
        record_needs=ObservationNeeds(), needs=ObservationNeeds(), sensors=None,
        action_mapper=None, model=model, max_running_s=None, capture_ids=iter(()),
        index=1, output_dir=None, record_raw=False,
    )
    episode.run_id = epoch
    episode.started_ns = time.monotonic_ns()
    episode.recorder = recorder
    keyboard = Keyboard(estop_callback=state.request_emergency_stop)
    keyboard.active = True
    return episode, keyboard


def recorder_stub():
    return SimpleNamespace(
        start_attempted=True, writer_pending=False, finish=Mock(), append=Mock(),
        result=Mock(return_value=dict(status="published", path="episode", staging_path=None, row_count=2)),
    )


class EpisodeLifecycleTests(unittest.TestCase):
    def test_first_reason_classification_does_not_depend_on_detail_text(self):
        for reason, expected in (("operator", True), ("quit", True),
                                 ("duration_complete", True), ("prediction_timeout", False),
                                 ("timeout", False), ("executor_boundary", False)):
            with self.subTest(reason=reason):
                result = EpisodeResult(1, termination_reason=reason, termination_details="duration_timeout")
                self.assertEqual(result.normal_termination, expected)
                result.termination_details = "arbitrary human explanation"
                self.assertEqual(result.normal_termination, expected)
        result = EpisodeResult(1, termination_reason="duration_complete", errors=[{"stage": "close"}])
        self.assertFalse(result.normal_termination)

    def test_duration_and_prediction_deadlines_have_distinct_machine_reasons(self):
        for kind, reason in (("duration", "duration_complete"), ("wait", "prediction_timeout")):
            with self.subTest(kind=kind):
                episode, _ = make_episode()
                episode.stop_budget((0, kind))
                self.assertEqual(episode.result.termination_reason, reason)
                episode.stop("cleanup", run_end_reason=RunEndReason.RUNTIME_FAILURE)
                self.assertEqual(episode.result.termination_reason, reason)

    def test_stop_precedes_single_recording_finish_and_discard_is_explicit(self):
        for discard in (False, True):
            with self.subTest(discard=discard):
                recorder = recorder_stub()
                episode, keyboard = make_episode(recorder=recorder)
                epoch = episode.run_id
                episode.discard_requested = discard

                def finish(**kwargs):
                    self.assertFalse(episode.runtime_state.can_move(run_id=epoch))
                    episode.robot.stop.assert_called_once()
                    self.assertEqual(kwargs["save"], not discard)
                    self.assertEqual(kwargs["technical_status"], "normal")

                recorder.finish.side_effect = finish
                episode.step = lambda: episode.stop("S", run_end_reason=RunEndReason.OPERATOR)
                result = episode.run(keyboard=keyboard, poll_operator=lambda: None)
                self.assertIs(result, episode.result)
                recorder.finish.assert_called_once()
                recorder.result.assert_called_once()
                episode.model.discard_pending.assert_called_once()

    def test_cleanup_failure_keeps_control_reason_and_is_not_retried(self):
        recorder = recorder_stub()
        recorder.finish.side_effect = OSError("close failed")
        episode, keyboard = make_episode(recorder=recorder)
        episode.step = lambda: episode.stop("S", run_end_reason=RunEndReason.OPERATOR)
        with self.assertLogs("dexmani_real.deployment.rollout", level="ERROR"):
            with self.assertRaisesRegex(OSError, "close failed"):
                episode.run(keyboard=keyboard, poll_operator=lambda: None)
        self.assertEqual(episode.result.termination_reason, "operator")
        self.assertEqual(episode.result.errors[0]["stage"], "recorder_finalize")
        recorder.finish.assert_called_once()

    def test_escape_callback_remains_live_while_writer_finish_blocks(self):
        entered, release = threading.Event(), threading.Event()
        recorder = recorder_stub()

        def finish(**kwargs):
            entered.set()
            if not release.wait(1):
                raise TimeoutError("test release was not signalled")

        recorder.finish.side_effect = finish
        episode, keyboard = make_episode(recorder=recorder)
        episode.step = lambda: episode.stop("S", run_end_reason=RunEndReason.OPERATOR)
        errors = []

        def run():
            try:
                episode.run(keyboard=keyboard, poll_operator=lambda: None)
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            keyboard.handle_key("esc", True)
            self.assertTrue(episode.runtime_state.emergency_signal.is_set())
            self.assertTrue(episode.runtime_state.stop_signal.is_set())
            self.assertTrue(thread.is_alive())
        finally:
            release.set()
            thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(episode.result.termination_reason, "operator")

    def test_late_prediction_is_discarded_before_another_episode_can_start(self):
        worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
        retired = Future()
        worker.future, worker.operation = retired, "predict"
        episode, keyboard = make_episode(model=worker)
        episode.step = lambda: episode.stop("S", run_end_reason=RunEndReason.OPERATOR)
        session = PolicySession.__new__(PolicySession)
        session.runtime_state, session.model, session.episode = episode.runtime_state, worker, episode
        session.start_epoch = None
        entered, done = threading.Event(), threading.Event()
        original_discard = worker.discard_pending

        def discard(timeout_s):
            entered.set()
            return original_discard(timeout_s)

        worker.discard_pending = discard
        errors = []

        def run():
            try:
                episode.run(keyboard=keyboard, poll_operator=lambda: None)
            except BaseException as exc:
                errors.append(exc)
            finally:
                done.set()

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(done.is_set())
            self.assertFalse(session.request_start())
            retired.set_result(SimpleNamespace(actions=np.ones((3, 21)), duration_ns=1))
            thread.join(1)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertFalse(worker.has_pending_task)
            self.assertIsNone(episode.query)
            self.assertIsNone(episode.plan)
        finally:
            if not retired.done():
                retired.set_result(None)
            thread.join(1)
            worker.close()
            self.assertEqual(worker.wait_closed(1)[0], "completed")

    def test_timeout_keeps_model_task_owned_and_blocks_start(self):
        worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
        retired = Future()
        worker.future, worker.operation = retired, "predict"
        episode, keyboard = make_episode(model=worker)
        episode.step = lambda: episode.stop("S", run_end_reason=RunEndReason.OPERATOR)
        try:
            with patch("dexmani_real.deployment.rollout.SHUTDOWN_TIMEOUT_S", 0), \
                    self.assertLogs("dexmani_real.deployment.rollout", level="ERROR"):
                with self.assertRaisesRegex(TimeoutError, "remains pending"):
                    episode.run(keyboard=keyboard, poll_operator=lambda: None)
            self.assertTrue(worker.has_pending_task)
            session = PolicySession.__new__(PolicySession)
            session.runtime_state, session.model, session.episode = episode.runtime_state, worker, episode
            self.assertFalse(session.request_start())
        finally:
            retired.set_result(None)
            worker.close()
            self.assertEqual(worker.wait_closed(1)[0], "completed")

    def test_unused_retired_prediction_failure_preserves_raw_but_exits_session(self):
        worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
        retired = Future()
        retired.set_exception(ValueError("prediction exploded"))
        worker.future, worker.operation = retired, "predict"
        recorder = recorder_stub()
        episode, keyboard = make_episode(model=worker, recorder=recorder)
        episode.step = lambda: episode.stop("S", run_end_reason=RunEndReason.OPERATOR)
        try:
            with self.assertLogs("dexmani_real.deployment.rollout", level="ERROR"):
                with self.assertRaisesRegex(ValueError, "prediction exploded"):
                    episode.run(keyboard=keyboard, poll_operator=lambda: None)
            self.assertEqual(episode.result.termination_reason, "operator")
            self.assertFalse(episode.result.normal_termination)
            self.assertEqual(recorder.finish.call_args.kwargs["technical_status"], "normal")
            self.assertEqual(episode.result.errors[0]["stage"], "retired_model")
            recorder.finish.assert_called_once()
            self.assertFalse(worker.has_pending_task)
        finally:
            worker.close()
            self.assertEqual(worker.wait_closed(1)[0], "completed")

    def test_two_episodes_reuse_one_loaded_model(self):
        loads, resets = [], []
        policy = SimpleNamespace(reset_episode=lambda: resets.append(True), close=lambda: None)

        def factory():
            loads.append(True)
            return policy

        worker = PolicyWorker(factory, build_observation=lambda rows: rows)
        try:
            worker.load()
            worker.future.result(timeout=1)
            self.assertEqual(worker.poll(), ("load", None))
            for _ in range(2):
                episode, keyboard = make_episode(model=worker)
                worker.reset()
                episode.step = lambda: episode.stop("S", run_end_reason=RunEndReason.OPERATOR)
                episode.run(keyboard=keyboard, poll_operator=lambda: None)
            self.assertEqual(loads, [True])
            self.assertEqual(resets, [True, True])
        finally:
            worker.close()
            self.assertEqual(worker.wait_closed(1)[0], "completed")

    def test_rtc_guidance_freezes_only_after_prediction_submission(self):
        episode, _ = make_episode()
        episode.execution = ExecutionConfig(mode="rtc", prefetch_steps=2, rtc_guidance_cap=1)
        episode.plan = ActionChunk(np.zeros((8, 19)), 0, 4, {}, 123)
        episode.model.predict = Mock()
        episode.history = deque([SimpleNamespace(observation_timestamp_ns=123)])
        episode.action_mapper = SimpleNamespace(
            control_from_command=lambda command, mode: np.concatenate((command.arm_qpos, command.hand_qpos)),
        )
        episode.policy_info.action_mode = "joint"
        command = RobotCommand(episode.run_id, np.ones(7), np.ones(12))
        episode.map_command = lambda action, row, previous, **kwargs: (command, None)
        episode.robot.check = lambda: None
        prefix, commands, issue = episode.prepare_prefix(None, 2)
        self.assertIsNone(issue)
        self.assertEqual(episode.plan.frozen, {})
        np.testing.assert_array_equal(prefix[:2], np.ones((2, 19)))
        self.assertIsNone(episode.submit(2, handoff_slot=4, prefix=prefix, commands=commands))
        self.assertIs(episode.plan.frozen[2], command)
        self.assertIs(episode.plan.frozen[3], command)
        self.assertEqual(episode.model.predict.call_args.kwargs["delay_steps"], 2)
        np.testing.assert_array_equal(episode.plan.actions, np.zeros((8, 19)))


if __name__ == "__main__":
    unittest.main()

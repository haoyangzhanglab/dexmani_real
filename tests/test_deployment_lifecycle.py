"""模型线程、单轮结果和会话清理契约；模型、设备与 recorder 均为离线替身。"""

import json
import tempfile
import threading
import unittest
from dataclasses import replace
from itertools import count
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from raw_fixture import raw_row, start_raw_recorder
from test_runtime_contract import armed_state

from dexmani_real.config import ExperimentConfig
from dexmani_real.data.raw.reader import EpisodeReader, RawDataError
from dexmani_real.data.session_results import SessionResults
from dexmani_real.deployment.inference import InferenceWorker, ModelResult
from dexmani_real.deployment.policy_observation import PolicyObservationRow
from dexmani_real.deployment.rollout.action_schedule import ActionPlan, PolicyQuery
from dexmani_real.deployment.rollout.episode import PolicyEpisode
from dexmani_real.deployment.session import PolicySession
from dexmani_real.robot.commands import RobotCommand, RobotError
from dexmani_real.robot.feedback import ARM_STATE_DTYPE
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.state import RunEndReason


class InferenceTests(unittest.TestCase):
    def test_serial_ownership_and_completed_future_still_blocks_submit(self):
        events = []
        def event(name, value=None):
            events.append((name, threading.get_ident()))
            return value
        model = SimpleNamespace(warmup=lambda: event('warmup'), reset_episode=lambda: event('reset'),
                                predict=lambda values: event('predict', values + 1), close=lambda: event('close'))
        worker = InferenceWorker(lambda: event('load', model),
                                 build_observation=lambda values: event('observation', values),
                                 reset_observation=lambda: event('reset_observation'))
        try:
            worker.submit('load')
            worker.future.result(timeout=2)
            self.assertTrue(worker.has_pending_task)
            with self.assertRaisesRegex(RuntimeError, 'unreclaimed'):
                worker.submit('reset_episode')
            self.assertIsNone(worker.poll()[1].error)
            worker.submit('reset_episode')
            worker.future.result(timeout=2)
            worker.poll()
            worker.submit('predict', np.array([2]), query_context={'run_id': 7})
            worker.future.result(timeout=2)
            _, result = worker.poll()
            np.testing.assert_array_equal(result.value, [3])
            self.assertLessEqual(result.started_ns, result.model_started_ns)
            self.assertLessEqual(result.model_started_ns, result.completed_ns)
        finally:
            worker.close().result(timeout=2)
        self.assertEqual([name for name, _ in events],
                         ['load', 'warmup', 'reset_observation', 'reset', 'observation', 'predict', 'close'])
        self.assertEqual(len({ident for _, ident in events}), 1)
        self.assertNotEqual(events[0][1], threading.get_ident())

    def test_retired_failure_remains_visible_and_close_still_runs(self):
        model = Mock()
        model.predict.side_effect = ValueError('prediction failed')
        worker = InferenceWorker(lambda: model, build_observation=lambda value: value)
        try:
            worker.submit('load')
            worker.future.result(timeout=2)
            worker.poll()
            context = dict(run_id=3, query_id=8)
            worker.submit('predict', (), query_context=context)
            worker.future.result(timeout=2)
            with self.assertLogs('dexmani_real.deployment.inference', level='ERROR'):
                worker.close().result(timeout=2)
            outcome = worker.close_result()
            self.assertEqual(outcome['status'], 'failed')
            self.assertEqual(outcome['retired_task']['query_context'], context)
            self.assertEqual(outcome['retired_task']['exception_type'], 'ValueError')
            self.assertIn('prediction failed', outcome['reason'])
            model.close.assert_called_once()
        finally:
            worker.close().result(timeout=2)

    def test_close_pending_is_not_reported_completed(self):
        release, entered = threading.Event(), threading.Event()
        def warmup():
            entered.set()
            if not release.wait(timeout=3):
                raise TimeoutError('test did not release model')
        worker = InferenceWorker(lambda: SimpleNamespace(warmup=warmup, close=lambda: None),
                                 build_observation=lambda value: value)
        try:
            worker.submit('load')
            self.assertTrue(entered.wait(timeout=2))
            worker.close()
            self.assertEqual(worker.close_result()['status'], 'pending')
        finally:
            release.set()
            worker.close().result(timeout=2)
        self.assertEqual(worker.close_result()['status'], 'completed')


class DeploymentLifecycleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def episode(self, *, running=False, begin=True, recorder=None):
        state, runtime = armed_state(), ExperimentConfig()
        info = SimpleNamespace(n_obs_steps=2, hand_history_steps=None, control_dt_s=.1,
                               observation_fields=('joint_state',), action_mode='joint', task_name='offline')
        robot, model = Mock(stop_required=False), Mock(has_pending_task=False)
        results = SessionResults(self.root, 'policy')
        episode = PolicyEpisode(
            state, runtime, info, preparing_epoch=state.run_id, robot=robot,
            control_needs=ObservationNeeds(), record_needs=ObservationNeeds(), dependencies=frozenset(),
            sensors=Mock(), action_mapper=Mock(), model_runtime=model, max_running_s=None,
            query_ids=count(1), capture_ids=count(1), results=results, recorder=recorder,
        )
        if begin:
            results.begin_episode()
        if running:
            episode.run_id, episode.started_ns = state.begin_motion()
            results.entered(episode.run_id)
        return episode

    def session(self, episode):
        session = PolicySession(episode.runtime, episode.policy_info, control_needs=ObservationNeeds(),
                                record_needs=ObservationNeeds(), dependencies=frozenset(),
                                results_dir=self.root, run_config={}, num_episodes=None)
        session.runtime_state, session.robot, session.model = episode.runtime_state, episode.robot, episode.model
        session.results, session.episode, session.recorder = episode.results, episode, episode.recorder
        session.keyboard = Mock()
        session.sensors = Mock()
        return session

    def test_cancelled_preparation_does_not_count_as_executed_episode(self):
        episode = self.episode(begin=False)
        episode.robot.reset_hand_feedback.side_effect = episode.runtime_state.request_stop
        episode.start_observation = Mock(return_value=object())
        session = self.session(episode)
        session.advance_episode()
        self.assertEqual(session.completed, 0)
        self.assertIsNone(session.episode)
        result = episode.results.session['episodes'][0]
        self.assertFalse(result['started'])
        self.assertEqual(result['termination_reason'], 'start_cancelled')
        self.assertEqual(result['duration_s'], 0)
        episode.model.submit.assert_not_called()

    def test_first_termination_survives_stop_error_and_partial_recording(self):
        events = []
        recorder = Mock(start_attempted=True, written_frames=1, published_path=None,
                        staging_path=self.root/'staging', writer_pending=False)
        recorder.add_frame.side_effect = lambda *a: events.append('frame')
        recorder.save_episode.side_effect = lambda **kw: events.append('save') or self.root/'episode_001'
        episode = self.episode(running=True, recorder=recorder)
        episode.pending_frame = ('row', 'attempt')
        def stop():
            events.append('stop')
            self.assertFalse(episode.runtime_state.command_may_cross_sdk(run_id=episode.termination.run_id))
            raise RobotError('stop unresolved')
        episode.robot.stop.side_effect = stop
        with self.assertRaisesRegex(RobotError, 'stop unresolved'):
            episode.finish('operator requested', run_end_reason=RunEndReason.OPERATOR)
        episode.finish('later cleanup', run_end_reason=RunEndReason.RUNTIME_SHUTDOWN)
        self.assertEqual(events, ['stop', 'frame', 'save'])
        self.assertEqual(episode.termination.reason, 'operator')
        self.assertIsNone(episode.pending_frame)
        saved = json.loads(episode.results.path.read_text())['episodes'][0]
        self.assertEqual(saved['termination_reason'], 'operator')
        self.assertEqual(saved['errors'][0]['stage'], 'stop')
        self.assertEqual(saved['raw']['row_count'], 1)
        self.assertEqual(saved['raw']['status'], 'published')
        self.assertEqual(recorder.save_episode.call_args.kwargs['technical_status'], 'failed')

    def test_stop_outcome_reaches_real_raw_metadata_and_completion_admission(self):
        root = self.root
        for reason, detail in ((RunEndReason.OPERATOR, 'operator requested'),
                               (RunEndReason.QUIT, 'quit'), (RunEndReason.TIMEOUT, 'duration_timeout')):
            for stop_fails in (False, True):
                with self.subTest(reason=reason, stop_fails=stop_fails):
                    self.root = root / f'{reason.name}_{stop_fails}'
                    recorder = start_raw_recorder(self.root / 'raw')
                    self.addCleanup(recorder.close)
                    episode = self.episode(running=True, recorder=recorder)
                    command = RobotCommand(episode.run_id, np.zeros(7), np.zeros(12))
                    recorder.add_frame(raw_row(0), command)
                    episode.pending_frame = (raw_row(1), command)
                    failure = RobotError('stop unresolved')
                    if stop_fails:
                        episode.robot.stop.side_effect = failure
                        with self.assertRaises(RobotError) as raised:
                            episode.finish(detail, run_end_reason=reason)
                        self.assertIs(raised.exception, failure)
                    else:
                        episode.finish(detail, run_end_reason=reason)
                    episode.finish('later cleanup')
                    episode.robot.stop.assert_called_once()
                    self.assertIsNone(episode.pending_frame)
                    self.assertEqual(recorder.submitted_frames, 2)
                    with EpisodeReader(recorder.published_path) as reader:
                        self.assertEqual(reader.num_frames, 2)
                        self.assertEqual(reader.meta['termination_reason'], reason.name.lower())
                        self.assertEqual(reader.meta['technical_status'], 'failed' if stop_fails else 'normal')
                        np.testing.assert_array_equal(reader['timestamp'][:], [0, .1])
                        if stop_fails:
                            with self.assertRaisesRegex(RawDataError, 'technical failure'):
                                reader.validate_completion()
                            details = json.loads(reader.data_file['meta/termination_details'][()])
                            self.assertEqual(details[-1]['stage'], 'stop')
                            self.assertEqual(details[-1]['message'], 'stop unresolved')
                        else:
                            reader.validate_completion()
                    saved = episode.results.session['episodes'][0]
                    self.assertEqual(saved['termination_reason'], reason.name.lower())
                    self.assertEqual(saved['raw']['status'], 'published')

    def test_existing_control_error_is_not_cleared_by_successful_stop(self):
        recorder = Mock(start_attempted=True, published_path=None, staging_path=None, written_frames=1)
        episode = self.episode(running=True, recorder=recorder)
        episode.finish('operator requested', run_end_reason=RunEndReason.OPERATOR,
                       errors=[dict(stage='control', message='control failed')])
        self.assertEqual(episode.termination.reason, 'operator')
        self.assertEqual(recorder.save_episode.call_args.kwargs['technical_status'], 'failed')
        self.assertEqual(recorder.save_episode.call_args.kwargs['details'][-1]['stage'], 'control')

    def test_wait_timeout_remains_a_technical_failure(self):
        recorder = Mock(start_attempted=True, published_path=None, staging_path=None, written_frames=1)
        episode = self.episode(running=True, recorder=recorder)
        episode.finish('wait_timeout', run_end_reason=RunEndReason.TIMEOUT)
        self.assertEqual(episode.termination.reason, 'timeout')
        self.assertEqual(recorder.save_episode.call_args.kwargs['technical_status'], 'failed')

    def test_published_raw_is_unchanged_by_independent_session_stop_failure(self):
        recorder = start_raw_recorder(self.root / 'raw')
        self.addCleanup(recorder.close)
        episode = self.episode(running=True, recorder=recorder)
        episode.pending_frame = (raw_row(), RobotCommand(episode.run_id, np.zeros(7), np.zeros(12)))
        episode.finish('operator requested', run_end_reason=RunEndReason.OPERATOR)
        before = {path.name: path.read_bytes() for path in recorder.published_path.iterdir()}
        session = self.session(episode)
        session.robot.stop_required = True
        session.robot.stop.side_effect = RobotError('session stop unresolved')
        session.model.close_result.return_value = dict(status='completed')
        session.sensors.shutdown.return_value = True
        self.assertEqual(session.close(clean=True, failure=None), 1)
        self.assertEqual({path.name: path.read_bytes() for path in recorder.published_path.iterdir()}, before)
        with EpisodeReader(recorder.published_path) as reader:
            reader.validate_completion()
            self.assertEqual(reader.meta['termination_reason'], 'operator')
        result = json.loads(session.results.path.read_text())
        self.assertEqual(result['outcome'], 'fault')
        self.assertEqual(result['errors'][0]['stage'], 'stop')
        self.assertEqual(result['episodes'][0]['termination_reason'], 'operator')

    def test_recorder_finalize_failure_preserves_staging_and_result(self):
        recorder = Mock(start_attempted=True, written_frames=2, published_path=None,
                        staging_path=self.root/'staging', writer_pending=True)
        recorder.save_episode.side_effect = RuntimeError('writer incomplete')
        episode = self.episode(running=True, recorder=recorder)
        with self.assertRaisesRegex(RuntimeError, 'writer incomplete'):
            episode.finish('duration_timeout', run_end_reason=RunEndReason.TIMEOUT)
        saved = json.loads(episode.results.path.read_text())['episodes'][0]
        self.assertEqual(saved['termination_reason'], 'timeout')
        self.assertEqual(saved['raw'], dict(status='pending', path=None, staging_path=str(self.root/'staging'), row_count=2))
        self.assertEqual(saved['errors'][0]['stage'], 'recorder_finalize')

    def test_failed_frame_enqueue_is_not_retried(self):
        recorder = Mock(start_attempted=True, published_path=None, staging_path=None, written_frames=1)
        recorder.add_frame.side_effect = RuntimeError('enqueue failed')
        episode = self.episode(running=True, recorder=recorder)
        episode.pending_frame = ('row', 'attempt')
        with self.assertRaisesRegex(RuntimeError, 'enqueue failed'):
            episode.finish('operator requested', run_end_reason=RunEndReason.OPERATOR)
        episode.record_pending()
        recorder.add_frame.assert_called_once()
        self.assertEqual(recorder.save_episode.call_args.kwargs['technical_status'], 'failed')
        self.assertEqual(recorder.save_episode.call_args.kwargs['details'][-1]['stage'], 'recording')
        self.assertEqual(episode.termination.reason, 'operator')

    def test_rtc_physical_prefix_is_committed_only_after_query_submission(self):
        episode = self.episode(running=True)
        episode.execution = replace(episode.execution, execution_mode='rtc', prefetch_steps=2)
        episode.plan = ActionPlan(np.zeros((5, 19)), 0, 3, 2_000_000_000, {})
        episode.action_mapper.map.return_value = SimpleNamespace(arm_qpos=np.ones(7)*.01, hand_qpos=np.zeros(12))
        episode.action_mapper.control_from_command.side_effect = lambda command, mode: np.concatenate((command.arm_qpos, command.hand_qpos))
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        row = SimpleNamespace(arm=arm)
        rows = [PolicyObservationRow(1_000_000_000, (1_000_000_000,), {})]
        with patch('dexmani_real.deployment.rollout.episode.time.monotonic_ns', return_value=1_000_000_001):
            prefix, commands, issue = episode.prepare_prefix(row, 0)
            self.assertIsNone(issue)
            self.assertEqual(episode.plan.frozen, {})
            episode.model.submit.side_effect = RuntimeError('busy')
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                episode.submit(rows, 0, handoff_slot=2, prefix=prefix, commands=commands)
            self.assertEqual(episode.plan.frozen, {})
            self.assertIsNone(episode.query)
            episode.model.submit.side_effect = None
            self.assertIsNone(episode.submit(rows, 0, handoff_slot=2, prefix=prefix, commands=commands))
        episode.action_mapper.frozen_rejection.return_value = None
        count_before = episode.action_mapper.map.call_count
        command, issue = episode.command_for_slot(row, 0)
        self.assertIs(command, commands[0])
        self.assertIsNone(issue)
        self.assertEqual(episode.action_mapper.map.call_count, count_before)
        self.assertFalse(command.arm_qpos.flags.writeable)
        np.testing.assert_array_equal(episode.plan.actions, np.zeros((5, 19)))

    def test_model_result_identity_mismatch_is_not_consumed(self):
        episode = self.episode(running=True)
        episode.query = PolicyQuery(3, episode.run_id, 0, 1000, None)
        result = ModelResult(np.zeros((2, 19)), None, 1, 2, dict(run_id=episode.run_id-1, query_id=3))
        episode.model.poll.return_value = ('predict', result)
        with self.assertRaisesRegex(RuntimeError, 'ownership mismatch'):
            episode.poll_model()
        self.assertIsNone(episode.query.result)

    def test_session_close_order_and_recorder_failure_visibility(self):
        episode = self.episode(begin=False)
        session = self.session(episode)
        session.episode = None
        events = []
        session.robot.stop_required = True
        session.robot.stop.side_effect = lambda: events.append('stop')
        session.robot.close.side_effect = lambda: events.append('robot_close')
        session.model.close.side_effect = lambda: events.append('model_close')
        session.model.close_result.side_effect = lambda timeout: events.append('model_result') or dict(status='completed')
        session.keyboard.stop.side_effect = lambda: events.append('keyboard')
        session.sensors.shutdown.side_effect = lambda **kw: events.append('sensors') or True
        def close_recorder():
            events.append('recorder_close')
            raise RuntimeError('writer still owns resources')
        session.recorder = SimpleNamespace(close=close_recorder)
        self.assertEqual(session.close(clean=True, failure=None), 1)
        self.assertEqual(events, ['stop', 'model_close', 'recorder_close', 'robot_close', 'keyboard', 'sensors', 'model_result'])
        result = json.loads(session.results.path.read_text())
        self.assertEqual(result['outcome'], 'fault')
        self.assertFalse(result['shutdown_clean'])
        self.assertEqual(result['errors'][0]['stage'], 'recorder_close')
        self.assertEqual(session.results.path.name, 'session_result.json')

    def test_results_refuse_to_overwrite_existing_session(self):
        SessionResults(self.root, 'policy')
        with self.assertRaises(FileExistsError):
            SessionResults(self.root, 'policy')

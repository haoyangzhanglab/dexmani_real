"""只验证现有边界；设备、时钟与反馈使用离线替身。"""

import contextlib
import io
import tempfile
import threading
import unittest
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.rollout import EpisodeResult, Rollout
from dexmani_real.deployment.session import PolicySession
from dexmani_real.robot.commands import (
    HomeTimeout,
    OperationCancelled,
    RobotCommand,
    RobotError,
)
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.home import home_hand
from dexmani_real.runtime.observation import (
    Observation,
    ObservationNeeds,
    check_device_feedback,
    check_observation,
)
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase
from dexmani_real.teleop.run import run_teleop


class SafetyContracts(unittest.TestCase):
    def setUp(self):
        self.runtime = RealConfig()
        self.state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
        self.now = 10**10

    def test_mapping_clips_but_submission_rejects(self):
        cfg = self.runtime
        mapper = ActionMapper.for_mode('joint', arm=cfg.arm, hand=cfg.hand,
                                       motion=cfg.motion, environment=cfg.environment)
        mapped = mapper.map_joint(np.full(7, 100), np.full(12, 100))
        np.testing.assert_array_equal(mapped.arm_qpos, cfg.arm.joint_limit_upper)
        np.testing.assert_array_equal(mapped.hand_qpos, cfg.hand.qpos_max_rad)
        robot = Robot(cfg.arm, cfg.hand, runtime_state=self.state)
        robot.connected = True
        robot.arm = Mock(is_connected=True)
        robot.hand = Mock(is_connected=True)
        for command in (RobotCommand(arm_qpos=np.full(7, 100)),
                        RobotCommand(hand_qpos=np.full(12, 100)),
                        RobotCommand(arm_qpos=np.zeros(6)),
                        RobotCommand(hand_qpos=np.full(12, np.nan))):
            with self.assertRaises(ValueError):
                robot.send_action(command, expires_ns=self.now)
        robot.arm.submit.assert_not_called()
        robot.hand._send_action.assert_not_called()

    def test_feedback_and_tactile_expiry(self):
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm['timestamp_ns'] = self.now
        for name in ('timestamp_ns', 'tactile_aggregate_timestamp_ns', 'tactile_dense_timestamp_ns'):
            hand[name] = self.now
        row = Observation(arm, hand, None, None, None, self.now)
        self.assertEqual(check_device_feedback(row, self.runtime, now_ns=self.now), [])
        deadline = self.now + int(self.runtime.arm.feedback_max_age_s * 1e9)
        self.assertEqual(check_device_feedback(row, self.runtime, now_ns=deadline-1), [])
        self.assertIn('arm_feedback_stale', check_device_feedback(row, self.runtime, now_ns=deadline))
        needs = ObservationNeeds(required_tactile=('aggregate', 'dense'))
        deadline = self.now + int(self.runtime.hand.feedback_max_age_s * 1e9)
        self.assertEqual(len(check_observation(row, self.runtime, needs, now_ns=deadline)), 2)
        self.assertIn('arm_feedback_stale', check_device_feedback(row, self.runtime, now_ns=self.now-1))

    def test_lease_and_cancellation_priority(self):
        io = NS(stop=threading.Event(), emergency=threading.Event(), closing=threading.Event(), fault=NS(value=b''))
        owner = ArmController(self.runtime.arm, io, None, None)
        owner.status['timestamp_ns'] = self.now
        self.assertIsNone(owner.command_issue({'expires_ns': self.now+1}, self.now))
        self.assertEqual(owner.command_issue({'expires_ns': self.now}, self.now), 'command_expired')
        io.emergency.set()
        self.assertEqual(owner.command_issue({'expires_ns': self.now}, self.now), 'cancelled')
        io.emergency.clear()
        self.assertEqual(owner.command_issue({'expires_ns': self.now+10**9},
                         self.now+int(self.runtime.arm.feedback_max_age_s*1e9)), 'arm_feedback_expired')
        self.state.request_stop()
        self.assertEqual(self.state.cancel_epoch, 2)
        self.state.request_emergency_stop()
        self.state.stop_signal.clear()
        self.assertFalse(self.state.requests_allow_motion())

    def test_home_failure_stops_and_stop_failure_propagates(self):
        feedback = np.zeros(1, dtype=HAND_STATE_DTYPE)
        feedback['timestamp_ns'] = self.now
        for stop_error in (None, RobotError('stop failed')):
            with self.subTest(stop_error=stop_error):
                robot = Mock(hand_config=self.runtime.hand)
                robot.hand_feedback.snapshot.return_value = feedback
                robot.send_hand_home.side_effect = HomeTimeout('HOME timed out')
                robot.stop.side_effect = [None, stop_error]
                with (
                    patch('dexmani_real.runtime.home.time.monotonic_ns', return_value=self.now),
                    self.assertRaisesRegex(type(stop_error) if stop_error else HomeTimeout,
                                           'stop failed' if stop_error else 'HOME timed out'),
                ):
                    home_hand(self.state, robot, cancel_epoch=self.state.cancel_epoch,
                              cancel_requested=lambda: False)
                self.assertEqual(robot.stop.call_count, 2)
                self.assertEqual(robot.method_calls[-1][0], 'stop')

    def make_rollout(self):
        runtime = replace(self.runtime, execution=replace(self.runtime.execution, control_dt_s=0.1, action_steps=2))
        info = NS(n_obs_steps=2, observation_fields=('rgb',), action_mode='joint', requires_prev_action=False)
        builder = ObservationBuilder(info, runtime)
        robot = Mock()
        rollout = Rollout(self.state, runtime, info, cancel_epoch=self.state.cancel_epoch,
                          robot=robot, sensors=None, observations=builder, read_needs=builder.needs,
                          record_needs=ObservationNeeds(), action_mapper=None, model=Mock(future=None),
                          max_running_s=None, index=1, output_dir=None, record_raw=False)
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm['timestamp_ns'] = hand['timestamp_ns'] = self.now
        arm['qpos'] = self.runtime.arm.home_qpos
        row = Observation(arm, hand, {'timestamp_ns': self.now, 'rgb': np.ones((2, 2, 3), np.uint8)},
                          None, None, self.now)
        return rollout, row

    def test_stale_input_discards_plan_and_new_episode_refills_history(self):
        rollout, row = self.make_rollout()
        with patch('dexmani_real.deployment.rollout.time.monotonic_ns', return_value=self.now):
            self.assertTrue(rollout.update_history(row))
            rollout.plan = rollout.query = object()
            rollout.prev_action = np.ones(19)
            stale = replace(row, camera=dict(row.camera, timestamp_ns=self.now-10**9))
            self.assertFalse(rollout.update_history(stale))
            self.assertTrue(self.state.stop_signal.is_set())
            self.assertEqual(len(rollout.history), 0)
            self.assertIsNone(rollout.plan)
            self.assertIsNone(rollout.query)
            self.assertIsNone(rollout.prev_action)
            rollout.step()
            rollout.robot.send_action.assert_not_called()
            new, fresh = self.make_rollout()
            self.assertEqual(len(new.history), 0)
            self.assertTrue(new.update_history(fresh))
            self.assertEqual(len(new.history), 2)
            for sample in new.history:
                np.testing.assert_array_equal(sample['rgb'], fresh.camera['rgb'])
            self.state.stop_signal.clear()  # 模拟设备完成新段准备；测试不调用硬件。
            new.submit(None)
            self.assertTrue(new.model.predict.call_args.kwargs['reset_episode'])

    def test_initial_observation_resamples_and_cancellation_wins(self):
        rollout, row = self.make_rollout()
        stale = replace(row, camera=dict(row.camera, timestamp_ns=self.now-10**9))
        self.state.stop_signal.set()
        with patch('dexmani_real.deployment.rollout.time.monotonic_ns', return_value=self.now), \
                patch('dexmani_real.deployment.rollout.time.sleep'), \
                patch.object(rollout, 'read_observations', side_effect=[(stale, stale), (row, row)]) as read:
            self.assertIs(rollout.start_observation(lambda: None), row)
            self.assertEqual(read.call_count, 2)
            self.assertTrue(self.state.stop_signal.is_set())
            rollout.robot.send_action.assert_not_called()
        for cancel in (self.state.request_stop, self.state.request_emergency_stop):
            rollout.cancel_epoch = self.state.cancel_epoch
            with self.subTest(cancel=cancel), patch.object(rollout, 'read_observations') as read:
                with self.assertRaises(OperationCancelled):
                    rollout.start_observation(cancel)
                read.assert_not_called()

    def test_rollout_stops_and_propagates_fatal_errors(self):
        for error in (RobotError('required sensor process exited'), RobotError('SDK failed')):
            rollout, _ = self.make_rollout()
            with patch.object(rollout, 'prepare', side_effect=error), self.assertRaisesRegex(RobotError, str(error)):
                rollout.run(poll_operator=Mock())
            rollout.robot.stop.assert_called_once()
            self.assertTrue(rollout.result.errors)
        rollout, _ = self.make_rollout()
        rollout.robot.stop.side_effect = RobotError('stop failed')
        with patch.object(rollout, 'prepare', side_effect=lambda poll: rollout.stop('camera_source_stale')), \
                self.assertRaisesRegex(RobotError, 'stop failed'):
            rollout.run(poll_operator=Mock())
        self.assertEqual(rollout.result.errors[-1]['stage'], 'stop')

    def test_episode_finalization_consumes_late_prediction_before_idle(self):
        episode, _ = self.make_rollout()
        episode.stop('camera_source_stale')
        session = self.make_policy_session(episode)
        session.episode = episode
        future: Future[np.ndarray] = Future()
        future.set_result(np.ones((2, 19)))
        session.model = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
        session.model.future = future
        self.addCleanup(lambda: session.model.close().result(timeout=1))
        with patch.object(session, 'poll_operator'), patch.object(session, 'save_results'), \
                patch.object(episode, 'finish_recording', side_effect=[False, True]) as finish:
            session.finish_episode()
        self.assertEqual(finish.call_count, 2)
        self.assertIsNone(session.model.future)
        self.assertIsNone(session.episode)
        self.assertIsNone(episode.plan)
        self.assertTrue(self.state.stop_signal.is_set())
        self.assertIsNone(session.handle_command_batch([]))

    def test_preparation_end_is_reported_before_restart_hint(self):
        episode, _ = self.make_rollout()
        episode.stop('device feedback: arm_feedback_stale')
        session = self.make_policy_session(episode)
        with self.assertLogs('dexmani_real.deployment.session', level='INFO') as logs, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.run_policy_episode(session, episode)
        self.assertIn('arm_feedback_stale', logs.output[0])
        self.assertIn('停止流程已完成', output.getvalue())
        self.assertIsNone(episode.result.started_utc)
        self.assertIsNone(session.episode)
        episode.robot.stop.assert_called_once()

    def test_teleop_restart_hint_only_after_successful_stop(self):
        for error in (None, RobotError('stop failed')):
            recorder, _ = self.make_recorder('published')
            loop = self.make_teleop(recorder)
            loop.robot.stop.side_effect = error
            with self.assertLogs('dexmani_real.teleop.loop', level='INFO') as logs:
                if error:
                    with self.assertRaises(RobotError):
                        loop.end_episode()
                else:
                    loop.end_episode()
            output = '\n'.join(logs.output)
            self.assertEqual('停止流程已完成' in output, error is None)
            self.assertEqual('按 B 开始' in output, error is None)
            if error:
                self.assertIs(loop.recorder, recorder)
                recorder.poll_finish.assert_not_called()
            loop.controller.clear.assert_called_once()

    def make_recorder(self, status, *, truncated=None, staging_available=True):
        temporary = tempfile.TemporaryDirectory(prefix='dexmani_hint_', dir='/tmp')
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name)
        recorder = Mock(spec=EpisodeRecorder,
                        error=RecordingError('disk write failed') if status == 'failed' else None,
                        writer_pending=False, written_frames=0 if status == 'not_published' else 3,
                        published_path=path if status == 'published' else None,
                        staging_path=path if status == 'failed' and staging_available else None,
                        recording_truncated_reason=truncated, finish_request=NS(save=True, reason='operator'))
        # 复用真实结果生成逻辑，避免测试另行维护一套 Raw 字段和状态规则。
        recorder.result.side_effect = lambda: EpisodeRecorder.result(recorder)
        recorder.poll_finish.return_value = True
        return recorder, recorder.result()

    def make_teleop(self, recorder=None):
        loop = TeleopLoop(self.state, self.runtime, Mock(), task_label='test', sensors=None,
                          controller=Mock(), recording_dir='/tmp' if recorder is not None else None,
                          max_duration_s=1, keyboard=Mock(), audio=Mock())
        loop.recorder = recorder
        loop.phase = TeleopPhase.ACTIVE
        loop.pending_termination_reason = 'operator'
        loop.keyboard.poll.return_value = ([], False)
        loop.keyboard.accept_operation.return_value = self.state.cancel_epoch
        return loop

    def assert_recording_hint(self, output, raw):
        if raw is not None and raw['status'] == 'failed':
            self.assertIn('Raw 录制失败，未发布有效 Episode', output)
            self.assertIn(raw['error'], output)
            self.assertIn('Staging：' + (raw['staging_path'] or '不可用'), output)
            self.assertIn('本次录制失败仍会保留在实验错误记录中', output)
            self.assertNotIn('停止流程已完成；', output)
            self.assertNotIn('Raw 保存成功', output)
            self.assertNotIn('部分数据已保存', output)
        else:
            self.assertIn('停止流程已完成；', output)
            self.assertNotIn('Raw 录制失败', output)
            if raw is not None and raw['status'] == 'published':
                self.assertIn(raw['path'], output)
                if raw['recording_truncated_reason']:
                    self.assertIn('部分数据已保存，录制发生截断', output)
                    self.assertIn(raw['recording_truncated_reason'], output)
                    self.assertNotIn('Raw 保存成功', output)
                else:
                    self.assertIn('Raw 保存成功', output)
                    self.assertNotIn('部分数据已保存', output)
            elif raw is not None:
                self.assertIn('Raw 未发布', output)
                self.assertNotIn('Raw 保存成功', output)
        self.assertIn('按 B 开始', output)

    def test_teleop_recording_result_hints_and_history(self):
        for status, truncated, history in (
            ('failed', None, False), ('published', None, False),
            ('published', 'recording_queue_full', False), (None, None, False),
            ('not_published', None, False), ('published', None, True),
        ):
            with self.subTest(status=status, truncated=truncated, history=history):
                recorder, raw = self.make_recorder(status, truncated=truncated) if status else (None, None)
                loop = self.make_teleop(recorder)
                loop.failed = history
                with self.assertLogs('dexmani_real.teleop.loop', level='INFO') as logs:
                    loop.end_episode()
                self.assert_recording_hint('\n'.join(logs.output), raw)
                self.assertEqual(loop.failed, history or status == 'failed')
                self.assertEqual(any(d['stage'] == 'recording' for d in loop.termination_details),
                                 status == 'failed')
                self.assertIsNone(loop.recorder)
                self.assertIs(loop.phase, TeleopPhase.IDLE)
                loop.robot.stop.assert_called_once()
                if recorder is not None:
                    recorder.poll_finish.assert_called_once()
                with patch.object(loop, 'run_episode') as run:
                    self.assertTrue(loop.handle_operator_command('b'))
                    run.assert_called_once_with(self.state.cancel_epoch)

    def make_policy_session(self, episode):
        with patch('dexmani_real.deployment.session.SessionState', return_value=self.state):
            session = PolicySession(self.runtime, 'offline.pt', weights=None, inference_steps=None,
                output_root=None, device='cpu', seed=0, max_running_s=1, num_episodes=None,
                record_raw=episode.recorder is not None, camera_calibration_path=None)
        session.robot = episode.robot
        session.info = episode.policy_info
        session.observations = episode.observations
        session.read_needs = episode.read_needs
        session.record_needs = episode.record_needs
        session.action_mapper = None
        session.keyboard = Mock()
        session.keyboard.accept_operation.return_value = self.state.cancel_epoch
        session.model = episode.model
        closed: Future[None] = Future()
        closed.set_result(None)
        session.model.close.return_value = closed
        return session

    def run_policy_episode(self, session, episode):
        def poll(**kwargs):
            if session.episode is not None:
                return None
            epoch = session.handle_command_batch(['b'])
            self.assertEqual(epoch, self.state.cancel_epoch)
            if episode.result in session.episodes:
                self.state.request_quit()
                return None
            return epoch

        with patch('dexmani_real.deployment.session.Rollout', return_value=episode), \
                patch.object(episode, 'prepare', side_effect=lambda poll: episode.stop('operator')), \
                patch.object(session, 'poll_operator', side_effect=poll):
            session.run_control_loop()

    def test_policy_recording_result_hints_and_history(self):
        for status, truncated, history in (
            ('failed', None, False), ('published', None, False),
            ('published', 'recording_queue_full', False), (None, None, False),
            ('not_published', None, False), ('published', None, True),
        ):
            with self.subTest(status=status, truncated=truncated, history=history):
                self.state.quit_requested = False
                episode, _ = self.make_rollout()
                recorder, raw = self.make_recorder(status, truncated=truncated) if status else (None, None)
                episode.recorder = recorder
                session = self.make_policy_session(episode)
                old = EpisodeResult(0, errors=[{'stage': 'recording', 'message': 'previous disk failure'}])
                if history:
                    session.episodes.append(old)
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.run_policy_episode(session, episode)
                self.assert_recording_hint(output.getvalue(), raw)
                self.assertEqual(episode.result.raw, raw)
                self.assertEqual(bool(episode.result.errors), status == 'failed')
                if status == 'failed':
                    assert raw is not None
                    self.assertEqual(episode.result.errors[-1]['stage'], 'recording')
                    self.assertEqual(episode.result.errors[-1]['message'], raw['error'])
                self.assertIsNone(session.episode)
                self.assertIsNone(episode.recorder)
                episode.robot.stop.assert_called_once()
                with patch('dexmani_real.deployment.session.close_runtime'):
                    self.assertEqual(session.close(), int(history or status == 'failed'))
                if history:
                    self.assertEqual(old.errors[0]['message'], 'previous disk failure')
                if status == 'failed':
                    self.assertTrue(episode.result.errors)

    def test_recording_failure_reports_unavailable_staging(self):
        for mode in ('teleop', 'policy'):
            for missing in (None, 'removed'):
                with self.subTest(mode=mode, missing=missing):
                    self.state.quit_requested = False
                    recorder, raw = self.make_recorder('failed', staging_available=False)
                    if missing:
                        temporary = tempfile.TemporaryDirectory(prefix='dexmani_removed_', dir='/tmp')
                        recorder.staging_path = Path(temporary.name)
                        raw = recorder.result()
                        temporary.cleanup()
                    if mode == 'teleop':
                        with self.assertLogs('dexmani_real.teleop.loop', level='INFO') as logs:
                            self.make_teleop(recorder).end_episode()
                        output = '\n'.join(logs.output)
                    else:
                        episode, _ = self.make_rollout()
                        episode.recorder = recorder
                        with contextlib.redirect_stdout(io.StringIO()) as stdout:
                            self.run_policy_episode(self.make_policy_session(episode), episode)
                        output = stdout.getvalue()
                    self.assertIn('Staging：不可用', output)
                    self.assertIn(raw['error'], output)
                    if missing:
                        self.assertIn(raw['staging_path'], output)

    def test_recording_pending_defers_release_and_restart_hint(self):
        for mode in ('teleop', 'policy'):
            with self.subTest(mode=mode):
                self.state.quit_requested = False
                recorder, raw = self.make_recorder('failed')
                recorder.writer_pending = True
                owner = self.make_teleop(recorder) if mode == 'teleop' else self.make_rollout()[0]
                owner.recorder = recorder
                output = io.StringIO()

                def poll_finish(owner=owner, recorder=recorder, output=output):
                    owner.robot.stop.assert_called_once()
                    self.assertIs(owner.recorder, recorder)
                    self.assertEqual(recorder.result()['status'], 'pending')
                    text = '\n'.join(logs.output) + output.getvalue()
                    self.assertNotIn('停止流程已完成', text)
                    self.assertNotIn('按 B 开始', text)
                    if recorder.poll_finish.call_count == 1:
                        return False
                    recorder.writer_pending = False
                    return True

                recorder.poll_finish.side_effect = poll_finish
                with self.assertLogs('dexmani_real', level='INFO') as logs, \
                        contextlib.redirect_stdout(output):
                    if mode == 'teleop':
                        owner.end_episode()
                    else:
                        self.run_policy_episode(self.make_policy_session(owner), owner)
                self.assertEqual(recorder.poll_finish.call_count, 2)
                self.assertIsNone(owner.recorder)
                self.assert_recording_hint('\n'.join(logs.output) + output.getvalue(), raw)

    def test_recording_interruption_retains_owner_without_restart_hint(self):
        for mode in ('teleop', 'policy'):
            for interrupt in (self.state.request_quit, self.state.request_emergency_stop):
                with self.subTest(mode=mode, interrupt=interrupt):
                    self.state.quit_requested = False
                    self.state.emergency_signal.clear()
                    recorder, _ = self.make_recorder('failed')
                    recorder.writer_pending = True
                    owner = self.make_teleop(recorder) if mode == 'teleop' else self.make_rollout()[0]
                    owner.recorder = recorder

                    def poll_finish(interrupt=interrupt):
                        interrupt()
                        return False

                    recorder.poll_finish.side_effect = poll_finish
                    with patch('dexmani_real.teleop.loop.logger') as log, \
                            contextlib.redirect_stdout(io.StringIO()) as output:
                        if mode == 'teleop':
                            owner.end_episode()
                        else:
                            session = self.make_policy_session(owner)
                            self.run_policy_episode(session, owner)
                            self.assertIs(session.episode, owner)
                    text = str(log.mock_calls) + output.getvalue()
                    self.assertNotIn('停止流程已完成', text)
                    self.assertNotIn('按 B 开始', text)
                    self.assertIs(owner.recorder, recorder)
                    self.assertEqual(recorder.result()['status'], 'pending')
                    recorder.poll_finish.assert_called_once()

    def test_recording_timeout_propagates_without_restart_hint(self):
        for mode in ('teleop', 'policy'):
            with self.subTest(mode=mode):
                recorder, _ = self.make_recorder('failed')
                recorder.writer_pending = True
                recorder.poll_finish.side_effect = TimeoutError('writer still running')
                owner = self.make_teleop(recorder) if mode == 'teleop' else self.make_rollout()[0]
                owner.recorder = recorder
                with patch('dexmani_real.teleop.loop.logger') as log, \
                        contextlib.redirect_stdout(io.StringIO()) as output, \
                        self.assertRaisesRegex(TimeoutError, 'writer still running'):
                    if mode == 'teleop':
                        owner.end_episode()
                    else:
                        self.run_policy_episode(self.make_policy_session(owner), owner)
                text = str(log.mock_calls) + output.getvalue()
                self.assertNotIn('停止流程已完成', text)
                self.assertNotIn('按 B 开始', text)
                self.assertIs(owner.recorder, recorder)

    def test_policy_stop_failure_never_prints_restart_hint(self):
        episode, _ = self.make_rollout()
        recorder, _ = self.make_recorder('published')
        episode.recorder = recorder
        session = self.make_policy_session(episode)
        episode.robot.stop.side_effect = RobotError('stop failed')
        with contextlib.redirect_stdout(io.StringIO()) as output, \
                self.assertRaisesRegex(RobotError, 'stop failed'):
            self.run_policy_episode(session, episode)
        self.assertNotIn('停止流程已完成', output.getvalue())
        self.assertNotIn('按 B 开始', output.getvalue())
        self.assertEqual(episode.result.errors[-1]['stage'], 'stop')
        self.assertIs(episode.recorder, recorder)
        self.assertIs(session.episode, episode)
        recorder.poll_finish.assert_not_called()

    def test_teleop_failure_exit_code_survives_later_success(self):
        recorder, _ = self.make_recorder('failed')
        loop = self.make_teleop(recorder)

        def run():
            loop.end_episode()
            self.assertTrue(loop.failed)
            # 新段重置结束详情，但必须保留会话失败；硬件准备完全由替身隔离。
            loop.phase = TeleopPhase.ACTIVE
            loop.pending_termination_reason = 'operator'
            loop.termination_details = []
            loop.recorder, raw = self.make_recorder('published')
            with self.assertLogs('dexmani_real.teleop.loop', level='INFO') as logs:
                loop.end_episode()
            self.assert_recording_hint('\n'.join(logs.output), raw)
            self.state.request_quit()

        with contextlib.ExitStack() as stack:
            for name in ('configure_logging', 'load_vr_transform', 'ActionMapper',
                         'build_hand_retargeter', 'VRController', 'SensorChannels',
                         'SensorWorkers', 'AudioFeedback', 'initialize_teleop', 'close_runtime'):
                stack.enter_context(patch('dexmani_real.teleop.run.' + name))
            stack.enter_context(patch('dexmani_real.teleop.run.SessionState', return_value=self.state))
            stack.enter_context(patch('dexmani_real.teleop.run.Robot', return_value=loop.robot))
            stack.enter_context(patch('dexmani_real.teleop.run.Keyboard', return_value=Mock(bindings={})))
            stack.enter_context(patch('dexmani_real.teleop.run.TeleopLoop', return_value=loop))
            stack.enter_context(patch.object(loop, 'run', side_effect=run))
            stack.enter_context(self.assertLogs('dexmani_real.teleop.loop', level='INFO'))
            self.assertEqual(run_teleop(self.runtime, record=False, episodes_dir='/tmp', max_duration_s=1), 1)
        self.assertTrue(loop.failed)

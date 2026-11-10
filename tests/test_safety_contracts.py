"""只验证现有边界；设备、时钟与反馈使用离线替身。"""

import threading
import unittest
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.rollout import Rollout
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
        with patch('dexmani_real.deployment.session.SessionState', return_value=self.state):
            session = PolicySession(self.runtime, 'offline.pt', weights=None, inference_steps=None,
                output_root=None, device='cpu', seed=0, max_running_s=1, num_episodes=1,
                record_raw=False, camera_calibration_path=None)
        episode, _ = self.make_rollout()
        episode.stop('camera_source_stale')
        session.episode = episode
        session.robot = episode.robot
        future = Future()
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
        import contextlib
        import io

        with patch('dexmani_real.deployment.session.SessionState', return_value=self.state):
            session = PolicySession(self.runtime, 'offline.pt', weights=None, inference_steps=None,
                output_root=None, device='cpu', seed=0, max_running_s=1, num_episodes=1,
                record_raw=False, camera_calibration_path=None)
        episode, _ = self.make_rollout()
        episode.stop('device feedback: arm_feedback_stale')
        session.robot = episode.robot
        session.info = episode.policy_info
        session.observations = episode.observations
        session.read_needs = episode.read_needs
        session.record_needs = episode.record_needs
        session.action_mapper = None
        session.keyboard = Mock()
        calls = []

        def poll():
            if calls:
                self.state.request_quit()
                return None
            calls.append('B')
            return self.state.cancel_epoch

        with patch('dexmani_real.deployment.session.Rollout', return_value=episode), \
                patch.object(episode, 'run'), patch.object(session, 'finish_episode'), \
                patch.object(session, 'poll_operator', side_effect=poll), \
                self.assertLogs('dexmani_real.deployment.session', level='INFO') as logs, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            session.run_control_loop()
        self.assertIn('arm_feedback_stale', logs.output[0])
        self.assertIn('停止流程已完成', output.getvalue())
        self.assertIsNone(episode.result.started_utc)

    def test_teleop_restart_hint_only_after_successful_stop(self):
        for error in (None, RobotError('stop failed')):
            robot, controller = Mock(), Mock()
            robot.stop.side_effect = error
            loop = TeleopLoop(self.state, self.runtime, robot, task_label='test', sensors=None,
                              controller=controller, recording_dir=None, max_duration_s=1,
                              keyboard=Mock(), audio=Mock())
            loop.phase = TeleopPhase.ACTIVE
            with patch('dexmani_real.teleop.loop.logger.info') as log:
                if error:
                    with self.assertRaises(RobotError):
                        loop.end_episode()
                else:
                    loop.end_episode()
                hints = [c for c in log.call_args_list if '停止流程已完成' in c.args[0]]
                self.assertEqual(len(hints), int(error is None))
            controller.clear.assert_called_once()

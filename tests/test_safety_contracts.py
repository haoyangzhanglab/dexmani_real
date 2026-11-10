"""只验证现有边界；设备、时钟与反馈使用离线替身。"""

import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.robot.commands import HomeTimeout, RobotCommand, RobotError
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

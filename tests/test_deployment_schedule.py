"""动作块调度与命令期限的离线回归；不运行模型或连接设备。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config.deployment import ACTION_TIMEOUT_NS, ExecutionConfig
from dexmani_real.config.real import RealConfig
from dexmani_real.deployment.observation import ObservationBuilder
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.schedule import Prediction, handoff_plan, plan_rejection
from dexmani_real.replay.joint_replay import JointReplay, ReplayStatus
from dexmani_real.replay.trajectory import ReplayBuffer
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.runtime.state import RunEndReason


class DeploymentScheduleTests(unittest.TestCase):
    def setUp(self):
        self.now = 100_000_000_000
        clock = patch('time.monotonic_ns', side_effect=lambda: self.now)
        clock.start()
        self.addCleanup(clock.stop)
        self.runtime = RealConfig(execution=ExecutionConfig(action_steps=64))
        info = SimpleNamespace(
            control_dt_s=0.05, n_obs_steps=1, hand_history_steps=None,
            observation_fields=('joint_state',),
        )
        needs = ObservationNeeds()
        observations = ObservationBuilder(
            info, self.runtime, dependencies={'arm', 'hand'}, needs=needs,
        )
        self.robot = SimpleNamespace(
            read_state=self.feedback, send_action=Mock(),
        )
        self.rollout = Rollout(
            SimpleNamespace(can_move=lambda **kwargs: True), self.runtime, info,
            preparing_epoch=1, robot=self.robot, observations=observations,
            record_needs=needs, needs=needs, sensors=None, action_mapper=None,
            model=SimpleNamespace(poll=lambda: None, predict=Mock()),
            max_running_s=None, capture_ids=iter(()), index=0,
            output_dir=None, record_raw=False,
        )
        self.rollout.run_id = 1
        self.rollout.next_step_ns = self.now
        self.rollout.read_observation = lambda: (self.feedback(), None, self.now + 100_000_000)
        self.rollout.command_for_slot = Mock(return_value=(RobotCommand(1, np.zeros(7)), None))
        self.rollout.stop = Mock()

    def feedback(self):
        return SimpleNamespace(
            arm={'qpos': np.array([self.runtime.arm.home_qpos]), 'timestamp_ns': [self.now]},
            hand={'qpos': np.deg2rad([self.runtime.hand.home_qpos_deg]), 'timestamp_ns': [self.now]},
            observation_timestamp_ns=self.now,
        )

    def test_long_chunk_executes_while_each_command_renews_deadline(self):
        self.rollout.step()
        self.now += 100_000_000  # 模拟同步推理耗时。
        self.rollout.query.result = np.zeros((64, 19))
        for _ in range(64):
            self.rollout.step()
            sent = self.robot.send_action.call_args.kwargs
            self.assertEqual(sent['execute_until_ns'], self.now + ACTION_TIMEOUT_NS)
            self.now = self.rollout.next_step_ns
        self.assertEqual(self.robot.send_action.call_count, 64)
        self.assertIsNone(self.rollout.plan)
        self.rollout.stop.assert_not_called()

    def test_wait_timeout_rejects_late_prediction(self):
        self.rollout.step()
        self.now += ACTION_TIMEOUT_NS
        self.rollout.query.result = np.zeros((64, 19))
        self.rollout.step()
        self.robot.send_action.assert_not_called()
        self.rollout.stop.assert_any_call('wait_timeout', run_end_reason=RunEndReason.TIMEOUT)

    def test_episode_deadline_caps_command_lifetime(self):
        deadline = self.now + 500_000_000
        self.rollout.run_deadline_ns = deadline
        self.rollout.step()
        self.rollout.query.result = np.zeros((64, 19))
        self.rollout.step()
        self.assertEqual(self.robot.send_action.call_args.kwargs['execute_until_ns'], deadline)
        self.rollout.stop.assert_not_called()

    def test_async_uses_remaining_actions_and_rejects_exhausted_prediction(self):
        query = Prediction(1, 0, 8, np.zeros((12, 19)))
        plan, issue = handoff_plan(
            query, run_id=1, slot=9, mode='async', action_steps=8, authorized=True,
        )
        self.assertIsNone(issue)
        self.assertEqual((plan.anchor_slot, plan.segment_end), (0, 12))
        self.assertIsNone(plan_rejection(plan, 11, 'async'))
        self.assertEqual(plan_rejection(plan, 12, 'async'), 'plan_exhausted_without_handoff')
        _, issue = handoff_plan(
            query, run_id=1, slot=12, mode='async', action_steps=8, authorized=True,
        )
        self.assertEqual(issue, 'handoff_prediction_exhausted')

    def test_rtc_still_requires_result_at_exact_handoff(self):
        query = Prediction(1, 0, 8)
        _, issue = handoff_plan(
            query, run_id=1, slot=8, mode='rtc', action_steps=8, authorized=True,
        )
        self.assertEqual(issue, 'handoff_result_not_ready')
        query.result = np.zeros((16, 19))
        _, issue = handoff_plan(
            query, run_id=1, slot=9, mode='rtc', action_steps=8, authorized=True,
        )
        self.assertEqual(issue, 'handoff_slot_missed')

    def test_replay_uses_command_timeout_and_keeps_submission_deadline(self):
        replay = JointReplay(None, self.runtime, None, robot=self.robot)
        replay.epoch = 1
        replay.capture = ReplayBuffer(2)
        deadline = self.now + 50_000_000
        row = self.feedback()
        replay.submit_row(0, row, np.zeros(7), np.zeros(12), deadline)
        sent = self.robot.send_action.call_args.kwargs
        self.assertEqual(sent['execute_until_ns'], self.now + ACTION_TIMEOUT_NS)
        self.assertEqual(sent['valid_until_ns'], deadline)
        self.now = deadline
        with self.assertRaisesRegex(TimeoutError, 'expired before submission'):
            replay.submit_row(1, row, np.zeros(7), np.zeros(12), deadline)
        self.robot.send_action.assert_called_once()

    def test_hand_start_wait_does_not_outlive_arm_command(self):
        replay = JointReplay(None, self.runtime, None, robot=self.robot)
        replay.epoch = 1

        def checked_target(*args, **kwargs):
            self.now += 10_000_000

        def send(*args, **kwargs):
            self.now += 20_000_000

        def wait(deadline):
            self.now = max(self.now, deadline)
            if self.robot.send_action.call_count >= 2:
                expires = self.robot.send_action.call_args.kwargs['execute_until_ns']
                self.now = max(self.now, expires + 1_000_000)

        replay.action_mapper = SimpleNamespace(constraints=SimpleNamespace(rejection=checked_target))
        replay.read_feedback = Mock(side_effect=lambda: (self.feedback(), self.now + 100_000_000))
        replay.wait_replay = wait
        self.robot.send_action.side_effect = send
        trajectory = SimpleNamespace(fps=20, action_hand_joint=self.feedback().hand['qpos'])
        with patch('builtins.print'):
            result = replay.prepare_hand(trajectory, self.feedback(), 0.05, 50_000_000)
        self.assertEqual(result, (ReplayStatus.REJECTED, 'hand start convergence timed out'))
        # 两次渐变提交后已超过保持目标的期限，不应再读取到位反馈。
        self.assertEqual(replay.read_feedback.call_count, 2)


if __name__ == '__main__':
    unittest.main()

"""验证重定向与遥操作的提交边界；设备均为替身。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_array_equal
from test_hand_retargeting import make_retargeter

from dexmani_real.config.teleop import TeleopConfig
from dexmani_real.geometry.pose import Pose
from dexmani_real.teleop.loop import TeleopLoop
from dexmani_real.teleop.vr import VRController


def observation(home, landmarks, *, sequence=1, timestamp=1_000_000_000):
    return SimpleNamespace(
        arm={"qpos": np.zeros((1, 7))},
        hand={"qpos": home[None, :].copy()},
        vr={
            "ring_sequence": sequence, "recv_ts_ns": timestamp,
            "wrist_pos": np.zeros(3), "wrist_quat_wxyz": np.array([1, 0, 0, 0]),
            "landmarks": landmarks.copy(),
        },
        observation_timestamp_ns=timestamp,
    )


def make_controller():
    retargeter, home, landmarks = make_retargeter()
    mapper = Mock()
    mapper.constraints.target_fk.forward.return_value = (
        np.zeros(3), np.array([1, 0, 0, 0, 1, 0]),
    )
    mapper.map_eef.side_effect = lambda pose, hand, *_: SimpleNamespace(
        arm_qpos=np.zeros(7), hand_qpos=hand.copy(), eef_pose=pose, rejection_reason=None,
    )
    controller = VRController(mapper, retargeter, TeleopConfig(), np.eye(3))
    row = observation(home, landmarks)
    controller.reset(row)
    return controller, home, landmarks


class ControllerTests(unittest.TestCase):
    def test_precompute_and_discard_do_not_advance_hand_command(self):
        controller, home, landmarks = make_controller()
        result = controller.compute_target(observation(home, landmarks, timestamp=1_062_500_000))
        self.assertIsNotNone(result)
        assert_array_equal(controller.hand_retargeter.previous_command, home)
        controller.clear()
        assert_array_equal(controller.hand_retargeter.previous_command, home)

    def test_committed_result_updates_hand_and_arm_together(self):
        controller, home, landmarks = make_controller()
        row = observation(home, landmarks, timestamp=1_062_500_000)
        result = controller.compute_target(row)
        # 模拟下游准入投影：必须以提交结果为准，不能使用滤波前的优化解。
        result.hand_qpos = (result.hand_qpos + home) / 2
        controller.commit(result)
        assert_array_equal(controller.hand_retargeter.previous_command, result.hand_qpos)
        self.assertEqual(controller.hand_retargeter.command_timestamp_ns, row.observation_timestamp_ns)

    def test_duplicate_frame_does_not_solve_or_filter_again(self):
        controller, home, landmarks = make_controller()
        original = controller.hand_retargeter.retarget
        controller.hand_retargeter.retarget = Mock(wraps=original)
        first = controller.compute_target(observation(home, landmarks, timestamp=1_062_500_000))
        controller.commit(first)
        second = controller.compute_target(observation(home, landmarks, timestamp=1_125_000_000))
        assert_array_equal(first.hand_qpos, second.hand_qpos)
        self.assertEqual(controller.hand_retargeter.retarget.call_count, 1)

    def test_failed_frame_reason_is_preserved_and_not_retried(self):
        controller, home, _ = make_controller()
        row = observation(home, np.zeros((21, 3)), timestamp=1_062_500_000)
        original = controller.hand_retargeter.retarget
        controller.hand_retargeter.retarget = Mock(wraps=original)
        self.assertIsNone(controller.compute_target(row))
        self.assertIn("bone", controller.rejection_reason)
        self.assertIsNone(controller.compute_target(row))
        self.assertEqual(controller.hand_retargeter.retarget.call_count, 1)
        assert_array_equal(controller.hand_retargeter.previous_command, home)

    def test_resume_reanchors_to_measurement_and_clears_cached_target(self):
        controller, home, landmarks = make_controller()
        result = controller.compute_target(observation(home, landmarks, timestamp=1_062_500_000))
        controller.commit(result)
        controller.reset(observation(home, landmarks, sequence=1, timestamp=2_000_000_000))
        assert_array_equal(controller.hand_retargeter.previous_command, home)
        self.assertIsNone(controller.hand_observation_id)
        self.assertIsNone(controller.hand_target)

    def test_hand_disabled_path_does_not_require_calibration(self):
        mapper = Mock()
        mapper.constraints.target_fk.forward.return_value = (
            np.zeros(3), np.array([1, 0, 0, 0, 1, 0]),
        )
        controller = VRController(mapper, None, TeleopConfig(), np.eye(3))
        row = observation(np.zeros(12), np.zeros((21, 3)))
        row.hand = None
        controller.reset(row)
        controller.compute_target(row)
        self.assertIsNone(mapper.map_eef.call_args.args[1])
        controller.commit(SimpleNamespace(arm_qpos=np.zeros(7), eef_pose=Pose.identity()))


class SubmissionTests(unittest.TestCase):
    def run_step(self, outcome):
        controller, home, landmarks = make_controller()
        row = observation(home, landmarks, timestamp=1_062_500_000)
        row.control_timestamp_ns = row.observation_timestamp_ns
        owner = SimpleNamespace(
            run_id=1, runtime=SimpleNamespace(teleop=TeleopConfig()), controller=controller,
            recorder=None, robot=Mock(), runtime_state=Mock(), tick_index=2,
            record_control_timing=Mock(), record_error=Mock(), abort_episode=Mock(),
        )
        owner.runtime_state.can_move.return_value = True
        controller.commit = Mock(wraps=controller.commit)
        now = 1_063_000_000
        if outcome == "send_timeout":
            owner.robot.send_action.side_effect = TimeoutError("simulated submission timeout")
        if outcome == "arm_rejection":
            original = controller.action_mapper.map_eef.side_effect

            def reject(*args):
                result = original(*args)
                result.rejection_reason = "collision"
                return result

            controller.action_mapper.map_eef.side_effect = reject
        if outcome == "deadline":
            now = 2_000_000_000
        with patch("dexmani_real.teleop.loop.time.monotonic_ns", return_value=now):
            TeleopLoop.execute_control_step(owner, row, 1_100_000_000, 1_100_000_000)
        return owner, home

    def test_successful_send_commits_once(self):
        owner, _ = self.run_step("success")
        owner.robot.send_action.assert_called_once()
        owner.controller.commit.assert_called_once()
        owner.abort_episode.assert_not_called()

    def test_send_timeout_arm_rejection_and_deadline_never_commit_candidate(self):
        for outcome in ("send_timeout", "arm_rejection", "deadline"):
            with self.subTest(outcome=outcome):
                owner, home = self.run_step(outcome)
                owner.controller.commit.assert_not_called()
                owner.abort_episode.assert_called_once()
                assert_array_equal(owner.controller.hand_retargeter.previous_command, home)
                if outcome != "send_timeout":
                    owner.robot.send_action.assert_not_called()


if __name__ == "__main__":
    unittest.main()

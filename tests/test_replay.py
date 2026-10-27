import json
import unittest
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import h5py
import numpy as np

from dexmani_real.config import load_experiment_config
from dexmani_real.data.raw.schema import ARM_ACTION_SEMANTICS, TIMING_ROW_DTYPE
from dexmani_real.geometry.kinematics import eef_pose_sequence
from dexmani_real.geometry.pose import rot6d_to_rotmat
from dexmani_real.replay import metrics, player
from dexmani_real.replay.data import (
    ReplayBuffer,
    Trajectory,
    load_trajectory,
    verify_replay_preflight,
)
from dexmani_real.robot.commands import RobotCommand, RobotError
from dexmani_real.robot.model import XARM7_JOINT_NAMES, XHAND_SDK_JOINT_NAMES
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.state import SessionState


def trajectory(count=3):
    return Trajectory(
        "episode",
        16,
        "test",
        np.zeros((count, 7)),
        np.zeros((count, 12)),
        np.zeros((count, 7)),
        np.zeros((count, 12)),
    )


def observation(stamp):
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    arm["timestamp_ns"], hand["timestamp_ns"] = stamp, stamp
    return SimpleNamespace(arm=arm, hand=hand, observation_timestamp_ns=stamp)


class ReplayDataTests(unittest.TestCase):
    def test_raw_loading_and_rejection_before_motion(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            with h5py.File(path / "data.h5", "w") as data:
                data.create_group("meta").attrs.update(
                    {
                        "arm_action_semantics": ARM_ACTION_SEMANTICS,
                        "timestamp_source": "host_monotonic_relative",
                        "joint_units": "rad",
                        "arm_joint_names": json.dumps(list(XARM7_JOINT_NAMES)),
                        "hand_joint_names": json.dumps(list(XHAND_SDK_JOINT_NAMES)),
                        "control_hz": 16,
                        "collection_source": "teleop",
                        "technical_status": "normal",
                        "task_label": "test",
                    }
                )
                for name, width in (
                    ("arm_qpos", 7),
                    ("hand_qpos", 12),
                    ("action_arm_joint_target", 7),
                    ("action_hand_joint_target", 12),
                ):
                    data[name] = np.zeros((2, width))
                data["timestamp"] = np.array([0, 1 / 16])
                timing = np.zeros(2, dtype=TIMING_ROW_DTYPE)
                for name in ("observation_ns", "control_ns", "arm_ns", "hand_ns"):
                    timing[name] = [1_000_000_000, 1_062_500_000]
                data["timing/rows"] = timing
            loaded = load_trajectory(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded.fps, 16)
            with h5py.File(path / "data.h5", "r+") as data:
                data["action_arm_joint_target"][0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                load_trajectory(path)
            loaded.action_arm_joint[0, 0] = 100
            with self.assertRaisesRegex(ValueError, "absolute limits"):
                verify_replay_preflight(loaded, load_experiment_config())

    def test_buffer_owns_samples_and_marks_unattempted_targets(self):
        buffer = ReplayBuffer(2)
        row = observation(123)
        command = RobotCommand(1, np.ones(7), np.ones(12))
        buffer.append(row, command)
        row.arm["qpos"] = 5
        captured = buffer.to_dict()
        np.testing.assert_array_equal(captured["arm_qpos"], np.zeros((1, 7)))
        np.testing.assert_array_equal(captured["arm_candidate"], np.ones((1, 7)))
        self.assertEqual(captured["submit_started_ns"].tolist(), [-1])
        captured["arm_candidate"][:] = 9
        np.testing.assert_array_equal(buffer.to_dict()["arm_candidate"], np.ones((1, 7)))

    def test_metrics_use_matching_prefix_and_known_geometric_errors(self):
        source = trajectory(5)
        replayed = {"arm_qpos": np.zeros((2, 7)), "hand_qpos": np.full((2, 12), np.deg2rad(2))}
        replayed["arm_qpos"][:, 0] = np.deg2rad(1)
        result = metrics.compute_metrics(source, replayed)
        self.assertEqual(result["matching_frames"], 2)
        self.assertAlmostEqual(result["arm_joint"]["mae_per_joint_deg"][0], 1)
        self.assertAlmostEqual(result["hand_joint"]["pooled_rmse_deg"], 2)
        a = eef_pose_sequence(source.arm_qpos[:2])[0]
        b = eef_pose_sequence(replayed["arm_qpos"])[0]
        angle = np.rad2deg(
            np.arccos(
                np.clip(
                    (np.trace(rot6d_to_rotmat(a[3:]) @ rot6d_to_rotmat(b[3:]).T) - 1) / 2, -1, 1
                )
            )
        )
        self.assertAlmostEqual(result["eef_orientation"]["mean_error_deg"], angle, places=6)
        self.assertAlmostEqual(
            result["eef_position"]["mean_error_mm"], np.linalg.norm(a[:3] - b[:3]) * 1000
        )

    def test_identical_and_empty_replay(self):
        source = trajectory()
        result = metrics.compute_metrics(
            source, {"arm_qpos": source.arm_qpos, "hand_qpos": source.hand_qpos}
        )
        self.assertEqual(result["arm_joint"]["mae_overall_deg"], 0)
        self.assertLess(result["eef_orientation"]["max_error_deg"], 1e-5)
        with TemporaryDirectory() as directory, patch.object(metrics, "ArmFK") as fk:
            metrics.evaluate_replay(source, ReplayBuffer(0).to_dict(), output_dir=directory)
            fk.assert_not_called()
            saved = json.loads((Path(directory) / "metrics.json").read_text())
            self.assertEqual(saved["matching_frames"], 0)
            self.assertIsNone(saved["arm_joint"])

    def test_raw_capture_survives_fk_failure(self):
        buffer = ReplayBuffer(1)
        buffer.append(observation(1), RobotCommand(1, np.zeros(7), np.zeros(12)))
        with (
            TemporaryDirectory() as directory,
            patch.object(metrics, "ArmFK", side_effect=RuntimeError("FK failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "FK failed"):
                metrics.evaluate_replay(trajectory(), buffer.to_dict(), output_dir=directory)
            with np.load(Path(directory) / "replay_data.npz") as saved:
                self.assertEqual(saved["arm_qpos"].shape, (1, 7))
                self.assertNotIn("technical_status", saved)
            self.assertFalse((Path(directory) / "metrics.json").exists())


class ReplayControlTests(unittest.TestCase):
    def setUp(self):
        self.clock = 1_000_000_000
        self.runtime = load_experiment_config()
        self.state = SessionState()
        self.state.arm()
        self.state.stop_signal.clear()
        self.robot = Mock()
        self.keys = Mock(healthy=True)
        self.keys.poll.return_value = []
        self.driver = player.ReplayPlayer(self.state, self.runtime, self.keys, robot=self.robot)
        self.driver.read_feedback = lambda: (observation(self.clock), self.clock + 100_000_000)
        self.mapper = Mock()
        self.mapper.prepare_joint_target.side_effect = lambda arm, hand, *_: (
            arm.copy(),
            hand.copy(),
            None,
        )
        self.mapper.constraints.rejection.return_value = None
        self.source = trajectory()
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.object(player.ActionMapper, "for_mode", return_value=self.mapper))
        stack.enter_context(
            patch.object(player.time, "monotonic_ns", side_effect=lambda: self.clock)
        )
        stack.enter_context(patch.object(player.time, "sleep", side_effect=self.advance))
        stack.enter_context(patch.object(player.logger, "exception"))
        stack.enter_context(patch("builtins.print"))

    def advance(self, seconds):
        self.clock += max(1, round(seconds * 1e9))

    def run_replay(self, duration=0):
        return self.driver.run(self.source, hand_start_duration_s=duration)

    def test_completed_replay_records_attempt_times(self):
        result = self.run_replay()
        self.assertEqual(result.status, player.ReplayStatus.COMPLETED)
        self.assertEqual(self.robot.send_action.call_count, len(self.source))
        self.assertTrue(np.all(result.replay_data["submit_started_ns"] > 0))
        self.assertTrue(
            np.all(
                result.replay_data["submit_completed_ns"] >= result.replay_data["submit_started_ns"]
            )
        )
        self.robot.stop.assert_called_once()
        self.assertFalse(self.state.can_move(run_id=self.driver.epoch))

    def test_rejected_target_is_recorded_without_submission(self):
        self.mapper.prepare_joint_target.side_effect = None
        self.mapper.prepare_joint_target.return_value = (np.zeros(7), np.zeros(12), "collision")
        result = self.run_replay()
        self.assertEqual(result.status, player.ReplayStatus.REJECTED)
        self.assertEqual(result.replay_data["submit_started_ns"].tolist(), [-1])
        self.robot.send_action.assert_not_called()

    def test_send_and_stop_failures_preserve_both_causes_and_prefix(self):
        self.robot.send_action.side_effect = RobotError("send failed")
        self.robot.stop.side_effect = RobotError("stop failed")
        result = self.run_replay()
        self.assertEqual(result.status, player.ReplayStatus.FAULT)
        self.assertIn("send failed", result.reason)
        self.assertIn("stop failed", result.reason)
        self.assertEqual(len(result.replay_data["arm_qpos"]), 1)
        self.assertGreater(result.replay_data["submit_completed_ns"][0], 0)

    def test_quit_preserves_partial_data(self):
        self.robot.send_action.side_effect = lambda *args, **kwargs: self.state.request_quit()
        result = self.run_replay()
        self.assertEqual(result.status, player.ReplayStatus.USER_QUIT)
        self.assertEqual(len(result.replay_data["arm_qpos"]), 1)

    def test_keyboard_interrupt_latches_estop_and_preserves_attempt(self):
        self.robot.send_action.side_effect = KeyboardInterrupt
        result = self.run_replay()
        self.assertEqual(result.status, player.ReplayStatus.ESTOP)
        self.assertTrue(self.state.estop_request)
        self.assertEqual(len(result.replay_data["arm_qpos"]), 1)
        self.robot.stop.assert_called_once()

    def test_late_submission_stops_without_committing_reference(self):
        self.source.action_arm_joint[:] = 0.01
        self.robot.send_action.side_effect = lambda *args, **kwargs: self.advance(0.1)
        result = self.run_replay()
        self.assertEqual(result.status, player.ReplayStatus.REJECTED)
        self.assertIn("after deadline", result.reason)
        np.testing.assert_array_equal(self.driver.previous_arm, np.zeros(7))
        self.robot.send_action.assert_called_once()

    def test_missed_slot_does_not_catch_up(self):
        self.keys.poll.side_effect = lambda **kwargs: self.advance(0.1) or []
        result = self.run_replay()
        self.assertEqual(result.status, player.ReplayStatus.REJECTED)
        self.assertEqual(len(result.replay_data["arm_qpos"]), 0)
        self.robot.send_action.assert_not_called()

    def test_stale_feedback_does_not_refresh_health(self):
        with patch.object(player, "read_observation", return_value=observation(1)):
            with self.assertRaisesRegex(TimeoutError, "stale"):
                player.ReplayPlayer.read_feedback(self.driver)
        self.robot.arm.update_health.assert_not_called()

    def test_hand_preparation_stays_outside_replay_rows(self):
        hand = np.clip(np.zeros(12), self.runtime.hand.qpos_min_rad, self.runtime.hand.qpos_max_rad)
        self.source.hand_qpos[:] = hand
        self.source.action_hand_joint[:] = hand

        def feedback():
            row = observation(self.clock)
            row.hand["qpos"] = hand
            return row, self.clock + 100_000_000

        self.driver.read_feedback = feedback
        result = self.run_replay(0.05)
        self.assertEqual(result.status, player.ReplayStatus.COMPLETED, result.reason)
        self.assertEqual(len(result.replay_data["arm_qpos"]), len(self.source))
        self.assertGreater(self.robot.send_action.call_count, len(self.source))
        self.robot.end_hand_send_interval.assert_called_once()


if __name__ == "__main__":
    unittest.main()

import unittest
from collections import deque
from unittest.mock import patch

import numpy as np

from dexmani_real.config.real import RealConfig
from dexmani_real.data.hoi import align_hoi_episode
from dexmani_real.data.raw import TIMING_ROW_DTYPE, RawDataError
from dexmani_real.robot.system import Robot, RobotState
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.runtime.state import SessionState
from dexmani_real.utils.timing import interpolate_joints

MS = 1_000_000


def raw_joints(camera_ms, arm_ms, hand_ms):
    timing = np.zeros(len(camera_ms), dtype=TIMING_ROW_DTYPE)
    timing["camera_ns"] = np.asarray(camera_ms) * MS
    timing["arm_ns"] = np.asarray(arm_ms) * MS
    timing["hand_ns"] = np.asarray(hand_ms) * MS
    latest_feedback = np.maximum.reduce([timing[key] for key in ("camera_ns", "arm_ns", "hand_ns")])
    timing["observation_ns"] = latest_feedback + np.arange(1, len(timing) + 1) * MS
    timing["control_ns"] = timing["observation_ns"] - MS // 2
    reader = {
        "timing/rows": timing,
        "arm_qpos": np.tile(np.asarray(arm_ms)[:, None], (1, 7)).astype(float),
        "hand_qpos": np.tile(np.asarray(hand_ms)[:, None], (1, 12)).astype(float),
    }
    return reader, timing


class InterpolateJointsTest(unittest.TestCase):
    def test_exact_hits_ignore_surrounding_gap(self):
        positions, valid = interpolate_joints(
            [10 * MS, 50 * MS], [[1.0, -1.0], [5.0, -5.0]], [10 * MS, 50 * MS], 0.01,
        )
        np.testing.assert_array_equal(positions, [[1.0, -1.0], [5.0, -5.0]])
        np.testing.assert_array_equal(valid, [True, True])

    def test_linear_interpolation_at_gap_limit(self):
        positions, valid = interpolate_joints(
            [10 * MS, 30 * MS], [[1.0, -1.0], [5.0, -5.0]], [15 * MS], 0.02,
        )
        np.testing.assert_array_equal(positions, [[2.0, -2.0]])
        np.testing.assert_array_equal(valid, [True])

    def test_missing_left_or_right_endpoint(self):
        positions, valid = interpolate_joints(
            [10 * MS, 20 * MS], [[1.0], [2.0]], [5 * MS, 25 * MS], 0.10,
        )
        self.assertTrue(np.isnan(positions).all())
        np.testing.assert_array_equal(valid, [False, False])

    def test_bracket_over_gap_limit(self):
        positions, valid = interpolate_joints(
            [10 * MS, 50 * MS], [[1.0], [5.0]], 20 * MS, 0.03,
        )
        self.assertTrue(np.isnan(positions[0]).all())
        self.assertFalse(valid[0])

    def test_repeated_identical_states(self):
        positions, valid = interpolate_joints(
            [10 * MS, 10 * MS, 30 * MS], [[1.0], [1.0], [3.0]], [10 * MS, 20 * MS], 0.10,
        )
        np.testing.assert_array_equal(positions, [[1.0], [2.0]])
        np.testing.assert_array_equal(valid, [True, True])

    def test_repeated_timestamps_with_conflicting_states(self):
        with self.assertRaisesRegex(ValueError, "Conflicting joint states at the same timestamp"):
            interpolate_joints([10 * MS, 10 * MS], [[1.0], [2.0]], 10 * MS, 0.10)


class OnlineJointAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.robot = Robot.__new__(Robot)
        self.robot.joint_history = {
            "arm": deque([(10 * MS, np.full(7, 1.0)), (30 * MS, np.full(7, 3.0))]),
            "hand": deque([(20 * MS, np.full(12, 7.0)), (35 * MS, np.full(12, 8.0))]),
        }

    def test_both_devices_use_camera_timestamp(self):
        aligned = self.robot.aligned_joints(20 * MS, 0.10)
        self.assertEqual(set(aligned), {"arm_qpos", "hand_qpos"})
        np.testing.assert_array_equal(aligned["arm_qpos"], np.full(7, 2.0))
        np.testing.assert_array_equal(aligned["hand_qpos"], np.full(12, 7.0))

    def test_one_device_has_no_bracket(self):
        self.assertIsNone(self.robot.aligned_joints(15 * MS, 0.10))
        self.assertIsNone(self.robot.aligned_joints(33 * MS, 0.10))

    def test_one_device_has_no_history(self):
        self.robot.joint_history["hand"].clear()
        self.assertIsNone(self.robot.aligned_joints(20 * MS, 0.10))

    def test_robot_uses_session_gap_config(self):
        self.assertIsNone(self.robot.aligned_joints(20 * MS, 0.01))

    def test_robot_rejects_conflicting_timestamp_from_feedback(self):
        config = RealConfig()
        robot = Robot(config.arm, config.hand, runtime_state=SessionState())
        robot.enable_joint_history(0.10, devices=("arm",))
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        arm["timestamp_ns"] = 30 * MS
        arm["qpos"] = 1.0
        with patch.object(robot, "read_state", return_value=RobotState(arm, None)):
            robot.service_feedback()
            robot.feedback_next_ns = 0
            robot.service_feedback()
            self.assertEqual(len(robot.joint_history["arm"]), 1)

            arm["qpos"] = 2.0
            robot.feedback_next_ns = 0
            with self.assertRaisesRegex(ValueError, "Conflicting joint states at the same timestamp"):
                robot.service_feedback()
        self.assertTrue(robot.runtime_state.stop_signal.is_set())


class OfflineJointAlignmentTest(unittest.TestCase):
    def test_never_interpolates_across_segments(self):
        reader, timing = raw_joints([15, 25], [10, 30], [10, 30])
        boundaries = np.array([0, 1, 2])
        aligned = align_hoi_episode(reader, timing, boundaries, 0.10)
        self.assertTrue(np.isnan(aligned["arm_qpos"]).all())
        self.assertTrue(np.isnan(aligned["hand_qpos"]).all())
        with self.assertRaisesRegex(RawDataError, r"arm: unalignable row 0 in segment 0"):
            align_hoi_episode(reader, timing, boundaries, 0.10, require_complete=True)

    def test_complete_export_reports_missing_device_and_row(self):
        reader, timing = raw_joints([10, 20, 30], [10, 20, 30], [10, 10, 20])
        with self.assertRaisesRegex(RawDataError, r"hand: unalignable row 2 in segment 0"):
            align_hoi_episode(reader, timing, np.array([0, 3]), 0.10, require_complete=True)

    def test_prefix_can_use_neighbors_in_the_same_segment(self):
        reader, timing = raw_joints([20, 30, 40, 50], [10, 15, 25, 30], [10, 15, 25, 30])
        aligned = align_hoi_episode(reader, timing[:1], np.array([0, 4]), 0.10)
        np.testing.assert_array_equal(aligned["arm_qpos"], np.full((1, 7), 20.0))
        np.testing.assert_array_equal(aligned["hand_qpos"], np.full((1, 12), 20.0))

    def test_prefix_checks_conflicts_in_used_timestamp_groups(self):
        for joints_ms, camera_ms, duplicate_row in (
            ([10, 10, 30], 10, 1),
            ([10, 20, 20], 20, 2),
            ([10, 20, 20, 30], 25, 2),
        ):
            with self.subTest(joints_ms=joints_ms, camera_ms=camera_ms):
                reader, timing = raw_joints([camera_ms] * len(joints_ms), joints_ms, joints_ms)
                reader["arm_qpos"][duplicate_row] += 1.0
                with self.assertRaisesRegex(ValueError, "Conflicting joint states at the same timestamp"):
                    align_hoi_episode(reader, timing[:1], np.array([0, len(joints_ms)]), 0.10)

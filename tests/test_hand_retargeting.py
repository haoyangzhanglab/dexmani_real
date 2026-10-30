"""真实 XHand URDF / Pinocchio / NLopt 数值回归；不连接设备。"""

import unittest
from dataclasses import replace
from unittest.mock import Mock

import nlopt
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from dexmani_real.config.real import load_config
from dexmani_real.config.teleop import HandRetargetingConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.teleop.retargeting.adaptive import AdaptiveHandRetargeter
from dexmani_real.teleop.retargeting.landmarks import HandCalibration, prepare_landmarks
from dexmani_real.teleop.retargeting.objective import HandObjective, huber


def model_landmarks(retargeter, qpos):
    """用真实模型生成定骨长的 21 点夹具，DIP 为末段内部点。"""
    matrices = retargeter.fk.forward(qpos, retargeter.frame_ids)
    points = matrices[:, :3, 3].reshape(5, 3, 3)
    human = np.zeros((21, 3))
    human[1] = retargeter.thumb_cmc
    human[2:5] = points[0]
    for finger in range(1, 5):
        offset = 1 + 4 * finger
        human[offset:offset + 2] = points[finger, :2]
        human[offset + 2] = (points[finger, 1] + points[finger, 2]) / 2
        human[offset + 3] = points[finger, 2]
    return human


def make_retargeter(**overrides):
    config = replace(HandRetargetingConfig(), max_solve_time_s=0.5, **overrides)
    hand = HandConfig()
    retargeter = AdaptiveHandRetargeter(config, hand, control_hz=16)
    home = np.deg2rad(hand.home_qpos_deg)
    landmarks = model_landmarks(retargeter, home)
    retargeter.calibrate([landmarks.copy() for _ in range(config.calibration_samples)])
    retargeter.reset(home, timestamp_ns=1_000_000_000)
    return retargeter, home, landmarks


def finite_difference(function, point, step=1e-6):
    result = np.empty_like(point)
    for index in range(len(point)):
        delta = np.zeros_like(point)
        delta[index] = step
        result[index] = (function(point + delta) - function(point - delta)) / (2 * step)
    return result


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retargeter, cls.home, cls.landmarks = make_retargeter()

    def test_palm_frame_is_rigid_transform_invariant(self):
        rotation = Rotation.from_rotvec([0.7, -1.2, 0.4]).as_matrix()
        transformed = self.landmarks @ rotation.T + [2, -3, 1]
        assert_allclose(prepare_landmarks(transformed), prepare_landmarks(self.landmarks), atol=1e-14)

    def test_input_is_not_mutated(self):
        points = self.landmarks.copy()
        before = points.copy()
        canonical = prepare_landmarks(points)
        r = self.retargeter
        r.calibration.targets(canonical, r.robot_points, r.thumb_cmc, r.config)
        assert_array_equal(points, before)

    def test_calibration_and_targets_are_body_scale_invariant(self):
        r = self.retargeter
        canonical = prepare_landmarks(self.landmarks)
        expected = r.calibration.targets(canonical, r.robot_points, r.thumb_cmc, r.config)
        for scale in (0.7, 1.3):
            scaled = self.landmarks * scale
            calibration = HandCalibration.from_frames(
                [scaled] * 25, min_samples=25, max_relative_mad=0.1,
            )
            actual = calibration.targets(
                prepare_landmarks(scaled), r.robot_points, r.thumb_cmc, r.config,
            )
            for a, e in zip(actual, expected):
                assert_allclose(a, e, atol=1e-14)

    def test_calibration_uses_bone_lengths_under_flexion(self):
        r = self.retargeter
        pose = self.home.copy()
        pose[[4, 5, 6, 7, 8, 9, 10, 11]] = 1.2
        bent = model_landmarks(r, pose)
        calibration = HandCalibration.from_frames(
            [bent] * 25, min_samples=25, max_relative_mad=0.1,
        )
        assert_allclose(calibration.bone_lengths, r.calibration.bone_lengths, atol=1e-14)
        r.calibration.validate_frame(prepare_landmarks(bent), 0.35)

    def test_bad_units_nan_and_degenerate_geometry_are_rejected(self):
        for points in (
            self.landmarks * 1000, np.zeros((21, 3)), np.full((21, 3), np.nan),
            np.zeros((20, 3)),
        ):
            with self.subTest(shape=points.shape):
                with self.assertRaises(ValueError):
                    prepare_landmarks(points)

    def test_calibration_rejects_insufficient_or_unstable_samples(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            HandCalibration.from_frames([self.landmarks], min_samples=25, max_relative_mad=0.1)
        frames = [self.landmarks * scale for scale in np.linspace(0.65, 1.35, 25)]
        with self.assertRaisesRegex(ValueError, "unstable"):
            HandCalibration.from_frames(frames, min_samples=25, max_relative_mad=0.1)

    def test_changed_bone_length_rejected_against_frozen_calibration(self):
        points = prepare_landmarks(self.landmarks)
        points[8] += [0.05, 0, 0]
        with self.assertRaisesRegex(ValueError, "changed"):
            self.retargeter.calibration.validate_frame(points, 0.35)

    def test_pinch_targets_and_activation_are_continuous_at_both_thresholds(self):
        r = self.retargeter
        canonical = prepare_landmarks(self.landmarks)
        scale = np.linalg.norm(r.robot_points[1, 0] - r.robot_points[4, 0]) / r.calibration.palm_width

        def features(distance):
            points = canonical.copy()
            points[8] = points[4] + np.array([distance / scale, 0, 0])
            _, _, activation, targets = r.calibration.targets(
                points, r.robot_points, r.thumb_cmc, r.config,
            )
            return np.array([activation[0], targets[0]])

        for boundary in (r.config.pinch_full_m, r.config.pinch_start_m):
            epsilon = 1e-7
            center = features(boundary)
            before, after = features(boundary - epsilon), features(boundary + epsilon)
            assert_allclose(before, after, atol=3e-7)
            assert_allclose((center - before) / epsilon, (after - center) / epsilon, atol=1e-3)
        self.assertTrue(np.isfinite(features(0)).all())

    def test_saved_calibration_round_trip_and_validation(self):
        metadata = self.retargeter.calibration_metadata()
        restored = HandCalibration.from_metadata(metadata)
        assert_array_equal(restored.bone_lengths, self.retargeter.calibration.bone_lengths)
        for changes in (
            {"method": "unknown"}, {"human_bone_lengths_m": [[0.03]]},
            {"human_palm_width_m": float("nan")}, {"sample_count": 0},
        ):
            with self.assertRaises(ValueError):
                HandCalibration.from_metadata(metadata | changes)


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retargeter, cls.home, cls.landmarks = make_retargeter()

    def test_sdk_order_frame_jacobians_match_real_urdf_finite_differences(self):
        r = self.retargeter
        rng = np.random.default_rng(824)
        for _ in range(5):
            q = rng.uniform(r.limits[:, 0], r.limits[:, 1])
            _, jac = r.fk.positions_and_jacobians(q, r.frame_ids)
            for joint in range(12):
                delta = np.eye(12)[joint] * 1e-6
                plus = r.fk.forward(q + delta, r.frame_ids)[:, :3, 3]
                minus = r.fk.forward(q - delta, r.frame_ids)[:, :3, 3]
                assert_allclose(jac[:, :, joint], (plus - minus) / 2e-6, atol=3e-9)

    def test_full_objective_gradient_including_temporal_and_pinch_terms(self):
        r, rng = self.retargeter, np.random.default_rng(912)
        for index in range(20):
            goal = rng.uniform(r.limits[:, 0], r.limits[:, 1])
            q = rng.uniform(r.limits[:, 0], r.limits[:, 1])
            points = r.fk.forward(goal, r.frame_ids)[:, :3, 3].reshape(5, 3, 3)
            terminal = points[:, 2] - points[:, 1]
            direction = terminal / np.linalg.norm(terminal, axis=1)[:, None]
            activation = np.array([0, 0.2, 0.7, 1])
            distances = np.array([0.08, 0.04, 0.025, 0.012])
            objective = HandObjective(
                r.fk, r.frame_ids, (points, direction, activation, distances),
                self.home, (1 / 16, 1 / 30, 1 / 60)[index % 3], r.config,
            )
            analytic = np.empty(12)
            objective(q, analytic)
            numeric = finite_difference(lambda x: objective(x, np.empty(0)), q)
            assert_allclose(analytic, numeric, atol=3e-7, rtol=3e-6)

    def test_huber_zero_is_finite_and_rotationally_invariant(self):
        e = np.array([[0, 0, 0], [0.2, 0.1, 0.4]])
        rotation = Rotation.from_rotvec([0.5, 0.7, -0.3]).as_matrix()
        value, derivative = huber(e, 0.2)
        rotated_value, rotated_derivative = huber(e @ rotation.T, 0.2)
        assert_allclose(value, rotated_value, atol=1e-15)
        assert_allclose(derivative @ rotation.T, rotated_derivative, atol=1e-15)
        self.assertTrue(np.isfinite(derivative).all())

    def test_reachable_goal_converges_with_actual_nlopt(self):
        r = self.retargeter
        goal = self.home + np.array([0.2, -0.1, 0.2, 0.03, 0.5, 0.4, 0.5, 0.4, 0.4, 0.3, 0.3, 0.5])
        points = r.fk.forward(goal, r.frame_ids)[:, :3, 3].reshape(5, 3, 3)
        terminal = points[:, 2] - points[:, 1]
        direction = terminal / np.linalg.norm(terminal, axis=1)[:, None]
        objective = HandObjective(
            r.fk, r.frame_ids, (points, direction, np.zeros(4), np.ones(4)),
            self.home, 1 / 16, replace(r.config, damping_s=0),
        )
        opt = nlopt.opt(nlopt.LD_SLSQP, 12)
        opt.set_lower_bounds(r.limits[:, 0])
        opt.set_upper_bounds(r.limits[:, 1])
        opt.set_maxeval(200)
        opt.set_ftol_abs(1e-12)
        opt.set_min_objective(objective)
        solved = opt.optimize(self.home)
        actual = r.fk.forward(solved, r.frame_ids)[:, :3, 3].reshape(5, 3, 3)
        self.assertLess(np.max(np.linalg.norm(actual[:, 2] - points[:, 2], axis=1)), 1e-5)


class SolverStateTests(unittest.TestCase):
    def test_proposal_does_not_advance_state_and_commit_uses_submitted_command(self):
        r, home, points = make_retargeter()
        timestamp = 1_062_500_000
        first = r.retarget(points, timestamp_ns=timestamp)
        self.assertIsNotNone(first, r.rejection_reason)
        assert_array_equal(r.previous_command, home)
        second = r.retarget(points, timestamp_ns=timestamp)
        assert_allclose(first, second, atol=1e-12)
        submitted = (first + home) / 2
        r.commit(submitted, timestamp_ns=timestamp)
        assert_array_equal(r.previous_command, submitted)
        self.assertEqual(r.command_timestamp_ns, timestamp)

    def test_position_and_velocity_bounds_over_a_sequence(self):
        r, _, _ = make_retargeter()
        rng = np.random.default_rng(731)
        stamp = 1_000_000_000
        for dt in (1 / 16, 1 / 30, 1 / 60) * 8:
            pose = rng.uniform(r.limits[:, 0], r.limits[:, 1])
            # index MCP abduction slightly changes palm span; still within calibration tolerance.
            points = model_landmarks(r, pose)
            previous = r.previous_command.copy()
            stamp += round(dt * 1e9)
            result = r.retarget(points, timestamp_ns=stamp)
            self.assertIsNotNone(result, r.rejection_reason)
            self.assertTrue(np.all(result >= r.limits[:, 0]))
            self.assertTrue(np.all(result <= r.limits[:, 1]))
            self.assertLessEqual(np.max(np.abs(result - previous)), 4 * dt + 1e-8)
            r.commit(result, timestamp_ns=stamp)

    def test_invalid_geometry_does_not_become_a_held_success(self):
        r, home, _ = make_retargeter()
        self.assertIsNone(r.retarget(np.zeros((21, 3)), timestamp_ns=1_062_500_000))
        self.assertIn("bone", r.rejection_reason)
        assert_array_equal(r.previous_command, home)

    def test_filter_does_not_reduce_the_requested_speed_limit_at_higher_rates(self):
        for hz in (16, 30, 60):
            with self.subTest(hz=hz):
                r, home, _ = make_retargeter(max_joint_speed_rad_s=0.1, damping_s=0)
                goal = home.copy()
                goal[6] += 1.0
                dt_ns = round(1e9 / hz)
                result = r.retarget(
                    model_landmarks(r, goal), timestamp_ns=1_000_000_000 + dt_ns,
                )
                self.assertIsNotNone(result, r.rejection_reason)
                speed = np.max(np.abs(result - home)) / (dt_ns * 1e-9)
                self.assertAlmostEqual(speed, 0.1, places=6)

    def test_negative_status_nan_out_of_bounds_and_worse_solution_are_rejected(self):
        for mode in ("status", "nan", "bounds", "worse"):
            with self.subTest(mode=mode):
                r, home, points = make_retargeter()
                optimizer = Mock()
                optimizer.last_optimize_result.return_value = -1 if mode == "status" else 3
                optimizer.get_numevals.return_value = 1
                if mode == "nan":
                    candidate = np.full(12, np.nan)
                elif mode == "bounds":
                    candidate = np.full(12, 50.0)
                else:
                    # 弯折的极端候选在放宽步长后仍不优于当前近似匹配姿态。
                    candidate = r.limits[:, 1].copy()
                optimizer.optimize.return_value = candidate
                r.optimizer = optimizer
                r.config = replace(r.config, max_joint_speed_rad_s=100)
                self.assertIsNone(r.retarget(points, timestamp_ns=1_062_500_000))
                self.assertIsNotNone(r.rejection_reason)
                assert_array_equal(r.previous_command, home)

    def test_roundoff_failure_is_visible_and_does_not_commit(self):
        r, home, points = make_retargeter()
        optimizer = Mock()
        optimizer.optimize.side_effect = nlopt.RoundoffLimited("test roundoff")
        optimizer.last_optimize_result.return_value = -4
        r.optimizer = optimizer
        self.assertIsNone(r.retarget(points, timestamp_ns=1_062_500_000))
        self.assertIn("RoundoffLimited", r.rejection_reason)
        assert_array_equal(r.previous_command, home)

    def test_budget_stop_is_reported_but_bounded_nonworse_candidate_is_usable(self):
        r, home, points = make_retargeter()
        optimizer = Mock()
        optimizer.optimize.return_value = home.copy()
        optimizer.last_optimize_result.return_value = nlopt.MAXEVAL_REACHED
        optimizer.get_numevals.return_value = 2
        r.optimizer = optimizer
        self.assertIsNotNone(r.retarget(points, timestamp_ns=1_062_500_000))
        self.assertTrue(r.last_diagnostics["budget_limited"])

    def test_nonmonotonic_time_and_invalid_commit_are_rejected(self):
        r, _, points = make_retargeter()
        with self.assertRaisesRegex(ValueError, "backwards"):
            r.retarget(points, timestamp_ns=999_999_999)
        with self.assertRaises(ValueError):
            r.commit(np.full(12, 50), timestamp_ns=1_062_500_000)

    def test_reset_preserves_morphology_and_clears_previous_episode_state(self):
        r, home, points = make_retargeter()
        calibration = r.calibration
        result = r.retarget(points, timestamp_ns=1_062_500_000)
        r.commit(result, timestamp_ns=1_062_500_000)
        r.reset(home, timestamp_ns=2_000_000_000)
        self.assertIs(r.calibration, calibration)
        assert_array_equal(r.previous_command, home)
        self.assertEqual(r.last_diagnostics, {})


class ConfigTests(unittest.TestCase):
    def test_invalid_parameters_and_obsolete_yaml_section_are_rejected(self):
        for override in (
            dict(pinch_start_m=0.02), dict(max_evaluations=0), dict(smoothing_tau_s=-1),
            dict(position_scale_m=0), dict(damping_s=np.nan), dict(tip_weight=0),
        ):
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    replace(HandRetargetingConfig(), **override).validate()
        with self.assertRaises(TypeError):
            load_config(overrides={"dexpilot_retargeting": {"low_pass_alpha": 0.5}})
        config = load_config(overrides={"hand_retargeting": {"smoothing_tau_s": 0.05}})
        self.assertEqual(config.hand_retargeting.smoothing_tau_s, 0.05)


if __name__ == "__main__":
    unittest.main()

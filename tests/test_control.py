import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.control import homing
from dexmani_real.control.action_mapping import ActionIntent, ActionMapper
from dexmani_real.control.collision import CollisionModel
from dexmani_real.control.config import (
    EnvironmentConfig,
    MotionConfig,
    StaticCollisionBox,
)
from dexmani_real.control.constraints import (
    joint_target_rejection,
    wrap_nearest_equivalent,
)
from dexmani_real.control.ik import IKSolver
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.model import XARM7_EQUIVALENT_JOINT_MASK
from dexmani_real.robot.xarm7.config import ArmConfig, HomingConfig
from dexmani_real.robot.xhand.config import HandConfig


class JointTests(unittest.TestCase):
    def test_equivalent_axes_respect_narrow_operational_limits(self):
        home = np.asarray(ArmConfig().home_qpos)
        mask = np.asarray(XARM7_EQUIVALENT_JOINT_MASK)
        target = home + 2 * np.pi
        result = wrap_nearest_equivalent(target, home, home - 0.1, home + 0.1)
        np.testing.assert_allclose(result[mask], home[mask], atol=1e-14)
        np.testing.assert_array_equal(result[~mask], target[~mask])
        np.testing.assert_array_equal(target, home + 2 * np.pi)

    def test_unavailable_equivalent_branch_preserves_target(self):
        target = np.full(7, np.pi)
        result = wrap_nearest_equivalent(target, np.zeros(7), [-0.1] * 7, [0.1] * 7)
        np.testing.assert_array_equal(result, target)

    def test_jump_is_absolute_and_invalid_output_is_rejected(self):
        limits = np.array([[-2 * np.pi, 2 * np.pi]] * 7)
        current = np.zeros(7)
        target = current.copy()
        target[0] = 2 * np.pi
        self.assertEqual(joint_target_rejection(target, current, current, limits), "jump")
        target[0] = np.nan
        self.assertEqual(joint_target_rejection(target, current, current, limits), "invalid_target")


class IKSearchTests(unittest.TestCase):
    def setUp(self):
        self.solver = IKSolver.__new__(IKSolver)
        self.solver.limits = np.array([[-1.0, 1.0]] * 7)
        self.solver.joint_ranges = np.full(7, 2.0)
        self.solver.limit_center = np.zeros(7)
        self.solver.limit_radius = np.ones(7)
        self.solver.position_tolerance = 0.02
        self.solver.rotation_tolerance = 0.1
        self.solver.eef_link_id = 7
        self.solver.model = SimpleNamespace(compute_IK_CLIK=Mock())
        self.solver.constraints = SimpleNamespace(
            joint_rejection=lambda q, current, previous: joint_target_rejection(
                q, current, previous, self.solver.limits
            ),
            endpoint_rejection=Mock(return_value=None),
            collision_model=SimpleNamespace(set_hand_qpos=Mock()),
        )
        self.target = Pose.identity()
        self.solver.compute_eef_pose_base = Mock(return_value=self.target)
        self.hand = np.zeros(12)

    def solve(self, current=None, previous=None):
        current = np.zeros(7) if current is None else current
        previous = current if previous is None else previous
        return self.solver.solve(self.target, current, previous, hand_qpos=self.hand)

    def test_continuation_accepts_once_and_receives_explicit_hand_pose(self):
        self.solver.model.compute_IK_CLIK.return_value = (np.zeros(7), True, None)
        result = self.solve()
        self.assertTrue(result.success)
        self.solver.model.compute_IK_CLIK.assert_called_once()
        self.solver.constraints.endpoint_rejection.assert_called_once()
        self.solver.constraints.collision_model.set_hand_qpos.assert_called_once_with(self.hand)

    def test_same_seed_is_not_retried(self):
        self.solver.model.compute_IK_CLIK.return_value = (np.zeros(7), False, None)
        result = self.solve()
        self.assertFalse(result.success)
        self.assertEqual(result.reason, "solver_no_convergence")
        self.solver.model.compute_IK_CLIK.assert_called_once()
        self.solver.constraints.endpoint_rejection.assert_not_called()

    def test_duplicate_candidates_are_checked_once(self):
        current = np.full(7, 0.9)
        self.solver.model.compute_IK_CLIK.return_value = (current, True, None)
        self.solver.nullspace_direction = Mock(return_value=None)
        self.assertTrue(self.solve(current, current - 0.01).success)
        self.assertEqual(self.solver.model.compute_IK_CLIK.call_count, 2)
        self.solver.constraints.endpoint_rejection.assert_called_once()

    def test_collision_anchor_can_find_a_valid_nullspace_solution(self):
        current = np.full(7, 0.85)
        candidates = [np.full(7, value) for value in (0.90, 0.91, 0.92, 0.93)]
        self.solver.model.compute_IK_CLIK.side_effect = [(q, True, None) for q in candidates]
        self.solver.constraints.endpoint_rejection.side_effect = [
            "collision",
            "collision",
            None,
            None,
        ]
        self.solver.compute_eef_jacobian_base = Mock(return_value=np.eye(6, 7))
        with patch.object(np.linalg, "svd", wraps=np.linalg.svd) as svd:
            result = self.solve(current, current + 0.01)
        self.assertTrue(result.success)
        np.testing.assert_array_equal(result.qpos, candidates[2])
        self.assertEqual(self.solver.model.compute_IK_CLIK.call_count, 4)
        self.assertEqual(self.solver.constraints.endpoint_rejection.call_count, 4)
        svd.assert_called_once()

    def test_backend_errors_and_malformed_outputs_propagate(self):
        failure = RuntimeError("CLIK backend failed")
        self.solver.model.compute_IK_CLIK.side_effect = failure
        with self.assertRaises(RuntimeError) as raised:
            self.solve()
        self.assertIs(raised.exception, failure)
        self.solver.model.compute_IK_CLIK.side_effect = None
        for raw in (None, np.zeros(6), np.full(7, np.nan)):
            with self.subTest(raw=raw):
                self.solver.model.compute_IK_CLIK.return_value = (raw, True, None)
                with self.assertRaises(ValueError):
                    self.solve()

    def test_pose_error_and_collision_are_normal_rejections(self):
        self.solver.model.compute_IK_CLIK.return_value = (np.zeros(7), True, None)
        self.solver.compute_eef_pose_base.return_value = Pose([1, 0, 0], [1, 0, 0, 0])
        self.assertIn("pose_error", self.solve().reason)
        self.solver.compute_eef_pose_base.return_value = self.target
        self.solver.constraints.endpoint_rejection.return_value = "self_collision"
        self.solver.nullspace_direction = Mock(return_value=None)
        self.assertIn("self_collision", self.solve().reason)

    def test_rank_guard_and_nullspace_step_bounds(self):
        self.solver.compute_eef_jacobian_base = Mock(return_value=np.eye(6, 7))
        direction = self.solver.nullspace_direction(np.zeros(7))
        anchor = np.full(7, 0.99)
        seeds = self.solver.nullspace_seeds(anchor, direction)
        self.assertEqual(len(seeds), 2)
        for seed in seeds:
            self.assertTrue(np.all(np.abs(seed) <= 1))
            self.assertLessEqual(np.max(np.abs(seed - anchor)), np.deg2rad(5) + 1e-12)
        self.solver.compute_eef_jacobian_base.return_value[0] = 0
        self.assertIsNone(self.solver.nullspace_direction(np.zeros(7)))


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm, cls.hand = ArmConfig(), HandConfig()
        cls.mapper = ActionMapper.for_mode(
            "eef",
            arm=cls.arm,
            hand=cls.hand,
            motion=MotionConfig(),
            environment=EnvironmentConfig(),
        )

    def test_real_kinematics_agree_and_continuous_targets_solve(self):
        rng = np.random.default_rng(3)
        current = np.asarray(self.arm.home_qpos)
        solver = self.mapper.solver
        for _ in range(12):
            target_q = current + rng.uniform(-0.025, 0.025, 7)
            target = solver.compute_eef_pose_base(target_q)
            position, _ = self.mapper.constraints.target_fk.compute(target_q)
            np.testing.assert_allclose(target.p, position, atol=1e-8)
            result = solver.solve(
                target, current, current, hand_qpos=np.deg2rad(self.hand.home_qpos_deg)
            )
            self.assertTrue(result.success, result.reason)
            current = result.qpos

    def test_unified_scene_preserves_obstacles_table_and_model_limits(self):
        q = np.asarray(self.arm.home_qpos)
        hand = np.deg2rad(self.hand.home_qpos_deg)
        collision = CollisionModel(
            static_boxes=(StaticCollisionBox(name="box", size_xyz_m=(4, 4, 4)),)
        )
        collision.set_hand_qpos(hand)
        self.assertIn("environment_collision", collision.rejection(q))
        self.assertIn("box", collision.rejection(q))
        table = CollisionModel(table=EnvironmentConfig().table, table_plane_abcd=(0, 0, 1, 1))
        table.set_hand_qpos(hand)
        self.assertIsNone(table.rejection(q))
        self.assertGreater(table.minimum_table_distance(q), 0.5)
        hand[0] = 10
        table.set_hand_qpos(hand)
        self.assertEqual(table.rejection(q), "model_joint_limits")
        with self.assertRaises(ValueError):
            table.rejection(np.full(7, np.nan))

    def test_policy_commands_roundtrip_joint_and_eef_control_layouts(self):
        current = np.asarray(self.arm.home_qpos)
        hand = np.deg2rad(self.hand.home_qpos_deg)
        for mode, target in (
            ("joint", current + 0.01),
            ("eef", self.mapper.solver.compute_eef_pose_base(current + 0.01)),
        ):
            with self.subTest(mode=mode):
                result = self.mapper.map(ActionIntent(mode, target, hand), current)
                self.assertIsNone(result.rejection_reason)
                command = RobotCommand(1, result.arm_qpos, result.hand_qpos)
                control = self.mapper.control_from_command(command, mode)
                self.assertEqual(control.shape, (19,) if mode == "joint" else (21,))
                np.testing.assert_array_equal(control[-12:], command.hand_qpos)
                self.assertIsNone(self.mapper.frozen_rejection(command, current, current, mode))
                if mode == "joint":
                    np.testing.assert_array_equal(control[:7], command.arm_qpos)
                else:
                    position, rotation = self.mapper.constraints.target_fk.compute(command.arm_qpos)
                    np.testing.assert_array_equal(control[:9], np.r_[position, rotation])

    def test_real_home_uses_narrow_limits_and_custom_convergence(self):
        target = np.asarray(self.arm.home_qpos)
        arm = replace(
            self.arm,
            joint_limit_lower=tuple(target - 0.02),
            joint_limit_upper=tuple(target + 0.02),
            homing=HomingConfig(convergence_rad=0.0001),
        )
        arm.validate()
        environment = EnvironmentConfig()
        environment = replace(environment, table=replace(environment.table, enabled=False))
        planner = homing.HomePlanner(environment, arm, table_plane_abcd=None)
        planner.set_hand_qpos(np.deg2rad(self.hand.home_qpos_deg))
        current = target + 0.001
        np.testing.assert_array_equal(planner.plan(current, target), [current, target])


class HomeGeometryTests(unittest.TestCase):
    def setUp(self):
        self.path = np.array([np.full(7, value) for value in (0, 0.01, 0.02)])
        self.planner = SimpleNamespace(
            arm=ArmConfig(),
            limits=np.array([[-1.0, 1.0]] * 7),
            workspace_bounds=np.array([[-1.0, 1.0]] * 3),
            arm_fk=SimpleNamespace(compute=Mock(return_value=(np.zeros(3), np.zeros(6)))),
            collision_model=SimpleNamespace(
                rejection=Mock(return_value=None),
                table=SimpleNamespace(soft_clearance_m=0.02),
                minimum_table_distance=Mock(side_effect=[0.01, 0.015, 0.03]),
            ),
        )

    def check_path(self, *, escape=True):
        homing.check_home_path(
            self.path,
            self.planner,
            allow_table_soft_escape=escape,
        )

    def test_soft_escape_requires_monotonic_clearance_and_no_mesh_collision(self):
        self.check_path()
        for distances in ([0.01, 0, 0.03], [0.01, 0.015, 0.019], [0.03, 0.01, 0.03]):
            with self.subTest(distances=distances):
                self.planner.collision_model.minimum_table_distance.side_effect = distances
                with self.assertRaises(homing.HomePathRejected):
                    self.check_path()
        self.planner.collision_model.rejection.return_value = "self_collision"
        with self.assertRaisesRegex(homing.HomePathRejected, "collision"):
            self.check_path()

    def test_workspace_and_backend_failures_are_distinct(self):
        self.planner.arm_fk.compute.return_value = (np.ones(3) * 2, None)
        with self.assertRaisesRegex(homing.HomePathRejected, "workspace"):
            self.check_path()
        self.planner.collision_model.rejection.side_effect = RuntimeError("geometry failure")
        with self.assertRaisesRegex(RuntimeError, "geometry failure"):
            self.check_path()

    def test_home_candidates_and_canonical_branch_alignment(self):
        planner = homing.HomePlanner.__new__(homing.HomePlanner)
        planner.arm = ArmConfig()
        planner.canonicalize_qpos = Mock(side_effect=lambda q, ref: q)
        current, target = np.zeros(7), np.full(7, 0.2)
        with patch.object(
            homing,
            "check_home_path",
            side_effect=[
                homing.HomePathRejected("direct"),
                homing.HomePathRejected("proximal"),
                None,
            ],
        ):
            path = planner.plan(current, target)
        np.testing.assert_array_equal(path[1, :4], current[:4])
        np.testing.assert_array_equal(path[1, 4:], target[4:])
        wrapped = target.copy()
        wrapped[0] += 2 * np.pi
        planner.canonicalize_qpos.return_value = wrapped
        planner.canonicalize_qpos.side_effect = None
        with (
            patch.object(
                homing,
                "compute_joint_home_path",
                side_effect=[
                    homing.HomePathRejected("canonical"),
                    np.stack([current, wrapped]),
                ],
            ),
            patch.object(homing, "check_home_path") as check,
        ):
            path = planner.plan(current, target)
        np.testing.assert_array_equal(path[-1], target)
        self.assertFalse(check.call_args.kwargs["allow_table_soft_escape"])

    def test_technical_failure_does_not_try_another_home_candidate(self):
        planner = homing.HomePlanner.__new__(homing.HomePlanner)
        planner.arm = ArmConfig()
        with patch.object(homing, "check_home_path", side_effect=RuntimeError("backend")) as check:
            with self.assertRaisesRegex(RuntimeError, "backend"):
                planner.plan(np.zeros(7), np.ones(7) * 0.2)
        check.assert_called_once()

    def test_planning_uses_execution_tolerance_and_operational_limits(self):
        planner = homing.HomePlanner.__new__(homing.HomePlanner)
        planner.arm = replace(ArmConfig(), homing=HomingConfig(convergence_rad=0.0001))
        planner.limits = np.array([[-0.5, 0.5]] * 7)
        current, target = np.zeros(7), np.full(7, 0.001)
        with patch.object(homing, "check_home_path") as check:
            path = planner.plan(current, target)
            np.testing.assert_array_equal(path, [current, target])
            self.assertEqual(planner.plan(target, target).shape, (0, 7))
        self.assertEqual(check.call_count, 2)
        wrapped = target.copy()
        wrapped[0] += 2 * np.pi
        np.testing.assert_allclose(planner.canonicalize_qpos(wrapped, wrapped), target)
        self.planner.limits[0, 1] = 0.015
        with self.assertRaisesRegex(homing.HomePathRejected, "operational joint limits"):
            self.check_path()
        self.planner.collision_model.rejection.assert_not_called()

    def test_empty_home_still_checks_geometry_and_invalid_input_is_not_retried(self):
        planner = homing.HomePlanner.__new__(homing.HomePlanner)
        planner.arm = ArmConfig()
        planner.limits = np.array([[-1.0, 1.0]] * 7)
        with patch.object(
            homing, "check_home_path", side_effect=homing.HomePathRejected("collision")
        ) as check:
            with self.assertRaisesRegex(homing.HomePathRejected, "collision"):
                planner.plan(np.zeros(7), np.zeros(7))
            check.assert_called_once()
        with self.assertRaisesRegex(ValueError, "finite shape"):
            planner.plan(np.full(7, np.nan), np.zeros(7))


if __name__ == "__main__":
    unittest.main()

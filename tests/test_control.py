"""控制候选、等价分支和 HOME 准入；替身只验证软件边界。"""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from dexmani_real.config import ExperimentConfig
from dexmani_real.control.action_mapping import ActionIntent, ActionMapper
from dexmani_real.control.collision import CollisionModel
from dexmani_real.control.config import StaticCollisionBox
from dexmani_real.control.homing import (
    HomePlanner,
    check_home_path,
    compute_band_alignment_path,
    compute_joint_home_path,
)
from dexmani_real.control.joint_angles import (
    equivalent_joint_delta,
    joint_target_rejection,
    wrap_nearest_equivalent,
)
from dexmani_real.control.online_ik import (
    IKFailureKind,
    IKResult,
    OnlineIKConfig,
    make_online_ik_solver,
)
from dexmani_real.control.target_validation import TargetChecker
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot.commands import RobotCommand


class ControlTests(unittest.TestCase):
    def mapper(self, solver=None):
        collision = Mock()
        collision.check_self_collision.return_value = False
        collision.within_joint_limits.return_value = True
        collision.check_environment_collision_details.return_value = SimpleNamespace(in_collision=False)
        checker = TargetChecker(
            collision, joint_lower_rad=np.full(7, -4 * np.pi),
            joint_upper_rad=np.full(7, 4 * np.pi), workspace=[[-1, 1]] * 3,
            workspace_margin_m=0, target_fk=SimpleNamespace(compute=lambda q: (np.zeros(3), None)),
        )
        return ActionMapper(checker, hand_enabled=True, hand_lower_rad=np.zeros(12),
                            hand_upper_rad=np.ones(12), home_hand_qpos=np.zeros(12), solver=solver)

    def test_equivalent_axes_and_absolute_jump_are_distinct(self):
        q = np.full(7, 2 * np.pi)
        wrapped = wrap_nearest_equivalent(q, np.zeros(7), np.full(7, -4*np.pi), np.full(7, 4*np.pi))
        self.assertEqual(wrapped[0], 0)
        self.assertEqual(wrapped[1], 2*np.pi)
        self.assertEqual(equivalent_joint_delta(q, np.zeros(7))[0], 0)
        self.assertEqual(joint_target_rejection(q, np.zeros(7), np.zeros(7),
                         np.tile([-4*np.pi, 4*np.pi], (7, 1)), np.ones(7)), 'jump')
        np.testing.assert_array_equal(q, np.full(7, 2*np.pi))
        # 无合法等价分支时保留原值，由下游明确拒绝。
        out = wrap_nearest_equivalent(q, np.zeros(7), np.full(7, 1.), np.full(7, 2.))
        self.assertEqual(out[0], q[0])
        for invalid in (np.zeros(6), np.full(7, np.nan)):
            with self.assertRaises(ValueError):
                wrap_nearest_equivalent(invalid, np.zeros(7), np.zeros(7), np.ones(7))

    def test_joint_mapping_limits_and_rejected_candidate(self):
        mapper = self.mapper()
        intent = ActionIntent('joint', np.full(7, .01), np.full(12, 2.))
        result = mapper.map(intent, np.zeros(7))
        np.testing.assert_array_equal(result.hand_qpos, np.ones(12))
        np.testing.assert_array_equal(result.arm_qpos, intent.arm)
        mapper.target_checker.collision_model.check_self_collision.return_value = True
        candidate, _, reason = mapper.prepare_joint_target(intent.arm, intent.hand, np.zeros(7))
        self.assertIsNotNone(candidate)
        self.assertEqual(reason, 'self_collision')
        self.assertIsNone(mapper.map(intent, np.zeros(7)).arm_qpos)

    def test_eef_projection_preserves_source_and_failure_category(self):
        for kind in IKFailureKind:
            solver = Mock()
            solver.solve.return_value = IKResult(success=False, qpos=None, reason='rejected', failure_kind=kind)
            mapper = self.mapper(solver)
            target = Pose([2, 0, 0], [1, 0, 0, 0])
            result = mapper.map(ActionIntent('eef', target, np.zeros(12)), np.zeros(7))
            self.assertIsNone(result.arm_qpos)
            self.assertEqual(result.ik_result.failure_kind, kind)
            self.assertEqual(target.p[0], 2)
            self.assertEqual(result.eef_pose.p[0], 1)

    def test_frozen_command_cannot_switch_equivalent_branch(self):
        mapper = self.mapper()
        q = np.zeros(7)
        q[0] = 2*np.pi
        command = RobotCommand(3, arm_qpos=q, hand_qpos=np.zeros(12))
        self.assertEqual(mapper.frozen_rejection(command, np.zeros(7), None, 'joint'),
                         'frozen_equivalent_branch_changed')
        self.assertEqual(command.arm_qpos[0], 2*np.pi)
        self.assertFalse(command.arm_qpos.flags.writeable)

    def test_real_ik_and_home_scenes_remain_separate(self):
        cfg = ExperimentConfig()
        solver = make_online_ik_solver(OnlineIKConfig())
        home = HomePlanner(cfg.environment, table_plane_abcd=(0, 0, 1, 0))
        self.assertFalse(solver.collision_model.has_table)
        self.assertTrue(home.collision_model.has_table)
        q = np.asarray(cfg.arm.home_qpos)
        solver.collision_model.set_hand_qpos(np.deg2rad(cfg.hand.home_qpos_deg))
        pose = solver.kin.compute_eef_pose_base(q)
        result = solver.solve(pose, q, q, endpoint_rejection=lambda candidate: None)
        self.assertTrue(result.success, result.reason)
        np.testing.assert_allclose(result.qpos, q, atol=1e-12, rtol=0)

    def test_collision_diagnostics_keep_robot_and_obstacle_identity(self):
        cfg = ExperimentConfig()
        model = CollisionModel(static_boxes=(
            StaticCollisionBox(name='test_obstacle', center_xyz_m=(0, 0, .5), size_xyz_m=(2, 2, 2)),
        ))
        model.set_hand_qpos(np.deg2rad(cfg.hand.home_qpos_deg))
        q = np.asarray(cfg.arm.home_qpos)
        info = model.check_environment_collision_details(q)
        self.assertTrue(info.in_collision)
        self.assertGreater(info.num_contacts, 0)
        self.assertEqual(len(info.sample_qpos_rad), 19)
        link_names = {frame.name for frame in model.model.frames}
        for contact in info.collision_pairs:
            self.assertIn(contact.link_name1, link_names)
            self.assertEqual(contact.link_name2, 'test_obstacle')
            self.assertEqual(contact.object_name2, 'test_obstacle')
            self.assertEqual(contact.collision_type, 'environment')
        self.assertTrue(info.collision_pairs)
        with (
            patch.object(model, 'get_geom_link_name', side_effect=RuntimeError('details unavailable')),
            self.assertLogs('dexmani_real.control.collision', level='WARNING'),
        ):
            unavailable = model.check_environment_collision_details(q)
        self.assertTrue(unavailable.in_collision)
        self.assertEqual(unavailable.num_contacts, 1)
        self.assertEqual(unavailable.collision_pairs, ())


class HomePathTests(unittest.TestCase):
    options = dict(table_z_surface_m=0., hand_safety_margin_m=0.)

    def test_candidate_order_direct_proximal_distal(self):
        planner = SimpleNamespace(canonicalize_qpos=lambda target, current: target.copy())
        start, end = np.zeros(7), np.full(7, .1)
        with patch('dexmani_real.control.homing.check_home_path', side_effect=[ValueError('direct'), ValueError('proximal'), None]) as check:
            path = compute_joint_home_path(start, end, planner, **self.options)
        self.assertEqual(check.call_count, 3)
        np.testing.assert_array_equal(check.call_args_list[0].args[0], [start, end])
        np.testing.assert_array_equal(check.call_args_list[1].args[0][1], [.1, .1, .1, .1, 0, 0, 0])
        np.testing.assert_array_equal(path[1], [0, 0, 0, 0, .1, .1, .1])

    def test_soft_escape_never_overrides_mesh_collision(self):
        model = SimpleNamespace(has_table=True, table_soft_clearance_m=.02,
                                minimum_table_distance=lambda q: q[0], check_path=lambda path: None)
        planner = SimpleNamespace(collision_model=model, check_workspace_path=lambda path: None)
        path = np.zeros((2, 7))
        path[:, 0] = [.01, .03]
        check_home_path(path, planner, allow_table_soft_escape=True, **self.options)
        with self.assertRaisesRegex(ValueError, 'table_clearance'):
            check_home_path(path, planner, allow_table_soft_escape=False, **self.options)
        model.check_path = lambda path: (0, 'mesh contact')
        with self.assertRaisesRegex(ValueError, 'collision at sample'):
            check_home_path(path, planner, allow_table_soft_escape=True, **self.options)

    def test_band_alignment_keeps_absolute_target_and_disables_escape(self):
        start, end = np.zeros(7), np.zeros(7)
        start[0] = 2*np.pi
        with patch('dexmani_real.control.homing.check_home_path') as check:
            path = compute_band_alignment_path(start, end, object(), **self.options)
        np.testing.assert_array_equal(path, [start, end])
        self.assertFalse(check.call_args.kwargs['allow_table_soft_escape'])


class ReplayConsistencyTests(unittest.TestCase):
    def test_absolute_joint_error_and_geometric_pose_error_keep_their_units(self):
        from dexmani_real.replay.consistency import compute_metrics

        pose = np.tile([.1, .2, .3, 1, 0, 0, 0, 1, 0], (2, 1))
        trajectory = SimpleNamespace(fps=10, arm_qpos=np.zeros((2, 7)), hand_qpos=np.zeros((2, 12)),
                                     arm_ee=pose, episode_path='offline', task_label='test')
        measured = np.zeros((2, 7))
        measured[:, 0] = 2*np.pi
        replay = dict(arm_qpos=measured, hand_qpos=np.zeros((2, 12)), arm_candidate=measured+.1,
                      submit_started_ns=np.array([-1, 100]), eef_pos=pose[:, :3]+[.001, 0, 0],
                      eef_rot6d=pose[:, 3:])
        metrics = compute_metrics(trajectory, replay)
        self.assertEqual(metrics['arm_joint']['mae_per_joint_deg'][0], 360)
        self.assertEqual(metrics['eef_position']['mean_error_mm'], 1)
        self.assertEqual(metrics['eef_orientation']['mean_error_deg'], 0)
        self.assertEqual(metrics['pre_submit_arm_target_gap']['mean_deg'], 5.73)
        self.assertEqual(metrics['trajectory_alignment']['per_active_joint_offset_frames'], {})
        self.assertEqual(metrics['valid_frames']['arm'], 2)
        replay['hand_qpos'] = np.zeros((1, 12))
        with self.assertRaisesRegex(ValueError, 'same row count'):
            compute_metrics(trajectory, replay)

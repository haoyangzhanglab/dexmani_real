"""HOME 路径能力、几何约束与查询复用；不连接设备。"""

import unittest
from types import SimpleNamespace

import numpy as np

from dexmani_real.control.config import EnvironmentConfig
from dexmani_real.control.homing import (
    HomePathRejected,
    HomePlanner,
    interpolate_waypoints,
)
from dexmani_real.robot.xarm7.config import HOME_DWELL_S, ArmConfig


def joints(proximal, distal=0.):
    q = np.zeros(7)
    q[[0, 4]] = proximal, distal
    return q


class HomingTests(unittest.TestCase):
    def setUp(self):
        self.planner = HomePlanner.__new__(HomePlanner)
        self.planner.arm = ArmConfig()
        self.planner.environment = EnvironmentConfig()
        self.planner.limits = np.column_stack((
            self.planner.arm.joint_limit_lower, self.planner.arm.joint_limit_upper,
        ))
        self.planner.workspace_bounds = self.planner.environment.workspace.as_array()
        self.queries = []
        self.obstacle = lambda q: False
        self.clearance = lambda q: .1
        self.position = lambda q: np.array([.5, 0., .25])
        self.planner.collision_model = SimpleNamespace(
            table=SimpleNamespace(soft_clearance_m=.02),
            rejection=self.query,
            eef_position=lambda: self.position(self.q),
            minimum_table_distance=lambda: self.clearance(self.q) + .02,
        )

    def query(self, q):
        self.q = q.copy()
        self.queries.append(tuple(q))
        return 'test_obstacle' if self.obstacle(q) else None

    def cost(self, path):
        return (np.max(np.abs(np.diff(path, axis=0)), axis=1).sum()
                / self.planner.arm.homing.max_speed_rad_per_s
                + (len(path) - 1) * HOME_DWELL_S)

    def test_direct_path_and_already_home(self):
        start, target = joints(.4, .2), joints(0.)
        np.testing.assert_array_equal(self.planner.plan(start, target), [start, target])
        self.assertEqual(self.planner.plan(target + .001, target).shape, (0, 7))

    def test_preserves_proximal_and_distal_detours(self):
        start, target = joints(0.), joints(1., 1.)
        for proximal_first in (True, False):
            with self.subTest(proximal_first=proximal_first):
                self.obstacle = (lambda q: q[0] < .8 and q[4] > .05) if proximal_first else (
                    lambda q: q[4] < .8 and q[0] > .05
                )
                mid = joints(1., 0.) if proximal_first else joints(0., 1.)
                np.testing.assert_array_equal(self.planner.plan(start, target), [start, mid, target])

    def test_preserves_equivalent_branch_and_exact_canonical_finish(self):
        start, target = joints(5.8, 1.), joints(-.2)
        self.obstacle = lambda q: 0. < q[4] < 1. and q[0] < 6.
        path = self.planner.plan(start, target)
        np.testing.assert_array_equal(path[0], start)
        np.testing.assert_array_equal(path[-1], target)
        self.assertTrue(np.any(path[:, 0] > 6.))
        self.assertFalse(any(self.obstacle(q) for q in interpolate_waypoints(path, np.deg2rad(1.))))

    def test_selects_lower_cost_complete_path_instead_of_first_feasible_branch(self):
        start, target = joints(6., 5.), joints(0.)
        self.obstacle = lambda q: abs(q[0] - 3.) < .04 and abs(q[4] - 2.5) < .04
        old_first_feasible = np.array([start, joints(0., 5.), target])
        path = self.planner.plan(start, target)
        self.assertLess(self.cost(path), self.cost(old_first_feasible))
        np.testing.assert_array_equal(path[[0, -1]], [start, target])
        self.assertFalse(any(self.obstacle(q) for q in interpolate_waypoints(path, np.deg2rad(1.))))

    def test_alignment_segment_is_checked_before_path_is_accepted(self):
        self.obstacle = lambda q: (
            (0. < q[4] < 1. and q[0] < 6.) or (2.8 < q[0] < 3.2 and abs(q[4]) < .01)
        )
        with self.assertRaisesRegex(HomePathRejected, 'test_obstacle'):
            self.planner.plan(joints(5.8, 1.), joints(-.2))

    def test_soft_clearance_escape_and_reentry(self):
        start, target = joints(0.), joints(.2)
        self.clearance = lambda q: -.01 + .1 * q[0]
        np.testing.assert_array_equal(self.planner.plan(start, target), [start, target])
        for clearance in (
            lambda q: -.01 + .1 * q[0] - .02 * np.sin(np.pi * q[0] / .2),
            lambda q: -.005 if .08 < q[0] < .12 else .01,
        ):
            self.clearance = clearance
            with self.assertRaisesRegex(HomePathRejected, 'table_clearance'):
                self.planner.plan(start, target)

    def test_already_home_still_requires_clearance(self):
        self.clearance = lambda q: -.0001 if q[0] > 0 else .1
        with self.assertRaisesRegex(HomePathRejected, 'table_clearance'):
            self.planner.plan(joints(.001), joints(0.))

    def test_validity_cache_is_shared_within_plan_and_discarded_afterwards(self):
        start, target = joints(.3), joints(0.)
        self.planner.plan(start, target)
        self.assertEqual(len(self.queries), len(set(self.queries)))
        self.obstacle = lambda q: True
        with self.assertRaisesRegex(HomePathRejected, 'test_obstacle'):
            self.planner.plan(start, target)

    def test_invalid_common_endpoint_is_rejected_without_scanning_paths(self):
        self.obstacle = lambda q: q[0] == 0.
        with self.assertRaisesRegex(HomePathRejected, 'test_obstacle'):
            self.planner.plan(joints(6.), joints(0.))
        self.assertEqual(len(self.queries), 2)

    def test_midpoint_rejection_is_early_but_other_samples_are_not_skipped(self):
        self.obstacle = lambda q: 1.3 < q[0] < 1.5
        with self.assertRaises(HomePathRejected):
            self.planner.plan(joints(0.), joints(2.8))
        self.assertLess(len(self.queries), 10)
        self.obstacle = lambda q: .14 < q[0] < .16
        with self.assertRaises(HomePathRejected):
            self.planner.plan(joints(0.), joints(.4))

    def test_workspace_and_nonfinite_geometry_remain_errors(self):
        self.position = lambda q: np.array([.1, 0., .25])
        with self.assertRaisesRegex(HomePathRejected, 'workspace'):
            self.planner.plan(joints(.2), joints(0.))
        self.position = lambda q: np.array([.5, 0., .25])
        self.clearance = lambda q: np.nan
        with self.assertRaisesRegex(RuntimeError, 'table_check_nonfinite'):
            self.planner.plan(joints(.2), joints(0.))

    def test_bad_inputs_are_rejected_before_geometry_queries(self):
        for q in (np.zeros(6), np.full(7, np.nan)):
            with self.assertRaises(ValueError):
                self.planner.plan(q, joints(0.))
        with self.assertRaisesRegex(HomePathRejected, 'joint limits'):
            self.planner.plan(joints(7.), joints(0.))
        self.assertEqual(self.queries, [])

    def test_interpolation_preserves_waypoints_and_resolution(self):
        path = np.array([joints(.35, .03), joints(.35, -.17), joints(-.23, -.17)])
        step = np.deg2rad(1.)
        samples = interpolate_waypoints(path, step)
        self.assertLessEqual(np.max(np.abs(np.diff(samples, axis=0))), step + 1e-12)
        for q in path:
            self.assertTrue(np.any(np.all(samples == q, axis=1)))


if __name__ == '__main__':
    unittest.main()

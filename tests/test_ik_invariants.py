from __future__ import annotations

import numpy as np
import pytest

from dexmani_real.planning.ik import TeleopIKSolver, nullspace_projector
from dexmani_real.planning.path_utils import wrap_nearest_equivalent
from dexmani_real.planning.planner import XArm7MotionPlanner
from dexmani_real.planning.types import Pose, TeleopProfile

LOWER = np.deg2rad(np.array([-360, -118, -360, -11, -360, -97, -360], dtype=np.float64))
UPPER = np.deg2rad(np.array([360, 120, 360, 225, 360, 180, 360], dtype=np.float64))
EQUIVALENT = (UPPER - LOWER) > 2.0 * np.pi


@pytest.mark.parametrize(
    ("raw_deg", "reference_deg", "expected_deg"),
    [(540.0, 359.0, 180.0), (-540.0, -359.0, -180.0), (720.0, 350.0, 360.0)],
)
def test_wrap_nearest_equivalent_boundary_cases(raw_deg, reference_deg, expected_deg):
    qpos = np.zeros(7)
    reference = np.zeros(7)
    qpos[0] = np.deg2rad(raw_deg)
    reference[0] = np.deg2rad(reference_deg)

    result = wrap_nearest_equivalent(qpos, reference, tuple(LOWER), tuple(UPPER))

    assert np.rad2deg(result[0]) == pytest.approx(expected_deg)
    assert (result[0] - qpos[0]) / (2.0 * np.pi) == pytest.approx(round((result[0] - qpos[0]) / (2.0 * np.pi)))


def test_wrap_nearest_equivalent_random_properties():
    rng = np.random.default_rng(20260808)
    for _ in range(5000):
        qpos = rng.uniform(-4.0 * np.pi, 4.0 * np.pi, 7)
        reference = rng.uniform(LOWER, UPPER)
        result = wrap_nearest_equivalent(qpos, reference, tuple(LOWER), tuple(UPPER))

        for joint in np.flatnonzero(EQUIVALENT):
            legal = [
                qpos[joint] + k * 2.0 * np.pi
                for k in range(-6, 7)
                if LOWER[joint] <= qpos[joint] + k * 2.0 * np.pi <= UPPER[joint]
            ]
            if not legal:
                assert result[joint] == qpos[joint]
                continue
            assert LOWER[joint] <= result[joint] <= UPPER[joint]
            assert (result[joint] - qpos[joint]) / (2.0 * np.pi) == pytest.approx(
                round((result[joint] - qpos[joint]) / (2.0 * np.pi))
            )
            assert abs(result[joint] - reference[joint]) == pytest.approx(
                min(abs(value - reference[joint]) for value in legal)
            )


def test_wrap_does_not_modify_non_equivalent_joints():
    qpos = np.array([0.0, 4.0, 0.0, -2.0, 0.0, 5.0, 0.0])
    result = wrap_nearest_equivalent(qpos, np.zeros(7), tuple(LOWER), tuple(UPPER))
    np.testing.assert_array_equal(result[~EQUIVALENT], qpos[~EQUIVALENT])


@pytest.mark.parametrize("rank", [6, 5, 3])
def test_nullspace_projector_tracks_numerical_rank(rank):
    rng = np.random.default_rng(rank)
    u, _, vt = np.linalg.svd(rng.normal(size=(6, 7)), full_matrices=False)
    singular_values = np.zeros(6)
    singular_values[:rank] = np.linspace(2.0, 1.0, rank)
    jacobian = (u * singular_values) @ vt

    projector = nullspace_projector(jacobian)

    np.testing.assert_allclose(jacobian @ projector, 0.0, atol=1e-10)
    np.testing.assert_allclose(projector @ projector, projector, atol=1e-10)
    assert np.trace(projector) == pytest.approx(7 - rank, abs=1e-10)


def test_final_ik_validation_rejects_nonlinear_nullspace_drift():
    class FakeKinematics:
        dof = 7

        @staticmethod
        def compute_eef_jacobian_and_pose_world(qpos):
            return np.column_stack([np.eye(6), np.zeros(6)]), Pose.identity()

        @staticmethod
        def compute_world_pose_error(_target, qpos):
            return float(qpos[6] ** 2), 0.0

    class FakeManager:
        joint_limits = np.column_stack((-np.ones(7), np.ones(7)))

        @staticmethod
        def canonicalize_qpos(qpos, _reference):
            return np.asarray(qpos).copy()

        @staticmethod
        def limit_violation(qpos, limits):
            outside = (qpos < limits[:, 0]) | (qpos > limits[:, 1])
            return outside, np.zeros(7)

        @staticmethod
        def compute_qpos_delta(qpos, reference):
            return qpos - reference

    profile = TeleopProfile(
        check_self_collision=False,
        max_pose_error_pos_m=0.001,
        nullspace_step_size_deg=10.0,
    )
    solver = TeleopIKSolver(FakeKinematics(), FakeManager(), profile, home_qpos=np.array([0, 0, 0, 0, 0, 0, 0.5]))

    result = solver._command_from_target_qpos(Pose.identity(), np.zeros(7), np.zeros(7), np.zeros(7), profile, {})

    assert not result.success
    assert result.held
    assert "pose-error" in result.reason


def test_workspace_segment_checks_intermediate_states():
    planner = object.__new__(XArm7MotionPlanner)
    planner.workspace_bounds = np.array([[-0.5, 0.5], [-1.0, 1.0], [-1.0, 1.0]])
    # Both endpoints are inside, but the nonlinear midpoint leaves the bound.
    planner.compute_eef_pose_world = lambda q: Pose([0.8 * np.sin(q[0]), 0.0, 0.0], [1, 0, 0, 0])
    start = np.zeros(7)
    end = np.zeros(7)
    end[0] = np.pi

    assert not planner.is_workspace_segment_safe(start, end)

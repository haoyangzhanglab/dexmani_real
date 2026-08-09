from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.defaults import HandParams
from dexmani_real.robot.xhand import XHand, XHandConfig
from dexmani_real.teleop.tag_retargeting.optimizer import HandOptimizer


def _optimizer_without_kinematics() -> HandOptimizer:
    optimizer = object.__new__(HandOptimizer)
    optimizer.dof = 3
    optimizer.finger_num = 2
    optimizer.joint_limits_lower = np.array([0.1, -0.2, 0.0])
    optimizer.joint_limits_upper = np.array([1.0, 0.8, 0.5])
    optimizer._default_qpos = (optimizer.joint_limits_lower + optimizer.joint_limits_upper) / 2.0
    optimizer.last_qpos = optimizer._default_qpos.copy()
    optimizer.qpos_stage1 = None
    optimizer.pinch_factors = np.zeros(2)
    optimizer._bounds_warn = lambda *args, **kwargs: None
    optimizer.feedback_bound_tolerance_rad = 0.03
    optimizer._feedback_bound_checks = 0
    optimizer._feedback_bound_clips = 0
    optimizer._feedback_bound_within_tolerance = 0
    optimizer._feedback_bound_over_tolerance = 0
    optimizer._feedback_bound_max_violation_rad = 0.0
    optimizer._feedback_bound_joint_clip_counts = np.zeros(3, dtype=np.int64)
    return optimizer


def test_reset_clips_hardware_warm_start_to_nlopt_bounds() -> None:
    optimizer = _optimizer_without_kinematics()

    optimizer.reset(np.array([0.08, 0.9, 0.25]))

    np.testing.assert_allclose(optimizer.last_qpos, [0.1, 0.8, 0.25])
    stats = optimizer.feedback_bound_stats
    assert stats["checks"] == 1
    assert stats["clipped_checks"] == 1
    assert stats["over_tolerance_checks"] == 1
    assert stats["max_violation_rad"] == pytest.approx(0.1)
    np.testing.assert_array_equal(stats["per_joint_clip_counts"], [1, 1, 0])


def test_feedback_bound_stats_classify_small_excursion_as_tolerated() -> None:
    optimizer = _optimizer_without_kinematics()

    optimizer.reset(np.array([0.08, 0.5, 0.25]))

    stats = optimizer.feedback_bound_stats
    assert stats["within_tolerance_checks"] == 1
    assert stats["over_tolerance_checks"] == 0
    assert stats["max_violation_rad"] == pytest.approx(0.02)


def test_reset_without_valid_hardware_state_restores_midpoint() -> None:
    optimizer = _optimizer_without_kinematics()
    optimizer.last_qpos[:] = 0.123

    optimizer.reset(np.full(3, np.nan))

    np.testing.assert_allclose(optimizer.last_qpos, optimizer._default_qpos)


def test_stage1_failure_is_reported_instead_of_false_success() -> None:
    class FailingOpt:
        warm_start: np.ndarray | None = None

        def optimize(self, qpos: np.ndarray) -> np.ndarray:
            self.warm_start = qpos.copy()
            raise ValueError("synthetic NLopt failure")

    optimizer = _optimizer_without_kinematics()
    optimizer.last_qpos = np.array([0.08, 0.9, 0.25])
    optimizer.finger_scale = np.ones(2)
    optimizer.opt_s1 = FailingOpt()
    warnings: list[str] = []
    optimizer._stage1_warn = lambda message, *args, **kwargs: warnings.append(message)

    result = optimizer.solve(np.zeros((2, 3)))

    assert result is None
    np.testing.assert_allclose(optimizer.opt_s1.warm_start, [0.1, 0.8, 0.25])
    np.testing.assert_allclose(optimizer.last_qpos, [0.1, 0.8, 0.25])
    assert warnings == ["HandOptimizer: Stage 1 NLopt failed — caller will hold last command"]


@pytest.mark.parametrize("tolerance", [-0.01, float("nan"), float("inf")])
def test_hand_feedback_bound_tolerance_must_be_finite_nonnegative(tolerance: float) -> None:
    with pytest.raises(ValueError, match="feedback_bound_tolerance_rad"):
        HandParams(feedback_bound_tolerance_rad=tolerance)

    with pytest.raises(ValueError, match="feedback_bound_tolerance_rad"):
        XHandConfig(feedback_bound_tolerance_rad=tolerance)


def test_xhand_feedback_stats_are_independent_from_command_clipping() -> None:
    config = XHandConfig(feedback_bound_tolerance_rad=0.01, clip_report_tolerance=0.5)
    driver = XHand(config)
    qpos = config.qpos_min.copy()

    driver._record_feedback_bound_check(qpos)
    qpos[0] -= 0.005
    driver._record_feedback_bound_check(qpos)
    qpos[0] -= 0.010
    driver._record_feedback_bound_check(qpos)

    stats = driver.feedback_bound_stats
    assert stats["checks"] == 3
    assert stats["outside_bounds_frames"] == 2
    assert stats["over_tolerance_frames"] == 1
    assert stats["max_violation_rad"] == pytest.approx(0.015)
    np.testing.assert_array_equal(stats["per_joint_outside_counts"], [2] + [0] * 11)
    np.testing.assert_array_equal(stats["per_joint_over_tolerance_counts"], [1] + [0] * 11)

    command = config.qpos_min.copy()
    command[0] -= 0.1
    limited = driver._limit_joint_range(command)
    assert limited[0] == pytest.approx(config.qpos_min[0])
    assert driver.last_joint_limit_clipped is False  # separate command-report tolerance
    assert driver.feedback_bound_stats["checks"] == 3  # commands are not feedback samples


def test_xhand_rejects_invalid_action_without_calling_sdk() -> None:
    driver = XHand(XHandConfig())
    driver.control = Mock()
    driver.hand_command = object()
    driver.last_qpos_cmd = np.zeros(12)

    assert not driver.send_action(np.full(12, np.nan))
    assert not driver.send_action(np.zeros(11))
    driver.control.send_command.assert_not_called()


def test_xhand_tactile_startup_load_check_fails_closed() -> None:
    driver = XHand(XHandConfig(tactile_contact_threshold=1.0))

    assert driver._tactile_load_present(None)
    assert driver._tactile_load_present(np.full((5, 3), np.nan))
    assert driver._tactile_load_present(np.array([[2.0, 0.0, 0.0]] + [[0.0, 0.0, 0.0]] * 4))
    assert not driver._tactile_load_present(np.zeros((5, 3)))

    with pytest.raises(ValueError, match="tactile_contact_threshold"):
        XHandConfig(tactile_contact_threshold=float("nan"))

"""用真实运动学验证数学性质；优化器替身只用于输出边界与精确滤波断言。"""

from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import nlopt
import numpy as np
import pytest

from dexmani_real.data.raw import RawReader
from dexmani_real.teleop.retargeting.landmarks import HandTargets, prepare_landmarks
from dexmani_real.teleop.retargeting.objective import HandObjective, huber
from dexmani_real.teleop.retargeting.solver import HandRetargeter
from scripts.preview_hand_retargeting import preview_episode, read_recorded_parameters

SAMPLE = Path(__file__).parent / "fixtures" / "hand_retargeting"


@pytest.fixture
def recorded():
    with RawReader(SAMPLE) as reader:
        config, limits, calibration = read_recorded_parameters(reader)
        landmarks = reader["vr_landmarks"][:]
        qpos = reader["hand_qpos"][:]
    solver = HandRetargeter(config, limits)
    solver.calibration = calibration
    return solver, landmarks, qpos


def finite_difference(function, point, step=1e-6):
    result = np.empty((*np.shape(function(point)), point.size))
    for index in range(point.size):
        offset = np.zeros_like(point)
        offset.flat[index] = step
        result[..., index] = (function(point + offset) - function(point - offset)) / (2 * step)
    return result


@pytest.mark.parametrize("radius", [0., 1e-8, .1, .3 - 1e-5, .3, .3 + 1e-5, 1.])
def test_radial_huber_gradient(radius):
    error = radius * np.array([.6, 0., .8])
    value, derivative = huber(error, .3)
    assert value >= 0
    numerical = finite_difference(lambda x: huber(x, .3)[0], error, step=1e-7)
    np.testing.assert_allclose(derivative, numerical, atol=3e-8, rtol=1e-6)


@pytest.mark.parametrize("fraction", [1e-4, .37, 1 - 1e-4])
def test_hand_jacobian_in_sdk_order_and_owned_outputs(recorded, fraction):
    solver, _, _ = recorded
    lower, upper = solver.limits.T
    qpos = lower + fraction * (upper - lower)
    positions, jacobian = solver.fk.positions_and_jacobians(qpos, solver.frame_ids)
    numerical = finite_difference(lambda q: solver.fk.forward(q, solver.frame_ids)[:, :3, 3], qpos)
    np.testing.assert_allclose(jacobian, numerical, atol=1e-9, rtol=1e-6)
    saved_positions, saved_jacobian = positions.copy(), jacobian.copy()
    solver.fk.positions_and_jacobians((lower + upper) / 2, solver.frame_ids)
    np.testing.assert_array_equal(positions, saved_positions)
    np.testing.assert_array_equal(jacobian, saved_jacobian)


@pytest.mark.parametrize("loss", ["position", "proximal", "terminal", "pinch", "damping", "total"])
@pytest.mark.parametrize("activation", [0., .4, 1.])
@pytest.mark.parametrize("dt", [.008, .0625, .2])
def test_objective_gradient_for_each_loss(recorded, loss, activation, dt):
    solver, landmarks, _ = recorded
    config = solver.config
    if loss != "total":
        weights = dict(shape_weight=0., tip_weight=0., proximal_direction_weight=0.,
                       direction_weight=0., pinch_weight=0., damping_s=0.)
        weights.update({
            "position": dict(shape_weight=.25, tip_weight=2.),
            "proximal": dict(proximal_direction_weight=2.),
            "terminal": dict(direction_weight=1.),
            "pinch": dict(pinch_weight=8.),
            "damping": dict(damping_s=.002),
        }[loss])
        config = replace(config, **weights)
    targets = solver.calibration.targets(
        prepare_landmarks(landmarks[150]), solver.geometry, config,
        fk=solver.fk, frame_ids=solver.frame_ids,
    )
    targets = replace(targets, activation=np.full(4, activation))
    lower, upper = solver.limits.T
    previous = lower + .6 * (upper - lower)
    objective = HandObjective(solver.fk, solver.frame_ids, targets, previous, dt, config)
    for fraction in (1e-4, .37, 1 - 1e-4):
        qpos = lower + fraction * (upper - lower)
        gradient = np.empty(12)
        value = objective(qpos, gradient)
        scalar = lambda q: objective(q, np.empty(0))
        assert value == pytest.approx(scalar(qpos), abs=1e-12)
        numerical = finite_difference(scalar, qpos)
        np.testing.assert_allclose(gradient, numerical, atol=1e-7, rtol=1e-5)


def test_exact_geometry_is_stationary(recorded):
    solver, _, _ = recorded
    qpos = solver.limits.mean(axis=1)
    points = solver.fk.forward(qpos, solver.frame_ids)[:, :3, 3].reshape(5, 3, 3)
    segments = np.diff(points, axis=1)
    targets = HandTargets(
        points, segments / np.linalg.norm(segments, axis=-1, keepdims=True),
        np.zeros(4), np.linalg.norm(points[1:, 2] - points[0, 2], axis=1),
    )
    gradient = np.empty(12)
    objective = HandObjective(solver.fk, solver.frame_ids, targets, qpos, .0625, solver.config)
    assert objective(qpos, gradient) == pytest.approx(0., abs=1e-15)
    np.testing.assert_allclose(gradient, 0., atol=1e-12)


@pytest.mark.parametrize("tau", [0., .02, .2])
@pytest.mark.parametrize("dt", [.001, .0625, .5])
def test_final_rate_and_smoothing_at_optimizer_boundary(recorded, tau, dt):
    solver, landmarks, _ = recorded
    solver.config = replace(solver.config, smoothing_tau_s=tau)
    previous = solver.limits.mean(axis=1)
    solver.reset(previous, timestamp_ns=1_000_000_000)
    optimizer = solver.optimizer = Mock()
    optimizer.optimize.side_effect = lambda _: optimizer.set_upper_bounds.call_args.args[0]
    optimizer.last_optimize_result.return_value = nlopt.SUCCESS
    optimizer.last_optimum_value.return_value = 1.
    command = solver.retarget(landmarks[150], timestamp_ns=1_000_000_000 + round(dt * 1e9))
    alpha = 1. if tau == 0 else -np.expm1(-dt / tau)
    expected_step = np.minimum(solver.config.max_joint_speed_rad_s * dt,
                               alpha * (solver.limits[:, 1] - previous))
    np.testing.assert_allclose(command - previous, expected_step, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(solver.reference_qpos, previous)
    assert solver.reference_timestamp_ns == 1_000_000_000


@pytest.mark.parametrize("failure", ["geometry", "roundoff"])
def test_rejected_step_keeps_reference_and_elapsed_time(recorded, failure):
    solver, landmarks, qpos = recorded
    solver.reset(qpos[0], timestamp_ns=1_000_000_000)
    original = solver.optimizer
    invalid = landmarks[1].copy()
    if failure == "geometry":
        invalid[2] = invalid[1]
    else:
        solver.optimizer = Mock(wraps=original)
        solver.optimizer.optimize.side_effect = nlopt.RoundoffLimited("test roundoff")
    assert solver.retarget(invalid, timestamp_ns=1_050_000_000) is None
    assert solver.rejection_reason
    np.testing.assert_array_equal(solver.reference_qpos, qpos[0])
    assert solver.reference_timestamp_ns == 1_000_000_000
    solver.optimizer = original
    recovered = solver.retarget(landmarks[2], timestamp_ns=1_100_000_000)
    # 未提交的拒绝行等价于没有调用，恢复仍使用距最后提交的完整 dt。
    solver.reset(qpos[0], timestamp_ns=1_000_000_000)
    direct = solver.retarget(landmarks[2], timestamp_ns=1_100_000_000)
    np.testing.assert_allclose(recovered, direct, atol=1e-12, rtol=0)
    solver.commit(recovered, timestamp_ns=1_100_000_000)
    solver.reset(qpos[0], timestamp_ns=2_000_000_000)
    np.testing.assert_array_equal(solver.reference_qpos, qpos[0])
    assert solver.reference_timestamp_ns == 2_000_000_000


@pytest.mark.parametrize("bad_solution", ["nan", "shape", "bounds", "objective", "status"])
def test_invalid_optimizer_output_is_not_committed(recorded, bad_solution):
    solver, landmarks, qpos = recorded
    solver.reset(qpos[0], timestamp_ns=1_000_000_000)
    optimizer = solver.optimizer = Mock()
    optimizer.optimize.return_value = {
        "nan": np.full(12, np.nan), "shape": np.zeros(11),
        "bounds": solver.limits[:, 1] + 1.,
    }.get(bad_solution, qpos[0].copy())
    optimizer.last_optimize_result.return_value = -1 if bad_solution == "status" else nlopt.SUCCESS
    optimizer.last_optimum_value.return_value = np.nan if bad_solution == "objective" else 1.
    with pytest.raises((ValueError, RuntimeError)):
        solver.retarget(landmarks[1], timestamp_ns=1_050_000_000)
    np.testing.assert_array_equal(solver.reference_qpos, qpos[0])
    assert solver.reference_timestamp_ns == 1_000_000_000


def test_recorded_trajectory_limits_and_final_rate(recorded):
    solver, _, _ = recorded
    result = preview_episode(SAMPLE, view=False)
    assert not any(result["rejection_reasons"])
    commands = result["reference_qpos"]
    assert np.all(commands >= solver.limits[:, 0])
    assert np.all(commands <= solver.limits[:, 1])
    dt = np.diff(result["observation_ns"]) * 1e-9
    assert np.all(np.abs(np.diff(commands, axis=0)) <= solver.config.max_joint_speed_rad_s * dt[:, None] + 1e-12)

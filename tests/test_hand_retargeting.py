from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dexmani_real.config.teleop import HandRetargetingConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.control.actions import StartPoseRejected
from dexmani_real.teleop.retargeting.landmarks import (
    HandGeometryRejected,
    hand_lengths,
    prepare_landmarks,
)


@pytest.fixture
def landmarks():
    points = np.zeros((21, 3))
    for finger in range(5):
        points[1 + 4 * finger:5 + 4 * finger] = [
            [.01, .04 - .02 * finger, .05 + .02 * joint] for joint in range(4)
        ]
    return points


@pytest.fixture
def retargeter(landmarks):
    pytest.importorskip('nlopt')
    pytest.importorskip('pinocchio')
    from dexmani_real.teleop.retargeting.solver import HandRetargeter

    config = HandRetargetingConfig()
    retargeter = HandRetargeter(config, HandConfig())
    measurement = hand_lengths(prepare_landmarks(landmarks))
    retargeter.calibrate([measurement] * config.calibration_samples)
    retargeter.reset(retargeter.limits.mean(axis=1), timestamp_ns=1)
    return retargeter


@pytest.mark.parametrize('index', [9, 17])
def test_undefined_palm_is_an_explicit_geometry_rejection(landmarks, index):
    landmarks[index] = landmarks[0 if index == 9 else 5]
    with np.errstate(divide='raise', invalid='raise'), pytest.raises(HandGeometryRejected):
        prepare_landmarks(landmarks)


def test_bad_geometry_is_held_but_target_code_error_propagates(retargeter, landmarks):
    reference = retargeter.reference_qpos.copy()
    degenerate = landmarks.copy()
    degenerate[17] = degenerate[5]
    assert retargeter.retarget(degenerate, timestamp_ns=62500001) is None
    assert 'degenerate palm' in retargeter.rejection_reason
    np.testing.assert_array_equal(retargeter.reference_qpos, reference)
    retargeter.calibration = SimpleNamespace(targets=Mock(side_effect=ValueError('target bug')))
    with pytest.raises(ValueError, match='target bug'):
        retargeter.retarget(landmarks, timestamp_ns=62500001)


@pytest.mark.parametrize('malformed', [np.zeros(11), np.full(12, np.nan)])
def test_malformed_feedback_is_not_a_retryable_start(retargeter, malformed):
    reference = retargeter.reference_qpos.copy()
    with pytest.raises(ValueError) as caught:
        retargeter.reset(malformed, timestamp_ns=2)
    assert not isinstance(caught.value, StartPoseRejected)
    np.testing.assert_array_equal(retargeter.reference_qpos, reference)
    assert retargeter.reference_timestamp_ns == 1


@pytest.mark.parametrize('status', [5, 6])
def test_valid_budget_limited_solution_is_accepted(retargeter, landmarks, status):
    reference = retargeter.reference_qpos.copy()
    optimizer = Mock()
    optimizer.optimize.return_value = reference + .01
    optimizer.last_optimize_result.return_value = status
    optimizer.last_optimum_value.return_value = .2
    retargeter.optimizer = optimizer
    command = retargeter.retarget(landmarks, timestamp_ns=62500001)
    assert command is not None
    assert np.all(command > reference)
    assert np.max(command - reference) <= retargeter.config.max_joint_speed_rad_s * .0625
    np.testing.assert_array_equal(retargeter.reference_qpos, reference)
    optimizer.optimize.assert_called_once()


@pytest.mark.parametrize('kind', ['shape', 'nonfinite', 'bounds', 'loss', 'status'])
def test_invalid_backend_output_stops_instead_of_holding(retargeter, landmarks, kind):
    reference = retargeter.reference_qpos.copy()
    optimizer = Mock()
    optimizer.optimize.return_value = {
        'shape': np.zeros(11), 'nonfinite': np.full(12, np.nan),
        'bounds': reference + 10,
    }.get(kind, reference)
    optimizer.last_optimize_result.return_value = -1 if kind == 'status' else 3
    optimizer.last_optimum_value.return_value = np.nan if kind == 'loss' else .2
    retargeter.optimizer = optimizer
    with pytest.raises((ValueError, RuntimeError)):
        retargeter.retarget(landmarks, timestamp_ns=62500001)
    assert retargeter.rejection_reason is None
    np.testing.assert_array_equal(retargeter.reference_qpos, reference)
    assert retargeter.reference_timestamp_ns == 1


def test_known_roundoff_failure_is_a_hold(retargeter, landmarks):
    import nlopt

    retargeter.optimizer.optimize = Mock(side_effect=nlopt.RoundoffLimited('roundoff'))
    assert retargeter.retarget(landmarks, timestamp_ns=62500001) is None
    assert 'RoundoffLimited' in retargeter.rejection_reason
    assert retargeter.reference_timestamp_ns == 1


@pytest.mark.parametrize('program_error', [False, True])
def test_calibration_skips_only_expected_geometry(monkeypatch, landmarks, program_error):
    from dexmani_real.teleop import run
    from dexmani_real.teleop.retargeting import landmarks as geometry

    retargeter = Mock()
    loop = SimpleNamespace(
        controller=SimpleNamespace(hand_retargeter=retargeter),
        runtime=SimpleNamespace(
            hand_retargeting=SimpleNamespace(calibration_timeout_s=5., calibration_samples=2),
            teleop=SimpleNamespace(vr_stale_threshold_s=.5),
        ),
        runtime_state=SimpleNamespace(cancel_epoch=1),
        sensors=SimpleNamespace(channels=None),
        robot=Mock(),
        poll_blocking_commands=Mock(return_value=False),
    )
    bad = landmarks.copy()
    bad[17] = bad[5]
    frames = [{"recv_ts_ns": i + 1, "landmarks": points}
              for i, points in enumerate((bad, landmarks, landmarks))]
    read = Mock(side_effect=frames)
    monkeypatch.setattr(run, 'read_vr_frame', read)
    monkeypatch.setattr(run.time, 'sleep', lambda _: None)
    monkeypatch.setattr(run.time, 'monotonic_ns', lambda: 1000000)
    if program_error:
        monkeypatch.setattr(geometry, 'prepare_landmarks', Mock(side_effect=ValueError('geometry bug')))
        with pytest.raises(ValueError, match='geometry bug'):
            run.calibrate_hand(loop, 1)
        retargeter.calibrate.assert_not_called()
    else:
        run.calibrate_hand(loop, 1)
        assert read.call_count == 3
        measurements = retargeter.calibrate.call_args.args[0]
        assert len(measurements) == 2
        np.testing.assert_array_equal(measurements[0], measurements[1])


def test_real_hand_gradient_and_filter_limits():
    pytest.importorskip('pinocchio')
    pytest.importorskip('nlopt')
    from dexmani_real.teleop.retargeting.landmarks import HandTargets
    from dexmani_real.teleop.retargeting.objective import HandObjective
    from dexmani_real.teleop.retargeting.solver import HandRetargeter
    cfg = HandRetargetingConfig()
    r = HandRetargeter(cfg, HandConfig())
    q = r.limits.mean(axis=1)
    points, jac = r.fk.positions_and_jacobians(q, r.frame_ids)
    points = points.reshape(5, 3, 3)
    segments = np.diff(points, axis=1)
    directions = segments / np.linalg.norm(segments, axis=2)[..., None]
    targets = HandTargets(points + .001, directions + .03, np.ones(4), np.ones(4)*.02, True)
    obj = HandObjective(r.fk, r.frame_ids, targets, q + .01, .06, cfg)
    gradient = np.empty(12)
    obj(q, gradient)
    eps = 1e-6
    numeric = np.array([(obj(q+e, np.empty(0))-obj(q-e, np.empty(0)))/(2*eps) for e in eps*np.eye(12)])
    np.testing.assert_allclose(gradient, numeric, atol=1e-7)
    numerical_jac = np.stack([(r.fk.forward(q+e, r.frame_ids)[:, :3, 3]-r.fk.forward(q-e, r.frame_ids)[:, :3, 3])/(2*eps) for e in eps*np.eye(12)], axis=2)
    np.testing.assert_allclose(jac, numerical_jac, atol=1e-8)
    for tau in (0., .02):
        r.config = replace(cfg, smoothing_tau_s=tau)
        r.calibration = SimpleNamespace(targets=lambda *a, **k: targets)
        # Geometry is tested separately; this isolates optimizer/filter time semantics.
        from unittest.mock import patch
        with patch('dexmani_real.teleop.retargeting.solver.prepare_landmarks', lambda x: x):
            for dt in (0, 1, 1000000, 62500000):
                r.reset(q, timestamp_ns=100)
                output = r.retarget(None, timestamp_ns=100+dt)
                assert output is not None
                assert np.max(np.abs(output-q)) <= cfg.max_joint_speed_rad_s*dt*1e-9 + 1e-12
                np.testing.assert_array_equal(r.reference_qpos, q)



def test_palm_rigid_invariance():
    from dexmani_real.teleop.retargeting.landmarks import prepare_landmarks
    points = np.zeros((21, 3))
    for finger in range(5):
        points[1+4*finger:5+4*finger] = [[.01, .04-.02*finger, .05+.02*j] for j in range(4)]
    rotation = Rotation.from_rotvec([.4, -.2, .6]).as_matrix()
    np.testing.assert_allclose(prepare_landmarks(points), prepare_landmarks(points@rotation.T + [1., 2., 3.]), atol=1e-14)



def test_optimizer_callback_error_is_not_a_hold(monkeypatch):
    pytest.importorskip('nlopt')
    pytest.importorskip('pinocchio')
    from dexmani_real.teleop.retargeting import solver
    r = solver.HandRetargeter(HandRetargetingConfig(), HandConfig())
    q = r.limits.mean(axis=1)
    r.reset(q, timestamp_ns=1)
    r.calibration = SimpleNamespace(targets=lambda *a, **k: None)
    monkeypatch.setattr(solver, 'prepare_landmarks', lambda x: x)
    def broken(*args):
        raise RuntimeError('model bug')
    monkeypatch.setattr(solver, 'HandObjective', lambda *a: broken)
    with pytest.raises(RuntimeError, match='model bug'):
        r.retarget(None, timestamp_ns=62500001)
    np.testing.assert_array_equal(r.reference_qpos, q)

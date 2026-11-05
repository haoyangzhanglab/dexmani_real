import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.teleop import TeleopConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.control.actions import ActionMapper, StartPoseRejected
from dexmani_real.control.ik import IKResult, IKSolver
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot.commands import (
    hand_reference_from_feedback,
    validate_hand_target,
)
from dexmani_real.teleop.vr import VRController


@pytest.mark.parametrize('side', [0, 1])
def test_feedback_roundoff_is_not_command_permission(side):
    cfg = HandConfig()
    lower, upper = np.asarray(cfg.qpos_min_rad), np.asarray(cfg.qpos_max_rad)
    q = (lower + upper) / 2
    q[2] = (lower[2] - 5e-9) if side == 0 else (upper[2] + 5e-9)
    saved = q.copy()
    reference = hand_reference_from_feedback(q, lower, upper)
    np.testing.assert_array_equal(q, saved)
    assert reference[2] == (lower if side == 0 else upper)[2]
    with pytest.raises(ValueError):
        validate_hand_target(q, lower, upper)
    q[2] += -.01 if side == 0 else .01
    with pytest.raises(ValueError):
        hand_reference_from_feedback(q, lower, upper)


def test_vr_start_and_policy_clipping_are_separate():
    solver = SimpleNamespace(
        constraints=SimpleNamespace(workspace=np.array([[0, 1], [-1, 1], [0, .5]])),
        compute_eef_pose_base=Mock(return_value=Pose(np.array([.4, 0, .51]), np.array([1., 0, 0, 0]))),
        solve=Mock(return_value=SimpleNamespace(qpos=np.zeros(7))),
    )
    mapper = ActionMapper(joint_limits=np.tile([-6., 6.], (7, 1)), hand_enabled=False,
                          hand_lower_rad=None, hand_upper_rad=None, solver=solver)
    controller = VRController(mapper, None, TeleopConfig(), np.eye(3))
    row = SimpleNamespace(arm={'qpos': [np.zeros(7)]}, vr={'wrist_pos': np.zeros(3),
                          'wrist_quat_wxyz': np.array([1., 0, 0, 0])})
    with pytest.raises(StartPoseRejected):
        controller.reset(row)
    solver.solve.assert_not_called()
    assert controller.anchor is None
    result = mapper.map_eef(solver.compute_eef_pose_base(), None, np.zeros(7))
    assert result.eef_pose.p[2] == .5
    solver.compute_eef_pose_base.return_value = Pose(np.array([.4, 0, .3]), np.array([1., 0, 0, 0]))
    controller.reset(row)
    np.testing.assert_allclose(controller._wrist_target(row.vr['wrist_pos'], row.vr['wrist_quat_wxyz']).p, [.4, 0, .3])
    solver.compute_eef_pose_base.return_value = Pose(np.array([.5, 0, .4]), np.array([1., 0, 0, 0]))
    controller.reset(row)
    np.testing.assert_allclose(controller.previous_eef_target.p, [.5, 0, .4])


@pytest.fixture
def ik_solver(monkeypatch):
    monkeypatch.setitem(sys.modules, 'mplib', SimpleNamespace(Pose=Pose))
    solver = IKSolver.__new__(IKSolver)
    solver.eef_link_id = 0
    solver.model = SimpleNamespace(compute_IK_CLIK=Mock(return_value=(np.zeros(7), False, None)))
    solver.limits = np.tile([-1., 1.], (7, 1))
    solver.joint_ranges = np.ones(7)*2
    solver.limit_center = np.zeros(7)
    solver.limit_radius = np.ones(7)
    solver.position_tolerance, solver.rotation_tolerance = .02, .1
    solver.constraints = SimpleNamespace(collision_model=Mock(), endpoint_rejection=Mock(return_value=None))
    solver.compute_eef_pose_base = Mock(return_value=Pose(np.zeros(3), [1., 0, 0, 0]))
    return solver


@pytest.mark.parametrize('raw,error,collision,accepted', [
    (np.zeros(7), 0., None, True), (np.zeros(7), .1, None, False),
    (np.zeros(7), 0., 'collision', False), (np.ones(7)*10, 0., None, False),
    (None, 0., None, False),
])
def test_ik_uses_actual_candidate(ik_solver, raw, error, collision, accepted):
    solver = ik_solver
    solver.model.compute_IK_CLIK.return_value = raw, False, None
    solver.constraints.endpoint_rejection.return_value = collision
    target = Pose(np.zeros(3), np.array([1., 0, 0, 0]))
    solver.compute_eef_pose_base.return_value = Pose([error, 0, 0], target.q)
    candidate, _ = solver.try_clik(np.zeros(7), target, target, np.zeros(7), np.zeros(7), [])
    assert (candidate is not None and candidate.rejection_reason is None) == accepted
    assert solver.model.compute_IK_CLIK.call_args.kwargs['max_iter'] == 50


@pytest.mark.parametrize('raw', [np.full(7, np.nan), np.full(7, np.inf), np.zeros(6), np.zeros((7, 1))])
def test_ik_rejects_malformed_backend_output(ik_solver, raw):
    ik_solver.model.compute_IK_CLIK.return_value = raw, False, None
    with pytest.raises(ValueError, match='CLIK output'):
        ik_solver.solve(Pose(np.zeros(3), [1., 0, 0, 0]), np.zeros(7), np.zeros(7), hand_qpos=np.zeros(12))


def test_ik_solve_accepts_unconverged_candidate(ik_solver):
    hand = np.ones(12)
    result = ik_solver.solve(Pose(np.zeros(3), [1., 0, 0, 0]), np.zeros(7), np.zeros(7), hand_qpos=hand)
    np.testing.assert_array_equal(result.qpos, np.zeros(7))
    ik_solver.model.compute_IK_CLIK.assert_called_once()
    ik_solver.constraints.collision_model.set_hand_qpos.assert_called_once_with(hand)


def test_ik_no_candidate_is_distinct_from_backend_failure(ik_solver):
    target = Pose(np.zeros(3), [1., 0, 0, 0])
    ik_solver.model.compute_IK_CLIK.return_value = None, False, None
    result = ik_solver.solve(target, np.zeros(7), np.zeros(7), hand_qpos=np.zeros(12))
    assert result.qpos is None
    assert result.reason == 'no_valid_candidate: solver_no_candidate'
    ik_solver.model.compute_IK_CLIK.side_effect = RuntimeError('backend failed')
    with pytest.raises(RuntimeError, match='backend failed'):
        ik_solver.solve(target, np.zeros(7), np.zeros(7), hand_qpos=np.zeros(12))


def test_joint_mapping_keeps_projection_without_loading_geometry():
    hand = HandConfig()
    lower, upper = np.full(7, -2*np.pi), np.full(7, 2*np.pi)
    lower[1], upper[1] = -1., 1.
    mapper = ActionMapper.for_mode(
        'joint', arm=SimpleNamespace(joint_limit_lower=lower, joint_limit_upper=upper),
        hand=hand, motion=None, environment=None,
    )
    arm = np.zeros(7)
    arm[:2] = 2*np.pi+.2, 3.
    result = mapper.map_joint(arm, np.full(12, 10.), np.zeros(7))
    assert mapper.solver is None
    np.testing.assert_allclose(result.arm_qpos[:2], [.2, 1.])
    np.testing.assert_array_equal(result.hand_qpos, hand.qpos_max_rad)




def test_heading_spread_does_not_block_save(tmp_path):
    import json

    from dexmani_real.utils.vr_heading import estimate_heading, save_vr_transform
    angles = np.deg2rad(np.linspace(-20, 20, 60))
    calibration = estimate_heading(np.column_stack((np.cos(angles), np.sin(angles), np.zeros(60))))
    path = tmp_path/'heading.json'
    save_vr_transform(calibration, path)
    assert set(json.loads(path.read_text())) == {'theta_deg'}
    with pytest.raises(ValueError):
        estimate_heading(np.zeros((60, 3)))


@pytest.mark.parametrize('outside', [True, False])
def test_jog_submission_keeps_measured_hold_and_commit_semantics(monkeypatch, outside):
    from dexmani_real.config.teleop import KeyboardTeleopConfig
    from dexmani_real.runtime import jog

    now_ns = 1000000000
    monkeypatch.setattr(jog.time, 'monotonic_ns', lambda: now_ns)
    row = SimpleNamespace(arm={'qpos': [np.zeros(7)], 'timestamp_ns': [now_ns]}, hand=None)
    monkeypatch.setattr(jog, 'read_observation', lambda *args: row)
    solver = Mock()
    solver.compute_eef_pose_base.return_value = Pose([.4, 0, .501 if outside else .3], [1., 0, 0, 0])
    solver.solve.return_value = IKResult(qpos=np.full(7, .1))
    controller = jog.CartesianJog(
        solver, KeyboardTeleopConfig(),
        (np.array([.255, -.495, .055]), np.array([.715, .495, .495])),
    )
    robot = Mock()
    keys = Mock(snapshot=Mock(return_value=frozenset({'j'})))
    state = SimpleNamespace(motion_lock=threading.Lock(), requests_allow_motion=lambda: True,
                            stop_signal=threading.Event())
    runtime = SimpleNamespace(arm=SimpleNamespace(feedback_max_age_s=.5),
                              hand=SimpleNamespace(enabled=False, home_qpos_deg=np.zeros(12)))
    if outside:
        controller.step(robot, keys, state, runtime)
        command = robot.send_action.call_args.args[0]
        assert command.arm_hold
        assert command.arm_qpos is None
        solver.solve.assert_not_called()
    else:
        robot.send_action.side_effect = RuntimeError('submission failed')
        with pytest.raises(RuntimeError, match='submission failed'):
            controller.step(robot, keys, state, runtime)
    assert controller.command_pose is None
    assert controller.command_qpos is None


def test_wrist_combined_rotation_and_uncommitted_smoothing():
    from dataclasses import replace

    from scipy.spatial.transform import Rotation

    config = replace(TeleopConfig(), ema_alpha_pos=.7, ema_alpha_rot=.6)
    base_rotation = Rotation.from_rotvec([.3, -.2, .4])
    wrist_start = Rotation.from_rotvec([-.4, .1, .2])
    eef_start = Rotation.from_rotvec([.2, .5, -.1])
    measured = Pose([.4, 0, .3], eef_start.as_quat()[[3, 0, 1, 2]])
    mapper = Mock()
    mapper.solver.constraints.workspace = np.array([[0, 1], [-1, 1], [0, 1]])
    mapper.solver.compute_eef_pose_base.return_value = measured
    row = SimpleNamespace(arm={'qpos': [np.zeros(7)]}, vr={
        'wrist_pos': np.array([.1, .2, .3]),
        'wrist_quat_wxyz': wrist_start.as_quat()[[3, 0, 1, 2]],
    })
    controller = VRController(mapper, None, config, base_rotation.as_matrix())
    controller.reset(row)
    zero = controller._wrist_target(row.vr['wrist_pos'], row.vr['wrist_quat_wxyz'])
    np.testing.assert_allclose(zero.p, measured.p)
    np.testing.assert_allclose(Rotation.from_quat(zero.q[[1, 2, 3, 0]]).as_matrix(), eef_start.as_matrix())
    wrist = Rotation.from_rotvec([.5, -.3, .7])
    moved = row.vr['wrist_pos'] + [.03, -.01, .02]
    desired = base_rotation * wrist * wrist_start.inv() * base_rotation.inv() * eef_start
    expected = eef_start * Rotation.from_rotvec(.6 * (eef_start.inv() * desired).as_rotvec())
    first = controller._wrist_target(moved, wrist.as_quat()[[3, 0, 1, 2]])
    second = controller._wrist_target(moved, wrist.as_quat()[[3, 0, 1, 2]])
    np.testing.assert_allclose(first.p, measured.p + .7 * config.vr_pos_scale * base_rotation.apply([.03, -.01, .02]))
    np.testing.assert_allclose(Rotation.from_quat(first.q[[1, 2, 3, 0]]).as_matrix(), expected.as_matrix())
    np.testing.assert_array_equal(first.p, second.p)
    np.testing.assert_array_equal(first.q, second.q)
    assert controller.previous_eef_target is measured

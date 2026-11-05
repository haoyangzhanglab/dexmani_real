import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.teleop import TeleopConfig
from dexmani_real.config.xarm7 import ArmConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.control.actions import ActionMapper, StartPoseRejected
from dexmani_real.control.constraints import joint_target_rejection, wrap_nearest_equivalent
from dexmani_real.control.ik import IKResult, IKSolver
from dexmani_real.geometry.pose import Pose, compute_pose_error
from dexmani_real.robot.commands import (
    hand_reference_from_feedback,
    validate_arm_target,
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


@pytest.mark.parametrize('offset,accepted', [(-5e-10, True), (5e-10, True), (2e-9, False)])
def test_vr_start_keeps_existing_roundoff_allowance(offset, accepted):
    solver = Mock()
    solver.constraints.workspace = np.array([[0, 1], [-1, 1], [0, .5]])
    solver.compute_eef_pose_base.return_value = Pose([.4, 0, .5+offset], [1., 0, 0, 0])
    controller = VRController(SimpleNamespace(solver=solver), None, TeleopConfig(), np.eye(3))
    row = SimpleNamespace(arm={'qpos': [np.zeros(7)]}, vr={
        'wrist_pos': np.zeros(3), 'wrist_quat_wxyz': np.array([1., 0, 0, 0]),
    })
    if accepted:
        controller.reset(row)
        assert controller.previous_eef_target.p[2] == .5+offset
    else:
        with pytest.raises(StartPoseRejected):
            controller.reset(row)
        assert controller.anchor is None


@pytest.fixture
def ik_solver():
    solver = IKSolver.__new__(IKSolver)
    solver.compute_clik = Mock(return_value=np.zeros(7))
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
])
def test_ik_uses_actual_candidate(ik_solver, raw, error, collision, accepted):
    solver = ik_solver
    solver.compute_clik.return_value = raw
    solver.constraints.endpoint_rejection.return_value = collision
    target = Pose(np.zeros(3), np.array([1., 0, 0, 0]))
    solver.compute_eef_pose_base.return_value = Pose([error, 0, 0], target.q)
    candidate, _ = solver.try_clik(np.zeros(7), target, np.zeros(7), np.zeros(7))
    assert (candidate is not None and candidate.rejection_reason is None) == accepted
    solver.compute_clik.assert_called_once()
    assert solver.compute_clik.call_args.args[1] is target


@pytest.mark.parametrize('raw', [np.full(7, np.nan), np.full(7, np.inf)])
def test_ik_rejects_nonfinite_iterate(ik_solver, raw):
    ik_solver.compute_clik.return_value = raw
    with pytest.raises(ValueError, match='IK iterate is non-finite'):
        ik_solver.solve(Pose(np.zeros(3), [1., 0, 0, 0]), np.zeros(7), np.zeros(7), hand_qpos=np.zeros(12))
    ik_solver.constraints.endpoint_rejection.assert_not_called()


def test_ik_fast_accepts_valid_previous_seed(ik_solver):
    hand = np.ones(12)
    result = ik_solver.solve(Pose(np.zeros(3), [1., 0, 0, 0]), np.full(7, .01), np.zeros(7), hand_qpos=hand)
    np.testing.assert_array_equal(result.qpos, np.zeros(7))
    ik_solver.compute_clik.assert_called_once()
    ik_solver.constraints.collision_model.set_hand_qpos.assert_called_once_with(hand)


def test_ik_checks_valid_neighbour_after_colliding_candidate(ik_solver):
    previous, current = np.zeros(7), np.full(7, 5e-5)
    ik_solver.compute_clik.side_effect = [previous.copy(), current.copy()]
    ik_solver.constraints.endpoint_rejection.side_effect = ['self_collision', None]
    # 禁止额外探索绕过第二个候选被误丢的问题。
    ik_solver.nullspace_direction = Mock(return_value=None)

    result = ik_solver.solve(
        Pose(np.zeros(3), [1., 0, 0, 0]), current, previous, hand_qpos=np.zeros(12),
    )

    assert ik_solver.compute_clik.call_count == 2
    for call, seed in zip(ik_solver.compute_clik.call_args_list, (previous, current)):
        np.testing.assert_array_equal(call.args[0], seed)
    assert ik_solver.compute_eef_pose_base.call_count == 2
    assert ik_solver.constraints.endpoint_rejection.call_count == 2
    np.testing.assert_array_equal(ik_solver.constraints.endpoint_rejection.call_args.args[0], current)
    np.testing.assert_array_equal(result.qpos, current)
    ik_solver.nullspace_direction.assert_not_called()


@pytest.mark.parametrize('lower,upper,raw_angle,boundary', [
    (-2*np.pi, .2, .2 - 2*np.pi, .2),
    (-.2, 2*np.pi, 2*np.pi - .2, -.2),
])
def test_ik_accepts_equivalent_angle_at_strict_limits(ik_solver, lower, upper, raw_angle, boundary):
    ik_solver.limits[0] = lower, upper
    ik_solver.joint_ranges = np.diff(ik_solver.limits, axis=1).ravel()
    ik_solver.limit_center = ik_solver.limits.mean(axis=1)
    ik_solver.limit_radius = ik_solver.joint_ranges / 2
    raw = np.zeros(7)
    raw[0] = raw_angle
    current = np.zeros(7)
    wrapped = wrap_nearest_equivalent(raw, current, *ik_solver.limits.T)

    assert np.all(wrapped >= ik_solver.limits[:, 0])
    assert np.all(wrapped <= ik_solver.limits[:, 1])
    assert wrapped[0] == boundary
    assert joint_target_rejection(wrapped, ik_solver.limits) is None
    validate_arm_target(wrapped, *ik_solver.limits.T)

    ik_solver.compute_clik.return_value = raw
    ik_solver.nullspace_direction = Mock(return_value=None)
    result = ik_solver.solve(
        Pose(np.zeros(3), [1., 0, 0, 0]), current, current, hand_qpos=np.zeros(12),
    )
    np.testing.assert_array_equal(result.qpos, wrapped)
    ik_solver.constraints.endpoint_rejection.assert_called_once()


@pytest.mark.parametrize('joint', [0, 1], ids=['no_equivalent_branch', 'non_equivalent_joint'])
def test_ik_preserves_and_rejects_unwrappable_angle(ik_solver, joint):
    raw = np.zeros(7)
    raw[joint] = np.nextafter(1., np.inf)
    current = np.zeros(7)
    wrapped = wrap_nearest_equivalent(raw, current, *ik_solver.limits.T)
    np.testing.assert_array_equal(wrapped, raw)
    assert joint_target_rejection(wrapped, ik_solver.limits) == 'operational_limits'
    with pytest.raises(ValueError, match='arm target violates joint limits'):
        validate_arm_target(raw, *ik_solver.limits.T)

    ik_solver.compute_clik.return_value = raw
    result = ik_solver.solve(
        Pose(np.zeros(3), [1., 0, 0, 0]), current, current, hand_qpos=np.zeros(12),
    )
    assert result.qpos is None
    assert result.reason == 'no_valid_candidate: operational_limits'
    ik_solver.compute_clik.assert_called_once()
    ik_solver.compute_eef_pose_base.assert_not_called()
    ik_solver.constraints.endpoint_rejection.assert_not_called()


def test_ik_propagates_clik_errors(ik_solver):
    target = Pose(np.zeros(3), [1., 0, 0, 0])
    ik_solver.compute_clik.side_effect = np.linalg.LinAlgError('factorization failed')
    with pytest.raises(np.linalg.LinAlgError, match='factorization failed'):
        ik_solver.solve(target, np.zeros(7), np.zeros(7), hand_qpos=np.zeros(12))


@pytest.fixture
def native_ik_solver():
    pytest.importorskip('pinocchio')
    arm = ArmConfig()
    constraints = SimpleNamespace(
        limits=np.column_stack((arm.joint_limit_lower, arm.joint_limit_upper)),
        collision_model=Mock(), endpoint_rejection=Mock(return_value=None),
    )
    return IKSolver(constraints, position_tolerance=.002, rotation_tolerance=np.deg2rad(2))


@pytest.mark.parametrize('qpos', [
    ArmConfig().home_qpos, np.zeros(7), [.4, -.3, .6, 1., -2., .8, -.5],
], ids=['home', 'singular', 'rotated'])
def test_native_eef_jacobian_matches_pose_difference(native_ik_solver, qpos):
    from scipy.spatial.transform import Rotation

    solver = native_ik_solver
    qpos = np.asarray(qpos)
    jacobian = solver.compute_eef_jacobian_base(qpos)
    finite_difference = np.empty((6, 7))
    for joint in range(7):
        delta = np.eye(7)[joint] * 1e-6
        plus = solver.compute_eef_pose_base(qpos + delta)
        minus = solver.compute_eef_pose_base(qpos - delta)
        finite_difference[:3, joint] = (plus.p - minus.p) / 2e-6
        relative = (Rotation.from_quat(plus.q[[1, 2, 3, 0]])
                    * Rotation.from_quat(minus.q[[1, 2, 3, 0]]).inv())
        finite_difference[3:, joint] = relative.as_rotvec() / 2e-6
    np.testing.assert_allclose(jacobian, finite_difference, atol=1e-8, rtol=0)


def test_native_clik_accepts_final_iterate_within_project_tolerance(native_ik_solver, monkeypatch):
    solver = native_ik_solver
    current = np.array(ArmConfig().home_qpos)
    saved = current.copy()
    target = solver.compute_eef_pose_base(current)
    target.p[0] += .008
    integrate = Mock(wraps=solver.fk.pin.integrate)
    monkeypatch.setattr(solver.fk.pin, 'integrate', integrate)

    result = solver.solve(target, current, current, hand_qpos=np.zeros(12))

    assert result.qpos is not None
    assert integrate.call_count == 50
    np.testing.assert_array_equal(current, saved)
    position_error, rotation_error = compute_pose_error(target, solver.compute_eef_pose_base(result.qpos))
    # 达到迭代预算时仍验收实际位姿，不把内部 1e-5 停止阈值当作成功标准。
    assert 1e-5 < position_error <= solver.position_tolerance
    assert rotation_error <= solver.rotation_tolerance
    solver.constraints.endpoint_rejection.assert_called_once()


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


@pytest.fixture
def jog_session(monkeypatch):
    from dataclasses import replace

    from dexmani_real.config.real import RealConfig
    from dexmani_real.control import constraints
    from dexmani_real.runtime import jog
    from dexmani_real.runtime.preflight import jog_bounds

    runtime = RealConfig()
    runtime = replace(runtime, hand=replace(runtime.hand, enabled=False))
    now_ns = 1_000_000_000
    monkeypatch.setattr(jog.time, 'monotonic_ns', lambda: now_ns)
    row = SimpleNamespace(arm={'qpos': [np.zeros(7)], 'timestamp_ns': [now_ns]}, hand=None)
    monkeypatch.setattr(jog, 'read_observation', lambda *args: row)
    collision = Mock(rejection=Mock(return_value=None))
    monkeypatch.setattr(constraints, 'CollisionModel', lambda **kwargs: collision)
    solver = Mock(constraints=constraints.MotionConstraints(runtime.arm, runtime.environment))
    solver.solve.return_value = IKResult(qpos=np.full(7, .1))
    controller = jog.CartesianJog(solver, runtime.keyboard_teleop, jog_bounds(runtime))
    keys = Mock(snapshot=Mock(return_value=frozenset()))
    state = SimpleNamespace(motion_lock=threading.Lock(), requests_allow_motion=lambda: True,
                            stop_signal=threading.Event())
    return SimpleNamespace(controller=controller, solver=solver, collision=collision,
                           robot=Mock(), keys=keys, state=state, runtime=runtime, row=row)


@pytest.mark.parametrize('z,key,delta,expected,solved', [
    (.494, 'up', .008, [.4, 0, .495], True),
    (.056, 'down', .008, [.4, 0, .055], True),
    (.496, 'down', .008, [.4, 0, .488], True),
    (.496, 'down', .0005, [.4, 0, .4955], True),
    (.496, 'up', .008, [.4, 0, .496], False),
    (.496, 'j', .008, [.4, 0, .496], True),
    (.496, 'w', .008, [.408, 0, .496], True),
    (.054, 'up', .008, [.4, 0, .062], True),
    (.054, 'down', .008, [.4, 0, .054], False),
    (.054, 'j', .008, [.4, 0, .054], True),
    (.054, 'w', .008, [.408, 0, .054], True),
])
def test_jog_boundary_reentry_after_release(jog_session, z, key, delta, expected, solved):
    from dataclasses import replace

    s = jog_session
    controller = s.controller
    controller.config = replace(controller.config, delta_pos_m=delta)
    previous = Pose([.4, 0, .495 if z > .3 else .055], [1., 0, 0, 0])
    controller.command_pose, controller.command_qpos = previous, np.full(7, .05)
    measured = Pose([.4, 0, z], [1., 0, 0, 0])
    s.solver.compute_eef_pose_base.return_value = measured
    s.collision.eef_position.return_value = measured.p

    # 走真实松键分支：先发布减速保持，再清除上次目标参考。
    def accept_hold(command, **kwargs):
        assert command.arm_hold
        assert controller.command_pose is previous

    s.robot.send_action.side_effect = accept_hold
    controller.step(s.robot, s.keys, s.state, s.runtime)
    assert controller.command_pose is None
    assert controller.command_qpos is None
    s.solver.solve.assert_not_called()

    s.keys.snapshot.return_value = frozenset({key})

    def accept_target(command, **kwargs):
        assert not command.arm_hold
        assert controller.command_pose is None
        assert controller.command_qpos is None

    s.robot.send_action.side_effect = accept_target
    controller.step(s.robot, s.keys, s.state, s.runtime)
    np.testing.assert_allclose(controller.command_pose.p, expected, atol=1e-15, rtol=0)
    assert s.robot.send_action.call_count == 2
    assert s.solver.solve.called == solved
    s.solver.compute_eef_pose_base.assert_called_once_with(s.row.arm['qpos'][0])
    if solved:
        args = s.solver.solve.call_args.args
        np.testing.assert_array_equal(args[0].p, controller.command_pose.p)
        np.testing.assert_array_equal(args[2], s.row.arm['qpos'][0])
    else:
        # 向外饱和仍保留原位目标，经过现有端点碰撞检查。
        s.collision.rejection.assert_called_once_with(s.row.arm['qpos'][0])
    command = s.robot.send_action.call_args.args[0]
    np.testing.assert_array_equal(controller.command_qpos, command.arm_qpos)
    if key in {'up', 'down'}:
        np.testing.assert_array_equal(controller.command_pose.p[:2], measured.p[:2])
    else:
        assert controller.command_pose.p[2] == z


@pytest.mark.parametrize('side', [0, 1])
def test_jog_accepts_exact_endpoint_workspace_boundary(jog_session, side):
    s = jog_session
    z = s.solver.constraints.workspace_with_tolerance[2, side]
    s.solver.compute_eef_pose_base.return_value = Pose([.4, 0, z], [1., 0, 0, 0])
    s.keys.snapshot.return_value = frozenset({'up' if side == 0 else 'down'})
    s.controller.step(s.robot, s.keys, s.state, s.runtime)
    target = s.solver.solve.call_args.args[0]
    assert target.p[2] > z if side == 0 else target.p[2] < z
    np.testing.assert_array_equal(s.controller.command_pose.p, target.p)
    assert not s.robot.send_action.call_args.args[0].arm_hold


@pytest.mark.parametrize('side', [0, 1])
@pytest.mark.parametrize('offset', [0., .001])
def test_jog_start_outside_endpoint_workspace_brakes(jog_session, side, offset):
    s = jog_session
    boundary = s.solver.constraints.workspace_with_tolerance[2, side]
    z = np.nextafter(boundary - offset if side == 0 else boundary + offset,
                     -np.inf if side == 0 else np.inf)
    s.solver.compute_eef_pose_base.return_value = Pose([.4, 0, z], [1., 0, 0, 0])
    assert s.solver.constraints.endpoint_rejection(
        s.row.arm['qpos'][0], position=np.array([.4, 0, z]),
    ) == 'final_eef_workspace'
    s.keys.snapshot.return_value = frozenset({'up' if z < .3 else 'down'})
    s.controller.step(s.robot, s.keys, s.state, s.runtime)
    assert s.robot.send_action.call_args.args[0].arm_hold
    s.solver.solve.assert_not_called()
    assert s.controller.command_pose is None
    assert s.controller.command_qpos is None


@pytest.mark.parametrize('committed', [False, True])
def test_jog_submission_failure_preserves_reference(jog_session, committed):
    s = jog_session
    previous_pose = Pose([.4, 0, .495], [1., 0, 0, 0]) if committed else None
    previous_qpos = np.full(7, .05) if committed else None
    s.controller.command_pose, s.controller.command_qpos = previous_pose, previous_qpos
    s.solver.compute_eef_pose_base.return_value = Pose([.4, 0, .496], [1., 0, 0, 0])
    s.keys.snapshot.return_value = frozenset({'down'})
    s.robot.send_action.side_effect = RuntimeError('submission failed')
    with pytest.raises(RuntimeError, match='submission failed'):
        s.controller.step(s.robot, s.keys, s.state, s.runtime)
    assert s.solver.solve.call_args.args[0].p[2] < .495
    assert s.controller.command_pose is previous_pose
    assert s.controller.command_qpos is previous_qpos
    if committed:
        np.testing.assert_array_equal(previous_pose.p, [.4, 0, .495])
        np.testing.assert_array_equal(previous_pose.q, [1., 0, 0, 0])


@pytest.mark.parametrize('dx,drpy', [([0., 0, 0], [0., 0, 0]), ([0., 0, 0], [.03, 0, 0])])
def test_jog_zero_translation_keeps_position_exactly(dx, drpy):
    from dexmani_real.runtime.jog import jog_pose

    start = Pose([.254, -.496, .496], [1., 0, 0, 0])
    pose, changed = jog_pose(start, np.array(dx), np.array(drpy),
                             np.array([.255, -.495, .055]), np.array([.715, .495, .495]))
    np.testing.assert_array_equal(pose.p, start.p)
    assert changed == bool(np.any(drpy))


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

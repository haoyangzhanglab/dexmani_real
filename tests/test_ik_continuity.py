from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.control.constraints import MotionConstraints
from dexmani_real.control.ik import IKReferenceError, IKSolver
from dexmani_real.geometry.pose import Pose
from dexmani_real.runtime.ik_reference import IKReference, prepare_reference

CASES = [
    (
        [
            6.159032380827766,
            1.346043214121702,
            -3.639382182611885,
            2.814915583600824,
            -1.4776921890261265,
            1.6187680844795305,
            -5.018550574121533,
        ],
        [0.6948947480914032, 0.09361343198041858, 0.46305470063391363],
        [
            0.6164158058017314,
            -0.43053084039190137,
            -0.11466317739336054,
            0.6492511883541717,
        ],
    ),
    (
        [0.0, 0.041888, 6.2830853071795865, 0.417134, -3.138102, 1.195551, 0.0],
        [0.3965039328933116, 0.0014100915242789296, 0.1891091048698622],
        [
            0.9999983975263151,
            0.0006394003867268489,
            -0.0000022005334708664135,
            -0.0016721564235718574,
        ],
    ),
]


@pytest.fixture
def solver():
    cfg = RealConfig()
    return IKSolver(
        MotionConstraints(cfg.arm, cfg.environment),
        target_speed=np.deg2rad(75),
        device_speed=cfg.arm.max_joint_velocity_rad_per_s,
        max_lead=np.deg2rad(10),
    )


def hand():
    return np.deg2rad(RealConfig().hand.home_qpos_deg)


@pytest.fixture
def reference_solver():
    solver = IKSolver.__new__(IKSolver)
    solver.limits = np.tile([-1.0, 1.0], (7, 1))
    return solver


@pytest.mark.parametrize("control_ns", [0, 1, 100_000_000, 10_000_000_000])
@pytest.mark.parametrize("period_s", [1 / 30, 1 / 60])
def test_first_reference_uses_nominal_period(reference_solver, control_ns, period_s):
    q = np.zeros(7)
    q[0] = 1 + 5e-7
    measured, command, anchor, h = prepare_reference(
        reference_solver, None, q, control_ns=control_ns, period_s=period_s
    )
    assert h == period_s
    assert command is measured and anchor is measured
    assert measured[0] == 1 and q[0] > 1


@pytest.mark.parametrize("elapsed", [-1, 0, 1_000_000, 100_000_000, 10_000_000_000])
def test_committed_reference_budget(reference_solver, elapsed):
    q = np.zeros(7)
    ref = IKReference(q + 0.01, q - 0.01, 100)
    if elapsed < 0:
        with pytest.raises(ValueError, match="clock moved backwards"):
            prepare_reference(
                reference_solver, ref, q, control_ns=100 + elapsed, period_s=1 / 30
            )
    else:
        measured, command, anchor, h = prepare_reference(
            reference_solver, ref, q, control_ns=100 + elapsed, period_s=1 / 30
        )
        np.testing.assert_array_equal(measured, q)
        assert command is ref.command and anchor is ref.anchor
        assert h == min(elapsed * 1e-9, 1 / 30)
    assert ref.submitted_ns == 100
    np.testing.assert_array_equal(ref.command, q + 0.01)
    np.testing.assert_array_equal(ref.anchor, q - 0.01)


@pytest.mark.parametrize("committed", [False, True])
@pytest.mark.parametrize("invalid", ["outside", "nan", "shape"])
def test_prepare_reference_validates_measured(reference_solver, committed, invalid):
    q = np.zeros(7)
    ref = IKReference(q.copy(), q.copy(), 100) if committed else None
    q[0] = 1.01 if invalid == "outside" else np.nan
    if invalid == "shape":
        q = np.zeros(6)
    with pytest.raises(IKReferenceError if invalid == "outside" else ValueError):
        prepare_reference(reference_solver, ref, q, control_ns=101, period_s=1 / 30)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("hz", [16, 30, 60])
def test_static_continuity(solver, case, hz):
    anchor = np.array(case[0])
    q = anchor.copy()
    target = Pose(*case[1:])
    tail = []
    for i in range(600):
        result = solver.solve(target, q, q, anchor, h=1 / hz, hand_qpos=hand())
        assert result.qpos is not None, result.reason
        delta = np.max(np.abs(result.qpos - q))
        assert delta <= solver.speed / hz + 1e-12
        if i > 500:
            tail.append(delta)
        q = result.qpos
    assert max(tail) < 1e-4
    assert np.max(np.abs(q - anchor)) < np.deg2rad(10)


@pytest.mark.parametrize("case", CASES)
def test_analytic_jacobian(solver, case):
    q = np.array(case[0])
    target = Pose(*case[1:])
    pin = solver.fk.pin
    goal = pin.SE3(pin.Quaternion(*target.q).matrix(), target.p)
    fun = lambda x: solver.residual_and_jacobian(x, goal, q, 0.04)
    _, jac = fun(q)
    eps = 1e-6
    numeric = np.column_stack(
        [(fun(q + eps * d)[0] - fun(q - eps * d)[0]) / (2 * eps) for d in np.eye(7)]
    )
    np.testing.assert_allclose(jac, numeric, atol=1e-7, rtol=1e-6)


def test_frozen_feedback_activates_lead(solver):
    measured = np.array(RealConfig().arm.home_qpos)
    q = measured.copy()
    target = solver.compute_eef_pose_base(measured)
    target.p[1] += 0.3
    # 隔离碰撞仅用于明确激活数值边界。
    solver.constraints.endpoint_rejection = Mock(return_value=None)
    hit = False
    for _ in range(160):
        new = solver.solve(
            target, measured, q, measured, h=1 / 30, hand_qpos=hand()
        ).qpos
        assert np.max(np.abs(new - q)) <= solver.speed / 30 + 1e-12
        assert np.max(np.abs(new - measured)) <= solver.lead + 1e-12
        hit |= np.max(np.abs(new - measured)) > solver.lead - 1e-5
        q = new
    assert hit
    with pytest.raises(IKReferenceError):
        solver.solve(
            target,
            measured - np.sign(q - measured) * 0.02,
            q,
            measured,
            h=1 / 30,
            hand_qpos=hand(),
        )


@pytest.mark.parametrize("h", [0, 0.02])
@pytest.mark.parametrize("reason", ["environment_collision", "final_eef_workspace"])
def test_actual_endpoint_rejection(solver, h, reason):
    from dexmani_real.config.control import StaticCollisionBox
    from dexmani_real.control.collision import CollisionModel

    q = np.array(RealConfig().arm.home_qpos)
    pose = solver.compute_eef_pose_base(q)
    assert solver.solve(pose, q, q, q, h=h, hand_qpos=hand()).qpos is not None
    if reason == "environment_collision":
        solver.constraints.collision_model = CollisionModel(
            static_boxes=[StaticCollisionBox(tuple(pose.p), (0.5, 0.5, 0.5))]
        )
    else:
        solver.constraints.workspace_with_tolerance[0, 1] = pose.p[0] - 0.01
    result = solver.solve(pose, q, q, q, h=h, hand_qpos=hand())
    assert result.qpos is None and result.reason.startswith(reason)
    np.testing.assert_array_equal(solver.constraints.collision_model.hand_qpos, hand())


def test_jog_reference_commit_failure_rejection_and_restart(reference_solver, monkeypatch):
    import threading

    from dexmani_real.control.ik import IKResult
    from dexmani_real.robot.commands import RobotError
    from dexmani_real.runtime.jog import CartesianJog

    cfg = RealConfig()
    q = np.zeros(7)
    workspace = np.tile([-1.0, 1.0], (3, 1))
    reference_solver.constraints = SimpleNamespace(workspace_with_tolerance=workspace)
    reference_solver.compute_eef_pose_base = Mock(return_value=Pose(np.zeros(3), [1, 0, 0, 0]))
    reference_solver.solve = Mock(return_value=IKResult(qpos=q + 0.01))
    jog = CartesianJog(reference_solver, cfg.keyboard_teleop, workspace.T)
    clock = [100_000_000]
    obs = SimpleNamespace(arm={"qpos": [q], "timestamp_ns": [clock[0]]}, hand=None)
    monkeypatch.setattr("dexmani_real.runtime.jog.read_observation", lambda *a: obs)
    monkeypatch.setattr("dexmani_real.runtime.jog.check_device_feedback", lambda *a: [])
    monkeypatch.setattr("dexmani_real.runtime.jog.time.monotonic_ns", lambda: clock[0])
    robot, keys = Mock(), Mock()
    keys.snapshot.return_value = {"w"}
    state = SimpleNamespace(
        motion_lock=threading.RLock(), stop_signal=threading.Event(),
        requests_allow_motion=lambda: True,
    )

    jog.propose({"w"}, q, hand_qpos=hand(), control_ns=clock[0])
    assert jog.ik_reference is None
    assert reference_solver.solve.call_args.kwargs["h"] == 1 / cfg.keyboard_teleop.control_hz
    jog.step(robot, keys, state, cfg)
    ref = jog.ik_reference
    assert ref.submitted_ns == clock[0]
    np.testing.assert_array_equal(ref.command, q + 0.01)
    np.testing.assert_array_equal(ref.anchor, q)

    clock[0] += 1_000_000
    robot.send_action.side_effect = RobotError("submission failed")
    with pytest.raises(RobotError):
        jog.step(robot, keys, state, cfg)
    assert jog.ik_reference is ref
    assert reference_solver.solve.call_args.kwargs["h"] == 0.001

    reference_solver.solve.return_value = IKResult(qpos=None, reason="collision")
    assert jog.propose({"w"}, q, hand_qpos=hand(), control_ns=clock[0]).qpos is None
    assert jog.ik_reference is ref
    robot.send_action.side_effect = None
    jog.step(robot, keys, state, cfg)
    assert robot.send_action.call_args.args[0].arm_hold
    assert jog.ik_reference is None

    reference_solver.solve.return_value = IKResult(qpos=q + 0.01)
    jog.propose({"w"}, q, hand_qpos=hand(), control_ns=clock[0])
    assert reference_solver.solve.call_args.kwargs["h"] == 1 / cfg.keyboard_teleop.control_hz
    assert jog.ik_reference is None


def test_budget_and_bad_candidate(solver, monkeypatch):
    import dexmani_real.control.ik as module

    q = np.array(RealConfig().arm.home_qpos)
    target = solver.compute_eef_pose_base(q)
    monkeypatch.setattr(
        module, "least_squares", lambda *a, **k: SimpleNamespace(x=q.copy(), status=0)
    )
    assert solver.solve(target, q, q, q, h=0.02, hand_qpos=hand()).qpos is not None
    monkeypatch.setattr(
        module, "least_squares", lambda *a, **k: SimpleNamespace(x=q + 10, status=0)
    )
    with pytest.raises(ValueError):
        solver.solve(target, q, q, q, h=0.02, hand_qpos=hand())


def test_closed_trajectory_and_noise(solver):
    anchor = np.array(RealConfig().arm.home_qpos)
    q = anchor.copy()
    origin = solver.compute_eef_pose_base(q)
    rng = np.random.default_rng(1)
    for angle in np.linspace(0, 4 * np.pi, 300):
        target = origin.copy()
        target.p += [0.01 * (1 - np.cos(angle)), 0.01 * np.sin(angle), 0]
        target.p += rng.normal(0, 1e-5, 3)
        new = solver.solve(target, q, q, anchor, h=1 / 30, hand_qpos=hand()).qpos
        assert new is not None
        assert np.max(np.abs(new - q)) <= solver.speed / 30 + 1e-12
        q = new


@pytest.mark.parametrize("side", [0, 1])
def test_actual_coordinates_near_limits(solver, side):
    anchor = np.array(RealConfig().arm.home_qpos)
    anchor[2] = solver.limits[2, side] + (1e-5 if side == 0 else -1e-5)
    q = anchor.copy()
    solver.constraints.endpoint_rejection = Mock(return_value=None)
    for delta in [0.001, -0.001] * 20:
        target = solver.compute_eef_pose_base(anchor)
        target.p[0] += delta
        new = solver.solve(target, q, q, anchor, h=0.02, hand_qpos=hand()).qpos
        assert np.max(np.abs(new - q)) <= solver.speed * 0.02 + 1e-12
        assert np.all(new >= solver.limits[:, 0]) and np.all(new <= solver.limits[:, 1])
        q = new


def test_jog_uses_accepted_fk_and_clears_on_hold(solver, monkeypatch):
    import threading

    from dexmani_real.runtime.jog import CartesianJog

    cfg = RealConfig()
    jog = CartesianJog(solver, cfg.keyboard_teleop, solver.constraints.workspace.T)
    q = np.array(cfg.arm.home_qpos)
    observation = SimpleNamespace(
        arm={"qpos": [q], "timestamp_ns": [1]}, hand={"qpos": [hand()]}
    )
    monkeypatch.setattr(
        "dexmani_real.runtime.jog.read_observation", lambda *a: observation
    )
    monkeypatch.setattr("dexmani_real.runtime.jog.check_device_feedback", lambda *a: [])
    monkeypatch.setattr("dexmani_real.runtime.jog.time.monotonic_ns", lambda: 100000000)
    keys = Mock()
    keys.snapshot.return_value = {"w"}
    robot = Mock()
    state = SimpleNamespace(
        motion_lock=threading.RLock(),
        stop_signal=threading.Event(),
        requests_allow_motion=lambda: True,
    )
    jog.step(robot, keys, state, cfg)
    command = robot.send_action.call_args.args[0]
    accepted_pose = solver.compute_eef_pose_base(command.arm_qpos)
    solve = Mock(wraps=solver.solve)
    monkeypatch.setattr(solver, "solve", solve)
    jog.propose({"w"}, q, hand_qpos=hand(), control_ns=200000000)
    np.testing.assert_allclose(
        solve.call_args.args[0].p,
        accepted_pose.p + [cfg.keyboard_teleop.delta_pos_m, 0, 0],
    )
    assert isinstance(jog.ik_reference, IKReference)
    keys.snapshot.return_value = set()
    jog.step(robot, keys, state, cfg)
    assert robot.send_action.call_args.args[0].arm_hold and jog.ik_reference is None


@pytest.mark.parametrize("speed", [float("inf"), float("nan"), 0.0, -1.0])
def test_invalid_speed_is_not_hidden_by_device_cap(solver, speed):
    with pytest.raises(ValueError, match="finite and positive"):
        IKSolver(solver.constraints, target_speed=speed, max_lead=0.1, device_speed=1.0)


def test_speed_cap_and_feedback_roundoff(solver):
    capped = IKSolver(
        solver.constraints, target_speed=2.0, max_lead=0.1, device_speed=0.5
    )
    assert capped.speed == 0.5
    q = np.array(RealConfig().arm.home_qpos)
    q[2] = solver.limits[2, 1] + 5e-7
    measured = solver.measured_reference(q)
    assert measured[2] == solver.limits[2, 1]
    assert q[2] > measured[2]
    from dexmani_real.robot.commands import validate_arm_target

    with pytest.raises(ValueError):
        validate_arm_target(q, *solver.limits.T)
    q[2] += 0.01
    with pytest.raises(IKReferenceError):
        solver.measured_reference(q)
    q[2] = np.nan
    with pytest.raises(ValueError):
        solver.measured_reference(q)


def test_jog_reference_failure_does_not_hold_or_reset(solver, monkeypatch):
    import threading

    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.runtime.jog import CartesianJog

    cfg = RealConfig()
    jog = CartesianJog(solver, cfg.keyboard_teleop, solver.constraints.workspace.T)
    q = np.array(cfg.arm.home_qpos)
    ref = IKReference(q + 0.3, q, 1)
    jog.ik_reference = ref
    obs = SimpleNamespace(
        arm={"qpos": [q], "timestamp_ns": [1]}, hand={"qpos": [hand()]}
    )
    monkeypatch.setattr("dexmani_real.runtime.jog.read_observation", lambda *a: obs)
    monkeypatch.setattr("dexmani_real.runtime.jog.check_device_feedback", lambda *a: [])
    robot = Mock()
    keys = Mock()
    keys.snapshot.return_value = {"w"}
    state = SimpleNamespace(
        motion_lock=threading.RLock(),
        requests_allow_motion=lambda: True,
        stop_signal=threading.Event(),
    )
    with pytest.raises(OperationCancelled):
        jog.step(robot, keys, state, cfg)
    robot.send_action.assert_not_called()
    assert jog.ik_reference is ref

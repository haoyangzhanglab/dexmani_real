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


def test_time_cap_and_zero_budget_geometry(solver):
    q = np.array(RealConfig().arm.home_qpos)
    reference = IKReference(q, q, 100)
    for elapsed in (1, 100_000_000, 10_000_000_000):
        args = prepare_reference(
            solver,
            reference,
            q,
            feedback_ns=100,
            control_ns=100 + elapsed,
            period_s=1 / 30,
        )
        assert args[-1] == min(elapsed * 1e-9, 1 / 30)
    solver.constraints.endpoint_rejection = Mock(return_value="collision")
    result = solver.solve(
        solver.compute_eef_pose_base(q), q, q, q, h=0, hand_qpos=hand()
    )
    assert result.qpos is None
    solver.constraints.endpoint_rejection.assert_called_once()


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

    from dexmani_real.runtime.ik_reference import IKReference
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
    jog.propose({"w"}, q, hand_qpos=hand(), feedback_ns=1, control_ns=200000000)
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
    from dexmani_real.runtime.jog import CartesianJog
    from dexmani_real.robot.commands import OperationCancelled

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

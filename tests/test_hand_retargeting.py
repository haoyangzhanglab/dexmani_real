"""XHand 机构几何、解析梯度及提交参考回归；无需设备或 episodes。"""

from dataclasses import replace

import nlopt
import numpy as np
import pytest

from dexmani_real.config.teleop import HandRetargetingConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.robot.model import XHAND_SDK_JOINT_NAMES
from dexmani_real.teleop.retargeting.landmarks import (
    HandCalibration,
    HandGeometryRejected,
    hand_lengths,
    project_direction,
    rotate_vectors,
)
from dexmani_real.teleop.retargeting.objective import HandObjective
from dexmani_real.teleop.retargeting.solver import HandRetargeter


@pytest.fixture
def hand():
    r = HandRetargeter(HandRetargetingConfig(), HandConfig())
    p = np.zeros((21, 3))
    g = r.geometry
    p[1] = r.fk.neutral_joint_geometry()[0][g.base_indices[0]]
    p[2:5] = g.positions[0]
    for i in range(1, 5):
        a, b, c = g.positions[i]
        p[1 + 4 * i : 5 + 4 * i] = [a, b, b + 0.4 * (c - b), c]
    r.calibrate([hand_lengths(p)] * r.config.calibration_samples)
    return r, p


def test_joint_axes_frames_and_sdk_order(hand):
    r, _ = hand
    fk = r.fk
    origins, axes = fk.neutral_joint_geometry()
    for i, name in enumerate(XHAND_SDK_JOINT_NAMES):
        jid = fk.model.getJointId(name)
        assert fk.urdf_to_sdk[i] == fk.model.joints[jid].idx_q
        fid = fk.model.getFrameId(name)
        poses = []
        for sign in [-1, 1]:
            q = np.zeros(12)
            q[i] = sign * 1e-6
            poses.append(fk.forward(q, [fid])[0])
        angular = fk.pin.log3(poses[1][:3, :3] @ poses[0][:3, :3].T) / 2e-6
        np.testing.assert_allclose(angular, axes[i], atol=1e-9)
        np.testing.assert_allclose(poses[0][:3, 3], origins[i], atol=1e-12)
    q = np.deg2rad(HandConfig().home_qpos_deg)
    pos, jac = fk.positions_and_jacobians(q, r.frame_ids)
    np.testing.assert_allclose(pos, fk.forward(q, r.frame_ids)[:, :3, 3])
    for i in range(12):
        dq = np.eye(12)[i] * 1e-6
        diff = (
            fk.forward(q + dq, r.frame_ids)[:, :3, 3]
            - fk.forward(q - dq, r.frame_ids)[:, :3, 3]
        ) / 2e-6
        np.testing.assert_allclose(diff, jac[:, :, i], atol=1e-8)
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(pos.reshape(5, 3, 3), axis=1), axis=2),
        r.geometry.segments,
        atol=1e-12,
    )


@pytest.mark.parametrize("angle", [-0.8, 0.12, 2.5])
def test_base_rotations_and_planes(hand, angle):
    r, p = hand
    g = r.geometry
    # 保持骨长，只旋转定义基部方位的骨段。
    for f in range(2):
        k = 1 + 4 * f
        v = p[k + 1] - p[k]
        changed = rotate_vectors(v[None], g.base_axes[f : f + 1], np.array([angle]))[0]
        if f == 0:
            p[k + 1 : k + 4] += changed - v
        else:
            p[k + 1 : k + 4] = p[k] + rotate_vectors(
                p[k + 1 : k + 4] - p[k], g.base_axes[f], np.array(angle),
            )
    t = r.calibration.targets(p, g, r.config, fk=r.fk, frame_ids=r.frame_ids)
    clipped = np.clip([angle, 0.5 * np.arctan2(np.sin(2*angle), np.cos(2*angle))], *g.limits[g.base_indices].T)
    q = np.zeros(12)
    q[[0, 3]] = clipped
    robot = r.fk.forward(q, r.frame_ids)[:, :3, 3].reshape(5, 3, 3)
    np.testing.assert_allclose(t.positions[:2, 0], robot[:2, 0], atol=1e-12)
    normals = g.flex_axes[:, 0].copy()
    normals[:2] = rotate_vectors(normals[:2], g.base_axes, clipped)
    np.testing.assert_allclose(
        np.sum(t.directions * normals[:, None], axis=2), 0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(np.diff(t.positions, axis=1), axis=2), g.segments, atol=1e-12
    )


def test_distal_calibration_spread_and_pinch(hand):
    r, p = hand
    g = r.geometry
    # 构造弯曲的远端两段，投影方向与固定长度应独立于瞬时骨长噪声。
    k = 9
    l = r.calibration.bone_lengths[2]
    p[k + 2] = p[k + 1] + l[1] * np.array([0.6, 0, 0.8])
    p[k + 3] = p[k + 2] + l[2] * np.array([1.0, 0, 0])
    t = r.calibration.targets(p, g, r.config, fk=r.fk, frame_ids=r.frame_ids)
    expected = project_direction(
        np.array([l[1] * 0.6 + l[2], 0, l[1] * 0.8]), g.flex_axes[2, 0]
    )
    np.testing.assert_allclose(
        (t.positions[2, 2] - t.positions[2, 1]) / g.segments[2, 1], expected
    )
    noisy = p.copy()  # 修改只作用于中指骨链。
    noisy[k + 2] = p[k + 1] + 1.1 * (p[k + 2] - p[k + 1])
    noisy[k + 3] = noisy[k + 2] + 0.9 * (p[k + 3] - p[k + 2])
    nt = r.calibration.targets(noisy, g, r.config, fk=r.fk, frame_ids=r.frame_ids)
    np.testing.assert_allclose(t.positions, nt.positions, atol=1e-12)
    for f in [2, 3, 4]:
        k = 1 + 4 * f
        bones = np.diff(p[k : k + 4], axis=0)
        spread = bones + np.linalg.norm(bones, axis=1)[:, None] * np.array([0, 0.2, 0])
        spread *= (
            np.linalg.norm(bones, axis=1)[:, None]
            / np.linalg.norm(spread, axis=1)[:, None]
        )
        p[k + 1 : k + 4] = p[k] + np.cumsum(spread, axis=0)
    spread_target = r.calibration.targets(p, g, r.config, fk=r.fk, frame_ids=r.frame_ids)
    np.testing.assert_allclose(t.positions, spread_target.positions, atol=1e-12)
    d = np.linalg.norm(t.positions[1:, 2] - t.positions[0, 2], axis=1)
    np.testing.assert_allclose(t.pinch_distances, d)
    # 对同一几何移动激活阈值，核查两个端点及过渡中点。
    for x in [0, 0.5, 1]:
        cfg = replace(
            r.config,
            pinch_start_m=float(d[0] + x * 0.01),
            pinch_full_m=float(d[0] - (1 - x) * 0.01),
        )
        assert r.calibration.targets(p, g, cfg, fk=r.fk, frame_ids=r.frame_ids).activation[0] == pytest.approx(
            x * x * (3 - 2 * x)
        )


def test_projection_and_calibration_rejections(hand):
    r, p = hand
    with pytest.raises(HandGeometryRejected, match="degenerate"):
        project_direction(np.array([0, 1, 0]), np.array([0, 1, 0]))
    bad = p.copy()
    bad[4] += 1
    with pytest.raises(HandGeometryRejected, match="tolerance"):
        r.calibration.targets(bad, r.geometry, r.config, fk=r.fk, frame_ids=r.frame_ids)
    with pytest.raises(ValueError, match="requires"):
        HandCalibration.from_measurements([hand_lengths(p)] * 24, r.config)
    samples = np.array([hand_lengths(p)] * 25)
    samples[:12] *= 0.6
    samples[13:] *= 1.4
    with pytest.raises(ValueError, match="unstable"):
        HandCalibration.from_measurements(samples, r.config)


@pytest.mark.parametrize(
    "term", ["position", "direction", "pinch", "temporal", "total"]
)
def test_analytical_gradient(hand, term):
    r, p = hand
    targets = r.calibration.targets(p, r.geometry, r.config, fk=r.fk, frame_ids=r.frame_ids)
    targets = replace(
        targets,
        activation=np.ones(4),
        pinch_distances=np.array([0.01, 0.03, 0.06, 0.09]),
    )
    weights = {
        "shape_weight": 0.0,
        "tip_weight": 0.0,
        "proximal_direction_weight": 0.0,
        "direction_weight": 0.0,
        "pinch_weight": 0.0,
        "damping_s": 0.0,
    }
    groups = {
        "position": ["shape_weight", "tip_weight"],
        "direction": ["proximal_direction_weight", "direction_weight"],
        "pinch": ["pinch_weight"],
        "temporal": ["damping_s"],
    }
    for key in weights:
        if term == "total" or key in groups[term]:
            weights[key] = getattr(r.config, key)
    cfg = replace(r.config, **weights)
    rng = np.random.default_rng(42)
    for _ in range(5):
        q = rng.uniform(*r.limits.T)
        previous = rng.uniform(*r.limits.T)
        obj = HandObjective(r.fk, r.frame_ids, targets, previous, 0.0625, cfg)
        grad = np.empty(12)
        value = obj(q, grad)
        assert value == pytest.approx(obj(q, np.empty(0)))
        numeric = np.array(
            [
                (obj(q + v, np.empty(0)) - obj(q - v, np.empty(0))) / 2e-6
                for v in np.eye(12) * 1e-6
            ]
        )
        np.testing.assert_allclose(grad, numeric, atol=1e-7, rtol=1e-5)


@pytest.mark.parametrize("tau", [0.0, 0.02])
def test_limits_rate_and_commit(hand, tau):
    r, p = hand
    r.config = replace(r.config, smoothing_tau_s=tau)
    q = np.deg2rad(HandConfig().home_qpos_deg)
    t = 1000000000
    r.reset(q, timestamp_ns=t)
    np.testing.assert_array_equal(r.retarget(p, timestamp_ns=t), q)
    with pytest.raises(ValueError, match="backwards"):
        r.retarget(p, timestamp_ns=t - 1)
    for dt in [0.001, 0.01, 0.0625, 0.2] * 3:
        stamp = t + int(dt * 1e9)
        command = r.retarget(p, timestamp_ns=stamp)
        assert command is not None
        np.testing.assert_array_equal(r.reference_qpos, q)
        assert np.max(np.abs(command - q)) <= 4 * dt + 1e-12
        assert np.all(command >= r.limits[:, 0]) and np.all(command <= r.limits[:, 1])
        r.commit(command, timestamp_ns=stamp)
        q = command.copy()
        t = stamp
        command[:] = 0
        np.testing.assert_array_equal(r.reference_qpos, q)
    bad = p.copy()
    bad[4] += 1
    assert r.retarget(bad, timestamp_ns=t + 1000000) is None
    assert r.reference_timestamp_ns == t
    np.testing.assert_array_equal(r.reference_qpos, q)


@pytest.mark.parametrize(
    "error", [nlopt.RoundoffLimited("roundoff"), RuntimeError("unexpected")]
)
def test_solver_failure_does_not_commit(hand, error):
    from types import SimpleNamespace

    r, p = hand
    q = np.deg2rad(HandConfig().home_qpos_deg)
    r.reset(q, timestamp_ns=1)

    def fail(seed):
        seed[:] = 0
        raise error

    r.optimizer = SimpleNamespace(
        set_lower_bounds=lambda x: None,
        set_upper_bounds=lambda x: None,
        set_min_objective=lambda x: None,
        optimize=fail,
    )
    if isinstance(error, nlopt.RoundoffLimited):
        assert r.retarget(p, timestamp_ns=100000001) is None
        assert "RoundoffLimited" in r.rejection_reason
    else:
        with pytest.raises(RuntimeError, match="unexpected"):
            r.retarget(p, timestamp_ns=100000001)
    np.testing.assert_array_equal(r.reference_qpos, q)
    assert r.reference_timestamp_ns == 1


def test_reachable_target_reconstruction_and_base_degeneracy(hand):
    r, p = hand
    q = np.array([0.6, 0.4, 0.5, 0.1, 0.7, 0.8, 0.4, 0.6, 0.5, 0.4, 0.3, 0.5])
    robot = r.fk.forward(q, r.frame_ids)[:, :3, 3].reshape(5, 3, 3)
    p[2:5] = robot[0]
    for f in range(1, 5):
        a, b, c = robot[f]
        p[1 + 4 * f : 5 + 4 * f] = [a, b, b + 0.4 * (c - b), c]
    target = r.calibration.targets(p, r.geometry, r.config, fk=r.fk, frame_ids=r.frame_ids)
    np.testing.assert_allclose(target.positions, robot, atol=1e-12)
    length = np.linalg.norm(p[2] - p[1])
    displacement = length * r.geometry.base_axes[0] - (p[2] - p[1])
    p[2:5] += displacement
    with pytest.raises(HandGeometryRejected, match="degenerate"):
        r.calibration.targets(p, r.geometry, r.config, fk=r.fk, frame_ids=r.frame_ids)


def synthetic_landmarks(r, q):
    """固定人体掌锚点，只从机器人取骨段方向；掌系变换应为恒等。"""
    pos = r.fk.forward(q, r.frame_ids)[:, :3, 3].reshape(5, 3, 3)
    p = np.zeros((21, 3))
    p[1] = r.fk.neutral_joint_geometry()[0][r.geometry.base_indices[0]]
    p[2:5] = pos[0]
    for f, side in zip(range(1, 5), [0.03, 0, -0.015, -0.03]):
        a, b, c = pos[f]
        anchor = np.array([0, side, 0.09])
        p[1+4*f:5+4*f] = np.array([a, b, b+0.4*(c-b), c]) + anchor-a
    return p


def test_random_full_pipeline_reconstruction(hand):
    from dexmani_real.teleop.retargeting.landmarks import prepare_landmarks

    r, _ = hand
    rng = np.random.default_rng(20261106)
    qs = rng.uniform(*r.limits.T, size=(1000, 12))
    qs[:3] = [r.limits[:, 0], r.limits[:, 1], r.limits.mean(axis=1)]
    qs[3:6, 4] = np.pi/2 + np.array([-1e-8, 0, 1e-8])
    p = synthetic_landmarks(r, qs[0])
    r.calibrate([hand_lengths(p)] * 25)
    for q in qs:
        p = synthetic_landmarks(r, q)
        np.testing.assert_allclose(prepare_landmarks(p), p, atol=1e-14)
        t = r.calibration.targets(p, r.geometry, r.config, fk=r.fk, frame_ids=r.frame_ids)
        np.testing.assert_allclose(t.requested_qpos, q, atol=1e-12)
        expected = r.fk.forward(q, r.frame_ids)[:, :3, 3].reshape(5, 3, 3)
        np.testing.assert_allclose(t.positions, expected, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(np.diff(t.positions, axis=1), axis=2), r.geometry.segments, atol=1e-12)
        np.testing.assert_allclose(t.directions, np.diff(expected, axis=1)/r.geometry.segments[:, :, None], atol=1e-12)


@pytest.mark.parametrize('offset', [-1e-6, 0, 1e-6])
@pytest.mark.parametrize('bound', [0, 1])
def test_operating_limit_projection(hand, offset, bound):
    r, _ = hand
    # 收窄全部操作范围，以合法机构构型检查独立裁剪而非硬件范围裁剪。
    limits = r.limits.copy()
    limits[:, 0] += 0.03
    limits[:, 1] -= 0.03
    geometry = replace(r.geometry, limits=limits)
    q = limits[:, bound] + offset
    p = synthetic_landmarks(r, q)
    r.calibrate([hand_lengths(p)] * 25)
    t = r.calibration.targets(p, geometry, r.config, fk=r.fk, frame_ids=r.frame_ids)
    expected_q = np.clip(q, *limits.T)
    np.testing.assert_allclose(t.requested_qpos, q, atol=1e-12)
    np.testing.assert_allclose(t.projected_qpos, expected_q, atol=1e-12)
    expected = r.fk.forward(expected_q, r.frame_ids)[:, :3, 3].reshape(5, 3, 3)
    np.testing.assert_allclose(t.positions, expected, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(np.diff(t.positions, axis=1), axis=2), geometry.segments, atol=1e-12)


def test_index_plane_unobservable_is_rejected(hand):
    r, p = hand
    lengths = r.calibration.bone_lengths[1]
    p[6:9] = p[5] + np.cumsum(lengths)[:, None] * r.geometry.base_axes[1]
    with pytest.raises(HandGeometryRejected, match='index flexion plane'):
        r.calibration.targets(p, r.geometry, r.config, fk=r.fk, frame_ids=r.frame_ids)

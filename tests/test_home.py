"""HOME 离线路径检查；不连接设备。"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.robot.commands import HomeTimeout, OperationCancelled
from dexmani_real.runtime import home


class CollisionProbe:
    arm_radii = np.array([.9, .8, .7, .6, .4, .3, .2])

    def __init__(self, reject=lambda point, count: None):
        self.points = []
        self.reject = reject

    def rejection(self, point, hand):
        self.points.append(point.copy())
        return self.reject(point, len(self.points))


@pytest.fixture
def linear_plan(monkeypatch):
    cfg = SimpleNamespace(
        home_qpos=np.ones(7), servo_hz=2.,
        joint_limit_lower=np.full(7, -3.), joint_limit_upper=np.full(7, 3.),
        homing=SimpleNamespace(
            max_speed_rad_per_s=1., max_acceleration_rad_per_s2=0.,
            timeout_s=10., collision_margin_m=.5, via_qpos=None,
        ),
    )
    trajectories = []

    class LinearTrajectory:
        duration = 1.

        def __init__(self, start, target):
            self.start, self.target = start.copy(), target.copy()
            self.calls = []

        def at_time(self, t):
            self.calls.append(t)
            return self.start + t * (self.target - self.start), np.ones(7), np.zeros(7)

    def calculate(cfg, q, v, target):
        trajectory = LinearTrajectory(q, target)
        trajectories.append(trajectory)
        return trajectory

    monkeypatch.setattr(home, "calculate_home_segment", calculate)
    monkeypatch.setattr(home, "validate_extrema", lambda *args: None)
    return cfg, trajectories


def test_real_ruckig_references_and_each_sample_evaluated_once(monkeypatch):
    base = RealConfig().arm
    cfg = replace(base, homing=replace(
        base.homing, max_jerk_deg_per_s3=4000., collision_margin_m=.01,
    ))
    q = np.asarray(cfg.home_qpos) + [.2, -.1, .2, -.1, .1, -.15, .1]
    v = np.zeros(7)
    trajectory = home.calculate_home_segment(cfg, q, v, np.asarray(cfg.home_qpos))
    times = np.arange(round(trajectory.duration * cfg.servo_hz) + 1) / cfg.servo_hz
    expected = np.array([trajectory.at_time(t)[0] for t in times])
    expected[-1] = cfg.home_qpos
    calls = []

    def at_time(t):
        calls.append(t)
        return trajectory.at_time(t)

    sampled = SimpleNamespace(
        duration=trajectory.duration, profiles=trajectory.profiles, at_time=at_time,
    )
    monkeypatch.setattr(home, "calculate_home_segment", lambda *args: sampled)
    collision = CollisionProbe()
    actual = home._plan_home(cfg, q, v, np.zeros(12), collision, lambda: None)
    np.testing.assert_array_equal(actual, expected)
    assert len(calls) == len(set(calls)) == len(collision.points)
    assert len(calls) > len(times)  # 本例确实进入曲线中点细分。


@pytest.mark.parametrize("via", [False, True])
def test_midpoint_collision_rejects_direct_and_tries_via(linear_plan, via):
    cfg, trajectories = linear_plan
    if via:
        cfg.homing.via_qpos = -np.ones(7)
    collision = CollisionProbe(
        lambda point, count: "midpoint collision" if count == 4 else None,
    )
    if via:
        references = home._plan_home(
            cfg, np.zeros(7), np.zeros(7), np.zeros(12), collision, lambda: None,
        )
        np.testing.assert_array_equal(references[:, 0], [0., -.5, -1., 0., 1.])
        assert len(trajectories) == 3
        np.testing.assert_array_equal(trajectories[1].target, cfg.homing.via_qpos)
        np.testing.assert_array_equal(trajectories[2].start, cfg.homing.via_qpos)
    else:
        with pytest.raises(home.HomePathRejected, match="Direct: midpoint collision"):
            home._plan_home(
                cfg, np.zeros(7), np.zeros(7), np.zeros(12), collision, lambda: None,
            )
        assert len(trajectories) == 1
    np.testing.assert_array_equal(np.array(collision.points)[:4, 0], [0., .5, 1., .75])


@pytest.mark.parametrize("failure", [OperationCancelled, HomeTimeout])
def test_refinement_propagates_cancellation_and_timeout(linear_plan, failure):
    cfg, trajectories = linear_plan
    cfg.homing.via_qpos = -np.ones(7)
    collision = CollisionProbe()

    def check():
        if len(collision.points) == 4:
            raise failure("stop refinement")

    with pytest.raises(failure, match="stop refinement"):
        home._plan_home(cfg, np.zeros(7), np.zeros(7), np.zeros(12), collision, check)
    assert len(trajectories) == 1
    assert len(collision.points) == 4


def test_trajectory_duration_timeout_precedes_sampling(linear_plan):
    cfg, trajectories = linear_plan
    cfg.homing.timeout_s = .5
    cfg.homing.via_qpos = -np.ones(7)
    collision = CollisionProbe()
    with pytest.raises(HomeTimeout, match="motion budget"):
        home._plan_home(
            cfg, np.zeros(7), np.zeros(7), np.zeros(12), collision, lambda: None,
        )
    assert len(trajectories) == 1
    assert trajectories[0].calls == []
    assert collision.points == []


@pytest.mark.parametrize("failure", [home.PositionLimitError, RuntimeError])
def test_only_position_extrema_rejection_tries_via(linear_plan, monkeypatch, failure):
    cfg, trajectories = linear_plan
    cfg.homing.via_qpos = -np.ones(7)

    def validate(*args):
        if len(trajectories) == 1:
            raise failure("extrema failure")

    monkeypatch.setattr(home, "validate_extrema", validate)
    collision = CollisionProbe()
    if failure is home.PositionLimitError:
        references = home._plan_home(
            cfg, np.zeros(7), np.zeros(7), np.zeros(12), collision, lambda: None,
        )
        np.testing.assert_array_equal(references[-1], cfg.home_qpos)
        assert len(trajectories) == 3
    else:
        with pytest.raises(RuntimeError, match="extrema failure"):
            home._plan_home(
                cfg, np.zeros(7), np.zeros(7), np.zeros(12), collision, lambda: None,
            )
        assert len(trajectories) == 1

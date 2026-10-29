from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.xarm7 import (
    HOME_CONVERGENCE_TIMEOUT_S,
    HOME_DWELL_S,
    ArmConfig,
)
from dexmani_real.config.xhand import HandConfig
from dexmani_real.control.homing import home_segment_timeout_s
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.runtime.home import home_arm
from dexmani_real.runtime.state import SafetyState


class Clock:
    def __init__(self):
        self.now_ns = 1_000_000_000

    def advance(self, seconds):
        self.now_ns += round(seconds * 1e9)

    def monotonic_ns(self):
        return self.now_ns

    def monotonic(self):
        return self.now_ns / 1e9


def test_home_budget_includes_acceleration_and_parent_phases(monkeypatch):
    from dexmani_real.runtime import home

    clock = Clock()
    monkeypatch.setattr(home.time, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(home.time, "monotonic", clock.monotonic)
    cfg = replace(ArmConfig(), max_joint_acceleration_deg_per_s2=np.rad2deg(0.01))
    segment = home_segment_timeout_s(0.1, cfg)
    assert segment > 2 * np.sqrt(0.1 / 0.01) + HOME_DWELL_S
    target = np.asarray(cfg.home_qpos)
    start = target.copy()
    start[0] += 0.1
    arm = np.zeros(1, dtype=[("qpos", "f8", (7,)), ("qvel", "f8", (7,)), ("timestamp_ns", "u8")])
    arm["qpos"] = start
    robot = SimpleNamespace(
        arm_config=cfg, hand_config=replace(HandConfig(), enabled=False),
        stop_required=False, home_arm=Mock(),
    )

    def feedback():
        clock.advance(0.01)
        arm["timestamp_ns"] = clock.now_ns

    robot.service_feedback = feedback
    robot.read_state = lambda: SimpleNamespace(arm=arm)
    state = SimpleNamespace(safety_state=SafetyState.ARMED, run_id=1, can_move=lambda **_: True)
    for waypoints in (np.stack((start, target)), np.empty((0, 7))):
        planner = SimpleNamespace(set_hand_qpos=Mock(), plan=lambda *_: waypoints)
        assert home_arm(state, robot, planner, cancel_requested=lambda: False).ok
        duration = (robot.home_arm.call_args.kwargs["deadline_ns"] - clock.now_ns) / 1e9
        assert duration >= HOME_CONVERGENCE_TIMEOUT_S + (
            segment if len(waypoints) else HOME_CONVERGENCE_TIMEOUT_S
        )


def test_empty_home_observes_dwell_without_counting_query_time(monkeypatch):
    from dexmani_real.robot.xarm7 import worker

    clock = Clock()
    monkeypatch.setattr(worker.time, "monotonic_ns", clock.monotonic_ns)
    cfg = replace(ArmConfig(), feedback_hz=100.0)
    target = np.asarray(cfg.home_qpos)

    def read():
        clock.advance(0.015)
        return target, np.zeros(7)

    io = SimpleNamespace(publish_status=Mock(), stop=SimpleNamespace(wait=clock.advance))
    driver = SimpleNamespace(read=read, read_status=lambda: SimpleNamespace(error_code=0))
    controller = ArmController(cfg, io, None, driver)
    controller.home(np.empty((0, 7)), target, check_abort=lambda: None)
    assert clock.monotonic() >= 1 + HOME_DWELL_S + 0.03

    with pytest.raises(TimeoutError, match="dwell not confirmed"):
        controller.wait_settled(lambda: None, stage="late query", timeout_s=0.01)

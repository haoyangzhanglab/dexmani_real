import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.deployment.rollout import Rollout
from dexmani_real.teleop.loop import TeleopLoop
from dexmani_real.utils.timing import next_tick_ns


@pytest.mark.parametrize("now,expected", [(100, 200), (105, 200), (455, 500)])
def test_next_grid(now, expected):
    assert next_tick_ns(100, 100, now) == expected


@pytest.mark.parametrize("kind", ["teleop", "sync"])
def test_tick_grid_and_action_order(monkeypatch, kind):
    clock = [100]
    if kind == "teleop":
        obj = TeleopLoop.__new__(TeleopLoop)
        obj.runtime_state = SimpleNamespace(stop_signal=threading.Event())
        obj.recorder = None
        obj.next_tick_ns = None
        obj.control_dt_ns = 100
        obj.target_deadline_ns = 10000
        obj.sensors = SimpleNamespace(channels=None)
        obj.robot = Mock()
        obj.runtime = obj.capture_needs = obj.control_needs = None
        obj.execute_control_step = Mock()
        for name, value in [
            ("read_observation", object()),
            ("check_device_feedback", []),
            ("check_observation", []),
            ("sources_available", True),
        ]:
            monkeypatch.setattr(
                "dexmani_real.teleop.loop." + name, Mock(return_value=value)
            )
        monkeypatch.setattr(
            "dexmani_real.teleop.loop.time.monotonic_ns", lambda: clock[0]
        )
        step = obj.run_control_tick
        count = lambda: obj.execute_control_step.call_count
        planned = lambda: obj.next_tick_ns
    else:
        obj = Rollout.__new__(Rollout)
        obj.next_step_ns = 100
        obj.dt_ns = 100
        obj.plan = SimpleNamespace(actions=np.arange(8))
        obj.cursor = 0
        obj.execution = SimpleNamespace(action_steps=8)
        obj.has_submitted = True
        obj.sample_observation = Mock(return_value=object())
        obj.update_history = Mock(return_value=True)
        obj.check_motion = Mock()
        obj.duration_expired = Mock(return_value=False)
        obj.poll_model = Mock()
        obj.prepare_command = Mock(side_effect=lambda action, row: action)
        obj.robot = Mock()
        obj.commit_arm = Mock()
        obj.record_row = Mock()
        monkeypatch.setattr(
            "dexmani_real.deployment.rollout.time.monotonic_ns", lambda: clock[0]
        )
        step = obj.step_sync
        count = lambda: obj.robot.send_action.call_count
        planned = lambda: obj.next_step_ns
    for now, expected, steps in [
        (100, 200, 1),
        (199, 200, 1),
        (205, 300, 2),
        (305, 400, 3),
        (655, 700, 4),
        (656, 700, 4),
        (705, 800, 5),
    ]:
        clock[0] = now
        step()
        assert planned() == expected
        assert count() == steps
    if kind == "sync":
        assert obj.cursor == 5
        assert [call.args[0] for call in obj.robot.send_action.call_args_list] == list(
            range(5)
        )
    else:
        clock[0] = 900
        obj.runtime_state.motion_lock = threading.RLock()
        obj.runtime_state.cancel_epoch = obj.cancel_epoch = 1
        obj.runtime_state.requests_allow_motion = lambda: True
        obj.runtime = SimpleNamespace(teleop=SimpleNamespace(target_max_age_s=0.5))
        obj.preparation_interrupted = Mock(return_value=False)
        obj.try_begin()
        assert obj.next_tick_ns is None
        step()
        assert planned() == 1000

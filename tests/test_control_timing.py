import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.deployment.rollout import Rollout
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop.keyboard import KeyboardTeleop
from dexmani_real.teleop.loop import TeleopLoop
from dexmani_real.utils.timing import next_tick_ns


@pytest.mark.parametrize("now,expected", [(100, 200), (105, 200), (455, 500)])
def test_next_grid(now, expected):
    assert next_tick_ns(100, 100, now) == expected


def test_keyboard_grid_and_idle_service(monkeypatch):
    obj = KeyboardTeleop(None)
    obj.runtime_state = SessionState()
    obj.robot, obj.keys, obj.jog = Mock(), Mock(), Mock()
    obj.keys.poll.return_value = ([], True)
    obj.period_ns = obj.tick_due_ns = 100_000_000
    times = iter([205_000_000, 305_000_000, 655_000_000, 705_000_000])
    clock, actual, planned, sleeps = [100_000_000], [], [], []

    def step(*args):
        actual.append(clock[0])
        planned.append(obj.tick_due_ns)
        if len(actual) == 5:
            obj.runtime_state.request_quit()
        else:
            # 下一轮从本次执行后 1 ns 开始，确实经过未到期分支。
            clock[0] += 1

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] = next(times)

    obj.jog.step.side_effect = step
    monkeypatch.setattr("dexmani_real.teleop.keyboard.time.monotonic_ns", lambda: clock[0])
    monkeypatch.setattr("dexmani_real.teleop.keyboard.time.sleep", sleep)
    assert obj.run_control_loop()
    assert actual == [100_000_000, 205_000_000, 305_000_000, 655_000_000, 705_000_000]
    assert planned == [200_000_000, 300_000_000, 400_000_000, 700_000_000, 800_000_000]
    assert obj.jog.step.call_count == obj.robot.service_feedback.call_count == 5
    assert obj.robot.service_idle.call_count == 4
    assert sleeps == [0.005] * 4


def test_keyboard_start_and_home_rebase_clock(monkeypatch):
    from dexmani_real.config.real import RealConfig

    cfg = RealConfig()
    obj = KeyboardTeleop(cfg)
    obj.jog_config = cfg.keyboard_teleop
    obj.runtime_state = SessionState()
    obj.robot, obj.keys = Mock(), Mock()
    obj.bounds = cfg.environment.workspace.as_array().T
    jog = Mock()
    monkeypatch.setattr("dexmani_real.teleop.keyboard.MotionConstraints", Mock())
    monkeypatch.setattr("dexmani_real.teleop.keyboard.IKSolver", Mock())
    monkeypatch.setattr("dexmani_real.teleop.keyboard.CartesianJog", Mock(return_value=jog))
    clock = [123]
    monkeypatch.setattr("dexmani_real.teleop.keyboard.time.monotonic_ns", lambda: clock[0])
    obj.start()
    assert obj.tick_due_ns == 123 + obj.period_ns
    jog.clear_reference.assert_called_once()
    obj.robot.prepare_streaming.assert_called_once_with(obj.runtime_state.cancel_epoch)
    assert obj.robot.send_action.call_args.args[0].arm_hold

    clock[0] = 987_654_321
    monkeypatch.setattr("dexmani_real.teleop.keyboard.home_hand", Mock())
    monkeypatch.setattr("dexmani_real.teleop.keyboard.home_arm", Mock())
    obj.robot.prepare_streaming.side_effect = lambda _: obj.runtime_state.stop_signal.clear()
    obj.run_home(obj.runtime_state.cancel_epoch)
    assert obj.tick_due_ns == clock[0] + obj.period_ns
    assert jog.clear_reference.call_count == 2
    assert obj.robot.send_action.call_args.args[0].arm_hold


@pytest.mark.parametrize("event", ["home", "quit", "estop"])
def test_keyboard_events_do_not_jog(monkeypatch, event):
    obj = KeyboardTeleop(None)
    obj.runtime_state = SessionState()
    obj.robot, obj.keys, obj.jog = Mock(), Mock(), Mock()
    obj.period_ns = obj.tick_due_ns = 100
    monkeypatch.setattr("dexmani_real.teleop.keyboard.time.monotonic_ns", lambda: 100)

    def poll(*args, **kwargs):
        if event == "quit":
            obj.runtime_state.request_quit()
        elif event == "estop":
            obj.runtime_state.request_emergency_stop()
        return (["r"] if event == "home" else []), True

    obj.keys.poll.side_effect = poll
    obj.keys.accept_operation.return_value = 7
    obj.run_home = Mock(side_effect=lambda _: obj.runtime_state.request_quit())
    assert obj.run_control_loop() == (event != "estop")
    obj.jog.step.assert_not_called()
    if event == "home":
        obj.run_home.assert_called_once_with(7)
    else:
        obj.run_home.assert_not_called()


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


@pytest.mark.parametrize("hz", [0.2, 0.5, 30.0])
@pytest.mark.parametrize("event", [None, "s", "q", "esc", "feedback_stale", "lease_expired", "slow_service"])
def test_replay_deadlines_and_wait_interruptions(monkeypatch, hz, event):
    import multiprocessing as mp

    import dexmani_real.replay.run as replay
    from dexmani_real.config.real import RealConfig
    from dexmani_real.replay.trajectory import Trajectory
    from dexmani_real.robot.commands import (
        COMMAND_LEASE_NS,
        OperationCancelled,
        RobotError,
    )
    from dexmani_real.robot.system import Robot
    from dexmani_real.robot.xarm7.protocol import ArmChannels
    from dexmani_real.robot.xarm7.worker import ArmController
    from dexmani_real.robot.xhand.feedback import XHandState

    clock = [1_000_000_000]
    monkeypatch.setattr("time.monotonic_ns", lambda: clock[0])
    monkeypatch.setattr("time.monotonic", lambda: clock[0] / 1e9)
    runtime = RealConfig()
    trajectory = Trajectory(hz, np.zeros((2, 7)), np.zeros((2, 12)))
    monkeypatch.setattr(replay, "load_trajectory", lambda *a: trajectory)
    trajectory_reference = SimpleNamespace(step=lambda *a, **kw: np.zeros(7))
    monkeypatch.setattr("dexmani_real.robot.xarm7.worker.ArmTrajectory", lambda *a: trajectory_reference)
    robots, submitted, returned = [], [], []

    class ReplayRobot(Robot):
        def connect(self):
            io = ArmChannels(mp.get_context("spawn"), stop_signal=self.runtime_state.stop_signal,
                             emergency_signal=self.runtime_state.emergency_signal)
            driver = Mock()
            driver.read.return_value = np.zeros(7), np.zeros(7)
            self.worker = ArmController(runtime.arm, io, Mock(), driver)
            self.arm = SimpleNamespace(
                check_error=lambda: None, is_connected=True, submit=io.publish_goal,
                snapshot=lambda: self.worker.status.reshape(1).copy(),
            )
            self.hand = Mock(is_connected=True)

            def sample():
                if event == "feedback_stale" and submitted:
                    return None
                return XHandState(np.zeros(12), None, None, clock[0])

            self.hand.get_state.side_effect = sample
            self.connected = True
            robots.append(self)
            self.service_feedback()

        def prepare_streaming(self, epoch):
            self.worker.reference = trajectory_reference

        def service_feedback(self):
            # 第三次服务发生在首行 deadline 建立后、提交前。
            self.services = getattr(self, "services", 0) + 1
            if event == "slow_service" and self.services == 3:
                clock[0] += 6_000_000_000
            self.worker.read_feedback()
            super().service_feedback()
            self.worker.tick()

        def send_action(self, command, *, expires_ns):
            submitted.append((clock[0], expires_ns))
            super().send_action(command, expires_ns=expires_ns)
            clock[0] += 10_000_000  # 末行留驻从发送返回时刻开始。
            returned.append(clock[0])

    monkeypatch.setattr(replay, "Robot", ReplayRobot)
    keys = Mock()

    def poll(state, **kwargs):
        if submitted and event in {"s", "q", "esc"}:
            {"s": state.request_stop, "q": state.request_quit,
             "esc": state.request_emergency_stop}[event]()
        return [], state.stop_signal.is_set()

    keys.poll.side_effect = poll
    monkeypatch.setattr(replay, "Keyboard", lambda **kwargs: keys)
    closed_at = []
    monkeypatch.setattr(replay, "close_runtime", lambda *a, **kw: closed_at.append(clock[0]))

    def sleep(seconds):
        clock[0] += round(seconds * 1e9)
        if event == "lease_expired" and submitted:
            clock[0] = submitted[0][1] + 1

    monkeypatch.setattr("time.sleep", sleep)
    if event in {"esc", "feedback_stale", "lease_expired"}:
        with pytest.raises((OperationCancelled, RobotError)) as error:
            replay.run_replay("unused", runtime)
        if event == "feedback_stale":
            assert "hand feedback unavailable/stale" in str(error.value)
        elif event == "lease_expired":
            assert "device stopped" in str(error.value)
            assert robots[0].worker.active is robots[0].worker.reference is None
        assert robots[0].runtime_state.stop_signal.is_set()
    else:
        replay.run_replay("unused", runtime)
    period_ns = round(1e9 / hz)
    assert submitted[0][1] == max(submitted[0][0], 1_000_000_000 + period_ns) + COMMAND_LEASE_NS
    if event is None or event == "slow_service":
        assert len(submitted) == 2
        assert submitted[1][0] == max(returned[0], 1_000_000_000 + period_ns)
        assert submitted[1][1] == submitted[1][0] + COMMAND_LEASE_NS
        assert closed_at == [returned[-1] + 500_000_000]
        assert not robots[0].runtime_state.stop_signal.is_set()
    elif event in {"s", "q", "esc", "lease_expired"}:
        assert len(submitted) == 1

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.xarm7 import ArmConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.robot.commands import FeedbackUnavailable, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7 import process, worker
from dexmani_real.robot.xarm7.driver import XArm7
from dexmani_real.robot.xarm7.protocol import STANDSTILL_DWELL_S
from dexmani_real.runtime.state import SessionState


@pytest.fixture
def robot_with_mock_arm(monkeypatch):
    cfg = ArmConfig()
    state = SessionState()
    robot = Robot(cfg, HandConfig(enabled=False), runtime_state=state)
    robot.arm = arm = process.ArmProcess(cfg, runtime_state=state)
    robot.connected = True
    arm.process = Mock(pid=1, exitcode=None)
    arm.process.is_alive.return_value = True
    arm.io.ready.set()
    driver = XArm7(cfg)
    driver.api = sdk = Mock()
    sdk.set_state.return_value = 0
    sdk.get_err_warn_code.return_value = 0, (0, 0)
    sdk.get_joint_states.return_value = 0, (np.zeros(7), np.zeros(7))
    owner = worker.ArmController(cfg, arm.io, arm.child_pipe, driver)

    clock = SimpleNamespace(now_ns=1_000_000_000)

    def advance(seconds):
        clock.now_ns += round(seconds * 1e9)

    mock_time = SimpleNamespace(
        monotonic_ns=lambda: clock.now_ns,
        monotonic=lambda: clock.now_ns / 1e9,
    )
    monkeypatch.setattr(worker, "time", mock_time)
    monkeypatch.setattr(process, "time", mock_time)
    monkeypatch.setattr(process, "EXECUTOR_TIMEOUT_S", 0.5)
    monkeypatch.setattr(arm.io.closing, "wait", advance)

    def acknowledge_stop(_timeout):
        assert state.stop_signal.is_set()
        if not arm.io.stopped.is_set():
            owner.stop()
        return arm.io.stopped.is_set()

    poll = arm.pipe.poll

    def run_owner(_timeout):
        if arm.child_pipe.poll():
            owner.operation(arm.child_pipe.recv())
        return poll(0)

    monkeypatch.setattr(arm.io.stopped, "wait", acknowledge_stop)
    monkeypatch.setattr(arm.pipe, "poll", run_owner)

    with arm.pipe, arm.child_pipe:
        yield robot, sdk, clock, advance


@pytest.mark.parametrize("motion", ["velocity", "position_drift"])
def test_sdk_stop_does_not_confirm_physical_stop(robot_with_mock_arm, monkeypatch, motion):
    robot, sdk, _, advance = robot_with_mock_arm

    def read(**_):
        qpos, qvel = np.zeros(7), np.zeros(7)
        if motion == "velocity":
            qvel[-1] = -2 * robot.arm_config.standstill_velocity_tolerance_rad_s
        else:
            qpos[-1] = (
                sdk.get_joint_states.call_count * robot.arm_config.standstill_position_tolerance_rad / 3
            )
        return 0, (qpos, qvel)

    sdk.get_joint_states.side_effect = read
    run_owner = robot.arm.pipe.poll
    reply_delayed = False

    def delayed_reply(timeout):
        nonlocal reply_delayed
        ready = run_owner(timeout)
        if ready and not reply_delayed:
            # Owner 用完采样预算，父进程在下一次轮询才收到回复。
            reply_delayed = True
            advance(timeout)
            return False
        return ready

    monkeypatch.setattr(robot.arm.pipe, "poll", delayed_reply)
    with pytest.raises(TimeoutError, match="arm physical stop execution deadline expired"):
        robot.stop()

    sdk.set_state.assert_called_once_with(4)
    assert robot.arm.io.stopped.is_set()
    assert not robot.arm.io.closing.is_set()
    assert sdk.get_joint_states.call_count > 1
    assert not robot.arm.physical_stop_confirmed


def test_stop_confirms_dwell_with_fresh_feedback(robot_with_mock_arm):
    robot, sdk, clock, _ = robot_with_mock_arm
    samples = []

    def read(**_):
        sdk.set_state.assert_called_once_with(4)
        samples.append(clock.now_ns)
        qvel = np.full(7, robot.arm_config.standstill_velocity_tolerance_rad_s / 2)
        return 0, (np.zeros(7), qvel)

    sdk.get_joint_states.side_effect = read
    robot.stop()
    assert robot.arm.physical_stop_confirmed
    assert len(samples) > 2
    assert samples[-1] - samples[0] >= round(STANDSTILL_DWELL_S * 1e9)
    assert samples[-2] - samples[0] < round(STANDSTILL_DWELL_S * 1e9)

    previous_count = len(samples)
    robot.service_idle()
    assert len(samples) == previous_count
    robot.stop()
    assert len(samples) > previous_count

    arm = robot.arm
    arm.resume(robot.runtime_state.cancel_epoch, deadline_ns=clock.now_ns + 1_000_000_000)
    assert not arm.physical_stop_confirmed


@pytest.mark.parametrize("delay_during_read", [True, False])
def test_physical_stop_rejects_stale_feedback(robot_with_mock_arm, monkeypatch, delay_during_read):
    robot, sdk, _, advance = robot_with_mock_arm
    max_age = robot.arm_config.feedback_max_age_s

    if delay_during_read:
        def read(**_):
            advance(max_age)
            return sdk.get_joint_states.return_value

        sdk.get_joint_states.side_effect = read
    else:
        monkeypatch.setattr(robot.arm.io.closing, "wait", lambda delay: advance(delay + max_age))

    with pytest.raises(FeedbackUnavailable, match="arm physical stop.*feedback"):
        robot.stop()

    assert not robot.arm.physical_stop_confirmed


def test_robot_stop_propagates_owner_stop_error(robot_with_mock_arm):
    robot, sdk, _, _ = robot_with_mock_arm
    sdk.set_state.return_value = 13

    with pytest.raises(RobotError, match="stop set_state.*SDK code 13"):
        robot.stop()

    sdk.get_joint_states.assert_not_called()
    assert not robot.arm.physical_stop_confirmed


def test_robot_stop_requires_feedback_from_live_owner(robot_with_mock_arm):
    robot, sdk, _, _ = robot_with_mock_arm
    arm = robot.arm
    arm.io.stopped.set()
    arm.io.closing.set()
    arm.process.exitcode = 0
    arm.process.is_alive.return_value = False

    with pytest.raises(RobotError, match="owner unavailable for physical stop verification"):
        robot.stop()

    sdk.get_joint_states.assert_not_called()
    assert not arm.physical_stop_confirmed


def test_robot_stop_fails_on_missing_sdk_acknowledgement(robot_with_mock_arm, monkeypatch):
    robot, sdk, _, advance = robot_with_mock_arm
    monkeypatch.setattr(robot.arm.io.stopped, "wait", advance)

    with pytest.raises(TimeoutError, match="SDK acceptance not confirmed"):
        robot.stop()

    sdk.get_joint_states.assert_not_called()
    assert not robot.arm.child_pipe.poll()
    assert not robot.arm.physical_stop_confirmed


def test_robot_stop_reports_existing_owner_fault_without_waiting(robot_with_mock_arm, monkeypatch):
    robot, sdk, _, _ = robot_with_mock_arm
    robot.arm.io.fault.value = b"controller connection lost"
    wait = Mock()
    monkeypatch.setattr(robot.arm.io.stopped, "wait", wait)

    with pytest.raises(RobotError, match="controller connection lost"):
        robot.stop()

    wait.assert_not_called()
    sdk.get_joint_states.assert_not_called()


def test_robot_stop_fails_on_missing_physical_stop_reply(robot_with_mock_arm, monkeypatch):
    robot, sdk, _, advance = robot_with_mock_arm

    def no_reply(timeout):
        advance(timeout)
        return False

    monkeypatch.setattr(robot.arm.pipe, "poll", no_reply)
    with pytest.raises(TimeoutError, match="arm physical stop unresolved"):
        robot.stop()

    sdk.set_state.assert_called_once_with(4)
    assert robot.arm.io.closing.is_set()
    assert not robot.arm.physical_stop_confirmed


def test_robot_stop_preserves_interrupt_during_owner_failure(robot_with_mock_arm, monkeypatch):
    robot, _, _, _ = robot_with_mock_arm

    def interrupted_reply(_timeout):
        robot.arm.io.fault.value = b"controller connection lost"
        raise KeyboardInterrupt

    monkeypatch.setattr(robot.arm.pipe, "poll", interrupted_reply)
    with pytest.raises(KeyboardInterrupt):
        robot.stop()

    assert robot.runtime_state.estop_request
    assert robot.arm.io.closing.is_set()
    assert not robot.arm.physical_stop_confirmed

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xhand.feedback import XHandState


@pytest.fixture
def robot(monkeypatch):
    clock = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.robot.system.time.monotonic_ns', lambda: clock[0])
    state = SimpleNamespace(stop_signal=threading.Event(), emergency_signal=threading.Event())
    robot = Robot(
        SimpleNamespace(joint_limit_lower=[-3.] * 7, joint_limit_upper=[3.] * 7),
        SimpleNamespace(feedback_max_age_s=.5),
        runtime_state=state,
    )
    robot.connected = True
    robot.arm = Mock(is_connected=True)
    robot.hand = Mock(is_connected=True)
    robot.hand_feedback.consume(XHandState(np.zeros(12), None, None, clock[0]))
    return robot, clock


@pytest.mark.parametrize('delay_ns', [100_000_000, 100_000_001])
def test_arm_publication_expiry_prevents_hand_send(robot, delay_ns):
    obj, clock = robot
    expires_ns = clock[0] + 100_000_000
    command = RobotCommand(np.zeros(7), np.zeros(12))

    def publish(command, *, expires_ns):
        assert clock[0] < expires_ns
        clock[0] += delay_ns

    obj.arm.submit.side_effect = publish
    with pytest.raises(OperationCancelled, match='expired before hand submission'):
        obj.send_action(command, expires_ns=expires_ns)
    obj.arm.submit.assert_called_once_with(command, expires_ns=expires_ns)
    obj.hand._send_action.assert_not_called()
    assert obj.runtime_state.stop_signal.is_set()
    assert not obj.hand_stop_required


@pytest.mark.parametrize('with_arm', [False, True])
def test_unexpired_command_sends_hand(robot, with_arm):
    obj, clock = robot
    expires_ns = clock[0] + 100_000_000
    command = RobotCommand(np.zeros(7) if with_arm else None, np.zeros(12))
    obj.send_action(command, expires_ns=expires_ns)
    assert obj.arm.submit.call_count == int(with_arm)
    np.testing.assert_array_equal(obj.hand._send_action.call_args.args[0], command.hand_qpos)
    obj.hand._send_action.assert_called_once()
    assert obj.hand_stop_required
    assert not obj.runtime_state.stop_signal.is_set()


def test_hand_only_expiry_during_feedback_check_prevents_send(robot):
    obj, clock = robot
    expires_ns = clock[0] + 100_000_000
    frame = obj.hand_feedback.snapshot()

    def snapshot():
        clock[0] = expires_ns
        return frame

    obj.hand_feedback.snapshot = snapshot
    with pytest.raises(OperationCancelled, match='expired before hand submission'):
        obj.send_action(RobotCommand(hand_qpos=np.zeros(12)), expires_ns=expires_ns)
    obj.arm.submit.assert_not_called()
    obj.hand._send_action.assert_not_called()
    assert obj.runtime_state.stop_signal.is_set()


def test_hand_sdk_crossing_deadline_does_not_invalidate_send(robot):
    obj, clock = robot
    expires_ns = clock[0] + 100_000_000

    def send(qpos):
        assert clock[0] < expires_ns
        clock[0] = expires_ns

    obj.hand._send_action.side_effect = send
    obj.send_action(RobotCommand(hand_qpos=np.zeros(12)), expires_ns=expires_ns)
    obj.hand._send_action.assert_called_once()
    assert not obj.runtime_state.stop_signal.is_set()


def test_invalid_hand_target_is_rejected_before_arm_publication(robot):
    obj, clock = robot
    command = RobotCommand(np.zeros(7), np.full(12, 99.))
    with pytest.raises(ValueError, match='hand target violates joint limits'):
        obj.send_action(command, expires_ns=clock[0] + 100_000_000)
    obj.arm.submit.assert_not_called()
    obj.hand._send_action.assert_not_called()


@pytest.mark.parametrize('after_publication', [False, True])
def test_stale_hand_feedback_blocks_the_next_device_submission(robot, after_publication):
    obj, clock = robot
    expires_ns = clock[0] + 1_000_000_000

    def delay():
        clock[0] += 500_000_001

    if after_publication:
        obj.arm.submit.side_effect = lambda *args, **kwargs: delay()
    else:
        delay()
    with pytest.raises(RobotError, match='hand feedback unavailable/stale'):
        obj.send_action(RobotCommand(np.zeros(7), np.zeros(12)), expires_ns=expires_ns)
    assert obj.arm.submit.call_count == int(after_publication)
    obj.hand._send_action.assert_not_called()
    assert obj.runtime_state.stop_signal.is_set()


def test_cancel_after_arm_publication_prevents_hand_send(robot):
    obj, clock = robot
    obj.arm.submit.side_effect = lambda *args, **kwargs: obj.runtime_state.stop_signal.set()
    with pytest.raises(OperationCancelled, match='device operation cancelled'):
        obj.send_action(RobotCommand(np.zeros(7), np.zeros(12)), expires_ns=clock[0] + 100_000_000)
    obj.arm.submit.assert_called_once()
    obj.hand._send_action.assert_not_called()
    assert obj.runtime_state.stop_signal.is_set()


def test_hand_sdk_failure_keeps_stop_required(robot):
    obj, clock = robot
    obj.hand._send_action.side_effect = RuntimeError('hand SDK failed')
    with pytest.raises(RuntimeError, match='hand SDK failed'):
        obj.send_action(RobotCommand(hand_qpos=np.zeros(12)), expires_ns=clock[0] + 100_000_000)
    assert obj.hand_stop_required
    assert obj.runtime_state.stop_signal.is_set()

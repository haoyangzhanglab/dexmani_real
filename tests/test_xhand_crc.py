import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.deployment import ExecutionConfig
from dexmani_real.config.teleop import HandRetargetingConfig, TeleopConfig
from dexmani_real.config.xarm7 import ArmConfig
from dexmani_real.config.xhand import HandConfig
from dexmani_real.control.actions import ActionMapper
from dexmani_real.deployment import rollout as deployment
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot import system
from dexmani_real.robot.commands import (
    CommandExpired,
    FeedbackUnavailable,
    HandCommandUnconfirmed,
    OperationCancelled,
    RobotCommand,
    RobotError,
)
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand import driver
from dexmani_real.robot.xhand.feedback import XHandState
from dexmani_real.runtime import observation
from dexmani_real.runtime.lifecycle import close_runtime
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop import loop as teleop
from dexmani_real.teleop.vr import VRController

CRC = driver.COMMUNICATION_CRC_ERROR_CODE


def sdk_error(code):
    return SimpleNamespace(error_code=code, error_message=f"SDK code {code}")


def sdk_state(qpos):
    return SimpleNamespace(finger_state=[
        SimpleNamespace(id=i, position=value, commboard_err=0, jonitboard_err=0, tipboard_err=0)
        for i, value in enumerate(qpos)
    ])


@pytest.fixture
def robot_with_mock_hand(monkeypatch):
    cfg = HandConfig()
    robot = system.Robot(ArmConfig(), cfg, runtime_state=SessionState())
    hand = robot.hand = driver.XHand(cfg)
    hand.control = Mock()
    hand.command = SimpleNamespace(finger_command=[SimpleNamespace() for _ in range(12)])
    hand.connected_flag = robot.connected = True
    robot.arm = Mock(is_connected=True)
    clock = SimpleNamespace(now_ns=1_000_000_000)

    def sleep(seconds):
        clock.now_ns += round(seconds * 1e9)

    mock_time = SimpleNamespace(monotonic_ns=lambda: clock.now_ns, sleep=Mock(side_effect=sleep))
    monkeypatch.setattr(driver, "time", mock_time)
    monkeypatch.setattr(system, "time", mock_time)
    target = np.deg2rad(cfg.home_qpos_deg)
    robot.hand_feedback.consume(XHandState(target, None, None, clock.now_ns))
    return robot, RobotCommand(arm_qpos=np.zeros(7), hand_qpos=target), clock, mock_time.sleep


def test_write_crc_retries_same_absolute_target_and_resets_count(robot_with_mock_hand):
    robot, command, clock, sleep = robot_with_mock_hand
    hand = robot.hand
    targets = []

    def send(_device, sdk_command):
        targets.append([joint.position for joint in sdk_command.finger_command])
        return sdk_error(CRC if len(targets) == 1 else 0)

    hand.control.send_command.side_effect = send
    robot.send_action(command, expires_ns=clock.now_ns + 1_000_000_000)
    np.testing.assert_array_equal(targets, np.repeat(command.hand_qpos[None], 2, axis=0))
    robot.arm.submit.assert_called_once()
    sleep.assert_called_once_with(hand.cfg.crc_retry_interval_s)
    assert hand.consecutive_write_crc == 0
    assert hand.crc_counts["write_errors"] == hand.crc_counts["write_recoveries"] == 1
    assert hand.crc_counts["unconfirmed_actions"] == 0
    assert not robot.runtime_state.stop_signal.is_set()

    sleep.reset_mock()
    robot.send_action(command, expires_ns=clock.now_ns + 1_000_000_000)
    sleep.assert_not_called()
    assert len(targets) == 3


def test_write_crc_exhaustion_stops_before_next_command_and_refreshes_hold(robot_with_mock_hand):
    robot, command, clock, _ = robot_with_mock_hand
    hand = robot.hand
    hand.control.send_command.return_value = sdk_error(CRC)

    with pytest.raises(HandCommandUnconfirmed, match="delivery unconfirmed"):
        robot.send_action(command, expires_ns=clock.now_ns + 1_000_000_000)

    robot.arm.submit.assert_called_once()
    assert hand.consecutive_write_crc == hand.control.send_command.call_count == 2
    assert robot.runtime_state.stop_signal.is_set()
    assert robot.feedback_next_ns == 0 and robot.hand_feedback_pending
    assert hand.crc_counts["unconfirmed_actions"] == 1
    assert hand.crc_counts["write_recoveries"] == 0

    with pytest.raises(FeedbackUnavailable, match="feedback pending"):
        robot.send_action(command, expires_ns=clock.now_ns + 1_000_000_000)

    hand.control.read_state.return_value = sdk_error(0), sdk_state(command.hand_qpos + 0.01)
    hand.control.send_command.return_value = sdk_error(0)
    robot.stop()
    assert not robot.hand_feedback_pending
    assert [operation[0] for operation in hand.control.mock_calls] == [
        "send_command", "send_command", "read_state", "send_command",
    ]
    held = [joint.position for joint in hand.command.finger_command]
    np.testing.assert_array_equal(held, command.hand_qpos + 0.01)
    with pytest.raises(OperationCancelled, match="device operation cancelled"):
        robot.send_action(command, expires_ns=clock.now_ns + 1_000_000_000)
    robot.arm.submit.assert_called_once()
    assert hand.control.send_command.call_count == 3


@pytest.mark.parametrize(("limited_by", "late_ns", "retry"), [
    ("command", 0, False), ("feedback", 1, False), ("command", 1, True),
])
def test_late_sdk_success_stops_without_fabricating_crc(robot_with_mock_hand, limited_by, late_ns, retry):
    robot, command, clock, _ = robot_with_mock_hand
    hand = robot.hand
    deadline_ns = clock.now_ns + 100_000_000
    if limited_by == "feedback":
        deadline_ns = clock.now_ns + int(hand.cfg.feedback_max_age_s * 1e9)
    expires_ns = deadline_ns if limited_by == "command" else deadline_ns + 1_000_000_000

    def send(*_):
        if retry and hand.control.send_command.call_count == 1:
            return sdk_error(CRC)
        clock.now_ns = deadline_ns + late_ns
        return sdk_error(0)

    hand.control.send_command.side_effect = send
    with pytest.raises(HandCommandUnconfirmed, match="SDK success reply arrived at or after"):
        robot.send_action(command, expires_ns=expires_ns)
    robot.arm.submit.assert_called_once()
    assert robot.runtime_state.stop_signal.is_set()
    assert robot.feedback_next_ns == 0 and robot.hand_feedback_pending
    assert hand.control.send_command.call_count == 1 + retry
    assert hand.consecutive_write_crc == hand.crc_counts["unconfirmed_actions"] == 0
    assert hand.crc_counts["write_errors"] == hand.crc_counts["write_recoveries"] == int(retry)


@pytest.mark.parametrize("limited_by", ["command", "feedback"])
def test_hand_command_expires_after_arm_publish_before_sdk_submission(robot_with_mock_hand, limited_by):
    robot, command, clock, _ = robot_with_mock_hand
    robot.hand.consecutive_write_crc = 1
    if limited_by == "feedback":
        clock.now_ns += int(robot.hand_config.feedback_max_age_s * 1e9) - 10_000_000
    deadline_ns = clock.now_ns + 10_000_000
    expires_ns = deadline_ns if limited_by == "command" else deadline_ns + 1_000_000_000
    robot.arm.submit.side_effect = lambda *a, **kw: setattr(clock, "now_ns", deadline_ns)
    with pytest.raises(CommandExpired, match="expired before SDK submission"):
        robot.send_action(command, expires_ns=expires_ns)
    robot.arm.submit.assert_called_once()
    robot.hand.control.send_command.assert_not_called()
    assert robot.runtime_state.stop_signal.is_set()
    assert not robot.hand_feedback_pending and not robot.hand_stop_required
    assert robot.hand.crc_counts["unconfirmed_actions"] == 0


@pytest.mark.parametrize("limited_by", ["command", "feedback"])
def test_write_crc_respects_command_and_feedback_deadlines(robot_with_mock_hand, limited_by):
    robot, command, clock, sleep = robot_with_mock_hand
    robot.hand.control.send_command.return_value = sdk_error(CRC)
    if limited_by == "feedback":
        clock.now_ns += int(robot.hand_config.feedback_max_age_s * 1e9) - 10_000_000
    deadline_ns = clock.now_ns + (10_000_000 if limited_by == "command" else 1_000_000_000)
    with pytest.raises(HandCommandUnconfirmed):
        robot.send_action(command, expires_ns=deadline_ns)
    robot.hand.control.send_command.assert_called_once()
    sleep.assert_not_called()


def test_write_crc_rechecks_stop_after_backoff(robot_with_mock_hand):
    robot, command, clock, sleep = robot_with_mock_hand
    robot.hand.control.send_command.return_value = sdk_error(CRC)
    sleep.side_effect = lambda _: robot.runtime_state.stop_signal.set()
    with pytest.raises(OperationCancelled, match="device operation cancelled"):
        robot.send_action(command, expires_ns=clock.now_ns + 1_000_000_000)
    robot.hand.control.send_command.assert_called_once()
    assert robot.hand.crc_counts["unconfirmed_actions"] == 1
    assert robot.hand_feedback_pending


@pytest.mark.parametrize("code", [123, driver.COMBINED_FORCE_UNAVAILABLE_CODE, driver.OVERCURRENT_WARNING_CODE])
def test_non_crc_sdk_error_propagates_without_retry(robot_with_mock_hand, code):
    robot, command, clock, sleep = robot_with_mock_hand
    hand = robot.hand
    hand.consecutive_write_crc = 1
    hand.control.send_command.return_value = sdk_error(code)
    with pytest.raises(driver.XHandError, match=f"code={code}"):
        robot.send_action(command, expires_ns=clock.now_ns + 1_000_000_000)
    hand.control.send_command.assert_called_once()
    assert hand.consecutive_write_crc == 1
    assert hand.crc_counts["write_recoveries"] == 0
    sleep.assert_not_called()
    assert robot.runtime_state.stop_signal.is_set()
    assert robot.hand_feedback_pending


def test_sdk_error_interrupts_crc_retry_and_keeps_feedback_pending(robot_with_mock_hand):
    robot, command, clock, _ = robot_with_mock_hand
    hand = robot.hand
    hand.control.send_command.side_effect = [sdk_error(CRC), sdk_error(123)]
    with pytest.raises(driver.XHandError, match="code=123"):
        robot.send_action(command, expires_ns=clock.now_ns + 1_000_000_000)
    assert hand.control.send_command.call_count == 2
    assert hand.crc_counts["unconfirmed_actions"] == 1
    assert robot.runtime_state.stop_signal.is_set() and robot.hand_feedback_pending


@pytest.mark.parametrize("stale", [False, True])
def test_read_crc_returns_only_fresh_valid_retry_payload(robot_with_mock_hand, stale):
    robot, command, clock, sleep = robot_with_mock_hand
    hand = robot.hand
    valid_state = sdk_state(command.hand_qpos)

    def read_state(*_):
        if hand.control.read_state.call_count == 1:
            return sdk_error(CRC), object()
        if stale:
            clock.now_ns += int(hand.cfg.feedback_max_age_s * 1e9)
        return sdk_error(driver.COMBINED_FORCE_UNAVAILABLE_CODE), valid_state

    hand.control.read_state.side_effect = read_state
    old_time = clock.now_ns

    state = hand.get_state()

    sleep.assert_called_once_with(hand.cfg.crc_retry_interval_s)
    assert hand.crc_counts["read_errors"] == 1
    if stale:
        assert state is None and hand.consecutive_read_crc == 1
        assert hand.crc_counts["read_recoveries"] == 0
    else:
        np.testing.assert_array_equal(state.qpos, command.hand_qpos)
        assert state.timestamp_ns == clock.now_ns > old_time
        assert hand.consecutive_read_crc == 0
        assert hand.crc_counts["read_recoveries"] == 1


def test_read_deadline_prevents_retry_and_rejects_late_success(robot_with_mock_hand):
    robot, command, clock, sleep = robot_with_mock_hand
    hand = robot.hand
    feedback = sdk_state(command.hand_qpos)

    def read_delayed(*_):
        clock.now_ns += 10_000_000
        return sdk_error(0), feedback

    hand.control.read_state.side_effect = read_delayed
    assert hand.get_state(deadline_ns=clock.now_ns + 5_000_000) is None

    hand.control.read_state.side_effect = [(sdk_error(CRC), object())]
    assert hand.get_state(deadline_ns=clock.now_ns + 20_000_000) is None
    assert hand.control.read_state.call_count == 2
    sleep.assert_not_called()


def test_persistent_read_crc_preserves_cache_and_only_stops_after_expiry(robot_with_mock_hand):
    robot, command, clock, _ = robot_with_mock_hand
    hand = robot.hand
    hand.control.read_state.return_value = sdk_error(CRC), object()
    old_frame = robot.hand_feedback.snapshot()
    assert hand.get_state() is None
    assert robot.service_feedback()
    assert robot.hand_feedback.snapshot() is old_frame
    assert robot.hand_feedback_pending
    assert not robot.runtime_state.stop_signal.is_set()
    assert hand.crc_counts["read_errors"] == 4

    with pytest.raises(FeedbackUnavailable, match="pending after CRC"):
        robot.send_action(command, expires_ns=clock.now_ns + 1_000_000_000)
    robot.arm.submit.assert_not_called()
    hand.control.send_command.assert_not_called()
    assert not robot.runtime_state.stop_signal.is_set()

    clock.now_ns += 200_000_000
    with pytest.raises(FeedbackUnavailable, match="feedback unavailable/stale"):
        robot.service_feedback()
    assert robot.runtime_state.stop_signal.is_set()


def test_teleop_skips_pending_feedback_and_commits_only_confirmed_action(monkeypatch):
    state, robot, controller = SessionState(), Mock(), Mock()
    controller.anchor = None
    controller.reset.side_effect = lambda _: setattr(controller, "anchor", object())
    runtime = SimpleNamespace(teleop=SimpleNamespace(target_max_age_s=0.25))
    loop = teleop.TeleopLoop(
        state, runtime, robot, control_dt_ns=50_000_000, task_label="test", sensors=Mock(),
        controller=controller, recording_dir=None, max_duration_s=None, keyboard=Mock(), audio=Mock(),
    )
    loop.episode_cancel_epoch = state.cancel_epoch
    loop.last_command = loop.target_deadline_ns = loop.hold_reason = None
    loop.recorder = Mock()
    loop.recorder.append.return_value = None
    monkeypatch.setattr(loop, "admit_observation", Mock(return_value=[]))
    controller.compute_target.return_value = SimpleNamespace(
        arm_qpos=np.zeros(7), hand_qpos=np.zeros(12), eef_pose=None, rejection_reason=None,
    )
    clock = SimpleNamespace(now_ns=1_000_000_000)
    monkeypatch.setattr(teleop, "time", SimpleNamespace(monotonic_ns=lambda: clock.now_ns))
    read_observation = Mock(side_effect=lambda *a, **kw: SimpleNamespace(
        control_timestamp_ns=clock.now_ns, arm={"qpos": np.zeros((1, 7))},
    ))
    monkeypatch.setattr(teleop, "read_observation", read_observation)

    robot.hand_feedback_pending = True
    loop.run_control_tick()
    read_observation.assert_not_called()
    robot.send_action.assert_not_called()

    robot.hand_feedback_pending = False
    clock.now_ns += loop.control_dt_ns
    loop.run_control_tick()
    controller.reset.assert_called_once()
    controller.commit.assert_called_once()
    assert loop.last_command is not None
    assert loop.recorder.append.call_args.kwargs["action_accepted"] is True


@pytest.mark.parametrize(("has_confirmed_target", "late_success"), [(False, False), (True, False), (True, True)])
def test_teleop_ends_episode_after_moved_write_without_on_time_ack(
    monkeypatch, robot_with_mock_hand, has_confirmed_target, late_success,
):
    from dexmani_real.teleop.retargeting.solver import HandRetargeter

    robot, command, clock, _ = robot_with_mock_hand
    state = robot.runtime_state
    cfg = TeleopConfig()
    runtime = SimpleNamespace(arm=robot.arm_config, hand=robot.hand_config, teleop=cfg)
    arm_limits = np.column_stack((robot.arm_lower, robot.arm_upper))
    arm_solver = Mock(limits=arm_limits, workspace=np.tile([-1.0, 1.0], (3, 1)))
    arm_solver.measured_reference.side_effect = lambda qpos: qpos
    arm_solver.compute_eef_pose_base.return_value = Pose(np.zeros(3), [1, 0, 0, 0])
    arm_solver.solve.return_value = SimpleNamespace(qpos=command.arm_qpos)
    mapper = ActionMapper(
        joint_limits=arm_limits, hand_enabled=True, hand_lower_rad=robot.hand_lower,
        hand_upper_rad=robot.hand_upper, solver=arm_solver,
    )
    retargeter = HandRetargeter(HandRetargetingConfig(), robot.hand_config)
    calibration = retargeter.calibration = object()
    controller = VRController(mapper, retargeter, cfg, np.eye(3))
    sensors = Mock()
    loop = teleop.TeleopLoop(
        state, runtime, robot, control_dt_ns=50_000_000, task_label="test", sensors=sensors,
        controller=controller, recording_dir=None, max_duration_s=None, keyboard=Mock(), audio=Mock(),
    )
    loop.episode_cancel_epoch = state.cancel_epoch
    loop.last_command = loop.target_deadline_ns = loop.hold_reason = None
    loop.recorder = Mock()
    mock_time = SimpleNamespace(monotonic_ns=lambda: clock.now_ns)
    monkeypatch.setattr(teleop, "time", mock_time)
    monkeypatch.setattr(observation, "time", mock_time)

    wrist_pos = np.zeros(3)
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    robot.arm.snapshot.return_value = arm

    def tick():
        if loop.next_tick_ns is not None:
            clock.now_ns = max(clock.now_ns, loop.next_tick_ns)
        arm["timestamp_ns"] = clock.now_ns
        sensors.channels.vr_ring.read.return_value = ({
            "recv_ts_ns": clock.now_ns, "wrist_pos": wrist_pos,
            "wrist_quat_wxyz": np.array([1, 0, 0, 0]), "landmarks": None,
        }, 1)
        loop.run_control_tick()

    lost_ack = False
    measured_qpos = command.hand_qpos.copy()

    def send(_device, sdk_command):
        if lost_ack:
            measured_qpos[0] = (command.hand_qpos[0] + sdk_command.finger_command[0].position) / 2
            if late_success:
                clock.now_ns += int(robot.hand_config.feedback_max_age_s * 1e9)
                return sdk_error(0)
            return sdk_error(CRC)
        return sdk_error(0)

    def append(_row, attempted, *, action_accepted):
        if lost_ack:
            assert state.stop_signal.is_set() and robot.hand_feedback_pending
            assert not action_accepted
            assert attempted.hand_qpos[0] > measured_qpos[0]

    loop.recorder.append.side_effect = append

    def retarget(_landmarks, *, timestamp_ns):
        reference = retargeter.reference
        assert reference is not None
        target = reference[0].copy()
        if lost_ack:
            target[0] += 0.1
        return target

    robot.hand.control.send_command.side_effect = send
    retarget_call = Mock(side_effect=retarget)
    monkeypatch.setattr(retargeter, "retarget", retarget_call)
    if has_confirmed_target:
        tick()
    deadline_ns = loop.target_deadline_ns
    last_command = loop.last_command
    if has_confirmed_target:
        previous_eef = controller.previous_eef_target
        reference_before = retargeter.reference
        wrist_pos = np.array([0.1, 0, 0])

    lost_ack = True
    tick()
    send_count = robot.hand.control.send_command.call_count
    assert send_count == 1 + has_confirmed_target + (not late_success)
    assert measured_qpos[0] > command.hand_qpos[0]
    if has_confirmed_target:
        assert retargeter.reference is reference_before
        assert controller.previous_eef_target is previous_eef
    assert robot.hand_feedback_pending and robot.hand.consecutive_read_crc == 0
    assert loop.target_deadline_ns == deadline_ns and loop.last_command is last_command
    assert loop.pending_termination_reason == "cancelled" and state.stop_signal.is_set()
    issue = loop.termination_details[0]["reason"]
    assert ("SDK success reply arrived at or after" if late_success else "delivery unconfirmed") in issue
    assert loop.recorder.append.call_count == 1 + has_confirmed_target

    lost_ack = False
    retarget_call.reset_mock()
    with monkeypatch.context() as read_patch:
        read_patch.setattr(robot.hand, "get_state", lambda: None)
        assert not robot.hold_hand()
    assert robot.hand_feedback_pending
    tick()
    retarget_call.assert_not_called()
    assert robot.hand.control.send_command.call_count == send_count

    robot.hand.control.read_state.return_value = sdk_error(0), sdk_state(measured_qpos)
    robot.stop()
    assert not robot.hand_feedback_pending
    assert retargeter.calibration is calibration
    tick()
    retarget_call.assert_not_called()
    assert robot.hand.control.send_command.call_count == send_count + 1
    assert robot.arm.submit.call_count == 1 + has_confirmed_target
    sent_qpos = [joint.position for joint in robot.hand.command.finger_command]
    np.testing.assert_array_equal(sent_qpos, measured_qpos)


def test_hold_retries_measured_position_and_preserves_torque_limit(robot_with_mock_hand):
    robot, command, clock, sleep = robot_with_mock_hand
    hand = robot.hand
    # HOME 保持沿用已发送的低输出限制。
    for joint in hand.command.finger_command:
        joint.tor_max = 77
    assert hand.send_position(command.hand_qpos, tor_max=hand.cfg.tor_max, deadline_ns=clock.now_ns) is None
    hand.control.send_command.assert_not_called()
    positions = []

    def send(_device, sdk_command):
        assert all(joint.mode == driver.POSITION_MODE and joint.tor_max == 77
                   for joint in sdk_command.finger_command)
        positions.append([joint.position for joint in sdk_command.finger_command])
        return sdk_error(CRC if len(positions) == 1 else 0)

    hand.control.send_command.side_effect = send
    assert hand.hold(command.hand_qpos, timestamp_ns=clock.now_ns)
    np.testing.assert_array_equal(positions, np.repeat(command.hand_qpos[None], 2, axis=0))
    sleep.assert_called_once_with(hand.cfg.crc_retry_interval_s)


@pytest.mark.parametrize("invalid", ["stale", "nonfinite"])
def test_hold_rejects_unusable_feedback_without_sending(robot_with_mock_hand, invalid):
    robot, command, clock, _ = robot_with_mock_hand
    hand = robot.hand
    qpos, stamp = command.hand_qpos, clock.now_ns
    if invalid == "stale":
        stamp -= int(hand.hold_feedback_max_age_s * 1e9)
    else:
        qpos = np.full_like(qpos, np.nan)

    assert hand.hold(qpos, timestamp_ns=stamp) is False
    hand.control.send_command.assert_not_called()


@pytest.mark.parametrize("acknowledged", [True, False])
def test_release_sends_only_passive_and_logs_result(robot_with_mock_hand, caplog, acknowledged):
    robot, command, _, _ = robot_with_mock_hand
    hand = robot.hand
    for joint, position in zip(hand.command.finger_command, command.hand_qpos):
        joint.position = position

    def send(_device, sdk_command):
        assert all(joint.mode == driver.PASSIVE_MODE for joint in sdk_command.finger_command)
        np.testing.assert_array_equal([joint.position for joint in sdk_command.finger_command], command.hand_qpos)
        return sdk_error(0 if acknowledged else CRC)

    hand.control.send_command.side_effect = send
    with caplog.at_level(logging.INFO, logger=driver.__name__):
        assert hand.release() is acknowledged

    result = "acknowledged" if acknowledged else "unconfirmed"
    assert f"XHand release: Passive command {result}" in caplog.text
    assert hand.control.send_command.call_count == (1 if acknowledged else 2)
    hand.control.read_state.assert_not_called()


@pytest.mark.parametrize("operation", ["hold", "release"])
def test_overcurrent_reply_does_not_confirm_hold_or_release(robot_with_mock_hand, operation):
    robot, command, clock, sleep = robot_with_mock_hand
    hand = robot.hand
    hand.control.send_command.return_value = sdk_error(driver.OVERCURRENT_WARNING_CODE)

    result = hand.hold(command.hand_qpos, timestamp_ns=clock.now_ns) if operation == "hold" else hand.release()

    assert result is False
    hand.control.send_command.assert_called_once()
    sleep.assert_not_called()


def test_robot_stop_propagates_failed_hold_without_releasing(robot_with_mock_hand):
    robot, _, _, _ = robot_with_mock_hand
    hand = robot.hand
    robot.hand_stop_required = True
    hand.consecutive_write_crc = 5

    def send(_device, sdk_command):
        assert robot.runtime_state.stop_signal.is_set()
        assert all(joint.mode == driver.POSITION_MODE for joint in sdk_command.finger_command)
        return sdk_error(CRC)

    hand.control.send_command.side_effect = send
    with pytest.raises(FeedbackUnavailable, match="position hold not confirmed"):
        robot.stop()

    assert robot.hand_stop_required
    assert hand.control.send_command.call_count == hand.cfg.crc_retries + 1
    assert hand.crc_counts["unconfirmed_actions"] == 0
    robot.arm.wait_physical_stop.assert_called_once()


def test_pause_and_normal_exit_reuse_confirmed_hold(robot_with_mock_hand):
    robot, command, _, _ = robot_with_mock_hand
    robot.hand_stop_required = True
    sdk = robot.hand.control
    sdk.send_command.return_value = sdk_error(0)

    robot.stop()
    close_runtime(robot.runtime_state, robot)

    sdk.send_command.assert_called_once()
    joints = sdk.send_command.call_args.args[1].finger_command
    assert all(joint.mode == driver.POSITION_MODE for joint in joints)
    np.testing.assert_array_equal([joint.position for joint in joints], command.hand_qpos)
    sdk.close_device.assert_called_once()


def test_normal_exit_refreshes_stale_pose_before_sending_hold(robot_with_mock_hand):
    robot, command, clock, _ = robot_with_mock_hand
    robot.hand_stop_required = True
    clock.now_ns += 250_000_000
    sdk = robot.hand.control
    measured = command.hand_qpos + 0.01
    sdk.read_state.return_value = sdk_error(driver.COMBINED_FORCE_UNAVAILABLE_CODE), sdk_state(measured)
    sdk.send_command.return_value = sdk_error(0)

    close_runtime(robot.runtime_state, robot)

    assert [entry[0] for entry in sdk.mock_calls] == ["read_state", "send_command", "close_device"]
    joints = sdk.send_command.call_args.args[1].finger_command
    assert all(joint.mode == driver.POSITION_MODE for joint in joints)
    np.testing.assert_array_equal([joint.position for joint in joints], measured)


def test_normal_exit_reports_unavailable_hold_and_still_closes_devices(robot_with_mock_hand):
    robot, _, clock, _ = robot_with_mock_hand
    robot.hand_stop_required = True
    clock.now_ns += 250_000_000
    arm, sdk = robot.arm, robot.hand.control
    sdk.read_state.return_value = sdk_error(CRC), object()

    with pytest.raises(FeedbackUnavailable, match="position hold not confirmed during close"):
        close_runtime(robot.runtime_state, robot)

    sdk.send_command.assert_not_called()
    sdk.close_device.assert_called_once()
    arm.close.assert_called_once()


@pytest.mark.parametrize("acknowledged", [True, False])
def test_explicit_exit_release_is_reported_before_disconnect(robot_with_mock_hand, acknowledged):
    robot, _, _, _ = robot_with_mock_hand
    robot.hand_stop_required = True
    sdk = robot.hand.control
    sdk.send_command.return_value = sdk_error(0 if acknowledged else CRC)

    if acknowledged:
        close_runtime(robot.runtime_state, robot, release_hand=True)
    else:
        with pytest.raises(RobotError, match="release not confirmed"):
            close_runtime(robot.runtime_state, robot, release_hand=True)

    assert [entry[0] for entry in sdk.mock_calls] == ["send_command"] * (
        1 if acknowledged else 2
    ) + ["close_device"]
    assert all(joint.mode == driver.PASSIVE_MODE for joint in sdk.send_command.call_args.args[1].finger_command)
    assert not robot.runtime_state.estop_request


def test_explicit_release_requires_arm_to_be_physically_stopped(robot_with_mock_hand):
    robot, _, _, _ = robot_with_mock_hand
    arm, sdk = robot.arm, robot.hand.control
    arm.wait_physical_stop.side_effect = TimeoutError("arm still moving")
    robot.hand_stop_required = True
    sdk.send_command.return_value = sdk_error(0)

    with pytest.raises(TimeoutError, match="arm still moving"):
        close_runtime(robot.runtime_state, robot, release_hand=True)

    arm.wait_physical_stop.assert_called_once()
    sdk.send_command.assert_called_once()
    assert all(joint.mode == driver.POSITION_MODE for joint in sdk.send_command.call_args.args[1].finger_command)
    arm.close.assert_called_once()
    sdk.close_device.assert_called_once()


@pytest.mark.parametrize("final_code", [0, driver.OVERCURRENT_WARNING_CODE, CRC])
def test_home_uses_bounded_crc_retry_without_losing_overcurrent_warning(robot_with_mock_hand, final_code):
    robot, command, clock, _ = robot_with_mock_hand
    robot.hand_config = replace(robot.hand_config, home_tor_max=77)
    sent = []

    def send(_device, sdk_command):
        sent.append([(joint.position, joint.tor_max) for joint in sdk_command.finger_command])
        return sdk_error(CRC if len(sent) == 1 else final_code)

    robot.hand.control.send_command.side_effect = send
    if final_code == CRC:
        with pytest.raises(FeedbackUnavailable, match="HOME XHand position submission unconfirmed"):
            robot.send_hand_home(command.hand_qpos, deadline_ns=clock.now_ns + 1_000_000_000, check_abort=Mock())
    else:
        warned = robot.send_hand_home(command.hand_qpos, deadline_ns=clock.now_ns + 1_000_000_000, check_abort=Mock())
        assert warned is (final_code == driver.OVERCURRENT_WARNING_CODE)
    assert sent == [list(zip(command.hand_qpos, [77] * 12))] * 2


def test_hoi_brakes_arm_during_hand_crc_and_holds_measured_posture_after_recovery(robot_with_mock_hand, monkeypatch):
    robot, _, clock, _ = robot_with_mock_hand
    measured = robot.hand_config.home_qpos_rad
    measured[2] = 0.0  # 合法实测姿态，位于普通动作的操作下界之外。
    robot.hand_feedback.consume(XHandState(measured, None, None, clock.now_ns))
    robot.hand.control.send_command.return_value = sdk_error(0)
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    arm["timestamp_ns"] = clock.now_ns
    robot.arm.snapshot.return_value = arm
    state = robot.runtime_state
    runtime = SimpleNamespace(arm=robot.arm_config, hand=robot.hand_config, execution=ExecutionConfig(control_dt_s=0.05))
    rollout = deployment.Rollout(
        state, runtime, SimpleNamespace(n_obs_steps=1), episode_cancel_epoch=state.cancel_epoch,
        robot=robot, observations=SimpleNamespace(needs=SimpleNamespace(hoi=True, hoi_joint_devices=())),
        sensors=SimpleNamespace(channels=None), action_mapper=None, model=Mock(),
        max_duration_s=None, index=0, output_dir=None, record_raw=False, effective_parameters={},
    )
    rollout.started_ns = clock.now_ns
    monkeypatch.setattr(deployment, "time", SimpleNamespace(monotonic_ns=lambda: clock.now_ns))
    monkeypatch.setattr(deployment, "check_hoi_status", lambda *_: ["hoi_stale"])
    robot.hand.consecutive_read_crc = 1

    rollout.step()

    sent_command = robot.arm.submit.call_args.args[0]
    assert sent_command.arm_hold and sent_command.hand_qpos is None
    assert rollout.state == "HOI_HOLD" and not state.stop_signal.is_set()
    robot.hand.control.send_command.assert_not_called()

    robot.hand.consecutive_read_crc = 0
    rollout._refresh_hoi_hold()

    joints = robot.hand.control.send_command.call_args.args[1].finger_command
    assert all(joint.mode == driver.POSITION_MODE for joint in joints)
    np.testing.assert_array_equal([joint.position for joint in joints], measured)

import logging
import sys
import time
from collections import deque
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.control.constraints import joint_target_rejection
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import (
    GOAL_DTYPE,
    MAX_TIME,
    LifecycleReply,
    PrepareRequest,
)
from dexmani_real.robot.xarm7.trajectory import ArmTrajectory
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.robot.xhand.config import HandConfig
from dexmani_real.robot.xhand.driver import (
    COMBINED_FORCE_UNAVAILABLE_CODE,
    DISTRIBUTED_FORCE_UNAVAILABLE_CODE,
    TEMPERATURE_UNAVAILABLE_CODE,
    XHand,
    XHandError,
)
from dexmani_real.robot.xhand.driver import (
    COMMUNICATION_CRC_ERROR_CODE as CRC,
)
from dexmani_real.robot.xhand.feedback import XHandFeedback
from dexmani_real.runtime.observation import ObservationNeeds, check_observation
from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase
from scripts.tools.control_xhand import XHandControlExample


@pytest.mark.parametrize("side,bad", [("arm", np.full(7, np.nan)), ("hand", np.full(12, np.nan)),
                                     ("hand", np.zeros(11)), ("arm", np.full(7, 100)),
                                     ("hand", np.full(12, 100))])
def test_combined_preflight_precedes_both_actuators(robot_fixture, clock, side, bad):
    env = robot_fixture
    arm = np.asarray(env.runtime.arm.home_qpos)
    hand = np.deg2rad(env.runtime.hand.home_qpos_deg)
    command = RobotCommand(env.state.run_id, bad if side == "arm" else arm,
                           bad if side == "hand" else hand)
    with pytest.raises(ValueError):
        env.robot.send_action(command, valid_until_ns=clock.now + 100, execute_until_ns=clock.now + 100)
    assert not env.sent and not env.hand_sent


def test_home_preflight_and_cancel_across_preparation(robot_fixture, clock, monkeypatch):
    env = robot_fixture
    begins = []

    def begin(run_id):
        begins.append(run_id)
        env.state.stop_signal.clear()

    monkeypatch.setattr(env.robot.arm, "begin_operation", begin)
    with pytest.raises(ValueError):
        env.robot.send_hand_home(RobotCommand(env.state.run_id, hand_qpos=np.full(12, np.nan)),
                                  valid_until_ns=clock.now + 100)
    assert not begins and not env.hand_sent
    command = RobotCommand(env.state.run_id, hand_qpos=np.deg2rad(env.runtime.hand.home_qpos_deg))
    env.robot.send_hand_home(command, valid_until_ns=clock.now + 100)
    assert len(begins) == len(env.hand_sent) == 1

    def cancelled_begin(run_id):
        env.state.run_signal.value += 1

    monkeypatch.setattr(env.robot.arm, "begin_operation", cancelled_begin)
    with pytest.raises(OperationCancelled):
        env.robot.send_hand_home(command, valid_until_ns=clock.now + 100)
    assert len(env.hand_sent) == 1


def make_hand(send_codes=(), read_codes=(), *, config=None):
    cfg = HandConfig() if config is None else config
    hand = XHand(cfg)
    hand.connected_flag = True
    hand.command = SimpleNamespace(finger_command=[SimpleNamespace() for _ in range(12)])

    def force():
        return SimpleNamespace(fx=1., fy=2., fz=3.)

    raw = SimpleNamespace(
        finger_state=[SimpleNamespace(id=i, position=float(q), torque=None,
                                      commboard_err=0, jonitboard_err=0, tipboard_err=0)
                      for i, q in enumerate(np.deg2rad(cfg.home_qpos_deg))],
        sensor_data=[SimpleNamespace(calc_force=force(), raw_force=[force() for _ in range(120)])
                     for _ in range(5)],
    )
    sends, reads = deque(send_codes), deque(read_codes)
    history = []

    def send(device, command):
        history.append([(joint.mode, getattr(joint, "position", None)) for joint in command.finger_command])
        code = sends.popleft() if sends else 0
        if isinstance(code, BaseException):
            raise code
        return SimpleNamespace(error_code=code, error_message="test reply")

    def read(device, refresh):
        code = reads.popleft() if reads else 0
        if isinstance(code, BaseException):
            raise code
        return SimpleNamespace(error_code=code, error_message="test reply"), raw

    hand.control = SimpleNamespace(send_command=send, read_state=read, raw_state=raw)
    return hand, history


def test_torque_none_and_read_crc_payload_discarded():
    hand, _ = make_hand(read_codes=[CRC, 0])
    assert hand.get_state() is None
    sample = hand.get_state()
    np.testing.assert_allclose(sample.qpos, np.deg2rad(hand.cfg.home_qpos_deg))
    assert not hasattr(sample, "current_ma") and not hasattr(sample, "commboard_err")


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nonfinite"])
def test_joint_payload_integrity_remains_required(fault):
    hand, _ = make_hand()
    joints = hand.control.raw_state.finger_state
    if fault == "missing":
        joints.pop()
    elif fault == "duplicate":
        joints[-1].id = joints[0].id
    else:
        joints[0].position = np.nan
    with pytest.raises(XHandError, match="invalid joint payload"):
        hand.get_state()


def test_joint_ids_determine_positions_and_tip_status_independent_of_payload_order():
    hand, _ = make_hand()
    raw = hand.control.raw_state
    raw.finger_state[2].tipboard_err = 201
    raw.finger_state.reverse()
    sample = hand.get_state()
    np.testing.assert_allclose(sample.qpos, np.deg2rad(hand.cfg.home_qpos_deg))
    assert sample.tactile_aggregate is not None and sample.tactile_dense is None


def test_crc_low_rate_recovery_and_continuous_fault(clock):
    hand, history = make_hand(send_codes=[CRC, 0, *([CRC] * 7), 0])
    target = np.deg2rad(hand.cfg.home_qpos_deg)
    hand.send_action(target)
    for ms in (40, 80, 120, 160):
        clock.at(ms)
        hand.get_state()
    assert hand.send_crc_count == 1  # 读取成功和长时间不发送均不是发送 ACK。
    clock.at(200)
    hand.send_action(target)
    assert hand.send_crc_count == 0
    for count in range(1, 5):
        clock.at(200 + count * 200)
        hand.send_action(target)
        assert hand.send_crc_count == count
    clock.at(1200)
    with pytest.raises(XHandError, match="continuous send CRC"):
        hand.send_action(target)
    assert len(history) == 7 and hand.send_crc_count == 5
    # 普通发送已达上限，停止仍可继续尝试，成功后清零。
    started = clock.now
    hand.stop()
    assert len(history) == 10 and hand.send_crc_count == 0
    assert clock.now - started == 20_000_000
    assert all(mode == 0 for mode, _ in history[-1])


def test_send_crc_returns_without_sleep_or_resending_old_target(clock, monkeypatch):
    hand, history = make_hand(send_codes=[CRC, 0])
    target = np.deg2rad(hand.cfg.home_qpos_deg)
    monkeypatch.setattr(time, "sleep", lambda _: pytest.fail("ordinary send must not sleep"))
    hand.send_action(target)
    assert len(history) == 1 and clock.now == clock.origin
    latest = target + 0.01
    hand.send_action(latest)
    assert len(history) == 2
    np.testing.assert_allclose([position for _, position in history[0]], target)
    np.testing.assert_allclose([position for _, position in history[1]], latest)


@pytest.mark.parametrize("code", [0, COMBINED_FORCE_UNAVAILABLE_CODE,
                                  DISTRIBUTED_FORCE_UNAVAILABLE_CODE, TEMPERATURE_UNAVAILABLE_CODE])
def test_accepted_send_response_resets_crc_sequence(code):
    hand, _ = make_hand(send_codes=[CRC, code, CRC])
    target = np.deg2rad(hand.cfg.home_qpos_deg)
    hand.send_action(target)
    assert hand.send_crc_count == 1
    hand.send_action(target)
    assert hand.send_crc_count == 0
    hand.send_action(target)
    assert hand.send_crc_count == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "5", None])
def test_send_crc_limit_requires_a_positive_integer(limit):
    with pytest.raises(ValueError, match="send_crc_limit"):
        replace(HandConfig(), send_crc_limit=limit).validate()


def test_configured_send_crc_limit():
    config = replace(HandConfig(), send_crc_limit=2)
    config.validate()
    hand, history = make_hand(send_codes=[CRC, CRC], config=config)
    target = np.deg2rad(config.home_qpos_deg)
    hand.send_action(target)
    with pytest.raises(XHandError, match="limit=2"):
        hand.send_action(target)
    assert len(history) == 2


def test_new_connection_resets_crc_count_but_existing_connection_does_not(clock, monkeypatch):
    hand, _ = make_hand(send_codes=[CRC])
    hand.connected_flag = False
    hand.send_crc_count = 4
    sdk = SimpleNamespace(HandCommand_t=lambda: SimpleNamespace(
        finger_command=[SimpleNamespace() for _ in range(12)],
    ))
    monkeypatch.setitem(sys.modules, "xhand_controller", SimpleNamespace(xhand_control=sdk))
    monkeypatch.setattr(hand, "open_device", lambda: None)
    monkeypatch.setattr(hand, "read_identity", lambda: None)
    hand.control.list_hands_id = lambda: [hand.cfg.device_id]
    hand.connect()
    assert hand.send_crc_count == 0
    hand.send_action(np.deg2rad(hand.cfg.home_qpos_deg))
    hand.connect()
    assert hand.send_crc_count == 1


def test_stop_requires_non_crc_response_and_validates_hold(clock):
    hand, history = make_hand(send_codes=[CRC, 0])
    hand.stop(np.full(12, 10.), timestamp_ns=clock.now)
    assert len(history) == 2
    np.testing.assert_allclose([position for _, position in history[-1]], hand.mechanical_upper)
    assert hand.send_crc_count == 0
    with pytest.raises(ValueError):
        hand.hold_current(np.full(12, np.nan))
    with pytest.raises(ValueError):
        hand.stop(np.zeros(11), timestamp_ns=clock.now)
    assert len(history) == 2
    failed, history = make_hand(send_codes=[CRC] * 100)
    with pytest.raises(RuntimeError, match="stop timed out"):
        failed.stop()
    assert all(mode == 0 for mode, _ in history[-1])
    assert 5 < len(history) < 100


@pytest.mark.parametrize("failure", [None, ValueError("SDK read broke"),
                                     XHandError("read", 123456, "unknown SDK error"), KeyboardInterrupt()])
def test_diagnostic_stop_attempts_passive_and_propagates_feedback_errors(clock, monkeypatch, failure):
    hand = XHandControlExample(HandConfig())
    hand.motion_pending = True
    stops = []

    def read():
        if failure is not None:
            raise failure

    monkeypatch.setattr(hand, "get_state", read)
    monkeypatch.setattr(hand, "stop", lambda qpos, **kwargs: stops.append(qpos))
    if failure is None:
        hand.stop_action()
    else:
        with pytest.raises(type(failure)) as caught:
            hand.stop_action()
        assert caught.value is failure
    assert stops == [None] and not hand.motion_pending


def test_diagnostic_stop_failure_keeps_feedback_error_as_context(clock, monkeypatch):
    hand = XHandControlExample(HandConfig())
    hand.motion_pending = True
    read_error = ValueError("SDK read broke")
    stop_error = RuntimeError("SDK stop broke")

    def read():
        raise read_error

    def stop(*args, **kwargs):
        raise stop_error

    monkeypatch.setattr(hand, "get_state", read)
    monkeypatch.setattr(hand, "stop", stop)
    with pytest.raises(RuntimeError) as caught:
        hand.stop_action()
    assert caught.value is stop_error and stop_error.__context__ is read_error
    assert hand.motion_pending


def test_read_crc_holds_trustworthy_snapshot_and_recovers(clock):
    hand, _ = make_hand(read_codes=[0, CRC, 0, *([CRC] * 4)])
    feedback = XHandFeedback(hand.cfg)
    feedback.reset(clock.now)
    first = feedback.consume(hand.get_state(), source_ns=clock.now, now_ns=clock.now)
    original = first.tobytes()
    hand.control.raw_state.finger_state[0].position += 0.01
    clock.at(40)
    held = feedback.consume(hand.get_state(), source_ns=clock.now, now_ns=clock.now)
    assert held is first and held.tobytes() == original
    clock.at(80)
    recovered = feedback.consume(hand.get_state(), source_ns=clock.now, now_ns=clock.now)
    assert int(recovered["timestamp_ns"][0]) == clock.now
    assert recovered["qpos"][0, 0] != first["qpos"][0, 0]
    assert first.tobytes() == original and not first.flags.writeable
    for ms in (120, 160, 200):
        clock.at(ms)
        assert feedback.consume(hand.get_state(), source_ns=clock.now, now_ns=clock.now) is recovered
    clock.at(215)
    with pytest.raises(RobotError, match="unavailable/stale"):
        feedback.consume(hand.get_state(), source_ns=clock.now, now_ns=clock.now)


@pytest.mark.parametrize("channel", ["aggregate", "dense"])
@pytest.mark.parametrize("fault", ["missing", "type", "shape", "nan", "inf", "overflow",
                                   "integer_overflow", "conversion"])
def test_bad_tactile_payload_only_discards_its_channel(channel, fault):
    hand, _ = make_hand()
    sensor = hand.control.raw_state.sensor_data[0]
    if fault == "missing":
        delattr(sensor, "calc_force" if channel == "aggregate" else "raw_force")
    elif fault == "shape":
        if channel == "aggregate":
            sensor.calc_force.fx = [1., 2.]
        else:
            sensor.raw_force.pop()
    else:
        force = sensor.calc_force if channel == "aggregate" else sensor.raw_force[0]
        force.fx = {"type": object(), "nan": np.nan, "inf": np.inf,
                    "overflow": 1e39, "integer_overflow": 10 ** 1000,
                    "conversion": "not a number"}[fault]
    sample = hand.get_state()
    np.testing.assert_allclose(sample.qpos, np.deg2rad(hand.cfg.home_qpos_deg))
    assert getattr(sample, f"tactile_{channel}") is None
    other = "dense" if channel == "aggregate" else "aggregate"
    valid = getattr(sample, f"tactile_{other}")
    assert valid.dtype == np.float32 and np.isfinite(valid).all()
    assert valid.shape == ((5, 120, 3) if other == "dense" else (5, 3))


@pytest.mark.parametrize("channel", ["aggregate", "dense"])
def test_whole_tactile_channel_holds_source_time_and_keeps_old_snapshots(clock, channel):
    hand, _ = make_hand()
    raw = hand.control.raw_state
    feedback = XHandFeedback(hand.cfg)
    feedback.reset(clock.now)
    first = feedback.consume(hand.get_state(), source_ns=clock.now, now_ns=clock.now)
    original = first.tobytes()
    sensor = raw.sensor_data[0]
    bad_force = sensor.calc_force if channel == "aggregate" else sensor.raw_force[0]
    bad_force.fx = np.nan
    # 同一坏帧内其他有效向量也不能推进该通道的值或来源时间。
    raw.sensor_data[1].calc_force.fx = 9.
    raw.sensor_data[1].raw_force[1].fx = 9.
    raw.finger_state[0].position += 0.01
    clock.at(40)
    sample = hand.get_state()
    held = feedback.consume(sample, source_ns=clock.now, now_ns=clock.now)
    np.testing.assert_array_equal(held[f"tactile_{channel}"], first[f"tactile_{channel}"])
    assert int(held[f"tactile_{channel}_timestamp_ns"][0]) == clock.origin
    other = "dense" if channel == "aggregate" else "aggregate"
    np.testing.assert_array_equal(held[f"tactile_{other}"][0], getattr(sample, f"tactile_{other}"))
    assert int(held[f"tactile_{other}_timestamp_ns"][0]) == clock.now
    assert int(held["timestamp_ns"][0]) == clock.now
    assert held["qpos"][0, 0] != first["qpos"][0, 0]
    held_original = held.tobytes()
    bad_force.fx = 8.
    clock.at(80)
    sample = hand.get_state()
    recovered = feedback.consume(sample, source_ns=clock.now, now_ns=clock.now)
    np.testing.assert_array_equal(recovered[f"tactile_{channel}"][0], getattr(sample, f"tactile_{channel}"))
    assert int(recovered[f"tactile_{channel}_timestamp_ns"][0]) == clock.now
    assert first.tobytes() == original and held.tobytes() == held_original
    assert not any(frame.flags.writeable for frame in (first, held, recovered))


def test_first_failed_tactile_channels_have_nan_values_and_zero_source_times(clock):
    hand, _ = make_hand()
    sensor = hand.control.raw_state.sensor_data[0]
    sensor.calc_force = None
    sensor.raw_force = None
    sample = hand.get_state()
    assert sample.tactile_aggregate is sample.tactile_dense is None
    feedback = XHandFeedback(hand.cfg)
    feedback.reset(clock.now)
    frame = feedback.consume(sample, source_ns=clock.now, now_ns=clock.now)
    for channel in ("aggregate", "dense"):
        assert np.isnan(frame[f"tactile_{channel}"]).all()
        assert int(frame[f"tactile_{channel}_timestamp_ns"][0]) == 0
    assert int(frame["timestamp_ns"][0]) == clock.now


def test_first_joint_payload_cannot_bypass_freshness_with_initialization_budget(clock):
    hand, _ = make_hand()
    feedback = XHandFeedback(hand.cfg)
    feedback.reset(clock.now)
    clock.at(135)
    with pytest.raises(RobotError, match="unavailable/stale"):
        feedback.consume(hand.get_state(), source_ns=clock.origin, now_ns=clock.now)


@pytest.mark.parametrize("comm,sdk_code,tip_code,allowed", [
    ("serial", 0, 0, (True, True)),
    ("serial", 0, 200, (False, False)),
    ("serial", 0, 201, (True, False)),
    ("serial", 0, 202, (True, True)),
    ("serial", COMBINED_FORCE_UNAVAILABLE_CODE, 0, (False, False)),
    ("serial", DISTRIBUTED_FORCE_UNAVAILABLE_CODE, 0, (True, False)),
    ("serial", TEMPERATURE_UNAVAILABLE_CODE, 0, (True, True)),
    ("ethercat", 0, 201, (False, False)),
    ("ethercat", TEMPERATURE_UNAVAILABLE_CODE, 0, (False, False)),
    ("ethercat", 0, 0, (True, True)),
    ("serial", 0, 999, (False, False)),
])
def test_tactile_status_semantics_gate_payload_access(comm, sdk_code, tip_code, allowed):
    hand, _ = make_hand(read_codes=[sdk_code], config=replace(HandConfig(), comm_type=comm))
    raw = hand.control.raw_state
    raw.finger_state[2].tipboard_err = tip_code

    class Sensor:
        def __init__(self, original):
            self.original = original

        def __getattr__(self, name):
            assert allowed[0 if name == "calc_force" else 1], "unavailable payload accessed"
            return getattr(self.original, name)

    raw.sensor_data = [Sensor(sensor) for sensor in raw.sensor_data]
    sample = hand.get_state()
    assert (sample.tactile_aggregate is not None, sample.tactile_dense is not None) == allowed


@pytest.mark.parametrize("status", [None, "invalid", np.inf, "missing"])
def test_invalid_tactile_status_keeps_valid_joints_without_accessing_payload(status):
    hand, _ = make_hand()
    raw = hand.control.raw_state
    if status == "missing":
        del raw.finger_state[2].tipboard_err
    else:
        raw.finger_state[2].tipboard_err = status
    del raw.sensor_data
    sample = hand.get_state()
    assert sample.tactile_aggregate is sample.tactile_dense is None
    np.testing.assert_allclose(sample.qpos, np.deg2rad(hand.cfg.home_qpos_deg))


@pytest.mark.parametrize("operation", ["read", "send"])
@pytest.mark.parametrize("failure", [1_501_035, 123456, ValueError("SDK broke"), RuntimeError("SDK broke"),
                                     AttributeError("SDK broke"), TypeError("SDK broke")])
def test_hardware_errors_and_unknown_sdk_failures_propagate(operation, failure):
    hand, _ = make_hand(**{f"{operation}_codes": [failure]})
    with pytest.raises(type(failure) if isinstance(failure, Exception) else XHandError) as caught:
        if operation == "read":
            hand.get_state()
        else:
            hand.send_action(np.deg2rad(hand.cfg.home_qpos_deg))
    if isinstance(failure, Exception):
        assert caught.value is failure
    else:
        assert caught.value.code == failure


def test_tactile_parser_does_not_swallow_unexpected_program_errors():
    hand, _ = make_hand()
    original = RuntimeError("unexpected tactile accessor failure")

    class BrokenSensor:
        @property
        def calc_force(self):
            raise original

    hand.control.raw_state.sensor_data[0] = BrokenSensor()
    with pytest.raises(RuntimeError) as failure:
        hand.get_state()
    assert failure.value is original


@pytest.mark.parametrize("field", ["commboard_err", "jonitboard_err"])
def test_joint_board_fault_is_not_masked_by_nonfatal_tactile_sdk_code(field):
    hand, _ = make_hand(read_codes=[COMBINED_FORCE_UNAVAILABLE_CODE])
    setattr(hand.control.raw_state.finger_state[0], field, 1)
    with pytest.raises(XHandError, match="joint/communication board fault"):
        hand.get_state()


def test_fault_logs_only_report_entry_and_recovery(clock, caplog):
    caplog.set_level(logging.INFO)
    hand, _ = make_hand(read_codes=[CRC, CRC, 0, 0], send_codes=[CRC, CRC, 0, 0])
    feedback = XHandFeedback(hand.cfg)
    feedback.reset(clock.now)
    for _ in range(4):
        feedback.consume(hand.get_state(), source_ns=clock.now, now_ns=clock.now)
    sensor = hand.control.raw_state.sensor_data[0]
    sensor.raw_force[0].fx = np.nan
    for _ in range(3):
        feedback.consume(hand.get_state(), source_ns=clock.now, now_ns=clock.now)
    sensor.raw_force[0].fx = 1.
    for _ in range(2):
        feedback.consume(hand.get_state(), source_ns=clock.now, now_ns=clock.now)
    for _ in range(4):
        hand.send_action(np.deg2rad(hand.cfg.home_qpos_deg))
    messages = [record.getMessage() for record in caplog.records]
    for fragment in ("read CRC", "trustworthy reads recovered", "tactile dense unavailable",
                     "tactile dense recovered", "send CRC", "send responses recovered"):
        assert sum(fragment in message for message in messages) == 1
    assert len(messages) == 6


def test_cache_reads_and_crc_never_refresh_source_time(robot_fixture, clock):
    env = robot_fixture
    stamp = int(env.robot.read_state().hand["timestamp_ns"][0])
    reads = env.robot.hand.reads
    clock.at(10)
    assert int(env.robot.read_state().hand["timestamp_ns"][0]) == stamp
    assert env.robot.hand.reads == reads
    env.robot.hand.crc = True
    for ms in (40, 80, 120):
        clock.at(ms)
        frame = env.robot.service_feedback()
        assert int(frame.hand["timestamp_ns"][0]) == stamp
        assert env.health[-1] <= stamp + int(env.runtime.hand.feedback_max_age_s * 1e9)
    clock.at(135)
    with pytest.raises(RobotError, match="unavailable/stale"):
        env.robot.service_feedback()
    assert env.state.stop_signal.is_set()


@pytest.mark.parametrize("required,aggregate_ready,dense_ready,wait_ms", [
    (("aggregate", "dense"), 0, 70, 70),
    (("aggregate",), 40, None, 40),
    (("dense",), None, 40, 40),
    ((), None, None, 0),
])
def test_initial_feedback_waits_only_for_required_tactile(
    robot_fixture, clock, monkeypatch, required, aggregate_ready, dense_ready, wait_ms,
):
    env = robot_fixture
    original = env.robot.hand.get_state

    def read():
        sample = original()
        for channel, ready in (("aggregate", aggregate_ready), ("dense", dense_ready)):
            if ready is None or clock.now < clock.origin + ready * 1_000_000:
                setattr(sample, f"tactile_{channel}", None)
        return sample

    monkeypatch.setattr(env.robot.hand, "get_state", read)
    deadline = clock.origin + 200_000_000
    env.robot.reset_hand_feedback(deadline_ns=deadline)
    row = env.robot.read_initial_state(required_tactile=required)
    assert clock.origin + wait_ms * 1_000_000 <= clock.now < deadline
    for channel in required:
        assert int(row.hand[f"tactile_{channel}_timestamp_ns"][0]) > 0
    for channel, ready in (("aggregate", aggregate_ready), ("dense", dense_ready)):
        if ready is None:
            assert int(row.hand[f"tactile_{channel}_timestamp_ns"][0]) == 0


@pytest.mark.parametrize("channel", ["aggregate", "dense"])
def test_initial_tactile_failure_times_out_at_original_deadline(robot_fixture, clock, monkeypatch, channel):
    env = robot_fixture
    original = env.robot.hand.get_state

    def read():
        sample = original()
        setattr(sample, f"tactile_{channel}", None)
        return sample

    monkeypatch.setattr(env.robot.hand, "get_state", read)
    deadline = clock.origin + 180_000_000
    env.robot.reset_hand_feedback(deadline_ns=deadline)
    with pytest.raises(RobotError, match="initial feedback acquisition timed out"):
        env.robot.read_initial_state(required_tactile=(channel,))
    assert clock.now == deadline
    assert int(env.robot.hand_feedback.latest["timestamp_ns"][0]) > clock.origin


def test_reentering_initial_wait_keeps_the_original_deadline(robot_fixture, clock, monkeypatch):
    env = robot_fixture
    original = env.robot.hand.get_state

    def read():
        sample = original()
        sample.tactile_dense = None
        return sample

    monkeypatch.setattr(env.robot.hand, "get_state", read)
    deadline = clock.origin + 100_000_000
    env.robot.reset_hand_feedback(deadline_ns=deadline)
    assert env.robot.read_initial_state().hand is not None
    clock.at(60)
    with pytest.raises(RobotError, match="initial feedback acquisition timed out"):
        env.robot.read_initial_state(required_tactile=("dense",))
    assert clock.now == deadline


def test_initial_feedback_requires_all_held_tactile_channels_to_remain_fresh(robot_fixture, clock, monkeypatch):
    env = robot_fixture
    original = env.robot.hand.get_state

    def read():
        sample = original()
        elapsed = clock.now - clock.origin
        if 0 < elapsed < 240_000_000:
            sample.tactile_aggregate = None
        if elapsed < 200_000_000:
            sample.tactile_dense = None
        return sample

    monkeypatch.setattr(env.robot.hand, "get_state", read)
    env.robot.reset_hand_feedback(deadline_ns=clock.origin + 300_000_000)
    row = env.robot.read_initial_state(required_tactile=("aggregate", "dense"))
    assert clock.now >= clock.origin + 240_000_000
    for channel in ("aggregate", "dense"):
        stamp = int(row.hand[f"tactile_{channel}_timestamp_ns"][0])
        assert 0 <= clock.now - stamp < int(env.runtime.hand.feedback_max_age_s * 1e9)


def test_initial_feedback_does_not_admit_a_read_completing_after_deadline(robot_fixture, clock, monkeypatch):
    env = robot_fixture
    original = env.robot.hand.get_state

    def read():
        clock.sleep(0.06)
        return original()

    monkeypatch.setattr(env.robot.hand, "get_state", read)
    env.robot.reset_hand_feedback(deadline_ns=clock.origin + 50_000_000)
    with pytest.raises(RobotError, match="initial feedback acquisition timed out"):
        env.robot.read_initial_state(required_tactile=("aggregate", "dense"))


@pytest.mark.parametrize("cancellation", ["callback", "run_id", "emergency"])
def test_initial_tactile_wait_preserves_cancellation_checks(robot_fixture, clock, monkeypatch, cancellation):
    env = robot_fixture
    original = env.robot.hand.get_state

    def read():
        sample = original()
        sample.tactile_dense = None
        return sample

    def cancel():
        if clock.now < clock.origin + 40_000_000:
            return False
        if cancellation == "run_id":
            env.state.run_signal.value += 1
        elif cancellation == "emergency":
            env.state.emergency_signal.set()
        return cancellation == "callback"

    monkeypatch.setattr(env.robot.hand, "get_state", read)
    env.robot.reset_hand_feedback()
    with pytest.raises(OperationCancelled):
        env.robot.read_initial_state(required_tactile=("dense",), cancel_requested=cancel)
    assert clock.now == clock.origin + 40_000_000
    assert not env.sent and not env.hand_sent


def test_teleop_preparation_includes_recording_tactile_needs(robot_fixture, clock, monkeypatch):
    env = robot_fixture
    original = env.robot.hand.get_state

    def read():
        sample = original()
        if clock.now < clock.origin + 70_000_000:
            sample.tactile_dense = None
        return sample

    monkeypatch.setattr(env.robot.hand, "get_state", read)
    loop = TeleopLoop(
        env.state, env.runtime, env.robot, task_label="test", control_needs=ObservationNeeds(),
        record_needs=ObservationNeeds(required_tactile=("aggregate", "dense")),
        sensors=SimpleNamespace(channels=None), controller=None, home_planner=None,
        recording_dir=None, keyboard=SimpleNamespace(poll=lambda **kwargs: ()), audio=None,
    )
    loop.start_episode(env.state.run_id)
    assert loop.phase is TeleopPhase.PREPARING
    assert loop.prepared_after_ns >= clock.origin + 70_000_000


def test_observation_deadline_matches_feedback_expiry(robot_fixture, clock):
    env = robot_fixture
    row = env.robot.read_state()
    expires = clock.now + int(env.runtime.arm.feedback_max_age_s * 1e9)
    clock.now = expires - 1
    issues, deadline = check_observation(row, env.runtime, ObservationNeeds())
    assert issues == [] and deadline == expires
    clock.now = expires
    issues, deadline = check_observation(row, env.runtime, ObservationNeeds())
    assert "arm_feedback_stale" in issues and deadline is None


def controller_fixture(env, clock):
    state = env.state
    state.stop_signal.clear()
    io = SimpleNamespace(run_id=state.run_signal, stop=state.stop_signal,
                         emergency=state.emergency_signal, closing=SimpleNamespace(is_set=lambda: False),
                         fault=SimpleNamespace(value=b""), health_until=SimpleNamespace(value=clock.now + 1_000_000_000),
                         execution_limit=[MAX_TIME, 0], input_generation=SimpleNamespace(value=3))
    driver = SimpleNamespace(mode=1, servo=lambda q: pytest.fail("unexpected servo side effect"))
    controller = ArmController(env.runtime.arm, io, None, driver)
    controller.status["valid"] = True
    controller.status["prepared_run_id"] = state.run_id
    controller.status["frame"] = env.robot.arm.snapshot()[0]
    goal = np.zeros((), dtype=GOAL_DTYPE)
    goal["run_id"] = state.run_id
    goal["submit_until_ns"] = clock.now + 1_000_000_000
    goal["execute_until_ns"] = clock.now + 60_000_000
    goal["input_generation"] = 3
    goal["sequence"] = 1
    return controller, goal


def test_health_cannot_extend_command_or_input_authority(robot_fixture, clock):
    env = robot_fixture
    controller, goal = controller_fixture(env, clock)
    assert controller.command_issue(goal, clock.now, first=True) is None
    clock.at(200)
    env.robot.service_feedback()
    controller.io.health_until.value = env.health[-1]
    controller.status["frame"] = env.robot.arm.snapshot()[0]
    assert controller.command_issue(goal, clock.now) == "input_decision_expired"
    goal["execute_until_ns"] = clock.now + 1000
    controller.io.input_generation.value += 1
    assert controller.command_issue(goal, clock.now, first=True) == "input_changed"
    env.state.run_signal.value += 1
    assert controller.command_issue(goal, clock.now) == "revoked"


def test_large_valid_target_and_braking_and_tracking_guards(robot_fixture, clock):
    env = robot_fixture
    cfg = env.runtime.arm
    q = np.asarray(cfg.home_qpos).copy()
    target = q.copy()
    target[0] += 0.6
    limits = np.column_stack((cfg.joint_limit_lower, cfg.joint_limit_upper))
    assert joint_target_rejection(target, limits) is None
    invalid = target.copy()
    invalid[0] = limits[0, 1] + 0.01
    assert joint_target_rejection(invalid, limits) == "operational_limits"
    trajectory = ArmTrajectory(cfg, q, np.zeros(7), clock.now)
    trajectory.step(clock.now, target)
    np.testing.assert_allclose(trajectory.step(clock.now + 2_000_000_000), target)
    q[0] = limits[0, 0] + 0.001
    velocity = np.zeros(7)
    velocity[0] = -1.
    with pytest.raises(ValueError, match="braking/position extremum"):
        ArmTrajectory(cfg, q, velocity, clock.now)
    controller, goal = controller_fixture(env, clock)
    controller.io.take_goal = lambda: None
    controller.active = goal
    reference = np.asarray(cfg.home_qpos).copy()
    reference[0] += cfg.tracking_error_rad + 0.01
    controller.reference = SimpleNamespace(step=lambda *args: reference)
    with pytest.raises(RuntimeError, match="tracking_error"):
        controller.tick()


def test_close_interrupt_does_not_skip_other_resources(robot_fixture, monkeypatch):
    env = robot_fixture
    arm, hand = env.robot.arm, env.robot.hand
    original = KeyboardInterrupt("stop interrupted")

    def interrupted(*args, **kwargs):
        raise original

    monkeypatch.setattr(hand, "stop", interrupted)
    with pytest.raises(KeyboardInterrupt) as failure:
        env.robot.close()
    assert failure.value is original
    assert not arm.is_connected and not hand.is_connected
    assert env.robot.stop_required


@pytest.mark.parametrize("expired", [False, True])
def test_combined_send_rechecks_after_arm_submission(robot_fixture, clock, monkeypatch, expired):
    env = robot_fixture
    env.robot.prepare_streaming(env.state.run_id, service=env.robot.service_feedback)
    original_submit = env.robot.arm.submit

    def blocked_submit(command, **kwargs):
        original_submit(command, **kwargs)
        if expired:
            clock.sleep(0.1)
        else:
            env.state.run_signal.value += 1

    monkeypatch.setattr(env.robot.arm, "submit", blocked_submit)
    command = RobotCommand(env.state.run_id, np.asarray(env.runtime.arm.home_qpos),
                           np.deg2rad(env.runtime.hand.home_qpos_deg))
    with pytest.raises(TimeoutError if expired else OperationCancelled):
        env.robot.send_action(command, valid_until_ns=clock.now + 50_000_000,
                              execute_until_ns=clock.now + 500_000_000)
    assert len(env.sent) == 1 and not env.hand_sent
    assert env.state.stop_signal.is_set()


def test_arm_operation_preserves_original_error_and_traceback(robot_fixture, clock, caplog):
    import threading

    env = robot_fixture
    controller, _ = controller_fixture(env, clock)
    replies = []
    controller.pipe = SimpleNamespace(send=replies.append)
    controller.io.stopped = threading.Event()
    controller.io.close_error = SimpleNamespace(value=b"")
    controller.io.take_goal = lambda **kwargs: None
    controller.io.publish_status = lambda status: None
    controller.driver.stop = lambda: None
    original = ValueError("planner malfunction")

    def fail(check_abort):
        raise original

    controller.wait_stationary = fail
    controller.operation(PrepareRequest(1, env.state.run_id))
    assert replies[0].error is original
    assert controller.io.stopped.is_set()
    assert caplog.records[-1].exc_info[1] is original


@pytest.mark.parametrize("operation_failed", [False, True])
def test_blocking_arm_operation_keeps_feedback_service_and_error_cause(robot_fixture, clock, operation_failed):
    env = robot_fixture
    process = ArmProcess.__new__(ArmProcess)
    process.cfg = env.runtime.arm
    process.io = SimpleNamespace(connected=SimpleNamespace(is_set=lambda: True),
                                 close_error=SimpleNamespace(value=b""))
    process.process = SimpleNamespace(is_alive=lambda: True)
    process.operation_unresolved = False
    original = ValueError("bad plan") if operation_failed else None

    def poll(timeout):
        clock.sleep(timeout)
        return clock.now >= clock.origin + 150_000_000

    process.pipe = SimpleNamespace(poll=poll, send=lambda request: None,
                                   recv=lambda: LifecycleReply(1, original))
    process.request_stop = env.robot.arm.request_stop
    args = dict(service=env.robot.service_feedback, abort=lambda: False,
                deadline_ns=clock.now + 1_000_000_000)
    if operation_failed:
        with pytest.raises(RobotError) as failure:
            process.request_operation(PrepareRequest(1, env.state.run_id), **args)
        assert failure.value.__cause__ is original
    else:
        process.request_operation(PrepareRequest(1, env.state.run_id), **args)
    assert env.robot.hand.reads >= 5
    assert int(env.robot.read_state().hand["timestamp_ns"][0]) >= clock.origin + 130_000_000


@pytest.mark.parametrize("case", ["pipe_failure", "owner_failed", "owner_pending"])
def test_arm_close_preserves_pending_owner_and_closes_other_pipe(case):
    process = ArmProcess.__new__(ArmProcess)
    process.request_stop = lambda: None
    process.io = SimpleNamespace(closing=SimpleNamespace(set=lambda: None),
                                 close_error=SimpleNamespace(value=b""))
    process.process = SimpleNamespace(pid=1, join=lambda timeout: None,
                                      is_alive=lambda: case == "owner_pending",
                                      exitcode=1 if case == "owner_failed" else 0)
    closed = []
    original = OSError("pipe close failed")

    def close_parent():
        closed.append("parent")
        if case == "pipe_failure":
            raise original

    process.pipe = SimpleNamespace(close=close_parent)
    process.child_pipe = SimpleNamespace(close=lambda: closed.append("child"))
    with pytest.raises(OSError if case == "pipe_failure" else RuntimeError if case == "owner_failed" else TimeoutError) as failure:
        process.close()
    assert closed == ([] if case == "owner_pending" else ["parent", "child"])
    if case == "pipe_failure":
        assert failure.value is original


def test_tag_stage_two_roundoff_uses_stage_one_only():
    import nlopt

    from dexmani_real.teleop.config import TAGConfig
    from dexmani_real.teleop.retargeting.tag import TAGRetargeter

    retargeter = TAGRetargeter(TAGConfig())
    first = (retargeter.lower + retargeter.upper) / 2
    retargeter.position_optimizer = SimpleNamespace(optimize=lambda seed: first.copy())

    def fail(error):
        def optimize(seed):
            raise error
        return SimpleNamespace(optimize=optimize)

    retargeter.pinch_optimizer = fail(nlopt.RoundoffLimited())
    np.testing.assert_array_equal(retargeter.solve(np.zeros((5, 3))), first)
    np.testing.assert_array_equal(retargeter.previous_solution, first)
    np.testing.assert_array_equal(retargeter.position_solution, first)
    original = ValueError("objective bug")
    retargeter.pinch_optimizer = fail(original)
    with pytest.raises(ValueError) as failure:
        retargeter.solve(np.zeros((5, 3)))
    assert failure.value is original
    retargeter.position_optimizer = fail(nlopt.RoundoffLimited())
    with pytest.raises(nlopt.RoundoffLimited):
        retargeter.solve(np.zeros((5, 3)))

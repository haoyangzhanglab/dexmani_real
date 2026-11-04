"""XHand 的串行 SDK 边界；Fake 只验证调用行为，不证明真实缓存语义。"""

import json
import sys
from dataclasses import asdict, replace
from itertools import chain, repeat
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.xhand import HandConfig
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.robot.xhand.driver import (
    COMMUNICATION_CRC_ERROR_CODE as CRC,
    COMBINED_FORCE_UNAVAILABLE_CODE,
    DISTRIBUTED_FORCE_UNAVAILABLE_CODE,
    OVERCURRENT_WARNING_CODE,
    TACTILE_SENSOR_IDS,
    XHand,
    XHandError,
)
from dexmani_real.robot.xhand.feedback import XHandFeedback


def reply(code=0):
    return SimpleNamespace(error_code=code, error_message=f'sdk {code}')


def target(value=.2):
    qpos = np.full(12, value)
    qpos[3] = .1
    return qpos


def payload(*, qpos=0.2, board=0):
    force = lambda: SimpleNamespace(fx=1., fy=2., fz=3.)
    return SimpleNamespace(
        finger_state=[SimpleNamespace(id=i, position=qpos if i != 3 else .1, commboard_err=0,
                                      jonitboard_err=board, tipboard_err=0) for i in range(12)],
        sensor_data=[SimpleNamespace(calc_force=force(), raw_force=[force() for _ in range(120)])
                     for _ in range(5)],
    )


class FakeControl:
    def __init__(self, *, writes=(), reads=(), resets=()):
        self.writes, self.reads, self.resets = iter(writes), iter(reads), iter(resets)
        self.sent, self.read_args, self.reset_ids = [], [], []
        self.closed = False

    def send_command(self, device_id, command):
        self.sent.append([(joint.mode, joint.position) for joint in command.finger_command])
        result = next(self.writes)
        if isinstance(result, BaseException):
            raise result
        return reply(result)

    def read_state(self, device_id, force_update):
        self.read_args.append((device_id, force_update))
        result = next(self.reads)
        if isinstance(result, BaseException):
            raise result
        code, state = result
        return reply(code), state

    def reset_sensor(self, device_id, sensor_id):
        self.reset_ids.append(sensor_id)
        result = next(self.resets)
        if isinstance(result, BaseException):
            raise result
        return reply(result)

    def close_device(self):
        self.closed = True


def make_hand(**kwargs):
    hand = XHand(HandConfig())
    hand.sdk = SimpleNamespace(HandCommand_t=lambda: SimpleNamespace(
        finger_command=[SimpleNamespace() for _ in range(12)]))
    hand.control = FakeControl(**kwargs)
    hand.connected_flag = True
    hand.command = hand.make_command(np.full(12, .2))
    return hand


@pytest.fixture
def clock(monkeypatch):
    now = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.robot.xhand.driver.time.monotonic_ns', lambda: now[0])
    monkeypatch.setattr('dexmani_real.robot.xhand.driver.time.monotonic', lambda: now[0] / 1e9)
    monkeypatch.setattr('dexmani_real.robot.xhand.driver.time.sleep',
                        lambda delay: now.__setitem__(0, now[0] + round(delay * 1e9)))
    return now


@pytest.mark.parametrize('integer', [int, np.int32, np.int64, np.uint64])
def test_hand_config_accepts_python_and_numpy_integer_scalars(integer, tmp_path):
    import yaml

    from dexmani_real.utils.io import write_yaml

    config = replace(HandConfig(), baudrate=integer(3_000_000), device_id=integer(0),
                     ki=integer(0), kd=integer(1), kp=(integer(100),) * 12,
                     tor_max=(integer(400),) * 12)
    config.validate()
    ethercat = replace(config, comm_type='ethercat', device_name=None,
                       ethercat_slave_position=integer(0))
    ethercat.validate()
    # 接纳配置之后，CLI JSON、Policy YAML 和原生 SDK 都消费同一原生整数。
    values = asdict(ethercat)
    serialized = json.loads(json.dumps(values))
    assert serialized['kp'] == [100] * 12
    write_yaml({'runtime': {'hand': values}}, tmp_path / 'run_config.yaml')
    assert yaml.safe_load((tmp_path / 'run_config.yaml').read_text())['runtime']['hand'] == serialized
    assert all(type(getattr(ethercat, name)) is int for name in (
        'baudrate', 'device_id', 'ethercat_slave_position', 'ki', 'kd',
    ))
    hand = XHand(config)
    hand.sdk = SimpleNamespace(HandCommand_t=lambda: SimpleNamespace(
        finger_command=[SimpleNamespace() for _ in range(12)]))
    command = hand.make_command(target())
    assert all(type(joint.kp) is int and type(joint.tor_max) is int
               for joint in command.finger_command)


@pytest.mark.parametrize('entrypoint', ['scripts.collect_teleop', 'scripts.run_policy',
                                      'scripts.tools.control_xhand'])
def test_numpy_integer_config_prints_through_real_cli(entrypoint, tmp_path, capsys):
    import importlib

    path = tmp_path / 'config.py'
    path.write_text(
        'from dataclasses import replace\n'
        'import numpy as np\n'
        'from dexmani_real.config.real import RealConfig\n'
        'base = RealConfig()\n'
        'CONFIG = replace(base, hand=replace(base.hand, device_id=np.int64(0), '
        'kp=(np.int64(100),) * 12, tor_max=(np.uint64(300),) * 12))\n'
    )
    assert importlib.import_module(entrypoint).main(['--config', str(path), '--print-config']) == 0
    hand = json.loads(capsys.readouterr().out)['hand']
    assert hand['device_id'] == 0 and hand['kp'] == [100] * 12 and hand['tor_max'] == [300] * 12


@pytest.mark.parametrize('field', ['baudrate', 'device_id', 'ki', 'kd', 'kp', 'tor_max',
                                  'ethercat_slave_position'])
@pytest.mark.parametrize('value', [True, np.bool_(True), 1.5, np.float64(1.5), 2.0])
def test_hand_config_rejects_booleans_and_floats(field, value):
    update = {field: (value,) * 12 if field in ('kp', 'tor_max') else value}
    if field == 'ethercat_slave_position':
        update.update(comm_type='ethercat', device_name=None)
    with pytest.raises(ValueError, match='integer'):
        replace(HandConfig(), **update).validate()


@pytest.mark.parametrize('field,value', [('baudrate', 0), ('device_id', -1), ('ki', -1),
                                       ('kd', -1), ('kp', 0), ('tor_max', -1),
                                       ('tor_max', 401), ('ethercat_slave_position', -2)])
def test_hand_config_keeps_integer_value_ranges(field, value):
    update = {field: (np.int64(value),) * 12 if field in ('kp', 'tor_max') else np.int64(value)}
    if field == 'ethercat_slave_position':
        update.update(comm_type='ethercat', device_name=None)
    with pytest.raises(ValueError):
        replace(HandConfig(), **update).validate()


@pytest.mark.parametrize('field', ['kp', 'tor_max'])
def test_hand_config_keeps_twelve_joint_gain_lengths(field):
    with pytest.raises(ValueError, match='twelve'):
        replace(HandConfig(), **{field: (np.int64(100),) * 11}).validate()


def test_periodic_crc_moves_on_to_latest_target_and_resets_only_on_write(clock, caplog):
    hand = make_hand(writes=[CRC, CRC, 0, CRC, 0], reads=[(0, payload())])
    for i, confirmed in enumerate((False, False, True, False)):
        assert hand.send_action(target(.2 + i * .01)) is confirmed
        clock[0] += 10_000_000
    assert hand.consecutive_write_crc == 1
    assert hand.get_state() is not None
    assert hand.consecutive_write_crc == 1
    hand.stop(np.full(12, .2), timestamp_ns=clock[0])
    assert hand.consecutive_write_crc == 0
    assert [row[0][1] for row in hand.control.sent[:4]] == pytest.approx([.2, .21, .22, .23])
    assert hand.counts['write_crc'] == 3
    assert sum('write CRC' in record.message for record in caplog.records) == 1


def test_fifth_periodic_crc_faults_but_stop_still_retries(clock):
    hand = make_hand(writes=[CRC] * 6 + [0])
    for _ in range(4):
        assert hand.send_action(target()) is False
    with pytest.raises(XHandError, match='5 consecutive'):
        hand.send_action(target())
    hand.stop(np.full(12, .2), timestamp_ns=clock[0])
    assert len(hand.control.sent) == 7
    assert hand.consecutive_write_crc == 0


def test_stop_keeps_old_freshness_and_finite_crc_budget(clock):
    hand = make_hand(writes=repeat(CRC))
    with pytest.raises(XHandError, match='stop timed out'):
        hand.stop(np.full(12, .2), timestamp_ns=clock[0] - 200_000_000)
    assert 1 <= len(hand.control.sent) <= 26
    assert all(mode == 0 for row in hand.control.sent for mode, _ in row)
    assert clock[0] == 1_250_000_000


@pytest.mark.parametrize('feedback', ['healthy', 'crc', 'device_fault'])
@pytest.mark.parametrize('idle_delay_s', [0, .2])
def test_arm_home_services_hand_feedback_until_idle_stop(clock, feedback, idle_delay_s):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.robot.commands import RobotCommand, RobotError
    from dexmani_real.robot.system import Robot
    from dexmani_real.runtime.home import home_arm
    from dexmani_real.runtime.state import SessionState

    config, state = RealConfig(), SessionState()
    # 独立 arm HOME 不消费手部 HOME 配置，也不等待触觉初值。
    hand_config = replace(config.hand, home_qpos_deg=(float('nan'),) * 12)
    robot = Robot(config.arm, hand_config, runtime_state=state)
    hand = make_hand(writes=repeat(0), reads=[(COMBINED_FORCE_UNAVAILABLE_CODE, payload())])
    robot.hand, robot.connected = hand, True
    robot.read_hand()
    initial_stamp = int(robot.hand_feedback.snapshot()['timestamp_ns'][0])
    robot.send_action(RobotCommand(hand_qpos=target(.3)), expires_ns=clock[0] + 1_000_000_000)
    assert robot.hand_stop_required and len(hand.control.sent) == 1
    # 失败前仍提供新反馈，确保设备错误发生时的停止样本尚新鲜。
    read_code = {'healthy': COMBINED_FORCE_UNAVAILABLE_CODE, 'crc': CRC, 'device_fault': 999}[feedback]
    hand.control.reads = chain([(COMBINED_FORCE_UNAVAILABLE_CODE, payload(qpos=.4))],
                               repeat((read_code, payload(qpos=.4))))
    started_ns = clock[0]
    confirmations = []

    def arm_home(epoch, *, service, abort, deadline_ns):
        assert epoch == state.cancel_epoch and deadline_ns > started_ns + 2_000_000_000
        # 模拟父端等待 Owner 的轮询；期间只允许手部反馈读取。
        while clock[0] < started_ns + 2_000_000_000:
            service()
            assert not abort()
            assert robot.hand_stop_required and len(hand.control.sent) == 1
            clock[0] += 5_000_000
        state.stop_signal.set()

    robot.arm = SimpleNamespace(home=arm_home, check_error=lambda: None,
                                wait_stopped=lambda: confirmations.append(True))
    run = lambda: home_arm(state, robot, cancel_epoch=state.cancel_epoch, cancel_requested=lambda: False)
    if feedback == 'healthy':
        run()
        assert clock[0] == started_ns + 2_000_000_000
        assert len(hand.control.read_args) > 50
    elif feedback == 'crc':
        with pytest.raises(RobotError, match='feedback unavailable/stale'):
            run()
    else:
        with pytest.raises(XHandError, match='999'):
            run()
    assert state.stop_signal.is_set() and robot.hand_stop_required
    assert len(hand.control.sent) == 1
    final_frame = robot.hand_feedback.snapshot()
    final_stamp = int(final_frame['timestamp_ns'][0])
    if feedback == 'healthy':
        assert final_stamp > initial_stamp
        assert clock[0] - final_stamp < 40_000_000
    else:
        # 读 CRC 和设备错误均不能给最后一份可信样本续期。
        assert final_stamp == initial_stamp == started_ns
    assert np.isnan(final_frame['tactile_aggregate']).all()
    assert np.isnan(final_frame['tactile_dense']).all()
    clock[0] += round(idle_delay_s * 1e9)
    # 收尾后的反馈正常，不能用这份晚到反馈反过来为停止续期。
    hand.control.reads = repeat((COMBINED_FORCE_UNAVAILABLE_CODE, payload(qpos=.4)))
    robot.service_idle()
    assert confirmations == [True]
    assert not robot.hand_stop_required and len(hand.control.sent) == 2
    stop_command = hand.control.sent[-1]
    expected_mode = 0 if feedback == 'crc' or idle_delay_s else 3
    assert all(mode == expected_mode for mode, _ in stop_command)
    if expected_mode == 3:
        np.testing.assert_array_equal([qpos for _, qpos in stop_command], target(.4))


def test_crc_does_not_refresh_any_source_and_channels_hold_independently(clock):
    aggregate_bad, dense_bad = payload(qpos=.3), payload(qpos=.4)
    aggregate_bad.sensor_data[0].calc_force.fx = np.nan
    dense_bad.sensor_data[0].raw_force = []
    hand = make_hand(reads=[(0, payload()), (CRC, payload(qpos=.9)),
                            (OVERCURRENT_WARNING_CODE, aggregate_bad), (0, dense_bad), (0, payload(qpos=.5))])
    cache = XHandFeedback()
    first = cache.consume(hand.get_state())
    clock[0] += 100_000_000
    assert cache.consume(hand.get_state()) is first
    clock[0] += 100_000_000
    frame = cache.consume(hand.get_state())
    assert frame['timestamp_ns'][0] == clock[0]
    assert frame['tactile_aggregate_timestamp_ns'][0] == first['timestamp_ns'][0]
    assert frame['tactile_dense_timestamp_ns'][0] == clock[0]
    dense_stamp = clock[0]
    clock[0] += 100_000_000
    frame = cache.consume(hand.get_state())
    assert frame['tactile_aggregate_timestamp_ns'][0] == clock[0]
    assert frame['tactile_dense_timestamp_ns'][0] == dense_stamp
    clock[0] += 100_000_000
    recovered = cache.consume(hand.get_state())
    assert recovered['tactile_dense_timestamp_ns'][0] == clock[0]
    assert np.isfinite(recovered['tactile_aggregate']).all()
    assert hand.counts == dict(read_crc=1, write_crc=0, aggregate=1, dense=1, overcurrent=1)
    assert hand.control.read_args == [(0, True)] * 5


def test_internal_tactile_presence_hold_and_policy_freshness_follow_sdk_sources(clock):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.runtime.observation import (
        Observation, ObservationNeeds, check_observation, missing_tactile_channels,
    )

    hand = make_hand(reads=[(DISTRIBUTED_FORCE_UNAVAILABLE_CODE, payload()), (0, payload()),
                            (COMBINED_FORCE_UNAVAILABLE_CODE, payload()), (0, payload())])
    cache = XHandFeedback()
    needs = ObservationNeeds(required_tactile=('aggregate', 'dense'))
    def observation(frame):
        arm = np.array([(clock[0],)], dtype=[('timestamp_ns', '<u8')])
        return Observation(arm=arm, hand=frame, camera=None, vr=None, point_cloud=None,
                           observation_timestamp_ns=clock[0])

    first = observation(cache.consume(hand.get_state()))
    assert missing_tactile_channels(first, needs.required_tactile) == ['dense']
    clock[0] += 10_000_000
    ready = observation(cache.consume(hand.get_state()))
    assert not missing_tactile_channels(ready, needs.required_tactile)
    assert missing_tactile_channels(replace(ready, observation_timestamp_ns=clock[0] - 1),
                                    needs.required_tactile) == ['aggregate', 'dense']
    clock[0] += 600_000_000
    held = observation(cache.consume(hand.get_state()))
    assert not missing_tactile_channels(held, needs.required_tactile)
    np.testing.assert_array_equal(held.hand['tactile_dense_timestamp_ns'],
                                  ready.hand['tactile_dense_timestamp_ns'])
    assert not check_observation(held, RealConfig(), needs, allow_tactile_hold=True)
    assert check_observation(held, RealConfig(), needs) == [
        'hand_tactile_aggregate_timestamp_ns_unavailable_or_stale',
        'hand_tactile_dense_timestamp_ns_unavailable_or_stale',
    ]
    cache.clear_tactile()
    assert missing_tactile_channels(observation(cache.snapshot()), needs.required_tactile) == [
        'aggregate', 'dense',
    ]
    clock[0] += 10_000_000
    recovered = observation(cache.consume(hand.get_state()))
    assert not check_observation(recovered, RealConfig(), needs)


@pytest.mark.parametrize('code,board', [(OVERCURRENT_WARNING_CODE, 0), (0, 110)])
def test_overcurrent_keeps_valid_joints_and_tactile(code, board, clock):
    hand = make_hand(reads=[(code, payload(board=board))], writes=[OVERCURRENT_WARNING_CODE])
    state = hand.get_state()
    assert state.tactile_aggregate is not None and state.tactile_dense is not None
    assert hand.send_action(target()) is True


def test_reset_one_pass_keeps_per_finger_results(clock):
    codes = [CRC, COMBINED_FORCE_UNAVAILABLE_CODE, DISTRIBUTED_FORCE_UNAVAILABLE_CODE,
             OVERCURRENT_WARNING_CODE, 0]
    hand = make_hand(resets=codes)
    assert hand.reset_tactile_sensors() == dict(zip(TACTILE_SENSOR_IDS, codes))
    assert hand.control.reset_ids == list(TACTILE_SENSOR_IDS)
    assert hand.tactile_reset_metadata['hand_tactile_reset_result'] == dict(zip(TACTILE_SENSOR_IDS, codes))


def test_reset_cancellation_preserves_completed_results_and_can_retry(clock):
    hand = make_hand(resets=[0] * 7)
    with pytest.raises(OperationCancelled, match='2/5'):
        hand.reset_tactile_sensors(cancel_requested=lambda: len(hand.control.reset_ids) == 2)
    assert hand.tactile_reset_metadata['hand_tactile_reset_result'] == {0x11: 0, 0x12: 0}
    assert len(hand.reset_tactile_sensors()) == 5
    assert hand.control.reset_ids == [0x11, 0x12, *TACTILE_SENSOR_IDS]


@pytest.mark.parametrize('kind', ['read', 'send', 'reset'])
def test_real_faults_are_not_treated_as_missing_tactile(kind, clock):
    hand = make_hand(reads=[(999, payload())], writes=[999], resets=[999])
    with pytest.raises(XHandError, match='999'):
        if kind == 'read':
            hand.get_state()
        elif kind == 'send':
            hand.send_action(target())
        else:
            hand.reset_tactile_sensors()


def test_invalid_joint_payload_remains_fault(clock):
    bad = payload()
    bad.finger_state[0].position = np.nan
    hand = make_hand(reads=[(0, bad)])
    with pytest.raises(XHandError, match='non-finite joint'):
        hand.get_state()


@pytest.mark.parametrize('kind', ['read', 'send', 'reset'])
def test_sdk_exceptions_keep_original_failure(kind, clock):
    failure = OSError('native failure')
    hand = make_hand(reads=[failure], writes=[failure], resets=[failure])
    with pytest.raises(OSError) as caught:
        if kind == 'read':
            hand.get_state()
        elif kind == 'send':
            hand.send_action(target())
        else:
            hand.reset_tactile_sensors()
    assert caught.value is failure


@pytest.mark.parametrize('kind', ['read', 'send'])
def test_robot_stops_and_preserves_native_sdk_exception(kind, clock):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.robot.commands import RobotCommand
    from dexmani_real.robot.system import Robot
    from dexmani_real.runtime.state import SessionState

    config, state = RealConfig(), SessionState()
    robot = Robot(config.arm, config.hand, runtime_state=state)
    failure = OSError('native failure')
    robot.hand = make_hand(reads=[(0, payload()), failure], writes=[failure])
    robot.connected = True
    robot.read_hand()
    with pytest.raises(OSError) as caught:
        if kind == 'read':
            robot.service_feedback()
        else:
            robot.send_action(RobotCommand(hand_qpos=target()), expires_ns=clock[0] + 1_000_000_000)
    assert caught.value is failure and state.stop_signal.is_set()


def test_disconnect_does_not_swallow_interrupt_after_another_error(clock):
    hand = make_hand()
    hand.cfg = replace(hand.cfg, comm_type='ethercat')
    def failed_init():
        raise RuntimeError('INIT failed')
    def interrupted_close():
        raise KeyboardInterrupt()
    hand.request_ethercat_init = failed_init
    hand.control.close_device = interrupted_close
    with pytest.raises(KeyboardInterrupt):
        hand.disconnect()
    assert not hand.is_connected


def test_connect_uses_real_joint_initial_position_without_tactile(monkeypatch, clock):
    hand = make_hand(reads=[(COMBINED_FORCE_UNAVAILABLE_CODE, payload(qpos=.42))])
    control, sdk = hand.control, hand.sdk
    hand.connected_flag = False
    control.list_hands_id = lambda: [0]
    hand.open_device = lambda: None
    monkeypatch.setitem(sys.modules, 'xhand_controller', SimpleNamespace(xhand_control=sdk))
    sample = hand.connect()
    assert sample.tactile_dense is None and sample.tactile_aggregate is None
    assert not control.sent
    np.testing.assert_array_equal([joint.position for joint in hand.command.finger_command], target(.42))


def test_diagnostic_one_shot_crc_retries_once_and_checks_measured_arrival(clock):
    from scripts.tools.control_xhand import XHandControlExample
    hand = XHandControlExample(HandConfig())
    source = make_hand(writes=[CRC, CRC], reads=repeat((0, payload())))
    hand.sdk, hand.control, hand.command = source.sdk, source.control, source.command
    hand.connected_flag = True
    hand.consecutive_write_crc = 5  # 一次性动作独立于周期升级阈值。
    hand.execute_action(target())
    assert len(hand.control.sent) == 2
    assert hand.motion_pending


@pytest.mark.parametrize('cancel_check', [1, 2, 3])
def test_robot_reset_cancellation_checks_stop_and_preserves_joint_feedback(clock, cancel_check):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.robot.system import Robot
    from dexmani_real.runtime.state import SessionState

    config = RealConfig()
    robot = Robot(config.arm, config.hand, runtime_state=SessionState())
    robot.hand = make_hand(reads=[(0, payload())])
    first = robot.read_hand()
    stopped = []
    robot.stop = lambda: stopped.append(True)
    checks = []
    def cancelled():
        checks.append(True)
        return len(checks) == cancel_check
    with pytest.raises(OperationCancelled):
        robot.reset_tactile_sensors(cancel_epoch=robot.runtime_state.cancel_epoch,
                                    cancel_requested=cancelled)
    assert bool(stopped) == (cancel_check > 1)
    assert not robot.hand.control.reset_ids
    current = robot.hand_feedback.snapshot()
    np.testing.assert_array_equal(current['qpos'], first['qpos'])
    assert current['timestamp_ns'][0] == first['timestamp_ns'][0]
    if cancel_check < 3:
        assert current is first
    else:
        # 清缓存之后取消无需回调恢复；下次开始重新等待真实触觉。
        assert current['tactile_aggregate_timestamp_ns'][0] == 0
        assert current['tactile_dense_timestamp_ns'][0] == 0


def test_connect_failure_closes_partial_resource(monkeypatch, clock):
    hand = make_hand(reads=[(999, payload())])
    control, sdk = hand.control, hand.sdk
    hand.connected_flag = False
    control.list_hands_id = lambda: [0]
    hand.open_device = lambda: None
    monkeypatch.setitem(sys.modules, 'xhand_controller', SimpleNamespace(xhand_control=sdk))
    with pytest.raises(XHandError, match='999'):
        hand.connect()
    assert control.closed and hand.control is None
    assert not hand.is_connected


@pytest.mark.parametrize('field', ['home_timeout_s', 'home_tolerance_deg'])
def test_diagnostic_rejects_invalid_operation_budget_before_hardware(monkeypatch, field):
    from scripts.tools import control_xhand
    config = replace(HandConfig(), **{field: 0})
    monkeypatch.setattr(control_xhand, 'load_config', lambda path: SimpleNamespace(hand=config))
    monkeypatch.setattr(control_xhand, 'run_isolated_hardware_session',
                        lambda argv: pytest.fail('hardware launcher reached'))
    with pytest.raises(SystemExit) as exc:
        control_xhand.main([])
    assert exc.value.code == 2


def test_robot_crc_continues_latest_submission_then_faults_and_stops(clock):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.robot.commands import RobotCommand
    from dexmani_real.robot.system import Robot
    from dexmani_real.runtime.state import SessionState

    config, state = RealConfig(), SessionState()
    robot = Robot(config.arm, config.hand, runtime_state=state)
    hand = make_hand(writes=[CRC, 0, *([CRC] * 5), CRC, 0], reads=repeat((0, payload())))
    events, submitted = [], []
    sdk_send = hand.control.send_command
    def send(device_id, command):
        events.append('hand')
        return sdk_send(device_id, command)
    hand.control.send_command = send
    def submit(command, *, expires_ns):
        events.append('arm')
        submitted.append(command)
    robot.arm = SimpleNamespace(is_connected=True, check_error=lambda: None, submit=submit,
                                wait_stopped=lambda: events.append('arm_stopped'))
    robot.hand, robot.connected = hand, True
    robot.read_hand()

    def command(value):
        return RobotCommand(arm_qpos=np.asarray(config.arm.home_qpos), hand_qpos=target(value))

    for value in (.2, .3):
        robot.send_action(command(value), expires_ns=clock[0] + 1_000_000_000)
        assert not state.stop_signal.is_set()
        clock[0] += 20_000_000
    assert hand.control.sent[0][0][1] == .2
    assert hand.control.sent[1][0][1] == .3
    np.testing.assert_array_equal(submitted[0].hand_qpos, target(.2))
    np.testing.assert_array_equal(submitted[1].hand_qpos, target(.3))

    for _ in range(4):
        robot.send_action(command(.4), expires_ns=clock[0] + 1_000_000_000)
        assert not state.stop_signal.is_set()
    with pytest.raises(XHandError, match='5 consecutive'):
        robot.send_action(command(.5), expires_ns=clock[0] + 1_000_000_000)
    assert state.stop_signal.is_set()
    assert events == ['arm', 'hand'] * 7
    robot.stop()
    assert state.stop_signal.is_set()
    assert events[-3:] == ['hand', 'hand', 'arm_stopped']
    assert len(hand.control.sent) == 9
    assert not robot.hand_stop_required
    assert hand.consecutive_write_crc == 0

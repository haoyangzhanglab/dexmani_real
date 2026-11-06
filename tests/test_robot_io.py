from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.robot.commands import (
    FEEDBACK_POSITION_ROUNDOFF_RAD,
    OperationCancelled,
)
from dexmani_real.robot.model import XHAND_RATED_QPOS_MAX_RAD, XHAND_RATED_QPOS_MIN_RAD
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.driver import (
    COMBINED_FORCE_UNAVAILABLE_CODE,
    MAX_CONSECUTIVE_WRITE_CRC,
    STOP_FEEDBACK_MAX_AGE_S,
    STOP_RETRY_TIMEOUT_S,
    XHand,
    XHandError,
)
from dexmani_real.robot.xhand.driver import (
    COMMUNICATION_CRC_ERROR_CODE as CRC,
)
from dexmani_real.runtime.state import SessionState


@pytest.fixture
def clock(monkeypatch):
    clock = SimpleNamespace(ns=1_000_000_000)

    def sleep(seconds):
        clock.ns += round(seconds * 1e9)

    clock.sleep = sleep
    monkeypatch.setattr("time.monotonic_ns", lambda: clock.ns)
    monkeypatch.setattr("time.monotonic", lambda: clock.ns / 1e9)
    monkeypatch.setattr("time.sleep", sleep)
    return clock


@pytest.fixture
def hand(clock):
    hand = XHand(RealConfig().hand)
    hand.connected_flag = True
    hand.command = SimpleNamespace(finger_command=[SimpleNamespace(mode=3, position=0.) for _ in range(12)])
    hand.control = Mock()
    hand.sent = []

    def send(device_id, command):
        hand.sent.append([(j.mode, j.position) for j in command.finger_command])
        return SimpleNamespace(error_code=0, error_message="ok")

    hand.control.send_command.side_effect = send
    raw = SimpleNamespace(finger_state=[
        SimpleNamespace(id=i, position=q, commboard_err=0, jonitboard_err=0, tipboard_err=0)
        for i, q in enumerate(XHAND_RATED_QPOS_MIN_RAD)
    ])
    hand.control.read_state.return_value = (
        SimpleNamespace(error_code=COMBINED_FORCE_UNAVAILABLE_CODE, error_message="no tactile"), raw,
    )
    return hand


@pytest.mark.parametrize("case", ["valid", "roundoff", "below", "above", "nan", "shape", "text", "stale", "missing"])
def test_hand_stop_uses_only_trusted_measured_positions(hand, clock, case, caplog):
    lower, upper = np.array(XHAND_RATED_QPOS_MIN_RAD), np.array(XHAND_RATED_QPOS_MAX_RAD)
    q = lower.copy()  # 包含实验操作范围外、机械范围内的合法位置。
    timestamp = clock.ns
    if case == "roundoff":
        q[0] -= FEEDBACK_POSITION_ROUNDOFF_RAD / 2
        q[1] = upper[1] + FEEDBACK_POSITION_ROUNDOFF_RAD / 2
    elif case == "below":
        q[0] = -0.2
    elif case == "above":
        q[0] = upper[0] + 0.2
    elif case == "nan":
        q[0] = np.nan
    elif case == "shape":
        q = q[:-1]
    elif case == "text":
        q = ["invalid"] * 12
    elif case == "stale":
        timestamp -= int(STOP_FEEDBACK_MAX_AGE_S * 1e9)
    elif case == "missing":
        q = None
    original = np.array(q, copy=True) if q is not None else None
    hand.stop(q, timestamp_ns=timestamp)
    modes, positions = np.array(hand.sent[-1]).T
    if case in {"valid", "roundoff"}:
        assert np.all(modes == 3)
        np.testing.assert_array_equal(positions, np.clip(original, lower, upper))
    else:
        assert np.all(modes == 0)
        if case != "missing":
            assert "stop uses passive" in caplog.text
    if q is not None:
        np.testing.assert_array_equal(q, original)


def test_stop_rechecks_age_after_crc(hand, clock):
    stamp = clock.ns
    sends = []

    def send(*args):
        sends.append([j.mode for j in hand.command.finger_command])
        clock.sleep(STOP_FEEDBACK_MAX_AGE_S)
        return SimpleNamespace(error_code=CRC if len(sends) == 1 else 0, error_message="reply")

    hand.control.send_command.side_effect = send
    hand.stop(np.array(XHAND_RATED_QPOS_MIN_RAD), timestamp_ns=stamp)
    assert sends == [[3] * 12, [0] * 12]


@pytest.mark.parametrize("code", [CRC, 12345])
def test_stop_failure_is_reported(hand, clock, code):
    hand.control.send_command.side_effect = lambda *a: SimpleNamespace(error_code=code, error_message="failure")
    start = clock.ns
    with pytest.raises(XHandError) as error:
        hand.stop(np.array(XHAND_RATED_QPOS_MIN_RAD), timestamp_ns=clock.ns)
    assert error.value.code == code
    assert clock.ns - start <= int(STOP_RETRY_TIMEOUT_S * 1e9)
    if code == CRC:
        assert "timed out" in str(error.value)
    else:
        assert error.value.operation == "send"


def test_stream_write_crc_threshold_and_reset(hand):
    q = np.array(XHAND_RATED_QPOS_MIN_RAD)
    hand.control.send_command.side_effect = lambda *a: SimpleNamespace(error_code=CRC, error_message="CRC")
    for _ in range(MAX_CONSECUTIVE_WRITE_CRC - 1):
        assert hand._send_action(q) is None
    with pytest.raises(XHandError, match="consecutive position write CRC"):
        hand._send_action(q)
    hand.control.send_command.side_effect = lambda *a: SimpleNamespace(error_code=0, error_message="ok")
    assert hand._send_action(q) is None
    assert hand.consecutive_write_crc == 0


@pytest.fixture
def startup_robot(monkeypatch, hand, clock):
    runtime = RealConfig()
    state = SessionState()
    robot = Robot(runtime.arm, runtime.hand, runtime_state=state)
    arm = Mock(is_connected=True)

    def snapshot():
        frame = np.zeros(1, dtype=ARM_STATE_DTYPE)
        frame["timestamp_ns"] = clock.ns
        return frame

    arm.snapshot.side_effect = snapshot

    def connect():
        robot.arm, robot.hand, robot.connected = arm, hand, True
        robot.hand_stop_required = True
        robot.hand_feedback.consume(hand.get_state())
        robot.feedback_next_ns = clock.ns + 33_333_333

    monkeypatch.setattr(robot, "connect", connect)
    return runtime, state, robot


@pytest.mark.parametrize("entry", ["deployment", "teleop"])
@pytest.mark.parametrize("outcome", ["recover", "crc_forever", "cancel", "fault"])
def test_startup_stays_stopped_through_blocking_initialization(
    monkeypatch, startup_robot, hand, clock, entry, outcome, caplog,
):
    import dexmani_real.teleop.run as teleop
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.runtime.observation import ObservationNeeds
    from dexmani_real.teleop.loop import TeleopLoop

    runtime, state, robot = startup_robot
    initial_epoch = state.cancel_epoch
    sensors = Mock()
    initial_stamp = clock.ns
    normal_reply = hand.control.read_state.return_value
    first_crc = True

    def read(*args):
        nonlocal first_crc
        if outcome == "fault":
            return SimpleNamespace(error_code=12345, error_message="device failed"), None
        if first_crc or outcome == "crc_forever":
            first_crc = False
            assert int(robot.hand_feedback.snapshot()["timestamp_ns"][0]) == initial_stamp
            return SimpleNamespace(error_code=CRC, error_message="CRC"), None
        return normal_reply

    def block(*args, **kwargs):
        assert state.stop_signal.is_set() and state.cancel_epoch == initial_epoch
        clock.sleep(2)
        hand.control.read_state.side_effect = read
        if outcome == "cancel":
            state.request_stop()

    sensors.start.side_effect = block
    if entry == "deployment":
        obj = PolicySession.__new__(PolicySession)
        obj.runtime, obj.runtime_state, obj.robot = runtime, state, robot
        obj.startup_cancel_epoch = initial_epoch
        obj.keyboard, obj.sensors, obj.process_context = Mock(), sensors, Mock()
        obj.capture_needs = ObservationNeeds(camera_fields=("rgb",))
        obj.model_needs = ObservationNeeds()
        obj.observations = Mock(camera_static=False)
        obj.camera_calibration_path, obj.record_raw = None, False
        obj.info = SimpleNamespace(observation_fields=(), n_obs_steps=1)
        obj.model = Mock()
        obj.model.poll.return_value.result.return_value = None
        obj.save_run_config = Mock()
        monkeypatch.setattr("dexmani_real.deployment.session.snapshot_camera_parameters", Mock())
        if outcome in {"crc_forever", "cancel", "fault"}:
            error_type, message = {
                "crc_forever": (ValueError, "warmup observation"),
                "cancel": (OperationCancelled, "startup cancelled"),
                "fault": (XHandError, "device failed"),
            }[outcome]
            with pytest.raises(error_type, match=message):
                obj.start_devices()
            if outcome == "crc_forever":
                assert clock.ns <= initial_stamp + 3_010_000_000
        else:
            obj.start_devices()
            assert obj.model.warmup.called
    else:
        # 保留真实入口、initialize_teleop、设备服务及有界 start_observation。
        monkeypatch.setattr(teleop, "Robot", lambda *a, **k: robot)
        monkeypatch.setattr(teleop, "SessionState", lambda: state)
        monkeypatch.setattr(teleop, "SensorChannels", Mock())
        monkeypatch.setattr(teleop, "SensorWorkers", lambda *a: sensors)
        sensors.channels.vr_ring.latest_sequence = 1
        starts = []

        def start(*args, **kwargs):
            if not starts:
                block()
            starts.append(True)

        sensors.start.side_effect = start
        keys = Mock(bindings={})
        keys.poll.return_value = ([], False)
        monkeypatch.setattr(teleop, "Keyboard", lambda **k: keys)
        monkeypatch.setattr(teleop, "AudioFeedback", Mock())
        monkeypatch.setattr(teleop, "ActionMapper", Mock())
        monkeypatch.setattr(teleop, "build_hand_retargeter", lambda *a: None)
        monkeypatch.setattr(teleop, "VRController", lambda *a: SimpleNamespace(hand_retargeter=None))
        monkeypatch.setattr(teleop, "load_vr_transform", lambda *a: np.eye(3))
        monkeypatch.setattr(teleop, "snapshot_camera_parameters", Mock())
        monkeypatch.setattr(teleop, "close_runtime", Mock())
        results = []

        def run(loop):
            loop.cancel_epoch = state.cancel_epoch
            loop.recording_dir = None
            loop.capture_needs = loop.control_needs = ObservationNeeds()
            results.append(loop.start_observation())
            state.request_quit()

        monkeypatch.setattr(TeleopLoop, "run", run)
        result = teleop.run_teleop(runtime, record=True, episodes_dir="unused", max_duration_s=10)
        assert result == (1 if outcome == "fault" else 0)
        if outcome == "fault":
            assert "device failed" in caplog.text
        if outcome in {"recover", "crc_forever"}:
            assert (results[0] is not None) == (outcome == "recover")
        if outcome == "crc_forever":
            assert clock.ns <= initial_stamp + 3_010_000_000
    assert state.stop_signal.is_set()
    if outcome == "recover":
        assert int(robot.hand_feedback.snapshot()["timestamp_ns"][0]) > initial_stamp
    elif outcome == "crc_forever":
        assert int(robot.hand_feedback.snapshot()["timestamp_ns"][0]) == initial_stamp

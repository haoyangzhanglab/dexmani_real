from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.robot.commands import (
    FEEDBACK_POSITION_ROUNDOFF_RAD,
    OperationCancelled,
    RobotError,
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
from dexmani_real.runtime.home import home_hand
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


@pytest.fixture
def home_robot(startup_robot, hand, clock, monkeypatch):
    _, state, robot = startup_robot
    robot.connect()
    target = np.deg2rad(robot.hand_config.home_qpos_deg)
    for joint, value in zip(hand.control.read_state.return_value[1].finger_state, target):
        joint.position = value
    send = robot.send_hand_home
    submissions = []

    def submit(*args, **kwargs):
        submissions.append(clock.ns)
        return send(*args, **kwargs)

    monkeypatch.setattr(robot, "send_hand_home", submit)
    epoch = state.cancel_epoch
    return SimpleNamespace(
        robot=robot, state=state, submissions=submissions,
        run=lambda: home_hand(state, robot, cancel_epoch=epoch, cancel_requested=lambda: False),
    )


@pytest.mark.parametrize("initial", ["stale_crc", "missing_crc", "not_due", "fresh"])
def test_hand_home_waits_for_fresh_joints(home_robot, hand, clock, initial):
    h = home_robot
    old_stamp = int(h.robot.hand_feedback.snapshot()["timestamp_ns"][0])
    if initial != "fresh":
        clock.sleep(1)
    if initial == "missing_crc":
        h.robot.hand_feedback.latest = None
    if initial == "not_due":
        h.robot.feedback_next_ns = clock.ns + 50_000_000
    normal = hand.control.read_state.return_value
    reads = []

    def read(*args):
        reads.append(clock.ns)
        if initial.endswith("crc") and len(reads) <= 2:
            assert not h.submissions
            frame = h.robot.hand_feedback.snapshot()
            assert frame is None or int(frame["timestamp_ns"][0]) == old_stamp
            if len(reads) == 1:
                return SimpleNamespace(error_code=CRC, error_message="CRC"), None
        return normal

    hand.control.read_state.side_effect = read
    start = clock.ns
    h.run()
    assert len(h.submissions) == 1
    if initial == "fresh":
        assert h.submissions[0] == start
    elif initial == "not_due":
        assert h.submissions[0] >= start + 50_000_000
    else:
        assert h.submissions[0] >= reads[1]
    assert sum(t > h.submissions[0] for t in reads) >= 3
    assert h.robot.hand_feedback.snapshot()["tactile_dense_timestamp_ns"][0] == 0
    assert h.state.stop_signal.is_set()
    assert not h.robot.hand_stop_required
    assert h.robot.arm.wait_stopped.call_count == 2


@pytest.mark.parametrize("event", [
    "timeout", "stop", "quit", "estop", "late_read", "cancel_read",
    "device_fault", "service_fault", "sdk_exception", "interrupt",
])
def test_hand_home_initial_wait_failures(home_robot, hand, clock, event):
    h = home_robot
    clock.sleep(1)
    normal = hand.control.read_state.return_value
    start = clock.ns
    deadline = start + int(h.robot.hand_config.home_timeout_s * 1e9)
    failure = RuntimeError("SDK read failed")
    calls = []

    def read(*args):
        calls.append(clock.ns)
        assert not h.submissions
        if event in {"stop", "quit", "estop"}:
            {"stop": h.state.request_stop, "quit": h.state.request_quit,
             "estop": h.state.request_emergency_stop}[event]()
        elif event == "late_read":
            clock.ns = deadline
            return normal
        elif event == "cancel_read":
            h.state.request_stop()
            return normal
        elif event == "device_fault":
            return SimpleNamespace(error_code=12345, error_message="device failed"), None
        elif event == "sdk_exception":
            raise failure
        elif event == "interrupt":
            raise KeyboardInterrupt()
        return SimpleNamespace(error_code=CRC, error_message="CRC"), None

    hand.control.read_state.side_effect = read
    if event == "service_fault":
        h.robot.check_services = lambda: False
    error = (TimeoutError if event in {"timeout", "late_read"} else
             XHandError if event == "device_fault" else
             RobotError if event == "service_fault" else
             RuntimeError if event == "sdk_exception" else
             KeyboardInterrupt if event == "interrupt" else OperationCancelled)
    with pytest.raises(error) as caught:
        h.run()
    assert not h.submissions
    assert clock.ns <= deadline
    if event == "timeout":
        assert clock.ns == deadline
        assert "initial feedback" in str(caught.value)
    elif event == "sdk_exception":
        assert caught.value is failure
    if event != "timeout":
        assert len(calls) <= 1
    assert h.state.stop_signal.is_set()
    assert h.state.estop_request == (event in {"estop", "interrupt"})
    assert h.robot.arm.wait_stopped.call_count == 2


@pytest.mark.parametrize("crc_writes", [0, 1, 2])
def test_hand_home_retry_requires_distinct_post_submission_frames(home_robot, hand, clock, crc_writes):
    h = home_robot
    normal = hand.control.read_state.return_value
    read_stamps = []
    writes = []
    original_send = hand.control.send_command.side_effect

    def send(*args):
        # 初始 stop 使用旧实测位置；HOME 和最终 stop 使用 HOME 位置。
        if np.allclose([j.position for j in hand.command.finger_command],
                       np.deg2rad(h.robot.hand_config.home_qpos_deg)):
            writes.append(clock.ns)
            if len(writes) <= crc_writes:
                return SimpleNamespace(error_code=CRC, error_message="CRC")
        return original_send(*args)

    def read(*args):
        read_stamps.append(clock.ns)
        return normal

    hand.control.send_command.side_effect = send
    hand.control.read_state.side_effect = read
    h.run()
    assert len(h.submissions) == (2 if crc_writes else 1)
    assert sum(t > h.submissions[-1] for t in read_stamps) >= 3
    assert len(writes) == len(h.submissions) + 1  # 最终 stop


def test_hand_home_repeated_cache_cannot_confirm_arrival(home_robot, hand, clock):
    h = home_robot
    normal = hand.control.read_state.return_value
    first = True

    def read(*args):
        nonlocal first
        if first:
            first = False
            return normal
        return SimpleNamespace(error_code=CRC, error_message="CRC"), None

    hand.control.read_state.side_effect = read
    start = clock.ns
    with pytest.raises(TimeoutError, match="convergence"):
        h.run()
    assert len(h.submissions) == 1
    assert clock.ns == start + int(h.robot.hand_config.home_timeout_s * 1e9)


@pytest.mark.parametrize("event", ["cancel", "deadline", "stale"])
def test_hand_home_rechecks_after_service_at_submission(home_robot, clock, event):
    h = home_robot
    deadline = clock.ns + 100_000_000

    def check_services():
        if event == "cancel":
            h.state.request_stop()
        elif event == "deadline":
            clock.ns = deadline
        else:
            clock.sleep(h.robot.hand_config.feedback_max_age_s)
        return True

    h.robot.check_services = check_services
    with pytest.raises({"cancel": OperationCancelled, "deadline": TimeoutError, "stale": RobotError}[event]):
        h.robot.send_hand_home(
            np.deg2rad(h.robot.hand_config.home_qpos_deg), deadline_ns=deadline,
            cancel_requested=lambda: h.state.cancel_epoch != 1,
        )
    assert not h.robot.hand.control.send_command.called


def test_hand_home_stop_time_consumes_total_budget(home_robot, clock):
    h = home_robot
    h.robot.arm.wait_stopped.side_effect = lambda: clock.sleep(h.robot.hand_config.home_timeout_s)
    with pytest.raises(TimeoutError, match="initial feedback"):
        h.run()
    assert not h.submissions


def test_hand_home_cleanup_failure_is_not_hidden_by_cancel(home_robot, hand):
    h = home_robot
    failure = RuntimeError("stop failed")

    def read(*args):
        h.state.request_stop()
        h.robot.arm.wait_stopped.side_effect = failure
        return hand.control.read_state.return_value

    h.robot.feedback_next_ns = 0
    hand.control.read_state.side_effect = read
    with pytest.raises(RuntimeError) as caught:
        h.run()
    assert caught.value is failure
    assert not h.submissions


@pytest.mark.parametrize("event", ["stop", "estop"])
def test_hand_home_cancel_during_wait_sleep(home_robot, hand, clock, monkeypatch, event):
    h = home_robot
    clock.sleep(1)
    normal = hand.control.read_state.return_value
    hand.control.read_state.return_value = (SimpleNamespace(error_code=CRC, error_message="CRC"), None)

    def sleep(seconds):
        clock.sleep(seconds)
        if event == "stop":
            h.state.request_stop()
        else:
            h.state.request_emergency_stop()

    monkeypatch.setattr("time.sleep", sleep)
    with pytest.raises(OperationCancelled):
        h.run()
    assert not h.submissions
    if event == "stop":
        # 普通取消后显式新 HOME 使用新 epoch；无需重建 Robot。
        monkeypatch.setattr("time.sleep", clock.sleep)
        hand.control.read_state.return_value = normal
        home_hand(h.state, h.robot, cancel_epoch=h.state.cancel_epoch, cancel_requested=lambda: False)
        assert len(h.submissions) == 1
    else:
        assert h.state.estop_request


def test_hand_home_initial_wait_does_not_restart_convergence_budget(home_robot, hand, clock):
    h = home_robot
    clock.sleep(1)
    start = clock.ns
    normal = hand.control.read_state.return_value
    recovered = False

    def read(*args):
        nonlocal recovered
        if not recovered and clock.ns - start >= 1_500_000_000:
            recovered = True
            return normal
        return SimpleNamespace(error_code=CRC, error_message="CRC"), None

    hand.control.read_state.side_effect = read
    with pytest.raises(TimeoutError, match="convergence"):
        h.run()
    assert len(h.submissions) == 1
    assert h.submissions[0] >= start + 1_500_000_000
    assert clock.ns == start + int(h.robot.hand_config.home_timeout_s * 1e9)


@pytest.mark.parametrize("stage", ["submission", "convergence_read"])
@pytest.mark.parametrize("cancel", [False, True])
def test_hand_home_blocking_completion_rechecks_cancel_before_timeout(home_robot, hand, clock, stage, cancel):
    h = home_robot
    deadline = clock.ns + int(h.robot.hand_config.home_timeout_s * 1e9)
    normal = hand.control.read_state.return_value

    def complete():
        clock.ns = deadline
        if cancel:
            h.state.request_stop()

    if stage == "submission":
        send = hand.control.send_command.side_effect

        def blocking_send(*args):
            if np.allclose([j.position for j in hand.command.finger_command],
                           np.deg2rad(h.robot.hand_config.home_qpos_deg)):
                complete()
            return send(*args)

        hand.control.send_command.side_effect = blocking_send
    else:
        def blocking_read(*args):
            complete()
            return normal

        hand.control.read_state.side_effect = blocking_read
    with pytest.raises(OperationCancelled if cancel else TimeoutError):
        h.run()
    assert len(h.submissions) == 1
    assert h.state.stop_signal.is_set()

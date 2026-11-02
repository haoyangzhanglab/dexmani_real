"""错误分类与收尾回归：调用真实运行边界，外部设备/IPC 使用替身。"""

import json
import multiprocessing as mp
import sys
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.control.homing import HomePathRejected
from dexmani_real.replay.joint_replay import JointReplay, ReplayStatus
from dexmani_real.replay.run import run_replay
from dexmani_real.replay.trajectory import Trajectory
from dexmani_real.robot import system
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7 import protocol
from dexmani_real.robot.xarm7 import worker as arm_worker
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import ArmChannels, PrepareRequest
from dexmani_real.robot.xhand.driver import XHand
from dexmani_real.robot.xhand.feedback import XHandState
from dexmani_real.runtime.home import _home_motion
from dexmani_real.runtime.lifecycle import SensorWorkers, close_runtime
from dexmani_real.runtime.state import SafetyState, SessionState
from dexmani_real.utils.timing import sample_is_fresh


@pytest.fixture
def devices(monkeypatch):
    runtime = RealConfig()
    state = SessionState(safety_state=SafetyState.RUNNING)
    calls = []
    arm = SimpleNamespace(
        is_connected=True,
        check_error=Mock(),
        submit=Mock(side_effect=lambda *args, **kwargs: calls.append("arm_submit")),
        request_stop=Mock(side_effect=lambda: calls.append("arm_stop")),
        wait_stopped=Mock(side_effect=lambda: calls.append("arm_stopped")),
    )
    hand = SimpleNamespace(
        is_connected=True,
        _send_action=Mock(side_effect=lambda target: calls.append("hand_submit")),
        stop=Mock(side_effect=lambda *args, **kwargs: calls.append("hand_stop")),
    )
    robot = Robot(
        runtime.arm, runtime.hand, run_signal=state.run_signal,
        stop_signal=state.stop_signal, emergency_signal=state.emergency_signal,
    )
    robot.arm, robot.hand, robot.connected = arm, hand, True
    now = 1_000_000_000
    monkeypatch.setattr(system, "time", SimpleNamespace(monotonic_ns=lambda: now))
    robot.hand_feedback.consume(XHandState(np.zeros(12), None, None, now))
    command = RobotCommand(state.run_id, np.asarray(runtime.arm.home_qpos), np.zeros(12))
    return runtime, state, robot, command, calls, now


@pytest.mark.parametrize("failure", ["expired", "stale", "missing", "sdk_timeout", "stop_failure"])
def test_replay_classifies_submission_and_stops_partial_command(devices, monkeypatch, failure):
    runtime, state, robot, command, calls, now = devices
    deadline = now + 100_000_000
    if failure == "expired":
        deadline = now
    elif failure == "stale":
        robot.hand_feedback.consume(XHandState(np.zeros(12), None, None, 1))
    elif failure == "missing":
        robot.hand_feedback.reset()
    elif failure == "sdk_timeout":
        robot.hand._send_action.side_effect = TimeoutError("SDK reply timed out")
    elif failure == "stop_failure":
        robot.hand_feedback.reset()
        robot.arm.wait_stopped.side_effect = TimeoutError("stop acceptance unresolved")

    # 映射和输入准入不是本测试对象；真实 run() 捕获真实组合提交的异常并执行 finally 停止。
    monkeypatch.setattr(
        "dexmani_real.replay.joint_replay.ActionMapper.for_mode", Mock(return_value=None),
    )
    replay = JointReplay(state, runtime, None, robot=robot)
    replay.admit_replay = lambda trajectory: robot.send_action(command, expires_ns=deadline)
    trajectory = Trajectory(
        "fake_episode", 30, "classification", command.arm_qpos[None], command.hand_qpos[None],
        command.arm_qpos, command.hand_qpos,
    )
    result = replay.run(trajectory, run_id=command.run_id, hand_start_duration_s=0)

    expected = ReplayStatus.FAULT if failure in {"sdk_timeout", "stop_failure"} else ReplayStatus.REJECTED
    assert result.status == expected
    assert calls[0] == "arm_submit"
    assert robot.arm.request_stop.call_count == 1
    assert robot.hand.stop.call_count == 1
    assert robot.arm.wait_stopped.call_count == 1
    assert state.stop_signal.is_set()
    if failure != "sdk_timeout":
        robot.hand._send_action.assert_not_called()
    else:
        robot.hand._send_action.assert_called_once()
        assert isinstance(result.errors[0], TimeoutError)
    if failure == "stop_failure":
        assert result.errors[0] is robot.stop_failure
        with pytest.raises(TimeoutError, match="stop acceptance unresolved"):
            robot.check()


@pytest.mark.parametrize("failure", ["expired", "mailbox_lock", "generation_lock", "stop_failure"])
def test_arm_publication_rejection_and_lock_faults_use_real_mailbox(devices, monkeypatch, failure):
    runtime, state, robot, command, _, now = devices
    channels = ArmChannels(
        mp.get_context("spawn"), run_signal=state.run_signal,
        stop_signal=state.stop_signal, emergency_signal=state.emergency_signal,
    )
    original_goal = bytes(channels.goal)
    deadline = now + 100_000_000
    if failure == "stop_failure":
        robot.arm.wait_stopped.side_effect = TimeoutError("stop acceptance unresolved")

    def submit(command, *, expires_ns):
        # 只在锁等待/时钟边界控制调度，目标发布与回放错误分类均走生产实现。
        acquired, release = threading.Event(), threading.Event()
        holder = None
        if failure in {"mailbox_lock", "generation_lock"}:
            lock = channels.goal_lock if failure == "mailbox_lock" else state.run_signal.get_lock()

            def hold_lock():
                with lock:
                    acquired.set()
                    assert release.wait(2)

            holder = threading.Thread(target=hold_lock)
            holder.start()
            assert acquired.wait(2)
            clock = lambda: deadline
        else:
            stamps = iter((now, now, deadline))
            clock = lambda: next(stamps)
        monkeypatch.setattr(protocol, "time", SimpleNamespace(monotonic_ns=clock))
        try:
            channels.publish_goal(command, expires_ns=expires_ns)
        finally:
            release.set()
            if holder is not None:
                holder.join(2)
                assert not holder.is_alive()

    robot.arm.submit.side_effect = submit
    monkeypatch.setattr("dexmani_real.replay.joint_replay.ActionMapper.for_mode", Mock(return_value=None))
    replay = JointReplay(state, runtime, None, robot=robot)
    replay.admit_replay = lambda trajectory: robot.send_action(command, expires_ns=deadline)
    trajectory = Trajectory(
        "fake_episode", 30, "classification", command.arm_qpos[None], command.hand_qpos[None],
        command.arm_qpos, command.hand_qpos,
    )
    result = replay.run(trajectory, run_id=command.run_id, hand_start_duration_s=0)

    assert result.status == (ReplayStatus.REJECTED if failure == "expired" else ReplayStatus.FAULT)
    if failure == "expired":
        assert result.reason == "arm target deadline expired"
        assert result.errors == []
    else:
        assert isinstance(result.errors[0], TimeoutError)
        if failure == "stop_failure":
            assert result.errors == [robot.stop_failure]
    assert bytes(channels.goal) == original_goal
    for lock in (channels.goal_lock, state.run_signal.get_lock()):
        assert lock.acquire(False)
        lock.release()
    robot.hand._send_action.assert_not_called()
    robot.arm.request_stop.assert_called_once()
    robot.arm.wait_stopped.assert_called_once()
    robot.hand.stop.assert_called_once()
    assert state.stop_signal.is_set()


def test_hand_expiration_during_feedback_snapshot_prevents_sdk(devices, monkeypatch):
    _, state, robot, command, _, now = devices
    deadline = now + 10_000_000
    clock = [now]
    snapshot = robot.hand_feedback.snapshot

    def delayed_snapshot():
        frame = snapshot()
        clock[0] = deadline  # 缓存读取期间调度迟延，反馈仍新鲜但目标已到期。
        return frame

    monkeypatch.setattr(system, "time", SimpleNamespace(monotonic_ns=lambda: clock[0]))
    monkeypatch.setattr(robot.hand_feedback, "snapshot", delayed_snapshot)
    with pytest.raises(OperationCancelled, match="target expired before device submission"):
        robot.send_action(command, expires_ns=deadline)
    robot.arm.submit.assert_called_once()
    robot.hand._send_action.assert_not_called()
    assert state.stop_signal.is_set()


def test_stop_attempts_both_devices_and_keeps_first_error(devices, caplog):
    _, _, robot, _, _, _ = devices
    first, second = RobotError("arm request failed"), RuntimeError("hand stop failed")
    robot.stop_required = True
    robot.arm.request_stop.side_effect = first
    robot.hand.stop.side_effect = second
    with pytest.raises(RobotError) as caught:
        robot.stop()
    assert caught.value is first
    robot.arm.wait_stopped.assert_called_once()
    robot.hand.stop.assert_called_once()
    assert robot.stop_required
    assert robot.stop_failure is first
    assert any(record.exc_info and record.exc_info[1] is second for record in caplog.records)
    assert not hasattr(first, "cleanup_errors")


def test_close_runtime_attempts_all_resources_and_reports_other_error(caplog):
    calls = []
    first, second = RuntimeError("robot close failed"), OSError("keyboard close failed")

    def close(stage, error=None):
        calls.append(stage)
        if error is not None:
            raise error

    state = SimpleNamespace(
        stop_signal=threading.Event(), revoke_motion=lambda *args, **kwargs: close("revoke"),
    )
    with pytest.raises(RuntimeError) as caught:
        close_runtime(
            state, SimpleNamespace(close=lambda: close("robot", first)),
            keyboard=SimpleNamespace(stop=lambda: close("keyboard", second)),
            sensors=SimpleNamespace(shutdown=lambda: close("sensors")),
        )
    assert caught.value is first
    assert calls == ["revoke", "robot", "keyboard", "sensors"]
    assert state.stop_signal.is_set()
    assert any(record.exc_info and record.exc_info[1] is second for record in caplog.records)
    assert not hasattr(first, "cleanup_errors")


def test_robot_close_covers_partial_connection_and_keeps_unresolved_handle(devices, caplog):
    _, _, robot, _, _, _ = devices
    first, second = RuntimeError("passive stop failed"), OSError("arm close failed")
    arm = robot.arm
    hand = robot.hand
    robot.connected = False
    arm.is_closed = False
    arm.close = Mock(side_effect=second)
    hand.control = None
    hand.stop.side_effect = first
    hand.disconnect = Mock()
    with pytest.raises(RuntimeError) as caught:
        robot.close()
    assert caught.value is first
    arm.close.assert_called_once()
    hand.disconnect.assert_called_once()
    assert robot.arm is arm
    assert robot.hand is None
    assert robot.stop_required
    assert any(record.exc_info and record.exc_info[1] is second for record in caplog.records)


@pytest.mark.parametrize("main", [OperationCancelled("cancelled"), HomePathRejected("no path")])
def test_home_rejection_cannot_mask_stop_failure(main):
    state = SessionState()
    stop_error = RobotError("HOME stop failed")
    with pytest.raises(RobotError) as caught, _home_motion(
        state, SimpleNamespace(stop=Mock(side_effect=stop_error)),
    ):
        raise main
    assert caught.value is stop_error
    assert caught.value.__context__ is main


def test_home_keeps_technical_primary_traceback(caplog):
    main, stop_error = ValueError("planner broke"), RobotError("HOME stop failed")

    def planning():
        raise main

    with pytest.raises(ValueError) as caught, _home_motion(
        SessionState(), SimpleNamespace(stop=Mock(side_effect=stop_error)),
    ):
        planning()
    assert caught.value is main
    traceback = main.__traceback__
    frames = []
    while traceback is not None:
        frames.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "planning" in frames
    assert any(record.exc_info and record.exc_info[1] is stop_error for record in caplog.records)
    assert not hasattr(main, "cleanup_errors")


def make_arm_owner():
    state = SessionState()
    owner = ArmProcess(
        RealConfig().arm, run_signal=state.run_signal, stop_signal=state.stop_signal,
        emergency_signal=state.emergency_signal,
    )
    owner.pipe.close()
    owner.child_pipe.close()
    owner.process = SimpleNamespace(is_alive=lambda: True, pid=123)
    owner.io.connected.set()
    owner.io.stopped.set()
    return owner


def test_cancelled_arm_request_with_pending_reply_is_fault():
    owner = make_arm_owner()
    owner.pipe = SimpleNamespace(send=Mock(), poll=Mock(return_value=False))
    owner.wait_stopped = Mock()
    abort = Mock(side_effect=[False, True])
    request = PrepareRequest(owner.io.run_id.value, time.monotonic_ns() + 1_000_000_000)
    with pytest.raises(TimeoutError, match="reply remains pending") as caught:
        owner.request_operation(request, service=lambda: None, abort=abort)
    assert isinstance(caught.value.__context__, OperationCancelled)
    assert owner.io.closing.is_set()
    assert owner.io.stop.is_set()
    assert owner.wait_stopped.call_count == 2


def test_cancelled_arm_request_does_not_ignore_timeout_reply():
    owner = make_arm_owner()
    sdk_error = TimeoutError("late SDK timeout")
    owner.pipe = SimpleNamespace(
        send=Mock(), poll=Mock(side_effect=[False, True]), recv=Mock(return_value=sdk_error),
    )
    owner.wait_stopped = Mock()
    request = PrepareRequest(owner.io.run_id.value, time.monotonic_ns() + 1_000_000_000)
    with pytest.raises(RobotError, match="operation also failed") as caught:
        owner.request_operation(request, service=lambda: None, abort=Mock(side_effect=[False, True]))
    assert owner.io.closing.is_set()
    assert owner.wait_stopped.call_count == 2
    assert caught.value.__cause__ is sdk_error


def test_arm_request_keeps_primary_when_cleanup_also_fails(caplog):
    owner = make_arm_owner()
    main, cleanup = ValueError("feedback decode failed"), TimeoutError("stop unresolved")
    owner.wait_stopped = Mock(side_effect=[None, cleanup])
    request = PrepareRequest(owner.io.run_id.value, time.monotonic_ns() + 1_000_000_000)
    with pytest.raises(ValueError) as caught:
        owner.request_operation(request, service=Mock(side_effect=main), abort=lambda: False)
    assert caught.value is main
    assert owner.io.closing.is_set()
    assert any(record.exc_info and record.exc_info[1] is cleanup for record in caplog.records)


def test_sensor_shutdown_attempts_all_and_retains_only_unresolved(caplog):
    workers = SensorWorkers()
    first, second = RuntimeError("first join failed"), OSError("second join failed")
    failed = [
        SimpleNamespace(name="first", join=Mock(side_effect=first)),
        SimpleNamespace(name="second", join=Mock(side_effect=second)),
    ]
    finished = SimpleNamespace(name="finished", join=Mock(), is_alive=lambda: False, exitcode=0)
    workers.started = [*failed, finished]
    with pytest.raises(RuntimeError) as caught:
        workers.shutdown()
    assert caught.value is first
    finished.join.assert_called_once()
    assert workers.started == failed
    assert any(record.exc_info and record.exc_info[1] is second for record in caplog.records)


def test_hand_driver_is_connection_error_boundary():
    driver = XHand(RealConfig().hand)
    with pytest.raises(RobotError, match="not connected"):
        driver.get_state()


@pytest.mark.parametrize("cancelled", [False, True])
def test_arm_request_stop_request_failure_still_confirms_stop(cancelled, caplog):
    owner = make_arm_owner()
    main = OperationCancelled("cancelled") if cancelled else ValueError("feedback failed")
    stop_error = OSError("stop request failed")
    owner.request_stop = Mock(side_effect=[None, stop_error])
    owner.wait_stopped = Mock()
    request = PrepareRequest(owner.io.run_id.value, time.monotonic_ns() + 1_000_000_000)
    expected = OSError if cancelled else ValueError
    with pytest.raises(expected) as caught:
        owner.request_operation(request, service=Mock(side_effect=main), abort=lambda: False)
    assert caught.value is (stop_error if cancelled else main)
    assert owner.wait_stopped.call_count == 2
    assert owner.io.closing.is_set()
    if not cancelled:
        assert any(record.exc_info and record.exc_info[1] is stop_error for record in caplog.records)


def test_hand_disconnect_attempts_device_close_after_init_failure(monkeypatch, caplog):
    driver = XHand(RealConfig().hand)
    driver.cfg = SimpleNamespace(comm_type="ethercat")
    first, second = RobotError("INIT rejected"), OSError("device close failed")
    driver.connected_flag = True
    driver.control = SimpleNamespace(close_device=Mock(side_effect=second))
    driver.request_ethercat_init = Mock(side_effect=first)
    monkeypatch.setattr("dexmani_real.robot.xhand.driver.time.sleep", lambda duration: None)
    with pytest.raises(RobotError) as caught:
        driver.disconnect()
    assert caught.value is first
    driver.control.close_device.assert_called_once()
    assert not driver.connected_flag
    assert any(record.exc_info and record.exc_info[1] is second for record in caplog.records)


def test_cancelled_arm_request_accepts_confirmed_owner_budget_reply():
    from dexmani_real.robot.xarm7.worker import _OperationTimeout

    owner = make_arm_owner()
    owner.pipe = SimpleNamespace(
        send=Mock(), poll=Mock(side_effect=[False, True]),
        recv=Mock(return_value=_OperationTimeout("owner wait budget ended")),
    )
    owner.wait_stopped = Mock()
    request = PrepareRequest(owner.io.run_id.value, time.monotonic_ns() + 1_000_000_000)
    with pytest.raises(OperationCancelled, match="arm operation cancelled"):
        owner.request_operation(request, service=lambda: None, abort=Mock(side_effect=[False, True]))
    assert owner.wait_stopped.call_count == 2
    assert owner.io.stop.is_set()
    assert not owner.io.closing.is_set()


@pytest.mark.parametrize("stage", ["health", "shutdown"])
@pytest.mark.parametrize("independent", [False, True])
def test_replay_session_reports_cached_fault_once(tmp_path, monkeypatch, stage, independent):
    from dexmani_real.replay import run
    from dexmani_real.replay.joint_replay import ReplayResult

    runtime = RealConfig()
    primary = RobotError("already reported by replay")
    later = OSError("independent failure") if independent else primary
    result = ReplayResult(ReplayStatus.FAULT, errors=[primary])
    robot = Mock()
    if stage == "health":
        robot.check.side_effect = [None, later]
    else:
        robot.close.side_effect = later
    keyboard = Mock()
    monkeypatch.setattr(run, "Robot", Mock(return_value=robot))
    monkeypatch.setattr(run, "Keyboard", Mock(return_value=keyboard))
    monkeypatch.setattr(run, "JointReplay", Mock(return_value=SimpleNamespace(run=lambda *a, **kw: result)))
    hand = np.deg2rad(runtime.hand.home_qpos_deg)
    trajectory = Trajectory(
        "fake_episode", 30, "classification", np.asarray(runtime.arm.home_qpos)[None],
        hand[None], np.asarray(runtime.arm.home_qpos), hand,
    )
    outcome = run_replay(trajectory, runtime, output_dir=tmp_path)
    assert outcome.errors == ([primary, later] if independent else [primary])
    assert outcome.status is ReplayStatus.FAULT
    assert outcome.reason.count(str(primary)) == 1
    robot.close.assert_called_once()
    keyboard.stop.assert_called_once()
    assert json.loads((tmp_path / "session_result.json").read_text())["status"] == "fault"


def test_hand_startup_cancel_cannot_hide_disconnect_failure(monkeypatch):
    driver = XHand(RealConfig().hand)
    failure = OSError("XHand close failed")
    control = SimpleNamespace(close_device=Mock(side_effect=failure))
    monkeypatch.setitem(sys.modules, "xhand_controller", SimpleNamespace(xhand_control=object()))
    monkeypatch.setattr(driver, "open_device", lambda: setattr(driver, "control", control))
    with pytest.raises(OSError) as caught:
        driver.connect(cancel_requested=Mock(side_effect=[False, True]))
    assert caught.value is failure
    assert isinstance(failure.__context__, OperationCancelled)
    assert driver.control is control and not driver.connected_flag


@pytest.mark.parametrize("independent", [False, True])
def test_replay_stop_does_not_repeat_cached_operation_error(devices, monkeypatch, independent):
    runtime, state, robot, command, _, _ = devices
    primary = RobotError("cached arm error")
    stop_error = RuntimeError("independent stop failure") if independent else primary
    monkeypatch.setattr("dexmani_real.replay.joint_replay.ActionMapper.for_mode", lambda *a, **kw: None)
    replay = JointReplay(state, runtime, None, robot=robot)
    replay.admit_replay = Mock(side_effect=primary)
    robot.stop = Mock(side_effect=stop_error)
    trajectory = Trajectory(
        "fake_episode", 30, "classification", command.arm_qpos[None], command.hand_qpos[None],
        command.arm_qpos, command.hand_qpos,
    )
    result = replay.run(trajectory, run_id=command.run_id, hand_start_duration_s=0)
    assert result.errors == ([primary, stop_error] if independent else [primary])
    assert result.status is ReplayStatus.FAULT


@pytest.mark.parametrize("stamp,expected", [
    (0, False), (1_000_000_001, False), (900_000_000, False),
    (900_000_001, True), (1_000_000_000, True),
])
def test_sample_freshness_boundaries(stamp, expected):
    assert sample_is_fresh(stamp, 0.1, 1_000_000_000) is expected


def test_feedback_rate_can_exceed_servo_rate():
    replace(RealConfig().arm, feedback_hz=50, servo_hz=25).validate()


@pytest.mark.parametrize("changes", [
    {"servo_hz": 0}, {"servo_hz": float("inf")},
    {"feedback_hz": -1}, {"feedback_hz": float("nan")},
    {"feedback_max_age_s": 0},
    {"feedback_hz": 50, "feedback_max_age_s": 0.02},
    {"feedback_hz": 50, "feedback_max_age_s": 0.01},
])
def test_feedback_config_still_rejects_invalid_rates_and_budget(changes):
    with pytest.raises(ValueError):
        replace(RealConfig().arm, **changes).validate()


def test_robot_cache_reads_preserve_each_source_timestamp(devices, monkeypatch):
    _, _, robot, _, _, now = devices
    robot.arm.snapshot = Mock(return_value={"timestamp_ns": np.array([now])})
    robot.hand.get_state = Mock(side_effect=AssertionError("cache read queried SDK"))
    robot.hand_feedback.consume(XHandState(
        np.zeros(12), np.ones((5, 3)), np.ones((5, 120, 3)), now,
    ))
    # 新关节反馈没有触觉数据时，两路触觉继续保留各自旧来源时间。
    robot.hand_feedback.consume(XHandState(np.ones(12), None, None, now + 1_000_000_000))
    monkeypatch.setattr(system, "time", SimpleNamespace(monotonic_ns=lambda: now + 2_000_000_000))
    for _ in range(2):
        state = robot.read_state()
        assert state.arm["timestamp_ns"][0] == now
        assert state.hand["timestamp_ns"][0] == now + 1_000_000_000
        for channel in ("aggregate", "dense"):
            assert state.hand[f"tactile_{channel}_timestamp_ns"][0] == now
            np.testing.assert_array_equal(state.hand[f"tactile_{channel}"], 1)
    robot.hand.get_state.assert_not_called()


@pytest.mark.parametrize("expires", ["command", "feedback"])
def test_arm_rechecks_expiration_after_trajectory_calculation(devices, monkeypatch, expires):
    runtime, state, _, command, _, now = devices
    channels = ArmChannels(
        mp.get_context("spawn"), run_signal=state.run_signal,
        stop_signal=state.stop_signal, emergency_signal=state.emergency_signal,
    )
    driver = SimpleNamespace(servo=Mock())
    owner = arm_worker.ArmController(runtime.arm, channels, None, driver)
    owner.status["qpos"] = runtime.arm.home_qpos
    owner.status["timestamp_ns"] = now
    owner.prepared_run_id = state.run_id
    target = command.arm_qpos.copy()
    target[0] += 0.01
    elapsed = (10_000_000 if expires == "command"
               else int(runtime.arm.feedback_max_age_s * 1e9))
    deadline = now + (elapsed if expires == "command" else 1_000_000_000)
    monkeypatch.setattr(protocol, "time", SimpleNamespace(monotonic_ns=lambda: now))
    channels.publish_goal(RobotCommand(state.run_id, target, None), expires_ns=deadline)
    clock = iter((now, now + elapsed))
    monkeypatch.setattr(arm_worker, "time", SimpleNamespace(monotonic_ns=lambda: next(clock)))
    assert owner.tick() is None
    assert owner.reference is not None  # 使用真实 Ruckig 计算后，才推进时钟到到期边界。
    driver.servo.assert_not_called()
    assert channels.stop.is_set()

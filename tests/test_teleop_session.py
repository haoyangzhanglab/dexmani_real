"""遥操作顺序生命周期：未完成准备不提交，取消/失败仍停止并释放本段。"""

import json
import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionResult
from dexmani_real.data.recorder import RecordingError
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import SensorWorkers
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop import loop as loop_module, run as run_module
from dexmani_real.teleop.loop import TeleopLoop


@pytest.fixture
def teleop():
    runtime = RealConfig()
    runtime = replace(runtime, hand=replace(runtime.hand, enabled=False))
    state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
    keys = Keyboard(
        bindings={"s": state.request_stop, "q": state.request_quit},
        estop_callback=state.request_emergency_stop,
    )
    keys.active = True
    keys.listener = Mock(is_alive=Mock(return_value=True))

    def read_state():
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        arm["timestamp_ns"] = time.monotonic_ns()
        return SimpleNamespace(arm=arm, hand=None)

    robot = Mock()
    robot.stop.side_effect = state.stop_signal.set
    robot.read_state.side_effect = read_state
    robot.prepare_streaming.side_effect = lambda *args, **kwargs: state.stop_signal.clear()
    # 提交一个目标后按 S；下一轮真实事件处理结束本段。
    robot.send_action.side_effect = lambda *args, **kwargs: keys.handle_key("s", True)
    vr_ring = SimpleNamespace(
        latest_sequence=1,
        read=lambda: ({"recv_ts_ns": time.monotonic_ns()}, 1),
    )
    controller = Mock(compute_target=Mock(return_value=ActionResult(np.zeros(7), None)))
    loop = TeleopLoop(
        state, runtime, robot, task_label="test", controller=controller,
        sensors=SensorWorkers(SimpleNamespace(vr_ring=vr_ring)),
        recording_dir=None, max_duration_s=10, keyboard=keys, audio=Mock(),
    )
    return loop


def test_two_episodes_stop_clear_and_restart(teleop):
    for _ in range(2):
        teleop.keyboard.allow_start()
        teleop.keyboard.handle_key("b", True)
        commands, _ = teleop.keyboard.poll(teleop.runtime_state, timeout=0)
        teleop.handle_commands(commands)
        assert teleop.cancel_epoch is None
        assert teleop.runtime_state.stop_signal.is_set()
        assert teleop.pending_termination_reason == "operator"
        teleop.keyboard.handle_key("b", False)
        teleop.keyboard.handle_key("s", False)
    assert teleop.robot.send_action.call_count == 2
    assert teleop.robot.prepare_streaming.call_count == 2
    assert teleop.controller.reset.call_count == 2
    assert teleop.controller.clear.call_count == 2
    assert teleop.robot.stop.call_count == 4


@pytest.mark.parametrize("interruption", ["s", "q", "esc", "fault"])
def test_interrupted_preparation_never_submits(teleop, interruption):
    def prepare(*args, **kwargs):
        teleop.runtime_state.stop_signal.clear()
        if interruption == "fault":
            raise RobotError("prepare failed")
        teleop.keyboard.handle_key(interruption, True)

    teleop.robot.prepare_streaming.side_effect = prepare
    if interruption == "fault":
        with pytest.raises(RobotError, match="prepare failed"):
            teleop.run_episode(teleop.runtime_state.cancel_epoch)
        assert teleop.failed
    else:
        teleop.run_episode(teleop.runtime_state.cancel_epoch)
    teleop.robot.send_action.assert_not_called()
    teleop.controller.compute_target.assert_not_called()
    teleop.controller.clear.assert_called_once()
    assert teleop.robot.stop.call_count == 2
    assert teleop.cancel_epoch is None
    assert teleop.runtime_state.stop_signal.is_set()


def test_observation_loss_after_preparation_never_submits(teleop, monkeypatch):
    monkeypatch.setattr(loop_module, "INITIAL_STATE_TIMEOUT_S", 0)

    def prepare(*args, **kwargs):
        teleop.runtime_state.stop_signal.clear()
        teleop.sensors.channels.vr_ring.read = lambda: None

    teleop.robot.prepare_streaming.side_effect = prepare
    teleop.run_episode(teleop.runtime_state.cancel_epoch)
    teleop.robot.prepare_streaming.assert_called_once()
    teleop.robot.send_action.assert_not_called()
    teleop.controller.clear.assert_called_once()
    assert "vr_unavailable" in teleop.termination_details[0]["reason"]
    assert teleop.cancel_epoch is None
    assert teleop.runtime_state.stop_signal.is_set()


def test_revoked_start_does_not_enter_or_cleanup_episode(teleop):
    epoch = teleop.runtime_state.cancel_epoch
    teleop.runtime_state.request_stop()
    teleop.run_episode(epoch)
    teleop.robot.prepare_streaming.assert_not_called()
    teleop.robot.send_action.assert_not_called()
    teleop.robot.stop.assert_not_called()
    teleop.controller.clear.assert_not_called()
    assert teleop.cancel_epoch is None


def test_partial_recorder_start_is_finished_before_return(teleop, monkeypatch, tmp_path):
    teleop.recording_dir = tmp_path
    teleop.robot.hand.tactile_reset_metadata = {}
    teleop.controller.rotation.as_matrix.return_value = np.eye(3)
    teleop.controller.hand_retargeter = None
    teleop.runtime = replace(
        teleop.runtime, arm=replace(teleop.runtime.arm, homing=object()),
        environment=replace(teleop.runtime.environment, home_model_confirmed=object(), static_boxes=(object(),)),
        execution=replace(teleop.runtime.execution, rtc_guidance_cap=object()),
    )
    monkeypatch.setattr(loop_module, "camera_metadata", lambda *args: {})
    recorder = Mock(
        error=RecordingError("writer failed"),
        start=Mock(side_effect=RecordingError("writer failed")),
        poll_finish=Mock(return_value=True),
        result=Mock(return_value=None),
    )
    monkeypatch.setattr(loop_module, "EpisodeRecorder", Mock(return_value=recorder))
    teleop.run_episode(teleop.runtime_state.cancel_epoch)
    metadata = recorder.start.call_args.kwargs["metadata"]
    parameters = json.loads(metadata["effective_parameters"])
    assert "homing" not in parameters["arm"]
    assert "home_model_confirmed" not in parameters["environment"]
    assert "execution" not in parameters
    assert parameters["vr_to_robot_rotation"] == np.eye(3).tolist()
    recorder.request_finish.assert_called_once()
    recorder.poll_finish.assert_called_once()
    teleop.robot.prepare_streaming.assert_not_called()
    teleop.robot.send_action.assert_not_called()
    assert teleop.recorder is None
    assert teleop.cancel_epoch is None
    assert teleop.failed


@pytest.mark.parametrize("outcome", ["success", "before_stop", "epoch", "poll_fault", "stop_fault"])
def test_tactile_reset_uses_callback_faults_and_preserves_cancel_stop(outcome):
    from dexmani_real.robot.system import Robot

    runtime = RealConfig()
    state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
    robot = Robot(runtime.arm, runtime.hand, runtime_state=state)
    robot.hand = Mock()
    robot.stop = Mock()
    robot.check = Mock()

    def poll():
        robot.check()
        if outcome == "before_stop":
            return True
        if outcome == "epoch" and callback.call_count == 2:
            state.request_stop()
        if outcome == "poll_fault" and callback.call_count == 3:
            raise RobotError("device failed during reset")
        return False

    callback = Mock(side_effect=poll)

    def reset(*, cancel_requested):
        assert not cancel_requested()
        return {2: 0}

    robot.hand.reset_tactile_sensors.side_effect = reset
    if outcome == "stop_fault":
        robot.stop.side_effect = RobotError("stop failed")
    if outcome == "success":
        assert robot.reset_tactile_sensors(cancel_epoch=1, cancel_requested=callback) == {2: 0}
    else:
        error = OperationCancelled if outcome in {"before_stop", "epoch"} else RobotError
        with pytest.raises(error):
            robot.reset_tactile_sensors(cancel_epoch=1, cancel_requested=callback)
    assert robot.check.call_count == callback.call_count
    assert robot.stop.call_count == (0 if outcome == "before_stop" else 1)
    if outcome in {"before_stop", "epoch", "stop_fault"}:
        robot.hand.reset_tactile_sensors.assert_not_called()


@pytest.mark.parametrize("outcome", ["ready", "cancel", "exited", "timeout"])
def test_vr_startup_uses_worker_wait_and_keeps_servicing(teleop, monkeypatch, outcome):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(run_module.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(run_module.time, "sleep", lambda dt: setattr(clock, "now", clock.now + dt))
    monkeypatch.setattr(run_module, "VR_READY_TIMEOUT_S", 0.2)
    calibrate = Mock()
    monkeypatch.setattr(run_module, "calibrate_hand", calibrate)
    ring = teleop.sensors.channels.vr_ring
    ring.latest_sequence = 0
    process = Mock(name="vr", is_alive=Mock(return_value=True))

    def service():
        if teleop.robot.service_idle.call_count < 3:
            return
        if outcome == "ready":
            ring.latest_sequence = 1
        elif outcome == "cancel":
            teleop.keyboard.handle_key("s", True)
        elif outcome == "exited":
            process.is_alive.return_value = False

    teleop.robot.service_idle.side_effect = service
    if outcome == "ready":
        run_module.initialize_teleop(teleop, process, cancel_epoch=teleop.runtime_state.cancel_epoch)
        calibrate.assert_called_once()
    else:
        expected = OperationCancelled if outcome == "cancel" else RuntimeError
        with pytest.raises(expected):
            run_module.initialize_teleop(teleop, process, cancel_epoch=teleop.runtime_state.cancel_epoch)
        calibrate.assert_not_called()
    process.start.assert_called_once()
    assert teleop.robot.service_idle.call_count >= 3
    assert process in teleop.sensors.processes

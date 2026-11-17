"""遥操作顺序生命周期：未完成准备不提交，取消/失败仍停止并释放本段。"""

import json
import threading
import time
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.control.actions import ActionResult
from dexmani_real.data.raw import camera_metadata, robot_recording_parameters
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.geometry.pose import Pose
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.lifecycle import SensorWorkers
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop import loop as loop_module
from dexmani_real.teleop import run as run_module
from dexmani_real.teleop.loop import TeleopLoop


@pytest.mark.parametrize("episodes_dir", ["captures", "~/captures"])
def test_collection_directory_is_resolved_from_cwd_or_home(tmp_path, monkeypatch, episodes_dir):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
    monkeypatch.setattr(run_module, "configure_logging", lambda: None)
    monkeypatch.setattr(run_module, "SessionState", lambda: state)
    monkeypatch.setattr(run_module.mp, "get_context", Mock())
    monkeypatch.setattr(run_module, "load_vr_transform", Mock())
    monkeypatch.setattr(run_module, "ActionMapper", Mock())
    monkeypatch.setattr(run_module, "build_hand_retargeter", Mock())
    monkeypatch.setattr(run_module, "VRController", Mock())
    monkeypatch.setattr(run_module, "SensorChannels", Mock())
    monkeypatch.setattr(run_module, "SensorWorkers", Mock())
    robot = Mock()
    monkeypatch.setattr(run_module, "Robot", robot)
    monkeypatch.setattr(run_module, "Keyboard", Mock(return_value=Mock(bindings={})))
    monkeypatch.setattr(run_module, "AudioFeedback", Mock())
    monkeypatch.setattr(run_module, "snapshot_camera_parameters", Mock())
    monkeypatch.setattr(run_module, "initialize_teleop", Mock())
    monkeypatch.setattr(run_module, "close_runtime", Mock())
    loop = Mock(session_failed=False)
    loop.run.side_effect = state.request_quit
    constructor = Mock(return_value=loop)
    monkeypatch.setattr(run_module, "TeleopLoop", constructor)
    assert run_module.run_teleop(
        RealConfig(), record=True, episodes_dir=episodes_dir, max_duration_s=10, task_name="experiment_label",
    ) == 0
    assert constructor.call_args.kwargs["recording_dir"] == tmp_path / "captures/experiment_label"
    assert constructor.call_args.kwargs["task_label"] == "experiment_label"


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


@pytest.fixture
def active_teleop(teleop, monkeypatch):
    clock = SimpleNamespace(now=10_000_000_000)
    monkeypatch.setattr(loop_module.time, "monotonic_ns", lambda: clock.now)
    monkeypatch.setattr(loop_module.time, "monotonic", lambda: clock.now / 1e9)
    teleop.robot.send_action.side_effect = None
    assert teleop.start_episode(teleop.runtime_state.cancel_epoch)
    return teleop, clock


def control_row(teleop):
    return loop_module.read_observation(
        teleop.sensors.channels, teleop.robot, needs=teleop.capture_needs,
    )


def test_two_episodes_stop_clear_and_restart(teleop):
    for _ in range(2):
        teleop.keyboard.allow_start()
        teleop.keyboard.handle_key("b", True)
        commands, _ = teleop.keyboard.poll(teleop.runtime_state, timeout=0)
        teleop.handle_commands(commands)
        assert teleop.episode_cancel_epoch is None
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
        assert teleop.session_failed
    else:
        teleop.run_episode(teleop.runtime_state.cancel_epoch)
    teleop.robot.send_action.assert_not_called()
    teleop.controller.compute_target.assert_not_called()
    teleop.controller.clear.assert_called_once()
    assert teleop.robot.stop.call_count == 2
    assert teleop.episode_cancel_epoch is None
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
    assert teleop.episode_cancel_epoch is None
    assert teleop.runtime_state.stop_signal.is_set()


def test_revoked_start_does_not_enter_or_cleanup_episode(teleop):
    epoch = teleop.runtime_state.cancel_epoch
    teleop.runtime_state.request_stop()
    teleop.run_episode(epoch)
    teleop.robot.prepare_streaming.assert_not_called()
    teleop.robot.send_action.assert_not_called()
    teleop.robot.stop.assert_not_called()
    teleop.controller.clear.assert_not_called()
    assert teleop.episode_cancel_epoch is None


@pytest.mark.parametrize("retargeting", [False, True])
def test_recording_parameters_and_start_failure_cleanup(teleop, monkeypatch, tmp_path, retargeting):
    teleop.recording_dir = tmp_path
    teleop.camera_snapshot = CameraSnapshot(
        "camera-123", CameraIntrinsics(16, 16, 20., 20., 8., 8., "none", (0.,) * 5), .001, np.eye(4),
    )
    teleop.robot.hand.tactile_reset_metadata = {
        "hand_tactile_reset_time_ns": 123456789, "hand_tactile_reset_result": {2: 0},
    }
    teleop.controller.rotation.as_matrix.return_value = np.eye(3)
    calibration = SimpleNamespace(bone_lengths=np.arange(20) / 100, palm_width=.08)
    teleop.controller.hand_retargeter = SimpleNamespace(calibration=calibration) if retargeting else None
    teleop.runtime = replace(
        teleop.runtime, arm=replace(teleop.runtime.arm, homing=object()),
        environment=replace(teleop.runtime.environment, home_model_confirmed=object(), static_boxes=(object(),)),
        execution=replace(teleop.runtime.execution, rtc_guidance_cap=object()),
    )
    recorder = Mock(
        error=RecordingError("writer failed"),
        start=Mock(side_effect=RecordingError("writer failed")),
        poll_finish=Mock(return_value=True),
        result=Mock(return_value=None),
    )
    monkeypatch.setattr(loop_module, "EpisodeRecorder", Mock(return_value=recorder))
    teleop.run_episode(teleop.runtime_state.cancel_epoch)
    parameters = json.loads(recorder.start.call_args.kwargs["metadata"]["effective_parameters"])
    runtime = teleop.runtime
    expected = {
        **robot_recording_parameters(runtime),
        "teleop": asdict(runtime.teleop), "max_duration_s": 10,
        "motion": asdict(runtime.motion),
        "environment": {
            "workspace": asdict(runtime.environment.workspace),
            "eef_workspace_tolerance_m": runtime.environment.eef_workspace_tolerance_m,
        },
        "camera_config": asdict(runtime.camera),
        "vr": asdict(runtime.vr), "vr_to_robot_rotation": np.eye(3).tolist(),
        "camera": camera_metadata(teleop.camera_snapshot),
        "tactile_reset": teleop.robot.hand.tactile_reset_metadata,
    }
    if retargeting:
        expected.update(
            hand_retargeting=asdict(runtime.hand_retargeting),
            hand_retargeting_calibration={"bone_lengths": calibration.bone_lengths.tolist(), "palm_width": .08},
        )
    assert parameters == json.loads(json.dumps(expected, allow_nan=False))
    recorder.request_finish.assert_called_once()
    recorder.poll_finish.assert_called_once()
    teleop.robot.prepare_streaming.assert_not_called()
    teleop.robot.send_action.assert_not_called()
    assert teleop.recorder is None
    assert teleop.episode_cancel_epoch is None
    assert teleop.session_failed


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


def test_failed_submission_records_attempt_without_advancing_state(active_teleop, tmp_path):
    teleop, clock = active_teleop
    teleop.execute_control_step(control_row(teleop))
    previous, deadline = teleop.last_command, teleop.target_deadline_ns
    teleop.controller.commit.reset_mock()
    clock.now += 10_000_000
    teleop.controller.compute_target.return_value = ActionResult(np.full(7, .1), None)
    recorder = teleop.recorder = EpisodeRecorder(tmp_path, control_hz=teleop.runtime.teleop.control_hz)
    recorder.is_recording = True  # 仅验证真实入队契约，不启动 writer。
    failure = RobotError("device submission failed")
    teleop.robot.send_action.side_effect = failure
    row = control_row(teleop)
    with pytest.raises(RobotError) as caught:
        teleop.execute_control_step(row)
    assert caught.value is failure
    recorded, command, _ = recorder.queue.get_nowait()
    assert command is teleop.robot.send_action.call_args.args[0]
    np.testing.assert_array_equal(command.arm_qpos, np.full(7, .1))
    assert recorded.control_timestamp_ns == row.control_timestamp_ns
    assert recorder.queue.empty()
    teleop.controller.commit.assert_not_called()
    assert teleop.last_command is previous
    assert teleop.target_deadline_ns == deadline
    assert teleop.runtime_state.stop_signal.is_set()


def test_rejected_ik_resubmits_previous_command_without_renewing_lease(active_teleop):
    teleop, clock = active_teleop
    pose = Pose(np.zeros(3), np.array([1., 0., 0., 0.]))
    teleop.controller.compute_target.return_value = ActionResult(np.zeros(7), None, eef_pose=pose)
    teleop.execute_control_step(control_row(teleop))
    command, deadline = teleop.last_command, teleop.target_deadline_ns
    teleop.controller.previous_eef_target = pose
    clock.now += 10_000_000
    teleop.controller.compute_target.return_value = ActionResult(None, None, "unreachable")
    teleop.execute_control_step(control_row(teleop))
    assert teleop.robot.send_action.call_count == 2
    assert teleop.robot.send_action.call_args.args[0] is command
    assert teleop.robot.send_action.call_args.kwargs["expires_ns"] == deadline
    assert teleop.controller.commit.call_args.kwargs["eef_pose"] is pose
    assert teleop.last_command is command and teleop.target_deadline_ns == deadline
    assert teleop.pending_termination_reason is None


@pytest.mark.parametrize("expired", ["observation", "previous_target"])
def test_expired_target_stops_episode_without_submission(active_teleop, expired):
    teleop, clock = active_teleop
    if expired == "previous_target":
        teleop.execute_control_step(control_row(teleop))
        clock.now = teleop.target_deadline_ns
    row = control_row(teleop)
    if expired == "observation":
        row = replace(row, control_timestamp_ns=clock.now - int(teleop.runtime.teleop.target_max_age_s * 1e9))
    teleop.robot.send_action.reset_mock()
    teleop.controller.commit.reset_mock()
    teleop.execute_control_step(row)
    teleop.robot.send_action.assert_not_called()
    teleop.controller.commit.assert_not_called()
    assert teleop.pending_termination_reason == "cancelled"
    assert teleop.termination_details == [{"stage": "termination", "reason": "target_expired"}]
    assert teleop.runtime_state.stop_signal.is_set()
    assert not teleop.service_episode(0)
    teleop.end_episode()
    assert teleop.episode_cancel_epoch is None
    teleop.controller.clear.assert_called_once()
    assert teleop.robot.stop.call_count == 2


def test_session_failure_survives_successful_new_episode(teleop):
    prepare = teleop.robot.prepare_streaming.side_effect
    teleop.robot.prepare_streaming.side_effect = RobotError("prepare failed")
    with pytest.raises(RobotError):
        teleop.run_episode(teleop.runtime_state.cancel_epoch)
    assert teleop.session_failed
    teleop.robot.prepare_streaming.side_effect = prepare
    teleop.run_episode(teleop.runtime_state.cancel_epoch)
    teleop.robot.send_action.assert_called_once()
    assert teleop.pending_termination_reason == "operator"
    assert teleop.session_failed and teleop.episode_cancel_epoch is None

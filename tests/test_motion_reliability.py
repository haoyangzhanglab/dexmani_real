from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.xarm7 import (
    HOME_CONVERGENCE_TIMEOUT_S,
    HOME_DWELL_S,
    ArmConfig,
)
from dexmani_real.config.xhand import HandConfig
from dexmani_real.control.homing import home_segment_timeout_s
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.runtime.home import home_arm
from dexmani_real.runtime.state import SafetyState, SessionState


class Clock:
    def __init__(self):
        self.now_ns = 1_000_000_000

    def advance(self, seconds):
        self.now_ns += round(seconds * 1e9)

    def monotonic_ns(self):
        return self.now_ns

    def monotonic(self):
        return self.now_ns / 1e9


def test_home_budget_includes_acceleration_and_parent_phases(monkeypatch):
    from dexmani_real.runtime import home

    clock = Clock()
    monkeypatch.setattr(home.time, "monotonic_ns", clock.monotonic_ns)
    monkeypatch.setattr(home.time, "monotonic", clock.monotonic)
    cfg = replace(ArmConfig(), max_joint_acceleration_deg_per_s2=np.rad2deg(0.01))
    segment = home_segment_timeout_s(0.1, cfg)
    assert segment > 2 * np.sqrt(0.1 / 0.01) + HOME_DWELL_S
    target = np.asarray(cfg.home_qpos)
    start = target.copy()
    start[0] += 0.1
    arm = np.zeros(1, dtype=[("qpos", "f8", (7,)), ("qvel", "f8", (7,)), ("timestamp_ns", "u8")])
    arm["qpos"] = start
    robot = SimpleNamespace(
        arm_config=cfg, hand_config=replace(HandConfig(), enabled=False),
        stop_required=False, home_arm=Mock(),
    )

    def feedback():
        clock.advance(0.01)
        arm["timestamp_ns"] = clock.now_ns

    robot.service_feedback = feedback
    robot.read_state = lambda: SimpleNamespace(arm=arm)
    state = SimpleNamespace(safety_state=SafetyState.ARMED, run_id=1, can_move=lambda **_: True)
    for waypoints in (np.stack((start, target)), np.empty((0, 7))):
        planner = SimpleNamespace(set_hand_qpos=Mock(), plan=lambda *_: waypoints)
        assert home_arm(state, robot, planner, cancel_requested=lambda: False).ok
        duration = (robot.home_arm.call_args.kwargs["deadline_ns"] - clock.now_ns) / 1e9
        assert duration >= HOME_CONVERGENCE_TIMEOUT_S + (
            segment if len(waypoints) else HOME_CONVERGENCE_TIMEOUT_S
        )


def test_empty_home_observes_dwell_without_counting_query_time(monkeypatch):
    from dexmani_real.robot.xarm7 import worker

    clock = Clock()
    monkeypatch.setattr(worker.time, "monotonic_ns", clock.monotonic_ns)
    cfg = replace(ArmConfig(), feedback_hz=100.0)
    target = np.asarray(cfg.home_qpos)

    def read():
        clock.advance(0.015)
        return target, np.zeros(7)

    io = SimpleNamespace(publish_status=Mock(), stop=SimpleNamespace(wait=clock.advance))
    driver = SimpleNamespace(read=read, read_status=lambda: SimpleNamespace(error_code=0))
    controller = ArmController(cfg, io, None, driver)
    controller.home(np.empty((0, 7)), target, check_abort=lambda: None)
    assert clock.monotonic() >= 1 + HOME_DWELL_S + 0.03

    with pytest.raises(TimeoutError, match="dwell not confirmed"):
        controller.wait_settled(lambda: None, stage="late query", timeout_s=0.01)


def test_partial_hand_connection_preserves_original_error_and_cleans_arm(monkeypatch):
    from dexmani_real.robot.system import Robot
    from dexmani_real.robot.xarm7 import process
    from dexmani_real.robot.xhand.driver import XHand

    state = SessionState()
    robot = Robot(
        ArmConfig(), HandConfig(), run_signal=state.run_signal,
        stop_signal=state.stop_signal, emergency_signal=state.emergency_signal,
        signal_lock=state.motion_lock,
    )
    arm = Mock()
    monkeypatch.setattr(process, "ArmProcess", lambda *_, **__: arm)
    monkeypatch.setattr(XHand, "connect", Mock(side_effect=ValueError("initial feedback failed")))
    with pytest.raises(ValueError, match="initial feedback failed"):
        robot.connect()
    arm.close.assert_called_once()
    assert robot.hand is None
    assert robot.stop_failure is None
    assert not robot.stop_required

    disconnected_hand = XHand(HandConfig())
    disconnected_hand.command = object()  # 曾完成初始化，断联不能视为从未执行。
    robot.hand = disconnected_hand
    with pytest.raises(RuntimeError, match="disconnected before stop"):
        robot.close()
    assert robot.stop_required
    assert robot.stop_failure is not None


def test_saturated_jog_keeps_accepted_target_and_rechecks_hand(monkeypatch):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.geometry.pose import Pose
    from dexmani_real.runtime import jog

    runtime = RealConfig()
    constraints = SimpleNamespace(rejection=Mock(return_value=None))
    solver = SimpleNamespace(constraints=constraints)
    motion = jog.CartesianJog(solver, runtime.keyboard_teleop, (np.full(3, -1), np.ones(3)))
    motion.command_pose = Pose(p=np.array([1.0, 0.0, 0.0]), q=np.array([1.0, 0.0, 0.0, 0.0]))
    accepted = np.full(7, 0.5)
    motion.command_qpos = accepted
    row = SimpleNamespace(arm={"qpos": [np.zeros(7)], "timestamp_ns": [1]}, hand={"qpos": [np.zeros(12)]})
    monkeypatch.setattr(jog, "read_observation", lambda *_: row)
    monkeypatch.setattr(jog, "check_observation", lambda *_, **__: ([], 1_000_000_000))
    state = SessionState(safety_state=SafetyState.RUNNING)
    robot, keys = SimpleNamespace(send_action=Mock()), SimpleNamespace(snapshot=lambda: frozenset("w"))
    for _ in range(3):
        motion.step(robot, keys, state, runtime)
        command = robot.send_action.call_args.args[0]
        assert not command.arm_hold
        np.testing.assert_array_equal(command.arm_qpos, accepted)
        np.testing.assert_array_equal(motion.command_qpos, accepted)
    row.hand["qpos"][0][0] = 0.2
    constraints.rejection.return_value = "hand collision"
    motion.step(robot, keys, state, runtime)
    assert robot.send_action.call_args.args[0].arm_hold
    assert motion.blocked_until_release
    assert constraints.rejection.call_args.kwargs["collision_hand_qpos"][0] == 0.2


def test_ik_scores_the_selected_absolute_branch():
    from dexmani_real.control.ik import IKSolver
    from dexmani_real.geometry.pose import Pose

    solver = IKSolver.__new__(IKSolver)
    solver.limits = np.tile([-3.1, 3.1], (7, 1))
    solver.joint_ranges = np.full(7, 6.2)
    solver.position_tolerance = solver.rotation_tolerance = 0.01
    raw, current = np.zeros(7), np.zeros(7)
    raw[0], current[0] = -3.1, 3.1
    solver.model = SimpleNamespace(compute_IK_CLIK=lambda *_: (raw, True, None))
    solver.eef_link_id = 0
    solver.constraints = SimpleNamespace(endpoint_rejection=lambda *_, **__: None)
    pose = Pose(p=np.zeros(3), q=np.array([1.0, 0.0, 0.0, 0.0]))
    solver.compute_eef_pose_base = lambda _: pose
    solver.joint_limit_penalty = lambda _: 0
    candidate, _ = solver.try_clik(current, pose, pose, current, current, [])
    assert candidate.displacement_rad == pytest.approx(6.2)
    assert candidate.cost > 2


@pytest.mark.parametrize("outside", [0.003, 0.006])
def test_home_and_online_workspace_share_the_configured_margin(outside):
    from dexmani_real.config.control import EnvironmentConfig
    from dexmani_real.control.constraints import MotionConstraints
    from dexmani_real.control.homing import HomePathRejected, HomePlanner

    environment = EnvironmentConfig()
    bounds = environment.workspace.as_array()
    position = bounds.mean(axis=1)
    position[0] = bounds[0, 1] + outside
    collision = SimpleNamespace(rejection=lambda _: None, eef_position=lambda: position)
    online = MotionConstraints.__new__(MotionConstraints)
    online.workspace, online.margin = bounds, environment.eef_workspace_margin_m
    online.collision_model = collision
    planner = HomePlanner.__new__(HomePlanner)
    planner.environment, planner.workspace_bounds, planner.collision_model = environment, bounds, collision
    planner._table_clearance = lambda: (1, 1, "test")
    rejected = online.endpoint_rejection(np.zeros(7), position=position) is not None
    if rejected:
        with pytest.raises(HomePathRejected, match="workspace"):
            planner._check_state(tuple(np.zeros(7)))
    else:
        planner._check_state(tuple(np.zeros(7)))


def teleop_loop():
    from dexmani_real.config.real import RealConfig
    from dexmani_real.runtime.observation import ObservationNeeds
    from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase

    state = SessionState(safety_state=SafetyState.RUNNING)
    loop = TeleopLoop(
        state, RealConfig(), SimpleNamespace(stop_required=True, stop=Mock(), check=Mock()),
        task_label="test", control_needs=ObservationNeeds(), record_needs=ObservationNeeds(),
        sensors=None, controller=Mock(), home_planner=None, recording_dir=None,
        keyboard=None, audio=Mock(),
    )
    loop.phase, loop.run_id = TeleopPhase.ACTIVE, state.run_id
    loop.recorder = Mock(start_attempted=True, published_path=None)
    return loop


@pytest.fixture
def teleop_preparation(monkeypatch, tmp_path):
    from dexmani_real.runtime.keyboard import Keyboard
    from dexmani_real.runtime.state import RunEndReason
    from dexmani_real.teleop import loop as teleop

    loop = teleop_loop()
    state = loop.runtime_state
    state.safety_state = SafetyState.ARMED
    loop.phase, loop.run_id, loop.recorder = teleop.TeleopPhase.IDLE, None, None
    loop.recording_dir, loop.sensors = tmp_path, SimpleNamespace(channels=None)
    loop.robot.reset_hand_feedback = Mock()
    loop.robot.read_initial_state = Mock()
    loop.robot.prepare_streaming = Mock(side_effect=lambda _: state.stop_signal.clear())
    loop.robot.hand = SimpleNamespace(get_tactile_reset_metadata=lambda: {})
    loop.robot.arm = SimpleNamespace(check_error=Mock())
    loop.robot.service_feedback = Mock()
    loop.keyboard = Keyboard(bindings={
        "s": state.request_stop,
        "q": lambda: state.request_stop(reason=RunEndReason.QUIT),
        "d": loop.request_discard,
    })
    loop.keyboard.active = True
    loop.keyboard.listener = SimpleNamespace(is_alive=Mock(return_value=True))
    recorder = Mock(start_attempted=True, published_path=None, is_recording=True)
    recorder.finish.side_effect = lambda **kwargs: tmp_path if kwargs["save"] else None
    row = SimpleNamespace(arm={"timestamp_ns": [10**30]}, vr={"recv_ts_ns": 10**30}, hand=None)
    monkeypatch.setattr(teleop, "EpisodeRecorder", lambda *_, **__: recorder)
    monkeypatch.setattr(teleop, "snapshot_recording_metadata", lambda *_, **__: {})
    monkeypatch.setattr(teleop, "read_observation", Mock(return_value=row))
    loop.admit_observation = Mock(return_value=((), 0))
    return loop, recorder, row


def test_teleop_preparation_still_admits_motion(teleop_preparation):
    from dexmani_real.teleop.loop import TeleopPhase

    loop, recorder, row = teleop_preparation
    loop.start_episode(loop.runtime_state.run_id)
    assert loop.phase is TeleopPhase.PREPARING
    loop.try_begin(row)
    assert loop.phase is TeleopPhase.ACTIVE
    assert loop.runtime_state.can_move(run_id=loop.run_id)
    loop.controller.reset.assert_called_once_with(row)
    recorder.finish.assert_not_called()


def interrupt_preparation(loop, recorder, row, stage, interrupt):
    from dexmani_real.robot.commands import OperationCancelled

    def cancel(*_, **__):
        interrupt()
        raise OperationCancelled("revoked during preparation")

    if stage == "stop":
        loop.robot.stop.side_effect = interrupt
    elif stage == "initial":
        loop.robot.read_initial_state.side_effect = cancel
    elif stage == "observation":
        loop.admit_observation.side_effect = lambda _: (interrupt() or (), 0)
    elif stage == "recording":
        recorder.start.side_effect = interrupt
    elif stage in ("streaming", "prepared"):
        loop.robot.prepare_streaming.side_effect = cancel if stage == "streaming" else interrupt
    loop.start_episode(loop.runtime_state.run_id)
    if stage == "begin":
        interrupt()
        loop.try_begin(row)
    elif stage == "authorization":
        loop.controller.reset.side_effect = interrupt
        loop.try_begin(row)


@pytest.mark.parametrize("command", ["s", "q", "d"])
@pytest.mark.parametrize("stage", ["stop", "initial", "observation", "recording", "streaming", "prepared", "begin", "authorization"])
def test_teleop_preparing_queued_cancel_keeps_command_semantics(teleop_preparation, command, stage):
    from dexmani_real.teleop.loop import TeleopPhase

    loop, recorder, row = teleop_preparation
    interrupt_preparation(loop, recorder, row, stage, lambda *_, **__: loop.keyboard.handle_key(command, True))
    assert loop.phase is TeleopPhase.IDLE
    assert loop.runtime_state.quit_requested == (command == "q")
    assert not loop.capture_failed
    assert not loop.runtime_state.stop_request
    loop.controller.clear.assert_called_once()
    if recorder.start.called:
        recorder.finish.assert_called_once()
        assert recorder.finish.call_args.kwargs["save"] == (command != "d")
        assert recorder.finish.call_args.kwargs["technical_status"] == "normal"
        assert recorder.finish.call_args.kwargs["reason"] == ("quit" if command == "q" else "operator")
    else:
        recorder.finish.assert_not_called()
    assert all(call.args[0] != "emergency" for call in loop.audio.play.call_args_list)


@pytest.mark.parametrize("command", ["s", "q", "d"])
@pytest.mark.parametrize("stage", ["streaming", "prepared", "begin", "authorization"])
def test_teleop_preparing_cancel_before_event_publication(teleop_preparation, monkeypatch, command, stage):
    from threading import Event, Thread
    from dexmani_real.teleop import loop as teleop

    loop, recorder, row = teleop_preparation
    loop.runtime_state.begin_motion = Mock(wraps=loop.runtime_state.begin_motion)
    revoked, publish = Event(), Event()
    callback = loop.keyboard.bindings[command]
    failures = []

    def delayed_callback():
        callback()
        revoked.set()
        if not publish.wait(5):
            raise TimeoutError("test did not release keyboard callback")

    loop.keyboard.bindings[command] = delayed_callback

    def press():
        try:
            loop.keyboard.handle_key(command, True)
        except BaseException as exc:
            failures.append(exc)

    thread = Thread(target=press)

    def interrupt(*_, **__):
        thread.start()
        assert revoked.wait(5)

    try:
        interrupt_preparation(loop, recorder, row, stage, interrupt)
        assert loop.phase is teleop.TeleopPhase.PREPARING
        assert loop.recorder is recorder
        assert loop.runtime_state.stop_request
        assert not loop.runtime_state.quit_requested  # Q 的早期回调只撤权。
        recorder.finish.assert_not_called()
        loop.runtime_state.begin_motion.assert_not_called()
        teleop.read_observation.reset_mock()
        loop.keyboard.listener.is_alive.reset_mock()
        loop.capture_started_s = 0  # 等待命令时也不能被录制时长判断抢先结束。
        tick = loop.run_control_tick

        def one_tick():
            tick()
            raise StopIteration("one controlled main-loop iteration")

        monkeypatch.setattr(loop, "run_control_tick", one_tick)
        with pytest.raises(StopIteration, match="controlled main-loop"):
            loop.run()
        loop.robot.service_feedback.assert_called_once()
        loop.robot.arm.check_error.assert_called_once()
        loop.keyboard.listener.is_alive.assert_called()
        recorder.check_error.assert_called_once()
        teleop.read_observation.assert_not_called()
        loop.runtime_state.begin_motion.assert_not_called()
        recorder.finish.assert_not_called()
        assert not loop.capture_failed
    finally:
        publish.set()
        thread.join(5)
    assert not thread.is_alive()
    assert not failures
    loop.handle_commands(loop.keyboard.poll(timeout=0), loop.runtime_state.run_id)
    loop.preparation_interrupted()  # 已结束的阶段不能再次结束 episode。
    assert loop.phase is teleop.TeleopPhase.IDLE
    assert loop.runtime_state.quit_requested == (command == "q")
    recorder.finish.assert_called_once()
    assert recorder.finish.call_args.kwargs["save"] == (command != "d")
    assert recorder.finish.call_args.kwargs["technical_status"] == "normal"
    assert not loop.capture_failed


@pytest.mark.parametrize("command", ["s", "q", "d"])
def test_teleop_preparing_stop_during_validity_check(teleop_preparation, monkeypatch, command):
    from dexmani_real.teleop.loop import TeleopPhase

    loop, recorder, _ = teleop_preparation
    loop.start_episode(loop.runtime_state.run_id)
    preparation_valid = loop.preparation_valid

    def revoke_before_validation():
        # 精确落在 helper 消费队列之后、观察到授权失效之前。
        loop.keyboard.handle_key(command, True)
        return preparation_valid()

    monkeypatch.setattr(loop, "preparation_valid", revoke_before_validation)
    assert loop.preparation_interrupted()
    assert loop.phase is TeleopPhase.PREPARING
    assert loop.runtime_state.stop_request
    assert loop.recorder is recorder
    assert not loop.capture_failed
    recorder.finish.assert_not_called()

    loop.handle_commands(loop.keyboard.poll(timeout=0), loop.runtime_state.run_id)
    assert loop.phase is TeleopPhase.IDLE
    assert loop.runtime_state.quit_requested == (command == "q")
    recorder.finish.assert_called_once()
    assert recorder.finish.call_args.kwargs["save"] == (command != "d")
    assert recorder.finish.call_args.kwargs["technical_status"] == "normal"


@pytest.mark.parametrize("failure", ["timeout", "sdk", "recording", "service", "estop", "runtime", "unexplained"])
def test_teleop_preparing_cancel_does_not_hide_failure(teleop_preparation, failure):
    from dexmani_real.data.recorder import RecordingError
    from dexmani_real.robot.commands import OperationCancelled, RobotError

    loop, recorder, _ = teleop_preparation

    def prepare(*_):
        if failure != "unexplained":
            loop.runtime_state.request_stop()
        if failure in ("timeout", "sdk", "recording"):
            raise {"timeout": TimeoutError, "sdk": RobotError, "recording": RecordingError}[failure]("independent failure")
        if failure == "service":
            loop.robot.check.side_effect = RobotError("independent service failure")
        elif failure == "estop":
            loop.runtime_state.request_emergency_stop()
        elif failure == "runtime":
            loop.runtime_state.mark_error()
        raise OperationCancelled("preparation interrupted")

    loop.robot.prepare_streaming.side_effect = prepare
    if failure in ("timeout", "sdk", "recording", "service"):
        with pytest.raises((TimeoutError, RobotError, RecordingError), match="independent") as caught:
            loop.start_episode(loop.runtime_state.run_id)
        loop.record_error("runtime", caught.value)
        loop.close()
    else:
        loop.start_episode(loop.runtime_state.run_id)
    assert loop.capture_failed
    assert recorder.finish.call_args.kwargs["technical_status"] == "failed"


def test_teleop_preparing_observation_failure_is_not_cancellation(teleop_preparation):
    loop, recorder, _ = teleop_preparation

    def invalid_observation(_):
        loop.runtime_state.request_stop()
        return ("independent stale observation",), 0

    loop.admit_observation.side_effect = invalid_observation
    loop.start_episode(loop.runtime_state.run_id)
    assert loop.capture_failed
    recorder.start.assert_not_called()


@pytest.mark.parametrize("reason_name,phase", [("OPERATOR", "compute"), ("QUIT", "send")])
def test_teleop_operator_cancellation_keeps_completed_rows(reason_name, phase):
    import time
    from dexmani_real.control.actions import ActionResult
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.runtime.state import RunEndReason

    loop = teleop_loop()
    recorder = loop.recorder
    result = ActionResult(np.zeros(7), np.zeros(12))

    def cancel(*_, **__):
        loop.runtime_state.request_stop(reason=getattr(RunEndReason, reason_name))
        if phase == "send":
            raise OperationCancelled("cancelled in SDK boundary")
        return result

    loop.controller.compute_target.side_effect = cancel if phase == "compute" else lambda _: result
    loop.robot.send_action = Mock(side_effect=cancel)
    deadline = time.monotonic_ns() + 1_000_000_000
    loop.execute_control_step(SimpleNamespace(vr={"recv_ts_ns": deadline}), deadline, deadline)
    recorder.append.assert_not_called()
    assert recorder.finish.call_args.kwargs["technical_status"] == "normal"
    assert recorder.finish.call_args.kwargs["reason"] == reason_name.lower()
    loop.controller.commit.assert_not_called()
    assert not loop.capture_failed


@pytest.mark.parametrize("failure", ["timeout", "sdk", "estop", "arm_fault", "business"])
def test_teleop_cancellation_does_not_hide_independent_failure(failure):
    import time
    from dexmani_real.control.actions import ActionResult
    from dexmani_real.robot.commands import OperationCancelled, RobotError

    loop = teleop_loop()
    recorder = loop.recorder
    loop.controller.compute_target.return_value = ActionResult(np.zeros(7), np.zeros(12))

    def send(*_, **__):
        loop.runtime_state.request_stop()
        if failure == "timeout":
            raise TimeoutError("real timeout")
        if failure == "sdk":
            raise RobotError("real SDK failure")
        if failure == "business":
            raise ValueError("retargeting failed")
        if failure == "estop":
            loop.runtime_state.request_emergency_stop()
        if failure == "arm_fault":
            loop.robot.check.side_effect = RobotError("owner fault")
        raise OperationCancelled("revoked")

    loop.robot.send_action = Mock(side_effect=send)
    deadline = time.monotonic_ns() + 1_000_000_000
    row = SimpleNamespace(vr={"recv_ts_ns": deadline})
    if failure in ("sdk", "arm_fault", "business"):
        with pytest.raises(ValueError if failure == "business" else RobotError):
            loop.execute_control_step(row, deadline, deadline)
    else:
        loop.execute_control_step(row, deadline, deadline)
    assert loop.capture_failed
    assert recorder.finish.call_args.kwargs["technical_status"] == "failed"
    if failure == "business":
        assert loop.pending_termination_reason == "operator"  # 首次 S 仍保留；技术异常单独记录。
        assert loop.termination_details[0]["exception_type"] == "ValueError"


def test_dexpilot_reset_seeds_the_real_filter():
    from dex_retargeting.optimizer_utils import LPFilter
    from dexmani_real.teleop.retargeting.dexpilot import DexPilotRetargeter

    retargeter = DexPilotRetargeter.__new__(DexPilotRetargeter)
    retargeter.limits = np.tile([0, 1], (12, 1))
    retargeter.optimizer = SimpleNamespace(projected=np.ones(10, dtype=bool))
    retargeter.filter = LPFilter(0.1)
    retargeter.reset(np.full(12, 0.2))
    np.testing.assert_allclose(retargeter.previous_solution, 0.2)
    np.testing.assert_allclose(retargeter.filter.next(np.full(12, 0.8)), 0.26)


def test_replay_admits_measured_start_then_prepares_that_posture(monkeypatch):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.replay import joint_replay

    clock, runtime = Clock(), RealConfig()
    monkeypatch.setattr(joint_replay.time, "monotonic_ns", clock.monotonic_ns)
    state = SessionState(safety_state=SafetyState.ARMED)
    robot = SimpleNamespace(prepare_streaming=Mock(), send_action=Mock())
    replay = joint_replay.JointReplay(state, runtime, None, robot=robot)
    replay.epoch = state.run_id
    arm = np.asarray(runtime.arm.home_qpos)
    measured = np.maximum(np.zeros(12), runtime.hand.qpos_min_rad)
    action = measured.copy()
    action[1] = np.deg2rad(40)
    trajectory = SimpleNamespace(
        arm_qpos=arm[None], hand_qpos=measured[None], action_hand_joint=action[None], fps=20,
    )
    row = SimpleNamespace(arm={"qpos": [arm]}, hand={"qpos": [measured], "timestamp_ns": [clock.now_ns]})

    def feedback():
        clock.advance(0.01)
        row.hand["timestamp_ns"][0] = clock.now_ns
        return row, clock.now_ns + 1_000_000_000

    replay.read_feedback = feedback
    status, _, _ = replay.admit_replay(trajectory)
    assert status == joint_replay.ReplayStatus.COMPLETED
    replay.wait_replay = lambda _: None
    replay.action_mapper = SimpleNamespace(constraints=SimpleNamespace(rejection=lambda *_, **__: None))
    assert replay.prepare_hand(trajectory, row, 0.05, 50_000_000) is None
    for call in robot.send_action.call_args_list:
        np.testing.assert_array_equal(call.args[0].hand_qpos, measured)
    robot.send_action.reset_mock()
    trajectory.hand_qpos = measured[None].copy()
    trajectory.hand_qpos[0, 0] = runtime.hand.qpos_max_rad[0] + 0.1
    with pytest.raises(ValueError, match="hand target violates"):
        replay.prepare_hand(trajectory, row, 0.05, 50_000_000)
    robot.send_action.assert_not_called()


@pytest.mark.parametrize("action", ["space", "enter", "home"])
def test_calibration_discards_old_one_shot_events_but_keeps_stop_and_held_keys(monkeypatch, action):
    from dexmani_real.runtime.home import HomeResult
    from dexmani_real.runtime.keyboard import Keyboard
    from dexmani_real.utils.camera_calibrate import calibrate

    keys = Keyboard()
    keys.held_keys.update(("w", "q"))
    keys.events.extend(("space", "enter", "backspace", "x", "r", "q", "esc"))
    calibration = calibrate.CameraCalibrator.__new__(calibrate.CameraCalibrator)
    calibration.keys, calibration.robot = keys, Mock()
    calibration.cancel_requested = lambda: False
    calibration.capture_sample, calibration.solve_and_save = Mock(), Mock()
    if action == "home":
        calibration.runtime_state = SimpleNamespace(estop_request=False)
        calibration.jog, calibration.home_planner, calibration.rate = Mock(), Mock(), Mock()
        monkeypatch.setattr(calibrate, "home_arm", lambda *_, **__: HomeResult(True))
        calibration.run_home()
    else:
        calibration.handle_sample_events([action, "space", "enter", "backspace"])
        assert calibration.capture_sample.call_count + calibration.solve_and_save.call_count == 1
    assert list(keys.events) == ["q", "esc"]
    assert keys.snapshot() == frozenset(("w", "q"))


@pytest.mark.parametrize("hardware", [False, True])
def test_replay_fault_classification_does_not_label_business_errors_as_policy(monkeypatch, hardware):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.replay import joint_replay
    from dexmani_real.replay.trajectory import Trajectory
    from dexmani_real.robot.commands import RobotError
    from dexmani_real.runtime.state import RunEndReason
    from dexmani_real.teleop.loop import failure_reason

    state = SessionState(safety_state=SafetyState.RUNNING)
    replay = joint_replay.JointReplay(state, RealConfig(), None, robot=SimpleNamespace(stop=Mock()))
    error = RobotError("device error") if hardware else ValueError("invalid replay input")
    replay.admit_replay = Mock(side_effect=error)
    monkeypatch.setattr(joint_replay.ActionMapper, "for_mode", lambda *_, **__: None)
    trajectory = Trajectory("unused", 20, "test", np.zeros((1, 7)), np.zeros((1, 12)), np.zeros((1, 7)), np.zeros((1, 12)))
    result = replay.run(trajectory, hand_start_duration_s=0)
    expected = RunEndReason.HARDWARE_FAULT if hardware else RunEndReason.RUNTIME_FAILURE
    assert result.status == joint_replay.ReplayStatus.FAULT
    assert state.run_ended_reason == expected
    assert failure_reason(error) == expected


@pytest.mark.parametrize("failure", ["cancelled", "lifecycle", "streaming"])
def test_arm_stop_keeps_independent_owner_failures_visible(failure):
    from dexmani_real.robot.commands import OperationCancelled, RobotError
    from dexmani_real.robot.xarm7.process import ArmProcess

    arm = ArmProcess.__new__(ArmProcess)
    error = ValueError("SDK failed during cancellation") if failure == "lifecycle" else OperationCancelled("revoked")
    arm.process = SimpleNamespace(is_alive=lambda: True)
    arm.pipe = SimpleNamespace(poll=lambda _: True, recv=lambda: error)
    arm.operation_unresolved = failure != "streaming"
    arm.io = SimpleNamespace(
        stopped=SimpleNamespace(wait=lambda _: True),
        close_error=SimpleNamespace(value=b""),
        fault=SimpleNamespace(value=b"SDK stream failed" if failure == "streaming" else b""),
    )
    if failure == "cancelled":
        arm.wait_stopped()
    else:
        with pytest.raises(RobotError, match="SDK") as caught:
            arm.wait_stopped()
        if failure == "lifecycle":
            assert caught.value.__cause__ is error
    assert not arm.operation_unresolved

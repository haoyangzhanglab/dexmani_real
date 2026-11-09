"""HOME、软暂停及设备协议的控制与取消回归；不连接 SDK。"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_freshness import make_rollout, make_teleop, observation

from dexmani_real.robot.commands import (
    FeedbackUnavailable,
    HomeTimeout,
    OperationCancelled,
    RobotCommand,
    RobotError,
)
from dexmani_real.runtime.observation import ObservationNeeds
from dexmani_real.teleop.loop import TeleopPhase


def live_keyboard(loop):
    from dexmani_real.runtime.keyboard import Keyboard

    state = loop.runtime_state
    keyboard = Keyboard(
        estop_callback=state.request_emergency_stop,
        bindings={
            "s": state.request_stop,
            "d": state.request_stop,
            "q": state.request_quit,
        },
    )
    keyboard.active = True
    keyboard.listener = Mock()
    loop.keyboard = keyboard
    return keyboard


@pytest.fixture(params=["teleop", "policy"])
def home_session(request, clock, runtime, monkeypatch):
    from test_freshness import hand_sample

    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.robot.system import Robot

    mode = request.param
    if mode == "teleop":
        owner = make_teleop(runtime, clock)
        owner.phase = TeleopPhase.IDLE
    else:
        owner = PolicySession(runtime, None, weights=None, inference_steps=None,
                              output_root=None, device="cpu", seed=0, max_running_s=None,
                              num_episodes=None, record_raw=False, camera_calibration_path=None)
    keyboard = live_keyboard(owner)
    keyboard.allow_start()
    robot = owner.robot = Robot(runtime.arm, runtime.hand, runtime_state=owner.runtime_state)
    robot.connected = True
    robot.arm = Mock(is_connected=True)
    robot.hand = Mock(is_connected=True)
    robot.hand.get_state.side_effect = lambda: hand_sample(clock.now_ns)
    robot.hand._send_position.return_value = 0
    robot.hand_feedback.consume(hand_sample(clock.now_ns))
    robot.feedback_next_ns = clock.now_ns + 1_000_000_000
    monkeypatch.setattr("time.sleep", lambda seconds: setattr(clock, "now_ns", clock.now_ns + round(seconds * 1e9)))
    return mode, owner


def configure_home_timeout(robot, clock, monkeypatch, stage):
    from test_freshness import hand_sample

    from dexmani_real.robot.xhand.driver import COMMUNICATION_CRC_ERROR_CODE

    deadline_ns = clock.now_ns + int(robot.hand_config.home_timeout_s * 1e9)
    robot.feedback_next_ns = clock.now_ns
    if stage in ("initial_before_sample", "initial_after_sample"):
        robot.hand_feedback.latest = None

        def missing_feedback():
            if stage == "initial_after_sample":
                clock.now_ns = deadline_ns

        monkeypatch.setattr(robot.hand.get_state, "side_effect", missing_feedback)
    elif stage == "convergence":
        def unconverged_feedback():
            sample = hand_sample(clock.now_ns)
            sample.qpos[1] += np.deg2rad(10)
            return sample

        monkeypatch.setattr(robot.hand.get_state, "side_effect", unconverged_feedback)
    else:
        assert stage in ("submission", "retry_submission")
        if stage == "retry_submission":
            monkeypatch.setattr(robot.hand._send_position, "side_effect", [COMMUNICATION_CRC_ERROR_CODE, 0])
        check_feedback = robot._check_hand_feedback

        def expire_after_feedback_check():
            check_feedback()
            if stage == "submission" or robot.hand._send_position.called:
                clock.now_ns = deadline_ns

        monkeypatch.setattr(robot, "_check_hand_feedback", expire_after_feedback_check)
    return deadline_ns


@pytest.mark.parametrize("stage,reason,submissions", [
    ("initial_before_sample", "initial feedback timed out", 0),
    ("initial_after_sample", "initial feedback timed out", 0),
    ("convergence", "convergence timed out", 1),
    ("submission", "budget exhausted before submission", 0),
    ("retry_submission", "budget exhausted before submission", 1),
])
def test_home_budget_exhaustion_waits_for_new_request(
    home_session, clock, monkeypatch, caplog, stage, reason, submissions,
):
    mode, owner = home_session
    robot, keyboard, state = owner.robot, owner.keyboard, owner.runtime_state
    caplog.set_level("INFO")
    initial_epoch = state.cancel_epoch
    with monkeypatch.context() as injection:
        deadline_ns = configure_home_timeout(robot, clock, injection, stage)
        if mode == "teleop":
            owner.handle_operator_command("h")
            assert owner.phase is TeleopPhase.IDLE
            assert "home_done" not in [c.args[0] for c in owner.audio.play.call_args_list]
        else:
            assert owner.run_home(keyboard.accept_operation(state))
            assert owner.episode is None
        assert clock.now_ns == deadline_ns
        assert f"本次 HOME 未完成：hand HOME {reason}" in caplog.text
        assert "已回初始位" not in caplog.text
        assert robot.arm.wait_stopped.call_count == 2
        assert robot.hand.stop.call_count == int(submissions > 0)
        assert robot.hand._send_position.call_count == submissions
        assert not robot.hand_stop_required
        assert state.stop_signal.is_set()
        assert not state.quit_requested and not state.estop_request
        assert state.cancel_epoch == initial_epoch
        assert robot.connected
        robot.arm.home.assert_not_called()
        robot.arm.prepare.assert_not_called()
        robot.arm.close.assert_not_called()
        robot.hand.disconnect.assert_not_called()
        assert keyboard.accept_operation(state) is None

    # 反馈恢复及空闲服务不能重发旧请求；下一次运动必须来自新的 H。
    clock.now_ns += 500_000_000
    robot.service_idle()
    assert robot.hand._send_position.call_count == submissions
    robot.arm.home.assert_not_called()
    assert state.stop_signal.is_set()
    keyboard.allow_start()
    keyboard.handle_key("h", True)
    if mode == "teleop":
        owner.handle_commands(keyboard.poll(state, timeout=0)[0])
        assert owner.phase is TeleopPhase.IDLE
        assert [c.args[0] for c in owner.audio.play.call_args_list].count("home_done") == 1
    else:
        assert owner.poll_operator() is None
        assert owner.episode is None
    assert robot.hand._send_position.call_count == submissions + 1
    robot.arm.home.assert_called_once()
    assert "已回初始位" in caplog.text
    assert state.cancel_epoch == initial_epoch


@pytest.mark.parametrize("key", ["s", "q", "esc"])
def test_home_timeout_preserves_operator_priority(home_session, clock, monkeypatch, caplog, key):
    mode, owner = home_session
    robot, keyboard, state = owner.robot, owner.keyboard, owner.runtime_state
    configure_home_timeout(robot, clock, monkeypatch, "initial_before_sample")
    initial_epoch = state.cancel_epoch
    caplog.set_level("INFO")

    def stopped():
        if robot.arm.wait_stopped.call_count == 2:
            # 已抛出真实业务超时，在 finally 收尾时同时收到操作者命令。
            keyboard.handle_key(key, True)

    robot.arm.wait_stopped.side_effect = stopped
    if mode == "teleop":
        owner.handle_operator_command("h")
        assert owner.phase is TeleopPhase.IDLE
        assert "home_done" not in [c.args[0] for c in owner.audio.play.call_args_list]
    else:
        assert owner.run_home(keyboard.accept_operation(state))
        assert owner.episode is None
    assert "initial feedback timed out" in caplog.text
    assert "本次 HOME 未完成" not in caplog.text
    assert "已回初始位" not in caplog.text
    assert robot.arm.wait_stopped.call_count == 2
    robot.arm.home.assert_not_called()
    robot.hand._send_position.assert_not_called()
    assert state.stop_signal.is_set()
    assert state.quit_requested == (key == "q")
    assert state.estop_request == (key == "esc")
    assert state.cancel_epoch == initial_epoch + int(key in ("s", "q"))
    keyboard.allow_start()
    assert (keyboard.accept_operation(state) is not None) == (key == "s")


@pytest.mark.parametrize("failure", ["hand_stop", "arm_stop", "fault"])
def test_home_timeout_does_not_hide_stop_failure_or_fault(home_session, clock, monkeypatch, caplog, failure):
    mode, owner = home_session
    robot = owner.robot
    configure_home_timeout(robot, clock, monkeypatch, "convergence")
    error = RobotError("device fault") if failure == "fault" else TimeoutError("stop unconfirmed")
    if failure == "hand_stop":
        robot.hand.stop.side_effect = error
    elif failure == "arm_stop":
        robot.arm.wait_stopped.side_effect = [None, error]
    else:
        def stopped():
            if robot.arm.wait_stopped.call_count == 2:
                robot.arm.check_error.side_effect = error

        robot.arm.wait_stopped.side_effect = stopped
    with pytest.raises(type(error)) as caught:
        if mode == "teleop":
            owner.handle_operator_command("h")
        else:
            owner.run_home(owner.keyboard.accept_operation(owner.runtime_state))
    assert caught.value is error
    if failure != "fault":
        assert isinstance(caught.value.__context__, HomeTimeout)
    assert robot.arm.wait_stopped.call_count == 2
    robot.arm.home.assert_not_called()
    assert owner.runtime_state.stop_signal.is_set()
    assert "本次 HOME 未完成" not in caplog.text


@pytest.mark.parametrize("home_session", ["teleop"], indirect=True)
def test_jog_home_timeout_leaves_original_exit_boundary(home_session, clock, runtime, monkeypatch):
    from dexmani_real.teleop.keyboard import KeyboardTeleop

    _, owner = home_session
    jog = KeyboardTeleop(runtime)
    jog.robot, jog.runtime_state = owner.robot, owner.runtime_state
    jog.prepare_jog = Mock()
    configure_home_timeout(jog.robot, clock, monkeypatch, "initial_before_sample")
    with pytest.raises(HomeTimeout) as caught:
        jog.run_home(jog.runtime_state.cancel_epoch)
    assert isinstance(caught.value, OperationCancelled)
    assert not isinstance(caught.value, TimeoutError)
    jog.prepare_jog.assert_not_called()
    jog.robot.arm.home.assert_not_called()
    assert jog.runtime_state.stop_signal.is_set()


@pytest.mark.parametrize("key", [None, "s", "q", "esc"])
def test_home_feedback_failure_preserves_session_and_operator_priority(home_session, clock, key, caplog):
    mode, owner = home_session
    robot, keyboard = owner.robot, owner.keyboard
    original_send = robot.send_hand_home

    def expire_before_send(*args, **kwargs):
        clock.now_ns += 210_000_000
        try:
            return original_send(*args, **kwargs)
        except FeedbackUnavailable:
            if key:
                keyboard.handle_key(key, True)
            raise

    robot.send_hand_home = expire_before_send
    initial_epoch = owner.runtime_state.cancel_epoch
    if mode == "teleop":
        owner.handle_operator_command("h")
        assert owner.phase is TeleopPhase.IDLE
        assert "home_done" not in [c.args[0] for c in owner.audio.play.call_args_list]
    else:
        epoch = keyboard.accept_operation(owner.runtime_state)
        assert owner.run_home(epoch)
        assert owner.episode is None
    assert robot.arm.wait_stopped.call_count == 2  # 开始停止 + 真实 HOME finally；上层不重复停止。
    robot.arm.home.assert_not_called()
    robot.hand._send_position.assert_not_called()
    assert owner.runtime_state.stop_signal.is_set()
    assert owner.runtime_state.quit_requested == (key == "q")
    assert owner.runtime_state.estop_request == (key == "esc")
    assert owner.runtime_state.cancel_epoch == initial_epoch + int(key in ("s", "q"))
    assert ("本次 HOME 未完成" in caplog.text) == (key is None)
    keyboard.allow_start()
    assert (keyboard.accept_operation(owner.runtime_state) is not None) == (key not in ("q", "esc"))


@pytest.mark.parametrize("failure", ["stop", "fault", "timeout", "program", "cancel"])
def test_home_real_cleanup_does_not_hide_failures(home_session, failure):
    mode, owner = home_session
    robot, keyboard = owner.robot, owner.keyboard
    if failure == "stop":
        robot.arm.wait_stopped.side_effect = [None, RobotError("stop unconfirmed")]
    error_type = {"stop": RobotError, "fault": RobotError, "timeout": TimeoutError,
                  "program": ValueError, "cancel": OperationCancelled}[failure]

    def fail_submission(*args, **kwargs):
        if failure == "fault":
            robot.arm.check_error.side_effect = RobotError("device fault")
        raise FeedbackUnavailable("feedback stale") if failure in ("stop", "fault") else error_type("injected")

    robot.hand._send_position.side_effect = fail_submission
    with pytest.raises(error_type):
        if mode == "teleop":
            owner.handle_operator_command("h")
        else:
            owner.run_home(keyboard.accept_operation(owner.runtime_state))
    robot.arm.home.assert_not_called()
    robot.hand._send_position.assert_called_once()


def test_arm_home_feedback_failure_drains_reply_before_next_operation(home_session, clock, caplog):
    from test_freshness import hand_sample

    from dexmani_real.robot.xarm7.process import ArmProcess
    from dexmani_real.robot.xarm7.protocol import HomeRequest

    mode, owner = home_session
    robot, keyboard = owner.robot, owner.keyboard
    arm = robot.arm = ArmProcess(robot.arm_config, runtime_state=owner.runtime_state)
    arm.process = Mock()
    arm.io.ready.set()
    arm.io.stopped.set()
    robot.read_hand = lambda: robot.hand_feedback.consume(hand_sample(clock.now_ns))
    robot.feedback_next_ns = clock.now_ns
    service_feedback = robot.service_feedback
    submitted = False
    replied = False

    def service_owner_and_feedback():
        nonlocal submitted, replied
        if arm.child_pipe.poll():
            assert isinstance(arm.child_pipe.recv(), HomeRequest)
            assert not submitted
            submitted = True
            arm.io.stopped.clear()
            # 手部 HOME 已完成；机械臂运动期间没有收到新手部反馈。
            clock.now_ns += 210_000_000
            robot.feedback_next_ns = clock.now_ns + 1_000_000_000
        elif submitted and owner.runtime_state.stop_signal.is_set() and not replied:
            arm.io.stopped.set()
            arm.child_pipe.send(OperationCancelled("owner stopped HOME"))
            replied = True
        return service_feedback()

    robot.service_feedback = service_owner_and_feedback
    try:
        if mode == "teleop":
            owner.handle_operator_command("h")
            assert owner.phase is TeleopPhase.IDLE
            assert "home_done" not in [c.args[0] for c in owner.audio.play.call_args_list]
        else:
            assert owner.run_home(keyboard.accept_operation(owner.runtime_state))
            assert owner.episode is None
        assert submitted and replied
        assert arm.io.stopped.is_set() and owner.runtime_state.stop_signal.is_set()
        assert not arm.io.closing.is_set()
        assert not arm.pipe.poll() and not arm.child_pipe.poll()
        assert "本次 HOME 未完成" in caplog.text
        keyboard.allow_start()
        assert keyboard.accept_operation(owner.runtime_state) is not None
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def fresh_vr(loop, clock, position=None):
    ring = loop.sensors.channels.vr_ring
    payload = ring.read()[0]
    payload["recv_ts_ns"] = clock.now_ns
    payload["wrist_pos"] = np.zeros(3) if position is None else position
    ring.write(**{name: payload[name] for name in ring.field_names})


def test_recovery_reanchors_from_post_prepare_feedback(clock, runtime):
    from dexmani_real.geometry.pose import Pose
    from dexmani_real.teleop.vr import VRController

    loop = make_teleop(runtime, clock)
    loop.run_control_tick()
    old = loop.last_command
    clock.now_ns += 500_000_000
    loop.run_control_tick()
    assert loop.phase is TeleopPhase.PAUSED
    assert loop.pending_termination_reason is None
    assert loop.last_command is loop.target_deadline_ns is None
    for _ in range(3):
        clock.now_ns += 62_500_000
        loop.run_control_tick()
    loop.robot.prepare_streaming.assert_not_called()
    assert loop.robot.send_action.call_count == 1

    # 实际 VRController.reset 重建 Arm/VR/Hand 参考，compute 不运行 IK。
    mapper = Mock()
    mapper.solver.measured_reference.side_effect = lambda q: q.copy()
    mapper.solver.compute_eef_pose_base.side_effect = lambda q: Pose(
        q[:3].copy(), np.array([1.0, 0, 0, 0])
    )
    mapper.solver.constraints.workspace = np.tile([-10.0, 10.0], (3, 1))
    hand = Mock()
    controller = VRController(mapper, hand, runtime.teleop, np.eye(3))
    controller.compute_target = Mock(
        return_value=SimpleNamespace(
            arm_qpos=np.asarray(runtime.arm.home_qpos) + 0.01,
            hand_qpos=observation(clock.now_ns).hand["qpos"][0],
            rejection_reason=None,
            eef_pose=None,
        )
    )
    controller.commit = Mock()
    loop.controller = controller

    def prepare(epoch, *, poll_operator):
        assert not poll_operator()
        loop.runtime_state.stop_signal.clear()  # fake 协议；生产代码只有 ArmProcess.resume 清除。

    loop.robot.prepare_streaming.side_effect = prepare
    clock.now_ns += 62_500_000
    fresh_vr(loop, clock, np.array([1.0, 2.0, 3.0]))
    loop.run_control_tick()
    assert loop.resume_after_ns == clock.now_ns
    assert loop.robot.send_action.call_count == 1
    prepared_ns = clock.now_ns
    clock.now_ns += 62_500_000
    # 准备前反馈仍新鲜，但禁止用它重新锚定。
    loop.robot.read_state.return_value = observation(prepared_ns - 1)
    loop.robot.read_state.side_effect = None
    loop.run_control_tick()
    assert loop.phase is TeleopPhase.PAUSED
    controller.compute_target.assert_not_called()
    clock.now_ns += 62_500_000
    current = observation(clock.now_ns)
    current.arm["qpos"][0, 0] += 0.02
    current = replace(current, hand=current.hand.copy())
    current.hand["qpos"][0, 0] += 0.03
    loop.robot.read_state.return_value = current
    fresh_vr(loop, clock, np.array([4.0, 5.0, 6.0]))
    loop.run_control_tick()
    assert loop.phase is TeleopPhase.ACTIVE
    assert loop.pending_termination_reason is None
    np.testing.assert_array_equal(controller.anchor[0], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(controller.anchor[1], current.arm["qpos"][0, :3])
    np.testing.assert_array_equal(
        hand.reset.call_args.kwargs["initial_qpos"], current.hand["qpos"][0]
    )
    assert loop.last_command is not old
    assert loop.robot.send_action.call_count == 2


@pytest.mark.parametrize("key", ["s", "d", "q", "esc"])
@pytest.mark.parametrize("during_prepare", [False, True])
def test_operator_preempts_paused_recovery(clock, runtime, key, during_prepare):
    loop = make_teleop(runtime, clock)
    keyboard = live_keyboard(loop)
    loop.pause("vr_stale")

    def prepare(epoch, *, poll_operator):
        keyboard.handle_key(key, True)
        poll_operator()
        raise OperationCancelled("cancelled during prepare")

    if during_prepare:
        loop.robot.prepare_streaming.side_effect = prepare
        loop.run_control_tick()
    else:
        keyboard.handle_key(key, True)
        assert not loop.service_episode(0)
    assert loop.pending_termination_reason == (
        "estop" if key == "esc" else "quit" if key == "q" else "operator"
    )
    assert loop.runtime_state.stop_signal.is_set()
    loop.robot.send_action.assert_not_called()
    assert loop.runtime_state.quit_requested == (key == "q")
    assert loop.runtime_state.estop_request == (key == "esc")
    assert loop.discard_requested == (key == "d")


@pytest.mark.parametrize("reason", ["command_expired", "arm_feedback_expired", None])
def test_internal_stop_requires_explicit_reason(clock, runtime, reason):
    loop = make_teleop(runtime, clock)
    live_keyboard(loop)
    loop.robot.arm.soft_stop_reason = reason
    loop.runtime_state.stop_signal.set()
    loop.service_episode(0)
    if reason is None:
        assert loop.pending_termination_reason == "cancelled"
        assert loop.phase is TeleopPhase.ACTIVE
    else:
        assert loop.pending_termination_reason is None
        assert loop.phase is TeleopPhase.PAUSED
        loop.robot.stop.assert_called_once()
    loop.robot.send_action.assert_not_called()


@pytest.mark.parametrize("stage", ["check", "stop"])
def test_hard_failure_never_admits_recovery(clock, runtime, stage):
    loop = make_teleop(runtime, clock)
    getattr(loop.robot, stage).side_effect = RobotError("injected hard fault")
    with pytest.raises(RobotError):
        loop.pause("vr_stale")
    assert loop.phase is not TeleopPhase.PAUSED
    loop.robot.prepare_streaming.assert_not_called()


@pytest.mark.parametrize("cancel", [None, "s", "esc"])
def test_arm_resume_discards_mailbox_and_honors_cancel(runtime, cancel):
    import time

    from dexmani_real.robot.xarm7.process import ArmProcess
    from dexmani_real.runtime.state import SessionState

    state = SessionState()
    arm = ArmProcess(runtime.arm, runtime_state=state)
    arm.process = Mock()
    arm.io.ready.set()
    arm.io.stopped.set()
    arm.io.publish_goal(
        RobotCommand(np.asarray(runtime.arm.home_qpos)),
        expires_ns=time.monotonic_ns() + 1_000_000_000,
    )
    epoch = state.cancel_epoch
    if cancel == "s":
        state.request_stop()
    if cancel == "esc":
        state.request_emergency_stop()
    try:
        if cancel:
            with pytest.raises(OperationCancelled):
                arm.resume(epoch, deadline_ns=time.monotonic_ns() + 1_000_000_000)
            assert state.stop_signal.is_set()
        else:
            arm.resume(epoch, deadline_ns=time.monotonic_ns() + 1_000_000_000)
            assert not state.stop_signal.is_set()
        assert arm.io.take_goal() is None
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_policy_feedback_loss_ends_episode_retains_future(clock, runtime):
    loop = make_rollout(runtime, ObservationNeeds(), clock)
    loop.prepare = Mock()
    loop.robot.service_feedback.side_effect = FeedbackUnavailable("hand expired")
    pending = loop.model.future = object()
    result = loop.run(poll_operator=lambda: None)
    assert result.termination_reason == "cancelled"
    assert loop.model.future is pending
    assert not loop.runtime_state.quit_requested
    loop.robot.stop.assert_called_once()
    loop.robot.send_action.assert_not_called()


def test_partial_submission_pauses_before_recovery(clock, runtime):
    from dexmani_real.robot.system import Robot

    loop = make_teleop(runtime, clock)
    robot = Robot(runtime.arm, runtime.hand, runtime_state=loop.runtime_state)
    robot.connected = True
    robot.arm = Mock(is_connected=True)
    robot.hand = Mock(is_connected=True)
    robot.hand_stop_required = True
    from test_freshness import hand_sample

    robot.hand_feedback.consume(hand_sample(clock.now_ns))
    robot.read_state = lambda: observation(clock.now_ns)

    def expire(*a, **k):
        clock.now_ns += 200_000_000

    robot.arm.submit.side_effect = expire
    loop.robot = robot
    loop.run_control_tick()
    assert loop.phase is TeleopPhase.PAUSED
    robot.arm.submit.assert_called_once()
    robot.hand._send_action.assert_not_called()
    robot.hand.stop.assert_called_once()
    robot.arm.wait_stopped.assert_called_once()
    assert loop.last_command is None
    loop.controller.commit.assert_not_called()


def test_new_arm_preparation_invalidates_old_stop_reason(clock, runtime, monkeypatch):
    import multiprocessing as mp

    from dexmani_real.robot.xarm7.protocol import ArmChannels, PrepareRequest
    from dexmani_real.robot.xarm7.worker import ArmController
    from dexmani_real.runtime.state import SessionState

    state = SessionState()
    io = ArmChannels(
        mp.get_context("spawn"),
        stop_signal=state.stop_signal,
        emergency_signal=state.emergency_signal,
    )
    io.stop_reason.value = b"command_expired"
    io.stopped.set()
    pipe = Mock()
    pipe.poll.return_value = False
    owner = ArmController(runtime.arm, io, pipe, Mock())
    owner.wait_settled = Mock()
    monkeypatch.setattr("dexmani_real.robot.xarm7.worker.ArmTrajectory", Mock())
    owner.operation(PrepareRequest(clock.now_ns + 1_000_000_000))
    assert io.stop_reason.value == b""
    assert not io.stopped.is_set()
    assert not io.fault.value
    pipe.send.assert_called_once_with(None)


def test_policy_finish_consumes_pending_prediction_before_next_episode(clock, runtime):
    from concurrent.futures import Future

    from dexmani_real.deployment.inference import PolicyWorker
    from dexmani_real.deployment.session import PolicySession

    rollout = make_rollout(runtime, ObservationNeeds(), clock)
    future = Future()
    model = PolicyWorker.__new__(PolicyWorker)
    model.future = future
    session = PolicySession.__new__(PolicySession)
    session.episode, session.model, session.robot = rollout, model, rollout.robot
    session.runtime_state = rollout.runtime_state
    session.poll_operator = Mock(return_value=None)
    session.save_results = Mock()
    observed = []

    def feedback():
        observed.append(session.episode is rollout and model.future is future)
        if len(observed) == 2:
            future.set_result(np.ones((2, 19)))

    rollout.robot.service_feedback.side_effect = feedback
    session.finish_episode()
    assert observed == [True, True]
    assert session.episode is None
    assert model.future is None


def test_unknown_preparation_cancel_does_not_reuse_old_soft_reason(clock, runtime):
    loop = make_teleop(runtime, clock)
    loop.pause("command_expired")
    loop.robot.arm.soft_stop_reason = "command_expired"
    loop.robot.prepare_streaming.side_effect = OperationCancelled(
        "unknown stop before owner accepted prepare"
    )
    loop.run_control_tick()
    assert loop.pending_termination_reason == "cancelled"
    assert loop.termination_details[-1]["reason"] == "resume_preparation_cancelled"
    loop.robot.send_action.assert_not_called()

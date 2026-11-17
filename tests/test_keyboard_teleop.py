"""HOME 返回 Jog 的无硬件回归；保留真实准备流程与取消互锁。"""

import logging
import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.system import Robot
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE, GOAL_DTYPE, ArmChannels
from dexmani_real.runtime.jog import CartesianJog
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop import keyboard


HOME_RESULTS = (
    pytest.param(
        keyboard.HomeConfigurationError("environment not confirmed"), id="configuration",
    ),
    pytest.param(keyboard.HomePathRejected("arm path intersects table"), id="path"),
    pytest.param(None, id="success"),
)


@pytest.mark.parametrize("outside_m,accepted", [(0.001, True), (0.020, False)])
def test_jog_from_outside_only_accepts_legal_ik_endpoint(outside_m, accepted):
    from dexmani_real.control.actions import ActionMapper
    from dexmani_real.geometry.kinematics import ArmFK

    runtime = RealConfig()
    measured = np.asarray(runtime.arm.home_qpos)
    position = ArmFK().forward(measured)[:3, 3]
    environment = replace(
        runtime.environment,
        workspace=replace(runtime.environment.workspace, x_min=position[0] + outside_m),
        eef_workspace_tolerance_m=0.,
    )
    mapper = ActionMapper.for_mode(
        "eef", arm=runtime.arm, hand=runtime.hand, motion=runtime.motion, environment=environment,
    )
    bounds = environment.workspace.as_array()
    jog = CartesianJog(mapper.solver, runtime.keyboard_teleop, (bounds[:, 0], bounds[:, 1]))
    result = jog.propose({"w"}, measured, control_ns=1_000_000_000)
    assert (result.qpos is not None) == accepted
    if accepted:
        endpoint = mapper.solver.compute_eef_pose_base(result.qpos).p
        assert np.all(endpoint >= bounds[:, 0]) and np.all(endpoint <= bounds[:, 1])
        assert np.all(np.abs(result.qpos - measured) <= mapper.solver.speed / runtime.keyboard_teleop.control_hz)
        assert np.all(np.abs(result.qpos - measured) <= mapper.solver.lead)
    else:
        assert result.reason == "final_eef_workspace"


def test_home_parameters_are_checked_only_for_home():
    from dexmani_real.config.control import StaticCollisionBox
    from dexmani_real.config.xarm7 import HomingConfig
    from dexmani_real.runtime.preflight import validate_robot_config

    runtime = RealConfig()
    runtime = replace(
        runtime,
        environment=replace(runtime.environment, home_model_confirmed=None,
                            static_boxes=(StaticCollisionBox(size_xyz_m=(-1, 1, 1)),)),
        arm=replace(runtime.arm, homing=HomingConfig(max_acceleration_deg_per_s2=float("nan"))),
        hand=replace(runtime.hand, home_tor_max=None),
    )
    validate_robot_config(runtime)
    runtime.environment.validate()
    with pytest.raises(ValueError, match="home_model_confirmed"):
        runtime.environment.validate_home()
    with pytest.raises(ValueError, match="positive"):
        replace(runtime.environment, home_model_confirmed=True).validate_home()
    with pytest.raises(ValueError):
        runtime.arm.validate_home()
    with pytest.raises(ValueError, match="HOME output"):
        runtime.hand.validate_home()
    assert RealConfig().environment.home_model_confirmed is False


def test_home_acceleration_uses_positive_limit_without_sdk_planner_minimum():
    from dexmani_real.config.xarm7 import HomingConfig

    runtime = RealConfig()
    homing = HomingConfig(
        max_jerk_deg_per_s3=1000., collision_margin_m=.01, tracking_error_rad=.1,
        hand_hold_tolerance_rad=.1, max_lateness_s=.005, max_acceleration_deg_per_s2=.001,
    )
    arm = replace(runtime.arm, homing=homing)
    arm.validate()
    arm.validate_home()
    for acceleration in (0., -1., float("nan"), arm.max_joint_acceleration_deg_per_s2 + 1):
        with pytest.raises(ValueError):
            replace(arm, homing=replace(homing, max_acceleration_deg_per_s2=acceleration)).validate_home()
    with pytest.raises(ValueError, match="SDK command range"):
        replace(arm, max_joint_acceleration_deg_per_s2=.001).validate()


@pytest.fixture
def teleop(monkeypatch, caplog):
    # CLI tests may already have configured the package logger for console output.
    monkeypatch.setattr(logging.getLogger("dexmani_real"), "propagate", True)
    caplog.set_level(logging.INFO, logger=keyboard.__name__)
    clock = SimpleNamespace(now_ns=10_000_000_000)

    def sleep(seconds):
        clock.now_ns += round(seconds * 1e9)

    monkeypatch.setattr(keyboard.time, "monotonic_ns", lambda: clock.now_ns)
    monkeypatch.setattr(keyboard.time, "monotonic", lambda: clock.now_ns / 1e9)
    monkeypatch.setattr(keyboard.time, "sleep", sleep)
    runtime = RealConfig()
    runtime = replace(runtime, hand=replace(runtime.hand, enabled=False))
    session = keyboard.KeyboardTeleop(runtime)
    state = session.runtime_state = SessionState(
        stop_signal=threading.Event(), emergency_signal=threading.Event(),
    )
    session.jog_config = runtime.keyboard_teleop
    session.jog = CartesianJog(None, session.jog_config, (None, None))
    session.jog.ik_reference = object()
    session.period_ns = keyboard.sampling_clock_ns(session.jog_config.control_hz)
    session.tick_due_ns = clock.now_ns - session.period_ns
    session.keys = Keyboard(
        bindings={"q": state.request_quit}, estop_callback=state.request_emergency_stop,
    )
    session.keys.active = True
    session.keys.listener = Mock(is_alive=Mock(return_value=True))

    # 仅替换共享存储、子进程和管道；resume/prepare/邮箱提交均执行生产方法。
    channels = ArmChannels.__new__(ArmChannels)
    channels.stop, channels.emergency = state.stop_signal, state.emergency_signal
    channels.closing, channels.stopped, channels.ready = (threading.Event() for _ in range(3))
    channels.stopped.set()
    channels.ready.set()
    channels.fault = SimpleNamespace(value=b"")
    channels.goal_lock, channels.status_lock = threading.Lock(), threading.Lock()
    channels.goal = bytearray(GOAL_DTYPE.itemsize)
    channels.status = bytearray(ARM_STATE_DTYPE.itemsize)
    frame = np.zeros(1, dtype=ARM_STATE_DTYPE)
    frame["timestamp_ns"] = clock.now_ns
    channels.publish_status(frame[0])
    channels.clear_goal = Mock(wraps=channels.clear_goal)
    channels.publish_goal = Mock(wraps=channels.publish_goal)
    arm = ArmProcess.__new__(ArmProcess)
    arm.runtime_state, arm.cfg, arm.io = state, runtime.arm, channels
    arm.process = Mock(pid=None, is_alive=Mock(return_value=True))
    arm.pipe = Mock(poll=Mock(return_value=True), recv=Mock(return_value=None))
    session.robot = Robot(runtime.arm, runtime.hand, runtime_state=state)
    session.robot.arm, session.robot.connected = arm, True
    home = Mock(side_effect=keyboard.HomeConfigurationError("HOME requires confirmed environment"))
    monkeypatch.setattr(keyboard, "home_robot", home)
    return session, clock, home


@pytest.mark.parametrize("home_error", HOME_RESULTS)
def test_home_prepares_jog_once_and_next_tick_runs(teleop, caplog, home_error):
    session, clock, home = teleop

    def finish_home(*args, **kwargs):
        assert session.runtime_state.stop_signal.is_set()
        clock.now_ns += 2_000_000_000
        if home_error is not None:
            raise home_error

    home.side_effect = finish_home
    session.run_home(session.runtime_state.cancel_epoch)

    arm = session.robot.arm
    assert session.jog.ik_reference is None
    assert not session.runtime_state.stop_signal.is_set()
    arm.io.clear_goal.assert_called_once()
    arm.pipe.send.assert_called_once()
    arm.io.publish_goal.assert_called_once()
    command = arm.io.publish_goal.call_args.args[0]
    assert command.arm_hold and command.hand_qpos is None
    assert arm.io.publish_goal.call_args.kwargs["expires_ns"] == (
        clock.now_ns + int(session.jog_config.target_max_age_s * 1e9)
    )
    assert session.tick_due_ns == clock.now_ns + session.period_ns
    assert caplog.messages.count("Keyboard jog ready") == 1
    assert caplog.messages.count("HOME completed") == int(home_error is None)
    if home_error is not None:
        assert str(home_error) in caplog.text

    # 一个新鲜反馈的控制周期执行真实 step（松键 hold），下一周期通过 Q 结束。
    frame = np.zeros(1, dtype=ARM_STATE_DTYPE)
    frame["timestamp_ns"] = clock.now_ns
    arm.io.publish_status(frame[0])
    poll = session.keys.poll

    def poll_then_quit(*args, **kwargs):
        if session.keys.poll.call_count == 2:
            session.keys.handle_key("q", True)
        return poll(*args, **kwargs)

    session.keys.poll = Mock(side_effect=poll_then_quit)
    session.jog.step = Mock(wraps=session.jog.step)
    assert session.run_control_loop()
    session.jog.step.assert_called_once()
    assert arm.io.publish_goal.call_count == 2


def cancel_motion(session, action):
    if action == "epoch":
        session.runtime_state.request_stop()
    else:
        session.keys.handle_key(action, True)


@pytest.mark.parametrize("action", ["q", "esc", "epoch"])
@pytest.mark.parametrize("home_error", HOME_RESULTS)
def test_cancel_before_recovery_does_not_prepare(teleop, caplog, action, home_error):
    session, _, home = teleop
    original_tick = session.tick_due_ns
    reference = session.jog.ik_reference

    def finish_home(*args, **kwargs):
        cancel_motion(session, action)
        if home_error is not None:
            raise home_error

    home.side_effect = finish_home
    session.run_home(session.runtime_state.cancel_epoch)
    session.robot.arm.io.clear_goal.assert_not_called()
    session.robot.arm.pipe.send.assert_not_called()
    session.robot.arm.io.publish_goal.assert_not_called()
    assert session.jog.ik_reference is reference
    assert session.tick_due_ns == original_tick
    assert session.runtime_state.stop_signal.is_set()
    assert "Keyboard jog ready" not in caplog.messages
    assert caplog.messages.count("HOME completed") == int(home_error is None)


@pytest.mark.parametrize("action", ["q", "esc", "epoch"])
@pytest.mark.parametrize("stage", ["resume", "reply", "prepared"])
def test_cancel_during_preparation_blocks_hold(teleop, caplog, action, stage):
    session, _, _ = teleop
    arm = session.robot.arm
    original_tick = session.tick_due_ns
    if stage == "resume":
        clear_goal = arm.io.clear_goal

        def clear_then_cancel(**kwargs):
            clear_goal(**kwargs)
            cancel_motion(session, action)

        arm.io.clear_goal = Mock(side_effect=clear_then_cancel)
    elif stage == "reply":
        arm.pipe.recv.side_effect = lambda: cancel_motion(session, action)
    else:
        prepare = session.robot.prepare_streaming

        def prepare_then_cancel(*args):
            prepare(*args)
            cancel_motion(session, action)

        session.robot.prepare_streaming = Mock(side_effect=prepare_then_cancel)

    with pytest.raises(OperationCancelled):
        session.run_home(session.runtime_state.cancel_epoch)
    assert session.jog.ik_reference is None
    assert arm.io.clear_goal.call_count == 1
    assert arm.pipe.send.call_count == int(stage != "resume")
    arm.io.publish_goal.assert_not_called()
    assert session.runtime_state.stop_signal.is_set()
    assert session.tick_due_ns == original_tick
    assert "Keyboard jog ready" not in caplog.messages
    assert "HOME completed" not in caplog.messages


@pytest.mark.parametrize(
    "stage", ["configuration_check", "path_check", "home", "home_cancel", "prepare"],
)
def test_errors_propagate_without_retry_or_success_log(teleop, caplog, stage):
    session, _, home = teleop
    failure = (
        OperationCancelled("HOME cancelled")
        if stage == "home_cancel" else RobotError("device fault")
    )
    original_tick = session.tick_due_ns
    if stage in {"configuration_check", "path_check"}:
        if stage == "path_check":
            home.side_effect = keyboard.HomePathRejected("path intersects table")
        session.robot.check = Mock(side_effect=failure)
    elif stage in {"home", "home_cancel"}:
        home.side_effect = failure
    else:
        session.robot.arm.pipe.recv.side_effect = failure

    with pytest.raises(type(failure)) as caught:
        session.run_home(session.runtime_state.cancel_epoch)
    assert caught.value is failure
    home.assert_called_once()
    assert session.robot.arm.pipe.send.call_count == int(stage == "prepare")
    session.robot.arm.io.publish_goal.assert_not_called()
    assert session.tick_due_ns == original_tick
    assert "Keyboard jog ready" not in caplog.messages
    assert "HOME completed" not in caplog.messages


@pytest.mark.parametrize('condition', ['valid', 'cancel', 'stale', 'expired'])
def test_hand_home_submission_boundary(condition):
    import time
    from dexmani_real.robot.commands import FeedbackUnavailable, HomeTimeout
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
    runtime = RealConfig()
    state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
    robot = Robot(runtime.arm, runtime.hand, runtime_state=state)
    robot.connected = True
    robot.hand_config = replace(runtime.hand, home_tor_max=100)
    robot.hand = Mock(is_connected=True)
    robot.hand._send_position.return_value = 0
    frame = np.zeros(1, dtype=HAND_STATE_DTYPE)
    frame['timestamp_ns'] = 1 if condition == 'stale' else time.monotonic_ns()
    robot.hand_feedback.latest = frame
    def check():
        if condition == 'cancel':
            raise OperationCancelled('cancelled')
        robot.check()
    deadline = 1 if condition == 'expired' else time.monotonic_ns()+1_000_000_000
    target = (np.asarray(runtime.hand.qpos_min_rad) + runtime.hand.qpos_max_rad) / 2
    if condition == 'valid':
        robot.send_hand_home(target, deadline_ns=deadline, check_abort=check)
        robot.hand._send_position.assert_called_once()
    else:
        with pytest.raises({'cancel':OperationCancelled,'stale':FeedbackUnavailable,'expired':HomeTimeout}[condition]):
            robot.send_hand_home(target, deadline_ns=deadline, check_abort=check)
        robot.hand._send_position.assert_not_called()

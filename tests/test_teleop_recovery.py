"""软暂停的控制、取消和设备协议回归；不连接 SDK。"""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from test_freshness import make_rollout, make_teleop, observation

from dexmani_real.robot.commands import (
    FeedbackUnavailable,
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

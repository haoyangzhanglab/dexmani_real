from collections import deque
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.config.xarm7 import ArmConfig
from dexmani_real.robot.commands import OperationCancelled, RobotError
from dexmani_real.robot.xarm7.driver import XArm7
from dexmani_real.robot.xarm7.process import EXECUTOR_TIMEOUT_S, ArmProcess
from dexmani_real.robot.xarm7.protocol import HomeRequest, PrepareRequest
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.runtime.state import SessionState


@pytest.fixture
def owner(monkeypatch):
    clock = [1_000_000_000]
    monkeypatch.setattr("time.monotonic_ns", lambda: clock[0])
    monkeypatch.setattr("time.monotonic", lambda: clock[0] / 1e9)

    def advance(seconds):
        clock[0] += round(seconds * 1e9)

    monkeypatch.setattr("time.sleep", advance)
    parent = ArmProcess(RealConfig().arm, runtime_state=SessionState())
    parent.pipe.close()
    parent.child_pipe.close()
    parent.process = Mock(pid=1, exitcode=0)
    parent.process.is_alive.return_value = True
    parent.io.ready.set()
    parent.io.stopped.set()
    requests, replies = deque(), deque()
    pump = Mock()

    def poll(timeout=0):
        pump()
        if not replies:
            advance(timeout)
        return bool(replies)

    parent.pipe = SimpleNamespace(send=requests.append, recv=replies.popleft, poll=poll)
    child_pipe = Mock()
    child_pipe.poll.return_value = False
    child_pipe.send.side_effect = replies.append
    child = ArmController(parent.cfg, parent.io, child_pipe, Mock())
    child.home = Mock(side_effect=lambda **kw: kw["check_abort"]())
    child.wait_settled = Mock()
    monkeypatch.setattr("dexmani_real.robot.xarm7.worker.ArmTrajectory", lambda *a: object())
    def wait_stopped(timeout):
        if not parent.io.stopped.is_set():
            advance(timeout)
        return parent.io.stopped.is_set()

    monkeypatch.setattr(parent.io.stopped, "wait", wait_stopped)
    return SimpleNamespace(parent=parent, child=child, requests=requests, replies=replies,
                           clock=clock, pump=pump, advance=advance)


def request(owner, *, service=lambda: None, kind=HomeRequest):
    parent = owner.parent
    epoch = parent.runtime_state.cancel_epoch
    parent.request_operation(
        kind(owner.clock[0] + 20_000_000_000), service=service,
        abort=lambda: parent.runtime_state.cancel_epoch != epoch or parent.io.emergency.is_set(),
        cancel_epoch=epoch,
    )


@pytest.mark.parametrize("stage", ["before_send", "queued", "executing", "cancel_reply", "completed", "after_recv"])
@pytest.mark.parametrize("next_kind", [HomeRequest, PrepareRequest])
def test_cancel_keeps_owner_and_consumes_reply(owner, stage, next_kind):
    parent, child = owner.parent, owner.child
    state = parent.runtime_state
    if stage == "after_recv":
        def receive():
            reply = owner.replies.popleft()
            state.request_stop()
            return reply

        parent.pipe.recv = receive

    def execute():
        if not owner.requests:
            return
        message = owner.requests.popleft()
        child.active = child.reference = object()
        if stage == "cancel_reply":
            state.request_stop()
        if stage == "executing":
            del child.home  # 使用真实 HOME 流程，在 SDK 接纳回位后触发取消。
            child.driver.move_home.side_effect = lambda *a: state.request_stop()
        child.operation(message)
        if stage == "completed":
            state.request_stop()

    def service():
        if stage == "before_send" and not state.stop_signal.is_set():
            state.request_stop()
        if stage == "queued" and owner.requests:
            state.request_stop()

    # queued: 首次 poll 未执行，父进程先看到取消，再消费 owner 回复。
    owner.pump.side_effect = (
        lambda: execute() if state.stop_signal.is_set() else None
    ) if stage == "queued" else execute
    with pytest.raises(OperationCancelled):
        request(owner, service=service)
    assert parent.is_connected and not parent.io.closing.is_set()
    assert state.stop_signal.is_set() and parent.io.stopped.is_set()
    assert child.active is child.reference is None
    assert not owner.requests and not owner.replies

    parent.pipe.recv = owner.replies.popleft
    child.home = Mock(side_effect=lambda **kw: kw["check_abort"]())
    owner.pump.side_effect = lambda: child.operation(owner.requests.popleft()) if owner.requests else None
    request(owner, kind=next_kind)
    assert parent.is_connected and not owner.replies
    if next_kind is PrepareRequest:
        assert child.reference is not None and not parent.io.stop.is_set()
    else:
        assert parent.io.stopped.is_set() and child.reference is None


@pytest.mark.parametrize("failure,error_type,message", [
    ("no_reply", TimeoutError, "lifecycle reply missing"),
    ("no_stop", TimeoutError, "stop or lifecycle reply missing"),
    ("stop_failure", RobotError, "stop failed"),
    ("fault", RobotError, "device failed"),
    ("reply_error", ValueError, "HOME failed"),
    ("exit", RobotError, "owner exited"),
    ("pipe", EOFError, "pipe closed"),
    ("emergency", OperationCancelled, "operation cancelled"),
    ("closing", OperationCancelled, "operation cancelled"),
])
def test_cancel_unresolved_or_fault_closes(owner, failure, error_type, message):
    parent, child = owner.parent, owner.child
    state = parent.runtime_state

    def service():
        if owner.requests and not state.stop_signal.is_set():
            state.request_stop()
            if failure == "no_stop":
                parent.io.stopped.clear()
                owner.replies.append(OperationCancelled("cancelled"))
            elif failure == "stop_failure":
                child.driver.stop.side_effect = RuntimeError("stop failed")
                child.operation(owner.requests.popleft())
            elif failure == "fault":
                parent.io.fault.value = b"device failed"
            elif failure == "reply_error":
                owner.replies.append(ValueError("HOME failed"))
            elif failure == "exit":
                parent.process.is_alive.return_value = False
            elif failure == "pipe":
                parent.pipe.poll = Mock(side_effect=EOFError("pipe closed"))
            elif failure == "emergency":
                state.request_emergency_stop()
            elif failure == "closing":
                parent.io.closing.set()

    with pytest.raises(error_type, match=message):
        request(owner, service=service)
    assert parent.io.closing.is_set() and parent.io.stop.is_set()
    if failure in {"no_reply", "no_stop"}:
        assert owner.clock[0] <= 1_010_000_000 + int(EXECUTOR_TIMEOUT_S * 1e9)


@pytest.mark.parametrize("late", ["stop", "reply"])
def test_cancellation_uses_one_budget(owner, late):
    parent = owner.parent
    start = None
    reply_sent = False

    def service():
        nonlocal start, reply_sent
        if not owner.requests:
            return
        if start is None:
            parent.runtime_state.request_stop()
            parent.io.stopped.clear()
            start = owner.clock[0]
        elapsed = owner.clock[0] - start
        if elapsed >= (4 if late == "stop" else 2) * 1_000_000_000:
            parent.io.stopped.set()
        if not reply_sent and elapsed >= (4 if late == "reply" else 2) * 1_000_000_000:
            owner.replies.append(OperationCancelled("cancelled"))
            reply_sent = True

    with pytest.raises(TimeoutError, match="cancellation unresolved"):
        request(owner, service=service)
    assert owner.clock[0] - start == int(EXECUTOR_TIMEOUT_S * 1e9)
    assert parent.io.closing.is_set()


@pytest.mark.parametrize("reason", ["emergency", "fault", "closing", "error"])
def test_worker_does_not_reuse_owner_after_fault(owner, reason):
    child = owner.child
    if reason == "emergency":
        child.io.emergency.set()
    elif reason == "fault":
        child.io.fault.value = b"device fault"
    elif reason == "closing":
        child.io.closing.set()
    else:
        child.home.side_effect = RuntimeError("HOME failed")
    child.operation(HomeRequest(owner.clock[0] + 1_000_000_000))
    assert child.io.closing.is_set() and child.io.stopped.is_set()
    assert child.active is child.reference is None
    assert isinstance(owner.replies.popleft(), Exception)


@pytest.mark.parametrize("caller,event", [
    (caller, event) for caller in ("teleop", "deployment", "keyboard")
    for event in ("s", "q", "esc") if caller != "keyboard" or event != "s"
])
def test_home_callers_preserve_idle_or_exit_intent(monkeypatch, caller, event):
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.teleop.keyboard import KeyboardTeleop
    from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase

    # 键盘点动没有 S 绑定；Q/Esc 沿现有外层取消路径退出。
    state = SessionState()
    cls = {"teleop": TeleopLoop, "deployment": PolicySession, "keyboard": KeyboardTeleop}[caller]
    obj = cls.__new__(cls)
    obj.runtime_state, obj.robot = state, Mock()
    obj.robot.arm.is_connected = True

    def cancel(*args, **kwargs):
        {"s": state.request_stop, "q": state.request_quit, "esc": state.request_emergency_stop}[event]()
        if event == "esc":
            obj.robot.arm.is_connected = False
        raise OperationCancelled("HOME cancelled")

    module = {"teleop": "dexmani_real.teleop.loop", "deployment": "dexmani_real.deployment.session",
              "keyboard": "dexmani_real.teleop.keyboard"}[caller]
    monkeypatch.setattr(module + ".home_hand", Mock())
    monkeypatch.setattr(module + ".home_arm", cancel)
    if caller == "teleop":
        obj.phase, obj.recorder = TeleopPhase.IDLE, None
        obj.keyboard, obj.audio = Mock(), Mock()
        obj.keyboard.accept_operation.return_value = state.cancel_epoch
        obj.poll_blocking_commands = Mock(return_value=False)
        obj.handle_operator_command("h")
        assert obj.phase is TeleopPhase.IDLE
    elif caller == "deployment":
        obj.episode = None
        assert obj.run_home(state.cancel_epoch)
    else:
        obj.prepare_jog = Mock()
        with pytest.raises(OperationCancelled):
            obj.run_home(state.cancel_epoch)
        obj.prepare_jog.assert_not_called()
    assert state.stop_signal.is_set()
    assert state.quit_requested == (event == "q")
    assert state.estop_request == (event == "esc")


@pytest.mark.parametrize("caller", ["teleop", "deployment"])
@pytest.mark.parametrize("event", ["s", "q", "esc", "fault"])
def test_preparation_cancellation_preserves_intent_and_fault(caller, event):
    from dexmani_real.deployment.rollout import EpisodeResult, Rollout
    from dexmani_real.teleop.loop import TeleopLoop

    cls = TeleopLoop if caller == "teleop" else Rollout
    obj = cls.__new__(cls)
    state = obj.runtime_state = SessionState()
    obj.cancel_epoch = state.cancel_epoch
    obj.robot = Mock()

    def cancel(*args):
        {"s": state.request_stop, "q": state.request_quit,
         "esc": state.request_emergency_stop, "fault": state.request_stop}[event]()
        obj.robot.arm.is_connected = event not in {"esc", "fault"}
        if event == "fault":
            obj.robot.check.side_effect = RobotError("device failed during cancellation")
        raise OperationCancelled("prepare cancelled")

    if caller == "teleop":
        obj.recorder = obj.recording_dir = None
        obj.preparation_interrupted = Mock(return_value=False)
        obj.poll_blocking_commands = Mock(return_value=False)
        obj.start_observation = Mock(return_value=object())
        obj.robot.prepare_streaming.side_effect = cancel
        run = lambda: obj.start_episode(obj.cancel_epoch)
    else:
        obj.result = EpisodeResult(1)
        obj.started_ns = 0
        obj.preparation_issue = None
        obj.history = deque()
        obj.prepare = cancel
        run = lambda: obj.run(poll_operator=lambda: None)
    if event == "fault":
        with pytest.raises(RobotError, match="device failed during cancellation"):
            run()
    else:
        run()
        if caller == "teleop":
            assert obj.pending_termination_reason == "cancelled"
        else:
            assert obj.result.termination_reason == "start_cancelled"
    assert state.stop_signal.is_set()
    assert state.quit_requested == (event == "q")
    assert state.estop_request == (event == "esc")


@pytest.mark.parametrize("mass", [0, 1.1, -1, float("nan"), float("inf")])
def test_bare_flange_mass(mass):
    cfg = replace(ArmConfig(), tcp_load_mass_kg=mass)
    if np.isfinite(mass) and mass >= 0:
        cfg.validate()
    else:
        with pytest.raises(ValueError):
            cfg.validate()


@pytest.mark.parametrize("first_mode", [0, 1])
def test_prepare_reuses_static_parameters_and_keeps_mode_enable(first_mode):
    driver = XArm7(ArmConfig())
    api = Mock()
    api.get_err_warn_code.return_value = (0, (0, 0))
    for name in (
        "motion_enable", "set_mode", "set_state", "set_collision_sensitivity",
        "set_tcp_load", "set_joint_maxacc",
    ):
        getattr(api, name).return_value = 0
    driver.api = api
    for mode in (first_mode, 1 - first_mode):
        driver.prepare_motion(mode, check_abort=lambda: None)
    assert api.set_state.call_args_list == [call(0), call(0)]
    assert api.set_mode.call_args_list == [call(first_mode), call(1 - first_mode)]
    api.set_collision_sensitivity.assert_called_once()
    api.set_tcp_load.assert_called_once()
    assert api.set_joint_maxacc.call_count == 2

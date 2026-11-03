"""机械臂轨迹、停止复用与异常退出的离线回归；SDK 操作使用 fake driver。"""
import multiprocessing as mp
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.xarm7 import EXECUTOR_TIMEOUT_S, ArmConfig
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, validate_arm_target
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import (
    GOAL_DTYPE,
    ArmChannels,
    HomeRequest,
    PrepareRequest,
)
from dexmani_real.robot.xarm7.trajectory import ArmTrajectory
from dexmani_real.robot.xarm7.worker import ArmController
from dexmani_real.runtime.state import SessionState


@pytest.fixture(autouse=True)
def in_process_sync(monkeypatch):
    # 本文件的 owner 均为线程或 fake process；同步不跨进程，Pipe 和共享缓冲仍用真实实现。
    ctx = mp.get_context('spawn')
    monkeypatch.setattr(ctx, 'Event', threading.Event)
    monkeypatch.setattr(ctx, 'Lock', threading.Lock)


def channels(state):
    return ArmChannels(mp.get_context('spawn'), stop_signal=state.stop_signal,
                       emergency_signal=state.emergency_signal)


@pytest.mark.parametrize('target', ['lower', 'upper', 'interior'])
def test_arm_trajectory_boundary_roundoff_and_commit(target):
    cfg = ArmConfig()
    start = np.asarray(cfg.home_qpos)
    goal = start.copy()
    goal[3] = dict(lower=cfg.joint_limit_lower[3], upper=cfg.joint_limit_upper[3],
                   interior=.1)[target]
    validate_arm_target(goal, cfg.joint_limit_lower, cfg.joint_limit_upper)
    trajectory = ArmTrajectory(cfg, start, np.zeros(7))
    previous = start
    # 覆盖完整运动和到位后的持续计算；每一帧都按 worker 的发送/commit 顺序推进。
    for _ in range(1000):
        q = trajectory.step(goal)
        validate_arm_target(q, cfg.joint_limit_lower, cfg.joint_limit_upper)
        np.testing.assert_array_equal(trajectory.input.current_position, previous)
        np.testing.assert_allclose(trajectory.output.new_position, q, rtol=0, atol=1e-12)
        trajectory.commit(q)
        np.testing.assert_array_equal(trajectory.input.current_position, q)
        previous = q
        if trajectory.output.time >= trajectory.output.trajectory.duration:
            np.testing.assert_allclose(q, goal, rtol=0, atol=1e-12)
            np.testing.assert_allclose(trajectory.input.current_velocity, 0, rtol=0, atol=1e-12)
    assert trajectory.output.time >= trajectory.output.trajectory.duration


@pytest.mark.parametrize('direction', [-1, 1], ids=['lower', 'upper'])
@pytest.mark.parametrize('hold', [False, True])
def test_arm_trajectory_rejects_braking_outside_limits(direction, hold):
    cfg = ArmConfig()
    goal = np.asarray(cfg.home_qpos)
    start, velocity = goal.copy(), np.zeros(7)
    bound = cfg.joint_limit_lower[3] if direction < 0 else cfg.joint_limit_upper[3]
    start[3] = bound - direction * .001
    velocity[3] = direction * .5
    validate_arm_target(goal, cfg.joint_limit_lower, cfg.joint_limit_upper)
    trajectory = ArmTrajectory(cfg, start, velocity)
    with pytest.raises(ValueError, match='extremum outside joint 4 limits'):
        trajectory.step(goal, hold=hold)
    np.testing.assert_array_equal(trajectory.input.current_position, start)


@pytest.fixture
def stubbed_arm_trajectory():
    from ruckig import Result
    cfg = ArmConfig()
    trajectory = ArmTrajectory(cfg, np.asarray(cfg.home_qpos), np.zeros(7))
    profiles = [SimpleNamespace(p=[p], v=[], a=[], j=[], t=[],
                                brake=SimpleNamespace(p=[], v=[], a=[], j=[], t=[]))
                for p in cfg.home_qpos]
    trajectory.output = SimpleNamespace(new_calculation=True, new_position=list(cfg.home_qpos),
                                         trajectory=SimpleNamespace(profiles=[profiles]))
    trajectory.otg = SimpleNamespace(update=lambda *args: Result.Working)
    return trajectory


@pytest.mark.parametrize('source', ['extremum', 'sample'])
@pytest.mark.parametrize('direction', [-1, 1], ids=['lower', 'upper'])
@pytest.mark.parametrize('offset', [5e-13, 2e-12, np.nan, np.inf, -np.inf])
def test_arm_trajectory_roundoff_tolerance_is_small_and_finite(
    stubbed_arm_trajectory, source, direction, offset,
):
    trajectory = stubbed_arm_trajectory
    bound = trajectory.lower[3] if direction < 0 else trajectory.upper[3]
    value = bound + direction * offset
    if source == 'extremum':
        trajectory.output.trajectory.profiles[0][3].p = [value]
    else:
        trajectory.output.new_calculation = False
        trajectory.output.new_position[3] = value
    if offset == 5e-13:
        q = trajectory.step()
        validate_arm_target(q, trajectory.lower, trajectory.upper)
        if source == 'sample':
            assert q[3] == bound
    else:
        with pytest.raises(ValueError, match='outside joint'):
            trajectory.step()


@pytest.mark.parametrize('segment', ['brake', 'profile'])
@pytest.mark.parametrize('jerk', [0., .6])
def test_arm_trajectory_checks_internal_extrema(stubbed_arm_trajectory, segment, jerk):
    trajectory = stubbed_arm_trajectory
    p = trajectory.lower[3] + .01
    v, a, duration = (-.2, .4, 1.) if jerk == 0 else (-.3, 0., 2.)
    end = p + duration * (v + duration * (a / 2 + duration * jerk / 6))
    assert trajectory.lower[3] < min(p, end) <= max(p, end) < trajectory.upper[3]
    profile = trajectory.output.trajectory.profiles[0][3]
    section = profile.brake if segment == 'brake' else profile
    section.p, section.v, section.a, section.j, section.t = [p, end], [v], [a], [jerk], [duration]
    # 两端合法，只有段内 v=0 的位置越界；制动段与主轨迹都必须检查。
    with pytest.raises(ValueError, match='extremum outside joint 4 limits'):
        trajectory.step()


@pytest.mark.parametrize('result', ['success', 'sdk_error', 'expired'])
def test_owner_commits_boundary_reference_only_after_send(monkeypatch, result):
    cfg = ArmConfig()
    goal = np.asarray(cfg.home_qpos).copy()
    goal[3] = cfg.joint_limit_lower[3]
    now = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.robot.xarm7.worker.time.monotonic_ns', lambda: now[0])
    io = channels(SessionState())
    expires_ns = now[0] + 1_000_000_000
    io.publish_goal(RobotCommand(arm_qpos=goal), expires_ns=expires_ns)
    sent = []

    def servo(q):
        validate_arm_target(q, cfg.joint_limit_lower, cfg.joint_limit_upper)
        sent.append(q.copy())
        if result == 'sdk_error' and q[3] == goal[3]:
            raise OSError('servo failed')

    owner = ArmController(cfg, io, None, SimpleNamespace(servo=servo))
    owner.status['qpos'] = cfg.home_qpos
    owner.status['timestamp_ns'] = now[0]
    owner.tick()
    trajectory = owner.reference
    ticks = int(np.ceil(trajectory.output.trajectory.duration * cfg.servo_hz))
    for _ in range(ticks - 2):
        owner.tick()
    previous = np.array([trajectory.input.current_position, trajectory.input.current_velocity,
                         trajectory.input.current_acceleration])
    if result == 'expired':
        step = trajectory.step

        def expire_during_step(*args, **kwargs):
            q = step(*args, **kwargs)
            now[0] = expires_ns
            return q

        monkeypatch.setattr(trajectory, 'step', expire_during_step)
    if result == 'sdk_error':
        with pytest.raises(OSError, match='servo failed'):
            owner.tick()
    else:
        owner.tick()
    assert trajectory.output.new_position[3] < goal[3]
    if result == 'success':
        assert sent[-1][3] == goal[3]
        np.testing.assert_array_equal(trajectory.input.current_position, sent[-1])
    else:
        np.testing.assert_array_equal(
            [trajectory.input.current_position, trajectory.input.current_velocity,
             trajectory.input.current_acceleration], previous,
        )
    assert len(sent) == ticks - (result == 'expired')
    assert io.stop.is_set() == (result == 'expired')


@pytest.mark.parametrize('invalid', ['shape', 'nan', 'inf', 'below', 'above'])
def test_public_arm_targets_remain_strict(invalid):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.robot.system import Robot
    cfg = RealConfig()
    q = np.asarray(cfg.arm.home_qpos).copy()
    if invalid == 'shape':
        q = q[:-1]
    else:
        q[3] = dict(nan=np.nan, inf=np.inf,
                    below=np.nextafter(cfg.arm.joint_limit_lower[3], -np.inf),
                    above=np.nextafter(cfg.arm.joint_limit_upper[3], np.inf))[invalid]
    robot = Robot(cfg.arm, cfg.hand, runtime_state=SessionState())
    robot.connected = True
    robot.arm = SimpleNamespace(is_connected=True,
                                submit=lambda *a, **kw: pytest.fail('invalid target submitted'))
    with pytest.raises(ValueError, match='shape|finite|joint limits'):
        robot.send_action(RobotCommand(arm_qpos=q), expires_ns=time.monotonic_ns() + 1_000_000_000)


def test_late_publisher_cannot_install_target_while_stopped():
    state = SessionState()
    io = channels(state)
    entered, release = threading.Event(), threading.Event()
    sent = []

    class PausedCommand:
        arm_qpos = np.zeros(7)

        @property
        def arm_hold(self):
            # publish_goal 已经检查 stop，且仍持有邮箱锁。
            entered.set()
            assert release.wait(2)
            return False

    owner = ArmController(ArmConfig(), io, None,
                          SimpleNamespace(stop=lambda: None, servo=sent.append))
    errors = []

    def publish():
        try:
            io.publish_goal(PausedCommand(), expires_ns=time.monotonic_ns() + 2_000_000_000)
        except BaseException as error:
            errors.append(error)

    producer = threading.Thread(target=publish)
    producer.start()
    try:
        assert entered.wait(2)
        owner.stop()
        owner.tick()
    finally:
        release.set()
        producer.join(2)
    assert not producer.is_alive()
    assert not errors
    owner.tick()
    assert owner.active is None and owner.reference is None
    assert np.frombuffer(io.goal, dtype=GOAL_DTYPE)['pending'][0]
    io.clear_goal(deadline_ns=time.monotonic_ns() + 1_000_000_000)
    assert io.take_goal() is None
    assert sent == []


def test_mailbox_clear_timeout_keeps_stop():
    state = SessionState()
    io = channels(state)
    state.stop_signal.set()
    with io.goal_lock:
        with pytest.raises(TimeoutError):
            io.clear_goal(deadline_ns=time.monotonic_ns())
    assert io.stop.is_set()


@pytest.mark.parametrize("cancel", ["request_stop", "request_quit", "request_emergency_stop"])
@pytest.mark.parametrize("stage", ["stop_confirmation", "mailbox_clear"])
@pytest.mark.parametrize("home", [False, True])
def test_cancel_during_recovery_is_not_cleared(cancel, stage, home):
    state = SessionState()
    process = ArmProcess(ArmConfig(), runtime_state=state)
    process.process = SimpleNamespace(is_alive=lambda: True)
    process.io.ready.set()
    process.check_error = lambda: None
    process.wait_stopped = lambda **kwargs: None
    if stage == "stop_confirmation":
        process.wait_stopped = lambda **kwargs: getattr(state, cancel)()
    else:
        clear_goal = process.io.clear_goal
        def clear_and_cancel(**kwargs):
            clear_goal(**kwargs)
            getattr(state, cancel)()
        process.io.clear_goal = clear_and_cancel
    # 启动早期的普通停止也必须改变取消上下文。
    epoch = state.cancel_epoch
    try:
        with pytest.raises(OperationCancelled):
            deadline = time.monotonic_ns() + 1_000_000_000
            request = HomeRequest(np.empty((0, 7)), deadline) if home else PrepareRequest(deadline)
            process.request_operation(request,
                                      service=lambda: None, abort=lambda: False,
                                      cancel_epoch=epoch)
        assert state.stop_signal.is_set()
    finally:
        process.pipe.close()
        process.child_pipe.close()


def test_unconsumed_reply_prevents_new_operation():
    state = SessionState()
    process = ArmProcess(ArmConfig(), runtime_state=state)
    process.process = SimpleNamespace(is_alive=lambda: True)
    process.io.ready.set()
    process.check_error = lambda: None
    process.reply_pending = True
    try:
        with pytest.raises(RuntimeError, match='reply remains pending'):
            process.prepare(state.cancel_epoch, service=lambda: None)
    finally:
        process.pipe.close()
        process.child_pipe.close()


def test_target_publication_after_stop_is_rejected():
    state = SessionState()
    io = channels(state)
    state.stop_signal.set()
    with pytest.raises(OperationCancelled):
        io.publish_goal(RobotCommand(arm_qpos=np.zeros(7)),
                        expires_ns=time.monotonic_ns() + 1_000_000_000)


def test_owner_prepare_and_new_target_after_residual_mailbox():
    state = SessionState()
    process = ArmProcess(ArmConfig(), runtime_state=state)
    sent = threading.Event()
    q = np.asarray(process.cfg.home_qpos)
    driver = SimpleNamespace(connect=lambda **kwargs: None, initialize=lambda **kwargs: None,
                             read=lambda: (q, np.zeros(7)), stop=lambda: None,
                             enter_mode=lambda *a, **kwargs: None, close=lambda: None,
                             servo=lambda target: sent.set())
    owner = ArmController(process.cfg, process.io, process.child_pipe, driver)
    thread = threading.Thread(target=owner.run, args=(time.monotonic_ns() + 5_000_000_000,))
    process.process = SimpleNamespace(is_alive=thread.is_alive, pid=1, exitcode=None)
    thread.start()
    try:
        assert process.io.ready.wait(3)
        process.request_stop()
        process.wait_stopped()
        # 模拟旧调用在 stop 后完成发布的残留目标。
        slot = np.frombuffer(process.io.goal, dtype=GOAL_DTYPE)
        with process.io.goal_lock:
            slot[0] = (True, False, time.monotonic_ns() + 1_000_000_000, q)
        process.prepare(state.cancel_epoch, service=lambda: None)
        assert not slot['pending'][0]
        assert not sent.is_set()
        process.submit(RobotCommand(arm_qpos=q), expires_ns=time.monotonic_ns() + 1_000_000_000)
        assert sent.wait(2)
        process.request_stop()
        process.wait_stopped()
        assert owner.active is None and owner.reference is None
    finally:
        process.io.closing.set()
        process.io.stop.set()
        thread.join(3)
        process.pipe.close()
        if not process.child_pipe.closed:
            process.child_pipe.close()
    assert not thread.is_alive()


@pytest.mark.parametrize("expired", ["goal", "feedback"])
def test_owner_stops_on_expired_goal_or_feedback(monkeypatch, expired):
    state = SessionState()
    io = channels(state)
    clock = [time.monotonic_ns()]
    monkeypatch.setattr('dexmani_real.robot.xarm7.worker.time.monotonic_ns', lambda: clock[0])
    sent = []
    owner = ArmController(ArmConfig(), io, None, SimpleNamespace(servo=sent.append))
    io.publish_goal(RobotCommand(arm_qpos=np.zeros(7)), expires_ns=clock[0] + (10_000_000 if expired == "goal" else 2_000_000_000))
    clock[0] += 20_000_000
    owner.status['timestamp_ns'] = clock[0] if expired == "goal" else clock[0] - 200_000_000
    owner.tick()
    assert io.stop.is_set()
    assert sent == []


def test_old_stop_confirmation_cannot_complete_stop_during_new_sdk_work():
    state = SessionState()
    process = ArmProcess(ArmConfig(), runtime_state=state)
    process.io.stopped.set()
    entered, release = threading.Event(), threading.Event()
    replies = []
    def enter_mode(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        kwargs['check_abort']()
    owner = ArmController(process.cfg, process.io,
                          SimpleNamespace(poll=lambda: False, send=replies.append),
                          SimpleNamespace(enter_mode=enter_mode, stop=lambda: None))
    thread = threading.Thread(target=owner.operation,
                              args=(PrepareRequest(time.monotonic_ns() + 2_000_000_000),))
    process.process = SimpleNamespace(is_alive=thread.is_alive, pid=1, exitcode=None)
    thread.start()
    try:
        assert entered.wait(2)
        process.request_stop()
        with pytest.raises(TimeoutError, match='stop unresolved'):
            process.wait_stopped(deadline=time.monotonic() + .01)
    finally:
        release.set()
        thread.join(2)
        process.pipe.close()
        process.child_pipe.close()
    assert not thread.is_alive()
    assert len(replies) == 1 and isinstance(replies[0], OperationCancelled)
    assert process.io.stopped.is_set()


def test_recovery_reports_owner_fault_even_when_operator_also_cancelled():
    from dexmani_real.robot.commands import RobotError
    state = SessionState()
    process = ArmProcess(ArmConfig(), runtime_state=state)
    process.process = SimpleNamespace(is_alive=lambda: True, pid=1, exitcode=None)
    process.io.ready.set()
    epoch = state.cancel_epoch
    state.request_stop()
    process.io.fault.value = b'OSError: SDK feedback failed'
    try:
        with pytest.raises(RobotError, match='SDK feedback failed'):
            process.prepare(epoch, service=lambda: None)
        assert not process.child_pipe.poll()
    finally:
        process.pipe.close()
        process.child_pipe.close()


@pytest.mark.parametrize('alive', [False, True])
def test_owner_fault_and_late_close_error_are_reported(alive):
    from dexmani_real.robot.commands import RobotError
    state = SessionState()
    arm = ArmProcess(ArmConfig(), runtime_state=state)
    arm.io.fault.value = b'original SDK fault'
    arm.io.stopped.set()
    close_error = arm.io.close_error
    def join(timeout):
        close_error.value = b'late disconnect failure'
    arm.process = SimpleNamespace(pid=1, exitcode=None if alive else 0,
                                  is_alive=lambda: alive, join=join)
    try:
        with pytest.raises(RobotError, match='original SDK fault'):
            arm.check_error()
        with pytest.raises(RobotError, match='original SDK fault'):
            arm.close()
        assert arm.is_closed == (not alive)
        assert arm._read_close_error().args == ('late disconnect failure',)
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_repeated_close_does_not_renew_pending_owner_budget(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr('dexmani_real.robot.xarm7.process.time.monotonic', lambda: clock[0])
    arm = ArmProcess(ArmConfig(), runtime_state=SessionState())
    alive = [True]
    waits = []
    def join(timeout):
        waits.append(timeout)
        clock[0] += timeout
    arm.process = SimpleNamespace(pid=1, exitcode=0, is_alive=lambda: alive[0], join=join)
    try:
        with pytest.raises(TimeoutError, match='still active'):
            arm.close()
        deadline = arm.close_deadline
        clock[0] += 10
        with pytest.raises(TimeoutError, match='still active'):
            arm.close()
        assert waits == [EXECUTOR_TIMEOUT_S, 0.0]
        assert arm.close_deadline == deadline
        assert not arm.is_closed and not arm.pipe.closed and not arm.child_pipe.closed
        # 真正退出后仍可检查晚到错误并关闭端点，无需新的等待窗口。
        alive[0] = False
        arm.io.stopped.set()
        arm.close()
        assert waits[-1] == 0.0 and arm.is_closed
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_cancelled_lifecycle_request_cannot_reuse_owner():
    from dexmani_real.robot.commands import OperationCancelled
    state = SessionState()
    arm = ArmProcess(ArmConfig(), runtime_state=state)
    arm.resume = lambda *args, **kwargs: None
    epoch = state.cancel_epoch
    try:
        with pytest.raises(OperationCancelled):
            arm.request_operation(
                PrepareRequest(time.monotonic_ns() + 1_000_000_000),
                service=state.request_stop,
                abort=lambda: state.cancel_epoch != epoch,
                cancel_epoch=epoch,
            )
        assert arm.io.stop.is_set() and arm.io.closing.is_set()
        assert not arm.reply_pending
        assert not arm.is_connected
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_cancel_inflight_keeps_reply_owned_and_closes_session():
    from dexmani_real.robot.commands import OperationCancelled
    state = SessionState()
    arm = ArmProcess(ArmConfig(), runtime_state=state)
    arm.process = SimpleNamespace(is_alive=lambda: True, pid=1, exitcode=None)
    arm.io.ready.set()
    arm.resume = lambda *args, **kwargs: None
    services = []
    def service():
        services.append(True)
        if len(services) == 2:
            state.request_stop()
    epoch = state.cancel_epoch
    try:
        with pytest.raises(OperationCancelled):
            arm.request_operation(
                PrepareRequest(time.monotonic_ns() + 1_000_000_000),
                service=service, abort=lambda: state.cancel_epoch != epoch,
                cancel_epoch=epoch,
            )
        assert arm.reply_pending and arm.io.closing.is_set()
        assert isinstance(arm.child_pipe.recv(), PrepareRequest)
        arm.resume = ArmProcess.resume.__get__(arm, ArmProcess)
        with pytest.raises(RuntimeError, match='unavailable'):
            arm.prepare(state.cancel_epoch, service=lambda: None)
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('error', [OperationCancelled('cancelled'), TimeoutError('stop unresolved')])
def test_preparation_resume_failure_also_closes_owner(error):
    state = SessionState()
    arm = ArmProcess(ArmConfig(), runtime_state=state)
    arm.resume = lambda *args, **kwargs: (_ for _ in ()).throw(error)
    try:
        with pytest.raises(type(error), match=str(error)):
            arm.prepare(state.cancel_epoch, service=lambda: None)
        assert arm.io.stop.is_set() and arm.io.closing.is_set()
        assert not arm.reply_pending and not arm.child_pipe.poll()
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_feedback_requires_live_ready_owner_and_survives_stop():
    arm = ArmProcess(ArmConfig(), runtime_state=SessionState())
    alive = [True]
    arm.process = SimpleNamespace(pid=1, exitcode=0, is_alive=lambda: alive[0])
    try:
        with pytest.raises(RuntimeError, match='unavailable'):
            arm.snapshot()
        arm.io.ready.set()
        assert arm.is_connected
        arm.snapshot()
        arm.request_stop()
        arm.io.stopped.set()
        assert arm.is_connected
        arm.snapshot()
        arm.io.closing.set()
        with pytest.raises(RuntimeError, match='unavailable'):
            arm.snapshot()
        alive[0] = False
        assert not arm.is_connected
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('operation', ['connect', 'snapshot', 'submit', 'prepare'])
def test_failed_startup_ready_never_authorizes_operation(monkeypatch, operation):
    from dexmani_real.robot.commands import RobotError
    arm = ArmProcess(ArmConfig(), runtime_state=SessionState())
    def fail_start():
        arm.io.fault.value = b'OSError: initialization failed'
        arm.io.ready.set()  # worker 的失败收尾也唤醒父进程。
    process = SimpleNamespace(pid=1, exitcode=None, is_alive=lambda: True, start=fail_start)
    monkeypatch.setattr(arm.ctx, 'Process', lambda **kwargs: process)
    try:
        with pytest.raises(RobotError, match='initialization failed'):
            if operation == 'connect':
                arm.connect()
            else:
                arm.process = process
                fail_start()
                if operation == 'snapshot':
                    arm.snapshot()
                elif operation == 'submit':
                    arm.submit(RobotCommand(arm_qpos=np.zeros(7)),
                               expires_ns=time.monotonic_ns() + 1_000_000_000)
                else:
                    arm.prepare(arm.runtime_state.cancel_epoch, service=lambda: None)
        assert arm.io.take_goal() is None
        assert not arm.reply_pending
    finally:
        arm.pipe.close()
        arm.child_pipe.close()

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
def test_arm_trajectory_boundary_roundoff_and_advance(target):
    cfg = ArmConfig()
    start = np.asarray(cfg.home_qpos)
    goal = start.copy()
    goal[3] = dict(lower=cfg.joint_limit_lower[3], upper=cfg.joint_limit_upper[3],
                   interior=.1)[target]
    validate_arm_target(goal, cfg.joint_limit_lower, cfg.joint_limit_upper)
    trajectory = ArmTrajectory(cfg, start, np.zeros(7))
    # 覆盖完整运动和到位后的持续计算；每一帧均验证归一位置和连续参考。
    for _ in range(1000):
        q = trajectory.step(goal)
        validate_arm_target(q, cfg.joint_limit_lower, cfg.joint_limit_upper)
        np.testing.assert_allclose(trajectory.output.new_position, q, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(trajectory.input.current_position, q)
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
                                         trajectory=SimpleNamespace(profiles=[profiles]),
                                         pass_to_input=lambda inp: None)
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
def test_owner_sends_normalized_reference_and_rejects_expired_candidate(monkeypatch, result):
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
    owner.reference = ArmTrajectory(cfg, np.asarray(cfg.home_qpos), np.zeros(7))
    owner.tick()
    trajectory = owner.reference
    ticks = int(np.ceil(trajectory.output.trajectory.duration * cfg.servo_hz))
    for _ in range(ticks - 2):
        owner.tick()
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
            request = HomeRequest(deadline) if home else PrepareRequest(deadline)
            process.request_operation(request,
                                      service=lambda: None, abort=lambda: False,
                                      cancel_epoch=epoch)
        assert state.stop_signal.is_set()
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
    driver = SimpleNamespace(connect=lambda **kwargs: None, read=lambda: (q, np.zeros(7)), stop=lambda: None,
                             prepare_motion=lambda *a, **kwargs: None, close=lambda: None,
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
        # 已确认停止、请求尚未发送的取消可复用同一 Owner。
        epoch = state.cancel_epoch
        with pytest.raises(OperationCancelled):
            process.prepare(epoch, service=state.request_stop)
        assert process.is_connected and process.io.stop.is_set()
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
    def prepare_motion(*args, **kwargs):
        entered.set()
        assert release.wait(2)
        kwargs['check_abort']()
    owner = ArmController(process.cfg, process.io,
                          SimpleNamespace(poll=lambda: False, send=replies.append),
                          SimpleNamespace(prepare_motion=prepare_motion, stop=lambda: None))
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
def test_owner_first_fault_survives_close(alive):
    from dexmani_real.robot.commands import RobotError
    state = SessionState()
    arm = ArmProcess(ArmConfig(), runtime_state=state)
    arm.io.fault.value = b'original SDK fault'
    arm.io.stopped.set()
    arm.process = SimpleNamespace(pid=1, exitcode=None if alive else 0,
                                  is_alive=lambda: alive, join=lambda timeout: None)
    try:
        with pytest.raises(RobotError, match='original SDK fault'):
            arm.check_error()
        with pytest.raises(RobotError, match='original SDK fault'):
            arm.close()
        assert arm.is_closed == (not alive)
        assert arm.fault == 'original SDK fault'
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_close_retains_pending_owner_until_exit():
    arm = ArmProcess(ArmConfig(), runtime_state=SessionState())
    alive = [True]
    waits = []
    def join(timeout):
        assert arm.io.stop.is_set() and arm.io.closing.is_set()
        waits.append(timeout)
    owner = arm.process = SimpleNamespace(pid=1, exitcode=None, is_alive=lambda: alive[0], join=join)
    try:
        with pytest.raises(TimeoutError, match='still active'):
            arm.close()
        with pytest.raises(TimeoutError, match='still active'):
            arm.close()
        assert waits and all(0 < timeout <= EXECUTOR_TIMEOUT_S for timeout in waits)
        assert arm.process is owner
        assert not arm.is_closed and not arm.pipe.closed and not arm.child_pipe.closed
        # 真正退出后仍可检查晚到错误并关闭端点。
        alive[0] = False
        owner.exitcode = 0
        arm.io.stopped.set()
        arm.close()
        assert waits[-1] == EXECUTOR_TIMEOUT_S and arm.is_closed
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('field', ['fault'])
def test_arm_close_reads_errors_first_reported_during_join(field):
    from dexmani_real.robot.commands import RobotError
    arm = ArmProcess(ArmConfig(), runtime_state=SessionState())
    arm.io.stopped.set()
    def join(timeout):
        getattr(arm.io, field).value = b'late owner failure'
    arm.process = SimpleNamespace(pid=1, exitcode=0, is_alive=lambda: False, join=join)
    try:
        for _ in range(2):
            with pytest.raises(RobotError, match='late owner failure'):
                arm.close()
            assert arm.is_closed
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_arm_join_interrupt_latches_before_pipe_cleanup():
    state = SessionState()
    arm = ArmProcess(ArmConfig(), runtime_state=state)
    arm.io.stopped.set()
    def join(timeout):
        raise KeyboardInterrupt('join interrupted')
    arm.process = SimpleNamespace(pid=1, exitcode=0, is_alive=lambda: False, join=join)
    arm.pipe.close()
    arm.child_pipe.close()
    closed = []
    def close_pipe():
        assert state.estop_request and state.stop_signal.is_set()
        closed.append(True)
    arm.pipe = arm.child_pipe = SimpleNamespace(close=close_pipe)
    with pytest.raises(KeyboardInterrupt, match='join interrupted'):
        arm.close()
    assert len(closed) == 2


def test_cancel_before_submission_keeps_healthy_owner():
    from dexmani_real.robot.commands import OperationCancelled
    state = SessionState()
    arm = ArmProcess(ArmConfig(), runtime_state=state)
    arm.process = SimpleNamespace(is_alive=lambda: True, pid=1, exitcode=None)
    arm.io.ready.set()
    arm.io.stopped.set()
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
        assert arm.io.stop.is_set() and not arm.io.closing.is_set()
        assert arm.is_connected
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_cancel_inflight_closes_session():
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
        assert arm.io.closing.is_set()
        assert isinstance(arm.child_pipe.recv(), PrepareRequest)
        arm.resume = ArmProcess.resume.__get__(arm, ArmProcess)
        with pytest.raises(RuntimeError, match='unavailable'):
            arm.prepare(state.cancel_epoch, service=lambda: None)
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('error', [TimeoutError('stop unresolved')])
def test_preparation_resume_failure_also_closes_owner(error):
    state = SessionState()
    arm = ArmProcess(ArmConfig(), runtime_state=state)
    arm.resume = lambda *args, **kwargs: (_ for _ in ()).throw(error)
    try:
        with pytest.raises(type(error), match=str(error)):
            arm.prepare(state.cancel_epoch, service=lambda: None)
        assert arm.io.stop.is_set() and arm.io.closing.is_set()
        assert not arm.child_pipe.poll()
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
        # 失败仅设置 fault，不发布成功 ready。
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
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('startup_failure', [False, True])
def test_owner_cleanup_preserves_first_error_and_logs_later_failures(startup_failure, caplog):
    io = channels(SessionState())
    def connect(**kwargs):
        if startup_failure:
            raise OSError('startup read failed')
    def stop():
        raise OSError('stop refused')
    def close():
        raise OSError('disconnect refused')
    def eof():
        raise EOFError
    pipe = SimpleNamespace(poll=lambda *a: io.ready.is_set(), recv=eof, close=lambda: None)
    cfg = ArmConfig()
    driver = SimpleNamespace(connect=connect, read=lambda: (cfg.home_qpos, np.zeros(7)),
                             stop=stop, close=close)
    ArmController(cfg, io, pipe, driver).run(time.monotonic_ns()+1_000_000_000)
    assert io.ready.is_set() == (not startup_failure)
    assert io.stopped.is_set()
    assert ('startup read failed' if startup_failure else 'stop refused') in io.fault.value.decode()
    assert 'stop refused' in caplog.text and 'disconnect refused' in caplog.text
    assert 'EOFError' not in io.fault.value.decode()


@pytest.mark.parametrize('mode', [0, 1])
def test_motion_configuration_finishes_with_state_zero(mode):
    from dataclasses import replace
    from dexmani_real.robot.xarm7.driver import XArm7
    cfg = ArmConfig()
    cfg = replace(cfg, homing=replace(cfg.homing, max_acceleration_deg_per_s2=300))
    events = []
    def call(name):
        def record(*args, **kwargs):
            events.append((name, args, kwargs))
            return 0
        return record
    driver = XArm7(cfg)
    driver.api = SimpleNamespace(mode=1, get_err_warn_code=lambda: (0, [0, 0]),
        **{name: call(name) for name in ('motion_enable', 'set_mode', 'set_state',
           'set_joint_maxacc', 'set_collision_sensitivity', 'set_tcp_load', 'set_servo_angle')})
    driver.prepare_motion(mode, check_abort=lambda: None)
    assert events[-1] == ('set_state', (0,), {})
    acceleration = next(args[0] for name, args, kw in events if name == 'set_joint_maxacc')
    assert acceleration == (cfg.homing.max_acceleration_rad_per_s2 if mode == 0
                            else cfg.max_joint_acceleration_rad_per_s2)
    driver.move_home(np.asarray(cfg.home_qpos))
    assert events[-1][2]['mvacc'] == cfg.homing.max_acceleration_rad_per_s2


def test_cancel_before_submission_does_not_hide_unconfirmed_stop():
    arm = ArmProcess(ArmConfig(), runtime_state=SessionState())
    arm.process = SimpleNamespace(pid=1, is_alive=lambda: True)
    arm.io.ready.set()
    arm.resume = lambda *a, **kw: None
    arm.wait_stopped = lambda **kw: (_ for _ in ()).throw(TimeoutError('stop unresolved'))
    try:
        with pytest.raises(TimeoutError, match='stop unresolved'):
            arm.request_operation(PrepareRequest(time.monotonic_ns()+1_000_000_000),
                                  service=lambda: None, abort=lambda: True,
                                  cancel_epoch=arm.runtime_state.cancel_epoch)
        assert arm.io.closing.is_set() and not arm.child_pipe.poll()
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('moving', [False, True])
def test_first_goal_uses_fresh_feedback_after_preparation(monkeypatch, moving):
    cfg = ArmConfig()
    now = 10_000_000_000
    monkeypatch.setattr('dexmani_real.robot.xarm7.worker.time.monotonic_ns', lambda: now)
    io = channels(SessionState())
    sent = []
    owner = ArmController(cfg, io, None, SimpleNamespace(servo=lambda q: sent.append(q.copy())))
    original = np.asarray(cfg.home_qpos)
    owner.reference = ArmTrajectory(cfg, original, np.zeros(7))
    current = original.copy()
    current[0] += .2
    owner.status['qpos'] = current
    owner.status['qvel'] = .2 if moving else 0
    owner.status['timestamp_ns'] = now
    io.publish_goal(RobotCommand(arm_qpos=current), expires_ns=now+1_000_000_000)
    if moving:
        with pytest.raises(RuntimeError, match='moved after standstill'):
            owner.tick()
        assert not sent
    else:
        owner.tick()
        np.testing.assert_allclose(sent[0], current, atol=1e-12, rtol=0)
        # 后续反馈偏移不能把已建立的规划参考每周期重置为实测。
        owner.status['qpos'] = original
        owner.tick()
        np.testing.assert_allclose(sent[1], current, atol=1e-12, rtol=0)


def test_arm_stop_is_confirmed_even_without_hand_commands():
    from dexmani_real.config.xhand import HandConfig
    from dexmani_real.robot.system import Robot
    state = SessionState()
    robot = Robot(ArmConfig(), HandConfig(enabled=False), runtime_state=state)
    def unresolved():
        assert state.stop_signal.is_set()
        raise TimeoutError('arm stop unresolved')
    robot.arm = SimpleNamespace(wait_stopped=unresolved)
    with pytest.raises(TimeoutError, match='arm stop unresolved'):
        robot.stop()


def test_cancel_cleanup_interrupt_latches_emergency():
    state = SessionState()
    arm = ArmProcess(ArmConfig(), runtime_state=state)
    arm.resume = lambda *a, **kw: None
    arm.wait_stopped = lambda **kw: (_ for _ in ()).throw(KeyboardInterrupt('during stop'))
    try:
        with pytest.raises(KeyboardInterrupt, match='during stop'):
            arm.request_operation(PrepareRequest(time.monotonic_ns()+1_000_000_000),
                                  service=lambda: None, abort=lambda: True,
                                  cancel_epoch=state.cancel_epoch)
        assert state.estop_request and arm.io.closing.is_set()
        assert not arm.child_pipe.poll()
    finally:
        arm.pipe.close()
        arm.child_pipe.close()

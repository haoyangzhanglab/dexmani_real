"""真实资源边界；只替换 SDK、终端和播放器 I/O。"""
import time
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import PrepareRequest
from dexmani_real.runtime.keyboard import Keyboard, KeyEvent
from dexmani_real.runtime.state import SessionState


@pytest.mark.parametrize('stage', ['stop', 'join', 'restore'])
def test_keyboard_shutdown_interrupt_keeps_first_and_cleans(monkeypatch, stage):
    state = SessionState()
    keyboard = Keyboard(get_run_id=lambda: state.run_id, estop_callback=state.request_emergency_stop)
    keyboard.active = True
    keyboard.saved_termios = object()
    calls = []
    def step(name):
        calls.append(name)
        if name == stage:
            raise KeyboardInterrupt()
        if name == 'restore':
            raise OSError('restore failed')
    keyboard.listener = SimpleNamespace(stop=lambda: step('stop'), join=lambda **kw: step('join'))
    monkeypatch.setattr('dexmani_real.runtime.keyboard.restore_terminal_echo', lambda _: step('restore'))
    with pytest.raises(KeyboardInterrupt):
        keyboard.stop()
    assert calls == ['stop', 'join', 'restore']
    assert state.estop_request and state.emergency_signal.is_set()
    assert keyboard.saved_termios is None
    keyboard.stop()


@pytest.mark.parametrize('stage', ['reply', 'recv', 'returned', 'stop'])
@pytest.mark.parametrize('cancel', [False, True])
def test_arm_rollback_interrupt_latches_and_preserves_owner(stage, cancel):
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    real_pipe = arm.pipe
    arm.process = Mock()
    arm.process.is_alive.return_value = True
    arm.io.connected.set()
    arm.check_error = Mock()
    primary = OperationCancelled('cancel') if cancel else OSError('primary')
    arm.pipe = Mock()
    arm.pipe.poll.side_effect = [primary, KeyboardInterrupt()] if stage == 'reply' else [primary, True]
    arm.pipe.recv.return_value = KeyboardInterrupt() if stage == 'returned' else None
    if stage == 'recv':
        arm.pipe.recv.side_effect = KeyboardInterrupt()
    arm.wait_stopped = Mock(side_effect=[None, KeyboardInterrupt() if stage == 'stop' else None])
    try:
        with pytest.raises(BaseException) as caught:
            arm.request_operation(PrepareRequest(state.run_id, time.monotonic_ns() + 10**9), service=lambda: None, abort=lambda: False)
        assert isinstance(caught.value, KeyboardInterrupt) if cancel else caught.value is primary
        assert state.estop_request and state.emergency_signal.is_set()
        assert arm.io.closing.is_set()
        assert arm.wait_stopped.call_count == 2
    finally:
        real_pipe.close()
        arm.child_pipe.close()


def test_worker_own_timeout_is_failure_not_pending():
    state = SessionState()
    worker = PolicyWorker(None, build_observation=lambda rows: rows)
    error = TimeoutError('native close timeout')
    worker.model = SimpleNamespace(close=Mock(side_effect=error))
    worker.close(state)
    worker.close_future.exception(2)
    with pytest.raises(TimeoutError) as caught:
        worker.wait_closed(0)
    assert caught.value is error


@pytest.mark.parametrize('reject', [False, True])
def test_replay_home_revokes_key_captured_before_enqueue(monkeypatch, reject):
    from dexmani_real.control.homing import HomePathRejected
    from dexmani_real.replay.joint_replay import ReplayResult, ReplayStatus
    from dexmani_real.replay.run import wait_for_home
    state = SessionState()
    state.arm()
    keyboard = Keyboard(get_run_id=lambda: state.run_id)
    keyboard.active = True
    keyboard.listener = Mock(is_alive=lambda: True)
    captured, release = Event(), Event()
    homes = []
    thread = None
    def identity():
        run_id = state.run_id
        captured.set()
        assert release.wait(2)
        return run_id
    keyboard.get_run_id = identity
    def home(*args, **kwargs):
        nonlocal thread
        homes.append(kwargs['run_id'])
        if len(homes) == 1:
            thread = Thread(target=keyboard.handle_key, args=('h', True))
            thread.start()
            assert captured.wait(2)
        if reject:
            raise HomePathRejected('geometry')
    batches = [0]
    poll = keyboard.poll
    def next_batch(**kwargs):
        batches[0] += 1
        if batches[0] == 1:
            return [KeyEvent('h', state.run_id)]
        if batches[0] == 2:
            release.set()
            thread.join(2)
            return [*poll(0), KeyEvent('h', state.run_id)]
        state.request_quit()
        return []
    keyboard.poll = next_batch
    monkeypatch.setattr('dexmani_real.replay.run.home_robot', home)
    outcome = ReplayResult(ReplayStatus.COMPLETED)
    results = []
    try:
        wait_for_home(state, RealConfig(), Mock(), keyboard, outcome, results)
        assert len(homes) == 2 and homes[0] != homes[1]
        assert len(results) == 2 and outcome.successful
    finally:
        release.set()
        if thread is not None:
            thread.join(2)


def test_audio_pending_replacement_and_popen_close_race(monkeypatch):
    from dexmani_real.teleop.audio import AudioFeedback
    entered, release = Event(), Event()
    processes, commands = [], []
    def popen(command, **kwargs):
        commands.append(command)
        proc = Mock(returncode=0)
        proc.poll.return_value = None
        proc.communicate.return_value = (b'', b'')
        processes.append(proc)
        entered.set()
        assert release.wait(2)
        return proc
    monkeypatch.setattr('dexmani_real.teleop.audio.shutil.which', lambda _: '/bin/player')
    monkeypatch.setattr('dexmani_real.teleop.audio.subprocess.Popen', popen)
    audio = AudioFeedback()
    try:
        audio.play('begin')
        assert entered.wait(2)
        audio.play('save')
        audio.play('home')
        assert audio.pending == audio.paths['home']
        # close 在 Popen 返回前发生；不依赖调度 sleep。
        audio.close()
        release.set()
        audio.worker.join(2)
        assert not audio.worker.is_alive()
        assert len(commands) == 1 and commands[0][-1] == str(audio.paths['begin'])
        processes[0].kill.assert_called_once()
        processes[0].communicate.assert_called_once_with(timeout=0.5)
    finally:
        release.set()
        audio.close()


@pytest.mark.parametrize('fresh', [True, False])
def test_home_arm_accepts_existing_fresh_hand_sample(monkeypatch, fresh):
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
    from dexmani_real.runtime.home import home_arm
    config = RealConfig()
    state = SessionState()
    run_id = state.arm()
    clock = [1_000_000_000]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    monkeypatch.setattr(time, 'monotonic', lambda: clock[0] / 1e9)
    monkeypatch.setattr(time, 'sleep', lambda _: clock.__setitem__(0, clock[0] + 10**9))
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    arm['timestamp_ns'] = clock[0] + 1
    hand['timestamp_ns'] = clock[0] - (1 if fresh else 500_000_000)
    arm['qpos'] = config.arm.home_qpos
    def feedback():
        clock[0] += 1
    robot = SimpleNamespace(arm_config=config.arm, hand_config=config.hand,
                            service_feedback=feedback, read_state=lambda: SimpleNamespace(arm=arm, hand=hand),
                            stop=Mock(), home_arm=Mock())
    planner = Mock()
    planner.plan.return_value = np.array([config.arm.home_qpos])
    if fresh:
        home_arm(state, robot, planner, run_id=run_id, cancel_requested=lambda: False)
        planner.set_hand_qpos.assert_called_once()
        robot.home_arm.assert_called_once()
    else:
        with pytest.raises(TimeoutError):
            home_arm(state, robot, planner, run_id=run_id, cancel_requested=lambda: False)
        planner.plan.assert_not_called()


def test_ruckig_preview_commit_and_braking():
    from dexmani_real.robot.xarm7.trajectory import ArmTrajectory
    cfg = RealConfig().arm
    q = np.asarray(cfg.home_qpos)
    trajectory = ArmTrajectory(cfg, q, np.zeros(7))
    goal = q + .05
    first = trajectory.step(goal)
    np.testing.assert_array_equal(trajectory.input.current_position, q)
    np.testing.assert_allclose(trajectory.step(goal), first)
    trajectory.commit()
    np.testing.assert_allclose(trajectory.input.current_position, first)
    for _ in range(100):
        candidate = trajectory.step(hold=True)
        assert np.all(candidate >= trajectory.lower) and np.all(candidate <= trajectory.upper)
        trajectory.commit()
    np.testing.assert_allclose(trajectory.input.current_velocity, 0, atol=1e-10)


@pytest.mark.parametrize('issue', ['generation_busy', 'revoked', 'command_expired'])
def test_arm_rechecks_after_preview_without_commit(monkeypatch, issue):
    from dexmani_real.robot.xarm7.worker import ArmController
    io = SimpleNamespace(take_goal=lambda: {'q_goal': np.zeros(7), 'hold': False}, stop=Event())
    driver = Mock()
    controller = ArmController(RealConfig().arm, io, None, driver)
    controller.reference = Mock()
    controller.command_issue = Mock(side_effect=[None, issue])
    controller.tick()
    controller.reference.step.assert_called_once()
    controller.reference.commit.assert_not_called()
    driver.servo.assert_not_called()
    assert io.stop.is_set() == (issue != 'generation_busy')


def test_worker_close_pending_retains_future():
    state = SessionState()
    entered, release = Event(), Event()
    def close():
        entered.set()
        assert release.wait(2)
    worker = PolicyWorker(None, build_observation=lambda rows: rows)
    worker.model = SimpleNamespace(close=close)
    future = worker.close(state)
    try:
        assert entered.wait(2)
        assert worker.wait_closed(0) is False and worker.close_future is future
    finally:
        release.set()
        assert worker.wait_closed(2) is True


def test_arm_close_interrupt_still_closes_pipes():
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    arm.process = Mock(pid=1, exitcode=0)
    arm.process.join.side_effect = KeyboardInterrupt()
    arm.process.is_alive.return_value = False
    arm.io.stopped.set()
    with pytest.raises(KeyboardInterrupt):
        arm.close()
    assert state.estop_request and arm.is_closed


def test_home_hand_old_sample_cannot_confirm(monkeypatch):
    from dataclasses import replace

    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
    from dexmani_real.runtime.home import home_hand
    config = RealConfig()
    state = SessionState()
    run_id = state.arm()
    clock = [1_000_000_000]
    hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
    hand['timestamp_ns'] = clock[0]
    hand['qpos'] = np.deg2rad(config.hand.home_qpos_deg)
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    monkeypatch.setattr(time, 'sleep', lambda _: clock.__setitem__(0, clock[0] + 10_000_000))
    robot = SimpleNamespace(hand_config=replace(config.hand, home_timeout_s=.03),
                            send_hand_home=Mock(), service_feedback=lambda: None,
                            hand_feedback=SimpleNamespace(snapshot=lambda: hand), stop=Mock())
    with pytest.raises(TimeoutError, match='convergence'):
        home_hand(state, robot, run_id=run_id, cancel_requested=lambda: False)
    robot.send_hand_home.assert_called_once()
    robot.stop.assert_called_once()


def test_audio_busy_plays_only_latest_pending(monkeypatch):
    from dexmani_real.teleop.audio import AudioFeedback
    entered, release, second = Event(), Event(), Event()
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        first = len(commands) == 1
        proc = Mock(returncode=0)
        proc.poll.return_value = 0
        def communicate(**kw):
            if first:
                entered.set()
                assert release.wait(2)
            else:
                second.set()
            return b'', b''
        proc.communicate.side_effect = communicate
        return proc
    monkeypatch.setattr('dexmani_real.teleop.audio.shutil.which', lambda _: '/bin/player')
    monkeypatch.setattr('dexmani_real.teleop.audio.subprocess.Popen', popen)
    audio = AudioFeedback()
    try:
        audio.play('begin')
        assert entered.wait(2)
        audio.play('save')
        audio.play('home_done')
        release.set()
        assert second.wait(2)
    finally:
        release.set()
        audio.close()
    assert [c[-1] for c in commands] == [str(audio.paths[k]) for k in ('begin', 'home_done')]


@pytest.mark.parametrize('missing', [False, True])
def test_audio_unavailable_is_warning(monkeypatch, caplog, missing):
    from dexmani_real.teleop.audio import AudioFeedback
    monkeypatch.setattr('dexmani_real.teleop.audio.shutil.which', lambda _: None if missing else '/bin/player')
    monkeypatch.setattr('dexmani_real.teleop.audio.subprocess.Popen', Mock(side_effect=OSError('player failed')))
    audio = AudioFeedback()
    audio.play('begin')
    if audio.worker is not None:
        audio.worker.join(2)
        assert not audio.worker.is_alive()
    audio.close()
    assert any(record.levelname == 'WARNING' for record in caplog.records)


def test_partial_hand_connection_still_closes_both_owners(monkeypatch):
    from dexmani_real.robot.system import Robot
    config, state = RealConfig(), SessionState()
    arm = Mock(is_closed=True)
    hand = Mock(control=object())
    failure = OSError('hand connection failed')
    hand.connect.side_effect = failure
    hand.disconnect.side_effect = lambda **kw: setattr(hand, 'control', None)
    monkeypatch.setattr('dexmani_real.robot.xarm7.process.ArmProcess', Mock(return_value=arm))
    monkeypatch.setattr('dexmani_real.robot.xhand.driver.XHand', Mock(return_value=hand))
    robot = Robot(config.arm, config.hand, runtime_state=state)
    with pytest.raises(OSError) as caught:
        robot.connect()
    assert caught.value is failure
    robot.close()
    arm.close.assert_called_once_with()
    hand.disconnect.assert_called_once_with(runtime_state=state)
    assert robot.arm is None and robot.hand is None


def test_source_cache_does_not_refresh_source_time(monkeypatch):
    from dexmani_real.robot.xhand.feedback import XHandFeedback, XHandState
    from dexmani_real.runtime.observation import read_observation
    cache = XHandFeedback()
    cache.consume(XHandState(np.zeros(12), np.zeros((5, 3)), np.zeros((5, 120, 3)), 100))
    robot = SimpleNamespace(read_state=lambda: SimpleNamespace(arm=None, hand=cache.snapshot()))
    ticks = iter([200, 210, 300, 310])
    monkeypatch.setattr(time, 'monotonic_ns', lambda: next(ticks))
    first, second = read_observation(None, robot), read_observation(None, robot)
    assert first.control_timestamp_ns == 200 and second.control_timestamp_ns == 300
    for name in ('timestamp_ns', 'tactile_aggregate_timestamp_ns', 'tactile_dense_timestamp_ns'):
        assert first.hand[name][0] == second.hand[name][0] == 100


def test_ruckig_rejects_outward_braking_extremum():
    from dexmani_real.robot.xarm7.trajectory import ArmTrajectory
    cfg = RealConfig().arm
    q = np.asarray(cfg.home_qpos).copy()
    q[0] = cfg.joint_limit_upper[0] - 1e-6
    v = np.zeros(7)
    v[0] = 1.
    trajectory = ArmTrajectory(cfg, q, v)
    with pytest.raises(ValueError, match='extremum'):
        trajectory.step(hold=True)


def test_replay_stop_boundary_revokes_late_home(monkeypatch):
    from dexmani_real.replay.joint_replay import JointReplay, ReplayStatus
    from dexmani_real.replay.run import wait_for_home
    state = SessionState()
    run_id = state.arm()
    keyboard = Keyboard(get_run_id=lambda: state.run_id)
    keyboard.active = True
    keyboard.listener = Mock(is_alive=lambda: True)
    captured, release = Event(), Event()
    def identity():
        identity = state.run_id
        captured.set()
        assert release.wait(2)
        return identity
    keyboard.get_run_id = identity
    thread = Thread(target=keyboard.handle_key, args=('h', True))
    def stop():
        thread.start()
        assert captured.wait(2)
    robot = Mock(stop=stop)
    replay = JointReplay(state, RealConfig(), keyboard, robot=robot)
    replay.admit_replay = lambda _: (ReplayStatus.COMPLETED, '', None)
    replay.replay_rows = lambda *args: (ReplayStatus.COMPLETED, '')
    class Trajectory:
        fps = 16
        def __len__(self):
            return 0
    monkeypatch.setattr('dexmani_real.replay.joint_replay.ActionMapper.for_mode', lambda *a, **kw: None)
    home = Mock()
    monkeypatch.setattr('dexmani_real.replay.run.home_robot', home)
    poll = keyboard.poll
    calls = []
    def poll_home(**kwargs):
        if calls:
            state.request_quit()
            return []
        calls.append(True)
        return poll(0)
    try:
        outcome = replay.run(Trajectory(), run_id=run_id, hand_start_duration_s=0)
        release.set()
        thread.join(2)
        keyboard.poll = poll_home
        wait_for_home(state, RealConfig(), robot, keyboard, outcome, [])
        home.assert_not_called()
        assert outcome.successful
    finally:
        release.set()
        if thread.ident is not None:
            thread.join(2)
        keyboard.stop()


def test_keyboard_start_primary_survives_cleanup_interrupt(monkeypatch):
    import sys
    state = SessionState()
    primary = OSError('listener start failed')
    listener = Mock()
    listener.start.side_effect = primary
    listener.join.side_effect = KeyboardInterrupt()
    keyboard = Keyboard(get_run_id=lambda: state.run_id, estop_callback=state.request_emergency_stop)
    keyboard.callbacks = lambda _: (None, None)
    monkeypatch.setitem(sys.modules, 'pynput', SimpleNamespace(keyboard=SimpleNamespace(Listener=Mock(return_value=listener))))
    monkeypatch.setattr('dexmani_real.runtime.keyboard.suppress_terminal_echo', lambda: object())
    restore = Mock()
    monkeypatch.setattr('dexmani_real.runtime.keyboard.restore_terminal_echo', restore)
    with pytest.raises(OSError) as caught:
        keyboard.start()
    assert caught.value is primary and state.estop_request
    restore.assert_called_once()
    listener.stop.assert_called_once()


@pytest.mark.parametrize('stage', ['stop', 'disconnect', 'fault'])
@pytest.mark.parametrize('existing', [False, True])
def test_owner_interrupt_exposes_emergency_even_after_fault(stage, existing):
    from dexmani_real.robot.xarm7.worker import ArmController
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    driver = Mock()
    owner = ArmController(arm.cfg, arm.io, arm.child_pipe, driver)
    original = b'OSError: original' if existing else b''
    arm.io.fault.value = original
    interrupt = KeyboardInterrupt()
    try:
        if stage == 'stop':
            driver.stop.side_effect = interrupt
            owner.stop()
            assert arm.io.stopped.is_set()
        elif stage == 'fault':
            owner.fault(interrupt)
        else:
            # 取消启动，走真实 run 的 stop/disconnect finally。
            driver.connect.side_effect = OperationCancelled('cancel startup')
            driver.close.side_effect = interrupt
            owner.run(state.run_id, time.monotonic_ns() + 30_000_000_000)
            assert arm.io.stopped.is_set() and arm.child_pipe.closed
            assert arm.io.close_error.value
        assert state.estop_request and state.stop_signal.is_set()
        if existing:
            assert arm.io.fault.value == original
        with pytest.raises(Exception):
            arm.check_error()
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('prior_stop', [False, True])
def test_shared_emergency_still_revokes_parent_and_cannot_rearm(prior_stop):
    from dexmani_real.runtime.state import RunEndReason, SafetyState
    state = SessionState()
    run_id = state.arm()
    state.stop_signal.clear()
    assert state.begin_motion(run_id=run_id) is not None
    if prior_stop:
        state.request_stop()
    state.emergency_signal.set()  # 模拟 owner 先报告急停。
    before = state.run_id
    assert state.estop_request and not state.can_move(run_id=run_id)
    state.request_emergency_stop()
    assert state.run_id > before
    reason = RunEndReason.OPERATOR if prior_stop else RunEndReason.ESTOP
    assert state.run_ended_reason == reason
    state.revoke_motion(SafetyState.DISARMED)
    state.request_emergency_stop()
    assert state.safety_state == SafetyState.DISARMED
    assert state.run_ended_reason == reason
    state.stop_signal.clear()
    assert state.begin_motion(run_id=run_id) is None
    assert state.begin_motion(run_id=state.run_id) is None
    with pytest.raises(OperationCancelled):
        state.arm()


@pytest.mark.parametrize('mode', ['late', 'missed', 'expired', 'error', 'cancel'])
def test_replay_keeps_original_slots_and_submission_times(monkeypatch, mode):
    from dexmani_real.replay.joint_replay import JointReplay, ReplayStatus
    from dexmani_real.replay.trajectory import ReplayBuffer
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
    state = SessionState()
    clock = [1_000_000_000]
    origin = clock[0]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    monkeypatch.setattr(time, 'sleep', lambda delay: clock.__setitem__(0, clock[0] + round(delay * 1e9)))
    row = SimpleNamespace(arm=np.zeros(1, dtype=ARM_STATE_DTYPE), hand=np.zeros(1, dtype=HAND_STATE_DTYPE))
    robot = Mock()
    keyboard = SimpleNamespace(poll=lambda **kw: [])
    replay = JointReplay(state, RealConfig(), keyboard, robot=robot)
    replay.run_id = state.run_id
    replay.capture = ReplayBuffer(2)
    replay.read_feedback = lambda: row
    sends = []
    error = OSError('device failed') if mode == 'error' else OperationCancelled('device lease refused')
    def send(command, **kw):
        sends.append(clock[0])
        clock[0] = origin + (125_000_000 if mode == 'missed' else 63_000_000)
        if mode in ('error', 'cancel'):
            raise error
    robot.send_action.side_effect = send
    def mapped(*args):
        if mode == 'expired':
            clock[0] = origin + 62_500_000
        return SimpleNamespace(arm_qpos=np.zeros(7), hand_qpos=np.zeros(12), rejection_reason=None)
    replay.action_mapper = SimpleNamespace(map_joint=mapped)
    class Trajectory:
        action_arm_joint = np.zeros((2, 7))
        action_hand_joint = np.zeros((2, 12))
        def __len__(self):
            return 2
    if mode == 'late':
        assert replay.replay_rows(Trajectory(), 62_500_000) == (ReplayStatus.COMPLETED, '')
        assert sends == [origin, origin + 63_000_000]
    else:
        with pytest.raises((OSError, OperationCancelled)) as caught:
            replay.replay_rows(Trajectory(), 62_500_000)
        if mode in ('error', 'cancel'):
            assert caught.value is error
        assert len(sends) == (0 if mode == 'expired' else 1)
    assert replay.capture.data['submit_started_ns'][0] == (-1 if mode == 'expired' else origin)
    assert replay.capture.data['submit_completed_ns'][0] == (
        -1 if mode == 'expired' else origin + (125_000_000 if mode == 'missed' else 63_000_000)
    )


@pytest.mark.parametrize('mode', ['success', 'stationary', 'timeout', 'cancel', 'empty', 'away', 'velocity', 'stale'])
def test_home_consumes_one_deadline_with_real_dwell(monkeypatch, mode):
    from dexmani_real.robot.xarm7.protocol import HomeRequest
    from dexmani_real.robot.xarm7.worker import ArmController, _OperationTimeout
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    clock = [1_000_000_000]
    origin = clock[0]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    monkeypatch.setattr(arm.io.stop, 'wait', lambda delay: clock.__setitem__(0, clock[0] + round(delay * 1e9)))
    q = np.array(arm.cfg.home_qpos)
    targets = np.array([q + .02, q + .01, q])
    empty = mode in ('empty', 'away', 'velocity', 'stale')
    if empty:
        targets = np.empty((0, 7))
    current = [q.copy() if empty else targets[0].copy()]
    moves = []
    ready_at = [origin + (16_000_000_000 if mode == 'stationary' else 0)]
    driver = Mock()
    def move(target):
        moves.append(clock[0])
        current[0] = target.copy()
        # 第一段超过旧的距离预算，但仍在总预算内。
        ready_at[0] = clock[0] + (2_000_000_000 if len(moves) == 1 else 0)
    def read():
        if mode == 'cancel' and moves:
            arm.io.stop.set()
        if mode == 'stale':
            clock[0] += 200_000_000
        pos = current[0] + (.1 if mode == 'away' else 0)
        velocity = np.ones(7) if clock[0] < ready_at[0] or mode == 'velocity' else np.zeros(7)
        return pos, velocity
    driver.read.side_effect = read
    driver.move_home_milestone.side_effect = move
    owner = ArmController(arm.cfg, arm.io, arm.child_pipe, driver)
    deadline = origin + (2_700_000_000 if mode == 'timeout' else 4_000_000_000)
    if mode == 'stationary':
        deadline = origin + 20_000_000_000
    try:
        owner.operation(HomeRequest(state.run_id, targets, q, deadline))
        assert arm.pipe.poll(1)
        reply = arm.pipe.recv()
        if mode in ('success', 'stationary', 'empty'):
            assert reply is None
            assert clock[0] < deadline and clock[0] - origin >= 300_000_000
            assert len(moves) == (0 if empty else 2)
        elif mode in ('timeout', 'stale'):
            assert isinstance(reply, _OperationTimeout)
        elif mode == 'cancel':
            assert isinstance(reply, OperationCancelled) and len(moves) == 1
        else:
            assert isinstance(reply, RuntimeError)
        if mode == 'timeout':
            assert clock[0] == deadline and len(moves) == 2
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('stage', ['startup', 'prepare'])
def test_standstill_outside_home_retains_independent_budget(monkeypatch, stage):
    from dexmani_real.config.xarm7 import HOME_CONVERGENCE_TIMEOUT_S
    from dexmani_real.robot.xarm7.worker import ArmController, _OperationTimeout
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    clock = [1_000_000_000]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    monkeypatch.setattr(arm.io.stop, 'wait', lambda delay: clock.__setitem__(0, clock[0] + round(delay * 1e9)))
    driver = Mock()
    driver.read.return_value = (np.array(arm.cfg.home_qpos), np.ones(7))
    owner = ArmController(arm.cfg, arm.io, arm.child_pipe, driver)
    try:
        if stage == 'startup':
            owner.run(state.run_id, time.monotonic_ns() + 30_000_000_000)
            assert b'_OperationTimeout' in arm.io.fault.value
            driver.initialize.assert_not_called()
        else:
            owner.operation(PrepareRequest(state.run_id, clock[0] + 30_000_000_000))
            assert isinstance(arm.pipe.recv(), _OperationTimeout)
            assert not owner.prepared_run_id
        assert clock[0] == 1_000_000_000 + int(HOME_CONVERGENCE_TIMEOUT_S * 1e9)
        assert arm.io.stopped.is_set()
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_arm_interrupt_reply_traverses_real_pipe():
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    arm.process = Mock()
    arm.process.is_alive.return_value = True
    arm.io.connected.set()
    arm.io.stopped.set()
    arm.child_pipe.send(KeyboardInterrupt())
    try:
        with pytest.raises(KeyboardInterrupt):
            arm.request_operation(
                PrepareRequest(state.run_id, time.monotonic_ns() + 10**9),
                service=arm.io.stopped.set, abort=lambda: False,
            )
        assert state.estop_request and state.stop_signal.is_set()
        assert not arm.is_closed and not arm.pipe.closed
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_arm_startup_wait_interrupt_latches_before_cleanup(monkeypatch):
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    process = Mock()
    monkeypatch.setattr(arm.ctx, 'Process', lambda **kwargs: process)
    monkeypatch.setattr(arm.io.ready, 'wait', Mock(side_effect=KeyboardInterrupt()))
    stop = arm.request_stop
    def stop_after_interrupt():
        assert state.estop_request and state.stop_signal.is_set()
        stop()
    arm.request_stop = stop_after_interrupt
    try:
        with pytest.raises(KeyboardInterrupt):
            arm.connect()
        assert arm.io.closing.is_set()
        assert arm.process is process and not arm.pipe.closed
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('operation', ['startup', 'home', 'prepare'])
@pytest.mark.parametrize('ending', ['release', 'stop', 'emergency', 'closing', 'generation', 'eof', 'timeout', 'overlap'])
def test_lifecycle_lock_contention(monkeypatch, operation, ending):
    from dexmani_real.robot.xarm7.protocol import HomeRequest
    from dexmani_real.robot.xarm7.worker import ArmController, _OperationTimeout
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    run_id = state.run_id
    clock = [1_000_000_000]
    deadline = clock[0] + 100_000_000
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    driver = Mock()
    driver.read.return_value = (np.array(arm.cfg.home_qpos), np.zeros(7))
    owner = ArmController(arm.cfg, arm.io, arm.child_pipe, driver)
    owner.wait_settled = lambda check_abort, **kw: check_abort()
    ready = arm.io.ready.set
    def ready_then_exit():
        ready()
        arm.io.closing.set()
    monkeypatch.setattr(arm.io.ready, 'set', ready_then_exit)
    driver.connect.side_effect = lambda **kw: kw['check_abort']()
    # 真锁由另一线程持有；受控时钟推进超过一个伺服周期后才释放/撤权。
    held, release = Event(), Event()
    lock = state.run_signal.get_lock()
    def holder():
        with lock:
            held.set()
            assert release.wait(3)
    thread = Thread(target=holder)
    thread.start()
    assert held.wait(2)
    acquire = lock.acquire
    calls = []
    def contended(*args, **kwargs):
        calls.append(kwargs.get('timeout', 0))
        if not release.is_set():
            assert not driver.connect.called and not driver.enter_mode.called
        clock[0] += 5_000_000
        if len(calls) == 4:
            if ending == 'generation':
                state.run_signal.get_obj().value += 1
            elif ending in ('stop', 'emergency', 'closing'):
                getattr(arm.io, ending).set()
            elif ending == 'eof':
                arm.pipe.close()
            elif ending == 'overlap':
                arm.pipe.send(PrepareRequest(run_id, deadline))
            if ending in ('release', 'generation'):
                release.set()
                thread.join(2)
        return acquire(*args, **kwargs)
    monkeypatch.setattr(lock, 'acquire', contended)
    try:
        if operation == 'startup':
            owner.run(run_id, deadline)
            if ending == 'release':
                driver.initialize.assert_called_once()
                driver.read.assert_called_once()
            else:
                driver.connect.assert_not_called()
        else:
            request = (PrepareRequest(run_id, deadline) if operation == 'prepare' else
                       HomeRequest(run_id, np.empty((0, 7)), np.array(arm.cfg.home_qpos), deadline))
            owner.operation(request)
            if ending != 'eof':
                reply = arm.pipe.recv()
                if ending == 'release':
                    assert reply is None
                else:
                    expected = _OperationTimeout if ending == 'timeout' else RuntimeError if ending == 'overlap' else OperationCancelled
                    assert isinstance(reply, expected)
        if ending not in ('release',):
            driver.stop.assert_called_once()
        if ending not in ('timeout', 'overlap'):
            assert not arm.io.fault.value
        if ending == 'timeout':
            assert clock[0] == deadline
        assert len(calls) >= 4
    finally:
        release.set()
        thread.join(2)
        arm.pipe.close()
        arm.child_pipe.close()


def test_servo_authorization_does_not_wait_on_busy_lock(monkeypatch):
    from dexmani_real.robot.xarm7.worker import ArmController
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    owner = ArmController(arm.cfg, arm.io, arm.child_pipe, Mock())
    acquire = Mock(return_value=False)
    monkeypatch.setattr(state.run_signal.get_lock(), 'acquire', acquire)
    try:
        assert owner.authorized(1) is None
        acquire.assert_called_once_with(timeout=0.0)
        owner.driver.servo.assert_not_called()
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('stage', ['startup', 'prepare'])
@pytest.mark.parametrize('budget_ns', [10_000_000_000, 30_000_000_000])
def test_standstill_lock_wait_uses_stage_deadline(monkeypatch, stage, budget_ns):
    from dexmani_real.config.xarm7 import HOME_CONVERGENCE_TIMEOUT_S
    from dexmani_real.robot.xarm7.worker import ArmController, _OperationTimeout
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    clock = [1_000_000_000]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    owner = ArmController(arm.cfg, arm.io, arm.child_pipe, Mock())
    expected_deadline = clock[0] + min(budget_ns, int(HOME_CONVERGENCE_TIMEOUT_S * 1e9))
    def settle(check_abort, *, deadline_ns, **kw):
        assert deadline_ns == expected_deadline
        clock[0] = deadline_ns - 1_000_000
        def busy(**kw):
            clock[0] += round(kw['timeout'] * 1e9)
            return False
        monkeypatch.setattr(state.run_signal.get_lock(), 'acquire', busy)
        check_abort()
        pytest.fail('expired stage authorized')
    owner.wait_settled = settle
    try:
        if stage == 'startup':
            owner.run(state.run_id, clock[0] + budget_ns)
            owner.driver.initialize.assert_not_called()
        else:
            owner.operation(PrepareRequest(state.run_id, clock[0] + budget_ns))
            assert isinstance(arm.pipe.recv(), _OperationTimeout)
            assert owner.prepared_run_id == 0
        assert clock[0] == expected_deadline
        owner.driver.stop.assert_called_once()
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('boundary', ['revoked', 'expired'])
def test_parent_startup_passes_identity_and_deadline(monkeypatch, boundary):
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    clock = [1_000_000_000]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    captured = {}
    process = Mock()
    def create(**kwargs):
        captured.update(kwargs)
        return process
    def start():
        assert captured['args'][3] == 1
        if boundary == 'revoked':
            state.request_stop()
        else:
            clock[0] = captured['args'][4]
        arm.io.ready.set()
    process.start.side_effect = start
    monkeypatch.setattr(arm.ctx, 'Process', create)
    try:
        with pytest.raises(OperationCancelled if boundary == 'revoked' else TimeoutError):
            arm.connect()
        assert arm.io.closing.is_set()
        assert arm.process is process
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('case', ['ready', 'cancel', 'fault', 'timeout', 'stale'])
def test_initial_feedback_callback_boundaries(monkeypatch, case):
    from dexmani_real.robot.commands import RobotError
    from dexmani_real.robot.system import Robot
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    cfg = RealConfig()
    robot = Robot(cfg.arm, cfg.hand, runtime_state=SessionState())
    clock = [1_000_000_000]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    robot.check = Mock(side_effect=AssertionError('duplicate internal check'))
    frame = np.zeros(1, dtype=ARM_STATE_DTYPE)
    frame['timestamp_ns'] = clock[0]
    events = []
    error = OSError('real feedback fault')
    def cancel():
        events.append('check')
        if len(events) > 1 and case == 'fault':
            raise error
        return len(events) > 1 and case == 'cancel'
    def read():
        events.append('read')
        if case == 'timeout':
            clock[0] += 100_000_000_000
        if case == 'stale':
            frame['timestamp_ns'] = 0
        return SimpleNamespace(arm=frame, hand=None)
    robot.read_state = read
    robot.service_feedback = Mock(side_effect=error)
    if case == 'ready':
        assert robot._read_initial_state(cancel_requested=cancel).arm is frame
    else:
        expected = OperationCancelled if case == 'cancel' else RobotError if case == 'timeout' else OSError
        with pytest.raises(expected) as caught:
            robot._read_initial_state(cancel_requested=cancel)
        if case in ('fault', 'stale'):
            assert caught.value is error
    assert events[:3] == ['check', 'read', 'check']
    robot.check.assert_not_called()


@pytest.mark.parametrize('cancel', ['none', 'stop', 'emergency'])
def test_tactile_reset_accepts_own_stop_but_observes_operator(cancel):
    from dexmani_real.robot.system import Robot
    cfg = RealConfig()
    state = SessionState()
    run_id = state.arm()
    robot = Robot(cfg.arm, cfg.hand, runtime_state=state)
    robot.hand = Mock()
    def reset(*, cancel_requested):
        assert state.stop_signal.is_set()
        assert not cancel_requested()
        if cancel == 'stop':
            state.request_stop()
        elif cancel == 'emergency':
            state.request_emergency_stop()
    robot.hand.reset_tactile_sensors.side_effect = reset
    def initial(**kwargs):
        if kwargs['cancel_requested']():
            raise OperationCancelled('reset cancelled')
    robot._read_initial_state = Mock(side_effect=initial)
    if cancel == 'none':
        robot.reset_tactile_sensors(run_id=run_id, cancel_requested=lambda: False)
        robot._read_initial_state.assert_called_once()
    else:
        with pytest.raises(OperationCancelled):
            robot.reset_tactile_sensors(run_id=run_id, cancel_requested=lambda: False)


@pytest.mark.parametrize('stage', ['startup', 'prepare'])
def test_sdk_timeout_remains_permanent_device_fault(stage):
    from dexmani_real.robot.xarm7.worker import ArmController
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    driver = Mock()
    error = TimeoutError('SDK timeout')
    driver.connect.side_effect = error
    driver.enter_mode.side_effect = error
    owner = ArmController(arm.cfg, arm.io, arm.child_pipe, driver)
    try:
        if stage == 'startup':
            owner.run(state.run_id, time.monotonic_ns() + 10**9)
        else:
            owner.operation(PrepareRequest(state.run_id, time.monotonic_ns() + 10**9))
            assert type(arm.pipe.recv()) is TimeoutError
        assert b'TimeoutError: SDK timeout' in arm.io.fault.value
        driver.stop.assert_called_once()
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def test_owner_rejects_startup_identity_captured_before_revocation():
    from dexmani_real.robot.xarm7.worker import ArmController
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    expected = state.run_id
    state.run_signal.value += 1
    owner = ArmController(arm.cfg, arm.io, arm.child_pipe, Mock())
    try:
        owner.run(expected, time.monotonic_ns() + 10**9)
        owner.driver.connect.assert_not_called()
        owner.driver.stop.assert_called_once()
        owner.driver.close.assert_called_once()
        assert not arm.io.fault.value
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('reason', [None, 'collision', 'final_eef_workspace'])
@pytest.mark.parametrize('hand_enabled', [False, True])
def test_projected_joint_still_checks_endpoint(reason, hand_enabled):
    from dexmani_real.control.actions import ActionMapper
    from dexmani_real.control.constraints import MotionConstraints
    constraints = MotionConstraints.__new__(MotionConstraints)
    constraints.limits = np.tile([-1., 1.], (7, 1))
    constraints.collision_model = Mock()
    constraints.collision_model.rejection.return_value = 'collision' if reason == 'collision' else None
    constraints.collision_model.eef_position.return_value = np.full(3, 2. if reason == 'final_eef_workspace' else 0.)
    constraints.workspace_with_margin = np.tile([-1., 1.], (3, 1))
    mapper = ActionMapper(constraints, hand_enabled=hand_enabled, hand_lower_rad=np.zeros(12),
                          hand_upper_rad=np.ones(12), home_hand_qpos=np.full(12, .2))
    hand = np.full(12, .4) if hand_enabled else None
    result = mapper.map_joint(np.full(7, 2.), hand, np.zeros(7))
    assert result.rejection_reason == reason
    assert np.all(result.arm_qpos <= 1) and np.all(result.arm_qpos >= -1)
    np.testing.assert_array_equal(constraints.collision_model.set_hand_qpos.call_args.args[0],
                                  hand if hand_enabled else mapper.home_hand_qpos)
    assert constraints.rejection(np.full(7, 2.), collision_hand_qpos=np.zeros(12)) == 'operational_limits'


@pytest.mark.parametrize('message', ['eof', 'overlap'])
def test_lifecycle_observes_pipe_arrival_while_acquiring_lock(monkeypatch, message):
    from dexmani_real.robot.xarm7.worker import ArmController
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    owner = ArmController(arm.cfg, arm.io, arm.child_pipe, Mock())
    request = PrepareRequest(state.run_id, time.monotonic_ns() + 10**9)
    lock = state.run_signal.get_lock()
    acquire = lock.acquire
    def arrive(**kwargs):
        if message == 'eof':
            arm.pipe.close()
        else:
            arm.pipe.send(request)
        return acquire(**kwargs)
    monkeypatch.setattr(lock, 'acquire', arrive)
    try:
        owner.operation(request)
        owner.driver.enter_mode.assert_not_called()
        owner.driver.stop.assert_called_once()
        if message == 'eof':
            assert arm.io.closing.is_set() and not arm.io.fault.value
        else:
            assert isinstance(arm.pipe.recv(), RuntimeError)
            assert b'overlapping' in arm.io.fault.value
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


@pytest.mark.parametrize('ending', ['release', 'generation', 'stop', 'emergency', 'closing', 'timeout', 'fault'])
def test_parent_ready_generation_wait_is_cancellable(monkeypatch, ending):
    from dexmani_real.robot.commands import RobotError
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    process = Mock()
    monkeypatch.setattr(arm.ctx, 'Process', lambda **kwargs: process)
    clock = [1_000_000_000]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    lock = state.run_signal.get_lock()
    acquired, release = Event(), Event()
    def hold():
        with lock:
            acquired.set()
            assert release.wait(3)
            if ending == 'generation':
                state.run_signal.get_obj().value += 1
    thread = Thread(target=hold)
    def start():
        thread.start()
        assert acquired.wait(2)
        arm.io.ready.set()
    process.start.side_effect = start
    waits = []
    def wait(delay):
        waits.append(delay)
        assert 0 < delay <= .005
        if ending in ('release', 'generation'):
            release.set()
            thread.join(2)
        elif ending == 'timeout':
            clock[0] += 30_000_000_000
        elif ending == 'fault':
            arm.io.fault.value = b'OSError: startup SDK failure'
            arm.io.stop.set()
        else:
            getattr(arm.io, ending).set()
    monkeypatch.setattr(arm.io.stop, 'wait', wait)
    try:
        if ending == 'release':
            arm.connect()
            assert not arm.io.closing.is_set()
        else:
            expected = TimeoutError if ending == 'timeout' else RobotError if ending == 'fault' else OperationCancelled
            with pytest.raises(expected):
                arm.connect()
            assert arm.io.closing.is_set() and arm.io.stop.is_set()
        assert len(waits) == 1
        assert arm.process is process
    finally:
        release.set()
        thread.join(2)
        arm.pipe.close()
        arm.child_pipe.close()


def test_process_start_interrupt_preserves_handle_and_signals(monkeypatch):
    state = SessionState()
    arm = ArmProcess(RealConfig().arm, runtime_state=state)
    process = Mock()
    process.start.side_effect = KeyboardInterrupt()
    monkeypatch.setattr(arm.ctx, 'Process', lambda **kwargs: process)
    try:
        with pytest.raises(KeyboardInterrupt):
            arm.connect()
        assert state.estop_request and arm.io.stop.is_set() and arm.io.closing.is_set()
        assert arm.process is process
        assert not arm.pipe.closed and not arm.child_pipe.closed
    finally:
        arm.pipe.close()
        arm.child_pipe.close()

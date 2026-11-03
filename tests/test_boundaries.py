"""真实资源边界；只替换 SDK、终端和播放器 I/O。"""
import time
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.robot.xarm7.process import ArmProcess
from dexmani_real.robot.xarm7.protocol import PrepareRequest
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.runtime.keyboard import Keyboard, KeyEvent
from dexmani_real.runtime.state import SessionState
from dexmani_real.deployment.inference import PolicyWorker


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
    from dexmani_real.replay.run import wait_for_home
    from dexmani_real.replay.joint_replay import ReplayResult, ReplayStatus
    from dexmani_real.control.homing import HomePathRejected
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
    from dexmani_real.runtime.home import home_arm
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
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
    from dexmani_real.runtime.home import home_hand
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
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
            owner.run()
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
    from dexmani_real.runtime.state import SafetyState, RunEndReason
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
            owner.run()
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

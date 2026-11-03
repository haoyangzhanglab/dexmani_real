"""期限和输入身份竞争；时钟与 Event 控制调度，不访问设备。"""

import time
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_policy import make_rollout

from dexmani_real.config.real import RealConfig
from dexmani_real.deployment.schedule import ActionChunk
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.state import SessionState
from dexmani_real.teleop.keyboard import KeyboardTeleop


@pytest.mark.parametrize('duration,lease,expected', [
    (.05, None, 'duration_complete'), (4, None, 'prediction_timeout'),
    (2, None, 'duration_complete'), (3, .02, 'prediction_timeout'),
])
def test_deadline_reason(tmp_path, monkeypatch, duration, lease, expected):
    episode, _ = make_rollout(tmp_path)
    origin = episode.started_ns
    monkeypatch.setattr(time, 'monotonic_ns', lambda: origin)
    episode.run_deadline_ns = origin + int(duration * 1e9)
    episode.target_expires_ns = origin + int(lease * 1e9) if lease else None
    episode.submit(0)
    deadline, _ = episode.budget_limit()
    monkeypatch.setattr(time, 'monotonic_ns', lambda: deadline)
    episode.step()
    assert episode.result.termination_reason == expected
    assert episode.result.normal_termination == (expected == 'duration_complete')
    recorder = episode.recorder = Mock(start_attempted=True)
    episode.finish_recording()
    assert recorder.finish.call_args.kwargs['technical_status'] == (
        'normal' if expected == 'duration_complete' else 'failed')


@pytest.mark.parametrize('mode', ['sync', 'async', 'rtc'])
def test_observation_crosses_duration(tmp_path, monkeypatch, mode):
    import numpy as np
    episode, _ = make_rollout(tmp_path, mode)
    clock = [episode.started_ns]
    episode.run_deadline_ns = clock[0] + 1
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    read = episode.read_observation
    def slow_read():
        clock[0] += 2
        return read()
    episode.read_observation = slow_read
    if mode != 'sync':
        episode.plan = ActionChunk(np.zeros((6, 19)), -2)
    episode.step()
    assert not episode.model.calls
    assert episode.result.termination_reason == 'duration_complete'


@pytest.mark.parametrize('boundary', ['home', 'prepare'])
def test_late_r_cannot_repeat_home(monkeypatch, boundary):
    jog = KeyboardTeleop(RealConfig())
    state = jog.runtime_state = SessionState()
    run_id = state.arm()
    state.stop_signal.clear()
    # 使用生产绑定，不启动监听器或设备。
    monkeypatch.setattr('dexmani_real.teleop.keyboard.validate_robot_config', lambda _: None)
    monkeypatch.setattr('dexmani_real.teleop.keyboard.jog_bounds', lambda _: None)
    monkeypatch.setattr('dexmani_real.teleop.keyboard.SessionState', lambda: state)
    monkeypatch.setattr('dexmani_real.teleop.keyboard.Robot', Mock())
    monkeypatch.setattr('dexmani_real.teleop.keyboard.close_runtime', Mock())
    jog.start = lambda: None
    jog.run_control_loop = lambda: True
    jog.run()
    state.begin_motion(run_id=run_id)
    original = jog.keys.get_run_id
    captured, release = Event(), Event()
    def delayed():
        identity = original()
        captured.set()
        assert release.wait(2)
        return identity
    jog.keys.get_run_id = delayed
    jog.keys.active = True
    jog.keys.listener = Mock()
    thread = Thread(target=jog.keys.handle_key, args=('r', True))
    def pause():
        if thread.ident is None:
            thread.start()
            assert captured.wait(2)
    homes, prepared = [], []
    def home(*args, **kwargs):
        homes.append(kwargs['run_id'])
        if boundary == 'home':
            pause()
    def prepare(run_id):
        prepared.append(run_id)
        if boundary == 'prepare':
            pause()
        state.stop_signal.clear()
    def send(command, **kwargs):
        assert command.run_id == prepared[-1] == state.run_id
        assert state.can_move(run_id=command.run_id)
    jog.robot = SimpleNamespace(prepare_streaming=prepare, send_action=send, check=lambda: None)
    jog.jog = Mock()
    jog.period_ns = 40_000_000
    monkeypatch.setattr('dexmani_real.teleop.keyboard.home_robot', home)
    try:
        jog.run_home(run_id)
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        for event in jog.keys.poll(0):
            jog.run_home(event.run_id)
        assert len(homes) == 1
        jog.keys.get_run_id = original
        jog.keys.handle_key('r', False)
        jog.keys.handle_key('r', True)
        for event in jog.keys.poll(0):
            jog.run_home(event.run_id)
        assert len(homes) == 2
    finally:
        release.set()
        thread.join(2)
        jog.keys.stop()


@pytest.mark.parametrize('stop_method', ['request_stop', 'request_quit', 'request_emergency_stop'])
def test_stop_visible_before_lock_and_begin_refused(stop_method):
    state = SessionState()
    run_id = state.arm()
    state.stop_signal.clear()
    thread = Thread(target=getattr(state, stop_method))
    with state.motion_lock:
        thread.start()
        assert state.stop_signal.wait(2)
        assert state.begin_motion(run_id=run_id) is None
    thread.join(2)
    assert not thread.is_alive()


def test_begin_rejects_old_run_id():
    state = SessionState()
    old = state.arm()
    state.revoke_motion()
    state.stop_signal.clear()
    assert state.begin_motion(run_id=old) is None
    assert state.begin_motion(run_id=state.run_id) is not None


@pytest.mark.parametrize('key', ['h', 't'])
def test_start_intent_blocks_same_batch_and_stop_invalidates(tmp_path, key):
    from test_policy import make_session

    from dexmani_real.runtime.keyboard import KeyEvent
    session, _ = make_session(tmp_path)
    state = session.runtime_state = SessionState()
    run_id = state.arm()
    session.robot.reset_tactile_sensors = Mock()
    selected = session.handle_command_batch([KeyEvent('b', run_id), KeyEvent(key, run_id)], allow_tactile_reset=True)
    assert selected == run_id
    session.robot.reset_tactile_sensors.assert_not_called()
    state.request_stop()
    # 回调不修改主线程意图；主线程消费时拒绝已撤销的代际。
    assert selected == run_id
    def idle():
        state.request_quit()
    session.robot.service_idle = idle
    session.poll_operator = lambda **kwargs: selected
    session.run_control_loop()
    assert session.episode is None


@pytest.mark.parametrize('error', [ValueError('model failed'), RuntimeError('writer failed')])
def test_real_error_precedes_expired_duration(tmp_path, monkeypatch, error):
    episode, _ = make_rollout(tmp_path)
    monkeypatch.setattr(time, 'monotonic_ns', lambda: episode.started_ns)
    episode.run_deadline_ns = episode.started_ns
    if isinstance(error, ValueError):
        episode.model.poll = Mock(side_effect=error)
    else:
        episode.recorder = Mock()
        episode.recorder.check_error.side_effect = error
    with pytest.raises(type(error)) as caught:
        episode.step()
    assert caught.value is error
    assert not episode.result.normal_termination


@pytest.mark.parametrize('boundary', ['home', 'prepare'])
@pytest.mark.parametrize('key', ['q', 'esc'])
def test_jog_stop_during_home_and_prepare(monkeypatch, boundary, key):
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.runtime.state import SafetyState
    jog = KeyboardTeleop(RealConfig())
    state = jog.runtime_state = SessionState()
    run_id = state.arm()
    state.stop_signal.clear()
    state.begin_motion(run_id=run_id)
    jog.keys = Keyboard(get_run_id=jog.running_run_id,
                        estop_callback=state.request_emergency_stop,
                        bindings={'q': state.request_quit})
    jog.keys.active = True
    jog.jog = Mock()
    jog.jog_config = jog.runtime.keyboard_teleop
    jog.period_ns = 40_000_000
    def interrupt():
        jog.keys.handle_key(key, True)
        assert state.stop_signal.is_set()
    def home(*args, **kwargs):
        if boundary == 'home':
            interrupt()
    def prepare(run_id):
        if boundary == 'prepare':
            interrupt()
    jog.robot = Mock()
    jog.robot.prepare_streaming.side_effect = prepare
    monkeypatch.setattr('dexmani_real.teleop.keyboard.home_robot', home)
    with pytest.raises(OperationCancelled):
        jog.run_home(run_id)
    assert state.safety_state != SafetyState.RUNNING
    jog.robot.send_action.assert_not_called()
    assert state.quit_requested == (key == 'q')
    assert state.emergency_signal.is_set() == (key == 'esc')
    jog.keys.stop()


@pytest.mark.parametrize('caller', ['policy', 'replay'])
@pytest.mark.parametrize('stop_method', ['request_stop', 'request_emergency_stop'])
def test_preparation_cannot_reauthorize(tmp_path, caller, stop_method):
    import numpy as np

    from dexmani_real.replay.joint_replay import JointReplay, ReplayStatus
    from dexmani_real.runtime.state import SafetyState
    episode, _ = make_rollout(tmp_path)
    state = episode.runtime_state
    state.safety_state = SafetyState.ARMED
    episode.robot.prepare_streaming = lambda run_id: getattr(state, stop_method)()
    if caller == 'policy':
        episode.start_observation = Mock(return_value=object())
        episode.prepare(lambda: None)
        assert episode.finished
    else:
        replay = JointReplay(state, episode.runtime, None, robot=episode.robot)
        replay.run_id = episode.run_id
        replay.read_feedback = lambda: SimpleNamespace(
            arm={'qpos': np.zeros((1, 7))}, hand={'qpos': np.zeros((1, 12))})
        status, _, _ = replay.admit_replay(SimpleNamespace(
            initial_arm_qpos=np.zeros(7), initial_hand_qpos=np.zeros(12)))
        assert status == ReplayStatus.REJECTED
    assert state.safety_state != SafetyState.RUNNING


def test_successful_rollout_returns_idle_only_after_reclaim(tmp_path, monkeypatch):
    from threading import get_ident

    from test_policy import make_session

    from dexmani_real.deployment.inference import PolicyWorker
    from dexmani_real.runtime.state import RunEndReason, SafetyState
    episode, _ = make_rollout(tmp_path)
    state = episode.runtime_state
    state.safety_state = SafetyState.ARMED
    entered, release = Event(), Event()
    owners = []
    def predict(rows):
        owners.append(get_ident())
        entered.set()
        assert release.wait(2)
    def mark(*args, **kwargs):
        owners.append(get_ident())
    def load():
        mark()
        return SimpleNamespace(warmup=mark, predict=predict, close=mark)
    worker = episode.model = PolicyWorker(load, build_observation=lambda rows: rows)
    worker.load()
    worker.future.result(2)
    worker.poll()
    worker.warmup(())
    worker.future.result(2)
    worker.poll()
    session, _ = make_session(tmp_path)
    session.runtime_state = state
    session.model = worker
    session.num_episodes = None
    session.action_mapper = session.camera_snapshot = None
    session.robot.service_idle = lambda: None
    session.keyboard = Mock()
    def prepare(poll):
        assert state.begin_motion(run_id=episode.run_id) is not None
        worker.predict(())
        assert entered.wait(2)
        episode.stop('operator', run_end_reason=RunEndReason.OPERATOR)
    episode.prepare = prepare
    def finish():
        assert session.episode is episode and worker.future is not None
        release.set()
        worker.future.result(2)
    episode.finish_recording = finish
    def poll(**kwargs):
        if session.episodes:
            assert session.episode is None and worker.future is None
            state.request_quit()
        return episode.run_id
    session.poll_operator = poll
    factory = Mock(return_value=episode)
    monkeypatch.setattr('dexmani_real.deployment.session.Rollout', factory)
    try:
        session.run_control_loop()
        factory.assert_called_once()
        assert session.episode is None and worker.future is None
    finally:
        release.set()
        worker.close(state)
        assert worker.wait_closed(2) is True
    assert len(owners) == 4 and len(set(owners)) == 1
    assert owners[0] != get_ident()


@pytest.mark.parametrize('ordering', ['duration', 'wait', 'equal'])
def test_sync_next_prediction_after_executed_chunk(tmp_path, monkeypatch, ordering):
    import numpy as np
    episode, sent = make_rollout(tmp_path)
    clock = [episode.started_ns]
    monkeypatch.setattr(time, 'monotonic_ns', lambda: clock[0])
    episode.step()
    episode.model.result = np.zeros((6, 19))
    for _ in range(3):
        clock[0] += episode.dt_ns
        episode.step()
    assert len(sent) == 3 and episode.plan is None
    clock[0] += episode.dt_ns
    lease = episode.target_expires_ns
    episode.run_deadline_ns = (
        clock[0] + 1 if ordering == 'duration' else lease + (1 if ordering == 'wait' else 0))
    episode.step()
    assert len(episode.model.calls) == 2
    assert not episode.model.calls[-1][1]['reset_episode']
    clock[0] = min(lease, episode.run_deadline_ns)
    episode.step()
    assert episode.result.termination_reason == (
        'prediction_timeout' if ordering == 'wait' else 'duration_complete')


def test_operator_stop_precedes_elapsed_budget(tmp_path, monkeypatch):
    episode, _ = make_rollout(tmp_path)
    monkeypatch.setattr(time, 'monotonic_ns', lambda: episode.started_ns)
    episode.run_deadline_ns = episode.started_ns
    episode.runtime_state.request_stop()
    episode.step()
    assert episode.result.termination_reason == 'operator'


@pytest.mark.parametrize('events', [('busy', 'fresh'), ('old', 'fresh'), ('fresh',)])
def test_jog_consumes_fresh_r_after_invalid_events(monkeypatch, events):
    from dexmani_real.runtime.keyboard import KeyEvent

    jog = KeyboardTeleop(RealConfig())
    state = jog.runtime_state = SessionState()
    run_id = state.arm()
    state.stop_signal.clear()
    state.begin_motion(run_id=run_id)
    jog.keys = Keyboard(get_run_id=jog.running_run_id)
    jog.keys.active = True
    jog.keys.listener = Mock()
    jog.keys.handle_key('w', True)
    identities = {'busy': None, 'old': run_id - 1, 'fresh': run_id}
    jog.keys.events.extend(KeyEvent('r', identities[event]) for event in events)
    jog.tick_due_ns = 100
    jog.period_ns = 40
    jog.robot = Mock()
    jog.jog = Mock()
    home = Mock(side_effect=lambda identity: state.request_quit())
    jog.run_home = home
    monkeypatch.setattr(time, 'monotonic_ns', lambda: 100)
    try:
        assert jog.run_control_loop()
        home.assert_called_once_with(run_id)
        assert 'w' in jog.keys.snapshot()
        jog.jog.step.assert_not_called()
    finally:
        jog.keys.stop()


def test_stop_between_start_selection_and_admission(tmp_path, monkeypatch):
    from test_policy import make_session

    session, _ = make_session(tmp_path)
    state = session.runtime_state = SessionState()
    run_id = state.arm()
    can_move = state.can_move
    def stop_after_check(**kwargs):
        allowed = can_move(**kwargs)
        state.request_stop()
        return allowed
    with monkeypatch.context() as patch:
        patch.setattr(state, 'can_move', stop_after_check)
        from dexmani_real.runtime.keyboard import KeyEvent
        selected = session.handle_command_batch([KeyEvent("b", run_id)])
    assert selected == run_id
    session.poll_operator = lambda **kwargs: selected
    # 第一轮消费旧意图，第二轮退出；不能构造任何 Rollout。
    calls = [0]
    def idle():
        calls[0] += 1
        if calls[0] == 2:
            state.request_quit()
    session.robot.service_idle = idle
    factory = Mock(side_effect=AssertionError('revoked start admitted'))
    monkeypatch.setattr('dexmani_real.deployment.session.Rollout', factory)
    session.run_control_loop()
    factory.assert_not_called()
    assert session.episode is None

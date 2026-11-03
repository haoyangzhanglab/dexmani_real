from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.state import SessionState


def test_busy_press_requires_release_and_new_press():
    keyboard = Keyboard()
    keyboard.active = True
    keyboard.handle_key('b', True)
    keyboard.allow_start()
    keyboard.handle_key('b', True)
    assert keyboard.events == keyboard.events.__class__()
    keyboard.handle_key('b', False)
    keyboard.handle_key('b', True)
    assert [event.key for event in keyboard.events] == ['b']


def test_cancel_between_poll_and_admission_prevents_start():
    state = SessionState()
    keyboard = Keyboard(bindings={'s': state.request_stop})
    keyboard.active = True
    keyboard.allow_start()
    keyboard.handle_key('s', True)
    assert keyboard.accept_operation(state) is None
    assert state.stop_signal.is_set()


def test_listener_pending_is_reported_on_repeated_stop():
    joins = []
    keyboard = Keyboard()
    keyboard.active = True
    keyboard.listener = SimpleNamespace(ident=1, stop=lambda: None, join=lambda timeout: joins.append(timeout),
                                        is_alive=lambda: True)
    for _ in range(2):
        with pytest.raises(TimeoutError, match='listener remains active'):
            keyboard.stop()
    assert len(joins) == 2


@pytest.mark.parametrize('outcome', ['none', 'task_timeout', 'pending'])
def test_model_completion_is_distinct_from_pending(outcome):
    worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
    future = worker.future = Future()
    try:
        if outcome == 'none':
            future.set_result(None)
            worker.discard_pending(0)
        else:
            if outcome == 'task_timeout':
                future.set_exception(TimeoutError('task failed'))
            with pytest.raises(TimeoutError):
                worker.discard_pending(0)
        assert (worker.future is future) == (outcome == 'pending')
    finally:
        worker.executor.shutdown(wait=False)


def test_model_cleanup_deadline_is_not_restarted(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr('dexmani_real.deployment.inference.time.monotonic', lambda: clock[0])
    worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
    worker.future = Future()
    worker.close_future = Future()
    try:
        with pytest.raises(TimeoutError):
            worker.discard_pending(0)
        clock[0] += 10
        assert not worker.wait_closed(65)
        assert worker.cleanup_deadline == 100
    finally:
        worker.executor.shutdown(wait=False)


def test_busy_admission_and_cancellation_preserve_operator_intent():
    state = SessionState()
    keyboard = Keyboard(bindings={'s': state.request_stop})
    keyboard.active = True
    keyboard.allow_start()
    epoch = keyboard.accept_operation(state)
    # 内部准备停止不应成为一次用户取消，或提前占据结束原因。
    state.stop_signal.set()
    assert state.cancel_epoch == epoch
    state.stop_signal.clear()
    assert state.cancel_epoch == epoch and not state.stop_signal.is_set()
    keyboard.handle_key('s', True)
    assert [event.key for event in keyboard.events] == ["s"]
    assert state.cancel_epoch != epoch


@pytest.mark.parametrize('late_discard', [False, True])
def test_discard_intent_freezes_before_writer_wait(late_discard):
    from dexmani_real.teleop.loop import TeleopLoop
    state = SessionState()
    keyboard = Keyboard(bindings={'d': state.request_stop, 'q': state.request_quit})
    keyboard.active = True
    loop = TeleopLoop.__new__(TeleopLoop)
    loop.keyboard, loop.runtime_state = keyboard, state
    loop.pending_termination_reason = 'quit'
    loop.termination_details = []
    loop.discard_requested = False
    loop.audio = SimpleNamespace(play=lambda sound: None)
    decisions = []
    def finish(**kwargs):
        decisions.append(kwargs['save'])
        if late_discard:
            keyboard.handle_key('d', True)
        return None
    loop.recorder = SimpleNamespace(finish=finish, recording_truncated_reason=None)
    if not late_discard:
        keyboard.handle_key('d', True)
        loop.discard_requested = True  # 主线程已接纳 D。
    keyboard.handle_key('q', True)
    loop.finish_recording()
    assert decisions == [late_discard]
    assert state.quit_requested


def test_fast_stop_and_event_publish_are_one_admission_boundary():
    import threading
    state = SessionState()
    paused, release = threading.Event(), threading.Event()
    def stop():
        paused.set()
        assert release.wait(2)
        state.request_stop()
    keyboard = Keyboard(bindings={'s': stop})
    keyboard.active = True
    keyboard.allow_start()
    callback = threading.Thread(target=keyboard.handle_key, args=('s', True))
    accepted = []
    admission = threading.Thread(target=lambda: accepted.append(keyboard.accept_operation(state)))
    callback.start()
    try:
        assert paused.wait(2)
        admission.start()
    finally:
        release.set()
        callback.join(2)
        admission.join(2)
    assert not callback.is_alive() and not admission.is_alive()
    assert accepted == [None]
    assert state.stop_signal.is_set()


def test_unstarted_sensor_is_owned_but_not_joined():
    from dexmani_real.runtime.lifecycle import SensorWorkers
    workers = SensorWorkers()
    process = SimpleNamespace(pid=None, name='camera', join=lambda **kwargs: pytest.fail('unstarted join'),
                              start=lambda: (_ for _ in ()).throw(OSError('start failed')))
    with pytest.raises(OSError, match='start failed'):
        workers.start(process, ring=None, may_start=lambda: True)
    workers.shutdown(SessionState())
    assert not workers.processes


def test_failed_archive_worker_is_reported_without_retaining_dead_handle():
    from dexmani_real.runtime.lifecycle import SensorWorkers
    workers = SensorWorkers()
    workers.processes = [SimpleNamespace(name='camera', pid=10, exitcode=1,
                                        join=lambda **kwargs: None, is_alive=lambda: False)]
    with pytest.raises(RuntimeError, match='exitcode=1'):
        workers.shutdown(SessionState())
    assert not workers.processes


@pytest.mark.parametrize('keys', [('x',), ('x', 'b')])
def test_policy_ignores_unbound_keys_before_start_admission(keys):
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.runtime.keyboard import KeyEvent
    session = PolicySession.__new__(PolicySession)
    session.runtime_state = SessionState()
    session.keyboard = Keyboard()
    session.keyboard.allow_start()
    session.episode = None
    epoch = session.handle_command_batch([KeyEvent(key) for key in keys])
    if 'b' in keys:
        assert epoch == session.runtime_state.cancel_epoch
    else:
        assert epoch is None
        # 无关按键不能消耗接纳，随后 B 仍可开始。
        assert session.handle_command_batch([KeyEvent('b')]) == session.runtime_state.cancel_epoch


@pytest.mark.parametrize('key', ['s', 'c', 'd', 'q', 'esc'])
def test_policy_startup_cancel_does_not_wait_for_model(monkeypatch, key):
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.robot.commands import OperationCancelled
    session = PolicySession.__new__(PolicySession)
    session.runtime_state = state = SessionState()
    session.startup_cancel_epoch = state.cancel_epoch
    session.robot = SimpleNamespace(service_feedback=lambda: None)
    session.keyboard = Keyboard(estop_callback=state.request_emergency_stop,
                                bindings={k: state.request_stop for k in ('s', 'c', 'd')})
    session.keyboard.bindings['q'] = state.request_quit
    session.keyboard.active = True
    session.keyboard.listener = SimpleNamespace(is_alive=lambda: True)
    task = Future()
    session.model = SimpleNamespace(poll=lambda: None, future=task)
    now = [0.0]
    monkeypatch.setattr('time.monotonic', lambda: now[0])
    def sleep(seconds):
        now[0] += seconds
        if now[0] >= .01:
            session.keyboard.handle_key(key, True)
    monkeypatch.setattr('time.sleep', sleep)
    with pytest.raises(OperationCancelled):
        session.wait_model(10)
    assert now[0] == .01
    assert session.model.future is task and not task.done()


@pytest.mark.parametrize('failure', [False, True])
def test_sensor_ready_and_cancel_together_never_admits(failure):
    from dexmani_real.runtime.lifecycle import SensorWorkers
    from dexmani_real.robot.commands import OperationCancelled
    state = SessionState()
    epoch = state.cancel_epoch
    class Ring:
        @property
        def latest_sequence(self):
            state.request_stop()
            return 1
    process = SimpleNamespace(name='camera', start=lambda: None,
                              is_alive=lambda: not failure, exitcode=1)
    workers = SensorWorkers()
    with pytest.raises(RuntimeError if failure else OperationCancelled):
        workers.start(process, ring=Ring(), timeout_s=1,
                      may_start=lambda: state.cancel_epoch == epoch)
    assert workers.processes == [process]


def test_model_completion_and_cancel_together_rechecks_admission():
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.robot.commands import OperationCancelled
    session = PolicySession.__new__(PolicySession)
    session.runtime_state = state = SessionState()
    session.startup_cancel_epoch = state.cancel_epoch
    session.keyboard = SimpleNamespace(check_health=lambda: True)
    session.robot = SimpleNamespace(service_feedback=lambda: None)
    done = Future()
    done.set_result(None)
    def poll():
        state.request_stop()
        return done
    session.model = SimpleNamespace(poll=poll)
    import time
    with pytest.raises(OperationCancelled):
        session.wait_model(time.monotonic() + 1)


def test_poll_waits_for_fast_stop_and_returns_consistent_snapshot():
    import threading
    state = SessionState()
    entered, release = threading.Event(), threading.Event()
    def stop():
        state.request_stop()
        entered.set()
        assert release.wait(2)
    keyboard = Keyboard(bindings={'s': stop})
    keyboard.active = True
    keyboard.listener = SimpleNamespace(is_alive=lambda: True)
    callback = threading.Thread(target=keyboard.handle_key, args=('s', True))
    snapshots = []
    reader = threading.Thread(target=lambda: snapshots.append(keyboard.poll(state, timeout=0)))
    callback.start()
    try:
        assert entered.wait(2)
        reader.start()
        assert snapshots == []
    finally:
        release.set()
        callback.join(2)
        reader.join(2)
    events, stopped = snapshots[0]
    assert [event.key for event in events] == ['s'] and stopped


@pytest.mark.parametrize('stage,key,exit_code', [
    ('vr', 's', 0), ('vr', 'c', 0), ('vr', 'd', 0), ('vr', 'q', 0), ('vr', 'esc', 1),
    ('connect', None, 1), ('running', 's', 1), ('cleanup', None, 1), ('idle', 'esc', 1),
])
def test_teleop_startup_cancel_has_one_exit_path(monkeypatch, stage, key, exit_code):
    import threading

    import numpy as np

    from dexmani_real.config.real import RealConfig
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.teleop import run

    keyboards, closed, sounds = [], [], []
    def start_keyboard(self):
        keyboards.append(self)
        self.active = True
        self.listener = SimpleNamespace(is_alive=lambda: True)
    monkeypatch.setattr(Keyboard, 'start', start_keyboard)
    monkeypatch.setattr(Keyboard, 'stop', lambda self: closed.append('keyboard'))

    class Process:
        pid = None
        exitcode = None
        def __init__(self, *, name, **kwargs):
            self.name = name
        def start(self):
            self.pid = 1
            if stage == 'vr':
                keyboards[0].handle_key(key, True)
        def is_alive(self):
            return self.pid is not None and self.exitcode is None
        def join(self, **kwargs):
            self.exitcode = 0
            closed.append('vr')

    class Robot:
        def __init__(self, arm, hand, *, runtime_state, **kwargs):
            self.state = runtime_state
        def connect(self):
            if stage == 'connect':
                self.state.stop_signal.set()
                raise OperationCancelled('device stopped without operator input')
        def check(self):
            pass
        def service_idle(self):
            pass
        def close(self):
            closed.append('robot')
            if stage == 'cleanup':
                raise OSError('robot close failed')

    monkeypatch.setattr(run.mp, 'get_context', lambda kind: SimpleNamespace(Process=Process, Event=threading.Event))
    monkeypatch.setattr(run, 'SensorChannels', lambda *a, **kw: SimpleNamespace(
        is_running=SimpleNamespace(value=True), vr_ring=SimpleNamespace(latest_sequence=1)))
    monkeypatch.setattr(run, 'Robot', Robot)
    monkeypatch.setattr(run, 'load_vr_transform', lambda path: np.eye(3))
    monkeypatch.setattr(run.ActionMapper, 'for_mode', lambda *a, **kw: None)
    monkeypatch.setattr(run, 'build_hand_retargeter', lambda runtime: None)
    monkeypatch.setattr(run, 'VRController', lambda *args: SimpleNamespace(hand_retargeter=None))
    monkeypatch.setattr(run, 'AudioFeedback', lambda: SimpleNamespace(play=sounds.append,
                                                                   close=lambda: closed.append('audio')))
    monkeypatch.setattr(run, 'initialize_hand', lambda *args, **kwargs: None)
    run_loop_idle = run.TeleopLoop.run
    def run_loop(self):
        if stage == 'idle':
            self.keyboard.handle_key(key, True)
            run_loop_idle(self)
            return
        if stage == 'cleanup':
            self.runtime_state.request_quit()
            return
        if stage == 'running':
            self.runtime_state.request_stop()  # 之前一段的正常 S 不能授权吞掉新故障。
            raise OperationCancelled('unexpected running interruption')
        pytest.fail('cancelled startup entered control')
    monkeypatch.setattr(run.TeleopLoop, 'run', run_loop)
    result = run.run_teleop(RealConfig(), record=False, episodes_dir='unused', max_duration_s=1,
                            runtime_config_json=None)
    assert result == exit_code
    if stage in {'connect', 'running', 'cleanup'} or key == 'esc':
        assert 'emergency' in sounds
    assert 'robot' in closed and 'keyboard' in closed and 'audio' in closed
    if stage == 'vr':
        assert 'vr' in closed


def test_policy_does_not_treat_internal_stop_as_startup_cancel():
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.robot.commands import OperationCancelled

    session = PolicySession.__new__(PolicySession)
    session.runtime_state = state = SessionState()
    session.startup_cancel_epoch = state.cancel_epoch
    session.info = SimpleNamespace(checkpoint_path=SimpleNamespace(name='fake'), weights='test',
                                   inference_steps=1, control_dt_s=.0625)
    session.runtime = SimpleNamespace(execution=SimpleNamespace(mode='sync', action_steps=1))
    session.save_results = lambda **kwargs: None
    session.robot = SimpleNamespace(check=lambda: None)
    session.summary = {}
    failures = []
    def prepare():
        state.stop_signal.set()
        raise OperationCancelled('device preparation unexpectedly stopped')
    session.prepare_resources = prepare
    session.close = lambda **kwargs: failures.append(kwargs['failure']) or 1
    assert session.run() == 1
    assert isinstance(failures[0], OperationCancelled)
    assert not state.quit_requested


def test_unconsumed_discard_does_not_change_finished_segment():
    from dexmani_real.teleop.loop import TeleopLoop
    state = SessionState()
    keyboard = Keyboard(bindings={'d': state.request_stop})
    keyboard.active = True
    loop = TeleopLoop.__new__(TeleopLoop)
    loop.keyboard, loop.runtime_state = keyboard, state
    loop.pending_termination_reason = 'operator'
    loop.termination_details = []
    loop.discard_requested = False
    loop.audio = SimpleNamespace(play=lambda sound: None)
    decisions = []
    loop.recorder = SimpleNamespace(
        finish=lambda **kwargs: decisions.append(kwargs['save']),
        recording_truncated_reason=None,
    )
    keyboard.handle_key('d', True)
    loop.finish_recording()
    assert decisions == [True]
    assert state.stop_signal.is_set()


def test_emergency_latches_before_lock_and_survives_concurrent_stop_clear():
    import threading
    state = SessionState()
    request = threading.Thread(target=state.request_emergency_stop)
    with state.motion_lock:
        request.start()
        assert state.emergency_signal.wait(1)
        state.stop_signal.clear()  # 模拟准备边界已经读完条件、随后清 stop。
    request.join(1)
    assert not request.is_alive()
    assert state.estop_request and state.stop_signal.is_set()


@pytest.mark.parametrize('record_raw', [False, True])
def test_policy_starts_only_sensors_requested_by_control_or_recording(monkeypatch, tmp_path, record_raw):
    import sys
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment import session as deployment
    from dexmani_real.runtime.observation import ObservationNeeds

    channels = []
    monkeypatch.setitem(sys.modules, 'dexmani_policy.deployment', SimpleNamespace(load_policy=lambda *a, **kw: None))
    monkeypatch.setattr(deployment, 'AudioFeedback', lambda: None)
    monkeypatch.setattr(deployment.ActionMapper, 'for_mode', lambda *a, **kw: None)
    monkeypatch.setattr(deployment, 'Keyboard', lambda **kwargs: SimpleNamespace(
        start=lambda: None, check_health=lambda: True,
    ))
    monkeypatch.setattr(deployment, 'PolicyWorker', lambda *a, **kw: SimpleNamespace(load=lambda: None))
    def sensor_channels(*args, **kwargs):
        channels.append(kwargs)
        return object()
    monkeypatch.setattr(deployment, 'SensorChannels', sensor_channels)
    session = deployment.PolicySession(
        RealConfig(), SimpleNamespace(action_mode='joint'), output_dir=tmp_path,
        run_config={}, device='cpu', seed=0, max_running_s=60, num_episodes=1,
        record_raw=record_raw, camera_calibration_path=None,
        observations=SimpleNamespace(needs=ObservationNeeds(), camera_static=False, build=lambda rows: rows),
    )
    session.prepare_resources()
    assert bool(channels) == record_raw
    assert session.capture_needs.camera_worker == record_raw
    assert session.capture_needs.required_tactile == (('aggregate', 'dense') if record_raw else ())

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
    from dexmani_real.runtime.state import SafetyState
    state = SessionState(safety_state=SafetyState.ARMED)
    keyboard = Keyboard(bindings={'s': state.request_stop})
    keyboard.active = True
    keyboard.allow_start()
    epoch = keyboard.accept_operation(state)
    # 内部准备停止不应成为一次用户取消，或提前占据结束原因。
    state.revoke_motion()
    assert state.cancel_epoch == epoch
    state.stop_signal.clear()
    assert state.begin_motion(cancel_epoch=epoch) is not None
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
    loop.discard_requested = loop.capture_failed = False
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
    workers = SensorWorkers(required_names=("camera",))
    process = SimpleNamespace(pid=None, name='camera', join=lambda **kwargs: pytest.fail('unstarted join'),
                              start=lambda: (_ for _ in ()).throw(OSError('start failed')))
    with pytest.raises(OSError, match='start failed'):
        workers.start(process, ring=None, may_start=lambda: True)
    workers.shutdown(SessionState())
    assert not workers.processes


@pytest.mark.parametrize('required', [False, True])
def test_finished_archive_worker_is_not_a_pending_or_close_failure(required):
    from dexmani_real.runtime.lifecycle import SensorWorkers
    workers = SensorWorkers(required_names=('camera',) if required else ())
    workers.processes = [SimpleNamespace(name='camera', pid=10, exitcode=1,
                                        join=lambda **kwargs: None, is_alive=lambda: False)]
    if required:
        with pytest.raises(RuntimeError, match='exitcode=1'):
            workers.shutdown(SessionState())
    else:
        workers.shutdown(SessionState())
    assert not workers.processes


@pytest.mark.parametrize('keys', [('x',), ('x', 'b')])
def test_policy_ignores_unbound_keys_before_start_admission(keys):
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.runtime.keyboard import KeyEvent
    from dexmani_real.runtime.state import SafetyState
    session = PolicySession.__new__(PolicySession)
    session.runtime_state = SessionState(safety_state=SafetyState.ARMED)
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
    workers = SensorWorkers(required_names=('camera',))
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
    ('connect', None, 1),
])
def test_teleop_startup_cancel_has_one_exit_path(monkeypatch, stage, key, exit_code):
    import threading

    import numpy as np

    from dexmani_real.config.real import RealConfig
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.teleop import run

    keyboards, closed = [], []
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
        def close(self):
            closed.append('robot')

    monkeypatch.setattr(run.mp, 'get_context', lambda kind: SimpleNamespace(Process=Process, Event=threading.Event))
    monkeypatch.setattr(run, 'SensorChannels', lambda *a, **kw: SimpleNamespace(
        is_running=SimpleNamespace(value=True), vr_ring=SimpleNamespace(latest_sequence=1)))
    monkeypatch.setattr(run, 'Robot', Robot)
    monkeypatch.setattr(run, 'load_vr_transform', lambda path: np.eye(3))
    monkeypatch.setattr(run.ActionMapper, 'for_mode', lambda *a, **kw: None)
    monkeypatch.setattr(run, 'build_hand_retargeter', lambda runtime: None)
    monkeypatch.setattr(run, 'VRController', lambda *args: SimpleNamespace(hand_retargeter=None))
    monkeypatch.setattr(run, 'AudioFeedback', lambda: SimpleNamespace(play=lambda sound: None,
                                                                   close=lambda: closed.append('audio')))
    monkeypatch.setattr(run, 'initialize_hand', lambda *args, **kwargs: None)
    monkeypatch.setattr(run.TeleopLoop, 'run', lambda self: pytest.fail('cancelled startup entered control'))
    result = run.run_teleop(RealConfig(), record=False, episodes_dir='unused', max_duration_s=1,
                            runtime_config_json=None)
    assert result == exit_code
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
        state.revoke_motion()
        raise OperationCancelled('device preparation unexpectedly stopped')
    session.prepare_resources = prepare
    session.close = lambda **kwargs: failures.append(kwargs['failure']) or 1
    assert session.run() == 1
    assert isinstance(failures[0], OperationCancelled)
    assert not state.quit_requested

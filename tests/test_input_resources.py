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
    from dexmani_real.runtime.state import RunEndReason, SafetyState
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
    assert state.run_ended_reason == RunEndReason.OPERATOR
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

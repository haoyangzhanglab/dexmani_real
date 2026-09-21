"""The listener can revoke motion while session orchestration blocks in home."""
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from dexmani_real.deployment import session
from dexmani_real.runtime.operator_input import OperatorCommand as Command
from test_deployment_evidence import _fake_shared


@pytest.mark.parametrize('key', ['stop', 'quit', 'estop'])
def test_immediate_callback_remains_live_during_home(key):
    shared = _fake_shared()
    shared.arm_home_completed_generation = SimpleNamespace(value=0)
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    callbacks = {}
    errors = []

    class Keyboard:
        healthy = True
        estop_latched = False
        def __init__(self, **kwargs): callbacks.update(kwargs)
        def start(self): pass
        def stop(self): pass
        def drain_signal(self, command): pass
        def poll(self, timeout): return [Command.HOME]

    def home(*args, abort_requested):
        entered.set()
        assert release.wait(3)
        assert abort_requested()
        return False

    def run():
        try:
            session.run_operator_control(shared, None, object(), stop_event=stopped, execute=True)
        except BaseException as exc:
            errors.append(exc)

    with patch.object(session, 'KeyboardInput', Keyboard), patch.object(session, 'home_policy_robot', home):
        thread = threading.Thread(target=run)
        thread.start()
        try:
            assert entered.wait(3)
            generation = shared.run_generation.value
            callbacks[key + '_callback']()
            if key == 'estop':
                assert shared.estop_request.value
            else:
                assert shared.run_generation.value > generation
                assert shared.stop_request.value
                if key == 'quit': assert shared.quit_requested.value
        finally:
            stopped.set()
            release.set()
            thread.join(3)
        assert not thread.is_alive()
        assert not errors
        assert not shared.physical_home_completed.value


def test_same_batch_stop_suppresses_begin():
    shared = _fake_shared()
    stopped = threading.Event()
    keyboard = SimpleNamespace(start=lambda: None, stop=lambda: None,
                               healthy=True, estop_latched=False)
    def poll(timeout):
        stopped.set()
        return [Command.BEGIN, Command.STOP]
    keyboard.poll = poll
    with patch.object(session, 'KeyboardInput', return_value=keyboard):
        session.run_operator_control(shared, None, None, stop_event=stopped, execute=False)
    assert not shared.start_request.value

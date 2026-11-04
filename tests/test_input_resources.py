import signal
import threading
from concurrent.futures import Future, TimeoutError as FutureTimeoutError, wait
from types import SimpleNamespace

import pytest

from dexmani_real.deployment.inference import PolicyWorker
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.state import SessionState


@pytest.fixture
def resource_logs(monkeypatch, caplog):
    import logging
    monkeypatch.setattr(logging.getLogger('dexmani_real'), 'handlers', [])
    monkeypatch.setattr(logging.getLogger('dexmani_real'), 'propagate', True)
    return caplog


@pytest.mark.parametrize('error_types', [
    (OSError, None, None), (None, OSError, None), (None, None, OSError),
    (KeyboardInterrupt, None, None), (OSError, KeyboardInterrupt, OSError),
    (OSError, None, KeyboardInterrupt),
])
@pytest.mark.parametrize('alive', [False, True])
def test_keyboard_stop_attempts_join_and_restore_after_failures(monkeypatch, resource_logs, error_types, alive):
    from dexmani_real.runtime import keyboard as keyboard_module
    state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
    keyboard = Keyboard(estop_callback=state.request_emergency_stop)
    keyboard.active = True
    keyboard.events.append('b')
    keyboard.held_keys.add('w')
    saved = keyboard.saved_termios = object()
    errors = [kind('cleanup failed') if kind else None for kind in error_types]
    calls = []
    def action(index):
        assert not keyboard.active and not keyboard.events and not keyboard.held_keys
        if KeyboardInterrupt in error_types[:index]:
            assert state.estop_request and state.stop_signal.is_set()
        calls.append(index)
        if errors[index] is not None:
            raise errors[index]
    def join(timeout):
        assert timeout == keyboard_module.LISTENER_STOP_TIMEOUT_S
        action(1)
    def restore(value):
        assert value is saved
        action(2)
    listener = keyboard.listener = SimpleNamespace(
        ident=1, stop=lambda: action(0), join=join, is_alive=lambda: alive,
    )
    monkeypatch.setattr(keyboard_module, 'restore_terminal_echo', restore)
    failures = [error for error in errors if error is not None]
    with pytest.raises(type(failures[0])) as caught:
        keyboard.stop()
    assert caught.value is failures[0]
    assert calls == [0, 1, 2]
    assert keyboard.saved_termios is (saved if errors[2] else None)
    assert keyboard.listener is listener
    assert state.estop_request == (KeyboardInterrupt in error_types)
    reported = [record for record in resource_logs.records if record.exc_info]
    assert [record.exc_info[1] for record in reported] == failures[1:]
    for record in reported:
        assert record.exc_info[2] is not None
        stage = 'listener join' if record.exc_info[1] is errors[1] else 'terminal restore'
        assert stage in record.getMessage()
    assert ('listener remains active' in resource_logs.text) == alive


def test_keyboard_stop_attempts_cleanup_when_emergency_callback_fails(monkeypatch, resource_logs):
    calls = []
    interrupted = KeyboardInterrupt('stop interrupted')
    callback_error = RuntimeError('emergency callback failed')
    join_error = OSError('join failed')
    latched = threading.Event()
    def emergency_stop():
        latched.set()
        raise callback_error
    keyboard = Keyboard(estop_callback=emergency_stop)
    saved = keyboard.saved_termios = object()
    def stop():
        calls.append('stop')
        raise interrupted
    def join(timeout):
        assert latched.is_set()
        calls.append('join')
        raise join_error
    def restore(value):
        assert latched.is_set() and value is saved
        calls.append('restore')
    keyboard.listener = SimpleNamespace(ident=1, stop=stop, join=join, is_alive=lambda: False)
    monkeypatch.setattr('dexmani_real.runtime.keyboard.restore_terminal_echo', restore)
    with pytest.raises(RuntimeError) as caught:
        keyboard.stop()
    assert caught.value is callback_error and callback_error.__context__ is interrupted
    assert calls == ['stop', 'join', 'restore'] and keyboard.saved_termios is None
    reported = [record for record in resource_logs.records if record.exc_info]
    assert len(reported) == 1 and reported[0].exc_info[1] is join_error
    assert reported[0].exc_info[2] is not None and 'listener join' in reported[0].getMessage()


def test_keyboard_stop_without_started_listener_restores_terminal(monkeypatch):
    keyboard = Keyboard()
    keyboard.saved_termios = saved = object()
    keyboard.listener = SimpleNamespace(ident=None, stop=lambda: None, is_alive=lambda: False,
                                        join=lambda **kw: pytest.fail('unstarted listener join'))
    restored = []
    monkeypatch.setattr('dexmani_real.runtime.keyboard.restore_terminal_echo', restored.append)
    keyboard.stop()
    assert restored == [saved] and keyboard.saved_termios is None


def test_busy_press_requires_release_and_new_press():
    keyboard = Keyboard()
    keyboard.active = True
    keyboard.handle_key('b', True)
    keyboard.allow_start()
    keyboard.handle_key('b', True)
    assert keyboard.events == keyboard.events.__class__()
    keyboard.handle_key('b', False)
    keyboard.handle_key('b', True)
    assert list(keyboard.events) == ['b']


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


@pytest.mark.parametrize('prediction_fails', [False, True])
def test_model_close_waits_for_prediction_on_its_thread(prediction_fails):
    from threading import Event, get_ident
    started, release = Event(), Event()
    calls = []
    def record(stage):
        calls.append((stage, get_ident()))
    def predict(rows):
        record('predict')
        started.set()
        assert release.wait(2)
        if prediction_fails:
            raise ValueError('prediction failed')
    def load():
        record('load')
        return SimpleNamespace(predict=predict, close=lambda: record('close'),
                               warmup=lambda rows: record('warmup'),
                               reset_episode=lambda: record('reset'))
    worker = PolicyWorker(load, build_observation=lambda rows: rows)
    try:
        worker.load()
        worker.future.result(timeout=1)
        worker.poll().result()
        worker.warmup([])
        worker.future.result(timeout=1)
        worker.poll().result()
        worker.predict([], reset_episode=True)
        assert started.wait(1)
        with pytest.raises(TimeoutError):
            worker.discard_pending(0)
        with pytest.raises(RuntimeError, match='unreclaimed task'):
            worker.predict([])
        closed = worker.close()
        assert worker.close() is closed
        with pytest.raises(RuntimeError, match='closing'):
            worker.predict([])
        with pytest.raises(FutureTimeoutError):
            closed.result(timeout=0)
        release.set()
        if prediction_fails:
            with pytest.raises(ValueError, match='prediction failed'):
                closed.result(timeout=1)
        else:
            closed.result(timeout=1)
        assert [stage for stage, _ in calls] == ['load', 'warmup', 'reset', 'predict', 'close']
        assert len({thread for _, thread in calls}) == 1
        assert calls[0][1] != get_ident()
    finally:
        release.set()
        worker.executor.shutdown(wait=False)


@pytest.fixture
def model_session():
    from dexmani_real.deployment.session import PolicySession
    session = PolicySession.__new__(PolicySession)
    session.runtime_state = SessionState()
    session.robot = session.sensors = session.keyboard = session.episode = session.audio = None
    session.errors = []
    session.summary = {'termination_reason': None}
    session.save_results = lambda: None
    return session


@pytest.mark.parametrize('prior_failure', [None, 'robot', 'recording', 'model'])
@pytest.mark.parametrize('audio_error_type', [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize('initial_reason', [None, 'startup_cancelled'])
def test_policy_audio_cleanup_only_ordinary_errors_are_optional(
    model_session, resource_logs, prior_failure, audio_error_type, initial_reason,
):
    session = model_session
    session.summary['termination_reason'] = initial_reason
    session.model = None
    primary = OSError('primary failure')
    audio_error = audio_error_type('audio close failed')
    calls = []
    def fail():
        raise primary
    if prior_failure == 'robot':
        session.robot = SimpleNamespace(close=fail)
    elif prior_failure == 'recording':
        session.episode = SimpleNamespace(finish_recording=fail)
    elif prior_failure == 'model':
        future = Future()
        future.set_exception(primary)
        session.model = SimpleNamespace(close=lambda: future)
    def close_audio():
        calls.append('audio')
        raise audio_error
    def save_results():
        if audio_error_type is KeyboardInterrupt:
            assert session.runtime_state.estop_request
        calls.append('results')
    session.audio = SimpleNamespace(close=close_audio)
    session.save_results = save_results
    fatal_audio = audio_error_type is not OSError
    assert session.close() == int(prior_failure is not None or fatal_audio)
    assert calls == ['audio', 'results']
    assert len(session.errors) == int(prior_failure is not None) + int(fatal_audio)
    if prior_failure is not None:
        assert session.errors[0]['message'] == 'primary failure'
    reports = [record for record in resource_logs.records if record.exc_info and record.exc_info[1] is audio_error]
    assert len(reports) == 1
    assert reports[0].levelname == ('ERROR' if fatal_audio else 'WARNING')
    assert session.runtime_state.estop_request == (audio_error_type is KeyboardInterrupt)
    assert session.summary['termination_reason'] == (
        'estop' if audio_error_type is KeyboardInterrupt
        else 'failed' if prior_failure is not None or fatal_audio else initial_reason or 'quit'
    )


def test_policy_clean_close_preserves_startup_cancellation(model_session):
    model_session.model = None
    model_session.summary['termination_reason'] = 'startup_cancelled'
    assert model_session.close() == 0
    assert model_session.summary['termination_reason'] == 'startup_cancelled'


@pytest.mark.parametrize('alive', [False, True])
def test_audio_close_cancels_pending_and_reports_actual_thread_state(alive, resource_logs):
    import threading
    from dexmani_real.teleop.audio import AudioFeedback
    audio = AudioFeedback.__new__(AudioFeedback)
    audio.condition = threading.Condition()
    audio.closed, audio.pending = False, object()
    waits = []
    def join(timeout):
        assert audio.closed and audio.pending is None
        waits.append(timeout)
    worker = audio.worker = SimpleNamespace(join=join, is_alive=lambda: alive)
    if alive:
        with pytest.raises(TimeoutError, match='still active'):
            audio.close()
    else:
        audio.close()
    assert waits == [1.0] and audio.worker is worker
    # 关闭错误交给运行入口报告，此处不重复 warning。
    assert not resource_logs.records


def test_audio_cancel_kills_and_reaps_player(monkeypatch):
    import threading
    from dexmani_real.teleop import audio as audio_module
    audio = audio_module.AudioFeedback.__new__(audio_module.AudioFeedback)
    audio.condition, audio.player, audio.closed = threading.Condition(), 'aplay', True
    calls = []
    player = SimpleNamespace(returncode=None)
    player.poll = lambda: player.returncode
    player.kill = lambda: calls.append('kill')
    def communicate(timeout):
        calls.append(('reap', timeout))
        player.returncode = -signal.SIGKILL
        return None, b''
    player.communicate = communicate
    monkeypatch.setattr(audio_module.subprocess, 'Popen', lambda *a, **kw: player)
    audio.play_one('unused.wav')
    assert calls == ['kill', ('reap', .5)] and player.returncode == -signal.SIGKILL


@pytest.mark.parametrize('task_type,close_type', [
    (ValueError, None), (None, OSError), (ValueError, OSError),
    (KeyboardInterrupt, None), (KeyboardInterrupt, OSError), (ValueError, KeyboardInterrupt),
    (TimeoutError, None), (None, TimeoutError),
])
def test_model_close_preserves_failures_and_interrupts(model_session, task_type, close_type, caplog):
    task_error = task_type('task failed') if task_type else None
    close_error = close_type('close failed') if close_type else None
    closes = []
    def predict(rows):
        if task_error is not None:
            raise task_error
    def close():
        closes.append(True)
        if close_error is not None:
            raise close_error
    worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
    worker.model = SimpleNamespace(predict=predict, close=close)
    model_session.model = worker
    try:
        worker.predict([])
        closed = worker.close()
        assert closed in wait((closed,), timeout=2).done
        assert model_session.close() == 1
        assert closes == [True]
        expected = task_error if task_type is KeyboardInterrupt else close_error or task_error
        assert closed.exception() is expected
        assert model_session.runtime_state.estop_request == (
            task_type is KeyboardInterrupt or close_type is KeyboardInterrupt
        )
        assert model_session.errors == [dict(stage='model_close',
                                             exception_type=type(expected).__name__, message=str(expected))]
        for error in (task_error, close_error):
            if error is not None:
                assert str(error) in caplog.text
        if task_error is not None and close_error is not None:
            if task_type is KeyboardInterrupt:
                assert task_error.__cause__ is close_error
            else:
                assert close_error.__context__ is task_error
    finally:
        worker.executor.shutdown(wait=True)


def test_session_close_timeout_retains_running_task(model_session, monkeypatch):
    from threading import Event
    started, release = Event(), Event()
    closes = []
    def predict(rows):
        started.set()
        assert release.wait(5)
    worker = PolicyWorker(lambda: None, build_observation=lambda rows: rows)
    worker.model = SimpleNamespace(predict=predict, close=lambda: closes.append(True))
    model_session.model = worker
    monkeypatch.setattr('dexmani_real.deployment.session.MODEL_SHUTDOWN_TIMEOUT_S', 0)
    try:
        worker.predict([])
        assert started.wait(1)
        assert model_session.close() == 1
        assert model_session.errors == [dict(stage='model_close', exception_type='TimeoutError',
                                             message='model close still pending; task handle retained')]
        assert not worker.close_future.done() and not closes
        with pytest.raises(RuntimeError, match='closing'):
            worker.predict([])
        release.set()
        worker.close_future.result(timeout=2)
        assert closes == [True]
    finally:
        release.set()
        worker.executor.shutdown(wait=True)


@pytest.fixture
def policy_info(tmp_path):
    from dexmani_policy.deployment import PolicyInfo
    return PolicyInfo(
        experiment_dir=tmp_path, checkpoint_path=tmp_path / 'policy.pt',
        weights='ema', inference_steps=1, policy_name='test', task_name='test',
        observation_fields=('joint_state',), n_obs_steps=2, n_action_steps=4,
        horizon=9, action_start_index=1, action_dim=19, control_action_dim=19,
        action_mode='joint', rgb_preprocessing={}, pc_dim=None,
        supported_execution_modes=('sync',), control_dt_s=.0625, depth_config=None,
        pointcloud_config=None, hand_pointcloud_config=None, fingertip_link_names=None,
    )


@pytest.mark.parametrize('field', ['joint_state', 'finger_tactile_force', 'taxel_tactile_force'])
def test_current_policy_metadata_to_live_observation(policy_info, field):
    from dataclasses import replace
    import numpy as np
    import torch
    from dexmani_policy.deployment import PolicyRuntime
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment.observation import ObservationBuilder
    from dexmani_real.deployment.preflight import resolve_deployment_config
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
    from dexmani_real.runtime.observation import Observation

    info = replace(policy_info, observation_fields=(field,))
    runtime = resolve_deployment_config(info, RealConfig())
    assert runtime.execution.action_steps == info.n_action_steps
    builder = ObservationBuilder(info, runtime)
    arm, hand = np.zeros(1, dtype=ARM_STATE_DTYPE), np.zeros(1, dtype=HAND_STATE_DTYPE)
    arm['qpos'], hand['qpos'] = .1, .2
    hand['tactile_aggregate'], hand['tactile_dense'] = .3, .4
    row = builder.project(Observation(arm, hand, None, None, None, 1))
    observation = builder.build([row, row])
    policy = PolicyRuntime(torch.nn.Linear(1, 1), info, seed=0)
    values = policy.prepare_observations(observation)[field]
    np.testing.assert_array_equal(values[0].numpy(), observation[field])
    assert values.dtype == torch.float32
    observation[field] = observation[field][:, :4]
    with pytest.raises(ValueError, match='trailing shape'):
        policy.prepare_observations(observation)


def test_cloud_recipe_can_use_defaults_but_rejects_invalid_values(policy_info):
    from dataclasses import replace
    from dexmani_real.config.perception import HandCloudConfig, PointCloudConfig
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment.preflight import resolve_deployment_config
    runtime = RealConfig()
    for field, attribute, config_type in (
        ('point_cloud', 'pointcloud', PointCloudConfig),
        ('hand_pointcloud', 'hand_pointcloud', HandCloudConfig),
    ):
        info = replace(policy_info, observation_fields=(field,),
                       **{f'{attribute}_config': {'num_points': 128}})
        resolved = resolve_deployment_config(info, runtime)
        assert getattr(resolved, attribute) == config_type(num_points=128)
        assert getattr(runtime, attribute) == config_type()
        with pytest.raises(TypeError):
            resolve_deployment_config(replace(info, **{f'{attribute}_config': {'num_ponts': 128}}), runtime)
        with pytest.raises(ValueError, match='positive integer'):
            resolve_deployment_config(replace(info, **{f'{attribute}_config': {'num_points': 0}}), runtime)


@pytest.mark.parametrize('frequency', [100, .001])
def test_jog_accepts_valid_clock_with_hand_enabled(frequency):
    from dataclasses import replace
    import numpy as np
    from dexmani_real.config.real import RealConfig
    from dexmani_real.runtime.preflight import jog_bounds

    runtime = RealConfig()
    assert runtime.hand.enabled
    runtime = replace(runtime, keyboard_teleop=replace(runtime.keyboard_teleop, control_hz=frequency))
    lower, upper = jog_bounds(runtime)
    workspace = runtime.environment.workspace.as_array()
    margin = runtime.keyboard_teleop.workspace_command_margin_m
    np.testing.assert_array_equal(lower, workspace[:, 0] + margin)
    np.testing.assert_array_equal(upper, workspace[:, 1] - margin)


@pytest.mark.parametrize('frequency', [0, -1, float('nan'), float('inf'), -float('inf'), 3e9, 1e-310])
def test_jog_rejects_invalid_clock_frequency(frequency):
    from dataclasses import replace
    from dexmani_real.config.real import RealConfig
    from dexmani_real.runtime.preflight import jog_bounds

    runtime = RealConfig()
    runtime = replace(runtime, keyboard_teleop=replace(runtime.keyboard_teleop, control_hz=frequency))
    with pytest.raises(ValueError, match='control_hz|control period'):
        jog_bounds(runtime)


def test_jog_still_rejects_empty_workspace():
    from dataclasses import replace
    from dexmani_real.config.real import RealConfig
    from dexmani_real.runtime.preflight import jog_bounds

    runtime = RealConfig()
    runtime = replace(runtime, keyboard_teleop=replace(
        runtime.keyboard_teleop, control_hz=100, workspace_command_margin_m=1,
    ))
    with pytest.raises(ValueError, match='no interior workspace'):
        jog_bounds(runtime)


@pytest.mark.parametrize('frequency,hand_enabled,message', [
    (100, True, '83 Hz'), (3e9, False, 'control period'), (1e-310, False, 'control period'),
])
def test_teleop_rejects_invalid_target_rate_before_startup(monkeypatch, frequency, hand_enabled, message):
    from dataclasses import replace
    from dexmani_real.config.real import RealConfig
    from dexmani_real.teleop import run

    runtime = RealConfig()
    runtime = replace(runtime, teleop=replace(runtime.teleop, control_hz=frequency),
                      hand=replace(runtime.hand, enabled=hand_enabled))
    monkeypatch.setattr(run, 'configure_logging', lambda: None)
    with pytest.raises(ValueError, match=message):
        run.run_teleop(runtime, record=False, episodes_dir='', max_duration_s=0,
                       runtime_config_json=None)


def test_policy_still_rejects_hand_targets_above_83_hz(policy_info):
    from dataclasses import replace
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment.preflight import resolve_deployment_config

    with pytest.raises(ValueError, match='83 Hz'):
        resolve_deployment_config(replace(policy_info, control_dt_s=.01), RealConfig())


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
    assert list(keyboard.events) == ["s"]
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


@pytest.mark.parametrize('exitcode', [1, -signal.SIGTERM, -signal.SIGKILL])
def test_failed_archive_worker_is_reported_without_retaining_dead_handle(exitcode):
    from dexmani_real.runtime.lifecycle import SensorWorkers
    workers = SensorWorkers()
    workers.processes = [SimpleNamespace(name='camera', pid=10, exitcode=exitcode,
                                        join=lambda **kwargs: None, is_alive=lambda: False)]
    with pytest.raises(RuntimeError, match=f'exitcode={exitcode}'):
        workers.shutdown(SessionState())
    assert not workers.processes


@pytest.mark.parametrize('initial_exitcode,terminate_exitcode,kill_exitcode,outcome', [
    (0, None, None, 'normal'),
    (None, -signal.SIGTERM, None, 'warning'),
    (None, None, -signal.SIGKILL, 'warning'),
    (None, None, -signal.SIGTERM, 'warning'),
    (None, 0, None, 'normal'),  # terminate 调用时已恰好自行退出。
    (None, 1, None, 'error'),
    (None, -signal.SIGKILL, None, 'error'),  # 本方只尝试过 SIGTERM。
    (None, None, 1, 'error'),
    (None, None, None, 'pending'),
])
def test_sensor_shutdown_matches_only_attempted_signals(
    caplog, initial_exitcode, terminate_exitcode, kill_exitcode, outcome,
):
    from dexmani_real.runtime.lifecycle import SensorWorkers
    workers = SensorWorkers(SimpleNamespace(is_running=SimpleNamespace(value=True)))
    calls = []
    process = SimpleNamespace(name='camera', pid=10, exitcode=initial_exitcode)
    process.join = lambda **kwargs: calls.append('join')
    process.is_alive = lambda: process.exitcode is None
    def terminate():
        calls.append('terminate')
        process.exitcode = terminate_exitcode
    def kill():
        calls.append('kill')
        process.exitcode = kill_exitcode
    process.terminate, process.kill = terminate, kill
    workers.processes = [process]
    if outcome in {'error', 'pending'}:
        message = 'could not be confirmed stopped' if outcome == 'pending' else 'exitcode='
        with pytest.raises(RuntimeError, match=message):
            workers.shutdown(SessionState())
    else:
        workers.shutdown(SessionState())
        assert bool(caplog.records) == (outcome == 'warning')
        assert all(record.levelname == 'WARNING' for record in caplog.records)
    expected_calls = ['join']
    if initial_exitcode is None:
        expected_calls += ['terminate', 'join']
        if terminate_exitcode is None:
            expected_calls += ['kill', 'join']
    assert calls == expected_calls
    assert not workers.channels.is_running.value
    assert workers.processes == ([process] if outcome == 'pending' else [])


@pytest.mark.parametrize('failed_call', ['join', 'terminate', 'kill'])
def test_sensor_cleanup_call_errors_are_reported(failed_call):
    from dexmani_real.runtime.lifecycle import SensorWorkers
    workers = SensorWorkers()
    process = SimpleNamespace(name='camera', pid=10, exitcode=None)
    def cleanup_call(name):
        if name == failed_call:
            if name == 'join':
                process.exitcode = 0  # 已退出仍不能掩盖 join 本身的错误。
            raise OSError(f'{name} failed')
    process.join = lambda **kwargs: cleanup_call('join')
    process.terminate = lambda: cleanup_call('terminate')
    process.kill = lambda: cleanup_call('kill')
    process.is_alive = lambda: process.exitcode is None
    joined = []
    other = SimpleNamespace(name='vr', pid=11, exitcode=0,
                            join=lambda **kwargs: joined.append('vr'), is_alive=lambda: False)
    workers.processes = [process, other]
    with pytest.raises(OSError, match=f'{failed_call} failed'):
        workers.shutdown(SessionState())
    assert joined == ['vr']
    assert workers.processes == ([] if failed_call == 'join' else [process])


def test_sensor_shutdown_latches_later_interrupt_and_continues(resource_logs):
    from dexmani_real.runtime.lifecycle import SensorWorkers
    state = SessionState()
    workers = SensorWorkers()
    failure, interrupted = OSError('join failed'), KeyboardInterrupt('join interrupted')
    def fail_join(**kw):
        raise failure
    def interrupt_join(**kw):
        raise interrupted
    calls = []
    def final_join(**kw):
        assert state.estop_request and state.stop_signal.is_set()
        calls.append('last')
    pending = SimpleNamespace(name='camera', pid=1, exitcode=None, join=fail_join)
    exited = SimpleNamespace(name='vr', pid=2, exitcode=0, join=interrupt_join)
    last = SimpleNamespace(name='other', pid=3, exitcode=0, join=final_join, is_alive=lambda: False)
    workers.processes = [pending, exited, last]
    with pytest.raises(OSError) as caught:
        workers.shutdown(state)
    assert caught.value is failure and calls == ['last']
    assert workers.processes == [pending]
    assert [record.exc_info[1] for record in resource_logs.records if record.exc_info] == [interrupted]


def test_close_runtime_keeps_order_and_latches_interrupt_before_next_resource(resource_logs):
    from dexmani_real.runtime.lifecycle import close_runtime
    state = SessionState()
    calls = []
    failure, interrupted = OSError('robot failed'), KeyboardInterrupt('keyboard interrupted')
    def close_robot():
        assert state.stop_signal.is_set()
        calls.append('robot')
        raise failure
    def stop_keyboard():
        calls.append('keyboard')
        raise interrupted
    def shutdown_sensors(runtime_state):
        assert runtime_state is state and state.estop_request
        calls.append('sensors')
    with pytest.raises(OSError) as caught:
        close_runtime(state, SimpleNamespace(close=close_robot),
                      keyboard=SimpleNamespace(stop=stop_keyboard),
                      sensors=SimpleNamespace(shutdown=shutdown_sensors))
    assert caught.value is failure and calls == ['robot', 'keyboard', 'sensors']
    assert [record.exc_info[1] for record in resource_logs.records if record.exc_info] == [interrupted]


@pytest.mark.parametrize('prior_result', ['session', 'stop', 'estop'])
def test_sensor_recovery_preserves_policy_failure(prior_result):
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.runtime.lifecycle import SensorWorkers
    from dexmani_real.utils.log import error_detail

    session = PolicySession.__new__(PolicySession)
    session.runtime_state = SessionState()
    session.robot = session.keyboard = session.episode = session.model = session.audio = None
    session.summary = {'termination_reason': None}
    session.save_results = lambda: None
    session.errors = []
    if prior_result == 'estop':
        session.runtime_state.request_emergency_stop()
    else:
        session.errors.append(error_detail(prior_result, OSError('existing failure')))
    original_errors = list(session.errors)
    session.sensors = SensorWorkers()
    process = SimpleNamespace(name='camera', pid=10, exitcode=None,
                              join=lambda **kwargs: None)
    process.is_alive = lambda: process.exitcode is None
    process.terminate = lambda: setattr(process, 'exitcode', -signal.SIGTERM)
    session.sensors.processes = [process]

    assert session.close() == 1
    assert session.errors == original_errors
    assert not session.sensors.processes
    assert session.summary['termination_reason'] == ('estop' if prior_result == 'estop' else 'failed')


@pytest.mark.parametrize('keys', [('x',), ('x', 'b')])
def test_policy_ignores_unbound_keys_before_start_admission(keys):
    from dexmani_real.deployment.session import PolicySession
    session = PolicySession.__new__(PolicySession)
    session.runtime_state = SessionState()
    session.keyboard = Keyboard()
    session.keyboard.allow_start()
    session.episode = None
    epoch = session.handle_command_batch(list(keys))
    if 'b' in keys:
        assert epoch == session.runtime_state.cancel_epoch
    else:
        assert epoch is None
        # 无关按键不能消耗接纳，随后 B 仍可开始。
        assert session.handle_command_batch(['b']) == session.runtime_state.cancel_epoch


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
    assert events == ['s'] and stopped


@pytest.mark.parametrize('stage,key,exit_code', [
    ('vr', 's', 0), ('vr', 'c', 0), ('vr', 'd', 0), ('vr', 'q', 0), ('vr', 'esc', 1),
    ('connect', None, 1), ('running', 's', 1), ('cleanup', None, 1), ('idle', 'esc', 1),
    ('normal', None, 0), ('recorded', None, 0), ('no_quit', None, 1),
    ('recording_cleanup', None, 1), ('audio_cleanup', None, 0),
    ('audio_interrupt', None, 1), ('audio_system_exit', None, 1),
    ('running_audio_cleanup', None, 1), ('recording_audio_cleanup', None, 1),
])
def test_teleop_startup_cancel_has_one_exit_path(
    monkeypatch, tmp_path, stage, key, exit_code, published_camera, resource_logs,
):
    import threading

    import numpy as np

    from dexmani_real.config.real import RealConfig
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.teleop import run

    monkeypatch.setattr(run, 'configure_logging', lambda: None)
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
            closed.append(self.name)

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
    published_camera.is_running = SimpleNamespace(value=True)
    published_camera.vr_ring = SimpleNamespace(latest_sequence=1)
    monkeypatch.setattr(run, 'SensorChannels', lambda *a, **kw: published_camera)
    calibration = tmp_path / 'cameras.json'
    calibration.write_text('{"other-camera": {}}')
    monkeypatch.setattr('dexmani_real.utils.camera_calibrate.extrinsics.CAMERAS_PATH', calibration)
    monkeypatch.setattr(run, 'Robot', Robot)
    monkeypatch.setattr(run, 'load_vr_transform', lambda path: np.eye(3))
    monkeypatch.setattr(run.ActionMapper, 'for_mode', lambda *a, **kw: None)
    monkeypatch.setattr(run, 'build_hand_retargeter', lambda runtime: None)
    monkeypatch.setattr(run, 'VRController', lambda *args: SimpleNamespace(hand_retargeter=None))
    def close_audio():
        closed.append('audio')
        if stage.endswith('audio_cleanup'):
            raise OSError('audio close failed')
        if stage == 'audio_interrupt':
            raise KeyboardInterrupt('audio interrupted')
        if stage == 'audio_system_exit':
            raise SystemExit('audio exited')
    monkeypatch.setattr(run, 'AudioFeedback', lambda: SimpleNamespace(play=sounds.append, close=close_audio))
    def finish_recording(self):
        closed.append('recording')
        if stage in {'recording_cleanup', 'recording_audio_cleanup'}:
            raise OSError('recording finish failed')
    monkeypatch.setattr(run.TeleopLoop, 'finish_recording', finish_recording)
    run_loop_idle = run.TeleopLoop.run
    def run_loop(self):
        if stage == 'idle':
            self.keyboard.handle_key(key, True)
            run_loop_idle(self)
            return
        if stage in {'normal', 'recorded', 'cleanup', 'recording_cleanup', 'audio_cleanup',
                     'audio_interrupt', 'audio_system_exit', 'recording_audio_cleanup'}:
            if stage == 'recorded':
                assert self.camera_snapshot.serial == 'actual-camera'
            self.runtime_state.request_quit()
            return
        if stage == 'no_quit':
            return
        if stage in {'running', 'running_audio_cleanup'}:
            self.runtime_state.request_stop()  # 之前一段的正常 S 不能授权吞掉新故障。
            raise OperationCancelled('unexpected running interruption')
        pytest.fail('cancelled startup entered control')
    monkeypatch.setattr(run.TeleopLoop, 'run', run_loop)
    result = run.run_teleop(RealConfig(), record=stage == 'recorded', episodes_dir='unused', max_duration_s=1,
                            runtime_config_json=None)
    assert result == exit_code
    audio_reports = [record for record in resource_logs.records if 'teleop audio cleanup failed' in record.message]
    if 'audio' in stage:
        assert len(audio_reports) == 1
        assert audio_reports[0].levelname == ('WARNING' if stage.endswith('audio_cleanup') else 'ERROR')
        assert audio_reports[0].exc_info is not None
    if stage == 'audio_interrupt':
        assert keyboards[0].estop_callback.__self__.estop_request
    if stage in {'connect', 'running', 'cleanup'} or key == 'esc':
        assert 'emergency' in sounds
    assert 'robot' in closed and 'keyboard' in closed and 'audio' in closed
    assert closed.index('robot') < closed.index('recording') < closed.index('audio')
    if stage == 'vr':
        assert 'vr' in closed


@pytest.mark.parametrize('case', [
    'primary', 'primary_interrupt', 'primary_stop', 'primary_stop_interrupt',
    'stop', 'finish_interrupt', 'runtime_interrupt',
])
def test_teleop_reports_primary_once_and_preserves_cleanup_failures(monkeypatch, resource_logs, case):
    import numpy as np
    from dexmani_real.config.real import RealConfig
    from dexmani_real.teleop import run
    from dexmani_real.teleop.loop import TeleopPhase

    monkeypatch.setattr(run, 'configure_logging', lambda: None)
    cause = ValueError('original cause')
    context = ValueError('original context')
    primary = KeyboardInterrupt('interrupted') if case == 'primary_interrupt' else OSError('failed')
    cleanup = KeyboardInterrupt('interrupted') if case.endswith('interrupt') else OSError('failed')
    loops, finishes, closed = [], [], []

    class Robot:
        def __init__(self, *a, runtime_state, **kw):
            self.state = runtime_state
        def connect(self):
            pass
        def check(self):
            pass
        def stop(self):
            assert self.state.stop_signal.is_set()
            if case == 'primary_interrupt':
                assert self.state.estop_request
            if case in {'primary_stop', 'primary_stop_interrupt', 'stop'}:
                raise cleanup
        def close(self):
            closed.append('robot')
            if case == 'runtime_interrupt':
                raise cleanup
            if case in {'primary_interrupt', 'primary_stop_interrupt', 'finish_interrupt'}:
                assert self.state.estop_request

    def stop_keyboard():
        closed.append('keyboard')
        if case.endswith('interrupt'):
            assert loops[0].runtime_state.estop_request

    monkeypatch.setattr(run, 'Robot', Robot)
    monkeypatch.setattr(run, 'Keyboard', lambda **kw: SimpleNamespace(
        bindings=kw['bindings'], start=lambda: None, stop=stop_keyboard, check_health=lambda: True,
    ))
    monkeypatch.setattr(run, 'SensorChannels', lambda *a, **kw: None)
    monkeypatch.setattr(run, 'AudioFeedback', lambda: SimpleNamespace(play=lambda sound: None, close=lambda: None))
    monkeypatch.setattr(run, 'load_vr_transform', lambda path: np.eye(3))
    monkeypatch.setattr(run.ActionMapper, 'for_mode', lambda *a, **kw: None)
    monkeypatch.setattr(run, 'build_hand_retargeter', lambda runtime: None)
    monkeypatch.setattr(run, 'VRController', lambda *a: SimpleNamespace(clear=lambda: None))
    monkeypatch.setattr(run, 'initialize_teleop', lambda *a, **kw: None)

    def run_loop(self):
        loops.append(self)
        def start(epoch):
            self.phase = TeleopPhase.ACTIVE
            if case in {'stop', 'finish_interrupt'}:
                self.request_episode_end('operator')
        def fail_tick():
            try:
                raise context
            except ValueError:
                raise primary from cause
        self.start_episode = start
        self.service_episode = lambda timeout: case not in {'stop', 'finish_interrupt'}
        self.run_control_tick = fail_tick
        self.run_episode(self.runtime_state.cancel_epoch)

    def finish(self):
        finishes.append((self.pending_termination_reason, list(self.termination_details)))
        if case == 'finish_interrupt' and len(finishes) == 1:
            raise cleanup

    monkeypatch.setattr(run.TeleopLoop, 'run', run_loop)
    monkeypatch.setattr(run.TeleopLoop, 'finish_recording', finish)
    assert run.run_teleop(RealConfig(), record=False, episodes_dir='unused', max_duration_s=1,
                          runtime_config_json=None) == 1
    details = [item for item in loops[0].termination_details if 'exception_type' in item]
    expected = [cleanup] if case in {'stop', 'finish_interrupt'} else [primary]
    if case in {'primary_stop', 'primary_stop_interrupt', 'runtime_interrupt'}:
        expected.append(cleanup)
    assert len(details) == len(expected)
    reports = [record.exc_info[1] for record in resource_logs.records if record.exc_info]
    assert len(reports) == len(expected)
    assert all(sum(error is reported for reported in reports) == 1 for error in expected)
    if primary in expected:
        assert primary.__cause__ is cause and primary.__context__ is context
    assert finishes[-1] == ('failed', loops[0].termination_details)
    assert closed == ['robot', 'keyboard']


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


@pytest.mark.parametrize('task_name,valid_directory', [
    ('.research', True), (' research', True), ('research ', True),
    ('.', False), ('..', False), ('a/b', False), ('a\\b', False), ('', False), ('a\n', False),
])
@pytest.mark.parametrize('record', [False, True])
def test_task_directory_rules_only_apply_to_recording(monkeypatch, task_name, valid_directory, record):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.teleop import run
    class ReachedResourceSetup(Exception):
        pass
    def load(path):
        raise ReachedResourceSetup()
    monkeypatch.setattr(run, 'load_vr_transform', load)
    with pytest.raises(ValueError if record and not valid_directory else ReachedResourceSetup):
        run.run_teleop(RealConfig(), record=record, task_name=task_name,
                      episodes_dir='unused', max_duration_s=1, runtime_config_json=None)


@pytest.fixture
def published_camera():
    import json
    color = dict(width=16, height=12, fx=100., fy=100., ppx=8., ppy=6.,
                 distortion_model='none', distortion_coeffs=[0.] * 5)
    return SimpleNamespace(camera_ring=SimpleNamespace(latest_sequence=1),
        camera_serial=SimpleNamespace(value=b'actual-camera'),
        camera_intrinsics=SimpleNamespace(value=json.dumps(color).encode()),
        camera_depth_scale=SimpleNamespace(value=.001))


@pytest.mark.parametrize('mode', ['optional', 'required', 'excluded'])
def test_camera_snapshot_uses_published_serial(monkeypatch, mode, published_camera):
    import json
    import numpy as np
    from dexmani_real.ipc import sensors
    channels = published_camera
    color = json.loads(channels.camera_intrinsics.value)
    calls = []
    def load(serial, path):
        calls.append((serial, path))
        return np.eye(4)
    monkeypatch.setattr(sensors, 'load_camera_extrinsics', load)
    monkeypatch.setattr(sensors, 'load_optional_camera_extrinsics', load)
    snapshot = sensors.snapshot_camera_parameters(channels, 'calibration.json',
        require_extrinsics=mode == 'required', include_extrinsics=mode != 'excluded')
    assert snapshot.serial == 'actual-camera' and snapshot.color.width == 16
    assert calls == ([] if mode == 'excluded' else [('actual-camera', 'calibration.json')])
    channels.camera_ring.latest_sequence = 0
    with pytest.raises(ValueError, match='published camera frame'):
        sensors.snapshot_camera_parameters(channels, None)
    channels.camera_ring.latest_sequence = 1
    channels.camera_depth_scale.value = 0
    with pytest.raises(ValueError, match='depth_scale'):
        sensors.snapshot_camera_parameters(channels, None, include_extrinsics=False)
    channels.camera_depth_scale.value = .001
    color['fx'] = 0
    channels.camera_intrinsics.value = json.dumps(color).encode()
    with pytest.raises(ValueError, match='focal lengths'):
        sensors.snapshot_camera_parameters(channels, None, include_extrinsics=False)


def test_keyboard_events_keep_order_and_deduplicate_presses():
    keyboard = Keyboard()
    keyboard.active = True
    keyboard.listener = SimpleNamespace(is_alive=lambda: True)
    for key in ['x', 'w', 'w', 's']:
        keyboard.handle_key(key, True)
    keyboard.handle_key('w', False)
    keyboard.handle_key('w', True)
    events, stopped = keyboard.poll(SessionState(), timeout=0)
    assert events == ['x', 'w', 's', 'w'] and not stopped
    assert keyboard.snapshot() == frozenset({'x', 'w', 's'})


@pytest.mark.parametrize('extrinsics', [False, True])
@pytest.mark.parametrize('record_raw', [False, True])
@pytest.mark.parametrize('calibration_state', ['valid', 'missing_file', 'missing_serial'])
def test_policy_camera_startup_uses_snapshot_interface(
    monkeypatch, tmp_path, published_camera, extrinsics, record_raw, calibration_state,
):
    import json
    import numpy as np
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment import session as deployment
    from dexmani_real.runtime.observation import ObservationNeeds

    calibration = tmp_path / 'camera.json'
    if calibration_state != 'missing_file':
        serial = 'other-camera' if calibration_state == 'missing_serial' else 'actual-camera'
        calibration.write_text(json.dumps({serial: {'T_base_from_camera': np.eye(4).tolist()}}))
    monkeypatch.setattr('dexmani_real.utils.camera_calibrate.extrinsics.CAMERAS_PATH', calibration)
    session = deployment.PolicySession.__new__(deployment.PolicySession)
    session.runtime = RealConfig()
    session.runtime_state = SessionState()
    session.startup_cancel_epoch = session.runtime_state.cancel_epoch
    session.capture_needs = ObservationNeeds(camera_fields=('rgb',))
    session.record_raw = record_raw
    session.info = SimpleNamespace(observation_fields=('camera_extrinsic',) if extrinsics else ())
    session.camera_calibration_path = None
    configured, started = [], []
    session.observations = SimpleNamespace(camera_static=True, configure_camera=configured.append)
    session.check_startup = lambda: None
    session.startup_allowed = lambda: True
    session.home_abort_requested = lambda: False
    session.warmup_policy = lambda: None
    session.robot = SimpleNamespace(connect=lambda: None)
    session.sensors = SimpleNamespace(channels=published_camera,
        start=lambda process, **kw: started.append(process.name))
    session.process_context = SimpleNamespace(Process=lambda **kw: SimpleNamespace(**kw))
    session.run_config, session.output_dir = {}, tmp_path
    if extrinsics and calibration_state != 'valid':
        with pytest.raises(FileNotFoundError if calibration_state == 'missing_file' else KeyError):
            session.start_devices()
        assert started == ['camera'] and not configured
        return
    session.start_devices()
    assert started == ['camera']
    assert configured == [session.camera_snapshot]
    assert session.run_config['effective_camera']['camera_serial'] == 'actual-camera'
    assert (session.camera_snapshot.T_xarm_base_from_color is not None) == (
        calibration_state == 'valid' and (extrinsics or record_raw)
    )


@pytest.mark.parametrize('outcome', ['ready', 'cancel', 'timeout'])
def test_policy_warmup_waits_for_its_real_inputs_without_home(monkeypatch, outcome):
    import numpy as np
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.robot.system import RobotState
    from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
    from dexmani_real.robot.xhand.feedback import XHandFeedback, XHandState
    from dexmani_real.runtime.observation import ObservationNeeds

    now = [1_000_000_000]
    monkeypatch.setattr('time.monotonic_ns', lambda: now[0])
    monkeypatch.setattr('time.monotonic', lambda: now[0] / 1e9)
    monkeypatch.setattr('time.sleep', lambda delay: now.__setitem__(0, now[0] + round(delay * 1e9)))
    session = PolicySession.__new__(PolicySession)
    session.runtime = RealConfig()
    session.runtime_state = SessionState()
    session.startup_cancel_epoch = session.runtime_state.cancel_epoch
    session.control_needs = ObservationNeeds(required_tactile=('aggregate',))
    session.info = SimpleNamespace(n_obs_steps=2)
    session.keyboard = SimpleNamespace(check_health=lambda: True)
    session.sensors = SimpleNamespace(channels=None)
    session.wait_model = lambda deadline: None
    feedback = XHandFeedback()
    measured_hand = np.deg2rad(session.runtime.hand.home_qpos_deg) + .02
    arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
    measured_arm = np.asarray(session.runtime.arm.home_qpos) + .03
    def service():
        if outcome == 'cancel' and now[0] > 1_000_000_000:
            session.runtime_state.request_quit()
        arm['qpos'], arm['timestamp_ns'] = measured_arm, now[0]
        aggregate = np.ones((5, 3)) if outcome == 'ready' and now[0] > 1_000_000_000 else None
        feedback.consume(XHandState(measured_hand, aggregate, None, now[0]))
    session.robot = SimpleNamespace(
        service_feedback=service,
        read_state=lambda: RobotState(arm.copy(), feedback.snapshot()),
    )
    session.observations = SimpleNamespace(project=lambda row: row)
    warmups = []
    session.model = SimpleNamespace(warmup=lambda rows, **kw: warmups.append(rows))
    if outcome == 'ready':
        session.warmup_policy()
        assert len(warmups) == 1
        row = warmups[0][0]
        np.testing.assert_array_equal(row.arm['qpos'][0], measured_arm)
        np.testing.assert_array_equal(row.hand['qpos'][0], measured_hand)
        np.testing.assert_array_equal(row.hand['tactile_aggregate'][0], np.ones((5, 3)))
    else:
        error = OperationCancelled if outcome == 'cancel' else ValueError
        with pytest.raises(error, match='startup cancelled' if outcome == 'cancel' else 'tactile_aggregate'):
            session.warmup_policy()
        assert not warmups


@pytest.mark.parametrize('commands,selected', [
    (['b', 's', 'q', 'd'], 'q'), (['q', 'd', 'esc'], 'esc'),
    (['s', 'd'], 'd'), (['c', 's'], 's'), (['h', 'b'], 'h'),
])
def test_teleop_string_commands_preserve_priority_and_discard(commands, selected):
    from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase
    loop = TeleopLoop.__new__(TeleopLoop)
    loop.phase = TeleopPhase.ACTIVE
    loop.discard_requested = False
    handled = []
    def handle(command):
        handled.append(command)
        return True
    loop.handle_operator_command = handle
    loop.handle_commands(commands)
    assert handled == [selected]
    assert loop.discard_requested == ('d' in commands)

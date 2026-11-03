"""分段意图、代际与键盘关闭竞态；外部设备由最小 fake 替换。"""

from pathlib import Path
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.robot.commands import OperationCancelled
from dexmani_real.runtime.keyboard import Keyboard, KeyEvent
from dexmani_real.runtime.state import RunEndReason, SessionState
from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase
from dexmani_real.teleop.run import initialize_teleop


def make_loop():
    state = SessionState()
    state.arm()
    runtime = SimpleNamespace(
        teleop=SimpleNamespace(control_hz=30, vr_stale_threshold_s=0.2),
        hand=SimpleNamespace(enabled=False),
    )
    robot = Mock()
    robot.prepare_streaming.side_effect = lambda run_id: state.stop_signal.clear()
    controller = Mock(hand_retargeter=None)
    controller.compute_target.return_value = SimpleNamespace(
        rejection_reason=None, arm_qpos=np.zeros(7), hand_qpos=None, eef_pose=None,
    )
    keyboard = Keyboard(get_run_id=lambda: state.run_id, bindings={
        's': state.request_stop, 'c': state.request_stop, 'q': state.request_quit,
    }, estop_callback=state.request_emergency_stop)
    keyboard.active = True
    keyboard.listener = Mock()
    keyboard.listener.is_alive.return_value = True
    loop = TeleopLoop(
        state, runtime, robot, task_label='test', sensors=SimpleNamespace(channels=None),
        controller=controller, recording_dir=None, max_duration_s=100,
        runtime_config_json=None, keyboard=keyboard, audio=Mock(),
    )
    keyboard.bindings['d'] = loop.request_discard
    return loop


def observation():
    return SimpleNamespace(
        observation_timestamp_ns=1, arm={'qpos': np.zeros((1, 7))}, hand=None,
        vr={'wrist_pos': np.zeros(3), 'wrist_quat_wxyz': np.array([1., 0., 0., 0.]),
            'recv_ts_ns': 1},
    )


def begin(loop):
    with patch('dexmani_real.teleop.loop.read_observation', return_value=observation()), \
         patch.object(loop, 'admit_observation', return_value=[]):
        loop.start_episode(loop.runtime_state.run_id)
    assert loop.phase is TeleopPhase.ACTIVE


class Recorder:
    start_attempted = True
    finish_request = None
    published_path = None
    submitted_frames = 1

    def __init__(self, on_finish=lambda: None):
        self.on_finish = on_finish
        self.calls = []

    def finish(self, **kwargs):
        self.calls.append(kwargs)
        self.finish_request = kwargs
        self.on_finish()
        return Path('/tmp/episode') if kwargs['save'] and self.submitted_frames else None


@pytest.mark.parametrize('key', ['s', 'c'])
def test_stop_ends_segment_and_new_b_reanchors(key):
    loop = make_loop()
    begin(loop)
    old_run_id = loop.run_id
    loop.keyboard.handle_key(key, True)
    loop.handle_commands(loop.keyboard.poll(0))
    assert loop.pending_termination_reason == 'operator'
    assert not loop.capture_failed
    loop.end_episode()
    assert loop.phase is TeleopPhase.IDLE
    loop.robot.stop.assert_called_once()
    begin(loop)
    assert loop.run_id != old_run_id
    assert loop.controller.reset.call_count == 2


def test_c_idle_does_not_start_or_create_recorder():
    loop = make_loop()
    loop.handle_commands([KeyEvent('c')])
    assert loop.phase is TeleopPhase.IDLE
    assert loop.recorder is None
    loop.robot.prepare_streaming.assert_not_called()


@pytest.mark.parametrize('key', ['s', 'c', 'd', 'q'])
def test_stop_event_consumption_does_not_repeat_keyboard_revocation(key):
    loop = make_loop()
    begin(loop)
    loop.keyboard.handle_key(key, True)
    revoked_run_id = loop.runtime_state.run_id
    assert loop.runtime_state.stop_signal.is_set()
    loop.handle_commands(loop.keyboard.poll(0))
    assert loop.runtime_state.run_id == revoked_run_id
    assert loop.discard_requested == (key == 'd')
    assert loop.runtime_state.quit_requested == (key == 'q')
    assert loop.pending_termination_reason == ('quit' if key == 'q' else 'operator')
    loop.robot.stop.assert_not_called()


@pytest.mark.parametrize('key', ['s', 'c', 'd', 'q', 'esc'])
def test_prepare_can_be_cancelled_by_keyboard(key):
    loop = make_loop()

    def prepare(run_id):
        loop.runtime_state.stop_signal.clear()
        loop.keyboard.handle_key(key, True)
        raise OperationCancelled('prepare revoked')

    loop.robot.prepare_streaming.side_effect = prepare
    loop.run_episode(loop.runtime_state.run_id)
    assert loop.phase is TeleopPhase.IDLE
    loop.robot.stop.assert_called_once()
    loop.controller.reset.assert_not_called()
    if key in ('s', 'c', 'd'):
        loop.audio.play.assert_not_called()
    assert loop.runtime_state.quit_requested == (key == 'q')
    assert loop.runtime_state.estop_request == (key == 'esc')


@pytest.mark.parametrize('keys', [('d', 'q'), ('q', 'd')])
def test_discard_and_single_q_do_not_publish(keys):
    loop = make_loop()
    begin(loop)
    recorder = loop.recorder = Recorder()
    for key in keys:
        loop.keyboard.handle_key(key, True)
    loop.handle_commands(loop.keyboard.poll(0))
    assert loop.pending_termination_reason == ('operator' if keys[0] == 'd' else 'quit')
    assert not loop.capture_failed
    loop.end_episode()
    assert loop.runtime_state.quit_requested
    assert recorder.calls[0]['save'] is False
    assert 'save' not in [call.args[0] for call in loop.audio.play.call_args_list]


def test_save_decision_precedes_late_discard_and_revokes_preparation_keys():
    loop = make_loop()
    begin(loop)

    def during_finish():
        loop.request_discard()
        for key in ('b', 'h', 't'):
            loop.keyboard.handle_key(key, True)

    recorder = loop.recorder = Recorder(during_finish)
    loop.keyboard.handle_key('s', True)
    loop.handle_commands(loop.keyboard.poll(0))
    loop.end_episode()
    assert recorder.calls[0]['save'] is True
    prompts = [call.args[0] for call in loop.audio.play.call_args_list]
    assert 'save' in prompts and 'discard' not in prompts
    loop.handle_commands(loop.keyboard.poll(0))
    assert loop.phase is TeleopPhase.IDLE


def test_discard_without_recording_stops_and_new_episode_resets_intent():
    loop = make_loop()
    loop.runtime_state.stop_signal.clear()
    loop.request_discard()
    assert loop.runtime_state.stop_signal.is_set()
    loop.robot.stop.assert_not_called()
    begin(loop)
    loop.keyboard.handle_key('d', True)
    loop.handle_commands(loop.keyboard.poll(0))
    loop.end_episode()
    begin(loop)
    recorder = loop.recorder = Recorder()
    loop.keyboard.handle_key('s', True)
    loop.handle_commands(loop.keyboard.poll(0))
    loop.end_episode()
    assert recorder.calls[0]['save'] is True


def test_h_active_does_not_stop_or_home():
    loop = make_loop()
    begin(loop)
    with patch('dexmani_real.teleop.loop.home_robot') as home:
        loop.handle_commands([KeyEvent('h', loop.runtime_state.run_id)])
    assert loop.phase is TeleopPhase.ACTIVE
    loop.robot.stop.assert_not_called()
    home.assert_not_called()


def test_old_start_intent_cannot_reauthorize():
    loop = make_loop()
    run_id = loop.runtime_state.run_id
    loop.runtime_state.request_stop()
    loop.start_episode(run_id)
    assert loop.phase is TeleopPhase.IDLE
    loop.robot.prepare_streaming.assert_not_called()


def test_keyboard_stop_allows_admitted_stop_callback_without_queueing():
    entered, release = Event(), Event()
    stopped = Mock()

    def get_run_id():
        entered.set()
        assert release.wait(2)
        return 1

    keyboard = Keyboard(get_run_id=get_run_id, bindings={'b': stopped})
    keyboard.active = True
    worker = Thread(target=keyboard.handle_key, args=('b', True))
    worker.start()
    assert entered.wait(2)
    keyboard.stop()
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    stopped.assert_called_once()
    assert not keyboard.events and not keyboard.held_keys
    keyboard.handle_key('b', True)
    stopped.assert_called_once()


def test_primary_error_survives_stop_and_writer_failures(caplog):
    loop = make_loop()
    primary = ValueError('mapping failed')
    writer = OSError('writer close failed')
    recorder = loop.recorder = Recorder()
    recorder.finish = Mock(side_effect=writer)

    def fail_start(run_id):
        loop.run_id = run_id
        loop.phase = TeleopPhase.PREPARING
        raise primary

    loop.start_episode = fail_start
    stop_error = loop.robot.stop.side_effect = RuntimeError('stop failed')
    with pytest.raises(ValueError) as raised:
        loop.run_episode(loop.runtime_state.run_id)
    assert raised.value is primary
    assert raised.traceback[-1].name == 'fail_start'
    for error in (stop_error, writer):
        assert sum(record.exc_info is not None and record.exc_info[1] is error for record in caplog.records) == 1
    assert not hasattr(primary, 'cleanup_errors')
    recorder.finish.assert_called_once()
    assert loop.runtime_state.error_state
    assert {item['stage'] for item in loop.termination_details} >= {
        'episode', 'stop', 'recording_cleanup',
    }


def test_recording_primary_is_not_reported_again_by_finish(tmp_path):
    loop = make_loop()
    begin(loop)
    recorder = loop.recorder = EpisodeRecorder(tmp_path, control_hz=30)
    recorder.start_attempted = True
    original = recorder.error = OSError('writer failed')
    with pytest.raises(RecordingError) as raised:
        loop.run_episode(loop.run_id)
    assert raised.value is recorder.error
    assert raised.value.__cause__ is original
    assert recorder.finish_request is not None
    assert [item['stage'] for item in loop.termination_details if 'exception_type' in item] == ['episode']
    loop.robot.stop.assert_called_once()


def test_discard_is_latched_before_stop_can_trigger_finish():
    loop = make_loop()
    begin(loop)
    recorder = loop.recorder = Recorder()
    stopped, release, finishing, finished = Event(), Event(), Event(), Event()
    original_set = loop.runtime_state.stop_signal.set

    def stop_at_boundary():
        original_set()
        if not stopped.is_set():
            stopped.set()
            assert release.wait(2)

    def finish():
        finishing.set()
        loop.finish_recording()
        finished.set()

    with patch.object(loop.runtime_state.stop_signal, 'set', side_effect=stop_at_boundary):
        callback = Thread(target=loop.request_discard)
        callback.start()
        control = Thread(target=finish)
        try:
            assert stopped.wait(2)
            control.start()
            assert finishing.wait(2)
            # 首次停止信号暴露时，D 必须仍持有保存决定使用的锁。
            acquired = loop.runtime_state.motion_lock.acquire(blocking=False)
            if acquired:
                loop.runtime_state.motion_lock.release()
            assert not acquired
        finally:
            release.set()
            callback.join(2)
            if control.ident is not None:
                control.join(2)
    assert not callback.is_alive() and not control.is_alive()
    assert finished.is_set()
    assert recorder.calls[0]['save'] is False
    prompts = [call.args[0] for call in loop.audio.play.call_args_list]
    assert 'discard' in prompts and 'save' not in prompts


@pytest.mark.parametrize('boundary', ['finish', 'startup', 'home', 'tactile'])
@pytest.mark.parametrize('key', ['b', 'h', 't'])
def test_preparation_event_delayed_until_after_boundary_is_revoked(boundary, key):
    loop = make_loop()
    captured, release = Event(), Event()
    workers = []

    def delayed_generation():
        run_id = loop.runtime_state.run_id
        captured.set()
        assert release.wait(2)
        return run_id

    def input_at_boundary():
        loop.keyboard.get_run_id = delayed_generation
        worker = Thread(target=loop.keyboard.handle_key, args=(key, True))
        workers.append(worker)
        worker.start()
        assert captured.wait(2)
        return True

    try:
        if boundary == 'finish':
            begin(loop)
            loop.recorder = Recorder(input_at_boundary)
            loop.keyboard.handle_key('s', True)
            loop.handle_commands(loop.keyboard.poll(0))
            loop.end_episode()
        elif boundary == 'startup':
            loop.sensors.channels = SimpleNamespace(vr_ring=SimpleNamespace(latest_sequence=1))
            loop.sensors.start = Mock()
            loop.sensors.check = input_at_boundary
            with patch('dexmani_real.teleop.run.initialize_hand'):
                assert initialize_teleop(loop, Mock(), run_id=loop.runtime_state.run_id)
        elif boundary == 'home':
            with patch('dexmani_real.teleop.loop.home_robot', side_effect=lambda *a, **kw: input_at_boundary()):
                loop.handle_operator_command(KeyEvent('h', loop.runtime_state.run_id))
        else:
            loop.runtime.hand.enabled = True
            loop.robot.reset_tactile_sensors.side_effect = lambda **kwargs: input_at_boundary()
            loop.handle_operator_command(KeyEvent('t', loop.runtime_state.run_id))
    finally:
        release.set()
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()
    queued = loop.keyboard.poll(0)
    assert len(queued) == 1 and queued[0].key == key
    assert queued[0].run_id != loop.runtime_state.run_id
    with patch.object(loop, 'run_episode') as run, patch('dexmani_real.teleop.loop.home_robot') as home:
        loop.handle_commands(queued)
        if key == 'b':
            # B 的最终代际准入在 start_episode，不能用单纯的 mock 冒充该检查。
            loop.start_episode(queued[0].run_id)
            assert loop.phase is TeleopPhase.IDLE
        else:
            run.assert_not_called()
        home.assert_not_called()
    begin(loop)


def test_target_rejection_without_recording_reports_reason_once(caplog):
    loop = make_loop()
    begin(loop)
    caplog.set_level('INFO', logger='dexmani_real.teleop.loop')
    loop.controller.compute_target.return_value = SimpleNamespace(
        rejection_reason='hand solver did not improve the committed-command seed', arm_qpos=None,
    )
    loop.execute_control_step(observation(), tick_end_ns=10**18)
    loop.robot.send_action.assert_not_called()
    assert loop.recorder is None
    loop.end_episode()
    messages = [record.getMessage() for record in caplog.records if '本段结束' in record.getMessage()]
    assert len(messages) == 1
    assert 'target_unavailable: hand solver did not improve the committed-command seed' in messages[0]


def test_revocation_returns_first_reason_without_revoking_new_run():
    loop = make_loop()
    begin(loop)
    state, run_id = loop.runtime_state, loop.run_id
    state.request_stop()
    assert state.revoke_motion(reason=RunEndReason.HARDWARE_FAULT,
                               expected_run_id=run_id) == RunEndReason.OPERATOR
    loop.request_episode_end('late SDK error', run_end_reason=RunEndReason.HARDWARE_FAULT)
    loop.end_episode()
    begin(loop)
    current = state.run_id
    state.revoke_motion(expected_run_id=run_id)
    assert state.run_id == current and state.can_move(run_id=current)


def test_emergency_signals_precede_motion_lock():
    state = SessionState()
    worker = Thread(target=state.request_emergency_stop)
    with state.motion_lock:
        worker.start()
        assert state.emergency_signal.wait(2)
        assert state.stop_signal.wait(2)
        assert state.estop_request  # 共享急停事实不等待本地锁。
    worker.join(2)
    assert not worker.is_alive() and state.estop_request



@pytest.mark.parametrize('read_ms,send', [(20, True), (62.5, False), (80, False)])
def test_first_control_read_uses_start_clock(monkeypatch, read_ms, send):
    loop = make_loop()
    begin(loop)
    loop.control_dt_ns = 62_500_000
    clock = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.teleop.loop.time.monotonic_ns', lambda: clock[0])
    def read(*args, **kwargs):
        row = observation()
        row.control_timestamp_ns = clock[0]
        clock[0] += int(read_ms * 1e6)
        row.observation_timestamp_ns = clock[0]
        return row
    monkeypatch.setattr('dexmani_real.teleop.loop.read_observation', read)
    loop.admit_observation = lambda row: []
    loop.run_control_tick()
    assert loop.robot.send_action.called == send
    assert loop.controller.compute_target.called == send
    assert loop.next_tick_ns == 1_000_000_000 + (62_500_000 if send else 125_000_000)
    if send:
        clock[0] = 1_000_000_000 + 3 * 62_500_000 + 1_000_000
        loop.run_control_tick()
        assert loop.robot.send_action.call_count == 2
        assert loop.next_tick_ns == 1_250_000_000


@pytest.mark.parametrize('old_key', ['b', 'h', 't'])
def test_old_preparation_does_not_consume_fresh_start(old_key):
    loop = make_loop()
    run_id = loop.runtime_state.run_id
    loop.run_episode = Mock()
    loop.handle_commands([KeyEvent(old_key, run_id-1), KeyEvent('b', run_id)])
    loop.run_episode.assert_called_once_with(run_id)


@pytest.mark.parametrize('cause', ['deadline', 'stop'])
def test_target_computation_cannot_submit_after_revocation_or_deadline(monkeypatch, cause):
    loop = make_loop()
    begin(loop)
    clock = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.teleop.loop.time.monotonic_ns', lambda: clock[0])
    target = loop.controller.compute_target.return_value
    def compute(*args, **kwargs):
        if cause == 'deadline':
            clock[0] += 100_000_000
        else:
            loop.runtime_state.request_stop()
        return target
    loop.controller.compute_target.side_effect = compute
    loop.execute_control_step(observation(), tick_end_ns=clock[0] + 62_500_000)
    loop.robot.send_action.assert_not_called()
    loop.controller.commit.assert_not_called()


@pytest.mark.parametrize('recording', [True, False])
def test_record_duration_only_applies_to_recording(monkeypatch, recording):
    loop = make_loop()
    begin(loop)
    loop.recorder = Mock(is_recording=recording)
    loop.capture_deadline_ns = 100
    loop.next_tick_ns = 1000
    monkeypatch.setattr('dexmani_real.teleop.loop.time.monotonic_ns', lambda: 100)
    loop.run_control_tick()
    assert loop.pending_termination_reason == ('duration_complete' if recording else None)


@pytest.mark.parametrize('key', ['s', 'c', 'd', 'q'])
def test_blocking_poll_handles_stop_before_event_is_enqueued(key):
    loop = make_loop()
    begin(loop)
    stopped, release = Event(), Event()
    callback = loop.keyboard.bindings[key]
    def delayed_callback():
        callback()
        stopped.set()
        assert release.wait(2)
    loop.keyboard.bindings[key] = delayed_callback
    thread = Thread(target=loop.keyboard.handle_key, args=(key, True))
    thread.start()
    try:
        assert stopped.wait(2)
        assert loop.poll_blocking_commands()
        assert not loop.capture_failed and loop.pending_termination_reason is None
        assert loop.service_episode(0) == (key != 'q')
        assert not loop.capture_failed
    finally:
        release.set()
        thread.join(2)
    assert not thread.is_alive()
    assert loop.poll_blocking_commands()
    assert not loop.capture_failed
    loop.end_episode()


@pytest.mark.parametrize('phase', [TeleopPhase.IDLE, TeleopPhase.PREPARING, TeleopPhase.ACTIVE])
def test_blocking_poll_drops_preparation_keys_without_termination(phase):
    loop = make_loop()
    if phase is not TeleopPhase.IDLE:
        begin(loop)
        loop.phase = phase
    loop.runtime.hand.enabled = True
    for key in ('b', 'h', 't'):
        loop.keyboard.handle_key(key, True)
    assert not loop.poll_blocking_commands()
    assert not loop.keyboard.poll(0)
    loop.robot.reset_tactile_sensors.assert_not_called()
    assert loop.robot.prepare_streaming.call_count == (phase is not TeleopPhase.IDLE)

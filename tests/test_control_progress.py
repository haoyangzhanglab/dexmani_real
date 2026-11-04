from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.keyboard import Keyboard
from dexmani_real.runtime.observation import Observation, ObservationNeeds
from dexmani_real.runtime.state import SessionState


def make_rollout(monkeypatch, *, mode='sync', history=1):
    clock = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.deployment.rollout.time.monotonic_ns', lambda: clock[0])
    sends = []
    runtime = RealConfig()
    runtime = replace(runtime, execution=replace(runtime.execution, mode=mode, action_steps=4, prefetch_steps=1))
    rollout = Rollout(SessionState(), runtime,
                      SimpleNamespace(control_dt_s=.0625, n_obs_steps=history), cancel_epoch=1,
                      robot=SimpleNamespace(arm=SimpleNamespace(is_connected=True), stop=lambda: None, check=lambda: None, send_action=lambda *a, **kw: sends.append(clock[0])),
                      observations=SimpleNamespace(project=lambda row: row, needs=ObservationNeeds()),
                      record_needs=ObservationNeeds(),
                      sensors=None, action_mapper=None,
                      model=SimpleNamespace(discard_pending=lambda timeout: None, future=None, poll=lambda: None, predict=lambda *a, **kw: None),
                      max_running_s=None, index=1, output_dir='/tmp', record_raw=False)
    rollout.started_ns = rollout.next_step_ns = clock[0]

    def observe():
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm['timestamp_ns'] = hand['timestamp_ns'] = clock[0]
        return Observation(arm, hand, None, None, None, clock[0], control_timestamp_ns=clock[0]), None

    rollout.read_observation = observe
    return rollout, clock, sends


def close_rollout_session(rollout, *, failure=None):
    from concurrent.futures import Future
    from dexmani_real.deployment.session import PolicySession
    session = PolicySession.__new__(PolicySession)
    session.runtime_state = rollout.runtime_state
    session.robot, session.episode, session.model = rollout.robot, rollout, rollout.model
    session.sensors = session.keyboard = None
    session.errors = []
    session.summary = {'termination_reason': None}
    session.save_results = lambda: None
    def close_model():
        future = Future()
        try:
            session.model.discard_pending(0)
        except BaseException as exc:
            future.set_exception(exc)
        else:
            future.set_result(None)
        return future
    session.model.close = close_model
    return session.close(failure=failure)


def test_slow_sync_mapping_submits_and_advances(monkeypatch):
    rollout, clock, sends = make_rollout(monkeypatch)
    rollout.sync_actions = np.zeros((4, 19))

    def map_slow(action, row):
        clock[0] += 63_000_000
        return RobotCommand(arm_qpos=np.zeros(7)), None

    rollout.map_command = map_slow
    for _ in range(4):
        clock[0] = max(clock[0], rollout.next_step_ns)
        rollout.step()
    assert len(sends) == 4
    assert rollout.cursor == 4
    assert np.diff(sends).tolist() == [63_000_000] * 3


def test_async_history_keeps_recent_accepted_samples(monkeypatch):
    rollout, clock, _ = make_rollout(monkeypatch, mode='async', history=2)
    origin = clock[0]
    for delta in (61_000_000, 125_100_000):
        clock[0] = origin + delta
        rollout.step()
    assert len(rollout.history) == 2
    # 调度间隔变大仍保留最近两次实际采样。
    clock[0] += 120_000_000
    rollout.step()
    assert len(rollout.history) == 2
    assert rollout.history[0].control_timestamp_ns == origin + 125_100_000


def test_duration_stops_waiting_model(monkeypatch):
    rollout, clock, sends = make_rollout(monkeypatch)
    rollout.run_deadline_ns = clock[0]
    rollout.step()
    assert rollout.result.termination_reason == 'duration_complete'
    assert not sends


def test_completed_none_prediction_is_invalid_not_pending(monkeypatch):
    from concurrent.futures import Future
    rollout, _, _ = make_rollout(monkeypatch)
    rollout.policy_info.future_steps = 4
    rollout.policy_info.control_action_dim = 19
    done = Future()
    done.set_result(None)
    rollout.model.poll = lambda: done
    with pytest.raises(ValueError, match='prediction requires finite shape'):
        rollout.poll_model()


def test_fault_closes_robot_before_slow_resources(monkeypatch):
    rollout, _, _ = make_rollout(monkeypatch)
    order = []
    rollout.runtime_state.quit_requested = False
    rollout.robot.stop = lambda: order.append('stop')
    rollout.robot.close = lambda: order.append('robot_close')
    rollout.prepare = lambda poll: (_ for _ in ()).throw(ValueError('prepare failed'))
    rollout.finish_recording = lambda: order.append('writer_finish')
    rollout.model.discard_pending = lambda timeout: order.append('model_finish')
    with pytest.raises(ValueError, match='prepare failed'):
        rollout.run(poll_operator=lambda: None)
    assert order == ['stop']
    assert close_rollout_session(rollout, failure=ValueError('prepare failed')) == 1
    assert order == ['stop', 'robot_close', 'writer_finish', 'model_finish']
    assert rollout.result.errors[0]['message'] == 'prepare failed'


def test_vr_expiring_during_mapping_cannot_receive_new_lease(monkeypatch):
    from dexmani_real.runtime.state import SessionState
    from dexmani_real.teleop.loop import TeleopLoop
    rollout, clock, sends = make_rollout(monkeypatch)
    row, _ = rollout.read_observation()
    row = replace(row, vr={'recv_ts_ns': clock[0]})
    loop = TeleopLoop.__new__(TeleopLoop)
    loop.runtime = rollout.runtime
    loop.runtime_state = SessionState()
    loop.cancel_epoch = loop.runtime_state.cancel_epoch
    loop.control_needs = ObservationNeeds(vr=True)
    loop.recorder = None
    loop.robot = rollout.robot
    ended = []
    loop.request_episode_end = lambda reason, **kwargs: ended.append(reason)
    def compute(*args, **kwargs):
        clock[0] += 600_000_000
        # 独立设备反馈仍新鲜，唯独此次候选依赖的 VR 已过期。
        row.arm['timestamp_ns'] = row.hand['timestamp_ns'] = clock[0]
        return SimpleNamespace(rejection_reason=None, arm_qpos=np.zeros(7), hand_qpos=np.zeros(12))
    loop.controller = SimpleNamespace(compute_target=compute)
    loop.execute_control_step(row)
    assert not sends
    assert ended == ['control_sources: vr_stale']


def test_stop_failure_survives_successful_close(monkeypatch):
    from dexmani_real.runtime.state import RunEndReason
    rollout, _, _ = make_rollout(monkeypatch)
    order = []
    rollout.runtime_state.quit_requested = False
    def failed_stop():
        order.append('stop')
        raise RuntimeError('stop failed')
    rollout.robot.stop = failed_stop
    rollout.robot.close = lambda: order.append('close')
    rollout.prepare = lambda poll: rollout.stop('operator', run_end_reason=RunEndReason.OPERATOR)
    rollout.finish_recording = lambda: order.append('writer')
    rollout.model.discard_pending = lambda timeout: order.append('model')
    with pytest.raises(RuntimeError, match='stop failed'):
        rollout.run(poll_operator=lambda: None)
    assert order == ['stop']
    assert close_rollout_session(rollout, failure=RuntimeError('stop failed')) == 1
    assert order == ['stop', 'close', 'writer', 'model']
    assert rollout.result.errors[0]['stage'] == 'stop'


@pytest.mark.parametrize('loss', ['missing', 'stale'])
def test_archive_source_loss_ends_recorded_segment(monkeypatch, tmp_path, loss):
    from dexmani_real.data.recorder import EpisodeRecorder
    rollout, clock, sends = make_rollout(monkeypatch)
    rollout.recorder = EpisodeRecorder(tmp_path, control_hz=16)
    rollout.recorder.is_recording = True
    rollout.record_needs = ObservationNeeds(camera_fields=('rgb', 'depth'))
    rollout.sensors = SimpleNamespace(check=lambda **kwargs: True)
    rollout.sync_actions = np.zeros((4, 19))
    observe = rollout.read_observation
    available = [True]
    def read():
        row, issue = observe()
        camera = dict(timestamp_ns=clock[0], ring_sequence=1,
                      rgb=np.zeros((4, 4, 3), dtype=np.uint8), depth=np.zeros((4, 4), dtype=np.uint16))
        if not available[0]:
            if loss == 'stale':
                camera['timestamp_ns'] -= 200_000_000
            elif loss == 'missing':
                camera = None
        return replace(row, camera=camera), issue
    rollout.read_observation = read
    rollout.map_command = lambda *args: (RobotCommand(arm_qpos=np.zeros(7)), None)
    for present in (True, False, True):
        available[0] = present
        clock[0] = max(clock[0], rollout.next_step_ns)
        rollout.step()
    assert len(sends) == 1
    assert rollout.finished
    assert rollout.recorder.queue.qsize() == 1
    assert rollout.recorder.recording_truncated_reason == {
        'missing': 'camera_source_unavailable_or_unmatched', 'stale': 'camera_source_stale',
    }[loss]
    assert rollout.recorder.finish_request is None


@pytest.mark.parametrize('mapping_ns', [63_000_000, 200_000_000])
def test_jog_late_candidate_depends_on_source_age_not_tick(monkeypatch, mapping_ns):
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.runtime.jog import CartesianJog
    from dexmani_real.runtime.state import SessionState
    rollout, clock, sends = make_rollout(monkeypatch)
    row, _ = rollout.read_observation()
    monkeypatch.setattr('dexmani_real.runtime.jog.read_observation', lambda *a: row)
    state = SessionState()
    jog = CartesianJog(None, SimpleNamespace(target_max_age_s=.5), (None, None))
    def propose(*args, **kwargs):
        clock[0] += mapping_ns
        return np.zeros(7), SimpleNamespace(qpos=np.zeros(7))
    jog.propose = propose
    keys = SimpleNamespace(snapshot=lambda: frozenset({'w'}))
    if mapping_ns > 133_333_333:
        with pytest.raises(OperationCancelled, match='mapping sources expired'):
            jog.step(rollout.robot, keys, state, rollout.runtime)
        assert not sends
    else:
        jog.step(rollout.robot, keys, state, rollout.runtime)
        assert sends == [clock[0]]


def test_teleop_slow_tick_commits_then_quits_through_episode_cleanup(monkeypatch):
    from dexmani_real.runtime.keyboard import Keyboard
    from dexmani_real.runtime.state import SessionState
    from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase
    rollout, clock, _ = make_rollout(monkeypatch)
    state = SessionState()
    keyboard = Keyboard(bindings={'q': state.request_quit})
    keyboard.active = True
    keyboard.listener = SimpleNamespace(is_alive=lambda: True)
    row, _ = rollout.read_observation()
    row = replace(row, vr=dict(recv_ts_ns=clock[0], wrist_pos=np.zeros(3),
                               wrist_quat_wxyz=np.array([1., 0, 0, 0])))
    monkeypatch.setattr('dexmani_real.teleop.loop.read_observation', lambda *a, **kw: row)
    order = []
    robot = SimpleNamespace(arm=SimpleNamespace(is_connected=True),check=lambda: None, service_feedback=lambda: None,
                            service_idle=lambda: None,
                            prepare_streaming=lambda epoch: state.stop_signal.clear(),
                            stop=lambda: order.append('stop'), close=lambda: order.append('close'),
                            send_action=lambda *a, **kw: order.append('send'))
    def compute(*args, **kwargs):
        clock[0] += 63_000_000
        return SimpleNamespace(rejection_reason=None, arm_qpos=np.zeros(7),
                               hand_qpos=np.zeros(12), eef_pose=None)
    def commit(*args, **kwargs):
        order.append('commit')
        keyboard.handle_key('q', True)
    controller = SimpleNamespace(compute_target=compute, commit=commit,
                                 reset=lambda row: None, clear=lambda: order.append('clear'))
    loop = TeleopLoop(state, rollout.runtime, robot, task_label='test',
                       sensors=SimpleNamespace(channels=None), controller=controller,
                       recording_dir=None, max_duration_s=60, runtime_config_json=None,
                       keyboard=keyboard, audio=SimpleNamespace(play=lambda sound: None))
    keyboard.allow_start()
    keyboard.handle_key('b', True)
    loop.handle_commands(keyboard.poll(state, timeout=0)[0])
    assert order == ['stop', 'send', 'commit', 'stop', 'clear']  # 设备关闭归会话入口。
    assert state.quit_requested and loop.phase is TeleopPhase.IDLE



def test_teleop_final_mapping_failure_preserves_valid_prefix(monkeypatch, tmp_path):
    from dexmani_real.data.recorder import EpisodeRecorder
    from dexmani_real.teleop.loop import TeleopLoop
    rollout, _, _ = make_rollout(monkeypatch)
    row, _ = rollout.read_observation()
    loop = TeleopLoop.__new__(TeleopLoop)
    loop.runtime = rollout.runtime
    loop.cancel_epoch = 1
    loop.recorder = EpisodeRecorder(tmp_path, control_hz=16)
    loop.recorder.is_recording = True
    loop.recorder.append(row, RobotCommand(np.zeros(7), np.zeros(12)))
    loop.record_needs = ObservationNeeds()
    loop.sensors = SimpleNamespace(check=lambda **kw: True)
    ended = []
    loop.request_episode_end = lambda reason, **kwargs: ended.append(reason)
    loop.controller = SimpleNamespace(compute_target=lambda *a, **kw: SimpleNamespace(
        rejection_reason='no feasible target', arm_qpos=None))
    loop.execute_control_step(row)
    assert ended == ['target_unavailable: no feasible target', 'recording_no_command']
    assert loop.recorder.queue.qsize() == 1
    assert loop.recorder.recording_truncated_reason == 'recording_no_command'
    assert loop.recorder.queue.qsize() == 1 and loop.recorder.finish_request is None


def test_close_interrupt_latches_emergency_before_waiting_for_writer(monkeypatch):
    from dexmani_real.runtime.state import RunEndReason, SessionState
    rollout, _, _ = make_rollout(monkeypatch)
    rollout.runtime_state = SessionState(quit_requested=True)
    rollout.robot.stop = lambda: None
    def close():
        raise KeyboardInterrupt()
    rollout.robot.close = close
    rollout.prepare = lambda poll: rollout.stop('quit', run_end_reason=RunEndReason.QUIT)
    cleanup = []
    def finish(stage):
        assert rollout.runtime_state.estop_request
        cleanup.append(stage)
    rollout.finish_recording = lambda: finish('writer')
    rollout.model.discard_pending = lambda timeout: finish('model')
    rollout.run(poll_operator=lambda: None)
    assert close_rollout_session(rollout) == 1
    assert cleanup == ['writer', 'model']


@pytest.mark.parametrize('prediction_ns,mapping_ns,steps,progress', [
    (600_000_000, 0, 8, False),
    (100_000_000, 63_000_000, 8, True),
    (62_500_000, 63_000_000, 2, False),
])
def test_async_progress_requires_valid_window(monkeypatch, prediction_ns, mapping_ns, steps, progress):
    from concurrent.futures import Future
    rollout, clock, sends = make_rollout(monkeypatch, mode='async')
    rollout.policy_info.future_steps = steps
    rollout.execution = replace(rollout.execution, action_steps=min(4, steps))
    rollout.policy_info.control_action_dim = 19
    deadline = clock[0] + 2_000_000_000
    pending = [None, None]
    def predict(*args, **kwargs):
        pending[:] = [Future(), clock[0] + prediction_ns]
    def poll():
        if pending[0] is None or clock[0] < pending[1]:
            return None
        future = pending[0]
        pending[0] = None
        future.set_result(np.zeros((steps, 19)))
        return future
    def mapping(*args):
        clock[0] += mapping_ns
        return RobotCommand(np.zeros(7)), None
    rollout.model.predict, rollout.model.poll = predict, poll
    rollout.map_command = mapping
    if progress:
        for _ in range(3000):
            rollout.step()
            clock[0] += 1_000_000
        assert len(sends) > 20
        assert clock[0] > deadline
        assert not rollout.finished
    else:
        for _ in range(4000):
            rollout.step()
            clock[0] += 1_000_000
        assert not sends and rollout.finished


def test_owner_stop_prevents_late_prediction_submission(monkeypatch):
    from dexmani_real.robot.commands import OperationCancelled
    rollout, clock, sends = make_rollout(monkeypatch)
    rollout.sync_actions = np.zeros((4, 19))
    rollout.runtime_state.stop_signal.set()
    with pytest.raises(OperationCancelled):
        rollout.step()
    assert not sends


@pytest.mark.parametrize('control_camera', [False, True])
def test_capture_needs_remain_fixed(monkeypatch, control_camera):
    rollout, _, _ = make_rollout(monkeypatch)
    control = ObservationNeeds(camera_fields=('rgb',) if control_camera else ())
    rollout.model_needs = control
    rollout.capture_needs = control.union(ObservationNeeds(camera_fields=('rgb', 'depth'), required_tactile=('dense',)))
    rollout.recorder = SimpleNamespace(is_recording=True)
    rollout.sensors = SimpleNamespace(channels=None)
    row, _ = rollout.read_observation()
    needs = []
    def read(*args, **kwargs):
        needs.append(kwargs['needs'])
        return row
    monkeypatch.setattr('dexmani_real.deployment.rollout.read_observation', read)
    monkeypatch.setattr('dexmani_real.deployment.rollout.check_observation', lambda *a, **kw: [])
    Rollout.read_observation(rollout)
    rollout.recorder.is_recording = False
    Rollout.read_observation(rollout)
    assert needs == [rollout.capture_needs, rollout.capture_needs]


@pytest.mark.parametrize('model_fails', [False, True])
def test_duration_and_owner_lease_stop_together_preserve_faults(monkeypatch, model_fails):
    from concurrent.futures import Future

    from dexmani_real.runtime.state import SessionState

    rollout, clock, sends = make_rollout(monkeypatch, mode='async')
    rollout.runtime_state = state = SessionState()
    rollout.cancel_epoch = state.cancel_epoch
    state.stop_signal.set()  # owner 已观察到目标租期到期。
    rollout.robot.check = lambda: None
    rollout.robot.stop = lambda: None
    rollout.run_deadline_ns = clock[0]
    if model_fails:
        done = Future()
        done.set_exception(ValueError('model computation failed'))
        rollout.model.poll = lambda: done
        with pytest.raises(ValueError, match='model computation failed'):
            rollout.step()
    else:
        rollout.step()
        assert rollout.result.termination_reason == 'duration_complete'
    assert not sends


def test_operator_cancel_at_duration_still_uses_current_input(monkeypatch):
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.runtime.state import SessionState

    rollout, clock, _ = make_rollout(monkeypatch)
    rollout.runtime_state = state = SessionState()
    rollout.cancel_epoch = state.cancel_epoch
    rollout.robot.check = lambda: None
    rollout.run_deadline_ns = clock[0]
    state.request_stop()
    with pytest.raises(OperationCancelled):
        rollout.step()
    assert not rollout.finished  # run 的操作者轮询消费终止输入后确定当前段原因。


@pytest.mark.parametrize('mode', ['async', 'rtc'])
def test_bootstrap_missing_snapshot_has_no_hidden_wait_budget(monkeypatch, mode):
    rollout, clock, sends = make_rollout(monkeypatch, mode=mode)
    rollout.model_needs = ObservationNeeds(camera_fields=('rgb',))
    deadline = clock[0] + 2_000_000_000
    rollout.step()
    assert not rollout.finished and not sends
    clock[0] = deadline + 1
    rollout.step()
    assert not rollout.finished and not sends



@pytest.mark.parametrize('mapping_ns,stale', [(62_499_999, False), (62_500_000, False), (63_000_000, False), (140_000_000, True)])
def test_async_full_tail_window_and_source_boundary(monkeypatch, mapping_ns, stale):
    from dexmani_real.deployment.schedule import ActionChunk
    rollout, clock, sends = make_rollout(monkeypatch, mode='async')
    rollout.robot.stop = lambda: None
    rollout.plan = ActionChunk(np.zeros((8, 19)), 0)
    rollout.last_slot = 6
    rollout.next_step_ns = clock[0] = rollout.started_ns + 7 * rollout.dt_ns
    def mapping(*args):
        clock[0] += mapping_ns
        return RobotCommand(np.zeros(7)), None
    rollout.map_command = mapping
    if stale:
        with pytest.raises(RuntimeError, match='device feedback'):
            rollout.step()
        assert not sends
        return
    rollout.step()
    assert bool(sends) == (mapping_ns < rollout.dt_ns)
    if not sends:
        assert rollout.finished


def test_sync_history_survives_prediction_wait(monkeypatch):
    from concurrent.futures import Future
    rollout, clock, sends = make_rollout(monkeypatch, history=2)
    rollout.execution = replace(rollout.execution, action_steps=1)
    rollout.policy_info.future_steps = 1
    rollout.policy_info.control_action_dim = 19
    submissions = []
    due = [0]
    def predict(rows, **kwargs):
        submissions.append([row.control_timestamp_ns for row in rows])
        rollout.model.future = Future()
        due[0] = clock[0] + 100_000_000
    def poll():
        future = rollout.model.future
        if future is None or clock[0] < due[0]:
            return None
        rollout.model.future = None
        future.set_result(np.zeros((1, 19)))
        return future
    rollout.model.predict, rollout.model.poll = predict, poll
    rollout.map_command = lambda *args: (RobotCommand(np.zeros(7)), None)
    origin = clock[0]
    for delta in (0, 62_500_000, 125_000_000, 162_500_000, 225_000_000):
        clock[0] = origin + delta
        rollout.step()
    assert submissions == [[origin, origin],
                           [origin + 125_000_000, origin + 225_000_000]]
    assert len(sends) == 1


def test_teleop_esc_arriving_in_poll_records_estop():
    import threading
    from dexmani_real.runtime.keyboard import Keyboard
    from dexmani_real.runtime.state import SessionState
    from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase
    state = SessionState(stop_signal=threading.Event(), emergency_signal=threading.Event())
    keyboard = Keyboard(estop_callback=state.request_emergency_stop)
    keyboard.active = True
    keyboard.listener = SimpleNamespace(is_alive=lambda: True)
    poll = keyboard.poll
    def interrupted_poll(**kwargs):
        keyboard.handle_key('esc', True)
        assert state.stop_signal.is_set()
        return poll(state, timeout=0)
    keyboard.poll = interrupted_poll
    loop = TeleopLoop.__new__(TeleopLoop)
    loop.runtime_state, loop.keyboard = state, keyboard
    loop.robot = SimpleNamespace(arm=SimpleNamespace(is_connected=True),service_feedback=lambda: None, service_idle=lambda: None)
    loop.phase, loop.next_tick_ns = TeleopPhase.ACTIVE, None
    loop.recorder = loop.pending_termination_reason = None
    loop.audio = SimpleNamespace(play=lambda sound: None)
    loop.termination_details = []
    assert not loop.service_episode(.005)
    assert loop.pending_termination_reason == 'estop'


@pytest.mark.parametrize('mode', ['async', 'rtc'])
def test_prefetch_uses_retained_history_with_real_model_future(monkeypatch, mode):
    from dexmani_real.deployment.inference import PolicyWorker

    rollout, clock, sends = make_rollout(monkeypatch, mode=mode, history=3)
    rollout.execution = replace(rollout.execution, action_steps=2, rtc_guidance_cap=0.0)
    rollout.policy_info.future_steps = 8
    rollout.policy_info.control_action_dim = 19
    rollout.policy_info.action_mode = 'joint'
    rollout.action_mapper = SimpleNamespace(normalize_policy_actions=lambda actions, mode: actions)
    rollout.map_command = lambda *args: (RobotCommand(np.zeros(7)), None)
    calls = []
    actions = np.arange(8 * 19).reshape(8, 19).astype(float)
    worker = PolicyWorker(lambda: None, build_observation=lambda rows: tuple(
        row.control_timestamp_ns for row in rows))
    def predict(observation, **kwargs):
        calls.append((observation, kwargs))
        return actions
    worker.model = SimpleNamespace(reset_episode=lambda: None, predict=predict, close=lambda: None)
    rollout.model = worker
    origin = clock[0]
    try:
        rollout.step()
        worker.future.result(timeout=1)
        clock[0] = origin + 62_500_000
        rollout.step()
        clock[0] = rollout.next_step_ns
        rollout.step()
        worker.future.result(timeout=1)
        assert len(sends) == len(calls) == 2
        assert calls[0][0] == (origin, origin, origin)
        assert calls[1][0] == (origin, origin + 62_500_000, origin + 125_000_000)
        if mode == 'rtc':
            assert calls[1][1]['delay_steps'] == 1
            np.testing.assert_array_equal(calls[1][1]['rtc_prefix'], actions[1:])
        else:
            assert calls[1][1] == {}
        assert not rollout.finished
    finally:
        worker.close().result(timeout=1)


def test_slow_first_prediction_has_only_episode_duration(monkeypatch):
    rollout, clock, sends = make_rollout(monkeypatch)
    rollout.run_deadline_ns = clock[0] + 60_000_000_000
    rollout.model.future = object()
    clock[0] += 3_000_000_000
    rollout.step()
    assert not rollout.finished and not sends
    clock[0] = rollout.run_deadline_ns
    rollout.step()
    assert rollout.result.termination_reason == 'duration_complete'


def test_recording_worker_exit_is_a_session_fault(monkeypatch, tmp_path):
    from dexmani_real.data.recorder import EpisodeRecorder
    rollout, _, _ = make_rollout(monkeypatch)
    rollout.recorder = EpisodeRecorder(tmp_path, control_hz=16)
    rollout.recorder.is_recording = True
    rollout.sensors = SimpleNamespace(check=lambda **kwargs: False)
    with pytest.raises(RuntimeError, match='camera worker exited'):
        rollout.step()
    assert rollout.runtime_state.stop_signal.is_set()
    assert rollout.recorder.queue.empty()


def test_policy_raw_is_explicit_opt_in():
    import inspect
    from dexmani_real.deployment.run import run_policy
    from scripts.run_policy import build_parser
    assert inspect.signature(run_policy).parameters['record_raw'].default is False
    parser = build_parser()
    assert not parser.parse_args([]).record_raw
    assert parser.parse_args(['--record-raw']).record_raw
    with pytest.raises(SystemExit):
        parser.parse_args(['--no-record'])


def test_mapping_crossing_episode_deadline_does_not_submit(monkeypatch):
    rollout, clock, sends = make_rollout(monkeypatch)
    rollout.run_deadline_ns = clock[0] + 10_000_000
    rollout.sync_actions = np.zeros((4, 19))
    def mapping(*args):
        clock[0] += 20_000_000
        return RobotCommand(np.zeros(7)), None
    rollout.map_command = mapping
    rollout.step()
    assert not sends
    assert rollout.result.termination_reason == 'duration_complete'


def test_policy_initial_observation_need_not_be_home(monkeypatch):
    rollout, _, _ = make_rollout(monkeypatch)
    rollout.robot.service_feedback = lambda: None
    row = rollout.start_observation()
    assert row is not None
    assert not np.allclose(row.arm['qpos'][0], rollout.runtime.arm.home_qpos)


@pytest.mark.parametrize('cancelled', [False, True])
def test_policy_prepare_only_accepts_current_operator_cancellation(monkeypatch, cancelled):
    from dexmani_real.robot.commands import OperationCancelled
    rollout, _, _ = make_rollout(monkeypatch)
    rollout.started_ns = 0
    def prepare(poll):
        if cancelled:
            rollout.runtime_state.request_stop()
        raise OperationCancelled('prepare interrupted')
    rollout.prepare = prepare
    if cancelled:
        rollout.run(poll_operator=lambda: None)
        assert not rollout.runtime_state.quit_requested
        assert rollout.result.termination_reason == 'start_cancelled'
    else:
        with pytest.raises(OperationCancelled, match='prepare interrupted'):
            rollout.run(poll_operator=lambda: None)
        assert not rollout.runtime_state.quit_requested
        assert rollout.result.termination_reason == 'preparation_failed'


@pytest.mark.parametrize('cancelled', [False, True])
def test_policy_home_only_accepts_current_operator_cancellation(monkeypatch, cancelled):
    from dexmani_real.deployment.session import PolicySession
    from dexmani_real.robot.commands import OperationCancelled
    session = PolicySession.__new__(PolicySession)
    session.runtime_state = state = SessionState()
    state.request_stop()  # 之前正常结束过一段，不能据此吞掉这次意外中断。
    epoch = state.cancel_epoch
    session.episode = None
    session.robot = SimpleNamespace(arm=SimpleNamespace(is_connected=True), stop=lambda: None, check=lambda: None)
    session.home_abort_requested = lambda: False
    def home(*args, **kwargs):
        if cancelled:
            state.request_stop()
        raise OperationCancelled('home interrupted')
    monkeypatch.setattr('dexmani_real.deployment.session.home_arm', home)
    monkeypatch.setattr('dexmani_real.deployment.session.home_hand', lambda *a, **kw: None)
    if cancelled:
        assert session.run_home(epoch)
        assert not state.quit_requested
    else:
        with pytest.raises(OperationCancelled, match='home interrupted'):
            session.run_home(epoch)
        assert not state.quit_requested


def test_rollout_releases_completed_recorder_once(monkeypatch):
    rollout, _, _ = make_rollout(monkeypatch)
    writes = []
    rollout.result.termination_reason = 'operator'
    rollout.recorder = SimpleNamespace(
        finish=lambda **kwargs: writes.append(kwargs) or '/published',
        result=lambda: {'recording_truncated_reason': None, 'path': '/published'},
    )
    rollout.finish_recording()
    rollout.finish_recording()
    assert len(writes) == 1
    assert rollout.recorder is None and rollout.result.raw['path'] == '/published'


@pytest.mark.parametrize('boundary', ['lower', 'upper'])
@pytest.mark.parametrize('offset', [0, .5e-6, 2e-6, .1, np.nan, np.inf])
def test_policy_and_trajectory_agree_on_feedback_admission(monkeypatch, boundary, offset):
    from dexmani_real.robot.xarm7.trajectory import ArmTrajectory
    rollout, _, _ = make_rollout(monkeypatch)
    row, _ = rollout.read_observation()
    cfg = rollout.runtime.arm
    bound = np.asarray(cfg.joint_limit_lower if boundary == 'lower' else cfg.joint_limit_upper)
    # offset=0 模拟 SDK float32 传输后转回 float64 的反馈。
    row.arm['qpos'][0] = (bound.astype(np.float32).astype(np.float64) if offset == 0
                          else bound + (-offset if boundary == 'lower' else offset))
    original = row.arm.copy()
    rollout.read_observation = lambda: (row, None)
    rollout.robot.service_feedback = lambda: None
    if offset in (0, .5e-6):
        assert rollout.start_observation() is row
        trajectory = ArmTrajectory(cfg, row.arm['qpos'][0], np.zeros(7))
        np.testing.assert_array_equal(
            trajectory.input.current_position,
            np.clip(row.arm['qpos'][0], cfg.joint_limit_lower, cfg.joint_limit_upper),
        )
    else:
        with pytest.raises(ValueError, match='finite|joint limits'):
            rollout.start_observation()
        with pytest.raises(ValueError, match='finite|joint limits'):
            ArmTrajectory(cfg, row.arm['qpos'][0], np.zeros(7))
    assert row.arm.tobytes() == original.tobytes()  # 原反馈和来源时间戳都保留。


@pytest.mark.parametrize('cancelled', [False, True])
def test_teleop_prepare_only_accepts_current_operator_cancellation(monkeypatch, cancelled):
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.teleop.loop import TeleopLoop
    rollout, _, _ = make_rollout(monkeypatch)
    state = rollout.runtime_state
    state.request_stop()
    epoch = state.cancel_epoch
    keyboard = Keyboard()
    loop = TeleopLoop(
        state, rollout.runtime, rollout.robot, task_label='test',
        sensors=None, controller=None, recording_dir=None, max_duration_s=60,
        runtime_config_json=None, keyboard=keyboard, audio=SimpleNamespace(play=lambda sound: None),
    )
    loop.poll_blocking_commands = lambda: False
    loop.start_observation = lambda: object()
    def prepare(epoch):
        if cancelled:
            state.request_stop()
        raise OperationCancelled('prepare interrupted')
    rollout.robot.prepare_streaming = prepare
    if cancelled:
        loop.start_episode(epoch)
        assert not state.quit_requested
    else:
        with pytest.raises(OperationCancelled, match='prepare interrupted'):
            loop.start_episode(epoch)
        assert not state.quit_requested


@pytest.mark.parametrize('scale,accepted', [(.001, True), (float(np.float32(.001)), True), (.01, False)])
def test_observation_builder_depth_scale(scale, accepted):
    from dexmani_real.deployment.observation import ObservationBuilder
    from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
    builder = ObservationBuilder(SimpleNamespace(observation_fields=('depth',),
                                 depth_config={'depth_scale_m': .001}), RealConfig())
    camera = CameraSnapshot('test', CameraIntrinsics(4, 4, 4., 4., 2., 2., 'none', (0.,) * 5),
                            scale, np.eye(4))
    if accepted:
        builder.configure_camera(camera)
    else:
        with pytest.raises(ValueError, match='Live depth scale'):
            builder.configure_camera(camera)


@pytest.mark.parametrize('before,after,accepted,rejected', [
    ({'w'}, {'w', 'x'}, True, False), ({'w', 'x'}, {'w'}, True, False),
    ({'w'}, set(), False, False), ({'w'}, {'s'}, False, False),
    ({'w'}, {'w', 'd'}, False, False), ({'w'}, {'w', 'r'}, False, False),
    ({'w'}, {'w', 'x'}, False, True),
])
def test_jog_compares_only_control_keys(monkeypatch, before, after, accepted, rejected):
    from dexmani_real.runtime.jog import CartesianJog
    rollout, _, _ = make_rollout(monkeypatch)
    row, _ = rollout.read_observation()
    monkeypatch.setattr('dexmani_real.runtime.jog.read_observation', lambda *a: row)
    keys = Keyboard()
    keys.held_keys = before
    jog = CartesianJog(None, SimpleNamespace(target_max_age_s=.5), (None, None))
    target = np.ones(7)
    jog.command_pose = jog.command_qpos = np.zeros(7)
    def propose(*args, **kwargs):
        keys.held_keys = after
        return target, SimpleNamespace(qpos=None if rejected else target, reason='IK rejected')
    jog.propose = propose
    sent = []
    rollout.robot.send_action = lambda command, **kw: sent.append(command)
    jog.step(rollout.robot, keys, SessionState(), rollout.runtime)
    assert sent[0].arm_hold == (not accepted)
    if accepted:
        np.testing.assert_array_equal(sent[0].arm_qpos, target)
        np.testing.assert_array_equal(jog.command_pose, target)
        np.testing.assert_array_equal(jog.command_qpos, target)
    else:
        assert jog.command_pose is jog.command_qpos is None


@pytest.mark.parametrize('key', ['q', 'esc'])
def test_jog_cancel_during_ik_reaches_publication_guard(monkeypatch, key):
    from dexmani_real.robot.commands import OperationCancelled
    from dexmani_real.robot.system import Robot
    from dexmani_real.robot.xarm7.process import ArmProcess
    from dexmani_real.runtime.jog import CartesianJog
    import threading
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    monkeypatch.setattr(ctx, 'Event', threading.Event)
    monkeypatch.setattr(ctx, 'Lock', threading.Lock)
    rollout, _, _ = make_rollout(monkeypatch)
    row, _ = rollout.read_observation()
    monkeypatch.setattr('dexmani_real.runtime.jog.read_observation', lambda *a: row)
    state = SessionState()
    robot = Robot(rollout.runtime.arm, rollout.runtime.hand, runtime_state=state)
    robot.connected = True
    arm = robot.arm = ArmProcess(rollout.runtime.arm, runtime_state=state)
    arm.process = SimpleNamespace(pid=1, exitcode=None, is_alive=lambda: True)
    arm.io.ready.set()
    keys = Keyboard(bindings={'q': state.request_quit}, estop_callback=state.request_emergency_stop)
    keys.active = True
    keys.handle_key('w', True)
    jog = CartesianJog(None, SimpleNamespace(target_max_age_s=.5), (None, None))
    def propose(*args, **kwargs):
        keys.handle_key(key, True)
        return np.zeros(7), SimpleNamespace(qpos=np.zeros(7))
    jog.propose = propose
    try:
        with pytest.raises(OperationCancelled):
            jog.step(robot, keys, state, rollout.runtime)
        assert arm.io.take_goal() is None
        assert jog.command_qpos is None
    finally:
        arm.pipe.close()
        arm.child_pipe.close()


def make_coordinated_robot(monkeypatch):
    from dexmani_real.robot.system import Robot
    from dexmani_real.robot.xhand.feedback import XHandState
    now = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.robot.system.time.monotonic_ns', lambda: now[0])
    cfg, state = RealConfig(), SessionState()
    robot = Robot(cfg.arm, cfg.hand, runtime_state=state)
    robot.connected = True
    submitted = []
    robot.arm = SimpleNamespace(is_connected=True, check_error=lambda: None,
                                submit=lambda command, **kw: submitted.append(command))
    robot.hand = SimpleNamespace(is_connected=True, get_state=lambda: None)
    robot.hand_feedback.consume(XHandState(np.zeros(12), np.ones((5, 3)),
                                          np.ones((5, 120, 3)), now[0]))
    return robot, now, submitted


def test_unconfirmed_hand_submission_advances_to_latest_target(monkeypatch):
    robot, now, submitted = make_coordinated_robot(monkeypatch)
    hand_targets = []
    def send(target):
        hand_targets.append(target.copy())
        return len(hand_targets) > 1
    robot.hand._send_action = send
    for value in (0., .1):
        robot.send_action(RobotCommand(np.zeros(7), np.full(12, value)), expires_ns=now[0] + 1)
        assert not robot.runtime_state.stop_signal.is_set()
    assert len(submitted) == 2
    np.testing.assert_array_equal(hand_targets[1], np.full(12, .1))


def test_hand_sync_send_has_no_arm_target_lease(monkeypatch):
    robot, now, _ = make_coordinated_robot(monkeypatch)
    sent = []
    robot.hand._send_action = lambda target: sent.append(target)
    robot.send_action(RobotCommand(hand_qpos=np.zeros(12)), expires_ns=now[0] - 1)
    assert len(sent) == 1


def test_active_feedback_service_detects_loss_without_another_send(monkeypatch):
    from dexmani_real.robot.commands import RobotError
    robot, now, submitted = make_coordinated_robot(monkeypatch)
    robot.hand_stop_required = True
    now[0] += 490_000_000
    robot.service_feedback()
    assert int(robot.hand_feedback.snapshot()['timestamp_ns'][0]) == 1_000_000_000
    now[0] += 20_000_000
    with pytest.raises(RobotError, match='hand feedback unavailable/stale'):
        robot.service_feedback()
    assert robot.runtime_state.stop_signal.is_set()
    assert not submitted


def test_expired_hand_feedback_rejects_before_arm_submission(monkeypatch):
    from dexmani_real.robot.commands import RobotError
    robot, now, submitted = make_coordinated_robot(monkeypatch)
    now[0] += 510_000_000
    robot.hand._send_action = lambda target: pytest.fail('stale hand sent')
    with pytest.raises(RobotError, match='hand feedback unavailable/stale'):
        robot.send_action(RobotCommand(np.zeros(7), np.zeros(12)), expires_ns=now[0] + 1_000_000_000)
    assert not submitted and robot.runtime_state.stop_signal.is_set()


def test_hand_feedback_expiring_during_arm_publication_rejects_hand_send(monkeypatch):
    from dexmani_real.robot.commands import RobotError
    robot, now, submitted = make_coordinated_robot(monkeypatch)
    def submit(command, **kwargs):
        submitted.append(command)
        now[0] += 510_000_000
    robot.arm.submit = submit
    robot.hand._send_action = lambda target: pytest.fail('stale hand sent')
    with pytest.raises(RobotError, match='hand feedback unavailable/stale'):
        robot.send_action(RobotCommand(np.zeros(7), np.zeros(12)), expires_ns=now[0] + 1_000_000_000)
    assert len(submitted) == 1 and robot.runtime_state.stop_signal.is_set()


def test_hand_home_callback_cannot_extend_operation_deadline(monkeypatch):
    robot, now, _ = make_coordinated_robot(monkeypatch)
    robot.hand._send_position = lambda target: pytest.fail('expired HOME submitted')
    def cancelled():
        now[0] += 110_000_000
        return False
    with pytest.raises(TimeoutError, match='HOME budget exhausted'):
        robot.send_hand_home(RobotCommand(hand_qpos=np.zeros(12)), deadline_ns=now[0] + 100_000_000,
                             cancel_requested=cancelled)
    assert not robot.hand_stop_required


def test_tactile_reset_clears_only_tactile_before_driver_requests(monkeypatch):
    robot, now, _ = make_coordinated_robot(monkeypatch)
    stopped = []
    robot.stop = lambda: stopped.append(True)
    def reset(*, cancel_requested):
        assert not cancel_requested()
        assert int(robot.hand_feedback.snapshot()['tactile_dense_timestamp_ns'][0]) == 0
        return {0x11: 1501070}
    robot.hand.reset_tactile_sensors = reset
    result = robot.reset_tactile_sensors(cancel_epoch=robot.runtime_state.cancel_epoch,
                                         cancel_requested=lambda: False)
    assert stopped and result == {0x11: 1501070}
    frame = robot.hand_feedback.snapshot()
    assert int(frame['timestamp_ns'][0]) == now[0]
    np.testing.assert_array_equal(frame['qpos'], np.zeros((1, 12)))
    for channel in ('aggregate', 'dense'):
        assert int(frame[f'tactile_{channel}_timestamp_ns'][0]) == 0
        assert np.isnan(frame[f'tactile_{channel}']).all()


@pytest.mark.parametrize('hand_limits', [
    {'qpos_min_rad': (float('nan'),) * 12},
    {'qpos_min_rad': (0.0,), 'qpos_max_rad': (1.0,)},
    {'qpos_min_rad': (10.0,) * 12},
    {'qpos_min_rad': (-10.0,) * 12},
    {'qpos_min_rad': ('unused',) * 12},
])
def test_joint_mapper_needs_no_geometry_or_disabled_hand_limits(monkeypatch, hand_limits):
    import builtins
    from dexmani_real.control.actions import ActionMapper
    from dexmani_real.runtime.preflight import validate_control_rate, validate_robot_config
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'pinocchio', 'hppfcl', 'mplib', 'trimesh'}:
            raise ModuleNotFoundError(name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    cfg = RealConfig()
    hand = replace(cfg.hand, enabled=False, home_qpos_deg=(float('nan'),)*12, **hand_limits)
    environment = replace(cfg.environment, workspace=replace(cfg.environment.workspace, x_min=10))
    runtime = replace(cfg, hand=hand, environment=environment)
    validate_robot_config(runtime)
    validate_control_rate(runtime, cfg.arm.servo_hz * 2)
    mapper = ActionMapper.for_mode('joint', arm=cfg.arm, hand=hand, motion=cfg.motion, environment=environment)
    result = mapper.map_joint(np.asarray(cfg.arm.home_qpos), None, cfg.arm.home_qpos)
    assert result.rejection_reason is None
    np.testing.assert_allclose(result.arm_qpos, cfg.arm.home_qpos)
    from dexmani_real.robot.system import Robot
    from dexmani_real.robot.commands import RobotCommand
    from dexmani_real.runtime.home import home_arm
    submitted = []
    homed = []
    def arm_home(*args, service, **kwargs):
        service()
        homed.append((args, kwargs))
    robot = Robot(cfg.arm, hand, runtime_state=SessionState())
    robot.connected = True
    robot.arm = SimpleNamespace(is_connected=True, check_error=lambda: None,
                                submit=lambda cmd, **kw: submitted.append(cmd),
                                home=arm_home)
    robot.send_action(RobotCommand(arm_qpos=result.arm_qpos), expires_ns=2_000_000_000)
    assert len(submitted) == 1
    home_arm(robot.runtime_state, robot, cancel_epoch=robot.runtime_state.cancel_epoch,
             cancel_requested=lambda: False)
    assert len(homed) == 1
    with pytest.raises(ModuleNotFoundError):
        ActionMapper.for_mode('eef', arm=cfg.arm, hand=cfg.hand, motion=cfg.motion, environment=cfg.environment)


@pytest.mark.parametrize('hand_changes, error', [
    ({'qpos_min_rad': (float('nan'),) * 12}, 'hand command limits'),
    ({'qpos_min_rad': (0.0,), 'qpos_max_rad': (1.0,)}, 'hand command limits'),
    ({'qpos_min_rad': (10.0,) * 12}, 'hand command limits'),
    ({'qpos_min_rad': (-10.0,) * 12}, 'hand command limits'),
    ({'home_qpos_deg': (float('nan'),) * 12}, 'hand target must be finite'),
    ({'home_qpos_deg': (0.0,)}, 'hand target must have shape'),
    ({'home_qpos_deg': (180.0,) * 12}, 'hand target violates joint limits'),
])
def test_disabled_hand_geometry_requires_valid_limits_and_fixed_pose(monkeypatch, hand_changes, error):
    from dexmani_real.control.actions import ActionMapper
    from dexmani_real.runtime.preflight import validate_robot_config
    from dexmani_real.teleop.keyboard import KeyboardTeleop

    cfg = RealConfig()
    runtime = replace(cfg, hand=replace(cfg.hand, enabled=False, **hand_changes))
    validate_robot_config(runtime)
    with pytest.raises(ValueError, match=error):
        ActionMapper.for_mode('eef', arm=runtime.arm, hand=runtime.hand,
                              motion=runtime.motion, environment=runtime.environment)
    monkeypatch.setattr('dexmani_real.teleop.keyboard.configure_logging', lambda: None)
    with pytest.raises(ValueError, match=error):
        KeyboardTeleop(runtime).run()


def test_enabled_hand_still_validates_device_limits():
    from dexmani_real.runtime.preflight import validate_robot_config

    cfg = RealConfig()
    runtime = replace(cfg, hand=replace(cfg.hand, qpos_min_rad=(float('nan'),) * 12))
    with pytest.raises(ValueError, match='hand command limits'):
        validate_robot_config(runtime)


def rtc_rollout(monkeypatch, *, length=16, action_steps=8, prefetch_steps=2):
    from concurrent.futures import Future
    from dexmani_real.deployment.schedule import ActionChunk
    rollout, clock, sends = make_rollout(monkeypatch, mode='rtc', history=3)
    rollout.policy_info.future_steps = 16
    rollout.policy_info.control_action_dim = 19
    rollout.policy_info.action_mode = 'joint'
    rollout.execution = replace(rollout.execution, action_steps=action_steps,
                                prefetch_steps=prefetch_steps, rtc_guidance_cap=0.)
    rollout.plan = ActionChunk(np.repeat(np.arange(length)[:, None], 19, axis=1).astype(float), 0)
    rollout.first_prediction = False
    rollout.action_mapper = SimpleNamespace(normalize_policy_actions=lambda actions, mode: actions)
    commands, requests = [], []
    def predict(rows, **kwargs):
        assert rollout.model.future is None
        requests.append((rows, kwargs))
        rollout.model.future = Future()
    def poll():
        future = rollout.model.future
        if future is not None and future.done():
            rollout.model.future = None
            return future
    rollout.model.predict, rollout.model.poll = predict, poll
    rollout.map_command = lambda action, row: (RobotCommand(np.full(7, action[0])), None)
    rollout.robot.send_action = lambda command, **kwargs: commands.append(command.arm_qpos[0])
    return rollout, clock, commands, requests


@pytest.mark.parametrize('ready_cursor,expected', [(7, 102), (11, 105)])
def test_rtc_handoff_uses_submissions_and_nominal_window(monkeypatch, ready_cursor, expected):
    rollout, clock, commands, requests = rtc_rollout(monkeypatch)
    rollout.cursor = 6
    rollout.step()
    frozen = requests[0][0]
    np.testing.assert_array_equal(requests[0][1]['rtc_prefix'][:, 0], np.arange(6, 16))
    while rollout.cursor < ready_cursor:
        clock[0] = rollout.next_step_ns
        rollout.step()
    rollout.model.future.set_result(np.repeat((100 + np.arange(16))[:, None], 19, axis=1))
    clock[0] = rollout.next_step_ns + 5 * rollout.dt_ns
    rollout.step()
    if ready_cursor == 7:
        assert commands[-1] == 7
        clock[0] = rollout.next_step_ns
        rollout.step()
    assert commands[-1] == expected
    assert rollout.cursor == 1
    assert len(rollout.plan.actions) == 16 - (expected - 100)
    assert tuple(requests[0][0]) == frozen
    assert not rollout.finished


@pytest.mark.parametrize('ready', [True, False])
def test_rtc_exhaustion_receives_ready_result_first(monkeypatch, ready):
    rollout, clock, commands, _ = rtc_rollout(monkeypatch, length=2)
    rollout.step()
    clock[0] = rollout.next_step_ns
    rollout.step()
    if ready:
        rollout.model.future.set_result(np.zeros((16, 19)))
    clock[0] = rollout.next_step_ns
    rollout.step()
    assert rollout.finished is not ready
    assert len(commands) == (3 if ready else 2)
    if ready:
        assert len(rollout.plan.actions) == 14


def test_rtc_short_window_prefetch_is_bounded_and_not_received_twice(monkeypatch):
    rollout, clock, commands, requests = rtc_rollout(monkeypatch, length=1)
    predict = rollout.model.predict
    def immediate(*args, **kwargs):
        predict(*args, **kwargs)
        rollout.model.future.set_result(np.zeros((16, 19)))
    rollout.model.predict = immediate
    rollout.step()
    assert len(requests) == len(commands) == 1
    assert requests[0][1]['delay_steps'] == 1
    assert rollout.cursor == 1 and len(rollout.plan.actions) == 1
    clock[0] = rollout.next_step_ns
    rollout.step()
    assert len(commands) == 2 and len(rollout.plan.actions) == 15


def test_rtc_early_model_error_is_not_delayed_until_window(monkeypatch):
    rollout, clock, _, _ = rtc_rollout(monkeypatch)
    rollout.cursor = 6
    rollout.step()
    rollout.model.future.set_exception(ValueError('model failed early'))
    clock[0] += 1
    with pytest.raises(ValueError, match='model failed early'):
        rollout.step()


def test_rtc_failed_submission_does_not_advance_cursor(monkeypatch):
    rollout, _, _, _ = rtc_rollout(monkeypatch)
    def fail(*args, **kwargs):
        raise RuntimeError('send failed')
    rollout.robot.send_action = fail
    with pytest.raises(RuntimeError, match='send failed'):
        rollout.step()
    assert rollout.cursor == 0


@pytest.mark.parametrize('mode', ['sync', 'async', 'rtc'])
def test_missing_model_input_preserves_old_actions_without_history(monkeypatch, mode):
    rollout, clock, commands, requests = rtc_rollout(monkeypatch)
    rollout.execution = replace(rollout.execution, mode=mode)
    rollout.sync_actions = np.zeros((4, 19))
    observe = rollout.read_observation
    def missing():
        row, _ = observe()
        return row, 'camera_source_stale'
    rollout.read_observation = missing
    rollout.step()
    assert len(commands) == 1 and not requests and not rollout.history
    assert not rollout.finished


def test_episode_failure_and_stop_failure_both_remain_visible(monkeypatch):
    rollout, _, _ = make_rollout(monkeypatch)
    def fail_prepare(poll):
        raise ValueError('primary')
    def fail_stop():
        raise RuntimeError('stop')
    rollout.prepare, rollout.robot.stop = fail_prepare, fail_stop
    with pytest.raises(RuntimeError, match='stop') as failure:
        rollout.run(poll_operator=lambda: None)
    assert isinstance(failure.value.__context__, ValueError)
    assert [error['message'] for error in rollout.result.errors] == ['primary', 'stop']


@pytest.mark.parametrize('mode', ['async', 'rtc'])
def test_execution_allows_full_nominal_window(mode):
    from dexmani_real.config.deployment import ExecutionConfig
    from dexmani_real.deployment.preflight import validate_execution_config
    info = SimpleNamespace(supported_execution_modes=('sync', 'async', 'rtc'),
                           control_dt_s=.0625, future_steps=8)
    validate_execution_config(info, ExecutionConfig(mode=mode, action_steps=8,
                              prefetch_steps=2, rtc_guidance_cap=0.))
    with pytest.raises(ValueError, match='d < P'):
        validate_execution_config(info, ExecutionConfig(mode=mode, action_steps=8,
                                  prefetch_steps=8, rtc_guidance_cap=0.))


def test_rtc_bootstrap_waits_and_starts_at_first_action(monkeypatch):
    rollout, clock, commands, requests = rtc_rollout(monkeypatch)
    rollout.plan = None
    rollout.first_prediction = True
    rollout.step()
    assert len(requests) == 1 and not commands
    frozen = requests[0][0]
    clock[0] += 10 * rollout.dt_ns
    rollout.step()
    assert not rollout.finished and not commands
    assert all(row.control_timestamp_ns == rollout.started_ns for row in frozen)
    assert rollout.history[-1].control_timestamp_ns == clock[0]
    rollout.model.future.set_result(np.repeat(np.arange(16)[:, None], 19, axis=1))
    clock[0] = rollout.next_step_ns
    rollout.step()
    assert commands == [0] and rollout.cursor == 1
    assert requests[0][1]['reset_episode']


def test_rtc_trimmed_window_recomputes_prefetch(monkeypatch):
    rollout, clock, commands, requests = rtc_rollout(monkeypatch, length=8)
    rollout.policy_info.future_steps = 8
    rollout.cursor = 6
    rollout.step()
    rollout.model.future.set_result(np.zeros((8, 19)))
    for _ in range(2):
        clock[0] = rollout.next_step_ns
        rollout.step()
    assert len(rollout.plan.actions) == 6 and rollout.cursor == 1
    for _ in range(3):
        clock[0] = rollout.next_step_ns
        rollout.step()
    assert rollout.cursor == 4 and len(requests) == 1
    clock[0] = rollout.next_step_ns
    rollout.step()
    assert len(requests) == 2 and len(requests[-1][1]['rtc_prefix']) == 2


def test_rtc_completed_but_fully_consumed_result_ends(monkeypatch):
    rollout, clock, commands, requests = rtc_rollout(monkeypatch, action_steps=2, prefetch_steps=2)
    for _ in range(16):
        clock[0] = rollout.next_step_ns
        rollout.step()
    rollout.model.future.set_result(np.zeros((16, 19)))
    clock[0] = rollout.next_step_ns
    rollout.step()
    assert rollout.finished and len(commands) == 16 and len(requests) == 1


def test_sync_completion_does_not_add_sampling_opportunity(monkeypatch):
    from concurrent.futures import Future
    rollout, clock, sends = make_rollout(monkeypatch, history=2)
    rollout.policy_info.future_steps = 4
    rollout.policy_info.control_action_dim = 19
    future = Future()
    rollout.model.predict = lambda *args, **kwargs: setattr(rollout.model, 'future', future)
    def poll():
        if rollout.model.future is not None and future.done():
            rollout.model.future = None
            return future
    rollout.model.poll = poll
    rollout.map_command = lambda *args: (RobotCommand(np.zeros(7)), None)
    rollout.step()
    first = tuple(rollout.history)
    future.set_result(np.zeros((4, 19)))
    clock[0] += 1
    rollout.step()
    assert tuple(rollout.history) == first and not sends
    clock[0] = rollout.next_step_ns
    rollout.step()
    assert len(sends) == 1 and rollout.history[-1].control_timestamp_ns == clock[0]


def test_device_failure_is_not_downgraded_to_raw_quality_end(monkeypatch):
    rollout, clock, _ = make_rollout(monkeypatch)
    row, _ = rollout.read_observation()
    row.arm['timestamp_ns'] = 0
    rollout.read_observation = lambda: (row, 'arm_feedback_stale')
    rollout.recorder = SimpleNamespace(check_error=lambda: None)
    rollout.recording_issue = lambda row: 'camera_source_stale'
    with pytest.raises(RuntimeError, match='arm_feedback_stale'):
        rollout.step()
    assert rollout.runtime_state.stop_signal.is_set()
    assert not rollout.finished


def test_rtc_result_completing_during_sampling_is_received_before_exhaustion(monkeypatch):
    rollout, clock, commands, _ = rtc_rollout(monkeypatch, length=1)
    rollout.step()
    observe = rollout.read_observation
    def read():
        rollout.model.future.set_result(np.zeros((16, 19)))
        return observe()
    rollout.read_observation = read
    clock[0] = rollout.next_step_ns
    rollout.step()
    assert not rollout.finished and len(commands) == 2


@pytest.mark.parametrize('mode', ['sync', 'async', 'rtc'])
@pytest.mark.parametrize('stage', ['read', 'project'])
def test_sampling_cancellation_does_not_submit_prediction(monkeypatch, mode, stage):
    from dexmani_real.robot.commands import OperationCancelled
    rollout, _, _ = make_rollout(monkeypatch, mode=mode)
    requests = []
    rollout.model.predict = lambda *args, **kwargs: requests.append(args)
    if stage == 'read':
        observe = rollout.read_observation
        def cancel_read():
            row = observe()
            rollout.runtime_state.request_stop()
            return row
        rollout.read_observation = cancel_read
    else:
        def cancel_project(row):
            rollout.runtime_state.request_quit()
            return row
        rollout.observations.project = cancel_project
    with pytest.raises(OperationCancelled):
        rollout.step()
    assert not requests


@pytest.mark.parametrize('mode', ['sync', 'async', 'rtc'])
def test_sampling_cancel_leaves_completed_prediction_for_reclamation(monkeypatch, mode):
    from dexmani_real.deployment.inference import PolicyWorker
    from dexmani_real.deployment.schedule import Prediction
    from dexmani_real.robot.commands import OperationCancelled
    from concurrent.futures import Future
    rollout, _, sends = make_rollout(monkeypatch, mode=mode)
    rollout.policy_info.future_steps = 4
    rollout.policy_info.control_action_dim = 19
    completed = Future()
    completed.set_exception(ValueError('late model failure'))
    rollout.model.future = completed
    rollout.model.poll = lambda: PolicyWorker.poll(rollout.model)
    rollout.query = None if mode == 'sync' else Prediction(0, None)
    observe = rollout.read_observation
    def cancel_read():
        row = observe()
        rollout.runtime_state.request_emergency_stop()
        return row
    rollout.read_observation = cancel_read
    with pytest.raises(OperationCancelled):
        rollout.step()
    assert not sends and rollout.model.future is completed
    with pytest.raises(ValueError, match='late model failure'):
        PolicyWorker.discard_pending(rollout.model, 0)
    assert rollout.model.future is None


def test_mapping_cancel_does_not_receive_async_replacement(monkeypatch):
    from concurrent.futures import Future
    from dexmani_real.deployment.inference import PolicyWorker
    from dexmani_real.deployment.schedule import ActionChunk, Prediction
    from dexmani_real.robot.commands import OperationCancelled
    rollout, clock, sends = make_rollout(monkeypatch, mode='async')
    rollout.policy_info.future_steps = 4
    rollout.policy_info.control_action_dim = 19
    rollout.plan = ActionChunk(np.zeros((4, 19)), 0)
    rollout.query = Prediction(2, None)
    future = rollout.model.future = Future()
    rollout.model.poll = lambda: PolicyWorker.poll(rollout.model)
    def mapping(action, row):
        clock[0] += 4 * rollout.dt_ns
        future.set_result(np.zeros((4, 19)))
        rollout.runtime_state.request_stop()
        return RobotCommand(np.zeros(7)), None
    rollout.map_command = mapping
    with pytest.raises(OperationCancelled):
        rollout.step()
    assert not sends and rollout.model.future is future
    assert rollout.query.result is None

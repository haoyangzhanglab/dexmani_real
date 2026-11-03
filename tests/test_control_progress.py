from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from dexmani_real.config.real import RealConfig
from dexmani_real.deployment.rollout import Rollout
from dexmani_real.deployment.schedule import Prediction
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
                      robot=SimpleNamespace(stop=lambda: None, check=lambda: None, send_action=lambda *a, **kw: sends.append(clock[0])),
                      observations=SimpleNamespace(project=lambda row: row),
                      control_needs=ObservationNeeds(), record_needs=ObservationNeeds(),
                      sensors=None, action_mapper=None,
                      model=SimpleNamespace(future=None, poll=lambda: None, predict=lambda *a, **kw: None),
                      max_running_s=None, index=1, output_dir='/tmp', record_raw=False,
                      audio=SimpleNamespace(play=lambda sound: None))
    rollout.started_ns = rollout.next_step_ns = clock[0]

    def observe():
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm['timestamp_ns'] = hand['timestamp_ns'] = clock[0]
        return Observation(arm, hand, None, None, None, clock[0], control_timestamp_ns=clock[0]), None

    rollout.read_observation = observe
    return rollout, clock, sends


def close_rollout_session(rollout, *, failure=None):
    from dexmani_real.deployment.session import PolicySession
    session = PolicySession.__new__(PolicySession)
    session.runtime_state = rollout.runtime_state
    session.robot, session.episode, session.model = rollout.robot, rollout, rollout.model
    session.sensors = session.keyboard = session.audio = None
    session.errors = []
    session.summary = {'termination_reason': None}
    session.save_results = lambda: None
    session.model.close = lambda state: session.model.discard_pending(0)
    session.model.wait_closed = lambda timeout: True
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
    assert rollout.sync_cursor == 4
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


def test_rtc_missed_handoff_does_not_fall_back():
    from dexmani_real.deployment.schedule import handoff_plan
    query = Prediction(0, 2, np.zeros((4, 19)))
    plan, issue = handoff_plan(query, slot=3, mode='rtc')
    assert plan is None and issue == 'handoff_slot_missed'


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
    rollout.sensors = SimpleNamespace(check=lambda **kwargs: available[0] or loss != "worker_exit")
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
        'worker_exit': 'recording_camera_exited',
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
    robot = SimpleNamespace(check=lambda: None, service_feedback=lambda: None,
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
    assert order == ['send', 'commit', 'stop', 'clear']  # 设备关闭归会话入口。
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
        assert not sends and not rollout.finished


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
    rollout.control_needs = control
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


def test_rtc_missing_committed_snapshot_stops(monkeypatch):
    from dexmani_real.deployment.schedule import ActionChunk
    rollout, clock, sends = make_rollout(monkeypatch, mode='rtc')
    stopped = []
    rollout.robot.stop = lambda: stopped.append(True)
    rollout.control_needs = ObservationNeeds(camera_fields=('rgb',))
    rollout.plan = ActionChunk(np.zeros((8, 19)), 0)
    observe = rollout.read_observation
    def read():
        row, issue = observe()
        return replace(row, camera=None if rollout.last_slot == 1 else
                       dict(timestamp_ns=clock[0])), issue
    rollout.read_observation = read
    rollout.map_command = lambda *args: (RobotCommand(np.zeros(7)), None)
    for _ in range(3):
        clock[0] = rollout.next_step_ns
        rollout.step()
    assert len(sends) == 1 and stopped == [True]
    assert rollout.finished
    assert rollout.result.termination_details == 'committed_slot_observation_unavailable'


@pytest.mark.parametrize('mode', ['async', 'rtc'])
def test_bootstrap_missing_snapshot_has_no_hidden_wait_budget(monkeypatch, mode):
    rollout, clock, sends = make_rollout(monkeypatch, mode=mode)
    rollout.control_needs = ObservationNeeds(camera_fields=('rgb',))
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
    rollout.step()
    assert bool(sends) == (mapping_ns < rollout.dt_ns)
    if stale:
        assert 'control_sources' in rollout.result.termination_details
    elif not sends:
        assert rollout.plan is None and not rollout.finished
        clock[0] += 3_000_000_000
        rollout.step()
        assert not rollout.finished


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
    assert submissions == [[origin, origin + 62_500_000],
                           [origin + 62_500_000, origin + 225_000_000]]
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
    loop.robot = SimpleNamespace(service_feedback=lambda: None, service_idle=lambda: None)
    loop.phase, loop.next_tick_ns = TeleopPhase.ACTIVE, None
    loop.recorder = loop.pending_termination_reason = None
    loop.capture_failed = False
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
        for delta in (0, 62_500_000, 125_000_000):
            clock[0] = origin + delta
            rollout.step()
        worker.future.result(timeout=1)
        # 预测等待形成不等间隔历史；窗口仍可用于预取。
        clock[0] += 100_000_000
        rollout.step()
        clock[0] = rollout.next_step_ns
        rollout.step()
        worker.future.result(timeout=1)
        assert len(sends) == len(calls) == 2
        assert calls[0][0] == (origin, origin + 62_500_000, origin + 125_000_000)
        assert calls[1][0] == (origin + 125_000_000, origin + 225_000_000, origin + 250_000_000)
        if mode == 'rtc':
            assert calls[1][1]['delay_steps'] == 1
            np.testing.assert_array_equal(calls[1][1]['rtc_prefix'], actions[1:])
        else:
            assert calls[1][1] == {}
        assert not rollout.finished
    finally:
        worker.close(rollout.runtime_state).result(timeout=1)


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
        assert rollout.runtime_state.quit_requested
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
    session.robot = SimpleNamespace(check=lambda: None)
    session.audio = SimpleNamespace(play=lambda sound: None)
    session.get_home_planner = lambda: None
    session.home_abort_requested = lambda: False
    def home(*args, **kwargs):
        if cancelled:
            state.request_stop()
        raise OperationCancelled('home interrupted')
    monkeypatch.setattr('dexmani_real.deployment.session.home_robot', home)
    if cancelled:
        assert session.run_home(epoch)
        assert state.quit_requested
    else:
        with pytest.raises(OperationCancelled, match='home interrupted'):
            session.run_home(epoch)
        assert not state.quit_requested


def test_rollout_releases_completed_recorder_once(monkeypatch):
    rollout, _, _ = make_rollout(monkeypatch)
    writes, sounds = [], []
    rollout.result.termination_reason = 'operator'
    rollout.audio.play = sounds.append
    rollout.recorder = SimpleNamespace(
        finish=lambda **kwargs: writes.append(kwargs) or '/published',
        result=lambda: {'recording_truncated_reason': None, 'path': '/published'},
    )
    rollout.finish_recording()
    rollout.finish_recording()
    assert len(writes) == 1 and sounds == ['save']
    assert rollout.recorder is None and rollout.result.raw['path'] == '/published'


def test_policy_secondary_sensor_error_does_not_poison_writer(monkeypatch):
    rollout, _, _ = make_rollout(monkeypatch)
    rollout.sync_actions = np.zeros((4, 19))
    rollout.recorder = SimpleNamespace(error=None, check_error=lambda: None, append=lambda *a, **kw: None)
    reads = []
    def quality(row):
        reads.append(True)
        if len(reads) == 2:
            raise RuntimeError('camera exited')
    rollout.recording_issue = quality
    def mapping(*args):
        raise ValueError('mapping failed')
    rollout.map_command = mapping
    with pytest.raises(ValueError, match='mapping failed'):
        rollout.step()
    assert rollout.recorder.error is None and len(reads) == 2
    assert rollout.runtime_state.stop_signal.is_set()


def test_policy_rejects_initial_arm_posture_outside_operational_limits(monkeypatch):
    rollout, _, _ = make_rollout(monkeypatch)
    row, _ = rollout.read_observation()
    row.arm['qpos'][0, 0] = rollout.runtime.arm.joint_limit_upper[0] + .1
    rollout.read_observation = lambda: (row, None)
    rollout.robot.service_feedback = lambda: None
    with pytest.raises(ValueError, match='operational joint limits'):
        rollout.start_observation()


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
        runtime_config_json=None, keyboard=keyboard, audio=rollout.audio,
    )
    loop.poll_blocking_commands = lambda: False
    def prepare(epoch):
        if cancelled:
            state.request_stop()
        raise OperationCancelled('prepare interrupted')
    rollout.robot.prepare_streaming = prepare
    if cancelled:
        loop.start_episode(epoch)
        assert state.quit_requested
    else:
        with pytest.raises(OperationCancelled, match='prepare interrupted'):
            loop.start_episode(epoch)
        assert not state.quit_requested

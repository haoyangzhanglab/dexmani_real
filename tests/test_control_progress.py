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
from dexmani_real.runtime.observation import Observation, ObservationNeeds


def make_rollout(monkeypatch, *, mode='sync', history=1):
    clock = [1_000_000_000]
    monkeypatch.setattr('dexmani_real.deployment.rollout.time.monotonic_ns', lambda: clock[0])
    sends = []
    runtime = RealConfig()
    runtime = replace(runtime, execution=replace(runtime.execution, mode=mode, action_steps=4, prefetch_steps=1))
    rollout = Rollout(SimpleNamespace(can_move=lambda **kwargs: True), runtime,
                      SimpleNamespace(control_dt_s=.0625, n_obs_steps=history), cancel_epoch=1,
                      robot=SimpleNamespace(send_action=lambda *a, **kw: sends.append(clock[0])),
                      observations=SimpleNamespace(project=lambda row: row),
                      control_needs=ObservationNeeds(), record_needs=ObservationNeeds(),
                      sensors=None, action_mapper=None,
                      model=SimpleNamespace(future=None, poll=lambda: None, predict=lambda *a, **kw: None),
                      max_running_s=None, index=1, output_dir='/tmp', record_raw=False,
                      audio=SimpleNamespace(play=lambda sound: None), keyboard=None)
    rollout.started_ns = rollout.next_step_ns = clock[0]
    rollout.wait_deadline_ns = clock[0] + 2_000_000_000

    def observe():
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm['timestamp_ns'] = hand['timestamp_ns'] = clock[0]
        return Observation(arm, hand, None, None, None, clock[0], control_timestamp_ns=clock[0]), None

    rollout.read_observation = observe
    return rollout, clock, sends


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


def test_async_history_uses_sample_gap(monkeypatch):
    rollout, clock, _ = make_rollout(monkeypatch, mode='async', history=2)
    origin = clock[0]
    for delta in (61_000_000, 125_100_000):
        clock[0] = origin + delta
        rollout.step()
    assert len(rollout.history) == 2
    # 真实大于 1.5dt 的断档，旧窗口被丢弃。
    clock[0] += 120_000_000
    rollout.step()
    assert len(rollout.history) == 1


def test_duration_wins_equal_wait_deadline(monkeypatch):
    rollout, clock, _ = make_rollout(monkeypatch)
    rollout.run_deadline_ns = rollout.wait_deadline_ns = clock[0]
    assert rollout.expired_budget() == (clock[0], 'duration')


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
    rollout.runtime_state.estop_request = False
    rollout.runtime_state.revoke_motion = lambda **kw: None
    rollout.robot.stop = lambda: order.append('stop')
    rollout.robot.close = lambda: order.append('robot_close')
    rollout.prepare = lambda poll: (_ for _ in ()).throw(ValueError('prepare failed'))
    rollout.finish_recording = lambda: order.append('writer_finish')
    rollout.model.discard_pending = lambda timeout: order.append('model_finish')
    with pytest.raises(ValueError, match='prepare failed'):
        rollout.run(poll_operator=lambda: None)
    assert order == ['stop', 'robot_close', 'writer_finish', 'model_finish']
    assert rollout.result.errors[0]['message'] == 'prepare failed'


def test_vr_expiring_during_mapping_cannot_receive_new_lease(monkeypatch):
    from dexmani_real.runtime.state import SafetyState, SessionState
    from dexmani_real.teleop.loop import TeleopLoop
    rollout, clock, sends = make_rollout(monkeypatch)
    row, _ = rollout.read_observation()
    row = replace(row, vr={'recv_ts_ns': clock[0]})
    loop = TeleopLoop.__new__(TeleopLoop)
    loop.runtime = rollout.runtime
    loop.runtime_state = SessionState(safety_state=SafetyState.RUNNING)
    loop.cancel_epoch = loop.runtime_state.cancel_epoch
    loop.control_needs = ObservationNeeds(vr=True)
    loop.recorder = None
    loop.robot = rollout.robot
    ended = []
    loop.request_episode_end = ended.append
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
    rollout.runtime_state.estop_request = False
    rollout.runtime_state.revoke_motion = lambda **kw: None
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
    assert order == ['stop', 'close', 'writer', 'model']
    assert rollout.result.errors[0]['stage'] == 'stop'


@pytest.mark.parametrize('loss', ['missing', 'stale', 'worker_exit'])
def test_archive_source_loss_freezes_recording_while_control_continues(monkeypatch, tmp_path, loss):
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
    assert len(sends) == 3
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
    from dexmani_real.runtime.state import SafetyState, SessionState
    rollout, clock, sends = make_rollout(monkeypatch)
    row, _ = rollout.read_observation()
    monkeypatch.setattr('dexmani_real.runtime.jog.read_observation', lambda *a: row)
    state = SessionState(safety_state=SafetyState.RUNNING)
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
    from dexmani_real.runtime.state import SafetyState, SessionState
    from dexmani_real.teleop.loop import TeleopLoop, TeleopPhase
    rollout, clock, _ = make_rollout(monkeypatch)
    state = SessionState(safety_state=SafetyState.ARMED)
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
    assert order == ['send', 'commit', 'stop', 'close', 'clear']
    assert state.quit_requested and loop.phase is TeleopPhase.IDLE



def test_teleop_final_mapping_failure_freezes_before_invalid_action_row(monkeypatch, tmp_path):
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
    loop.request_episode_end = ended.append
    loop.controller = SimpleNamespace(compute_target=lambda *a, **kw: SimpleNamespace(
        rejection_reason='no feasible target', arm_qpos=None))
    loop.execute_control_step(row)
    assert ended == ['target_unavailable: no feasible target']
    assert loop.recorder.queue.qsize() == 1
    assert loop.recorder.recording_truncated_reason == 'recording_no_command'
    assert loop.recorder.queue.qsize() == 1 and loop.recorder.finish_request is None


def test_close_interrupt_latches_emergency_before_waiting_for_writer(monkeypatch):
    from dexmani_real.runtime.state import RunEndReason, SafetyState, SessionState
    rollout, _, _ = make_rollout(monkeypatch)
    rollout.runtime_state = SessionState(safety_state=SafetyState.RUNNING, quit_requested=True)
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
    with pytest.raises(KeyboardInterrupt):
        rollout.run(poll_operator=lambda: None)
    assert cleanup == ['writer', 'model']
    assert rollout.result.errors[0]['stage'] == 'robot_close'


@pytest.mark.parametrize('prediction_ns,mapping_ns', [(600_000_000, 0), (100_000_000, 63_000_000)])
def test_async_zero_progress_keeps_first_absolute_budget(monkeypatch, prediction_ns, mapping_ns):
    from concurrent.futures import Future
    rollout, clock, sends = make_rollout(monkeypatch, mode='async')
    rollout.runtime_state.revoke_motion = lambda: None
    rollout.policy_info.future_steps = 8
    rollout.policy_info.control_action_dim = 19
    deadline = rollout.wait_deadline_ns
    pending = [None, None]
    def predict(*args, **kwargs):
        pending[:] = [Future(), clock[0] + prediction_ns]
    def poll():
        if pending[0] is None or clock[0] < pending[1]:
            return None
        future = pending[0]
        pending[0] = None
        future.set_result(np.zeros((8, 19)))
        return future
    def mapping(*args):
        clock[0] += mapping_ns
        return RobotCommand(np.zeros(7)), None
    rollout.model.predict, rollout.model.poll = predict, poll
    rollout.map_command = mapping
    with pytest.raises(TimeoutError, match='budget expired'):
        for _ in range(4000):
            rollout.step()
            clock[0] += 1_000_000
    assert not sends
    assert rollout.wait_deadline_ns == deadline
    assert clock[0] <= deadline + mapping_ns + 1_000_000


def test_target_lease_bounds_progress_even_with_plan(monkeypatch):
    rollout, clock, _ = make_rollout(monkeypatch, mode='async')
    rollout.wait_deadline_ns = None
    rollout.target_expires_ns = clock[0]
    assert rollout.expired_budget() == (clock[0], 'wait')
    rollout.run_deadline_ns = clock[0]
    assert rollout.expired_budget() == (clock[0], 'duration')


@pytest.mark.parametrize('control_camera', [False, True])
def test_frozen_recorder_stops_archive_only_reads(monkeypatch, control_camera):
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
    assert needs == [rollout.capture_needs, control]


@pytest.mark.parametrize('model_fails', [False, True])
def test_duration_and_owner_lease_stop_together_preserve_faults(monkeypatch, model_fails):
    from concurrent.futures import Future

    from dexmani_real.runtime.state import SafetyState, SessionState

    rollout, clock, sends = make_rollout(monkeypatch, mode='async')
    rollout.runtime_state = state = SessionState(safety_state=SafetyState.RUNNING)
    rollout.cancel_epoch = state.cancel_epoch
    state.stop_signal.set()  # owner 已观察到目标租期到期。
    rollout.robot.check = lambda: None
    rollout.robot.stop = lambda: None
    rollout.run_deadline_ns = rollout.target_expires_ns = clock[0]
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
    from dexmani_real.runtime.state import SafetyState, SessionState

    rollout, clock, _ = make_rollout(monkeypatch)
    rollout.runtime_state = state = SessionState(safety_state=SafetyState.RUNNING)
    rollout.cancel_epoch = state.cancel_epoch
    rollout.robot.check = lambda: None
    rollout.run_deadline_ns = clock[0]
    state.request_stop()
    with pytest.raises(OperationCancelled):
        rollout.step()
    assert not rollout.finished  # run 的操作者轮询消费终止输入后确定当前段原因。

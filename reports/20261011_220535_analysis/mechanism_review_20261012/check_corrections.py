"""Focused offline regression cases. All device calls are fakes; no hardware SDK is loaded."""
from concurrent.futures import CancelledError
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from dexmani_real.deployment import runner as runner_module
from dexmani_real.deployment.config import ExecutionConfig
from dexmani_real.deployment.runner import PolicyRunner, Plan, Query
from dexmani_real.deployment.observation import policy_sources
from dexmani_real.ipc.schema import ARM_STATE_DTYPE, HAND_STATE_DTYPE
from dexmani_real.robot.action import ActionRealization
from dexmani_real.robot.commands import DispatchStatus, RobotCommand
from dexmani_real.robot.drivers.xhand import XHand, XHandError
from dexmani_real.robot.robot import DexManiRobot, DispatchError, DispatchResult
from dexmani_real.runtime.observation import ObservationRow
from dexmani_real.runtime.safety import SafetyState, revoke_motion

logging.disable(logging.CRITICAL)
BASE = 1_000_000_000
DT = 62_500_000
cases = {}


def row_at(now):
    arm, hand = np.zeros(1, ARM_STATE_DTYPE), np.zeros(1, HAND_STATE_DTYPE)
    arm['timestamp_ns'] = hand['timestamp_ns'] = now
    return ObservationRow(arm, hand, None, None, None, now, control_timestamp_ns=now)


class Model:
    def __init__(self, clock, delay):
        self.clock, self.delay = clock, delay
        self.future, self.count, self.requests = None, 0, []

    def submit(self, operation, observation, **kwargs):
        assert self.future is None and operation == 'predict'
        self.count += 1
        actions = np.repeat(np.arange(15, dtype=float)[:, None], 19, axis=1)
        self.actions = (actions + self.count * 100) / 1000
        self.started = self.clock.now
        self.ready = self.started + self.delay
        self.future = object()
        self.requests.append(dict(at_ns=self.started, kwargs=kwargs))

    def poll(self):
        if self.future is None or self.clock.now < self.ready:
            return None
        self.future = None
        return 'predict', NS(value=self.actions, started_ns=self.started,
                             completed_ns=self.ready, error=None)


def make_runner(mode='async', *, a=4, d=1, delay=38_000_000, read_ns=12_000_000):
    clock = NS(now=BASE)
    shared = NS(sensors=NS(is_running=NS(value=True)), error_state=False, estop_request=False,
                quit_requested=False, run_id=7, safety_state=SafetyState.RUNNING,
                motion_lock=threading.Lock(), start_request=False, stop_request=0,
                run_ended_reason=0, run_ended_id=-1)
    execution = ExecutionConfig(action_steps=a, execution_mode=mode, prefetch_steps=d,
                                rtc_guidance_cap=0 if mode == 'rtc' else None)
    info = NS(control_dt_s=DT/1e9, horizon=16, n_obs_steps=2, n_action_steps=a,
              action_mode='joint', observation_fields=('joint_state',))
    execution.validate(info)
    runtime = NS(execution=execution, arm=NS(feedback_max_age_s=.133333333),
                 hand=NS(feedback_max_age_s=.133333333))
    model = Model(clock, delay)
    sends, rows, stops = [], [], []
    def send(command, *, valid_until_ns):
        assert clock.now < valid_until_ns
        sends.append(dict(at_ns=clock.now, index=int(round(command.arm_qpos[0]*1000)),
                          deadline=valid_until_ns))
        clock.now += 11_000_000
        return DispatchResult(DispatchStatus.ACCEPTED, DispatchStatus.ACCEPTED,
                              timestamp_ns=clock.now)
    robot = NS(check=lambda: None, stop=lambda: stops.append(clock.now), send_action=send)
    realizer = NS(realize=lambda intent, *args: ActionRealization(intent.arm, intent.hand),
                  frozen_rejection=lambda *args: None,
                  control_from_command=lambda command, *args: np.concatenate((command.arm_qpos, command.hand_qpos)))
    runner = PolicyRunner(shared, runtime, info, robot=robot, realizer=realizer,
                          model_runtime=model, kinematics=None, execute=True,
                          max_running_s=10, num_episodes=1)
    runner.run_id, runner.started_ns, runner.next_step_ns = 7, BASE, BASE
    def read():
        clock.now += read_ns
        row = row_at(clock.now)
        rows.append(row)
        return row
    runner._read_observation = read
    return runner, clock, sends, rows, stops


def advance(r, clock, sends, count):
    with patch.object(runner_module, 'time', NS(monotonic_ns=lambda: clock.now)):
        for _ in range(150):
            r.step()
            if len(sends) >= count or r.run_id is None:
                return
            pending = [r.next_step_ns]
            if r.model.future is not None:
                pending.append(r.model.ready)
            clock.now = max(clock.now, min(pending))
        raise AssertionError('scheduler stalled')


# Sync keeps the entire head and resumes as soon as model results are ready,
# with a new physical row before sending. Steady actions still have one dt spacing.
for a in (1, 4, 8):
    r, c, sends, rows, stops = make_runner('sync', a=a)
    advance(r, c, sends, 3*a)
    assert not stops and r.run_id == 7
    assert [s['index'] for s in sends] == [q*100+i for q in range(1, 4) for i in range(a)]
    expected_first = BASE + DT + 12_000_000 + 38_000_000 + 12_000_000
    assert sends[0]['at_ns'] == expected_first
    assert sends[0]['at_ns'] < BASE + 2*DT + 12_000_000
    assert rows[2].observation_timestamp_ns == sends[0]['at_ns']
    for group in range(3):
        times = [s['at_ns'] for s in sends[group*a:(group+1)*a]]
        assert np.all(np.diff(times) == DT)
    cases[f'sync_A{a}'] = dict(first_send_ms=(expected_first-BASE)/1e6,
                             action_indices=[s['index'] for s in sends], fresh_row=True)

# A synchronous model taking several control periods must still retain action 0.
r, c, sends, rows, stops = make_runner('sync', a=4, delay=150_000_000)
advance(r, c, sends, 8)
assert not stops and [s['index'] for s in sends] == list(range(100,104)) + list(range(200,204))
cases['slow_sync'] = dict(inference_ms=150, head_preserved=True)

# Async / RTC keep the control grid; new predictions start at d after bootstrap.
for mode in ('async', 'rtc'):
    for a, d in ((1, 1), (4, 1), (8, 2)):
        r, c, sends, rows, stops = make_runner(mode, a=a, d=d)
        advance(r, c, sends, 3*a)
        assert not stops and r.run_id == 7
        expected = list(range(100, 100+a))
        expected += [q*100+i for q in (2, 3) for i in range(d, d+a)]
        assert [s['index'] for s in sends] == expected
        assert np.all(np.diff([s['at_ns'] for s in sends]) == DT)
        if mode == 'rtc':
            prefix = r.model.requests[1]['kwargs']
            assert prefix['delay_steps'] == d
            assert np.array_equal(prefix['rtc_prefix'][:d, 0],
                                  np.array(list(range(100+a-d, 100+a)))/1000)
        cases[f'{mode}_A{a}_d{d}'] = dict(action_indices=expected, interval_ms=DT/1e6)

# A prediction completed 5 ms past the nominal tick is usable after the 12 ms read.
r, c, sends, rows, stops = make_runner(delay=55_500_000)
advance(r, c, sends, 8)
assert not stops and np.all(np.diff([s['at_ns'] for s in sends]) == DT)
cases['ready_after_nominal_tick'] = dict(complete_past_tick_ms=5, consumed_past_tick_ms=12,
                                        interval_ms=DT/1e6)

# Crossing the actual slot end is forbidden even if fresh device feedback arrives then.
r, c, sends, rows, stops = make_runner(read_ns=70_000_000)
r.plan = Plan(np.ones((15, 19)), 0, 4, {'joint_state': (BASE,)}, {}, 1)
with patch.object(runner_module, 'time', NS(monotonic_ns=lambda: c.now)):
    r.step()
assert not sends and stops and r.termination_timing['reason'] == 'dispatch_deadline_expired'
cases['slow_read'] = dict(read_ms=70, slot_ms=62.5, sends=0,
                         reason=r.termination_timing['reason'])

# Missing result, expired decision, overrun and revocation still stop; no stale fallback.
r, c, sends, rows, stops = make_runner(delay=100_000_000)
advance(r, c, sends, 8)
assert r.termination_timing['reason'] == 'handoff_result_not_ready' and len(sends) == 4
cases['handoff_not_ready'] = dict(sends=len(sends), reason=r.termination_timing['reason'])
for kind in ('missed_slot', 'revoked', 'expired'):
    r, c, sends, rows, stops = make_runner()
    r.plan = Plan(np.ones((15, 19)), 0, 4, {'joint_state': (BASE,)}, {}, 1)
    if kind == 'missed_slot':
        c.now += DT
    elif kind == 'revoked':
        revoke_motion(r.shared)
    else:
        r.plan.sources = {'joint_state': (BASE-1,)}
        r.execution = replace(r.execution, max_decision_age_s=.001)
    with patch.object(runner_module, 'time', NS(monotonic_ns=lambda: c.now)):
        r.step()
    assert not sends and stops and r.run_id is None
    cases[kind] = dict(sends=0, reason=r.termination_timing['reason'])

# Polling a retired result never installs a plan or re-grants authority.
r, c, sends, rows, stops = make_runner('sync')
with patch.object(runner_module, 'time', NS(monotonic_ns=lambda: c.now)):
    r.step(); c.now = r.next_step_ns; r.step()
    assert r.model.future is not None
    revoke_motion(r.shared)
    r.step()
    c.now = r.model.ready
    r._poll_model()
assert r.run_id is None and r.query is None and r.plan is None and not sends
cases['retired_result'] = 'discarded without sending'

# Runtime budgets remain effective without a warmup-derived startup gate.
r, c, sends, rows, stops = make_runner('sync', delay=200_000_000)
r.execution = replace(r.execution, max_wait_s=.08).validate(r.policy_info)
advance(r, c, sends, 1)
assert not sends and r.termination_timing['reason'] == 'wait_timeout'
cases['wait_budget'] = r.termination_timing['reason']
try:
    ExecutionConfig(action_steps=15, execution_mode='async', prefetch_steps=1).validate(r.policy_info)
    raise AssertionError('invalid horizon accepted')
except ValueError:
    pass
cases['horizon_alignment'] = 'A+d>P remains rejected'

# Use the actual XHand parsing/send methods with an in-memory SDK reply only.
def fake_hand(code=0, tip_code=0, joint_code=0, comm_code=0):
    hand = XHand.__new__(XHand)
    hand.cfg = NS(device_id=0, comm_type='serial', mechanical_qpos_min_rad=np.full(12, -2.),
                  mechanical_qpos_max_rad=np.full(12, 2.))
    hand.connected_flag = True
    fingers = [NS(id=i, position=.1, torque=10., commboard_err=0,
                  jonitboard_err=0, tipboard_err=0) for i in range(12)]
    fingers[2].tipboard_err = tip_code
    fingers[2].jonitboard_err = joint_code
    fingers[2].commboard_err = comm_code
    state = NS(finger_state=fingers, sensor_data=[
        NS(calc_force=NS(fx=1., fy=2., fz=3.),
           raw_force=[NS(fx=1., fy=2., fz=3.) for _ in range(120)]) for _ in range(5)])
    reply = NS(error_code=code, error_message='fake SDK status')
    hand._control = NS(read_state=lambda *args: (reply, state), send_command=lambda *args: reply)
    hand._command = NS(finger_command=[NS(mode=0, position=0.) for _ in range(12)])
    hand._tactile_reset_metadata = {}
    return hand

assert np.isfinite(fake_hand().get_state().tactile_dense).all()
state = fake_hand(1501019).get_state()
assert np.isfinite(state.tactile_aggregate).all() and np.isnan(state.tactile_dense).all()
assert fake_hand(1501070).get_state() is None
assert fake_hand(0, joint_code=307).get_state() is None
assert fake_hand(0, comm_code=307).get_state() is None
for tip in (200, 201, 202, 307):
    state = fake_hand(0, tip_code=tip).get_state()
    assert state is not None
    assert np.isfinite(state.tactile_aggregate[1:]).all()
    assert np.isfinite(state.tactile_dense[1:]).all()
    assert np.isfinite(state.tactile_aggregate[0]).all() == (tip in (201, 202))
    assert np.isfinite(state.tactile_dense[0]).all() == (tip == 202)
cases['XHand_reads'] = 'CRC / comm / joint failures rejected; partial tactile and thumb mapping preserved'
assert fake_hand(1501070)._send_command() == DispatchStatus.CRC_UNCONFIRMED
for code in (1501035, 1501002):
    try:
        fake_hand(code)._send_command()
        raise AssertionError('fault accepted')
    except XHandError as exc:
        assert exc.code == code and str(code) in str(exc)
# Owner records uncertain delivery rather than claiming rejection before a send.
owner = DexManiRobot.__new__(DexManiRobot)
owner._owner, owner._connected, owner._hand_stop_pending = threading.get_ident(), True, False
owner._arm_stopped, owner._motion_active = False, False
owner.hand, owner.arm = fake_hand(1501035), None
owner.runtime = NS(hand=owner.hand.cfg)
owner._authorized = lambda *args: True
try:
    owner.send_action(RobotCommand(7, hand_qpos=np.zeros(12)), valid_until_ns=10**30)
    raise AssertionError('overcurrent accepted by owner')
except DispatchError as exc:
    assert exc.result.hand == DispatchStatus.UNKNOWN and '1501035' in str(exc)
cases['XHand_send'] = 'overcurrent preserved as 1501035; UNKNOWN delivery; CRC remains unconfirmed'

# Reset once per physical fingertip, accurately report SDK calls, stop on cancellation/error.
for kind in ('complete', 'cancel', 'error'):
    hand, calls = fake_hand(), []
    def reset(device, sensor):
        calls.append(sensor)
        return NS(error_code=1 if kind == 'error' else 0, error_message='fake')
    hand._control.reset_sensor = reset
    try:
        ok = hand.reset_tactile_sensors(cancel_requested=lambda: kind == 'cancel' and len(calls) == 2)
    except CancelledError:
        assert kind == 'cancel'
    else:
        assert ok == (kind == 'complete')
    assert len(calls) == {'complete':5, 'cancel':2, 'error':1}[kind]
    assert calls == list(range(0x11, 0x11+len(calls)))
    meta = hand.get_tactile_reset_metadata()
    assert 'hand_tactile_zero_result' not in meta and meta['hand_tactile_processing'] == 'sdk_native'
    cases[f'reset_{kind}'] = meta['hand_tactile_reset_result']

# Follow the renamed reset metadata through the real recording boundary.
import inspect
from dexmani_real.recording.recorder import AsyncEpisodeRecorder, snapshot_recording_metadata
from dexmani_real.sensor.camera.geometry import CameraIntrinsics, RGBDGeometry
from dexmani_real.runtime.operator_input import _KEY_MAP, OperatorCommand
intrinsics = CameraIntrinsics(2, 2, 2., 2., 1., 1., 'none', (0.,)*5)
geometry = RGBDGeometry(intrinsics, intrinsics, np.eye(4))
shared = NS(sensors=NS(camera_serial=NS(value=b'fake-camera'),
            camera_geometry=NS(value=json.dumps(geometry.to_dict()).encode()),
            camera_depth_scale=NS(value=.001)))
metadata = snapshot_recording_metadata(shared, collection_source='policy_rollout',
                                       camera_calibration=None, hand=hand)
recorder = AsyncEpisodeRecorder.__new__(AsyncEpisodeRecorder)
recorder._rgb_shape, recorder.control_hz = (2, 2, 3), 16.
inspect.signature(recorder.start_episode).bind(task_label='offline', **metadata)
saved = recorder._snapshot_start_metadata(task_label='offline', **metadata)
assert saved['hand_tactile_reset_result'] == hand.get_tactile_reset_metadata()['hand_tactile_reset_result']
assert 'hand_tactile_zero_result' not in saved
assert _KEY_MAP['t'] is OperatorCommand.RESET_TACTILE
cases['reset_metadata'] = 'T key and driver -> recording metadata matched; no writer or device'

from dexmani_real.recording.recorder import RecordingError
from dexmani_real.robot.robot import DispatchInterrupted
# Each terminal control path stops once and submits the current row at most once.
for kind in ('sdk_error', 'revoked_partial', 'interrupt', 'writer_error', 'sdk_and_writer_error'):
    r, c, sends, rows, stops = make_runner()
    r.plan = Plan(np.ones((15, 19)), 0, 4, {'joint_state': (BASE,)}, {}, 1)
    events, recorded = [], []
    r.robot.stop = lambda: events.append('stop')
    def add(row, command, result, **kwargs):
        events.append('record'); recorded.append((row, command, result))
        if 'writer_error' in kind:
            raise RecordingError('fake writer failure')
    recorder = NS(check_error=lambda: None, add_frame=add, save_episode=lambda **k: None,
                  staging_path=None, written_frames=0)
    r.recorder, r.recording_started = recorder, True
    if kind != 'writer_error':
        def send(command, **kwargs):
            partial = DispatchResult(DispatchStatus.ACCEPTED,
                DispatchStatus.NOT_CALLED if kind == 'revoked_partial' else DispatchStatus.UNKNOWN,
                timestamp_ns=c.now)
            if kind == 'interrupt': raise DispatchInterrupted(partial)
            raise DispatchError('fake dispatch failure', partial,
                                revoked=kind == 'revoked_partial',
                                cause='authority_revoked' if kind == 'revoked_partial' else None)
        r.robot.send_action = send
    failure = None
    with patch.object(runner_module, 'time', NS(monotonic_ns=lambda: c.now)):
        try: r.step()
        except (DispatchError, DispatchInterrupted, RecordingError) as exc: failure = exc
    assert r.run_id is None and events.count('stop') == events.count('record') == 1
    assert (failure is None) == (kind == 'revoked_partial')
    if kind != 'writer_error':
        assert events.index('stop') < events.index('record')
        assert recorded[0][2].arm == DispatchStatus.ACCEPTED
    if kind == 'sdk_and_writer_error': assert isinstance(failure, DispatchError)
    cases[f'policy_{kind}'] = 'one stop; one row attempt; partial result and original failure preserved'

assert 'xhand_controller' not in sys.modules and 'pyrealsense2' not in sys.modules
cases['hardware'] = 'no hardware SDK loaded; only in-memory fake calls'
root = ROOT
files = ['dexmani_real/deployment/runner.py', 'dexmani_real/deployment/config.py',
         'dexmani_real/deployment/session.py', 'dexmani_real/robot/drivers/xhand.py']
report = dict(cases=cases, source_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                        for name in files})
Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
print(json.dumps(report, indent=2, ensure_ascii=False))

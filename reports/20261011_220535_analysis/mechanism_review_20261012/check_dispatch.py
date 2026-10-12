"""Offline dispatch regressions: real owners/runners, fake SDK, clock and writer."""
from contextlib import ExitStack
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
from dexmani_real.robot import robot as robot_module
from dexmani_real.robot.action import ActionRealization
from dexmani_real.robot.commands import DispatchStatus as Status, RobotCommand
from dexmani_real.robot.robot import DexManiRobot, DispatchError, DispatchInterrupted
from dexmani_real.recording.recorder import RecordingError
from dexmani_real.runtime import safety
from dexmani_real.runtime.observation import ObservationRow
from dexmani_real.teleop import runner as teleop_module
from dexmani_real.teleop.runner import TeleopRunner

logging.disable(logging.CRITICAL)
BASE, DT = 1_000_000_000, 62_500_000
cases = {}


def setup(*, recording=False):
    clock, events = NS(now=BASE), []
    shared = NS(sensors=NS(is_running=NS(value=True)), run_id=7,
                safety_state=safety.SafetyState.RUNNING, run_ended_id=-1, run_ended_reason=0,
                estop_request=False, error_state=False, quit_requested=False, stop_request=0,
                motion_lock=threading.RLock())
    runtime = NS(arm=NS(feedback_max_age_s=.2),
                 hand=NS(feedback_max_age_s=.2, state_read_failure_timeout_s=1.,
                         mechanical_qpos_min_rad=np.full(12, -2.),
                         mechanical_qpos_max_rad=np.full(12, 2.)),
                 teleop=NS(control_hz=16),
                 policy=NS(hand_enabled=True, max_record_duration_s=100.,
                           vr_mapping=NS(stale_threshold_s=.2)))
    robot = DexManiRobot.__new__(DexManiRobot)
    robot.shared, robot.runtime = shared, runtime
    robot._owner, robot._connected = threading.get_ident(), True
    robot._arm_stopped = robot._motion_active = robot._hand_stop_pending = False
    robot._last_hand = robot._hand_failure_started = robot._previous_errors = None
    robot._idle_next_ns = 0
    robot.check_services = lambda: events.append('health') or True
    def send_arm(q):
        events.append('arm_send'); clock.now += 2_000_000
        return 0
    def enter_mode6(*, check_abort):
        check_abort(); events.append('mode6'); clock.now += 80_000_000; check_abort()
    robot.arm = NS(is_connected=True, error_code=0, servo=send_arm,
                   enter_mode6=enter_mode6,
                   read=lambda: (np.zeros(7), np.zeros(7), np.zeros(7)),
                   stop=lambda: events.append('arm_stop'),
                   emergency_stop=lambda: events.append('arm_estop'))
    hand_state = NS(qpos=np.zeros(12), current_ma=np.zeros(12),
                    tactile_aggregate=np.zeros((5, 3)), tactile_dense=np.zeros((5, 120, 3)),
                    commboard_err=np.zeros(12), jointboard_err=np.zeros(12), tipboard_err=np.zeros(12))
    robot.hand = NS(is_connected=True, get_state=lambda: hand_state,
                    send_action=lambda q: events.append('hand_send') or Status.ACCEPTED,
                    stop=lambda *a, **k: events.append('hand_stop'))
    def read(*args, **kwargs):
        clock.now += 12_000_000
        state = robot.read_state()
        return ObservationRow(state.arm, state.hand, None, {'recv_ts_ns': clock.now},
                              None, clock.now, control_timestamp_ns=clock.now-12_000_000)
    realization = ActionRealization(np.zeros(7), np.zeros(12), eef_pose=np.zeros(9))
    runner = TeleopRunner.__new__(TeleopRunner)
    runner.shared, runner.runtime, runner.robot = shared, runtime, robot
    runner.controller = NS(compute_target=lambda row: realization,
                           commit_dispatched_target=lambda target: events.append('commit'),
                           clear_reference=lambda: None, reset_reference=lambda row: True)
    runner.active, runner.paused, runner.resume_requested = True, False, False
    runner.preparing_epoch, runner.control_run_id = None, 7
    runner.resume_after_ns, runner.next_tick_ns, runner.tick_origin_ns = 0, BASE, BASE
    runner.control_dt_ns, runner.tick_index, runner.capture_started_s = DT, 0, BASE/1e9
    runner.pending_termination_reason, runner.quit_pending = None, False
    runner.termination_details = []
    runner.audio = NS(play=lambda *a: None, queue=lambda *a: None,
                      wait_until_idle=lambda **k: True, close=lambda: None)
    runner.task_label, runner.camera_calibration = 'offline', None
    rows = []
    recorder = NS(is_recording=True, check_error=lambda: None)
    def add(row, command=None, dispatch=None, **kwargs):
        events.append('record'); rows.append((row, command, dispatch))
    def save(**kwargs):
        events.append('save'); recorder.is_recording=False
    recorder.add_frame, recorder.save_episode = add, save
    runner.recorder = recorder if recording else None
    fake_time = NS(monotonic_ns=lambda: clock.now, monotonic=lambda: clock.now/1e9)
    stack = ExitStack()
    for module in (robot_module, teleop_module, safety):
        stack.enter_context(patch.object(module, 'time', fake_time))
    stack.enter_context(patch.object(teleop_module, 'read_observation', read))
    return runner, robot, clock, events, rows, stack


# The old 15.625 ms gate rejected 5 ms scheduling + 12 ms read even without capture.
for recording in (False, True):
    for jitter_ms in (0, 5, 16):
        r, robot, c, events, rows, patches = setup(recording=recording)
        with patches:
            c.now += jitter_ms*1_000_000
            r._run_control_tick()
        assert events.count('arm_send') == events.count('hand_send') == 1 and r.active
        assert events.count('health') == 1 and 'arm_stop' not in events
        assert len(rows) == int(recording)
        if recording:
            assert rows[0][0].observation_timestamp_ns == BASE+(jitter_ms+12)*1_000_000
        cases[f'teleop_jitter_{jitter_ms}ms_recording_{recording}'] = 'sent; actual timestamps retained'

for kind in ('missed_slot', 'read_overrun', 'invalid_target'):
    for recording in (False, True):
        r, robot, c, events, rows, patches = setup(recording=recording)
        with patches:
            if kind == 'missed_slot':
                c.now += DT
            elif kind == 'read_overrun':
                c.now += DT-10_000_000
            else:
                r.controller.compute_target = lambda row: ActionRealization(None, None, rejection_reason='jump')
            r._run_control_tick()
        assert 'arm_send' not in events and 'hand_send' not in events and not r.active
        assert events.count('arm_stop') == 1
        assert len(rows) == int(recording and kind != 'missed_slot')
        cases[f'{kind}_recording_{recording}'] = 'stopped; no target sent'

# Check the real robot owner and runner, including partial dispatch evidence.
for kind in ('stop_between_devices', 'deadline_between_devices', 'arm_rejected',
             'hand_exception', 'hand_interrupt', 'late_hand_return'):
    r, robot, c, events, rows, patches = setup(recording=True)
    def arm(q):
        events.append('arm_send')
        if kind == 'stop_between_devices':
            safety.revoke_motion(r.shared, reason=safety.RunEndReason.OPERATOR)
        if kind == 'deadline_between_devices':
            c.now += DT
        return 42 if kind == 'arm_rejected' else 0
    def hand(q):
        events.append('hand_send')
        if kind == 'hand_exception': raise RuntimeError('fake SDK failure')
        if kind == 'hand_interrupt': raise KeyboardInterrupt()
        c.now += DT
        return Status.ACCEPTED
    robot.arm.servo, robot.hand.send_action = arm, hand
    failure = None
    with patches:
        try: r._run_control_tick()
        except (DispatchError, DispatchInterrupted) as exc: failure = exc
    assert not r.active and 'commit' not in events and len(rows) == 1
    stop = 'arm_estop' if kind == 'hand_interrupt' else 'arm_stop'
    assert events.count(stop) == 1 and events.index(stop) < events.index('record')
    result = rows[0][2]
    assert result.arm == (Status.REJECTED if kind == 'arm_rejected' else Status.ACCEPTED)
    assert result.hand == (Status.UNKNOWN if kind in ('hand_exception', 'hand_interrupt') else
                           Status.ACCEPTED if kind == 'late_hand_return' else Status.NOT_CALLED)
    assert bool(failure) == (kind in ('arm_rejected', 'hand_exception', 'hand_interrupt'))
    cases[kind] = dict(arm=result.arm.name, hand=result.hand.name, stop_before_record=True)

for kind in ('bad_hand', 'bad_arm', 'not_prepared', 'stop_unconfirmed'):
    r, robot, c, events, rows, patches = setup()
    arm, hand = np.zeros(7), np.zeros(12)
    if kind == 'bad_hand': hand[0] = np.nan
    if kind == 'bad_arm': arm[0] = 99.
    if kind == 'not_prepared': robot._arm_stopped = True
    if kind == 'stop_unconfirmed': robot._hand_stop_pending = True
    with patches:
        try:
            robot.send_action(RobotCommand(7, arm, hand), valid_until_ns=BASE+DT)
            raise AssertionError('invalid command sent')
        except DispatchError as exc:
            assert exc.result.arm == exc.result.hand == Status.NOT_CALLED
    assert not events
    cases[kind] = 'neither SDK called; no implicit mode switch'

# An explicit preparation can be revoked; granting a new epoch is never implicit.
for cancel in (False, True):
    r, robot, c, events, rows, patches = setup()
    r.shared.safety_state = safety.SafetyState.ARMED
    robot._arm_stopped = True
    if cancel:
        def mode(*, check_abort):
            safety.revoke_motion(r.shared); check_abort()
        robot.arm.enter_mode6 = mode
    with patches:
        try: robot.prepare_streaming(7)
        except DispatchError as exc: assert cancel and exc.revoked
    assert r.shared.safety_state == safety.SafetyState.ARMED
    assert robot._arm_stopped == cancel and 'arm_send' not in events
    cases[f'prepare_cancel_{cancel}'] = 'prepared' if not cancel else 'revocation preserved'

# Run the actual teleop loop through B -> preparation -> fresh reference -> one send.
r, robot, c, events, rows, patches = setup()
r.active, r.control_run_id, r.tick_origin_ns = False, None, None
r.shared.safety_state = safety.SafetyState.ARMED
r._initialize = lambda: True
def poll(*, timeout):
    c.now += max(1, int(timeout*1e9))
    if 'arm_send' in events:
        r.shared.quit_requested = True
    if 'mode6' not in events:
        return [teleop_module.OperatorCommand.BEGIN]
    return []
r.keyboard = NS(healthy=True, start=lambda: None, stop=lambda: None, poll=poll)
with patches:
    r.run()
assert events.count('mode6') == events.count('arm_send') == events.count('hand_send') == 1
assert events[events.index('mode6')+1:events.index('arm_send')].count('arm_stop') == 0
assert events.count('arm_stop') == 2  # start and final cleanup
cases['teleop_start_lifecycle'] = 'mode prepared once; fresh read; no idle re-stop before first send'

from dexmani_real.replay import replayer
from dexmani_real.calibration.camera import motion
from dexmani_real.planning import Pose
from dexmani_real.planning.kinematics.ik import IKResult
# Other arm dispatch entries must prepare before granting motion and refresh feedback.
for cancel in (False, True):
    r, robot, c, events, rows, patches = setup()
    r.shared.safety_state = safety.SafetyState.ARMED
    robot._arm_stopped = True
    r.runtime.arm.joint_limit_lower = np.full(7, -1.)
    r.runtime.arm.joint_limit_upper = np.full(7, 1.)
    r.runtime.hand.qpos_min_rad = np.full(12, -1.)
    r.runtime.hand.qpos_max_rad = np.full(12, 1.)
    trajectory = NS(num_frames=1, fps=16, arm_qpos=np.zeros((1, 7)),
                    hand_qpos=np.zeros((1, 12)), action_arm_joint=np.zeros((1, 7)),
                    action_hand_joint=np.zeros((1, 12)))
    with patches:
        def read(*args):
            row = teleop_module.read_observation(*args)
            if cancel: safety.revoke_motion(r.shared)
            return row
        teleop_module.time.sleep = lambda seconds: setattr(c, 'now', c.now+max(1, int(seconds*1e9)))
        patches.enter_context(patch.object(replayer, 'time', teleop_module.time))
        patches.enter_context(patch.object(replayer, 'read_observation', read))
        patches.enter_context(patch.object(replayer, 'make_arm_fk', lambda: NS(
            compute=lambda q: (np.zeros(3), np.array([1., 0, 0, 0, 1., 0])))))
        result = replayer.replay_targets(r.shared, r.runtime, trajectory,
            NS(healthy=True, poll=lambda **k: []), robot=robot, hand_start_duration_s=0)
    assert events.count('mode6') == 1 and events.count('arm_send') == int(not cancel)
    assert result.status == (replayer.ReplayStatus.REJECTED if cancel else replayer.ReplayStatus.COMPLETED)
    cases[f'replay_start_cancel_{cancel}'] = result.status.value

for cancel in (False, True):
    r, robot, c, events, rows, patches = setup()
    r.shared.safety_state = safety.SafetyState.ARMED
    robot._arm_stopped = True
    r.runtime.keyboard_teleop = NS(workspace_command_margin_m=.005)
    if cancel:
        def mode(*, check_abort):
            safety.revoke_motion(r.shared); check_abort()
        robot.arm.enter_mode6 = mode
    state = motion.CalibrationLoopState(samples=[], current_qpos=np.full(7, .1),
                                        feedback_timestamp_ns=BASE, frame=1)
    observed = []
    planner = NS(kin=NS(compute_eef_pose_world=lambda q: Pose(p=np.zeros(3), q=np.array([1., 0, 0, 0]))),
                 solve_online_ik=lambda pose, current, previous: observed.append(current.copy()) or
                 IKResult(success=True, qpos=np.zeros(7)))
    with patches:
        patches.enter_context(patch.object(motion, 'time', teleop_module.time))
        motion.run_calibration_motion_tick(r.shared, r.runtime, planner, np.array([[-1., 1.]]*3),
            NS(is_pressed=lambda key: key == 'w'), state,
            NS(delta_pos_m=.001, delta_rpy_rad=.01, status_interval_frames=20), robot)
    assert events.count('arm_send') == int(not cancel)
    if not cancel:
        assert np.array_equal(observed[0], np.zeros(7)) and state.feedback_timestamp_ns > BASE
    cases[f'calibration_start_cancel_{cancel}'] = 'no send' if cancel else 'IK uses post-mode feedback'

r, robot, c, events, rows, patches = setup(recording=True)
robot.arm.servo = lambda q: 42
def failed_stop(): raise RuntimeError('fake stop failure')
robot.arm.stop = failed_stop
with patches:
    try: r._run_control_tick()
    except DispatchError as exc: assert 'stop failures' in str(exc.__cause__)
    else: raise AssertionError('lost SDK/stop failure')
assert robot.stop_required and r.shared.error_state and len(rows) == 1 and r.active
cases['failed_stop'] = 'fault retained; capture attempted once; no normal idle reported'

# Neither controller acceptance nor SDK code 0 means physical arrival.
from dexmani_real.robot.drivers.xarm7 import XArm7
driver = XArm7.__new__(XArm7)
calls = []
driver.cfg = NS(max_joint_velocity_rad_per_s=1., max_joint_acceleration_rad_per_s2=2.)
driver._api = NS(set_servo_angle=lambda **kwargs: calls.append(kwargs) or 0)
assert driver.servo(np.zeros(7)) == 0 and len(calls) == 1 and calls[0]['wait'] is False
cases['arm_driver'] = 'one set_servo_angle(wait=False); no post-send read'

assert 'xarm' not in sys.modules and 'xhand_controller' not in sys.modules
files = ['robot/robot.py', 'robot/drivers/xarm7.py', 'robot/action.py', 'robot/projection.py',
         'teleop/control/controller.py', 'teleop/runner.py', 'teleop/keyboard_session.py',
         'deployment/runner.py', 'replay/replayer.py', 'calibration/camera/motion.py']
output = dict(cases=cases, hardware='fake I/O only; no SDK loaded', source_sha256={
    name: hashlib.sha256((ROOT/'dexmani_real'/name).read_bytes()).hexdigest() for name in files})
Path(__file__).with_suffix('.json').write_text(json.dumps(output, indent=2, ensure_ascii=False)+'\n')
print(f'{len(cases)} dispatch regressions passed; no hardware connected')

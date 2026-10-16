"""Parent-side xArm submission; one spawned process owns all xArm SDK calls.

Only the current goal, measured snapshot, authority and recording events cross the
boundary. Ordinary submission never waits for a servo tick or SDK reply.
"""
from dataclasses import dataclass
from importlib.metadata import version
import multiprocessing as mp
import time

import numpy as np

from dexmani_real.ipc.schema import ARM_STATE_DTYPE
from dexmani_real.recording.storage.servo import SERVO_EVENT_DTYPE, ServoEvidence
from dexmani_real.robot.command_validation import check_arm_target
from dexmani_real.runtime.safety import (
    SafetyState, RunEndReason, publish_arm_authority, revoke_motion,
)

_MAX_TIME = 2**63 - 1
_GOAL_DTYPE = np.dtype([
    ('sequence', '<i8'), ('run_id', '<i8'), ('capture_id', '<i8'),
    ('submit_until_ns', '<i8'), ('execute_until_ns', '<i8'), ('q_goal', '<f8', (7,)),
])
_STATUS_DTYPE = np.dtype([
    ('frame', ARM_STATE_DTYPE), ('valid', '?'), ('state', '<i4'), ('error_code', '<i4'),
    ('query_completed_ns', '<i8'), ('prepared_run_id', '<i8'),
    ('point', SERVO_EVENT_DTYPE), ('stats', '<i8', (4,)),
])
_STAT_NAMES = ('points', 'max_lateness_ns', 'max_call_ns', 'max_interval_ns')


@dataclass(frozen=True)
class ArmSnapshot:
    frame: np.ndarray
    valid: bool
    state: int
    error_code: int
    query_completed_ns: int


class _ArmChannels:
    """Fixed xArm channels created before spawn; no SDK or RuntimeState inside."""

    def __init__(self, ctx, capacity):
        self.authority = ctx.Value('q', 0)
        self.emergency = ctx.Event()
        self.closing, self.stop, self.stopped, self.ready = (ctx.Event() for _ in range(4))
        self.connected, self.cancel_operation = ctx.Event(), ctx.Event()
        self.fault = ctx.Array('c', 1024, lock=False)
        self.close_error = ctx.Array('c', 1024, lock=False)
        self.service_until = ctx.Value('q', 0)
        self.upstream_until = ctx.Value('q', 0)
        self.hand_until = ctx.Value('q', _MAX_TIME)
        self.execution_limit = ctx.Array('q', [_MAX_TIME, 0])  # deadline, last affected command
        self.goal_lock, self.status_lock = ctx.Lock(), ctx.Lock()
        self.goal = ctx.RawArray('b', _GOAL_DTYPE.itemsize)
        self.status = ctx.RawArray('b', _STATUS_DTYPE.itemsize)
        self.events = ctx.Queue(maxsize=capacity)
        self.evidence_failed = ctx.Event()

    def put_goal(self, goal):
        # A full slot is an explicit failure, never a silent target replacement.
        with self.goal_lock:
            slot = np.frombuffer(self.goal, dtype=_GOAL_DTYPE, count=1)
            if slot['sequence'][0]:
                raise RuntimeError('arm target slot is full')
            slot[0] = goal

    def take_goal(self):
        if not self.goal_lock.acquire(False):
            return None  # Never stall the servo on the parent holding this short lock.
        try:
            slot = np.frombuffer(self.goal, dtype=_GOAL_DTYPE, count=1)
            if not slot['sequence'][0]:
                return None
            goal = slot[0].copy()
            slot['sequence'][0] = 0
            return goal
        finally:
            self.goal_lock.release()

    def read_status(self):
        with self.status_lock:
            return np.frombuffer(self.status, dtype=_STATUS_DTYPE, count=1)[0].copy()

    def publish(self, status):
        # Both sides hold this lock only for a small snapshot copy, never SDK I/O.
        with self.status_lock:
            np.frombuffer(self.status, dtype=_STATUS_DTYPE, count=1)[0] = status


class ArmExecutor:
    def __init__(self, cfg, shared, *, driver_factory=None):
        cfg.validate()
        self.cfg, self.shared, self._factory = cfg, shared, driver_factory
        self._ctx = mp.get_context('spawn')
        self._io = _ArmChannels(self._ctx, cfg.servo_event_capacity)
        self._pipe, self._child_pipe = self._ctx.Pipe()
        self._process = None
        self._sequence = self._capture_id = self._operation_id = 0
        self._operation_unresolved = False
        self._evidence = None
        self._ended_evidence = None

    @property
    def is_connected(self):
        return self._process is not None and self._process.is_alive() and self._io.connected.is_set()

    @property
    def fault(self):
        return self._io.fault.value.decode(errors='replace') or None

    @property
    def stats(self):
        return dict(zip(_STAT_NAMES, map(int, self._io.read_status()['stats'])))

    @property
    def last_point(self):
        """Latest SDK attempt; completed_ns=0 explicitly means an unresolved call."""
        return self._io.read_status()['point']

    def connect(self):
        from ruckig import Ruckig
        from dexmani_real.robot.arm_worker import run_arm

        Ruckig(7)
        if self._process is not None:
            raise RuntimeError('arm executor is single-use')
        if not self.shared.running or self.shared.estop_request or self.shared.error_state:
            raise RuntimeError('arm startup unauthorized')
        self.shared.arm_authority, self.shared.arm_emergency = self._io.authority, self._io.emergency
        # Explicit connect authorizes initialization only, never streaming.
        self._io.authority.value = -int(self.shared.run_id)
        self._process = self._ctx.Process(
            target=run_arm, args=(self.cfg, self._io, self._child_pipe, self._factory),
            name='xarm-owner', daemon=False,
        )
        self._process.start()
        self._child_pipe.close()
        if not self._io.ready.wait(self.cfg.homing.convergence_timeout_s + self.cfg.executor_timeout_s):
            self.request_stop()
            self._io.closing.set()
            raise TimeoutError('arm initialization unresolved; owner process may still be in SDK')
        self.check_error()
        publish_arm_authority(self.shared)

    def check_error(self):
        issue = self.fault
        if not issue and self._process is not None and not self._process.is_alive():
            issue = f'arm owner exited: {self._process.exitcode}'
        if issue:
            self.shared.error_state = True
            revoke_motion(self.shared, SafetyState.FAULT, reason=RunEndReason.HARDWARE_FAULT)
            raise RuntimeError(f'xArm executor fault: {issue}')

    def snapshot(self):
        self.check_error()
        status = self._io.read_status()
        stamp = int(status['frame']['timestamp_ns'])
        if not status['valid'] or not 0 < stamp <= time.monotonic_ns() < stamp + int(self.cfg.feedback_max_age_s * 1e9):
            raise RuntimeError('arm feedback unavailable/stale')
        frame = status['frame'].copy().reshape(1)
        frame.flags.writeable = False
        return ArmSnapshot(frame, True, int(status['state']), int(status['error_code']),
                           int(status['query_completed_ns']))

    def update_health(self, *, service_until_ns=None, upstream_until_ns=None, hand_until_ns=None):
        for value, destination in ((service_until_ns, self._io.service_until),
                                   (upstream_until_ns, self._io.upstream_until),
                                   (hand_until_ns, self._io.hand_until)):
            if value is not None:
                destination.value = int(value)

    def limit_execution_until(self, deadline_ns):
        limit = self._io.execution_limit
        with limit.get_lock():
            limit[0] = min(limit[0], int(deadline_ns)) if limit[1] == self._sequence else int(deadline_ns)
            limit[1] = self._sequence

    def submit(self, command, *, submit_until_ns, execute_until_ns):
        self.check_error()
        issue = check_arm_target(command.arm_qpos,
                                 joint_limit_lower_rad=np.asarray(self.cfg.joint_limit_lower),
                                 joint_limit_upper_rad=np.asarray(self.cfg.joint_limit_upper))
        if issue:
            raise ValueError(issue)
        if (not self.is_connected or self._io.closing.is_set() or self._io.stop.is_set()
                or self._io.authority.value != command.run_id
                or self._io.read_status()['prepared_run_id'] != command.run_id
                or time.monotonic_ns() >= min(submit_until_ns, execute_until_ns)):
            raise RuntimeError('arm target unauthorized or expired')
        self._sequence += 1
        goal = np.zeros((), dtype=_GOAL_DTYPE)
        goal[()] = (self._sequence, command.run_id, self._capture_id if self._evidence else 0,
                    submit_until_ns, execute_until_ns, command.arm_qpos)
        self._io.put_goal(goal)
        return (int(command.run_id), self._sequence)

    def _operation(self, kind, payload=None, *, service=None, abort=None, timeout_s=None):
        if self._operation_unresolved and self._pipe.poll():
            reply_id, _ = self._pipe.recv()
            if reply_id != self._operation_id:
                raise RuntimeError('arm lifecycle reply mismatch')
            self._operation_unresolved = False
        if not self.is_connected or self._operation_unresolved:
            raise RuntimeError(f'arm {kind}: owner unavailable or previous operation unresolved')
        self._operation_id += 1
        identity = self._operation_id
        self._io.cancel_operation.clear()
        self._pipe.send((identity, kind, payload))
        deadline = time.monotonic() + (timeout_s or self.cfg.executor_timeout_s)
        service_due, received = 0, False
        try:
            while not self._pipe.poll(0.005):
                now = time.monotonic()
                if service is not None and now >= service_due:
                    service()  # XHand/input service remains exclusively in the parent.
                    service_due = now + self.cfg.homing.step_interval_s
                if abort is not None and abort():
                    from dexmani_real.robot.drivers.xarm7 import HomeAborted
                    raise HomeAborted(f'{kind} authority revoked')
                if now >= deadline or not self.is_connected:
                    raise TimeoutError(f'arm {kind} unresolved')
            reply_id, error = self._pipe.recv()
            received = True
            if reply_id != identity:
                raise RuntimeError('arm lifecycle reply mismatch')
            if error:
                if error.startswith('HomeAborted:'):
                    from dexmani_real.robot.drivers.xarm7 import HomeAborted
                    raise HomeAborted(error)
                raise RuntimeError(error)
        except BaseException:
            self._io.cancel_operation.set()
            self.request_stop()
            self._operation_unresolved = not received
            raise

    def prepare(self, epoch, *, service=None, abort=None):
        self.check_error()
        publish_arm_authority(self.shared)
        self._operation('prepare', epoch, service=service, abort=abort,
                        timeout_s=self.cfg.homing.convergence_timeout_s + 2)

    def home(self, waypoints, target, epoch, *, service, abort):
        self.check_error()
        publish_arm_authority(self.shared)
        self._operation('home', (waypoints, target, epoch), service=service, abort=abort, timeout_s=3600)

    def request_stop(self):
        self._io.authority.value = 0
        if self.shared.estop_request:
            self._io.emergency.set()
        self._io.stopped.clear()
        self._io.stop.set()

    def stop(self):
        self.request_stop()
        self.wait_stopped()

    def wait_stopped(self):
        deadline = time.monotonic() + self.cfg.executor_timeout_s
        if self._process is not None and self._process.is_alive():
            if not self._io.stopped.wait(self.cfg.executor_timeout_s):
                raise TimeoutError('arm stop unresolved; SDK acceptance/physical stop not confirmed')
            if self._operation_unresolved:
                if not self._pipe.poll(max(0, deadline - time.monotonic())):
                    raise TimeoutError('arm stopped but cancelled lifecycle operation remains unresolved')
                reply_id, _ = self._pipe.recv()
                if reply_id != self._operation_id:
                    raise RuntimeError('arm lifecycle reply mismatch')
                self._operation_unresolved = False
        elif self._process is not None and self._process.exitcode not in (None, 0):
            raise RuntimeError(f'arm owner exited without confirming stop: {self._process.exitcode}')
        if self._io.close_error.value:
            raise RuntimeError(self._io.close_error.value.decode(errors='replace'))

    def begin_evidence(self, target_dt):
        if self._evidence is not None:
            raise RuntimeError('previous arm capture has not ended')
        if self._ended_evidence is not None and (
                not self._ended_evidence.closed or self._ended_evidence.error is not None):
            raise RuntimeError('previous arm capture has not been completely drained')
        self._capture_id += 1
        self._io.evidence_failed.clear()
        metadata = dict(
            mode=1, implementation='ruckig_monotonic_process_v2', ruckig_version=version('ruckig'),
            sdk_version=version('xarm-python-sdk'), servo_hz=self.cfg.servo_hz, target_dt=target_dt,
            max_velocity_rad_s=self.cfg.max_joint_velocity_rad_per_s,
            max_acceleration_rad_s2=self.cfg.max_joint_acceleration_rad_per_s2,
            servo_max_interval_s=self.cfg.servo_max_interval_s, sdk_timeout_s=self.cfg.sdk_timeout_s,
            clock='host_monotonic_ns', receipt='submitted_sdk_results_in_servo_events',
            action='q_goal_absolute_joint_position_not_next_step_arrival',
            feedback='synchronous_query_start_host_monotonic; effort SDK-native',
            event_time='timestamp_ns=emission; planned_ns=tick; sampled_ns=reference; started/completed_ns=SDK',
            trajectory_time='actual_sample_ns_minus_activation_ns; no catch-up sends',
        )
        evidence = ServoEvidence(self.cfg.servo_event_capacity, metadata,
                                 queue=self._io.events, failure=self._io.evidence_failed)
        self._evidence = evidence
        try:
            self._operation('attach', self._capture_id)
        except BaseException as exc:
            evidence.fail(f'capture start unresolved: {exc}')
            raise
        return evidence

    def end_evidence(self, evidence):
        if evidence is not self._evidence:
            raise ValueError('arm capture identity mismatch')
        try:
            self._operation('detach', self._capture_id)
            evidence.producer_done = True
            self._ended_evidence = evidence
            self._evidence = None
        except BaseException as exc:
            evidence.fail(f'capture end unresolved: {exc}; last SDK attempt={self.last_point}')
            raise

    def close(self):
        self.request_stop()
        self._io.closing.set()
        if self._process is not None and self._process.pid is not None:
            self._process.join(self.cfg.executor_timeout_s)
            if self._process.is_alive():
                if self._evidence is not None:
                    self._evidence.fail(f'arm close unresolved; last SDK attempt={self.last_point}')
                raise TimeoutError('arm owner process still active; stop/disconnect unresolved')
            if self._process.exitcode != 0:
                raise RuntimeError(f'arm owner abnormal exit: {self._process.exitcode}')
        if self._evidence is not None and not self._evidence.closed:
            self._evidence.fail('arm closed before capture finalization')
        if self._ended_evidence is not None and not self._ended_evidence.closed:
            self._ended_evidence.fail('arm closed before capture events were drained')
        self._pipe.close()
        self._child_pipe.close()
        self._io.events.close()
        if self._io.close_error.value:
            raise RuntimeError(self._io.close_error.value.decode(errors='replace'))

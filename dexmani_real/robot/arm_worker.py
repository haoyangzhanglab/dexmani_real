"""xArm process: SDK lifetime, measured feedback and causal Mode-1 references.

No IK, XHand calls, callbacks into the parent, or file I/O occur here. This process
is not hard real time, and cannot interrupt its own blocked SDK call.
"""
import time

import numpy as np

from dexmani_real.robot.arm_executor import _MAX_TIME, _STATUS_DTYPE
from dexmani_real.robot.arm_reference import ArmReference
from dexmani_real.robot.commands import DispatchStatus
from dexmani_real.recording.storage.servo import ServoEvent, ServoEvidence
from dexmani_real.utils.control_clock import next_tick_ns


class _ArmWorker:
    def __init__(self, cfg, io, pipe, driver, *, clock_ns=time.monotonic_ns):
        self.cfg, self.io, self.pipe, self.driver = cfg, io, pipe, driver
        self.clock = clock_ns
        self.status = np.zeros((), dtype=_STATUS_DTYPE)
        self.active = self.reference = self.evidence = None
        self.capture_id = self.point_sequence = self.last_send_ns = 0

    def authorized(self, run_id, *, preparing=False):
        expected = -run_id if preparing else run_id
        return (self.io.authority.value == expected and not self.io.fault.value
                and not self.io.closing.is_set() and not self.io.stop.is_set()
                and not self.io.emergency.is_set())

    def emit(self, kind, goal=None, **fields):
        if self.evidence is not None:
            if goal is not None:
                if goal['capture_id'] != self.capture_id:
                    return
                fields.update(run_id=goal['run_id'], command_seq=goal['sequence'])
            self.evidence.emit(kind, self.clock(), **fields)

    def fault(self, reason):
        if not self.io.fault.value:
            self.io.fault.value = str(reason).encode()[:1023]
            try:
                self.emit(ServoEvent.FAULT, self.active, reason=reason)
            except Exception as exc:
                self.evidence.fail(exc)
        self.io.authority.value = 0
        self.io.stop.set()

    def read_feedback(self):
        started = self.clock()
        try:
            q, v, effort = self.driver.read()
            if any(np.asarray(x).shape != (7,) for x in (q, v, effort)) or not np.isfinite([q, v]).all():
                raise RuntimeError('invalid arm feedback shape/values')
            status = self.driver.read_status()
            completed = self.clock()
            if completed - started >= int(self.cfg.feedback_max_age_s * 1e9):
                raise TimeoutError('arm query exceeded feedback freshness budget')
            if not self.driver.is_connected or status.error_code:
                raise RuntimeError(f'arm disconnected/controller error {status.error_code}')
            self.status['frame']['qpos'] = q
            self.status['frame']['qvel'] = v
            self.status['frame']['effort'] = effort
            # Host query start is conservative; never replace this with cache-read time.
            self.status['frame']['timestamp_ns'] = started
            self.status['valid'] = True
            self.status['state'], self.status['error_code'] = status.state, status.error_code
            self.status['query_completed_ns'] = completed
        except Exception:
            self.status['valid'] = False
            raise
        finally:
            self.io.publish(self.status)

    def stationary_reference(self, check_abort):
        deadline = self.clock() + int(self.cfg.homing.convergence_timeout_s * 1e9)
        stable = anchor = None
        while self.clock() < deadline:
            check_abort()
            self.read_feedback()
            q, v = self.status['frame']['qpos'], self.status['frame']['qvel']
            now = self.clock()
            if np.max(np.abs(v)) <= self.cfg.homing.velocity_convergence_rad_s:
                if anchor is None or np.max(np.abs(q - anchor)) > self.cfg.homing.convergence_rad:
                    stable, anchor = None, q.copy()
                stable = now if stable is None else stable
                if now - stable >= int(self.cfg.homing.dwell_s * 1e9):
                    return ArmReference(self.cfg, q, v, now)
            else:
                stable = anchor = None
            self.io.stop.wait(1 / self.cfg.feedback_hz)
        raise TimeoutError('fresh measured arm standstill not confirmed')

    def end_active(self, reason):
        if self.active is not None:
            goal, self.active = self.active, None
            self.emit(ServoEvent.END, goal, reason=reason)

    def cancel_goal(self, goal, reason):
        if goal is not None:
            self.emit(ServoEvent.CANCEL, goal, status=DispatchStatus.NOT_CALLED,
                      q_goal=goal['q_goal'], reason=reason)

    def stop(self):
        self.status['prepared_run_id'] = 0
        self.reference = None
        try:
            self.end_active('fault' if self.io.fault.value else 'revoked')
            self.cancel_goal(self.io.take_goal(), 'revoked_before_sdk')
        except Exception as exc:
            if self.evidence is not None:
                self.evidence.fail(exc)
        try:
            if self.io.emergency.is_set():
                self.driver.emergency_stop()
            else:
                self.driver.stop()
        except Exception as exc:
            self.io.close_error.value = f'stop SDK failure: {exc}'.encode()[:1023]
            self.fault(exc)
        finally:
            self.io.stop.clear()
            self.io.stopped.set()  # SDK request processed; physical standstill is separate.
            self.io.publish(self.status)

    def operation(self, message):
        identity, kind, payload = message
        error = None
        try:
            if self.io.cancel_operation.is_set():
                raise RuntimeError(f'cancelled {kind}')
            if kind in ('prepare', 'home'):
                epoch = payload if kind == 'prepare' else payload[2]

                def check_abort():
                    if self.io.cancel_operation.is_set() or not self.authorized(epoch, preparing=True):
                        from dexmani_real.robot.drivers.xarm7 import HomeAborted
                        raise HomeAborted(f'{kind} authority revoked')

                check_abort()
                if self.active is not None:
                    raise RuntimeError('prepare/HOME requires stopped streaming')
                self.cancel_goal(self.io.take_goal(), 'lifecycle_boundary')
                self.status['prepared_run_id'] = 0
                self.io.stopped.clear()
                self.reference = self.stationary_reference(check_abort)
                if kind == 'home':
                    def abort_reason():
                        check_abort()
                        return None
                    self.driver.home(payload[0], payload[1], abort_check=abort_reason)
                else:
                    self.driver.enter_mode1(check_abort=check_abort)
                self.reference = self.stationary_reference(check_abort)
                check_abort()
                self.last_send_ns = 0
                self.io.execution_limit[:] = (_MAX_TIME, 0)
                self.status['prepared_run_id'] = epoch + 1
            elif kind == 'attach':
                if self.evidence is not None or self.active is not None:
                    raise RuntimeError('capture start requires quiescent arm')
                self.capture_id = payload
                self.evidence = ServoEvidence(self.cfg.servo_event_capacity, {},
                                              queue=self.io.events, failure=self.io.evidence_failed)
                self.emit(ServoEvent.BEGIN)
            elif kind == 'detach':
                if self.capture_id != payload or self.evidence is None or self.active is not None:
                    raise RuntimeError('capture end requires revoked, quiescent arm')
                self.cancel_goal(self.io.take_goal(), 'capture_end_before_sdk')
                self.emit(ServoEvent.FINISH)
                self.evidence = None
                self.capture_id = 0
            else:
                raise RuntimeError(f'unknown arm operation {kind}')
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            if kind in ('prepare', 'home') and not self.io.cancel_operation.is_set():
                from dexmani_real.robot.drivers.xarm7 import HomeAborted
                if not isinstance(exc, HomeAborted):
                    self.fault(exc)
        # Lifecycle replies include publication; never hold this lock over SDK I/O.
        with self.io.status_lock:
            np.frombuffer(self.io.status, dtype=_STATUS_DTYPE, count=1)[0] = self.status
        self.pipe.send((identity, error))

    def command_issue(self, goal, now, *, first=False):
        if not self.authorized(int(goal['run_id'])):
            return 'revoked'
        if first and now >= goal['submit_until_ns']:
            return 'submission_expired_before_sdk'
        if self.status['prepared_run_id'] != goal['run_id']:
            return 'unprepared_run'
        wait_until, wait_sequence = self.io.execution_limit[:]
        limit = wait_until if goal['sequence'] <= wait_sequence else _MAX_TIME
        if now >= min(goal['execute_until_ns'], limit):
            return 'input_decision_expired'
        if now >= min(self.io.service_until.value, self.io.upstream_until.value, self.io.hand_until.value):
            return 'upstream_feedback_or_service_expired'
        if (not self.status['valid'] or now - self.status['frame']['timestamp_ns']
                >= int(self.cfg.feedback_max_age_s * 1e9)):
            return 'arm_feedback_expired'
        if self.driver.mode != 1 or self.status['state'] not in (0, 1, 2):
            return 'controller_not_streaming'
        return None

    def tick(self, planned_ns):
        pending = self.io.take_goal()
        if self.active is None and pending is None:
            return
        now = self.clock()
        self.status['stats'][1] = max(self.status['stats'][1], now - planned_ns)
        if self.evidence is not None and self.io.evidence_failed.is_set():
            raise RuntimeError('servo evidence writer/buffer failed')
        goal = pending if pending is not None else self.active
        issue = self.command_issue(goal, now, first=pending is not None)
        if issue:
            self.cancel_goal(pending, issue)
            if issue != 'revoked':
                raise RuntimeError(issue)
            self.io.stop.set()
            return
        if pending is not None:
            if self.active is None:
                q, v = self.status['frame']['qpos'], self.status['frame']['qvel']
                if np.max(np.abs(v)) > self.cfg.homing.velocity_convergence_rad_s:
                    raise RuntimeError('arm moved after standstill preparation')
                self.reference = ArmReference(self.cfg, q, v, now)
            self.reference.replace(pending['q_goal'], now)
            self.end_active('superseded')
            self.active = pending
            self.emit(ServoEvent.ACTIVATE, pending, q_goal=pending['q_goal'], sampled_ns=now,
                      submit_until_ns=pending['submit_until_ns'], execute_until_ns=pending['execute_until_ns'])
        q, _, _ = self.reference.sample(now)
        started = self.clock()
        issue = self.command_issue(goal, started, first=pending is not None)
        if issue:
            if issue != 'revoked':
                raise RuntimeError(issue)
            self.io.stop.set()
            return
        if started - now > int(self.cfg.servo_max_interval_s * 1e9):
            raise TimeoutError('reference computation exceeded interval budget')
        if np.max(np.abs(self.status['frame']['qpos'] - q)) > self.cfg.tracking_error_rad:
            raise RuntimeError('tracking_error')
        point = self.status['point']
        point['run_id'], point['command_seq'] = goal['run_id'], goal['sequence']
        point['point_seq'] = self.point_sequence + 1
        point['planned_ns'], point['sampled_ns'], point['started_ns'] = planned_ns, now, started
        point['completed_ns'], point['status'], point['code'] = 0, DispatchStatus.UNKNOWN, -999
        point['q_ref'] = q
        self.io.publish(self.status)
        # Publication never authorizes a call: revocation can race the copy.
        started = self.clock()
        interval = started - self.last_send_ns if self.last_send_ns else 0
        issue = self.command_issue(goal, started, first=pending is not None)
        if issue is None and interval > int(self.cfg.servo_max_interval_s * 1e9):
            issue = 'servo interval exceeded; no catch-up sends'
        if issue is None and started - now >= int(self.cfg.servo_max_interval_s * 1e9):
            issue = 'servo dispatch stalled before SDK'
        if issue:
            point['status'], point['completed_ns'] = DispatchStatus.NOT_CALLED, self.clock()
            self.io.publish(self.status)
            if issue != 'revoked':
                raise RuntimeError(issue)
            self.io.stop.set()
            return
        # The public unresolved snapshot starts at the pre-call fence; the final
        # point carries the actual SDK call start and end, including any late result.
        point['started_ns'] = started
        code, error, result = None, '', DispatchStatus.UNKNOWN
        try:
            code = self.driver.servo(q)
            if isinstance(code, (int, np.integer)) and not isinstance(code, bool):
                code = int(code)
                result = (DispatchStatus.ACCEPTED if code == 0 else
                          DispatchStatus.UNKNOWN if code in (-1, -3, 3, 4, 5, 6, 7, 8, 10, 11)
                          else DispatchStatus.REJECTED)
            else:
                error, code = f'invalid SDK return: {code!r}', None
        except BaseException as exc:
            error = str(exc)
        completed = self.clock()
        self.last_send_ns = started
        self.point_sequence += 1
        self.status['stats'][0] += 1
        self.status['stats'][2] = max(self.status['stats'][2], completed - started)
        self.status['stats'][3] = max(self.status['stats'][3], interval)
        point['completed_ns'], point['status'], point['code'] = completed, result, -999 if code is None else code
        self.io.publish(self.status)
        self.emit(ServoEvent.POINT, goal, point_seq=self.point_sequence,
                  planned_ns=planned_ns, sampled_ns=now, started_ns=started, completed_ns=completed,
                  status=result, code=point['code'], q_ref=q, reason=error)
        if result != DispatchStatus.ACCEPTED:
            raise RuntimeError(error or f'servo SDK status={result.name} code={code}')
        if completed - started >= int(self.cfg.servo_max_interval_s * 1e9):
            raise TimeoutError('servo SDK blocked beyond interval budget')

    def run(self):
        try:
            self.driver.connect()
            self.io.connected.set()
            epoch = -self.io.authority.value

            def startup_abort():
                if not self.authorized(epoch, preparing=True):
                    raise RuntimeError('arm startup authority revoked')

            self.stationary_reference(startup_abort)
            self.driver.initialize(check_abort=startup_abort)
            self.read_feedback()
            self.io.ready.set()
            period = round(1e9 / self.cfg.servo_hz)
            tick = self.clock() + period
            feedback_due = self.clock() + round(1e9 / self.cfg.feedback_hz)
            while True:
                if self.io.stop.is_set():
                    self.stop()
                if self.io.closing.is_set():
                    break
                if self.pipe.poll():
                    self.operation(self.pipe.recv())
                    tick = self.clock() + period
                    feedback_due = self.clock()
                now = self.clock()
                if now >= tick:
                    try:
                        self.tick(tick)
                    except Exception as exc:
                        self.fault(exc)
                    tick = next_tick_ns(tick, period, self.clock())
                if self.active is not None and not self.authorized(int(self.active['run_id'])):
                    self.io.stop.set()
                if self.clock() >= feedback_due and not self.io.stop.is_set():
                    try:
                        self.read_feedback()
                    except Exception as exc:
                        self.fault(exc)
                    feedback_due = self.clock() + round(1e9 / self.cfg.feedback_hz)
                # A stop interrupts this wait; goals only activate on a scheduled tick.
                self.io.stop.wait(max(0, min(tick, feedback_due) - self.clock()) / 1e9)
        except BaseException as exc:
            self.fault(exc)
        finally:
            self.io.ready.set()
            if not self.io.stopped.is_set():
                self.stop()
            try:
                self.driver.close()
            except Exception as exc:
                self.io.close_error.value = f'disconnect failure: {exc}'.encode()[:1023]
            self.io.connected.clear()
            if self.evidence is not None:
                self.evidence.fail('arm exited before capture finalization')
            self.pipe.close()
            # If capture finalization failed, retain staging instead of deadlocking
            # process exit on an abandoned pipe. A completed stream was drained first.
            self.io.events.cancel_join_thread()
            self.io.events.close()


def run_arm(cfg, io, pipe, driver_factory=None):
    try:
        if driver_factory is None:
            from dexmani_real.robot.drivers.xarm7 import XArm7
            driver_factory = XArm7
        driver = driver_factory(cfg)
    except BaseException as exc:
        io.fault.value = f'driver construction failed: {exc}'.encode()[:1023]
        io.ready.set()
        pipe.close()
        return
    _ArmWorker(cfg, io, pipe, driver).run()

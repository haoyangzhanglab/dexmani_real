"""Own xArm SDK calls and causal references; a blocked native call cannot be interrupted here."""

import time

import numpy as np

from dexmani_real.robot.xarm7.protocol import (
    MAX_TIME,
    STATUS_DTYPE,
    SDK_CODE_UNAVAILABLE,
    PrepareRequest,
    HomeRequest,
    LifecycleReply,
    OperationStatus,
    OperationCancelled,
)
from dexmani_real.robot.xarm7.reference import ArmReference
from dexmani_real.robot.commands import DispatchStatus
from dexmani_real.utils.timing import next_tick_ns


class ArmWorker:
    def __init__(self, cfg, io, pipe, driver, *, clock_ns=time.monotonic_ns):
        self.cfg, self.io, self.pipe, self.driver = cfg, io, pipe, driver
        self.clock = clock_ns
        self.status = np.zeros((), dtype=STATUS_DTYPE)
        self.active = self.reference = None
        self.first_accepted = False
        self.last_send_ns = 0

    def authorized(self, run_id, *, preparing=False):
        expected = -run_id if preparing else run_id
        return (
            self.io.authority.value == expected
            and not self.io.fault.value
            and not self.io.closing.is_set()
            and not self.io.stop.is_set()
            and not self.io.emergency.is_set()
        )

    def fault(self, reason):
        if not self.io.fault.value:
            self.io.fault.value = str(reason).encode()[:1023]
        self.io.authority.value = 0
        self.io.stop.set()

    def read_feedback(self):
        started = self.clock()
        try:
            q, v, effort = self.driver.read()
            if (
                any(np.asarray(x).shape != (7,) for x in (q, v, effort))
                or not np.isfinite([q, v]).all()
            ):
                raise RuntimeError("invalid arm feedback shape/values")
            status = self.driver.read_status()
            completed = self.clock()
            if completed - started >= int(self.cfg.feedback_max_age_s * 1e9):
                raise TimeoutError("arm query exceeded feedback freshness budget")
            if not self.driver.is_connected or status.error_code:
                raise RuntimeError(f"arm disconnected/controller error {status.error_code}")
            self.status["frame"]["qpos"] = q
            self.status["frame"]["qvel"] = v
            self.status["frame"]["effort"] = effort
            # Host query start is conservative; never replace this with cache-read time.
            self.status["frame"]["timestamp_ns"] = started
            self.status["valid"] = True
            self.status["state"] = status.state
        except Exception:
            self.status["valid"] = False
            raise
        finally:
            self.io.publish(self.status)

    def wait_settled(
        self,
        check_abort,
        *,
        stage,
        timeout_s,
        interval_s,
        target=None,
        tolerance_rad=None,
        strict=False,
    ):
        """Require position/velocity dwell; strict HOME rejects any bad sample."""
        tolerance = self.cfg.homing.convergence_rad if tolerance_rad is None else tolerance_rad
        deadline = self.clock() + int(timeout_s * 1e9)
        stable = self.clock() if strict else None
        anchor = None
        while self.clock() < deadline:
            check_abort()
            if strict:
                self.io.stop.wait(min(interval_s, self.cfg.homing.dwell_s))
                check_abort()
            self.read_feedback()
            check_abort()
            q, v = self.status["frame"]["qpos"], self.status["frame"]["qvel"]
            now = self.clock()
            valid = np.max(np.abs(v)) <= self.cfg.homing.velocity_convergence_rad_s
            if target is not None:
                valid = valid and np.max(np.abs(q - target)) <= tolerance
            if strict and not valid:
                raise RuntimeError(f"{stage} interrupted by position/velocity")
            if valid:
                if target is None and (anchor is None or np.max(np.abs(q - anchor)) > tolerance):
                    stable, anchor = None, q.copy()
                stable = now if stable is None else stable
                if now - stable >= int(self.cfg.homing.dwell_s * 1e9):
                    return
            else:
                stable = anchor = None
            if not strict:
                self.io.stop.wait(interval_s)
        raise TimeoutError(f"{stage}: measured position/velocity dwell not confirmed")

    def wait_stationary(self, check_abort):
        self.wait_settled(
            check_abort,
            stage="arm standstill",
            timeout_s=self.cfg.homing.convergence_timeout_s,
            interval_s=1 / self.cfg.feedback_hz,
        )
        # Check braking extrema before Mode 1; seed the first causal goal from fresh feedback.
        ArmReference(
            self.cfg, self.status["frame"]["qpos"], self.status["frame"]["qvel"], self.clock()
        )

    def home(self, waypoints, final_qpos, *, check_abort):
        """Advance the admitted HOME milestones entirely inside the SDK owner."""
        waypoints = np.asarray(waypoints, dtype=np.float64)
        final_qpos = np.asarray(final_qpos, dtype=np.float64)
        lower, upper = np.asarray(self.cfg.joint_limit_lower), np.asarray(
            self.cfg.joint_limit_upper
        )
        if waypoints.ndim != 2 or waypoints.shape[1:] != (7,) or not np.isfinite(waypoints).all():
            raise ValueError("invalid home waypoint array")
        if final_qpos.shape != (7,) or not np.isfinite(final_qpos).all():
            raise ValueError("invalid home final_qpos")
        if np.any(final_qpos < lower) or np.any(final_qpos > upper):
            raise ValueError("home final target violates joint limits")
        if np.any(waypoints < lower) or np.any(waypoints > upper):
            raise ValueError("home waypoint violates joint limits")
        if len(waypoints) and np.max(np.abs(waypoints[-1] - final_qpos)) > 1e-6:
            raise ValueError("final milestone does not match canonical home")
        check_abort()
        self.read_feedback()
        check_abort()
        current = self.status["frame"]["qpos"].copy()
        velocity = self.status["frame"]["qvel"]
        if len(waypoints) == 0:
            if (
                np.max(np.abs(current - final_qpos)) > self.cfg.homing.convergence_rad
                or np.max(np.abs(velocity)) > self.cfg.homing.velocity_convergence_rad_s
            ):
                raise RuntimeError("empty home path while away from canonical home")
        elif np.max(np.abs(current - waypoints[0])) > self.cfg.homing.convergence_rad:
            raise RuntimeError("current state moved too far from planned path start")
        targets = waypoints[1:]
        if len(targets):
            check_abort()
            self.driver.enter_mode0(check_abort=check_abort)
            for index, target in enumerate(targets, 1):
                check_abort()
                self.driver.move_home_milestone(target)
                check_abort()
                nominal = float(np.max(np.abs(target - current))) / max(
                    self.cfg.homing.max_speed_rad_per_s, 1e-6
                )
                self.wait_settled(
                    check_abort,
                    stage=f"home milestone {index}",
                    timeout_s=2 * nominal + self.cfg.homing.target_timeout_s,
                    interval_s=self.cfg.homing.step_interval_s,
                    target=target,
                    tolerance_rad=min(self.cfg.homing.convergence_rad, np.deg2rad(0.5)),
                )
                current = self.status["frame"]["qpos"].copy()
            if np.max(np.abs(current - final_qpos)) > self.cfg.homing.convergence_rad:
                raise RuntimeError("home final target error")
        else:
            self.wait_settled(
                check_abort,
                stage="home dwell",
                strict=True,
                target=final_qpos,
                timeout_s=self.cfg.homing.dwell_s + self.cfg.homing.step_interval_s,
                interval_s=self.cfg.homing.step_interval_s,
            )
        check_abort()
        self.driver.enter_mode1(check_abort=check_abort)
        self.wait_settled(
            check_abort,
            stage="post-Mode1 home settle",
            target=final_qpos,
            timeout_s=self.cfg.homing.target_timeout_s
            + self.cfg.homing.dwell_s
            + self.cfg.homing.step_interval_s,
            interval_s=self.cfg.homing.step_interval_s,
        )

    def end_active(self):
        if self.active is not None:
            if not self.first_accepted:
                self.status["cancelled_before_first"] += 1
            self.active = None
            self.status["active_sequence"] = 0

    def cancel_goal(self, goal):
        if goal is not None:
            self.status["cancelled_before_first"] += 1

    def stop(self):
        self.io.authority.value = 0
        self.status["prepared_run_id"] = 0
        self.reference = None
        self.end_active()
        self.cancel_goal(self.io.take_goal(block=True))
        self.status["stop_started_ns"] = self.clock()
        self.status["stop_completed_ns"] = 0
        self.status["stop_accepted"] = False
        self.io.publish(self.status)
        try:
            code = self.driver.stop()
            if isinstance(code, bool) or not isinstance(code, (int, np.integer)) or code != 0:
                raise RuntimeError(f"stop acceptance unavailable/rejected: {code!r}")
            self.status["stop_accepted"] = True
        except BaseException as exc:
            if not self.io.close_error.value:
                self.io.close_error.value = f"stop SDK failure: {exc}".encode()[:1023]
            self.fault(exc)
        finally:
            self.status["stop_completed_ns"] = self.clock()
            self.io.stop.clear()
            self.io.publish(self.status)
            self.io.stopped.set()  # Processing finished; stop_accepted records the independent SDK result.

    def operation(self, message):
        identity = message.operation_id
        kind = "home" if isinstance(message, HomeRequest) else "prepare"
        detail = ""
        result = OperationStatus.COMPLETED
        try:
            if self.io.cancel_operation.is_set():
                raise OperationCancelled(f"cancelled {kind}")
            if not isinstance(message, (PrepareRequest, HomeRequest)):
                raise RuntimeError(f"unknown arm operation: {type(message).__name__}")
            epoch = message.epoch

            def check_abort():
                if self.io.cancel_operation.is_set() or not self.authorized(epoch, preparing=True):
                    raise OperationCancelled(f"{kind} authority revoked")
                # HOME uses parent-service and hand-feedback deadlines, separate from streaming.
                if self.clock() >= min(self.io.service_until.value, self.io.hand_until.value):
                    raise TimeoutError(f"{kind} hand feedback or parent service expired")

            check_abort()
            if self.active is not None:
                raise RuntimeError("prepare/HOME requires stopped streaming")
            self.cancel_goal(self.io.take_goal(block=True))
            self.status["prepared_run_id"] = 0
            self.io.stopped.clear()
            self.wait_stationary(check_abort)
            if isinstance(message, HomeRequest):
                self.home(message.waypoints, message.target, check_abort=check_abort)
            else:
                self.driver.enter_mode1(check_abort=check_abort)
            self.wait_stationary(check_abort)
            check_abort()
            self.last_send_ns = 0
            self.io.execution_limit[:] = (MAX_TIME, 0)
            # RuntimeState.begin_motion advances the prepared ARMED epoch once.
            self.status["prepared_run_id"] = epoch + 1
        except BaseException as exc:
            detail = f"{type(exc).__name__}: {exc}"
            result = (
                OperationStatus.CANCELLED
                if isinstance(exc, OperationCancelled)
                else OperationStatus.FAILED
            )
            if not isinstance(exc, OperationCancelled):
                self.fault(exc)
            # Do not wait for a parent blocked in another device's SDK.
            self.stop()
            if self.io.close_error.value:
                detail += "; " + self.io.close_error.value.decode(errors="replace")
                result = OperationStatus.FAILED
        # Lifecycle replies include publication; never hold this lock over SDK I/O.
        self.io.publish(self.status)
        self.pipe.send(LifecycleReply(identity, result, detail))

    def command_issue(self, goal, now, *, first=False):
        if not self.authorized(int(goal["run_id"])):
            return "revoked"
        if first and now >= goal["submit_until_ns"]:
            return "submission_expired_before_sdk"
        if self.status["prepared_run_id"] != goal["run_id"]:
            return "unprepared_run"
        wait_until, wait_sequence = self.io.execution_limit[:]
        limit = wait_until if goal["sequence"] <= wait_sequence else MAX_TIME
        if now >= min(goal["execute_until_ns"], limit):
            return "input_decision_expired"
        if now >= min(
            self.io.service_until.value, self.io.upstream_until.value, self.io.hand_until.value
        ):
            return "upstream_feedback_or_service_expired"
        if not self.status["valid"] or now - self.status["frame"]["timestamp_ns"] >= int(
            self.cfg.feedback_max_age_s * 1e9
        ):
            return "arm_feedback_expired"
        if self.driver.mode != 1 or self.status["state"] not in (0, 1, 2):
            return "controller_not_streaming"
        return None

    def tick(self, planned_ns):
        pending = self.io.take_goal()
        if self.active is None and pending is None:
            return
        now = self.clock()
        self.status["stats"]["max_lateness_ns"] = max(
            self.status["stats"]["max_lateness_ns"], now - planned_ns
        )
        goal = pending if pending is not None else self.active
        issue = self.command_issue(goal, now, first=pending is not None)
        if issue:
            self.cancel_goal(pending)
            if issue != "revoked":
                raise RuntimeError(issue)
            self.io.stop.set()
            return
        if pending is not None:
            try:
                if self.active is None:
                    q, v = self.status["frame"]["qpos"], self.status["frame"]["qvel"]
                    if np.max(np.abs(v)) > self.cfg.homing.velocity_convergence_rad_s:
                        raise RuntimeError("arm moved after standstill preparation")
                    self.reference = ArmReference(self.cfg, q, v, now)
                self.reference.replace(pending["q_goal"], now)
            except Exception:
                self.cancel_goal(pending)
                raise
            self.end_active()
            self.active = pending
            self.first_accepted = False
            self.status["active_sequence"] = pending["sequence"]
        q, _, _ = self.reference.sample(now)
        started = self.clock()
        issue = self.command_issue(goal, started, first=pending is not None)
        if issue:
            if issue != "revoked":
                raise RuntimeError(issue)
            self.io.stop.set()
            return
        if started - now > int(self.cfg.servo_max_interval_s * 1e9):
            raise TimeoutError("reference computation exceeded interval budget")
        if np.max(np.abs(self.status["frame"]["qpos"] - q)) > self.cfg.tracking_error_rad:
            raise RuntimeError("tracking_error")
        point = self.status["point"]
        point["run_id"], point["command_seq"] = goal["run_id"], goal["sequence"]
        point["planned_ns"], point["sampled_ns"], point["started_ns"] = planned_ns, now, started
        point["completed_ns"], point["status"], point["code"] = (
            0,
            DispatchStatus.UNKNOWN,
            SDK_CODE_UNAVAILABLE,
        )
        point["q_ref"] = q
        self.status["sdk_attempted"] += 1
        self.io.publish(self.status)
        # Publication never authorizes a call: revocation can race the copy.
        started = self.clock()
        interval = started - self.last_send_ns if self.last_send_ns else 0
        issue = self.command_issue(goal, started, first=pending is not None)
        if issue is None and interval > int(self.cfg.servo_max_interval_s * 1e9):
            issue = "servo interval exceeded; no catch-up sends"
        if issue is None and started - now >= int(self.cfg.servo_max_interval_s * 1e9):
            issue = "servo dispatch stalled before SDK"
        if issue:
            # The pre-call fence prevented this attempt from reaching the SDK.
            self.status["sdk_attempted"] -= 1
            point["status"], point["completed_ns"] = DispatchStatus.NOT_CALLED, self.clock()
            self.io.publish(self.status)
            if issue != "revoked":
                raise RuntimeError(issue)
            self.io.stop.set()
            return
        # Fence unresolved dispatch before the SDK call; finalized points retain actual start/end times.
        point["started_ns"] = started
        code, error, result = None, "", DispatchStatus.UNKNOWN
        try:
            code = self.driver.servo(q)
            if isinstance(code, (int, np.integer)) and not isinstance(code, bool):
                code = int(code)
                result = (
                    DispatchStatus.ACCEPTED
                    if code == 0
                    else (
                        DispatchStatus.UNKNOWN
                        if code in (-1, -3, 3, 4, 5, 6, 7, 8, 10, 11)
                        else DispatchStatus.REJECTED
                    )
                )
            else:
                error, code = f"invalid SDK return: {code!r}", None
        except BaseException as exc:
            error = str(exc)
        completed = self.clock()
        self.last_send_ns = started
        self.status["sdk_completed"] += 1
        if result == DispatchStatus.ACCEPTED:
            self.status["sdk_accepted"] += 1
            if not self.first_accepted:
                self.first_accepted = True
                self.status["first_accepted"] += 1
        self.status["stats"]["max_call_ns"] = max(
            self.status["stats"]["max_call_ns"], completed - started
        )
        self.status["stats"]["max_interval_ns"] = max(
            self.status["stats"]["max_interval_ns"], interval
        )
        point["completed_ns"], point["status"], point["code"] = (
            completed,
            result,
            SDK_CODE_UNAVAILABLE if code is None else code,
        )
        self.io.publish(self.status)
        if result != DispatchStatus.ACCEPTED:
            raise RuntimeError(error or f"servo SDK status={result.name} code={code}")
        if completed - started >= int(self.cfg.servo_max_interval_s * 1e9):
            raise TimeoutError("servo SDK blocked beyond interval budget")

    def run(self):
        try:
            self.driver.connect()
            self.io.connected.set()
            epoch = -self.io.authority.value

            def startup_abort():
                if not self.authorized(epoch, preparing=True):
                    raise RuntimeError("arm startup authority revoked")

            self.wait_stationary(startup_abort)
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
                if self.active is not None and not self.authorized(int(self.active["run_id"])):
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
                previous = self.io.close_error.value.decode(errors="replace")
                detail = f"disconnect failure: {exc}"
                self.io.close_error.value = (
                    f"{previous}; {detail}" if previous else detail
                ).encode()[:1023]
            self.io.connected.clear()
            self.pipe.close()


def run_arm(cfg, io, pipe, driver_factory=None):
    try:
        if driver_factory is None:
            from dexmani_real.robot.xarm7.driver import XArm7

            driver_factory = XArm7
        driver = driver_factory(cfg)
    except BaseException as exc:
        io.fault.value = f"driver construction failed: {exc}".encode()[:1023]
        io.ready.set()
        pipe.close()
        return
    ArmWorker(cfg, io, pipe, driver).run()

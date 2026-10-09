"""Operator-supervised teleop with one current observation per real control step."""

import logging
import time
from concurrent.futures import CancelledError

from dexmani_real.recording.recorder import RecordingError, snapshot_recording_metadata
from dexmani_real.robot.arm_homing import home_robot
from dexmani_real.robot.hand_homing import home_hand
from dexmani_real.robot.robot import DispatchError
from dexmani_real.runtime.observation import read_observation
from dexmani_real.runtime.operator_input import KeyboardInput, OperatorCommand
from dexmani_real.runtime.safety import RunEndReason, SafetyState, begin_motion, revoke_motion
from dexmani_real.teleop.audio_feedback import AudioFeedback
from dexmani_real.teleop.control.controller import execute_control_step
from dexmani_real.utils.control_clock import sampling_clock_ns


logger = logging.getLogger(__name__)
_DEBUG_FAILURE_LIMIT = 10
_END_AUDIO_GRACE_S = 3.0


class TeleopRunner:
    """Own teleop control and capture lifecycle with the session's optional recorder."""

    def __init__(
        self,
        shared,
        runtime,
        robot,
        *,
        task_label,
        controller,
        home_planner,
        start_vr,
        recorder,
        camera_calibration=None,
    ):
        self.shared = shared
        self.robot = robot
        self.start_vr = start_vr
        self.camera_calibration = camera_calibration
        self.task_label = task_label
        self.runtime = runtime
        self.recorder = recorder
        self.keyboard = KeyboardInput(
            estop_callback=lambda: setattr(shared, "estop_request", True),
            stop_callback=lambda: revoke_motion(shared, reason=RunEndReason.OPERATOR),
            quit_callback=lambda: revoke_motion(shared, reason=RunEndReason.QUIT),
        )
        self.audio = AudioFeedback()
        self.controller = controller
        self.active = False
        self.paused = False
        self.resume_requested = False
        self.preparing_epoch = None
        self.resume_after_ns = 0
        self.quit_pending = False
        self.pending_termination_reason = None
        self.termination_details = []
        self.control_run_id = None
        self.next_tick_ns = 0
        self.tick_origin_ns = None
        self.tick_index = 0
        self.control_dt_ns, self.sample_tolerance_ns = sampling_clock_ns(
            self.runtime.teleop.control_hz
        )
        self.failures = 0
        self.capture_started_s = 0.0

        self.home_planner = home_planner

    def _end_reason(self, fallback):
        with self.shared.motion_lock:
            if (
                self.control_run_id is not None
                and int(self.shared.run_ended_id) == self.control_run_id
            ):
                return RunEndReason(int(self.shared.run_ended_reason)).name.lower()
        return fallback

    def _stop_motion(self, stage):
        try:
            self.robot.stop()
        except Exception as exc:
            self.termination_details.append(
                dict(stage=stage, exception_type=type(exc).__name__, message=str(exc))
            )
            raise

    def _pause_control(self, reason, *, run_end_reason=RunEndReason.OPERATOR, stop_motion=True):
        revoke_motion(self.shared, reason=run_end_reason)
        if self.pending_termination_reason is None:
            self.pending_termination_reason = self._end_reason(reason)
        self.termination_details.append(dict(stage="termination", reason=reason))
        if stop_motion:
            self._stop_motion("pause_stop")
        self.controller.clear_reference()
        self.paused, self.resume_requested = True, False
        self.preparing_epoch = None

    def _end_segment(self, save, reason, *, announce=True):
        """Called only after revocation/stop, including sessions without a recorder."""
        self._finish_capture(save, reason, announce=announce)
        self.active = self.paused = self.resume_requested = self.quit_pending = False
        self.preparing_epoch = self.control_run_id = None

    def _finish_capture(self, save, reason, *, announce=True) -> None:
        if self.recorder is None or not self.recorder.is_recording:
            self.pending_termination_reason = None
            self.termination_details = []
            return
        reason = self._end_reason(self.pending_termination_reason or reason)
        published = None
        if save:
            published = self.recorder.save_episode(reason=reason, details=self.termination_details)
        else:
            self.recorder.discard_episode(reason=reason)
        self.pending_termination_reason = None
        self.termination_details = []
        if announce:
            self.audio.play("save" if published is not None else "discard")

    def _stop_capture(self, reason, *, stop_motion=True, row=None):
        failure = None
        try:
            self._pause_control(
                reason, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, stop_motion=stop_motion
            )
        except Exception as exc:
            failure = exc
        logger.warning("Teleop ended: %s", reason)
        self.audio.play("emergency")
        try:
            if row is not None and self.recorder is not None:
                self.recorder.add_frame(row)
        except Exception as exc:
            failure = failure or exc
            self.termination_details.append(dict(
                stage="recording", exception_type=type(exc).__name__, message=str(exc),
            ))
        try:
            if failure is None:
                self._end_segment(True, reason, announce=False)
            else:
                # Keep the fault visible; never advertise normal IDLE after stop failure.
                self._finish_capture(True, reason, announce=False)
        except Exception as exc:
            failure = failure or exc
        if failure is not None:
            raise failure

    def _poll_blocking_commands(self, commands=()):
        self.robot.check()
        aborted = False
        # Other keys are consumed, including the tail of the poll that contained H.
        # Q revokes HOME authority before any blocking capture finalization.
        for cmd in (*commands, *self.keyboard.poll(timeout=0)):
            if cmd in (OperatorCommand.EMERGENCY_STOP, OperatorCommand.QUIT,
                       OperatorCommand.STOP, OperatorCommand.DISCARD):
                self._handle_operator_command(cmd)
                aborted = True
        return bool(
            aborted
            or self.shared.estop_request
            or self.shared.error_state
            or self.shared.quit_requested
            or not self.shared.sensors.is_running.value
        )

    def _preparation_valid(self):
        return (
            self.preparing_epoch is not None
            and self.preparing_epoch == int(self.shared.run_id)
            and not self.shared.estop_request
            and not self.shared.error_state
            and not self.shared.quit_requested
            and self.shared.sensors.is_running.value
            and int(self.shared.safety_state) == int(SafetyState.ARMED)
        )

    def _start_segment(self, expected_run_id):
        if self.resume_requested:
            return
        with self.shared.motion_lock:
            # Bind B/C to the generation from before poll, never to a later STOP.
            if expected_run_id != int(self.shared.run_id):
                return
            revoke_motion(self.shared)
            self.preparing_epoch = int(self.shared.run_id)
        self.resume_requested = True
        self._stop_motion("segment_stop")
        self.paused = True
        if self.recorder is not None and self.recorder.is_recording:
            self._finish_capture(True, "pause")
        if not self._preparation_valid():
            self._stop_capture("preparation_revoked")
            return
        row = read_observation(
            self.shared,
            self.runtime,
            self.robot,
            require_hand=self.runtime.policy.hand_enabled,
            require_camera=self.recorder is not None,
            require_vr=True,
        )
        if row is None or not self._preparation_valid():
            print("Start requires fresh robot, VR and recording resources", flush=True)
            self._stop_capture("preparation_unavailable")
            return
        self.control_run_id = None
        if self.recorder is not None:
            self.recorder.start_episode(
                task_label=self.task_label,
                **snapshot_recording_metadata(
                    self.shared,
                    collection_source="teleop",
                    camera_calibration=self.camera_calibration,
                ),
            )
        self.capture_started_s = time.monotonic()
        self.pending_termination_reason = None
        self.termination_details = []
        # Disk finalization/START may block. Never anchor to observations from
        # before that boundary or append across the operator's wall-clock pause.
        self.resume_after_ns = time.monotonic_ns()
        if not self._preparation_valid():
            self._stop_capture("preparation_revoked")

    def _handle_operator_command(self, cmd, *, home_commands=(), expected_run_id=None) -> bool:
        if cmd is OperatorCommand.EMERGENCY_STOP:
            self.shared.estop_request = True
            revoke_motion(self.shared, reason=RunEndReason.ESTOP)
            self.audio.play("emergency")
            return True
        if (
            self.shared.estop_request
            or self.shared.error_state
            or self.shared.quit_requested
            or not self.shared.sensors.is_running.value
        ):
            return True
        if cmd is OperatorCommand.QUIT:
            was_idle = not self.active and not self.resume_requested and not self.quit_pending
            self._pause_control("quit", run_end_reason=RunEndReason.QUIT)
            has_capture = self.recorder is not None and self.recorder.is_recording
            if self.quit_pending or (was_idle and not has_capture):
                if has_capture:
                    self._end_segment(True, "quit", announce=False)
                self.audio.play("end")
                self.shared.quit_requested = True
                return True
            self.quit_pending = True
            self.audio.play("quit")
        elif cmd in (OperatorCommand.STOP, OperatorCommand.DISCARD):
            reason = cmd.value.lower()
            self._pause_control(reason)
            self._end_segment(cmd is OperatorCommand.STOP, reason)
        elif cmd is OperatorCommand.TARE:
            if (
                self.resume_requested
                or (self.active and not self.paused)
                or (self.recorder is not None and self.recorder.is_recording)
                or int(self.shared.safety_state) != int(SafetyState.ARMED)
            ):
                logger.warning("T ignored: tactile tare requires idle, no capture or preparation")
                return False
            try:
                result = self.robot.tare_tactile(cancel_requested=self._poll_blocking_commands)
                logger.info("Explicit tactile baseline aggregate/dense: %s", result)
            except CancelledError:
                logger.info("Tactile tare cancelled; no new baseline published")
            finally:
                self.keyboard.drain_signal(OperatorCommand.TARE)
                self.keyboard.drain_signal(OperatorCommand.BEGIN)
            return True
        elif cmd is OperatorCommand.HOME:
            self._pause_control("home")
            self._end_segment(True, "home")
            if not self._poll_blocking_commands(home_commands):
                self.audio.play("home")
                result = home_robot(
                    self.shared,
                    self.runtime,
                    self.home_planner,
                    robot=self.robot,
                    abort_requested=self._poll_blocking_commands,
                )
                if result.ok:
                    self.audio.queue("home_done")
                elif result.interrupted:
                    logger.info("HOME cancelled: %s", result.reason)
                else:
                    raise RuntimeError(f"HOME failed: {result.reason}")
            self._poll_blocking_commands()
            return True
        elif cmd is OperatorCommand.PAUSE and (self.active or self.quit_pending):
            if self.resume_requested:
                return False
            self.quit_pending = False
            if self.paused:
                self._start_segment(expected_run_id)
            else:
                self._pause_control("pause")
                self.audio.play("pause")
        elif cmd is OperatorCommand.BEGIN and not self.active and not self.quit_pending:
            self._start_segment(expected_run_id)
        return False

    def _handle_commands(self, commands, expected_run_id):
        terminal = {OperatorCommand.STOP, OperatorCommand.DISCARD,
                    OperatorCommand.QUIT, OperatorCommand.EMERGENCY_STOP}
        if terminal.intersection(commands):
            commands = [c for c in commands if c in terminal]
            if OperatorCommand.STOP in commands:
                commands = [c for c in commands if c is not OperatorCommand.DISCARD]
            if OperatorCommand.EMERGENCY_STOP in commands:
                commands = [OperatorCommand.EMERGENCY_STOP]
        for index, cmd in enumerate(commands):
            if self._handle_operator_command(
                cmd, expected_run_id=expected_run_id,
                home_commands=commands[index + 1:] if cmd is OperatorCommand.HOME else (),
            ):
                break

    def _try_resume(self, row) -> bool:
        if self.resume_requested and not self._preparation_valid():
            self._stop_capture("preparation_revoked")
            return False
        if self.resume_requested and all(
            int(stamp) > self.resume_after_ns
            for stamp in (
                row.arm["timestamp_ns"][0],
                row.vr["recv_ts_ns"],
                row.hand["timestamp_ns"][0]
                if row.hand is not None
                else row.observation_timestamp_ns,
            )
        ):
            prepared = self.controller.reset_reference(row)
            with self.shared.motion_lock:
                authorized = (
                    prepared
                    and self._preparation_valid()
                    and begin_motion(self.shared)
                )
                if authorized:
                    self.control_run_id = int(self.shared.run_id)
            if authorized:
                event = "resume" if self.active else "begin"
                self.active = True
                self.paused = self.resume_requested = False
                self.preparing_epoch = None
                self.failures = 0
                self.tick_origin_ns = None
                self.tick_index = 0
                self.next_tick_ns = 0
                # C may have just saved the previous segment; do not cut off
                # that result announcement when fresh observations arrive.
                self.audio.queue(event)
                return True
            self._stop_capture("resume_unavailable" if self.active else "begin_unavailable")
        return False

    def _execute_control_step(self, row, deadline_ns):
        step = execute_control_step(
            self.controller,
            self.shared,
            self.robot,
            row,
            self.recorder,
            termination_details=self.termination_details,
            action_deadline_ns=deadline_ns,
        )
        if step.interrupted:
            logger.info("teleop interrupted: dispatch=%s", step.dispatch)
            self._stop_capture("motion_revoked", stop_motion=False)
            return
        if self.recorder is not None and not step.target_dispatched:
            self._stop_capture("control_failure")
            return
        if (
            self.recorder is not None
            and time.monotonic() - self.capture_started_s
            >= self.runtime.policy.max_record_duration_s
        ):
            self._pause_control("max_record_duration", run_end_reason=RunEndReason.TIMEOUT)
            self._end_segment(True, "max_record_duration")
            return

        if self.recorder is None:
            self.failures = self.failures + 1 if not step.target_dispatched else 0
            if self.failures >= _DEBUG_FAILURE_LIMIT:
                self._stop_capture("control_failure")

    def _wait_for_vr(self) -> bool:
        deadline = time.monotonic() + self.runtime.safety.readiness_timeouts_s["vr"]
        while True:
            if not self.keyboard.healthy:
                self.shared.estop_request = True
            if self._poll_blocking_commands():
                return False
            self.robot.service_idle()
            if self.shared.sensors.vr_ready.wait(timeout=0.05):
                # Consume startup keystrokes before accepting a new B after readiness.
                return not self._poll_blocking_commands()
            if time.monotonic() >= deadline:
                raise RuntimeError("VR failed to become ready before the startup timeout")

    def _initialize(self) -> bool:
        # Keyboard callbacks are active before the startup hand HOME.
        home_result = home_hand(
            self.shared,
            self.runtime,
            robot=self.robot,
            abort_requested=self._poll_blocking_commands,
        )
        if not home_result.ok:
            if home_result.interrupted:
                self.shared.quit_requested = True
                return False
            raise RuntimeError(f"startup hand home failed: {home_result.reason}")
        if self._poll_blocking_commands():
            return False
        self.start_vr()
        if not self._wait_for_vr():
            return False
        print("准备进入遥操作；空闲且确认手部无接触后按 T 归零触觉，S/Q/ESC 可取消。"
              "dense 仅检查载荷/残差有限，不验证稳定性或无接触。", flush=True)
        return True

    def _run_control_tick(self):
        if (
            self.recorder is not None
            and self.recorder.is_recording
            and time.monotonic() - self.capture_started_s
            >= self.runtime.policy.max_record_duration_s
        ):
            self._pause_control("max_record_duration", run_end_reason=RunEndReason.TIMEOUT)
            self._end_segment(True, "max_record_duration")
            return
        now_ns = time.monotonic_ns()
        if not self.paused and self.tick_origin_ns is not None:
            if now_ns < self.next_tick_ns:
                return
            if now_ns > self.next_tick_ns + self.sample_tolerance_ns:
                self.termination_details.append(dict(
                    stage="sampling", reason="missed_tick", tick=self.tick_index,
                    due_ns=self.next_tick_ns, observed_ns=now_ns,
                ))
                self._stop_capture("missed_tick")
                return
        row = read_observation(
            self.shared,
            self.runtime,
            self.robot,
            require_hand=self.runtime.policy.hand_enabled,
            require_camera=self.recorder is not None,
            require_vr=True,
        )
        if row is None:
            if self.recorder is not None:
                from dexmani_real.runtime.observation import (
                    read_camera_frame,
                    sample_is_fresh,
                )

                camera = read_camera_frame(self.shared)
                if camera is None or not sample_is_fresh(
                    camera["timestamp_ns"], self.runtime.camera.max_frame_age_s
                ):
                    revoke_motion(self.shared, reason=RunEndReason.HARDWARE_FAULT)
                    self._stop_capture("camera_unavailable")
                    raise RuntimeError("required recording camera unavailable")
            self._stop_capture("observation_unavailable")
            return
        if self.paused:
            self._try_resume(row)
            return
        stamp = row.observation_timestamp_ns
        if self.tick_origin_ns is None:
            self.tick_origin_ns = stamp
            self.next_tick_ns = stamp
        if not self.next_tick_ns <= stamp <= self.next_tick_ns + self.sample_tolerance_ns:
            self.termination_details.append(dict(
                stage="sampling", reason="late_observation", tick=self.tick_index,
                due_ns=self.next_tick_ns, observed_ns=stamp,
            ))
            # Preserve the actual late row only after authority is revoked/stopped.
            self._stop_capture("late_observation", row=row)
            return
        self.tick_index += 1
        self.next_tick_ns = self.tick_origin_ns + self.tick_index * self.control_dt_ns
        self._execute_control_step(row, self.next_tick_ns)

    def _shutdown_capture_and_inputs(self, failure):
        revoke_motion(self.shared, reason=RunEndReason.RUNTIME_SHUTDOWN)
        try:
            if self.robot.stop_required:
                self._stop_motion("shutdown_stop")
        except Exception as exc:
            if failure is None:
                failure = exc
            logger.exception("teleop stop failed")
        try:
            self.controller.clear_reference()
        except Exception as exc:
            if failure is None:
                failure = exc
            self.termination_details.append(
                dict(
                    stage="reference_cleanup",
                    exception_type=type(exc).__name__,
                    message=str(exc),
                )
            )
            logger.exception("teleop reference cleanup failed")
        try:
            if self.recorder is not None:
                if self.recorder.is_recording:
                    self._finish_capture(True, "interrupted", announce=False)
        except Exception as exc:
            if failure is None:
                failure = exc
            logger.exception("teleop recording cleanup failed")
        finally:
            if self.recorder is not None:
                try:
                    self.recorder.close()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    logger.exception("teleop recorder close failed")
        try:
            # Motion is revoked and recording is finalized. Let the exit
            # or emergency cue finish before close() cancels the player.
            if not self.audio.wait_until_idle(timeout_s=_END_AUDIO_GRACE_S):
                logger.warning("Final audio did not finish within %.1fs", _END_AUDIO_GRACE_S)
        except Exception:
            logger.warning("Final audio unavailable", exc_info=True)
        for close in (self.keyboard.stop, self.audio.close):
            try:
                close()
            except Exception as exc:
                if failure is None:
                    failure = exc
                logger.exception("teleop resource cleanup failed")
        return failure

    def run(self) -> None:
        failure = None
        try:
            self.keyboard.start()
            check_services = self.robot.check_services
            self.robot.check_services = lambda: (
                (check_services is None or check_services()) and self.keyboard.healthy
            )
            ready = self._initialize()
            while ready and self.shared.sensors.is_running.value and not self.shared.quit_requested:
                self.robot.check()
                if not self.active or self.paused:
                    self.robot.service_idle()
                if not self.keyboard.healthy:
                    self.shared.estop_request = True
                if self.shared.estop_request or self.shared.error_state:
                    self._stop_capture(
                        "estop" if self.shared.estop_request else "runtime_failure"
                    )
                    break
                if self.recorder is not None:
                    self.recorder.check_error()
                timeout = 0.005
                if self.active and not self.paused:
                    timeout = min(timeout, max(0.0, (self.next_tick_ns - time.monotonic_ns()) / 1e9))
                epoch = int(self.shared.run_id)
                commands = self.keyboard.poll(timeout=timeout)
                self._handle_commands(commands, epoch)
                if (
                    self.shared.estop_request
                    or self.shared.error_state
                    or not self.shared.sensors.is_running.value
                ):
                    break
                if self.shared.quit_requested or (
                    not self.resume_requested and (not self.active or self.paused)
                ):
                    continue
                if not self.paused and time.monotonic_ns() < self.next_tick_ns:
                    continue
                self._run_control_tick()
        except KeyboardInterrupt:
            self.shared.estop_request = True
            revoke_motion(self.shared, reason=RunEndReason.ESTOP)
            raise
        except Exception as exc:
            failure = exc
            self.shared.error_state = True
            self.shared.sensors.is_running.value = False
            revoke_motion(
                self.shared,
                reason=RunEndReason.RECORDING_FAILURE
                if isinstance(exc, RecordingError)
                else RunEndReason.HARDWARE_FAULT
                if isinstance(exc, DispatchError)
                else RunEndReason.POLICY_FAILURE,
            )
            self.termination_details.append(dict(
                stage="runtime", exception_type=type(exc).__name__, message=str(exc),
            ))
            logger.exception("teleop failed")
            self.audio.play("emergency")
        finally:
            failure = self._shutdown_capture_and_inputs(failure)
        if failure is not None:
            raise failure

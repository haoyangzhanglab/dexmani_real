"""Operator-supervised teleop with one current observation per real control step."""

import logging
import time
from concurrent.futures import CancelledError
from enum import Enum

from dexmani_real.data.recorder import (
    AsyncEpisodeRecorder,
    RecordingError,
    snapshot_recording_metadata,
)
from dexmani_real.data.session_results import error_detail
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.runtime.motion.home import home_hand, home_robot
from dexmani_real.runtime.observation import (
    check_observation,
    control_deadline_ns,
    read_observation,
    service_feedback,
)
from dexmani_real.runtime.operator_input import OperatorCommand
from dexmani_real.runtime.state import RunEndReason, SafetyState
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)


class TeleopPhase(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    ACTIVE = "active"


class TeleopRunner:
    """持有遥操作控制和当前 capture；录制配置独立于 capture 是否存在。"""

    def __init__(
        self,
        runtime_state,
        runtime,
        robot,
        *,
        task_label,
        control_needs,
        record_needs,
        sensors,
        controller,
        home_planner,
        vr_process,
        recording_dir,
        keyboard,
        audio,
        camera_snapshot=None,
    ):
        self.sensors = sensors
        self.control_needs, self.record_needs = control_needs, record_needs
        self.needs = control_needs.union(record_needs)
        self.runtime_state = runtime_state
        self.robot = robot
        self.vr_process = vr_process
        self.camera_snapshot = camera_snapshot
        self.task_label = task_label
        self.runtime = runtime
        self.recording_dir = recording_dir
        self.recorder = None
        self.keyboard = keyboard
        self.audio = audio
        self.controller = controller
        self.phase = TeleopPhase.IDLE
        self.preparing_epoch = None
        self.prepared_after_ns = 0
        self.capture_failed = False
        self.pending_termination_reason = None
        self.termination_details = []
        self.control_run_id = None
        self.tick_origin_ns = None
        self.tick_index = 0
        self.control_dt_ns, _ = sampling_clock_ns(self.runtime.teleop.control_hz)
        self.capture_started_s = 0.0

        self.home_planner = home_planner

    @property
    def next_tick_ns(self):
        if self.tick_origin_ns is None:
            return None
        return self.tick_origin_ns + self.tick_index * self.control_dt_ns

    def end_reason(self, fallback):
        with self.runtime_state.motion_lock:
            if (
                self.control_run_id is not None
                and self.runtime_state.run_ended_id == self.control_run_id
            ):
                return self.runtime_state.run_ended_reason.name.lower()
        return fallback

    def stop_motion(self, stage):
        try:
            self.robot.stop()
        except Exception as exc:
            if self.recorder is not None and self.recorder.published_path is None:
                self.capture_failed = True
            self.termination_details.append(error_detail(stage, exc))
            raise

    def stop_control(self, reason, *, run_end_reason=RunEndReason.OPERATOR, expected_run_id=None):
        with self.runtime_state.motion_lock:
            if expected_run_id is not None and (
                expected_run_id != self.runtime_state.run_id
                or not self.runtime_state.requests_allow_motion()
            ):
                return False
            self.runtime_state.revoke_motion(reason=run_end_reason)
            stopped_epoch = self.runtime_state.run_id
            if self.pending_termination_reason is None:
                self.pending_termination_reason = self.end_reason(reason)
        self.termination_details.append(dict(stage="termination", reason=reason))
        self.stop_motion("operator_stop")
        self.runtime_state.acknowledge_stop(stopped_epoch)
        self.controller.clear_reference()
        self.preparing_epoch = None
        return True

    def end_segment(self, save, reason, *, announce=True):
        """Called only after revocation/stop, including sessions without a recorder."""
        self.finish_capture(save, reason, announce=announce)
        self.phase = TeleopPhase.IDLE
        self.preparing_epoch = self.control_run_id = None

    def finish_capture(self, save, reason, *, announce=True) -> None:
        if self.recorder is None or not self.recorder.start_attempted:
            self.pending_termination_reason = None
            self.termination_details = []
            return
        reason = self.pending_termination_reason or self.end_reason(reason)
        published = None
        if save:
            published = self.recorder.save_episode(
                reason=reason,
                details=self.termination_details,
                technical_status=(
                    "normal" if not self.capture_failed and reason in {"operator", "quit", "timeout"}
                    else "failed"
                ),
            )
        else:
            self.recorder.discard_episode(reason=reason)
        self.recorder.check_error()
        self.recorder = None
        self.pending_termination_reason = None
        self.termination_details = []
        if announce:
            self.audio.play("save" if published is not None else "discard")

    def stop_capture(
        self,
        reason,
        *,
        run_end_reason=RunEndReason.EXECUTOR_BOUNDARY,
        row=None,
        command=None,
    ):
        self.capture_failed = True
        failure = None
        try:
            self.stop_control(reason, run_end_reason=run_end_reason)
        except Exception as exc:
            failure = exc
        logger.warning("Teleop ended: %s", reason)
        self.audio.play("emergency")
        try:
            if row is not None and self.recorder is not None:
                self.recorder.add_frame(row, command)
        except Exception as exc:
            failure = failure or exc
            self.termination_details.append(error_detail("recording", exc))
        try:
            if failure is None:
                self.end_segment(True, reason, announce=False)
            else:
                # Keep the fault visible; never advertise normal IDLE after stop failure.
                self.finish_capture(True, reason, announce=False)
        except Exception as exc:
            failure = failure or exc
        if failure is not None:
            raise failure

    def poll_blocking_commands(self, commands=()):
        self.robot.check()
        aborted = False
        terminal = tuple(
            cmd for cmd in (*commands, *self.keyboard.poll(timeout=0))
            if cmd in (OperatorCommand.EMERGENCY_STOP, OperatorCommand.QUIT,
                       OperatorCommand.STOP, OperatorCommand.DISCARD)
        )
        if terminal:
            self.handle_commands(terminal, self.runtime_state.run_id)
            aborted = True
        return bool(
            aborted
            or self.runtime_state.stop_request
            or self.runtime_state.estop_request
            or self.runtime_state.error_state
            or self.runtime_state.quit_requested
            or not self.runtime_state.running
        )

    def preparation_valid(self):
        return self.runtime_state.command_may_cross_sdk(
            run_id=self.preparing_epoch,
            required_safety_state=SafetyState.ARMED,
        )

    def start_segment(self, expected_run_id):
        if self.phase is not TeleopPhase.IDLE:
            return
        if self.recorder is not None:
            raise RecordingError("previous capture still owns resources")
        with self.runtime_state.motion_lock:
            # B belongs to the generation from before poll, never to a later STOP.
            if expected_run_id != self.runtime_state.run_id:
                return
            self.runtime_state.revoke_motion()
            self.preparing_epoch = self.runtime_state.run_id
        self.stop_motion("segment_stop")
        self.phase = TeleopPhase.PREPARING
        if not self.preparation_valid():
            self.stop_capture("preparation_revoked")
            return
        self.robot.reset_hand_feedback()
        try:
            row = read_observation(
                self.sensors.channels, self.robot, needs=self.needs,
                wait_for_hand=True,
                cancel_requested=lambda: (
                    self.poll_blocking_commands() or not self.preparation_valid()
                ),
            )
            issues, _ = self.admit_observation(row)
        except CancelledError:
            # S/Q may already have finalized preparation in poll_blocking_commands.
            if self.preparing_epoch is not None:
                self.stop_capture("preparation_revoked")
            return
        if issues or not self.preparation_valid():
            print("Start requires fresh robot, VR and recording resources", flush=True)
            self.stop_capture("preparation_unavailable")
            return
        self.control_run_id = None
        if self.recording_dir is not None:
            self.recorder = AsyncEpisodeRecorder(
                self.recording_dir, control_hz=self.runtime.teleop.control_hz,
            )
            self.recorder.start_episode(
                task_label=self.task_label,
                collection_source="teleop",
                metadata=snapshot_recording_metadata(
                    self.camera_snapshot,
                    tactile_reset_metadata=self.robot.hand.get_tactile_reset_metadata(),
                    tactile_hold_max_age_s=self.runtime.hand.feedback_max_age_s,
                ),
            )
        self.capture_started_s = time.monotonic()
        self.capture_failed = False
        self.pending_termination_reason = None
        self.termination_details = []
        try:
            self.robot.prepare_streaming(
                self.preparing_epoch, service=lambda: service_feedback(self.robot, self.runtime),
            )
        except OperationCancelled:
            self.stop_capture("preparation_revoked")
            return
        # START/prepare must finish before the feedback and VR anchor used to begin.
        self.prepared_after_ns = time.monotonic_ns()
        if not self.preparation_valid():
            self.stop_capture("preparation_revoked")

    def handle_operator_command(self, cmd, *, home_commands=(), expected_run_id=None) -> bool:
        if cmd is OperatorCommand.EMERGENCY_STOP:
            self.runtime_state.request_emergency_stop()
            self.audio.play("emergency")
            return True
        if (
            self.runtime_state.estop_request
            or self.runtime_state.error_state
            or self.runtime_state.quit_requested
            or not self.runtime_state.running
        ):
            return True
        if cmd is OperatorCommand.QUIT:
            self.runtime_state.request_quit()
            self.stop_control("quit", run_end_reason=RunEndReason.QUIT)
            self.end_segment(True, "quit", announce=False)
            self.audio.play("end")
            return True
        elif cmd in (OperatorCommand.STOP, OperatorCommand.DISCARD):
            reason = cmd.value.lower()
            try:
                self.stop_control(reason)
            except Exception as stop_error:
                if cmd is OperatorCommand.DISCARD:
                    # 锁定已有 recorder 的丢弃请求，避免 close() 默认保存。
                    try:
                        self.finish_capture(False, reason, announce=False)
                    except Exception as discard_error:
                        raise stop_error from discard_error
                raise
            self.end_segment(cmd is OperatorCommand.STOP, reason)
        elif cmd is OperatorCommand.RESET_TACTILE:
            if (
                self.preparing_epoch is not None
                or self.phase is not TeleopPhase.IDLE
                or (self.recorder is not None and self.recorder.is_recording)
                or self.runtime_state.safety_state != SafetyState.ARMED
            ):
                logger.warning("T ignored: tactile reset requires idle, no capture or preparation")
                return False
            try:
                logger.info("正在请求触觉传感器复位，请保持手部无接触；S/Q/Esc 可中断后续请求。")
                self.robot.reset_tactile_sensors(cancel_requested=self.poll_blocking_commands)
            except CancelledError as exc:
                logger.info("%s", exc)
            finally:
                self.keyboard.drain_signal(OperatorCommand.RESET_TACTILE)
                self.keyboard.drain_signal(OperatorCommand.BEGIN)
            return True
        elif cmd is OperatorCommand.HOME:
            if not self.stop_control("home", expected_run_id=expected_run_id):
                return True
            self.end_segment(True, "home")
            if not self.poll_blocking_commands(home_commands):
                self.audio.play("home")
                result = home_robot(
                    self.runtime_state,
                    self.runtime.arm, self.runtime.hand,
                    self.home_planner,
                    robot=self.robot,
                    service=lambda: service_feedback(self.robot, self.runtime),
                    abort_requested=self.poll_blocking_commands,
                )
                if result.ok:
                    self.audio.play("home_done", interrupt=False)
                elif result.interrupted:
                    logger.info("HOME cancelled: %s", result.reason)
                else:
                    raise RuntimeError(f"HOME failed: {result.reason}")
            self.poll_blocking_commands()
            return True
        elif cmd is OperatorCommand.BEGIN and self.phase is TeleopPhase.IDLE:
            self.start_segment(expected_run_id)
        return False

    def handle_commands(self, commands, expected_run_id):
        for terminal in (
            OperatorCommand.EMERGENCY_STOP, OperatorCommand.QUIT,
            OperatorCommand.STOP, OperatorCommand.DISCARD,
        ):
            if terminal in commands:
                commands = [terminal]
                break
        for index, cmd in enumerate(commands):
            if self.handle_operator_command(
                cmd,
                expected_run_id=expected_run_id,
                home_commands=commands[index + 1 :] if cmd is OperatorCommand.HOME else (),
            ):
                break

    def try_begin(self, row) -> bool:
        if self.preparing_epoch is not None and not self.preparation_valid():
            self.stop_capture("preparation_revoked")
            return False
        if self.preparing_epoch is not None and all(
            int(stamp) > self.prepared_after_ns
            for stamp in (
                row.arm["timestamp_ns"][0],
                row.vr["recv_ts_ns"],
                (
                    row.hand["timestamp_ns"][0]
                    if row.hand is not None
                    else row.observation_timestamp_ns
                ),
            )
        ):
            prepared = self.controller.reset_reference(row)
            with self.runtime_state.motion_lock:
                authorized = (
                    prepared and self.preparation_valid() and self.runtime_state.begin_motion()
                )
                if authorized:
                    self.control_run_id = self.runtime_state.run_id
            if authorized:
                self.phase = TeleopPhase.ACTIVE
                self.preparing_epoch = None
                self.tick_origin_ns = None
                self.tick_index = 0
                self.audio.play("begin", interrupt=False)
                return True
            self.stop_capture("begin_unavailable")
        return False

    def execute_control_step(self, row, deadline_ns, feedback_until_ns):
        epoch = self.control_run_id
        deadline_ns = min(deadline_ns, feedback_until_ns)
        action_result = command = failure = None
        reason = None
        interrupted = False
        run_end_reason = RunEndReason.EXECUTOR_BOUNDARY
        try:
            action_result = self.controller.compute_target(row)
            if action_result is None or action_result.arm_qpos is None:
                detail = action_result.rejection_reason if action_result is not None else None
                reason = f"target_unavailable: {detail}"
            else:
                candidate = RobotCommand(epoch, action_result.arm_qpos, action_result.hand_qpos)
                if self.recorder is not None:
                    self.recorder.check_error()
                if not self.runtime_state.command_may_cross_sdk(run_id=epoch):
                    raise OperationCancelled("teleop cancelled before submission")
                if time.monotonic_ns() >= deadline_ns:
                    raise TimeoutError("teleop expired before submission")
                command = candidate
                self.robot.send_action(
                    command,
                    valid_until_ns=deadline_ns,
                    execute_until_ns=int(row.vr["recv_ts_ns"])
                    + int(self.runtime.teleop.vr_mapping.stale_threshold_s * 1e9),
                )
        except BaseException as exc:
            failure = exc
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            interrupted = isinstance(exc, (OperationCancelled, TimeoutError))
            reason = f"control_failed: {type(exc).__name__}: {exc}"
            if isinstance(exc, KeyboardInterrupt):
                run_end_reason = RunEndReason.ESTOP
            elif isinstance(exc, RecordingError):
                run_end_reason = RunEndReason.RECORDING_FAILURE
            elif interrupted:
                run_end_reason = RunEndReason.EXECUTOR_BOUNDARY
            elif isinstance(exc, RobotError):
                run_end_reason = RunEndReason.HARDWARE_FAULT
            else:
                run_end_reason = RunEndReason.POLICY_FAILURE
        if reason is None:
            if time.monotonic_ns() >= deadline_ns:
                reason = "submission_deadline_expired"
            elif (
                self.runtime_state.run_id != epoch
                or self.runtime_state.estop_request
                or self.runtime_state.error_state
                or self.runtime_state.quit_requested
                or self.runtime_state.stop_request
                or not self.runtime_state.running
            ):
                reason = "motion_revoked"
        if reason is not None:
            try:
                self.stop_capture(
                    reason,
                    run_end_reason=run_end_reason,
                    row=row,
                    command=command,
                )
            except Exception as exc:
                if failure is not None and not interrupted:
                    raise failure from exc
                raise
            if failure is not None and not interrupted:
                raise failure
            return
        self.controller.commit_submitted_target(action_result)
        if self.recorder is not None:
            self.recorder.add_frame(row, command)

    def wait_for_vr(self) -> bool:
        deadline = time.monotonic() + self.runtime.safety.vr_ready_timeout_s
        while True:
            if not self.keyboard.healthy:
                self.runtime_state.request_emergency_stop()
            if self.poll_blocking_commands():
                return False
            self.robot.service_idle()
            if self.sensors.channels.vr_ready.wait(timeout=0.05):
                # Consume startup keystrokes before accepting a new B after readiness.
                return not self.poll_blocking_commands()
            if time.monotonic() >= deadline:
                raise RuntimeError("VR failed to become ready before the startup timeout")

    def initialize(self) -> bool:
        # Keyboard callbacks are active before the startup hand HOME.
        home_result = home_hand(
            self.runtime_state,
            self.runtime.hand,
            robot=self.robot,
            abort_requested=self.poll_blocking_commands,
        )
        if not home_result.ok:
            if home_result.interrupted:
                self.runtime_state.request_quit()
                return False
            raise RuntimeError(f"startup hand home failed: {home_result.reason}")
        self.robot.stop()
        if self.poll_blocking_commands():
            return False
        self.sensors.start(
            self.vr_process,
            ready=self.sensors.channels.vr_ready,
            may_start=lambda: self.keyboard.healthy and self.runtime_state.requests_allow_motion(),
        )
        if not self.wait_for_vr():
            return False
        print(
            "准备进入遥操作；空闲且确认手部无接触后按 T 请求复位触觉，S/Q/ESC 可取消。", flush=True
        )
        return True

    def admit_observation(self, row):
        issues, feedback_until_ns = check_observation(
            row, self.runtime, self.control_needs, require_hand=self.runtime.hand.enabled,
        )
        if issues:
            return issues, feedback_until_ns
        if self.recording_dir is not None:
            record_issues, _ = check_observation(row, self.runtime, self.record_needs)
            if record_issues:
                raise RecordingError("recording sources: " + "; ".join(record_issues))
        self.robot.arm.update_health(
            health_until_ns=control_deadline_ns(feedback_until_ns, self.runtime.arm),
        )
        return issues, feedback_until_ns

    def run_control_tick(self):
        if (
            self.recorder is not None
            and self.recorder.is_recording
            and time.monotonic() - self.capture_started_s
            >= self.runtime.collection.max_duration_s
        ):
            self.stop_control("max_record_duration", run_end_reason=RunEndReason.TIMEOUT)
            self.end_segment(True, "max_record_duration")
            return
        now_ns = time.monotonic_ns()
        if self.phase is TeleopPhase.ACTIVE and self.tick_origin_ns is not None:
            if now_ns < self.next_tick_ns:
                return
            if now_ns >= self.next_tick_ns + self.control_dt_ns:
                self.termination_details.append(
                    dict(
                        stage="sampling",
                        reason="missed_tick",
                        tick=self.tick_index,
                        due_ns=self.next_tick_ns,
                        observed_ns=now_ns,
                    )
                )
                self.stop_capture("missed_tick")
                return
        row = read_observation(
            self.sensors.channels, self.robot, needs=self.needs,
        )
        issues, feedback_until_ns = self.admit_observation(row)
        if issues:
            self.stop_capture("control_sources: " + "; ".join(issues))
            return
        if self.phase is not TeleopPhase.ACTIVE:
            self.try_begin(row)
            return
        stamp = row.observation_timestamp_ns
        if self.tick_origin_ns is None:
            self.tick_origin_ns = stamp
        if stamp >= self.next_tick_ns + self.control_dt_ns:
            self.termination_details.append(
                dict(
                    stage="sampling",
                    reason="late_observation",
                    tick=self.tick_index,
                    due_ns=self.next_tick_ns,
                    observed_ns=stamp,
                )
            )
            # Preserve the actual late row only after authority is revoked/stopped.
            self.stop_capture("late_observation", row=row)
            return
        self.tick_index += 1
        self.execute_control_step(row, self.next_tick_ns, feedback_until_ns)

    def close(self):
        """Revoke and preserve the capture; the session owns input/audio resources."""
        failure = None
        self.runtime_state.revoke_motion(reason=RunEndReason.RUNTIME_SHUTDOWN)
        try:
            if self.robot.stop_required:
                self.stop_motion("shutdown_stop")
        except Exception as exc:
            failure = exc
            logger.exception("teleop stop failed")
        try:
            self.controller.clear_reference()
        except Exception as exc:
            if self.recorder is not None and self.recorder.published_path is None:
                self.capture_failed = True
            if failure is None:
                failure = exc
            self.termination_details.append(error_detail("reference_cleanup", exc))
            logger.exception("teleop reference cleanup failed")
        try:
            self.finish_capture(True, "interrupted", announce=False)
        except Exception as exc:
            if failure is None:
                failure = exc
            logger.exception("teleop recording cleanup failed")
        finally:
            if self.recorder is not None:
                # START 失败或首次收尾超时仍由当前 owner 等待；不重试写入。
                try:
                    self.recorder.close()
                except Exception as exc:
                    if failure is None:
                        failure = exc
                    logger.exception("teleop recorder close failed")
        if failure is not None:
            raise failure

    def run(self) -> None:
        try:
            ready = self.initialize()
            while ready and self.runtime_state.running and not self.runtime_state.quit_requested:
                self.robot.arm.check_error()
                if self.preparing_epoch is None and self.phase is not TeleopPhase.ACTIVE:
                    self.robot.service_idle()
                if not self.keyboard.healthy:
                    self.runtime_state.request_emergency_stop()
                if self.runtime_state.estop_request or self.runtime_state.error_state:
                    self.stop_capture(
                        "estop" if self.runtime_state.estop_request else "runtime_failure"
                    )
                    break
                if self.recorder is not None:
                    self.recorder.check_error()
                timeout = 0.005
                if self.phase is TeleopPhase.ACTIVE:
                    timeout = (
                        0.0
                        if self.next_tick_ns is None
                        else min(timeout, max(0.0, (self.next_tick_ns - time.monotonic_ns()) / 1e9))
                    )
                epoch = self.runtime_state.run_id
                commands = self.keyboard.poll(timeout=timeout)
                self.handle_commands(commands, epoch)
                if (
                    self.runtime_state.estop_request
                    or self.runtime_state.error_state
                    or not self.runtime_state.running
                ):
                    break
                if self.runtime_state.quit_requested or (
                    self.preparing_epoch is None and self.phase is not TeleopPhase.ACTIVE
                ):
                    continue
                if (
                    self.phase is TeleopPhase.ACTIVE
                    and self.next_tick_ns is not None
                    and time.monotonic_ns() < self.next_tick_ns
                ):
                    continue
                self.run_control_tick()
        except KeyboardInterrupt:
            self.runtime_state.request_emergency_stop()
            raise
        except Exception as exc:
            self.runtime_state.mark_error()
            if isinstance(exc, RecordingError):
                reason = RunEndReason.RECORDING_FAILURE
            elif isinstance(exc, RobotError):
                reason = RunEndReason.HARDWARE_FAULT
            else:
                reason = RunEndReason.POLICY_FAILURE
            self.runtime_state.revoke_motion(reason=reason)
            if self.recorder is not None and self.recorder.published_path is None:
                self.capture_failed = True
            self.sensors.request_shutdown()
            self.termination_details.append(error_detail("runtime", exc))
            logger.exception("teleop failed")
            self.audio.play("emergency")
            raise

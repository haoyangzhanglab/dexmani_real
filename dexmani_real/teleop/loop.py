"""遥操作控制循环与单段示教录制。"""

import logging
import time
from enum import Enum

from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import (
    EpisodeRecorder,
    RecordingError,
)
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.runtime.home import home_robot
from dexmani_real.runtime.observation import (
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState
from dexmani_real.utils.log import error_detail
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)


def failure_reason(exc):
    if isinstance(exc, KeyboardInterrupt):
        return RunEndReason.ESTOP
    if isinstance(exc, RecordingError):
        return RunEndReason.RECORDING_FAILURE
    if isinstance(exc, RobotError):
        return RunEndReason.HARDWARE_FAULT
    return RunEndReason.POLICY_FAILURE


class TeleopPhase(Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    ACTIVE = "active"


class TeleopLoop:
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
        recording_dir,
        keyboard,
        audio,
        camera_snapshot=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.robot = robot
        self.sensors = sensors
        self.control_needs, self.record_needs = control_needs, record_needs
        self.needs = control_needs.union(record_needs)
        self.controller = controller
        self.home_planner = home_planner
        self.recording_dir = recording_dir
        self.task_label = task_label
        self.camera_snapshot = camera_snapshot
        self.keyboard, self.audio = keyboard, audio

        self.phase = TeleopPhase.IDLE
        self.run_id = None
        self.prepared_after_ns = 0
        self.tick_origin_ns = None
        self.tick_index = 0
        self.control_dt_ns, _ = sampling_clock_ns(runtime.teleop.control_hz)

        self.recorder = None
        self.capture_started_s = 0.0
        self.capture_failed = False
        self.discard_requested = False
        self.pending_termination_reason = None
        self.termination_details = []
        self.last_error = None

    @property
    def next_tick_ns(self):
        if self.tick_origin_ns is None:
            return None
        return self.tick_origin_ns + self.tick_index * self.control_dt_ns

    def request_discard(self):
        with self.runtime_state.motion_lock:
            self.discard_requested = True
            self.runtime_state.request_stop()

    def record_error(self, stage, exc):
        if isinstance(exc, KeyboardInterrupt):
            self.runtime_state.request_emergency_stop()
        else:
            self.runtime_state.mark_error()
        if self.recorder is not None and self.recorder.published_path is None:
            self.capture_failed = True
        if exc is not self.last_error:
            self.termination_details.append(error_detail(stage, exc))
            self.last_error = exc

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
                self.pending_termination_reason = (
                    self.runtime_state.run_ended_reason
                    if self.runtime_state.run_ended_id == self.run_id else run_end_reason
                ).name.lower()
        self.termination_details.append(dict(stage="termination", reason=reason))
        if self.robot.stop_required:
            self.robot.stop()
        self.runtime_state.acknowledge_stop(stopped_epoch)
        return True

    def end_episode(
        self, save, reason, *, run_end_reason=RunEndReason.OPERATOR,
        expected_run_id=None, announce=True, row=None, command=None,
    ):
        if not save:
            self.discard_requested = True
        failure = None
        try:
            if not self.stop_control(
                reason, run_end_reason=run_end_reason, expected_run_id=expected_run_id,
            ):
                return False
        except BaseException as exc:
            self.record_error("stop", exc)
            failure = exc
        self.controller.clear()

        if announce and self.capture_failed:
            self.audio.play("emergency")
        try:
            if row is not None and self.recorder is not None:
                self.recorder.append(row, command)
        except BaseException as exc:
            self.record_error("recording_frame", exc)
            logger.exception("teleop final frame failed")
            failure = failure or exc
        try:
            self.finish_recording(save, announce=announce and not self.capture_failed)
        except BaseException as exc:
            self.record_error("recording_cleanup", exc)
            logger.exception("teleop recording cleanup failed")
            failure = failure or exc
        if failure is not None:
            raise failure
        self.phase = TeleopPhase.IDLE
        self.run_id = None
        return True

    def finish_recording(self, save, *, announce=True):
        published = None
        captured = self.recorder is not None and self.recorder.start_attempted
        if captured:
            reason = self.pending_termination_reason
            published = self.recorder.finish(
                save=save and not self.discard_requested,
                reason=reason,
                details=self.termination_details,
                technical_status=(
                    "normal" if not self.capture_failed and reason in {"operator", "quit", "timeout"}
                    else "failed"
                ),
            )
        # finish 失败时保留 recorder，让会话收尾继续持有未释放的 writer。
        self.recorder = None
        if announce and captured:
            self.audio.play("save" if published is not None else "discard")

    def abort_episode(
        self, reason, *, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, row=None, command=None,
    ):
        self.capture_failed = True
        self.end_episode(True, reason, run_end_reason=run_end_reason, row=row, command=command)
        logger.warning("Teleop ended: %s", reason)

    def poll_blocking_commands(self, commands=()):
        self.robot.check()
        terminal = tuple(
            cmd for cmd in (*commands, *self.keyboard.poll(timeout=0))
            if cmd in ("esc", "q", "s", "d")
        )
        if terminal:
            self.handle_commands(terminal, self.runtime_state.run_id)
        return bool(terminal) or not self.runtime_state.requests_allow_motion()

    def preparation_valid(self):
        return self.phase is TeleopPhase.PREPARING and self.runtime_state.can_move(
            run_id=self.run_id, required_safety_state=SafetyState.ARMED,
        )

    def start_episode(self, expected_run_id):
        if self.phase is not TeleopPhase.IDLE:
            return
        if self.recorder is not None:
            raise RecordingError("previous capture still owns resources")
        with self.runtime_state.motion_lock:
            # B 只能属于取出按键前的代际，不能越过稍后到达的 STOP。
            if (
                expected_run_id != self.runtime_state.run_id
                or not self.runtime_state.requests_allow_motion()
            ):
                return
            self.capture_failed = self.discard_requested = False
            self.pending_termination_reason = self.last_error = None
            self.termination_details = []
            self.runtime_state.revoke_motion()
            self.run_id = self.runtime_state.run_id
            self.phase = TeleopPhase.PREPARING
        self.robot.stop()
        if not self.preparation_valid():
            self.abort_episode("preparation_revoked")
            return

        self.robot.reset_hand_feedback()
        try:
            self.robot.read_initial_state(
                required_tactile=self.needs.required_tactile,
                cancel_requested=lambda: (
                    self.poll_blocking_commands() or not self.preparation_valid()
                ),
            )
            if not self.preparation_valid():
                raise OperationCancelled("cancelled during initial feedback acquisition")
            row = read_observation(self.sensors.channels, self.robot, needs=self.needs)
            issues, _ = self.admit_observation(row)
        except OperationCancelled:
            if self.phase is TeleopPhase.PREPARING:
                self.abort_episode("preparation_revoked")
            return
        if issues or not self.preparation_valid():
            self.abort_episode("preparation_unavailable")
            return

        if self.recording_dir is not None:
            self.recorder = EpisodeRecorder(
                self.recording_dir, control_hz=self.runtime.teleop.control_hz,
            )
            self.recorder.start(
                task_label=self.task_label, collection_source="teleop",
                metadata=snapshot_recording_metadata(
                    self.camera_snapshot,
                    tactile_reset_metadata=self.robot.hand.get_tactile_reset_metadata(),
                    tactile_hold_max_age_s=self.runtime.hand.feedback_max_age_s,
                ),
            )
        self.capture_started_s = time.monotonic()
        try:
            self.robot.prepare_streaming(
                self.run_id, service=self.robot.service_feedback,
            )
        except OperationCancelled:
            self.abort_episode("preparation_revoked")
            return
        # 锚点与反馈必须来自 prepare 完成之后。
        self.prepared_after_ns = time.monotonic_ns()
        if not self.preparation_valid():
            self.abort_episode("preparation_revoked")

    def handle_operator_command(self, cmd, *, home_commands=(), expected_run_id=None):
        if cmd == "esc":
            self.runtime_state.request_emergency_stop()
            self.audio.play("emergency")
            return True
        if (
            self.runtime_state.estop_request or self.runtime_state.error_state
            or self.runtime_state.quit_requested or not self.runtime_state.running
        ):
            return True

        if cmd == "q":
            self.runtime_state.request_quit()
            self.end_episode(True, "quit", run_end_reason=RunEndReason.QUIT, announce=False)
            self.audio.play("end")
            return True
        if cmd in ("s", "d"):
            self.end_episode(cmd == "s", "stop" if cmd == "s" else "discard")
        elif cmd == "t":
            if (
                self.phase is not TeleopPhase.IDLE or self.recorder is not None
                or self.runtime_state.safety_state != SafetyState.ARMED
            ):
                logger.warning("T ignored: tactile reset requires idle, no capture or preparation")
                return False
            try:
                logger.info("正在请求触觉传感器复位，请保持手部无接触；S/Q/Esc 可中断后续请求。")
                self.robot.reset_tactile_sensors(cancel_requested=self.poll_blocking_commands)
            except OperationCancelled as exc:
                logger.info("%s", exc)
            finally:
                self.keyboard.discard_events("t", "b")
            return True
        elif cmd == "h":
            if not self.end_episode(True, "home", expected_run_id=expected_run_id):
                return True
            if not self.poll_blocking_commands(home_commands):
                self.audio.play("home")
                result = home_robot(
                    self.runtime_state, self.robot, self.home_planner,
                    service=self.robot.service_feedback,
                    cancel_requested=self.poll_blocking_commands,
                )
                if result.ok:
                    self.audio.play("home_done", interrupt=False)
                elif result.interrupted:
                    logger.info("HOME cancelled: %s", result.reason)
                else:
                    raise RuntimeError(f"HOME failed: {result.reason}")
            self.poll_blocking_commands()
            return True
        elif cmd == "b":
            self.start_episode(expected_run_id)
        return False

    def handle_commands(self, commands, expected_run_id):
        for terminal in ("esc", "q", "s", "d"):
            if terminal in commands:
                commands = [terminal]
                break
        for index, cmd in enumerate(commands):
            if self.handle_operator_command(
                cmd,
                expected_run_id=expected_run_id,
                home_commands=commands[index + 1 :] if cmd == "h" else (),
            ):
                break

    def try_begin(self, row):
        if not self.preparation_valid():
            self.abort_episode("preparation_revoked")
            return
        stamps = [row.arm["timestamp_ns"][0], row.vr["recv_ts_ns"]]
        if row.hand is not None:
            stamps.append(row.hand["timestamp_ns"][0])
        if min(stamps) <= self.prepared_after_ns:
            return
        self.controller.reset(row)
        with self.runtime_state.motion_lock:
            authorized = self.preparation_valid() and self.runtime_state.begin_motion()
        if not authorized:
            self.abort_episode("begin_unavailable")
            return
        self.phase = TeleopPhase.ACTIVE
        self.tick_origin_ns = None
        self.tick_index = 0
        self.audio.play("begin", interrupt=False)

    def execute_control_step(self, row, deadline_ns, feedback_until_ns):
        epoch = self.run_id
        deadline_ns = min(deadline_ns, feedback_until_ns)
        command = None
        reason = None
        try:
            result = self.controller.compute_target(row)
            if result is None or result.rejection_reason is not None or result.arm_qpos is None:
                detail = result.rejection_reason if result is not None else "hand geometry rejected"
                reason = f"target_unavailable: {detail}"
            else:
                if self.recorder is not None:
                    self.recorder.check_error()
                try:
                    if not self.runtime_state.can_move(run_id=epoch):
                        raise OperationCancelled("teleop cancelled before submission")
                    if time.monotonic_ns() >= deadline_ns:
                        raise TimeoutError("teleop expired before submission")
                    command = RobotCommand(epoch, result.arm_qpos, result.hand_qpos)
                    self.robot.send_action(
                        command,
                        valid_until_ns=deadline_ns,
                        execute_until_ns=int(row.vr["recv_ts_ns"])
                        + int(self.runtime.teleop.vr_stale_threshold_s * 1e9),
                    )
                except (OperationCancelled, TimeoutError) as exc:
                    reason = f"control_interrupted: {exc}"
        except BaseException as exc:
            self.record_error("control", exc)
            try:
                self.abort_episode(
                    f"control_failed: {type(exc).__name__}: {exc}",
                    run_end_reason=failure_reason(exc),
                    row=row,
                    command=command,
                )
            except BaseException:
                logger.exception("teleop control cleanup also failed")
            raise

        if reason is None:
            if time.monotonic_ns() >= deadline_ns:
                reason = "submission_deadline_expired"
            elif not self.runtime_state.can_move(run_id=epoch):
                reason = "motion_revoked"
        if reason is not None:
            self.abort_episode(reason, row=row, command=command)
        else:
            self.controller.commit(result)
            if self.recorder is not None:
                self.recorder.append(row, command)

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
        return issues, feedback_until_ns

    def run_control_tick(self):
        if (
            self.recorder is not None
            and self.recorder.is_recording
            and time.monotonic() - self.capture_started_s
            >= self.runtime.collection.max_duration_s
        ):
            self.end_episode(True, "max_record_duration", run_end_reason=RunEndReason.TIMEOUT)
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
                self.abort_episode("missed_tick")
                return
        row = read_observation(
            self.sensors.channels, self.robot, needs=self.needs,
        )
        issues, feedback_until_ns = self.admit_observation(row)
        if issues:
            self.abort_episode("control_sources: " + "; ".join(issues))
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
            # 停止后才保存迟到的观测。
            self.abort_episode("late_observation", row=row)
            return
        self.tick_index += 1
        self.execute_control_step(row, self.next_tick_ns, feedback_until_ns)

    def close(self):
        self.end_episode(
            True, "interrupted", run_end_reason=RunEndReason.RUNTIME_SHUTDOWN, announce=False,
        )

    def run(self):
        while self.runtime_state.running and not self.runtime_state.quit_requested:
            self.robot.arm.check_error()
            if self.phase is TeleopPhase.IDLE:
                self.robot.service_idle()
            else:
                self.robot.service_feedback()
            if not self.keyboard.healthy:
                self.runtime_state.request_emergency_stop()
            if self.runtime_state.estop_request or self.runtime_state.error_state:
                self.abort_episode(
                    "estop" if self.runtime_state.estop_request else "runtime_failure"
                )
                break
            if self.recorder is not None:
                self.recorder.check_error()

            timeout = 0.005
            if self.phase is TeleopPhase.ACTIVE:
                due = self.next_tick_ns
                timeout = (
                    0.0
                    if due is None
                    else min(
                        timeout,
                        max(0.0, (due - time.monotonic_ns()) / 1e9),
                    )
                )
            epoch = self.runtime_state.run_id
            self.handle_commands(self.keyboard.poll(timeout=timeout), epoch)
            if (
                self.runtime_state.estop_request
                or self.runtime_state.error_state
                or not self.runtime_state.running
                or self.runtime_state.quit_requested
            ):
                break
            if self.phase is not TeleopPhase.IDLE:
                self.run_control_tick()

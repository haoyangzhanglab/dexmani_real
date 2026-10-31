"""遥操作控制循环与单段示教录制。"""

import json
import logging
import sys
import time
from dataclasses import asdict
from enum import Enum
from typing import cast

from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import (
    EpisodeRecorder,
    RecordingError,
)
from dexmani_real.robot.commands import OperationCancelled, RobotCommand
from dexmani_real.runtime.home import home_robot
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState, device_failure_reason
from dexmani_real.utils.log import error_detail
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)


def failure_reason(exc):
    if isinstance(exc, RecordingError):
        return RunEndReason.RECORDING_FAILURE
    return device_failure_reason(exc) or RunEndReason.RUNTIME_FAILURE


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
        sensors,
        controller,
        home_planner,
        recording_dir,
        keyboard,
        camera_snapshot=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.robot = robot
        self.sensors = sensors
        self.control_needs = ObservationNeeds(vr=True)
        self.record_needs = (
            ObservationNeeds(camera_fields=("rgb", "depth"), required_tactile=("aggregate", "dense"))
            if recording_dir is not None else ObservationNeeds()
        )
        self.controller = controller
        self.home_planner = home_planner
        self.recording_dir = recording_dir
        # 会话配置固定；连接设备前生成录制快照，避免分段开始时才发现序列化错误。
        self.runtime_config_json = (
            json.dumps(asdict(runtime), ensure_ascii=False, allow_nan=False)
            if recording_dir is not None else None
        )
        self.task_label = task_label
        self.camera_snapshot = camera_snapshot
        self.keyboard = keyboard

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
        self.termination_details.append(error_detail(stage, exc))

    def stop_control(self, reason, *, run_end_reason=RunEndReason.OPERATOR):
        with self.runtime_state.motion_lock:
            self.request_episode_end(reason, run_end_reason=run_end_reason, failed=False)
            stopped_epoch = self.runtime_state.run_id
        if self.robot.stop_required:
            self.robot.stop()
        self.runtime_state.acknowledge_stop(stopped_epoch)

    def end_episode(
        self, save, reason, *, run_end_reason=RunEndReason.OPERATOR,
    ):
        if not save:
            self.discard_requested = True
        failure = None
        try:
            self.stop_control(reason, run_end_reason=run_end_reason)
        except BaseException as exc:
            self.record_error("stop", exc)
            failure = exc
        self.controller.clear()

        try:
            self.finish_recording(save)
        except BaseException as exc:
            self.record_error("recording_cleanup", exc)
            failure = failure or exc
        self.phase = TeleopPhase.IDLE
        self.run_id = None
        if failure is not None:
            raise failure
        return True

    def finish_recording(self, save):
        recorder = self.recorder
        if recorder is None or not recorder.start_attempted:
            self.recorder = None
            return
        reason = self.pending_termination_reason
        published = recorder.finish(
            save=save and not self.discard_requested,
            reason=reason,
            details=self.termination_details,
            technical_status=(
                "normal" if not self.capture_failed and reason in {"operator", "quit", "duration_complete"}
                else "failed"
            ),
        )
        # finish 失败时保留 recorder，让会话收尾继续持有未释放的 writer。
        self.recorder = None
        logger.info("本段%s。", "已保存" if published is not None else "已丢弃")

    def request_episode_end(
        self, reason, *, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, failed=True,
    ):
        """先撤权并保留结束意图，资源收尾由 run_episode 的边界执行。"""
        self.capture_failed |= failed
        with self.runtime_state.motion_lock:
            self.runtime_state.revoke_motion(reason=run_end_reason, expected_run_id=self.run_id)
            if self.pending_termination_reason is None:
                self.pending_termination_reason = (
                    self.runtime_state.run_ended_reason
                    if self.runtime_state.run_ended_id == self.run_id else run_end_reason
                ).name.lower()
                self.termination_details.append(dict(stage="termination", reason=reason))

    def poll_blocking_commands(self):
        self.robot.check()
        terminal = tuple(
            cmd for cmd in self.keyboard.poll(timeout=0)
            if cmd in ("esc", "q", "s", "d")
        )
        if terminal:
            self.handle_commands(terminal, self.runtime_state.run_id)
        return bool(terminal) or not self.runtime_state.requests_allow_motion()

    def preparation_valid(self):
        return self.phase is TeleopPhase.PREPARING and self.runtime_state.can_move(
            run_id=self.run_id, required_safety_state=SafetyState.ARMED,
        )

    def preparation_interrupted(self):
        self.poll_blocking_commands()
        if self.pending_termination_reason is not None:
            return True
        if self.preparation_valid():
            return False
        if self.runtime_state.stop_request:
            return True  # 停止回调已撤权，按键随后给出保存/丢弃意图。
        self.request_episode_end("preparation_revoked")
        return True

    def start_episode(self, expected_run_id):
        if self.phase is not TeleopPhase.IDLE:
            return
        if self.recorder is not None:
            raise RecordingError("previous capture still owns resources")
        with self.runtime_state.motion_lock:
            # 开始只能属于取出按键前的代际，不能越过稍后到达的 STOP。
            if (
                expected_run_id != self.runtime_state.run_id
                or not self.runtime_state.requests_allow_motion()
            ):
                return
            self.runtime_state.revoke_motion()
            self.run_id = self.runtime_state.run_id
            self.phase = TeleopPhase.PREPARING
        self.capture_failed = self.discard_requested = False
        self.pending_termination_reason = None
        self.termination_details = []
        self.robot.stop()
        if self.preparation_interrupted():
            return

        self.robot.reset_hand_feedback()
        needs = self.control_needs.union(self.record_needs)
        self.robot.read_initial_state(
            required_tactile=needs.required_tactile,
            cancel_requested=lambda: self.poll_blocking_commands() or not self.preparation_valid(),
        )
        if self.preparation_interrupted():
            return
        row = read_observation(self.sensors.channels, self.robot, needs=needs)
        issues, _ = self.admit_observation(row)
        if issues:
            self.request_episode_end("preparation_unavailable")
            return
        if self.preparation_interrupted():
            return

        # 首次求解在准备阶段完成，不占用第一帧的控制期限，也不提交预计算目标。
        started_ns = time.monotonic_ns()
        try:
            self.controller.reset(row)
            result = self.controller.compute_target(row)
            hand_rejection = self.controller.rejection_reason
        finally:
            elapsed_ms = (time.monotonic_ns() - started_ns) / 1e6
            self.termination_details.append(dict(stage="precompute", elapsed_ms=elapsed_ms))
            self.controller.clear()
        if self.preparation_interrupted():
            return
        logger.info("Teleop precompute run=%s: %.2f ms", self.run_id, elapsed_ms)
        if result is None or result.rejection_reason is not None or result.arm_qpos is None:
            detail = (
                result.rejection_reason if result is not None
                else hand_rejection or "hand target rejected"
            )
            self.request_episode_end(f"preparation_target_unavailable: {detail}")
            return

        if self.recording_dir is not None:
            metadata = snapshot_recording_metadata(
                self.camera_snapshot,
                tactile_reset_metadata=self.robot.hand.get_tactile_reset_metadata(),
                tactile_hold_max_age_s=self.runtime.hand.feedback_max_age_s,
            )
            metadata.update(
                runtime_config_json=self.runtime_config_json,
                vr_to_robot_rotation=self.controller.rotation.as_matrix().tolist(),
                teleop_run_id=self.run_id,
            )
            if self.controller.hand_retargeter is not None:
                metadata["hand_retargeting_calibration_json"] = json.dumps(
                    self.controller.hand_retargeter.calibration_metadata(), allow_nan=False,
                )
            self.recorder = EpisodeRecorder(
                self.recording_dir, control_hz=self.runtime.teleop.control_hz,
            )
            self.recorder.start(
                task_label=self.task_label, collection_source="teleop",
                metadata=metadata,
            )
        if self.preparation_interrupted():
            return
        self.robot.prepare_streaming(self.run_id)
        # 锚点与反馈必须来自 prepare 完成之后。
        self.prepared_after_ns = time.monotonic_ns()
        self.preparation_interrupted()

    def handle_operator_command(self, cmd, *, expected_run_id=None):
        event = dict(
            stage="operator", key=cmd, phase=self.phase.value,
            run_id=self.run_id, timestamp_ns=time.monotonic_ns(),
        )
        if self.recorder is not None:
            self.termination_details.append(event)
        if cmd == "esc":
            self.runtime_state.request_emergency_stop()
            return True
        if (
            self.runtime_state.estop_request or self.runtime_state.error_state
            or self.runtime_state.quit_requested or not self.runtime_state.running
        ):
            return True

        if cmd == "q":
            self.runtime_state.request_quit()
            if self.phase is not TeleopPhase.IDLE:
                self.request_episode_end("quit", run_end_reason=RunEndReason.QUIT, failed=False)
            return True
        if cmd in ("s", "d"):
            if cmd == "d":
                self.discard_requested = True
            self.runtime_state.request_stop()
            if self.phase is not TeleopPhase.IDLE:
                self.request_episode_end(
                    "stop" if cmd == "s" else "discard", run_end_reason=RunEndReason.OPERATOR,
                    failed=False,
                )
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
            if self.phase is not TeleopPhase.IDLE or self.recorder is not None:
                logger.warning("HOME requires idle without an active capture")
                return True
            if not self.poll_blocking_commands():
                logger.info("正在回初始位…")
                result = home_robot(
                    self.runtime_state, self.robot, self.home_planner,
                    cancel_requested=self.poll_blocking_commands,
                )
                if result.ok:
                    logger.info("已回初始位。")
                elif result.interrupted:
                    logger.info("HOME cancelled: %s", result.reason)
                else:
                    raise RuntimeError(f"HOME failed: {result.reason}")
            self.poll_blocking_commands()
            return True
        elif cmd == "b":
            if self.phase is TeleopPhase.IDLE:
                self.run_episode(expected_run_id)
            return True
        return False

    def handle_commands(self, commands, expected_run_id):
        for terminal in ("esc", "q", "s", "d"):
            if terminal in commands:
                commands = [terminal]
                break
        for cmd in commands:
            previous_phase, previous_run = self.phase, self.run_id
            try:
                handled = self.handle_operator_command(
                    cmd,
                    expected_run_id=expected_run_id,
                )
            finally:
                # 停止按键先撤权，再由本段边界执行资源收尾。
                logger.info("Teleop key=%s phase=%s->%s run=%s->%s",
                            cmd.upper(), previous_phase.value, self.phase.value, previous_run, self.run_id)
            if handled:
                break

    def try_begin(self, row):
        if self.preparation_interrupted():
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
            if not self.preparation_interrupted():
                self.request_episode_end("begin_unavailable")
            return
        self.phase = TeleopPhase.ACTIVE
        self.capture_started_s = time.monotonic()
        self.tick_origin_ns = None
        self.tick_index = 0
        self.termination_details.append(dict(
            stage="teleop_anchor", run_id=self.run_id,
            observation_ns=int(row.observation_timestamp_ns),
            arm_qpos=row.arm["qpos"][0].tolist(),
            hand_qpos=row.hand["qpos"][0].tolist() if row.hand is not None else None,
            vr_wrist_pos=row.vr["wrist_pos"].tolist(),
            vr_wrist_quat_wxyz=row.vr["wrist_quat_wxyz"].tolist(),
            vr_recv_ts_ns=int(row.vr["recv_ts_ns"]),
        ))

    def record_control_timing(
        self, row, epoch, deadline_ns, compute_start_ns, compute_end_ns,
        submit_start_ns, submit_end_ns, previous_hand_observation_id,
    ):
        """仅失败时编码诊断；所有开始/完成时点已在实际事件处采集。"""
        ended_ns = time.monotonic_ns()
        timing = dict(
            stage="control_timing", run_id=epoch, tick=self.tick_index,
            control_ns=int(row.control_timestamp_ns), observation_ns=int(row.observation_timestamp_ns),
            deadline_ns=int(deadline_ns), compute_start_ns=compute_start_ns, ended_ns=ended_ns,
        )
        for name, stamp in (("compute_end_ns", compute_end_ns), ("submit_start_ns", submit_start_ns),
                            ("submit_end_ns", submit_end_ns)):
            if stamp is not None:
                timing[name] = stamp
        if (
            self.controller.hand_retargeter is not None
            and self.controller.hand_observation_id != previous_hand_observation_id
        ):
            timing["hand_retargeting"] = self.controller.hand_retargeter.last_diagnostics.copy()
        self.termination_details.append(timing)

    def execute_control_step(self, row, deadline_ns, feedback_until_ns):
        epoch = cast(int, self.run_id)  # 仅 ACTIVE 阶段调用，保留本段的授权 ID 快照。
        deadline_ns = min(deadline_ns, feedback_until_ns)
        command = None
        reason = None
        cancelled = False
        compute_start_ns = time.monotonic_ns()
        compute_end_ns = submit_start_ns = submit_end_ns = None
        previous_hand_observation_id = self.controller.hand_observation_id
        operator_end = False
        try:
            result = self.controller.compute_target(row)
            compute_end_ns = time.monotonic_ns()
            if result is None or result.rejection_reason is not None or result.arm_qpos is None:
                detail = (
                    result.rejection_reason if result is not None
                    else self.controller.rejection_reason or "hand target rejected"
                )
                reason = f"target_unavailable: {detail}"
            else:
                if self.recorder is not None:
                    self.recorder.check_error()
                try:
                    if not self.runtime_state.can_move(run_id=epoch):
                        raise OperationCancelled("teleop cancelled before submission")
                    if time.monotonic_ns() >= deadline_ns:
                        reason = "teleop expired before submission"
                    else:
                        command = RobotCommand(epoch, result.arm_qpos, result.hand_qpos)
                        submit_start_ns = time.monotonic_ns()
                        self.robot.send_action(
                            command,
                            valid_until_ns=deadline_ns,
                            execute_until_ns=int(row.vr["recv_ts_ns"])
                            + int(self.runtime.teleop.vr_stale_threshold_s * 1e9),
                        )
                        submit_end_ns = time.monotonic_ns()
                except OperationCancelled as exc:
                    cancelled = True
                    reason = f"control_interrupted: {exc}"

            if reason is None or cancelled:
                # 授权检查和内部状态提交同属一个临界区；锁内没有设备或写盘 I/O。
                with self.runtime_state.motion_lock:
                    if not self.runtime_state.can_move(run_id=epoch):
                        reason, cancelled = reason or "motion_revoked", True
                    elif time.monotonic_ns() >= deadline_ns:
                        reason, cancelled = "submission_deadline_expired", False
                    elif reason is None:
                        self.controller.commit(command, eef_pose=result.eef_pose)
            if cancelled:
                # 撤权不能掩盖同时发生的设备/服务故障。
                self.robot.check()
                with self.runtime_state.motion_lock:
                    ended = self.runtime_state.run_ended_reason
                    operator_end = (
                        self.runtime_state.run_ended_id == epoch
                        and ended in (RunEndReason.OPERATOR, RunEndReason.QUIT)
                        and not self.runtime_state.estop_request
                        and not self.runtime_state.error_state
                    )
                if operator_end:
                    # 回调先撤权、后入队；由按键决定保存/丢弃，未完成的行不补录。
                    return
            if reason is not None:
                self.record_control_timing(
                    row, epoch, deadline_ns, compute_start_ns, compute_end_ns,
                    submit_start_ns, submit_end_ns, previous_hand_observation_id,
                )
                self.request_episode_end(reason)
            elif self.tick_index == 1:
                logger.info("Teleop first target submitted run=%s", epoch)
        finally:
            primary_error = sys.exc_info()[1]
            if primary_error is not None:
                self.runtime_state.revoke_motion(
                    reason=failure_reason(primary_error), expected_run_id=epoch,
                )
                try:
                    self.record_control_timing(
                        row, epoch, deadline_ns, compute_start_ns, compute_end_ns,
                        submit_start_ns, submit_end_ns, previous_hand_observation_id,
                    )
                except BaseException:
                    logger.exception("teleop failure timing also failed")
            try:
                if self.recorder is not None and not operator_end:
                    self.recorder.append(row, command)
            except BaseException as exc:
                if primary_error is None:
                    raise
                self.termination_details.append(error_detail("recording_frame", exc))
                logger.exception("teleop frame recording also failed")

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
        if self.phase not in (TeleopPhase.PREPARING, TeleopPhase.ACTIVE):
            return
        if self.runtime_state.stop_request:
            return  # 等待已经撤权、尚未入队的停止按键。
        if self.phase is TeleopPhase.PREPARING and self.preparation_interrupted():
            return
        if (
            self.phase is TeleopPhase.ACTIVE
            and self.recorder is not None
            and self.recorder.is_recording
            and time.monotonic() - self.capture_started_s
            >= self.runtime.collection.max_duration_s
        ):
            self.request_episode_end(
                "max_record_duration", run_end_reason=RunEndReason.DURATION_COMPLETE, failed=False,
            )
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
                self.request_episode_end("missed_tick")
                return
        row = read_observation(
            self.sensors.channels, self.robot, needs=self.control_needs.union(self.record_needs),
        )
        issues, feedback_until_ns = self.admit_observation(row)
        if issues:
            self.request_episode_end("control_sources: " + "; ".join(issues))
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
            self.request_episode_end("late_observation")
            if self.recorder is not None:
                self.recorder.append(row)
            return
        self.tick_index += 1
        self.execute_control_step(row, self.next_tick_ns, feedback_until_ns)

    def close(self):
        if self.recorder is not None and self.recorder.finish_request is not None:
            if self.recorder.writer_pending:
                raise RecordingError("recording worker has not exited; capture remains unpublished")
            return
        if self.phase is not TeleopPhase.IDLE or self.recorder is not None:
            self.end_episode(
                True, "interrupted", run_end_reason=RunEndReason.RUNTIME_SHUTDOWN,
            )

    def run_episode(self, expected_run_id):
        """本段准备/控制栈先返回，再在唯一边界停止和收尾。"""
        try:
            self.start_episode(expected_run_id)
            while self.phase in (TeleopPhase.PREPARING, TeleopPhase.ACTIVE):
                self.robot.arm.check_error()
                if self.runtime_state.stop_request:
                    self.robot.service_idle()
                else:
                    self.robot.service_feedback()
                if not self.keyboard.check_health():
                    self.runtime_state.request_emergency_stop()
                if self.runtime_state.estop_request or self.runtime_state.error_state:
                    self.request_episode_end(
                        "estop" if self.runtime_state.estop_request else "runtime_failure",
                        run_end_reason=(
                            RunEndReason.ESTOP if self.runtime_state.estop_request
                            else RunEndReason.RUNTIME_FAILURE
                        ),
                    )
                if self.pending_termination_reason is not None:
                    break
                if self.recorder is not None:
                    self.recorder.check_error()
                timeout = 0.005
                if self.phase is TeleopPhase.ACTIVE and self.next_tick_ns is not None:
                    timeout = min(timeout, max(0.0, (self.next_tick_ns - time.monotonic_ns()) / 1e9))
                self.handle_commands(self.keyboard.poll(timeout=timeout), self.runtime_state.run_id)
                if self.pending_termination_reason is not None:
                    break
                if not self.runtime_state.running or self.runtime_state.quit_requested:
                    break
                self.run_control_tick()
        except OperationCancelled:
            self.poll_blocking_commands()
            if self.pending_termination_reason is None:
                self.request_episode_end("preparation_revoked", failed=False)
        except BaseException as exc:
            self.record_error("episode", exc)
            self.request_episode_end(str(exc), run_end_reason=failure_reason(exc))
            raise
        finally:
            primary_failure = sys.exc_info()[0] is not None
            try:
                if self.phase is not TeleopPhase.IDLE:
                    self.end_episode(
                        True, self.pending_termination_reason or "interrupted",
                        run_end_reason=RunEndReason.RUNTIME_SHUTDOWN,
                    )
            except BaseException:
                if not primary_failure:
                    raise
                logger.exception("teleop episode cleanup also failed")
            finally:
                self.keyboard.discard_events("b", "h", "t")

    def run(self):
        while self.runtime_state.running and not self.runtime_state.quit_requested:
            self.robot.arm.check_error()
            self.robot.service_idle()
            if not self.keyboard.check_health():
                self.runtime_state.request_emergency_stop()
            if self.runtime_state.estop_request or self.runtime_state.error_state:
                break
            if not self.robot.stop_required:
                self.runtime_state.acknowledge_stop(self.runtime_state.run_id)
            self.handle_commands(self.keyboard.poll(timeout=0.005), self.runtime_state.run_id)

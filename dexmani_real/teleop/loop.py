"""遥操作控制循环与单段示教录制。"""

import json
import logging
import sys
import time
from enum import Enum
from typing import cast

from dexmani_real.config.control import HomeConfigurationError
from dexmani_real.control.homing import HomePathRejected
from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import (
    EpisodeRecorder,
    RecordingError,
)
from dexmani_real.robot.commands import OperationCancelled, RobotCommand
from dexmani_real.runtime.home import build_home_planner, home_robot
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import RunEndReason, SafetyState, device_failure_reason
from dexmani_real.utils.log import attach_cleanup_errors, error_detail
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
    PAUSED = "paused"
    QUIT_PENDING = "quit_pending"


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
        recording_dir,
        max_duration_s,
        runtime_config_json,
        keyboard,
        audio,
        camera_snapshot=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.robot = robot
        self.sensors = sensors
        record_needs = (
            ObservationNeeds(camera_fields=("rgb", "depth"), required_tactile=("aggregate", "dense"))
            if recording_dir is not None else ObservationNeeds()
        )
        self.needs = ObservationNeeds(vr=True).union(record_needs)
        self.controller = controller
        self.home_planner = None
        self.recording_dir = recording_dir
        self.max_duration_s = max_duration_s
        self.runtime_config_json = runtime_config_json
        self.task_label = task_label
        self.camera_snapshot = camera_snapshot
        self.keyboard = keyboard
        self.audio = audio
        self.pending_commands = []

        self.phase = TeleopPhase.IDLE
        self.resuming = False
        self.run_id = None
        self.tick_origin_ns = None
        self.tick_index = 0
        self.control_dt_ns, _ = sampling_clock_ns(runtime.teleop.control_hz)

        self.recorder = None
        self.capture_started_s = 0.0
        self.capture_failed = False
        self.discard_requested = False
        self.pending_termination_reason = None
        self.termination_details = []
        self.soft_timeout_count = 0
        self.last_soft_timeout = None

    @property
    def next_tick_ns(self):
        if self.tick_origin_ns is None:
            return None
        return self.tick_origin_ns + self.tick_index * self.control_dt_ns

    def request_discard(self):
        self.discard_requested = True
        self.runtime_state.request_stop()

    def request_pause(self):
        # 键盘线程只撤权；暂停确认和录制决策仍由控制线程执行。
        if self.phase in (TeleopPhase.PREPARING, TeleopPhase.ACTIVE):
            self.runtime_state.request_stop()

    def pause_episode(self, *, quitting=False):
        self.runtime_state.revoke_motion(
            reason=RunEndReason.QUIT if quitting else RunEndReason.OPERATOR,
            expected_run_id=self.run_id,
        )
        self.robot.stop()
        self.controller.clear()
        self.phase = TeleopPhase.QUIT_PENDING if quitting else TeleopPhase.PAUSED
        self.audio.play("quit" if quitting else "pause")
        logger.info(
            "已暂停；再按 Q 保存退出，C/B 继续，S 保存本段，D 丢弃，H 保存后回初始位。"
            if quitting else "已暂停；C/B 保存前段并继续，S 保存本段，D 丢弃，H 保存后回初始位。"
        )

    def record_error(self, stage, exc):
        if isinstance(exc, KeyboardInterrupt):
            self.runtime_state.request_emergency_stop()
        else:
            self.runtime_state.mark_error()
        if self.recorder is not None and self.recorder.published_path is None:
            self.capture_failed = True
        self.termination_details.append(error_detail(stage, exc))

    def stop_control(self, reason, *, run_end_reason=RunEndReason.OPERATOR):
        self.request_episode_end(reason, run_end_reason=run_end_reason, failed=False)
        self.robot.stop()

    def end_episode(
        self, reason, *, run_end_reason=RunEndReason.OPERATOR, primary_failure=None,
    ):
        failure = None
        try:
            self.stop_control(reason, run_end_reason=run_end_reason)
        except BaseException as exc:
            self.record_error("stop", exc)
            failure = primary_failure if primary_failure is not None else exc
            attach_cleanup_errors(failure, [("episode_stop", exc)])
        self.controller.clear()

        if self.capture_failed:
            self.audio.play(
                "emergency" if self.runtime_state.estop_request or self.runtime_state.error_state else "end"
            )
        if self.soft_timeout_count:
            summary = {
                "stage": "soft_timeouts", "count": self.soft_timeout_count, "last": self.last_soft_timeout,
            }
            logger.info("Teleop soft timeouts: %s", summary)
            if self.recorder is not None:
                self.termination_details.append(summary)
            self.soft_timeout_count = 0
            self.last_soft_timeout = None
        try:
            self.finish_recording()
        except BaseException as exc:
            self.record_error("recording_cleanup", exc)
            if failure is None:
                failure = primary_failure if primary_failure is not None else exc
            attach_cleanup_errors(failure, [("recording_cleanup", exc)])
        self.phase = TeleopPhase.IDLE
        self.run_id = None
        self.pending_commands = [
            event for event in self.pending_commands if event.key in ("esc", "q", "s", "d")
        ]
        self.keyboard.discard_events("b", "c", "h", "t")
        if failure is not None:
            raise failure

    def finish_recording(self):
        recorder = self.recorder
        if recorder is None or not recorder.start_attempted:
            self.recorder = None
            return
        reason = self.pending_termination_reason
        published = recorder.finish(
            save=not self.discard_requested,
            reason=reason,
            details=self.termination_details,
            technical_status=(
                "normal" if not self.capture_failed and reason in {"operator", "quit", "duration_complete"}
                else "failed"
            ),
        )
        # finish 失败时保留 recorder，让会话收尾继续持有未释放的 writer。
        self.recorder = None
        if published is not None:
            logger.info("本段已保存。")
            if not self.capture_failed:
                self.audio.play("save", interrupt=False)
        elif recorder.submitted_frames > 0:
            logger.info("本段已丢弃。")
            if not self.capture_failed:
                self.audio.play("discard", interrupt=False)

    def request_episode_end(
        self, reason, *, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, failed=True,
    ):
        """先撤权并保留结束意图，资源收尾由 run_episode 的边界执行。"""
        self.capture_failed |= failed
        self.runtime_state.revoke_motion(reason=run_end_reason, expected_run_id=self.run_id)
        with self.runtime_state.motion_lock:
            if self.pending_termination_reason is None:
                self.pending_termination_reason = (
                    self.runtime_state.run_ended_reason
                    if self.runtime_state.run_ended_id == self.run_id else run_end_reason
                ).name.lower()
                self.termination_details.append(dict(stage="termination", reason=reason))

    def poll_blocking_commands(self, *, cancellation=None):
        self.robot.check(operation_error=cancellation)
        commands = self.keyboard.poll(timeout=0)
        if self.phase in (TeleopPhase.PREPARING, TeleopPhase.ACTIVE):
            # 准备或提交尚未返回时只收集意图；不在旧帧的 finally 中更换 recorder。
            self.pending_commands.extend(
                event for event in commands if event.key in ("esc", "q", "s", "d", "c", "h")
            )
            return (
                any(event.key in ("esc", "q", "s", "d", "c") for event in self.pending_commands)
                or not self.runtime_state.requests_allow_motion()
                # 快停止回调可能刚撤权，按键还未入队；下一次主循环再消费意图。
                or (self.runtime_state.stop_signal.is_set() and self.runtime_state.run_id != self.run_id)
            )
        terminal = tuple(
            event for event in commands
            if event.key in ("esc", "q", "s", "d")
        )
        if terminal:
            self.handle_commands(terminal)
        return bool(terminal) or not self.runtime_state.requests_allow_motion()

    def preparation_valid(self):
        return self.phase is TeleopPhase.PREPARING and self.runtime_state.can_move(
            run_id=self.run_id, required_safety_state=SafetyState.ARMED,
        )

    def get_home_planner(self):
        if self.home_planner is None:
            self.home_planner = build_home_planner(self.runtime.environment, self.runtime.arm)
        return self.home_planner

    def preparation_interrupted(self):
        if self.poll_blocking_commands():
            return True
        if self.pending_termination_reason is not None:
            return True
        if self.preparation_valid():
            return False
        if self.runtime_state.stop_signal.is_set():
            return True  # 停止回调已撤权，按键随后给出保存/丢弃意图。
        self.request_episode_end("preparation_revoked")
        return True

    def start_episode(self, expected_run_id):
        if self.phase is not TeleopPhase.IDLE:
            return
        if self.recorder is not None:
            raise RecordingError("previous capture still owns resources")
        with self.runtime_state.motion_lock:
            # 只接受入队时的身份；不能给 STOP 前的意图重新授权。
            if (
                expected_run_id != self.runtime_state.run_id
                or not self.runtime_state.requests_allow_motion()
            ):
                return
            self.runtime_state.revoke_motion()
            self.run_id = self.runtime_state.run_id
            self.phase = TeleopPhase.PREPARING
        logger.info("Teleop preparing run=%s", self.run_id)
        self.capture_failed = self.discard_requested = False
        self.pending_termination_reason = None
        self.termination_details = []
        self.soft_timeout_count = 0
        self.last_soft_timeout = None
        try:
            if self.preparation_interrupted():
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
            if self.preparation_interrupted():
                return
            self.robot.service_feedback()
            row = read_observation(self.sensors.channels, self.robot, needs=self.needs)
            issues = self.admit_observation(row)
            if issues:
                self.request_episode_end("preparation_sources: " + "; ".join(issues))
                return
            self.try_begin(row)
        except OperationCancelled as exc:
            if not self.poll_blocking_commands(cancellation=exc):
                self.request_episode_end("preparation_revoked", failed=False)

    def handle_operator_command(self, key_event):
        cmd = key_event.key
        expected_run_id = key_event.run_id
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
            or self.runtime_state.quit_requested
        ):
            return True

        if cmd == "q":
            if self.phase in (TeleopPhase.IDLE, TeleopPhase.QUIT_PENDING):
                self.runtime_state.request_quit()
                if self.phase is not TeleopPhase.IDLE:
                    self.request_episode_end("quit", run_end_reason=RunEndReason.QUIT, failed=False)
            else:
                self.pause_episode(quitting=True)
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
            if not self.runtime.hand.enabled:
                logger.info("T ignored: hand is disabled")
                return False
            if (
                self.phase is not TeleopPhase.IDLE or self.recorder is not None
                or not self.runtime_state.can_move(
                    run_id=expected_run_id, required_safety_state=SafetyState.ARMED,
                )
            ):
                logger.warning("T ignored: tactile reset requires idle, no capture or preparation")
                return False
            try:
                logger.info("正在请求触觉传感器复位，请保持手部无接触；S/Q/Esc 可中断后续请求。")
                self.robot.reset_tactile_sensors(
                    run_id=expected_run_id,
                    cancel_requested=self.poll_blocking_commands,
                    required_tactile=self.needs.required_tactile,
                )
            except OperationCancelled as exc:
                self.robot.check(operation_error=exc)
                logger.info("%s", exc)
            finally:
                self.keyboard.discard_events("t", "b")
            return True
        elif cmd == "h":
            with self.runtime_state.motion_lock:
                if expected_run_id != self.runtime_state.run_id:
                    logger.info("忽略已被停止指令撤销的 HOME。")
                    return True
                # 接受 H 时先撤销旧段，固定新代际贯穿保存和 HOME；后续停止仍可取消。
                self.runtime_state.revoke_motion(reason=RunEndReason.OPERATOR)
                expected_run_id = self.runtime_state.run_id
            if self.phase is not TeleopPhase.IDLE:
                self.end_episode("home")
            if not self.poll_blocking_commands():
                logger.info("正在回初始位…")
                self.audio.play("home", interrupt=False)
                try:
                    home_robot(
                        self.runtime_state, self.robot, self.get_home_planner,
                        run_id=expected_run_id,
                        cancel_requested=self.poll_blocking_commands,
                    )
                except OperationCancelled as exc:
                    self.robot.check(operation_error=exc)
                    logger.info("HOME cancelled: %s", exc)
                except (HomeConfigurationError, HomePathRejected) as exc:
                    self.robot.check(operation_error=exc)
                    logger.warning("本次 HOME 未完成（手部可能已回位）：%s", exc)
                else:
                    logger.info("已回初始位。")
                    self.audio.play("home_done", interrupt=False)
            self.poll_blocking_commands()
            return True
        elif cmd == "c" and self.phase in (TeleopPhase.PREPARING, TeleopPhase.ACTIVE):
            self.pause_episode()
            return True
        elif cmd in ("b", "c"):
            if self.phase is TeleopPhase.IDLE:
                if cmd == "b":
                    self.resuming = False
                    self.run_episode(expected_run_id)
            elif self.phase in (TeleopPhase.PAUSED, TeleopPhase.QUIT_PENDING):
                if not self.runtime_state.can_move(
                    run_id=expected_run_id, required_safety_state=SafetyState.ARMED,
                ):
                    return True
                self.end_episode("resume")
                self.resuming = True
                self.start_episode(expected_run_id)
            return True
        return False

    def handle_commands(self, commands):
        for terminal in ("esc", "q", "s", "d", "c", "h"):
            terminal_event = next((event for event in commands if event.key == terminal), None)
            if terminal_event is not None:
                commands = [terminal_event]
                break
        for event in commands:
            logger.info("Teleop key=%s phase=%s run=%s",
                        event.key.upper(), self.phase.value, self.run_id)
            handled = self.handle_operator_command(event)
            if handled:
                break

    def try_begin(self, row):
        if self.preparation_interrupted():
            return
        self.controller.reset(row)
        with self.runtime_state.motion_lock:
            started_ns = self.runtime_state.begin_motion() if self.preparation_valid() else None
        if started_ns is None:
            if not self.preparation_interrupted():
                self.request_episode_end("begin_unavailable")
            return
        self.phase = TeleopPhase.ACTIVE
        self.first_target_pending = True
        logger.info("Teleop active run=%s", self.run_id)
        self.capture_started_s = started_ns / 1e9
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
        submit_start_ns, submit_end_ns,
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
        if self.controller.hand_retargeter is not None:
            timing["hand_retargeting"] = self.controller.hand_retargeter.last_diagnostics.copy()
        self.termination_details.append(timing)

    def record_soft_timeout(self, stage, *, deadline_ns, observed_ns):
        self.soft_timeout_count += 1
        self.last_soft_timeout = {
            "stage": stage, "tick": self.tick_index,
            "deadline_ns": int(deadline_ns), "observed_ns": int(observed_ns),
        }

    def execute_control_step(self, row, tick_end_ns):
        epoch = cast(int, self.run_id)
        command = None
        compute_start_ns = time.monotonic_ns()
        compute_end_ns = submit_start_ns = submit_end_ns = None
        try:
            result = self.controller.compute_target(row, control_ns=compute_start_ns)
            compute_end_ns = time.monotonic_ns()
            if result.rejection_reason is not None or result.arm_qpos is None:
                self.record_control_timing(
                    row, epoch, tick_end_ns, compute_start_ns, compute_end_ns,
                    submit_start_ns, submit_end_ns,
                )
                self.request_episode_end(f"target_unavailable: {result.rejection_reason}")
                return
            if self.recorder is not None:
                self.recorder.check_error()
            if not self.runtime_state.can_move(run_id=epoch):
                raise OperationCancelled("teleop cancelled before submission")
            checked_ns = time.monotonic_ns()
            if checked_ns >= tick_end_ns:
                # 未发送的候选不推进滤波、关节或时间参考；下周期从最新输入继续。
                self.record_soft_timeout(
                    "compute", deadline_ns=tick_end_ns, observed_ns=checked_ns,
                )
                return
            command = RobotCommand(epoch, result.arm_qpos, result.hand_qpos)
            submit_start_ns = time.monotonic_ns()
            self.robot.send_action(
                command,
                expires_ns=submit_start_ns + int(self.runtime.teleop.vr_stale_threshold_s * 1e9),
            )
            submit_end_ns = time.monotonic_ns()
            self.controller.commit(command, eef_pose=result.eef_pose, submit_end_ns=submit_end_ns)
            if self.first_target_pending:
                logger.info("Teleop first target submitted run=%s", epoch)
                self.first_target_pending = False
                self.audio.play("resume" if self.resuming else "begin", interrupt=False)
        except OperationCancelled as exc:
            # 取消不能掩盖同时发生的 SDK/worker 故障；排队的按键随后决定保存意图。
            interrupted = self.poll_blocking_commands(cancellation=exc)
            if self.pending_termination_reason is None and not interrupted:
                self.request_episode_end(f"control_interrupted: {exc}")
        finally:
            primary_error = sys.exc_info()[1]
            if primary_error is not None:
                self.runtime_state.revoke_motion(
                    reason=failure_reason(primary_error), expected_run_id=epoch,
                )
                self.record_control_timing(
                    row, epoch, tick_end_ns, compute_start_ns, compute_end_ns,
                    submit_start_ns, submit_end_ns,
                )
            try:
                if self.recorder is not None:
                    self.recorder.append(row, command)
            except BaseException as exc:
                if primary_error is None:
                    raise
                attach_cleanup_errors(primary_error, [("recording_frame", exc)])
                self.termination_details.append(error_detail("recording_frame", exc))
                logger.exception("teleop frame recording also failed")

    def admit_observation(self, row):
        return check_observation(
            row, self.runtime, self.needs, require_hand=self.runtime.hand.enabled,
        )

    def run_control_tick(self):
        if self.phase is not TeleopPhase.ACTIVE:
            return
        if self.runtime_state.stop_signal.is_set():
            return  # 等待已经撤权、尚未入队的停止按键。
        if (
            self.recorder is not None
            and self.recorder.is_recording
            and time.monotonic() - self.capture_started_s
            >= self.max_duration_s
        ):
            self.request_episode_end(
                "max_record_duration", run_end_reason=RunEndReason.DURATION_COMPLETE, failed=False,
            )
            return
        now_ns = time.monotonic_ns()
        if self.tick_origin_ns is not None:
            if now_ns < self.next_tick_ns:
                return
            # 绝对网格上跳过错过的时点，不追赶历史控制决策。
            self.tick_index += (now_ns - self.next_tick_ns) // self.control_dt_ns
        row = read_observation(
            self.sensors.channels, self.robot, needs=self.needs,
        )
        issues = self.admit_observation(row)
        if issues:
            self.request_episode_end("control_sources: " + "; ".join(issues))
            return
        stamp = row.observation_timestamp_ns
        if self.tick_origin_ns is None:
            self.tick_origin_ns = stamp
        if stamp >= self.next_tick_ns + self.control_dt_ns:
            # 本次复制已错过提交机会；不映射、不提交，下个未来时点重新读。
            self.record_soft_timeout(
                "observation", deadline_ns=self.next_tick_ns + self.control_dt_ns, observed_ns=stamp,
            )
            self.tick_index += (stamp - self.next_tick_ns) // self.control_dt_ns + 1
            if self.recorder is not None:
                self.recorder.append(row)
            return
        self.tick_index += 1
        self.execute_control_step(row, self.next_tick_ns)

    def close(self):
        if self.recorder is not None and self.recorder.finish_request is not None:
            self.recorder.check_error()
            return
        if self.phase is not TeleopPhase.IDLE or self.recorder is not None:
            self.end_episode(
                "interrupted", run_end_reason=RunEndReason.RUNTIME_SHUTDOWN,
            )

    def service_episode(self, timeout):
        if self.runtime_state.stop_signal.is_set():
            self.robot.service_idle()
        else:
            self.robot.service_feedback()
        if self.runtime_state.estop_request or self.runtime_state.error_state:
            self.request_episode_end(
                "estop" if self.runtime_state.estop_request else "runtime_failure",
                run_end_reason=(
                    RunEndReason.ESTOP if self.runtime_state.estop_request
                    else RunEndReason.RUNTIME_FAILURE
                ),
            )
        if self.pending_termination_reason is not None:
            return False
        if self.recorder is not None:
            self.recorder.check_error()
        if self.phase is TeleopPhase.ACTIVE and self.next_tick_ns is not None:
            timeout = min(timeout, max(0.0, (self.next_tick_ns - time.monotonic_ns()) / 1e9))
        commands, self.pending_commands = self.pending_commands, []
        self.handle_commands((*commands, *self.keyboard.poll(timeout=timeout)))
        if (
            self.phase is TeleopPhase.ACTIVE and self.pending_termination_reason is None
            and self.runtime_state.stop_signal.is_set()
        ):
            # owner 的普通租期到期也会置 stop；不能只等永远不会出现的键盘事件。
            with self.runtime_state.motion_lock:
                operator_pending = (
                    self.runtime_state.run_ended_id == self.run_id
                    and self.runtime_state.run_ended_reason in (RunEndReason.OPERATOR, RunEndReason.QUIT)
                )
            if not operator_pending:
                self.request_episode_end("executor_stopped")
        return (
            self.pending_termination_reason is None
            and not self.runtime_state.quit_requested
        )

    def run_episode(self, expected_run_id):
        """顺序准备和控制；暂停保留录制，继续前先结束旧段。"""
        try:
            self.start_episode(expected_run_id)
            while self.phase is not TeleopPhase.IDLE:
                if not self.service_episode(0.005):
                    break
                self.run_control_tick()
        except BaseException as exc:
            self.record_error("episode", exc)
            self.request_episode_end(str(exc), run_end_reason=failure_reason(exc))
            raise
        finally:
            primary_failure = sys.exc_info()[1]
            try:
                if self.phase is not TeleopPhase.IDLE:
                    self.end_episode(
                        self.pending_termination_reason or "interrupted",
                        run_end_reason=RunEndReason.RUNTIME_SHUTDOWN,
                        primary_failure=primary_failure,
                    )
            except BaseException as exc:
                if primary_failure is None:
                    raise
                attach_cleanup_errors(primary_failure, [("episode_cleanup", exc)])
                logger.exception("teleop episode cleanup also failed")
            finally:
                self.keyboard.discard_events("b", "c", "h", "t")

    def run(self):
        while not self.runtime_state.quit_requested:
            self.robot.service_idle()
            if self.runtime_state.estop_request or self.runtime_state.error_state:
                break
            commands, self.pending_commands = self.pending_commands, []
            self.handle_commands((*commands, *self.keyboard.poll(timeout=0.005)))

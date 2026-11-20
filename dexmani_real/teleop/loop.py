"""遥操作控制循环与单段示教录制。"""

import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from importlib.metadata import version
from math import acos, degrees
from typing import cast

import numpy as np

from dexmani_real.control.actions import StartPoseRejected
from dexmani_real.control.ik import IKReferenceError
from dexmani_real.data.raw import episode_recording_metadata, robot_recording_parameters
from dexmani_real.data.recorder import (
    EpisodeRecorder,
    RecordingError,
    format_recording_result,
)
from dexmani_real.robot.commands import (
    CommandExpired,
    FeedbackUnavailable,
    HandCommandUnconfirmed,
    HomeTimeout,
    OperationCancelled,
    RobotCommand,
)
from dexmani_real.runtime.home import (
    HomeConfigurationError,
    HomePathRejected,
    home_robot,
)
from dexmani_real.runtime.observation import (
    RAW_RECORDING_NEEDS,
    Observation,
    ObservationNeeds,
    check_device_feedback,
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import INITIAL_STATE_TIMEOUT_S, RunEndReason
from dexmani_real.utils.log import error_detail
from dexmani_real.utils.timing import next_tick_ns

logger = logging.getLogger(__name__)


class TeleopLoop:
    def __init__(
        self,
        runtime_state,
        runtime,
        robot,
        *,
        control_dt_ns: int,
        task_label,
        sensors,
        controller,
        recording_dir,
        max_duration_s,
        keyboard,
        audio,
        camera_snapshot=None,
        timing=None,
    ):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.robot = robot
        self.sensors = sensors
        self.capture_needs = (
            replace(RAW_RECORDING_NEEDS, vr=True)
            if recording_dir is not None else ObservationNeeds(vr=True)
        )
        self.controller = controller
        self.recording_dir = recording_dir
        self.max_duration_s = max_duration_s
        self.task_label = task_label
        self.camera_snapshot = camera_snapshot
        self.keyboard = keyboard
        self.audio = audio
        self.timing = timing
        self.quit_presses = 0

        self.episode_cancel_epoch: int | None = None  # None 表示当前无活动段；Recorder 仍可能在收尾。
        self.episode_vr_epoch: int | None = None
        self.next_tick_ns: int | None = None
        self.control_dt_ns = control_dt_ns
        self.previous_wrist: tuple[int, int, np.ndarray, np.ndarray] | None = None

        self.recorder: EpisodeRecorder | None = None
        self.episode_deadline_ns: int | None = None
        self.discard_requested = False
        self.pending_termination_reason = None
        self.termination_details: list[dict[str, str]] = []
        self.session_failed = False  # 新段会重置结束状态，会话失败保留到退出。

    def record_error(self, stage, exc):
        if isinstance(exc, KeyboardInterrupt):
            self.runtime_state.request_emergency_stop()
        self.request_episode_end(stage, run_end_reason=RunEndReason.FAILED)
        self.termination_details.append(error_detail(stage, exc))
        self.audio.play("emergency")

    def end_episode(self):
        primary_failure = sys.exc_info()[1]
        failure = primary_failure
        # 仅在本次收尾保留引用，owner 释放后仍可读取当前段的最终录制结果。
        recorder = self.recorder
        try:
            self.robot.stop()
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            self.request_episode_end("stop_failed", run_end_reason=RunEndReason.FAILED)
            if failure is None:
                failure = exc
            else:
                self.record_error("stop", exc)
                logger.exception("teleop stop also failed")
        self.previous_wrist = None
        self.controller.clear()

        if self.runtime_state.estop_request:
            self.audio.play("emergency")
        try:
            if failure is None and not self.runtime_state.quit_requested and not self.runtime_state.estop_request:
                while self.recorder is not None:
                    self.poll_blocking_commands()
                    if self.runtime_state.quit_requested or self.runtime_state.estop_request:
                        break
                    self.robot.service_feedback()
                    if self.finish_recording(wait=False):
                        break
                    time.sleep(0.005)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            self.request_episode_end("recording_failed", run_end_reason=RunEndReason.FAILED)
            raise
        finally:
            self.episode_cancel_epoch = None
            self.episode_vr_epoch = None
        if failure is not None and primary_failure is None:
            raise failure
        if failure is None:
            raw = recorder.result() if recorder is not None else None
            level, text = format_recording_result(raw)
            status = "结束" if self.recorder is None else "控制结束，录制收尾未完成"
            logger.log(level, "Teleop 本段%s\nEnd reason: %s\nDetails: %s\n%s",
                       status, self.pending_termination_reason, self.termination_details, text)
        if (failure is None and self.recorder is None
                and not self.runtime_state.quit_requested and not self.runtime_state.estop_request):
            logger.info("停止流程已完成；遥操作控制锚点已清空。检查上方结束原因及录制结果后按 B 开始新段。")

    def finish_recording(self, *, wait=True) -> bool:
        """True 仅表示收尾完成；录制成败由 Recorder 结果报告。"""
        recorder = self.recorder
        if recorder is None:
            return True
        request = recorder.finish(
            save=not self.discard_requested, reason=self.pending_termination_reason, wait=wait,
        )
        if request is None:
            return False
        # 仅收尾完成后释放 owner；pending 或异常仍保留资源。
        self.recorder = None
        if recorder.error is not None:
            self.termination_details.append(error_detail("recording", recorder.error))
            self.request_episode_end("recording_failed", run_end_reason=RunEndReason.FAILED)
            return True
        if self.runtime_state.estop_request:
            self.audio.play("emergency")
            return True
        if request.reason not in {"operator", "quit", "duration_complete"}:
            return True
        if recorder.published_path is not None and recorder.recording_truncated_reason is None:
            self.audio.play("save")
        elif not request.save:
            logger.info("本段已丢弃。")
            self.audio.play("discard")
        return True

    def request_episode_end(
        self, reason, *, run_end_reason=RunEndReason.CANCELLED,
    ):
        """先请求停止并保留结束意图，资源收尾由 run_episode 的边界执行。"""
        self.runtime_state.stop_signal.set()
        if run_end_reason is RunEndReason.FAILED:
            self.session_failed = True
        if self.pending_termination_reason is None or (
            run_end_reason is RunEndReason.FAILED and self.pending_termination_reason != "failed"
        ):
            self.pending_termination_reason = run_end_reason.name.lower()
            self.termination_details.append({"stage": "termination", "reason": reason})
            if self.timing is not None:
                self.timing.add("episode_end", outcome=self.pending_termination_reason, reason=reason)
            logger.info("请求结束本段：%s（%s）。", self.pending_termination_reason, reason)

    def poll_blocking_commands(self):
        self.robot.check()
        commands = self.keyboard.poll(self.runtime_state, timeout=0)[0]
        terminal = tuple(event for event in commands if event in ("esc", "q", "s", "c", "d"))
        if terminal:
            self.handle_commands(terminal)
        return bool(terminal) or not self.runtime_state.requests_allow_motion()

    def preparation_valid(self):
        return (
            self.runtime_state.cancel_epoch == self.episode_cancel_epoch
            and self.runtime_state.requests_allow_motion()
        )

    def check_vr_connection(self) -> bool:
        if self.sensors.channels.vr_connection_epoch.value == self.episode_vr_epoch:
            return True
        self.request_episode_end("vr_disconnected")
        return False

    def preparation_interrupted(self):
        if (self.poll_blocking_commands() or self.pending_termination_reason is not None
                or not self.preparation_valid() or not self.check_vr_connection()):
            if self.pending_termination_reason is None:
                self.request_episode_end("preparation_cancelled")
            return True
        return False

    def start_observation(self, *, after_ns=None):
        until = time.monotonic_ns() + int(INITIAL_STATE_TIMEOUT_S * 1e9)
        while True:
            if self.preparation_interrupted():
                return None
            self.robot.service_feedback()
            row = read_observation(
                self.sensors.channels, self.robot, needs=self.capture_needs, include_visual_payloads=False,
            )
            issues = self.admit_observation(row)
            joints_ready = after_ns is None or (
                int(row.arm["timestamp_ns"][0]) >= after_ns
                and (not self.runtime.hand.enabled or row.hand is not None
                     and int(row.hand["timestamp_ns"][0]) >= after_ns)
            )
            if not issues and joints_ready and not self.robot.hand_feedback_pending:
                return row
            if time.monotonic_ns() >= until:
                if issues:
                    reason = "initial_observation_unavailable: " + "; ".join(issues)
                else:
                    reason = "initial_joint_feedback_unavailable"
                logger.warning("本段未开始：%s；等待停止与收尾。", reason)
                self.request_episode_end(reason)
                return None
            time.sleep(0.005)

    def start_episode(self, expected_cancel_epoch) -> bool:
        if self.recorder is not None:
            raise RuntimeError("previous capture still owns resources")
        with self.runtime_state.motion_lock:
            # 主线程接纳时捕获取消上下文；停止不能被随后准备覆盖。
            if (
                expected_cancel_epoch != self.runtime_state.cancel_epoch
                or not self.runtime_state.requests_allow_motion()
            ):
                return False
            self.runtime_state.stop_signal.set()
            self.episode_cancel_epoch = self.runtime_state.cancel_epoch
            self.episode_vr_epoch = self.sensors.channels.vr_connection_epoch.value
            self.discard_requested = False
        logger.info("Teleop preparing")
        self.pending_termination_reason = None
        self.termination_details = []
        try:
            if self.preparation_interrupted():
                return False

            self.robot.stop()
            if self.preparation_interrupted():
                return False
            # 各来源先于 writer 就绪；等待与资源初始化期间设备保持停止。
            if self.start_observation() is None or self.preparation_interrupted():
                return False
            if self.recording_dir is not None:
                runtime = self.runtime
                effective_parameters = {
                    "teleop": asdict(runtime.teleop),
                    "max_duration_s": self.max_duration_s,
                    **robot_recording_parameters(runtime),
                    "motion": asdict(runtime.motion),
                    "environment": {
                        "workspace": asdict(runtime.environment.workspace),
                        "eef_workspace_tolerance_m": runtime.environment.eef_workspace_tolerance_m,
                    },
                    "camera_config": asdict(runtime.camera),
                    "vr": asdict(runtime.vr),
                    "vr_to_robot_rotation": self.controller.rotation.as_matrix().tolist(),
                }
                retargeter = self.controller.hand_retargeter
                calibration = retargeter.calibration
                effective_parameters["hand_retargeting"] = asdict(retargeter.config)
                effective_parameters["dex_retargeting_version"] = version("dex-retargeting")
                effective_parameters["hand_retargeting_calibration"] = {
                    "bone_lengths": calibration.bone_lengths.tolist(),
                    "palm_width": calibration.palm_width,
                }
                metadata = episode_recording_metadata(
                    self.camera_snapshot, effective_parameters,
                    tactile_reset=self.robot.hand.tactile_reset_metadata,
                    tactile_calibration=self.robot.hand.tactile_calibration_metadata,
                )
                self.recorder = EpisodeRecorder(
                    self.recording_dir, control_hz=self.runtime.teleop.control_hz,
                    read_xhand_crc=self.robot.hand.crc_counts.copy,
                )
                self.recorder.start(
                    task_label=self.task_label, collection_source="teleop",
                    metadata=metadata,
                )
            # writer 可能阻塞超过反馈 TTL，准备运动前重新等候新反馈。
            if self.recorder is not None and self.start_observation(after_ns=time.monotonic_ns()) is None:
                return False
            if self.preparation_interrupted():
                return False

            def poll_prepare():
                if not self.poll_blocking_commands():
                    self.check_vr_connection()

            self.robot.prepare_streaming(self.episode_cancel_epoch, poll_operator=poll_prepare)
            if self.start_observation() is None:
                return False
            if self.preparation_interrupted():
                return False
            with self.runtime_state.motion_lock:
                started_ns = (
                    time.monotonic_ns()
                    if self.preparation_valid() and not self.runtime_state.stop_signal.is_set()
                    else None
                )
            if started_ns is None:
                if not self.preparation_interrupted():
                    self.request_episode_end("begin_unavailable")
                return False
            self.last_command: RobotCommand | None = None
            self.target_deadline_ns: int | None = None
            self.hold_reason: str | None = None
            self.episode_deadline_ns = (
                started_ns + int(self.max_duration_s * 1e9) if self.recording_dir is not None else None
            )
            self.next_tick_ns = None
            logger.info("Teleop active")
            return True
        except FeedbackUnavailable as exc:
            if not self.preparation_interrupted():
                self.request_episode_end(str(exc))
        except OperationCancelled:
            self.robot.check()
            if (self.check_vr_connection()
                    and self.runtime_state.cancel_epoch == self.episode_cancel_epoch
                    and self.runtime_state.requests_allow_motion()):
                raise
            self.poll_blocking_commands()
            if self.pending_termination_reason is None:
                self.request_episode_end("preparation_cancelled")
        return False

    def handle_quit_key(self) -> bool:
        """确认退出时在按键线程请求停止，并接纳 Q 事件。"""
        if not self.runtime_state.requests_allow_motion():
            return False
        self.quit_presses += 1
        if self.quit_presses == 2:
            self.runtime_state.request_quit()
            return True
        self.audio.play("quit")
        logger.info("准备退出遥操作，再按一次 Q 保存并退出。")
        return False

    def handle_operator_command(self, cmd: str):
        expected_cancel_epoch = None
        if cmd in {"b", "h", "t"}:
            expected_cancel_epoch = self.keyboard.accept_operation(self.runtime_state)
            if expected_cancel_epoch is None:
                return False
        if cmd == "esc":
            self.runtime_state.request_emergency_stop()
            self.audio.play("emergency")
            if self.episode_cancel_epoch is not None:
                self.request_episode_end("estop", run_end_reason=RunEndReason.ESTOP)
            return True
        if self.runtime_state.estop_request:
            return True

        if cmd == "q":
            if self.episode_cancel_epoch is not None:
                self.request_episode_end("quit", run_end_reason=RunEndReason.QUIT)
            return True
        if self.runtime_state.quit_requested:
            return True
        if cmd in ("s", "c", "d"):
            if self.episode_cancel_epoch is not None:
                self.request_episode_end(
                    "discard" if cmd == "d" else "stop", run_end_reason=RunEndReason.OPERATOR,
                )
        elif cmd == "t":
            return self._calibrate_tactile(expected_cancel_epoch)
        elif cmd == "h":
            return self._run_home(expected_cancel_epoch)
        elif cmd == "b":
            if (self.episode_cancel_epoch is None
                    and self.runtime_state.cancel_epoch == expected_cancel_epoch
                    and self.runtime_state.requests_allow_motion()):
                self.run_episode(expected_cancel_epoch)
                return True
            return False
        return False

    def _calibrate_tactile(self, expected_cancel_epoch):
        if not self.runtime.hand.enabled:
            logger.info("T ignored: hand is disabled")
            return False
        if (
            self.episode_cancel_epoch is not None or self.recorder is not None
            or self.runtime_state.cancel_epoch != expected_cancel_epoch
            or not self.runtime_state.requests_allow_motion()
        ):
            logger.warning("T ignored: tactile calibration requires idle, no capture or preparation")
            return False
        try:
            logger.info("正在校准触觉，请保持五指无接触；S 或 Esc 可中断，按两次 Q 退出。")
            self.robot.calibrate_tactile(
                no_contact=True,
                cancel_epoch=expected_cancel_epoch,
                cancel_requested=self.poll_blocking_commands,
            )
        except OperationCancelled as exc:
            self.robot.check()
            logger.info("%s", exc)
        return True

    def _run_home(self, expected_cancel_epoch):
        with self.runtime_state.motion_lock:
            if self.episode_cancel_epoch is not None or self.recorder is not None:
                logger.warning("H ignored: 请先按 S 结束本段，再回初始位。")
                return True
            if (self.runtime_state.cancel_epoch != expected_cancel_epoch
                    or not self.runtime_state.requests_allow_motion()):
                logger.info("忽略已被停止指令撤销的 HOME。")
                return False
            self.runtime_state.stop_signal.set()
            expected_cancel_epoch = self.runtime_state.cancel_epoch
        if not self.poll_blocking_commands():
            logger.info("正在回初始位…")
            self.audio.play("home")
            try:
                home_robot(
                    self.runtime_state, self.robot,
                    environment=self.runtime.environment,
                    cancel_epoch=expected_cancel_epoch,
                    cancel_requested=self.poll_blocking_commands,
                )
            except (FeedbackUnavailable, HomeTimeout) as exc:
                # 组合 HOME 已完成停止收尾。
                interrupted = self.poll_blocking_commands()
                if (interrupted or self.runtime_state.cancel_epoch != expected_cancel_epoch
                        or not self.runtime_state.requests_allow_motion()):
                    logger.info("HOME cancelled: %s", exc)
                else:
                    logger.warning("本次 HOME 未完成：%s", exc)
            except OperationCancelled as exc:
                self.robot.check()
                if (self.runtime_state.cancel_epoch == expected_cancel_epoch
                        and self.runtime_state.requests_allow_motion()):
                    raise
                logger.info("HOME cancelled: %s", exc)
            except (HomeConfigurationError, HomePathRejected) as exc:
                self.robot.check()
                logger.warning("本次 HOME 未完成：%s", exc)
            else:
                logger.info("已回初始位。")
                self.audio.play("home_done")
        self.poll_blocking_commands()
        return True

    def handle_commands(self, commands: Sequence[str]):
        if (self.pending_termination_reason is not None and self.recorder is not None
                and self.recorder.finish_request is not None):
            # 首次收尾请求已固定处置意图；此时只处理退出与急停。
            commands = tuple(event for event in commands if event in ("q", "esc"))
        if "d" in commands and self.episode_cancel_epoch is not None:
            self.discard_requested = True
        # 回调已置停止信号；先登记 D 意图，再按原优先级选择结束原因。
        for terminal in ("esc", "q", "d", "s", "c"):
            if terminal in commands:
                self.handle_operator_command(terminal)
                return
        for event in commands:
            if self.handle_operator_command(event):
                break

    def execute_control_step(self, row):
        timing = self.timing
        episode_cancel_epoch = self.episode_cancel_epoch
        submitted_command = None
        compute_start_ns = time.monotonic_ns()
        first_target = self.target_deadline_ns is None
        initializing_anchor = self.controller.anchor is None
        try:
            if initializing_anchor:
                self.controller.reset(row)
            result, ik_anchor, discarded_position_m = self.controller.compute_target(row, control_ns=compute_start_ns)
            if self.recorder is not None:
                self.recorder.check_error()
            if (self.runtime_state.cancel_epoch != episode_cancel_epoch
                    or not self.runtime_state.requests_allow_motion()
                    or self.runtime_state.stop_signal.is_set()):
                raise OperationCancelled("teleop cancelled before submission")
            issues = self.admit_observation(row)
            if issues:
                self.request_episode_end("control_sources: " + "; ".join(issues))
                return
            now_ns = time.monotonic_ns()
            if self.record_duration_expired(now_ns):
                return
            candidate_deadline_ns = row.control_timestamp_ns + int(self.runtime.teleop.target_max_age_s * 1e9)
            if (now_ns >= candidate_deadline_ns or
                    self.target_deadline_ns is not None and now_ns >= self.target_deadline_ns):
                self.request_episode_end("target_expired")
                return
            rejected = result.rejection_reason is not None or result.arm_qpos is None
            reason = (result.rejection_reason or "no_candidate") if rejected else None
            if reason != self.hold_reason:
                logger.info("Teleop %s", f"保持/等待：{reason}" if rejected else "恢复有效目标")
                self.hold_reason = reason
            if rejected:
                command = self.last_command
                if command is None:
                    if initializing_anchor:
                        self.controller.clear()
                    return
                expires_ns = cast(int, self.target_deadline_ns)
            else:
                command = RobotCommand(result.arm_qpos, result.hand_qpos)
                expires_ns = candidate_deadline_ns
            if not self.check_vr_connection():
                return
            arm_published_ns = None
            submit_start_ns = time.monotonic_ns() if timing is not None else None
            try:
                arm_published_ns = self.robot.send_action(command, expires_ns=expires_ns)
                submitted_command = command
            finally:
                submit_end_ns = time.monotonic_ns()
                if timing is not None:
                    timing.add(
                        "arm_submit",
                        timestamp_ns=submit_start_ns,
                        finished_ns=submit_end_ns,
                        control_ns=row.control_timestamp_ns,
                        arm_published_ns=arm_published_ns,
                        goal_qpos=command.arm_qpos,
                        hold=command.arm_hold,
                        accepted=submitted_command is not None,
                        rejection_reason=reason,
                    )
            self.controller.commit(
                command, ik_anchor=ik_anchor, arm_published_ns=cast(int, arm_published_ns),
                submit_end_ns=submit_end_ns, eef_pose=None if rejected else result.eef_pose,
                discarded_position_m=discarded_position_m,
            )
            self.last_command = command
            self.target_deadline_ns = expires_ns
            if first_target:
                logger.info("Teleop first target submitted")
                self.audio.play("begin")
        except StartPoseRejected as exc:
            logger.warning("%s", exc)
            self.request_episode_end(str(exc))
        except IKReferenceError as exc:
            self.request_episode_end(str(exc))
        except (FeedbackUnavailable, CommandExpired, HandCommandUnconfirmed) as exc:
            self.request_episode_end(str(exc))
        except OperationCancelled as exc:
            # 先检查 SDK/worker 故障，再即时登记已入队的终止意图。
            interrupted = self.poll_blocking_commands()
            if self.pending_termination_reason is None and not interrupted:
                self.handle_executor_stop(f"control_interrupted: {exc}")
        finally:
            primary_error = sys.exc_info()[1]
            if primary_error is not None:
                self.runtime_state.stop_signal.set()
            if self.recorder is not None and submitted_command is not None:
                try:
                    issue = self.recorder.append(row, submitted_command)
                    if issue is not None:
                        self.request_episode_end(issue)
                except BaseException as exc:
                    if primary_error is None:
                        raise
                    self.record_error("recording_append", exc)
                    logger.exception("teleop recording append also failed")

    def admit_observation(self, row):
        issues = check_device_feedback(row, self.runtime) + check_observation(
            row, self.runtime, self.capture_needs
        )
        if row.vr is not None and int(row.vr["connection_epoch"]) != self.episode_vr_epoch:
            issues.append("vr_waiting_for_current_connection")
        return issues

    def accept_wrist_observation(self, row: Observation) -> bool:
        """拦截腕姿跳变，并在首次下发前等待新腕包。"""
        vr = cast(dict, row.vr)
        control_ns = row.control_timestamp_ns
        recv_ns = int(vr["wrist_recv_ts_ns"])
        position = vr["wrist_pos"]
        quaternion = vr["wrist_quat_wxyz"]
        current = control_ns, recv_ns, position, quaternion
        if self.previous_wrist is None:
            self.previous_wrist = current
            return False

        previous_ns, previous_recv_ns, previous_position, previous_quaternion = self.previous_wrist
        interval_ns = control_ns - previous_ns
        # 漏拍后仍限制单次允许的位姿变化。
        budget_s = min(interval_ns, 1.5 * self.control_dt_ns) / 1e9
        displacement_m = float(np.linalg.norm(position - previous_position))
        dot = abs(float(np.dot(quaternion, previous_quaternion)))
        rotation_deg = degrees(2.0 * acos(min(dot, 1.0)))
        config = self.runtime.teleop
        if (
            displacement_m > config.vr_max_wrist_speed_m_s * budget_s
            or rotation_deg > config.vr_max_wrist_angular_speed_deg_s * budget_s
        ):
            self.request_episode_end("vr_pose_jump")
            sequence_id = int(vr["sequence_id"])
            logger.warning(
                "VR 腕姿突变，结束本段，序号 %d，位移 %.1f mm，转角 %.1f°，间隔 %.1f ms（限值使用 %.1f ms）",
                sequence_id, displacement_m * 1e3, rotation_deg, interval_ns / 1e6, budget_s * 1e3,
            )
            if self.timing is not None:
                self.timing.add(
                    "vr_pose_jump", timestamp_ns=control_ns, vr_sequence_id=sequence_id,
                    displacement_m=displacement_m, rotation_deg=rotation_deg,
                    interval_ns=interval_ns, budget_s=budget_s,
                )
            return False

        self.previous_wrist = current
        return self.target_deadline_ns is not None or recv_ns > previous_recv_ns

    def handle_executor_stop(self, unknown_reason="executor_stopped_unknown"):
        self.robot.check()
        self.request_episode_end(self.robot.arm.stop_reason or unknown_reason)

    def record_duration_expired(self, now_ns: int) -> bool:
        if self.episode_deadline_ns is None or now_ns < self.episode_deadline_ns:
            return False
        self.request_episode_end("max_record_duration", run_end_reason=RunEndReason.DURATION_COMPLETE)
        return True

    def run_control_tick(self):
        if self.pending_termination_reason is not None:
            return
        if not self.check_vr_connection():
            return
        now_ns = time.monotonic_ns()
        if self.record_duration_expired(now_ns) or self.runtime_state.stop_signal.is_set():
            return
        if self.next_tick_ns is not None and now_ns < self.next_tick_ns:
            return
        planned_ns = now_ns if self.next_tick_ns is None else self.next_tick_ns
        self.next_tick_ns = next_tick_ns(planned_ns, self.control_dt_ns, now_ns)
        timing = self.timing
        missed_slots = (self.next_tick_ns - planned_ns) // self.control_dt_ns - 1
        if missed_slots > 0:
            if self.recorder is not None:
                self.recorder.mark_gap()
            if timing is not None:
                timing.add("control_miss", timestamp_ns=now_ns, scheduled_ns=planned_ns, slots=missed_slots)
        if self.target_deadline_ns is not None and now_ns >= self.target_deadline_ns:
            if timing is not None:
                timing.add("control_skip", timestamp_ns=now_ns, scheduled_ns=planned_ns, reason="target_expired")
            self.request_episode_end("target_expired")
            return
        if self.robot.hand_feedback_pending:
            if timing is not None:
                timing.add("control_skip", timestamp_ns=now_ns, scheduled_ns=planned_ns, reason="hand_feedback_pending")
            if self.recorder is not None:
                self.recorder.mark_gap()
            return
        row = read_observation(self.sensors.channels, self.robot, needs=self.capture_needs)
        issues = self.admit_observation(row)
        if timing is not None:
            vr = row.vr
            timing.add(
                "control_tick",
                timestamp_ns=row.control_timestamp_ns,
                scheduled_ns=planned_ns,
                observation_ns=row.observation_timestamp_ns,
                arm_feedback_ns=row.arm["timestamp_ns"][0],
                wrist_recv_ns=vr["wrist_recv_ts_ns"] if vr is not None else None,
                vr_recv_ns=vr["recv_ts_ns"] if vr is not None else None,
                vr_sequence_id=vr["sequence_id"] if vr is not None else None,
                wrist_pos_m=vr["wrist_pos"] if vr is not None else None,
                wrist_quat_wxyz=vr["wrist_quat_wxyz"] if vr is not None else None,
                rejection_reason=", ".join(issues) if issues else None,
            )
        if issues:
            self.request_episode_end("control_sources: " + "; ".join(issues))
            return
        if self.accept_wrist_observation(row):
            self.execute_control_step(row)

    def service_episode(self, timeout):
        if not self.runtime_state.estop_request and not self.check_vr_connection():
            return False
        try:
            if self.runtime_state.stop_signal.is_set():
                self.robot.service_idle()
            else:
                self.robot.service_feedback()
        except FeedbackUnavailable as exc:
            self.request_episode_end(str(exc))
        if self.runtime_state.estop_request:
            self.request_episode_end(
                "estop", run_end_reason=RunEndReason.ESTOP,
            )
        if self.pending_termination_reason is not None:
            return False
        if self.recorder is not None:
            self.recorder.check_error()
        if self.next_tick_ns is not None:
            timeout = min(timeout, max(0.0, (self.next_tick_ns - time.monotonic_ns()) / 1e9))
        events, stopped = self.keyboard.poll(timeout=timeout, state=self.runtime_state)
        self.handle_commands(events)
        if self.pending_termination_reason is None:
            if self.runtime_state.cancel_epoch != self.episode_cancel_epoch:
                self.request_episode_end("operator_cancelled")
            elif stopped:
                self.handle_executor_stop()

        return (
            self.pending_termination_reason is None
            and not self.runtime_state.quit_requested
        )

    def run_episode(self, expected_cancel_epoch):
        """顺序准备、控制、停止与录制收尾；每次 B 都创建独立新段。"""
        try:
            if not self.start_episode(expected_cancel_epoch):
                return
            while self.service_episode(0.005):
                self.run_control_tick()
        except RecordingError:
            self.request_episode_end("recording_failed", run_end_reason=RunEndReason.FAILED)
            self.robot.check()
        except BaseException as exc:
            # 主错误由会话边界报告；阻塞停止前仍须锁存中断并标记本段失败。
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            self.request_episode_end("episode_failed", run_end_reason=RunEndReason.FAILED)
            raise
        finally:
            if self.episode_cancel_epoch is not None:
                if self.pending_termination_reason is None:
                    self.request_episode_end(
                        "interrupted",
                        run_end_reason=(
                            RunEndReason.QUIT if self.runtime_state.quit_requested
                            else RunEndReason.CANCELLED
                        ),
                    )
                self.end_episode()

    def run(self):
        while not self.runtime_state.quit_requested:
            self.robot.service_idle()
            if self.runtime_state.estop_request:
                break
            self.keyboard.allow_start()
            self.handle_commands(self.keyboard.poll(self.runtime_state, timeout=0.005)[0])

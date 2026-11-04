"""遥操作控制循环与单段示教录制。"""

import json
import logging
import sys
import time
from collections.abc import Sequence
from enum import Enum
from typing import cast

from dexmani_real.control.actions import HandStartRejected
from dexmani_real.data.raw import snapshot_recording_metadata
from dexmani_real.data.recorder import (
    EpisodeRecorder,
    RecordingError,
)
from dexmani_real.robot.commands import OperationCancelled, RobotCommand
from dexmani_real.runtime.home import HomeConfigurationError, home_arm, home_hand
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    check_device_feedback,
    check_observation,
    missing_tactile_channels,
    read_observation,
    sources_available,
)
from dexmani_real.runtime.state import INITIAL_STATE_TIMEOUT_S, RunEndReason
from dexmani_real.utils.log import error_detail
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)


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
        self.control_needs = ObservationNeeds(vr=True)
        self.record_needs = record_needs
        self.capture_needs = self.control_needs.union(record_needs)
        self.controller = controller
        self.recording_dir = recording_dir
        self.max_duration_s = max_duration_s
        self.runtime_config_json = runtime_config_json
        self.task_label = task_label
        self.camera_snapshot = camera_snapshot
        self.keyboard = keyboard
        self.audio = audio

        self.phase = TeleopPhase.IDLE
        self.cancel_epoch = None
        self.next_tick_ns = None
        self.control_dt_ns = sampling_clock_ns(runtime.teleop.control_hz)

        self.recorder = None
        self.capture_deadline_ns = None
        self.discard_requested = False
        self.pending_termination_reason = None
        self.termination_details = []

    def record_error(self, stage, exc):
        if isinstance(exc, KeyboardInterrupt):
            self.runtime_state.request_emergency_stop()
        else:
            self.runtime_state.stop_signal.set()
        self.pending_termination_reason = RunEndReason.FAILED.name.lower()
        self.termination_details.append(error_detail(stage, exc))
        self.audio.play("emergency")

    def end_episode(self):
        primary_failure = sys.exc_info()[1]
        failure = primary_failure
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
        self.controller.clear()

        if self.runtime_state.estop_request:
            self.audio.play("emergency")
        if failure is None and not self.runtime_state.quit_requested and not self.runtime_state.estop_request:
            try:
                self.finish_recording()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    self.runtime_state.request_emergency_stop()
                self.request_episode_end("recording_failed", run_end_reason=RunEndReason.FAILED)
                failure = exc
        self.phase = TeleopPhase.IDLE
        self.cancel_epoch = None
        if failure is not None and primary_failure is None:
            raise failure

    def finish_recording(self):
        recorder = self.recorder
        if recorder is None:
            return
        reason = self.pending_termination_reason
        save = not self.discard_requested
        published = recorder.finish(
            save=save,
            reason=reason,
            details=self.termination_details,
        )
        # finish 失败时保留 recorder，让会话收尾继续持有未释放的 writer。
        self.recorder = None
        request = recorder.finish_request
        if self.runtime_state.estop_request:
            self.audio.play("emergency")
            return
        if request.reason not in {"operator", "quit", "duration_complete"} or any(
            "exception_type" in detail for detail in request.details
        ):
            return
        if published is not None and recorder.recording_truncated_reason is None:
            logger.info("本段已保存。")
            self.audio.play("save")
        elif not request.save:
            logger.info("本段已丢弃。")
            self.audio.play("discard")

    def request_episode_end(
        self, reason, *, run_end_reason=RunEndReason.CANCELLED,
    ):
        """先请求停止并保留结束意图，资源收尾由 run_episode 的边界执行。"""
        self.runtime_state.stop_signal.set()
        if self.pending_termination_reason is None or run_end_reason is RunEndReason.FAILED:
            self.pending_termination_reason = run_end_reason.name.lower()
            self.termination_details.append(dict(stage="termination", reason=reason))
            logger.info("本段结束：%s（%s）。", self.pending_termination_reason, reason)

    def poll_blocking_commands(self):
        self.robot.check()
        commands = self.keyboard.poll(self.runtime_state, timeout=0)[0]
        terminal = tuple(
            event for event in commands
            if event in ("esc", "q", "s", "c", "d")
        )
        if terminal:
            self.handle_commands(terminal)
        return bool(terminal) or not self.runtime_state.requests_allow_motion()

    def preparation_valid(self):
        return (
            self.runtime_state.cancel_epoch == self.cancel_epoch
            and self.runtime_state.requests_allow_motion()
        )

    def preparation_interrupted(self):
        if (self.poll_blocking_commands() or self.pending_termination_reason is not None
                or not self.preparation_valid()):
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
            row = read_observation(self.sensors.channels, self.robot, needs=self.capture_needs)
            issues = self.admit_observation(row)
            if self.recording_dir is not None and (issue := self.recording_issue(row)) is not None:
                issues.append(issue)
            joints_ready = after_ns is None or (
                int(row.arm["timestamp_ns"][0]) >= after_ns
                and (not self.runtime.hand.enabled or row.hand is not None
                     and int(row.hand["timestamp_ns"][0]) >= after_ns)
            )
            if not issues and joints_ready:
                return row
            if time.monotonic_ns() >= until:
                missing = missing_tactile_channels(row, self.record_needs.required_tactile)
                if missing:
                    reason = "initial_tactile_unavailable: " + ", ".join(missing)
                elif issues:
                    reason = "initial_observation_unavailable: " + "; ".join(issues)
                else:
                    reason = "initial_joint_feedback_unavailable"
                logger.warning("本段未开始：%s；可稍后再次按 B。", reason)
                self.request_episode_end(reason)
                return None
            time.sleep(0.005)

    def start_episode(self, expected_cancel_epoch):
        if self.recorder is not None:
            raise RecordingError("previous capture still owns resources")
        with self.runtime_state.motion_lock:
            # 主线程接纳时捕获取消上下文；停止不能被随后准备覆盖。
            if (
                expected_cancel_epoch != self.runtime_state.cancel_epoch
                or not self.runtime_state.requests_allow_motion()
            ):
                return
            self.runtime_state.stop_signal.set()
            self.cancel_epoch = self.runtime_state.cancel_epoch
            self.discard_requested = False
            self.phase = TeleopPhase.PREPARING
        logger.info("Teleop preparing")
        self.pending_termination_reason = None
        self.termination_details = []
        try:
            if self.preparation_interrupted():
                return

            self.robot.stop()
            if self.preparation_interrupted():
                return
            # 全模态初值先于 writer；等待与资源初始化期间设备保持停止。
            if self.start_observation() is None or self.preparation_interrupted():
                return
            if self.recording_dir is not None:
                metadata = snapshot_recording_metadata(
                    self.camera_snapshot,
                    tactile_reset_metadata=self.robot.hand.tactile_reset_metadata,
                )
                metadata["vr_to_robot_rotation"] = self.controller.rotation.as_matrix().tolist()
                if self.runtime_config_json is not None:
                    metadata["runtime_config_json"] = self.runtime_config_json
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
            if self.recorder is not None:
                # writer 可能阻塞超过反馈 TTL；仍停止采样，允许一次 CRC 后恢复。
                if self.start_observation(after_ns=time.monotonic_ns()) is None:
                    return
            if self.preparation_interrupted():
                return
            self.robot.prepare_streaming(self.cancel_epoch)
            row = self.start_observation()
            if row is None:
                return
            self.try_begin(row)
        except OperationCancelled:
            self.robot.check()
            if (self.runtime_state.cancel_epoch == self.cancel_epoch
                    and self.runtime_state.requests_allow_motion()):
                raise
            self.poll_blocking_commands()
            if self.pending_termination_reason is None:
                self.request_episode_end("preparation_cancelled")
            if not self.robot.arm.is_connected:
                self.runtime_state.request_quit()

    def handle_operator_command(self, cmd: str):
        expected_cancel_epoch = None
        if cmd in {"b", "h", "t"}:
            expected_cancel_epoch = self.keyboard.accept_operation(self.runtime_state)
            if expected_cancel_epoch is None:
                return False
        logger.debug("Teleop key=%s phase=%s", cmd.upper(), self.phase.value)
        if cmd == "esc":
            self.runtime_state.request_emergency_stop()
            self.audio.play("emergency")
            if self.phase is not TeleopPhase.IDLE:
                self.request_episode_end("estop", run_end_reason=RunEndReason.ESTOP)
            return True
        if self.runtime_state.estop_request:
            return True

        if cmd == "q":
            self.audio.play("quit")
            if self.phase is not TeleopPhase.IDLE:
                self.request_episode_end("quit", run_end_reason=RunEndReason.QUIT)
            return True
        if self.runtime_state.quit_requested:
            return True
        if cmd in ("s", "c", "d"):
            if self.phase is not TeleopPhase.IDLE:
                self.request_episode_end(
                    "discard" if cmd == "d" else "stop", run_end_reason=RunEndReason.OPERATOR,
                )
        elif cmd == "t":
            if not self.runtime.hand.enabled:
                logger.info("T ignored: hand is disabled")
                return False
            if (
                self.phase is not TeleopPhase.IDLE or self.recorder is not None
                or self.runtime_state.cancel_epoch != expected_cancel_epoch
                or not self.runtime_state.requests_allow_motion()
            ):
                logger.warning("T ignored: tactile reset requires idle, no capture or preparation")
                return False
            try:
                logger.info("正在请求触觉传感器复位，请保持手部无接触；S/Q/Esc 可中断后续请求。")
                self.robot.reset_tactile_sensors(
                    cancel_epoch=expected_cancel_epoch,
                    cancel_requested=self.poll_blocking_commands,
                )
            except OperationCancelled as exc:
                self.robot.check()
                logger.info("%s", exc)
            return True
        elif cmd == "h":
            with self.runtime_state.motion_lock:
                if self.phase is not TeleopPhase.IDLE or self.recorder is not None:
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
                    home_hand(
                        self.runtime_state, self.robot,
                        cancel_epoch=expected_cancel_epoch,
                        cancel_requested=self.poll_blocking_commands,
                    )
                    home_arm(
                        self.runtime_state, self.robot,
                        cancel_epoch=expected_cancel_epoch,
                        cancel_requested=self.poll_blocking_commands,
                    )
                except OperationCancelled as exc:
                    self.robot.stop()
                    self.robot.check()
                    if (self.runtime_state.cancel_epoch == expected_cancel_epoch
                            and self.runtime_state.requests_allow_motion()):
                        raise
                    if not self.robot.arm.is_connected:
                        self.runtime_state.request_quit()
                    logger.info("HOME cancelled: %s", exc)
                except HomeConfigurationError as exc:
                    self.robot.check()
                    logger.warning("本次 HOME 未完成（手部可能已回位）：%s", exc)
                else:
                    logger.info("已回初始位。")
                    self.audio.play("home_done")
            self.poll_blocking_commands()
            return True
        elif cmd == "b":
            if (self.phase is TeleopPhase.IDLE
                    and self.runtime_state.cancel_epoch == expected_cancel_epoch
                    and self.runtime_state.requests_allow_motion()):
                self.run_episode(expected_cancel_epoch)
                return True
            return False
        return False

    def handle_commands(self, commands: Sequence[str]):
        if "d" in commands and self.phase is not TeleopPhase.IDLE:
            self.discard_requested = True
        # 回调已置停止信号；先登记 D 意图，再按原优先级选择结束原因。
        for terminal in ("esc", "q", "d", "s", "c"):
            if terminal in commands:
                self.handle_operator_command(terminal)
                return
        for event in commands:
            if self.handle_operator_command(event):
                break

    def try_begin(self, row):
        if self.preparation_interrupted():
            return
        try:
            self.controller.reset(row)
        except HandStartRejected as exc:
            logger.warning("%s", exc)
            self.request_episode_end(str(exc))
            return
        with self.runtime_state.motion_lock:
            started_ns = (
                time.monotonic_ns()
                if self.runtime_state.cancel_epoch == self.cancel_epoch
                and self.runtime_state.requests_allow_motion()
                and not self.runtime_state.stop_signal.is_set()
                else None
            )
        if started_ns is None:
            if not self.preparation_interrupted():
                self.request_episode_end("begin_unavailable")
            return
        self.phase = TeleopPhase.ACTIVE
        self.first_target_pending = True
        logger.info("Teleop active")
        self.capture_deadline_ns = (
            started_ns + int(self.max_duration_s * 1e9) if self.recorder is not None else None
        )
        self.next_tick_ns = None
        self.termination_details.append(dict(
            stage="teleop_anchor",
            observation_ns=int(row.observation_timestamp_ns),
            arm_qpos=row.arm["qpos"][0].tolist(),
            hand_qpos=row.hand["qpos"][0].tolist() if row.hand is not None else None,
            vr_wrist_pos=row.vr["wrist_pos"].tolist(),
            vr_wrist_quat_wxyz=row.vr["wrist_quat_wxyz"].tolist(),
            vr_recv_ts_ns=int(row.vr["recv_ts_ns"]),
        ))

    def execute_control_step(self, row):
        cancel_epoch = cast(int, self.cancel_epoch)
        command = None
        compute_start_ns = time.monotonic_ns()
        try:
            result = self.controller.compute_target(row, control_ns=compute_start_ns)
            if result.rejection_reason is not None or result.arm_qpos is None:
                self.request_episode_end(f"target_unavailable: {result.rejection_reason}")
                return
            if self.recorder is not None:
                self.recorder.check_error()
            if (self.runtime_state.cancel_epoch != cancel_epoch
                    or not self.runtime_state.requests_allow_motion()
                    or self.runtime_state.stop_signal.is_set()):
                raise OperationCancelled("teleop cancelled before submission")
            issues = self.admit_observation(row)
            if issues:
                self.request_episode_end("control_sources: " + "; ".join(issues))
                return
            command = RobotCommand(result.arm_qpos, result.hand_qpos)
            submit_start_ns = time.monotonic_ns()
            self.robot.send_action(
                command,
                expires_ns=submit_start_ns + int(self.runtime.teleop.target_max_age_s * 1e9),
            )
            submit_end_ns = time.monotonic_ns()
            self.controller.commit(command, eef_pose=result.eef_pose, submit_end_ns=submit_end_ns)
            if self.first_target_pending:
                logger.info("Teleop first target submitted")
                self.first_target_pending = False
                self.audio.play("begin")
        except OperationCancelled as exc:
            # 先检查 SDK/worker 故障，再即时登记已入队的终止意图。
            interrupted = self.poll_blocking_commands()
            if self.pending_termination_reason is None and not interrupted:
                self.request_episode_end(f"control_interrupted: {exc}")
        finally:
            primary_error = sys.exc_info()[1]
            if primary_error is not None:
                self.runtime_state.stop_signal.set()
            if self.recorder is not None:
                try:
                    # 示教行要求真实提交尝试；末次映射失败不能写 NaN 否定此前前缀。
                    issue = "recording_no_command" if command is None else None
                    issue = self.recorder.append(row, command, quality_issue=issue)
                    if issue is not None:
                        self.request_episode_end(issue)
                except BaseException as exc:
                    if primary_error is None:
                        raise
                    self.record_error("recording_append", exc)
                    logger.exception("teleop recording append also failed")

    def recording_issue(self, row):
        if not self.sensors.check(names=("camera",)):
            raise RuntimeError("recording camera worker exited")
        issues = check_observation(row, self.runtime, self.record_needs,
                                   now_ns=row.observation_timestamp_ns, allow_tactile_hold=True)
        return "; ".join(issues) if issues else None

    def admit_observation(self, row):
        return (check_device_feedback(row, self.runtime)
                + check_observation(row, self.runtime, self.control_needs))

    def run_control_tick(self):
        if self.runtime_state.stop_signal.is_set():
            return  # 下一次输入快照负责归因；设备边界已停止。
        if (
            self.recorder is not None
            and time.monotonic_ns() >= cast(int, self.capture_deadline_ns)
        ):
            self.request_episode_end(
                "max_record_duration", run_end_reason=RunEndReason.DURATION_COMPLETE,
            )
            return
        now_ns = time.monotonic_ns()
        if self.next_tick_ns is not None and now_ns < self.next_tick_ns:
            return
        self.next_tick_ns = now_ns + self.control_dt_ns
        row = read_observation(self.sensors.channels, self.robot, needs=self.capture_needs)
        devices = check_device_feedback(row, self.runtime)
        if devices:
            self.request_episode_end("control_sources: " + "; ".join(devices))
            return
        if self.recorder is not None and (quality := self.recording_issue(row)) is not None:
            self.recorder.append(row, quality_issue=quality)
            self.request_episode_end(quality)
            return
        if not sources_available(row, self.control_needs):
            return
        issues = check_observation(row, self.runtime, self.control_needs)
        if issues:
            self.request_episode_end("control_sources: " + "; ".join(issues))
            return
        self.execute_control_step(row)

    def service_episode(self, timeout):
        if self.runtime_state.stop_signal.is_set():
            self.robot.service_idle()
        else:
            self.robot.service_feedback()
        if self.runtime_state.estop_request:
            self.request_episode_end(
                "estop", run_end_reason=RunEndReason.ESTOP,
            )
        if self.pending_termination_reason is not None:
            return False
        if self.recorder is not None:
            self.recorder.check_error()
        if self.phase is TeleopPhase.ACTIVE and self.next_tick_ns is not None:
            timeout = min(timeout, max(0.0, (self.next_tick_ns - time.monotonic_ns()) / 1e9))
        events, stopped = self.keyboard.poll(timeout=timeout, state=self.runtime_state)
        self.handle_commands(events)
        if self.phase is TeleopPhase.ACTIVE and self.pending_termination_reason is None and stopped:
            self.request_episode_end("executor_stopped")

        return (
            self.pending_termination_reason is None
            and not self.runtime_state.quit_requested
        )

    def run_episode(self, expected_cancel_epoch):
        """顺序准备、控制、停止与录制收尾；每次 B 都创建独立新段。"""
        try:
            self.start_episode(expected_cancel_epoch)
            while self.phase is not TeleopPhase.IDLE:
                if not self.service_episode(0.005):
                    break
                self.run_control_tick()
        except BaseException as exc:
            # 主错误由会话边界报告；阻塞停止前仍须锁存中断并标记本段失败。
            if isinstance(exc, KeyboardInterrupt):
                self.runtime_state.request_emergency_stop()
            self.request_episode_end("episode_failed", run_end_reason=RunEndReason.FAILED)
            raise
        finally:
            if self.phase is not TeleopPhase.IDLE:
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

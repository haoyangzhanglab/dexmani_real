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
from dexmani_real.utils.log import error_detail
from dexmani_real.utils.timing import next_tick_ns, sampling_clock_ns

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

        self.phase = TeleopPhase.IDLE
        self.run_id = None
        self.next_tick_ns = None
        self.control_dt_ns = sampling_clock_ns(runtime.teleop.control_hz)

        self.recorder = None
        self.capture_deadline_ns = None
        self.capture_failed = False
        self.discard_requested = False
        self.pending_termination_reason = None
        self.termination_details = []
        self.soft_timeout_count = 0
        self.last_soft_timeout = None

    def request_discard(self):
        # 与保存决定排序；回调只锁存意图和撤权，不等待资源收尾。
        with self.runtime_state.motion_lock:
            if self.run_id is not None:
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

    def end_episode(self):
        primary_failure = sys.exc_info()[1]
        failure = primary_failure
        try:
            self.robot.stop()
        except BaseException as exc:
            if exc is not primary_failure:
                self.record_error("stop", exc)
                if failure is None:
                    failure = exc
                else:
                    logger.exception("teleop stop also failed")
        self.controller.clear()

        if self.capture_failed:
            self.audio.play("emergency")
        if self.soft_timeout_count:
            summary = {
                "stage": "soft_timeouts", "count": self.soft_timeout_count, "last": self.last_soft_timeout,
            }
            last_lateness_ms = (
                self.last_soft_timeout["observed_ns"] - self.last_soft_timeout["deadline_ns"]
            ) / 1_000_000
            logger.info("Teleop soft timeouts: %s; last_lateness_ms=%.3f", summary, last_lateness_ms)
            if self.recorder is not None:
                self.termination_details.append(summary)
            self.soft_timeout_count = 0
            self.last_soft_timeout = None
        try:
            self.finish_recording()
        except BaseException as exc:
            if exc is not failure:
                self.record_error("recording_cleanup", exc)
                if failure is None:
                    failure = exc
                else:
                    logger.exception("teleop recording cleanup also failed")
        # 入队前已捕获收尾代际的 B/H/T 也必须失效，不能只清空现有队列。
        self.runtime_state.revoke_motion()
        self.phase = TeleopPhase.IDLE
        self.run_id = None
        if failure is not None and primary_failure is None:
            raise failure

    def finish_recording(self):
        recorder = self.recorder
        if recorder is None or not recorder.start_attempted:
            self.recorder = None
            return
        reason = self.pending_termination_reason
        with self.runtime_state.motion_lock:
            save = not self.discard_requested
        published = recorder.finish(
            save=save,
            reason=reason,
            details=self.termination_details,
            technical_status=(
                "normal" if not self.capture_failed and reason in {"operator", "quit", "duration_complete"}
                else "failed"
            ),
        )
        # finish 失败时保留 recorder，让会话收尾继续持有未释放的 writer。
        self.recorder = None
        interrupted = self.runtime_state.estop_request or self.runtime_state.error_state
        if interrupted and not self.capture_failed:
            self.audio.play("emergency")
        if published is not None:
            logger.info("本段已保存。")
            if not self.capture_failed and not interrupted:
                self.audio.play("save")
        elif not save:
            logger.info("本段已丢弃。")
            if not self.capture_failed and not interrupted:
                self.audio.play("discard")

    def request_episode_end(
        self, reason, *, run_end_reason=RunEndReason.EXECUTOR_BOUNDARY, failed=True,
    ):
        """先撤权并保留结束意图，资源收尾由 run_episode 的边界执行。"""
        self.capture_failed |= failed
        reason_code = self.runtime_state.revoke_motion(
            reason=run_end_reason, expected_run_id=self.run_id,
        )
        if self.pending_termination_reason is None:
            self.pending_termination_reason = reason_code.name.lower()
            self.termination_details.append(dict(stage="termination", reason=reason))
            logger.info("本段结束：%s（%s）。", self.pending_termination_reason, reason)

    def poll_blocking_commands(self):
        self.robot.check()
        commands = self.keyboard.poll(timeout=0)
        terminal = tuple(
            event for event in commands
            if event.key in ("esc", "q", "s", "c", "d")
        )
        if terminal:
            self.handle_commands(terminal)
        return (
            bool(terminal) or not self.runtime_state.requests_allow_motion()
            # 快停止回调可能已撤权，但按键还未入队。
            or (
                self.phase in (TeleopPhase.PREPARING, TeleopPhase.ACTIVE)
                and self.runtime_state.stop_signal.is_set()
                and self.runtime_state.run_id != self.run_id
            )
        )

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
            self.capture_failed = self.discard_requested = False
            self.phase = TeleopPhase.PREPARING
        logger.info("Teleop preparing run=%s", self.run_id)
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
        except OperationCancelled:
            if not self.poll_blocking_commands():
                self.request_episode_end("preparation_revoked", failed=False)

    def handle_operator_command(self, key_event):
        cmd = key_event.key
        expected_run_id = key_event.run_id
        logger.debug("Teleop key=%s phase=%s run=%s", cmd.upper(), self.phase.value, self.run_id)
        if cmd == "esc":
            self.runtime_state.request_emergency_stop()
            return True
        if self.runtime_state.estop_request or self.runtime_state.error_state:
            return True

        if cmd == "q":
            self.audio.play("quit")
            if self.phase is not TeleopPhase.IDLE:
                self.request_episode_end("quit", run_end_reason=RunEndReason.QUIT, failed=False)
            return True
        if self.runtime_state.quit_requested:
            return True
        if cmd in ("s", "c", "d"):
            if self.phase is not TeleopPhase.IDLE:
                self.request_episode_end(
                    "discard" if cmd == "d" else "stop", run_end_reason=RunEndReason.OPERATOR,
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
                self.robot.check()
                logger.info("%s", exc)
            finally:
                self.runtime_state.revoke_motion()
            return True
        elif cmd == "h":
            with self.runtime_state.motion_lock:
                if self.phase is not TeleopPhase.IDLE or self.recorder is not None:
                    logger.warning("H ignored: 请先按 S 结束本段，再回初始位。")
                    return True
                if not self.runtime_state.can_move(
                    run_id=expected_run_id, required_safety_state=SafetyState.ARMED,
                ):
                    logger.info("忽略已被停止指令撤销的 HOME。")
                    return False
                self.runtime_state.revoke_motion(reason=RunEndReason.OPERATOR)
                expected_run_id = self.runtime_state.run_id
            try:
                if not self.poll_blocking_commands():
                    logger.info("正在回初始位…")
                    self.audio.play("home")
                    try:
                        home_robot(
                            self.runtime_state, self.robot, self.get_home_planner,
                            run_id=expected_run_id,
                            cancel_requested=self.poll_blocking_commands,
                        )
                    except OperationCancelled as exc:
                        self.robot.check()
                        logger.info("HOME cancelled: %s", exc)
                    except (HomeConfigurationError, HomePathRejected) as exc:
                        self.robot.check()
                        logger.warning("本次 HOME 未完成（手部可能已回位）：%s", exc)
                    else:
                        logger.info("已回初始位。")
                        self.audio.play("home_done")
                self.poll_blocking_commands()
            finally:
                self.runtime_state.revoke_motion()
            return True
        elif cmd == "b":
            if self.phase is TeleopPhase.IDLE and self.runtime_state.can_move(
                run_id=expected_run_id, required_safety_state=SafetyState.ARMED,
            ):
                self.run_episode(expected_run_id)
                return True
            return False
        return False

    def handle_commands(self, commands):
        if any(event.key == "esc" for event in commands):
            self.runtime_state.request_emergency_stop()
            return
        # 键盘回调已完成撤权及 D/Q 意图；这里只选择段收尾，不重新处理意图。
        for terminal in ("q", "d", "s", "c"):
            terminal_event = next((event for event in commands if event.key == terminal), None)
            if terminal_event is not None:
                self.handle_operator_command(terminal_event)
                return
        for event in commands:
            if self.handle_operator_command(event):
                break

    def try_begin(self, row):
        if self.preparation_interrupted():
            return
        self.controller.reset(row)
        started_ns = (
            self.runtime_state.begin_motion(run_id=self.run_id)
            if self.phase is TeleopPhase.PREPARING else None
        )
        if started_ns is None:
            if not self.preparation_interrupted():
                self.request_episode_end("begin_unavailable")
            return
        self.phase = TeleopPhase.ACTIVE
        self.first_target_pending = True
        logger.info("Teleop active run=%s", self.run_id)
        self.capture_deadline_ns = (
            started_ns + int(self.max_duration_s * 1e9) if self.recorder is not None else None
        )
        self.next_tick_ns = None
        self.termination_details.append(dict(
            stage="teleop_anchor", run_id=self.run_id,
            observation_ns=int(row.observation_timestamp_ns),
            arm_qpos=row.arm["qpos"][0].tolist(),
            hand_qpos=row.hand["qpos"][0].tolist() if row.hand is not None else None,
            vr_wrist_pos=row.vr["wrist_pos"].tolist(),
            vr_wrist_quat_wxyz=row.vr["wrist_quat_wxyz"].tolist(),
            vr_recv_ts_ns=int(row.vr["recv_ts_ns"]),
        ))

    def record_soft_timeout(self, stage, *, deadline_ns, observed_ns):
        self.soft_timeout_count += 1
        self.last_soft_timeout = {
            "stage": stage,
            "deadline_ns": int(deadline_ns), "observed_ns": int(observed_ns),
        }

    def execute_control_step(self, row, tick_end_ns):
        run_id = cast(int, self.run_id)
        command = None
        compute_start_ns = time.monotonic_ns()
        try:
            result = self.controller.compute_target(row, control_ns=compute_start_ns)
            if result.rejection_reason is not None or result.arm_qpos is None:
                self.request_episode_end(f"target_unavailable: {result.rejection_reason}")
                return
            if self.recorder is not None:
                self.recorder.check_error()
            if not self.runtime_state.can_move(run_id=run_id):
                raise OperationCancelled("teleop cancelled before submission")
            checked_ns = time.monotonic_ns()
            if checked_ns >= tick_end_ns:
                # 未发送的候选不推进滤波、关节或时间参考；下周期从最新输入继续。
                self.record_soft_timeout(
                    "compute", deadline_ns=tick_end_ns, observed_ns=checked_ns,
                )
                return
            command = RobotCommand(run_id, result.arm_qpos, result.hand_qpos)
            submit_start_ns = time.monotonic_ns()
            self.robot.send_action(
                command,
                expires_ns=submit_start_ns + int(self.runtime.teleop.vr_stale_threshold_s * 1e9),
            )
            submit_end_ns = time.monotonic_ns()
            self.controller.commit(command, eef_pose=result.eef_pose, submit_end_ns=submit_end_ns)
            if self.first_target_pending:
                logger.info("Teleop first target submitted run=%s", run_id)
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
                self.runtime_state.revoke_motion(
                    reason=failure_reason(primary_error), expected_run_id=run_id,
                )
            if self.recorder is not None:
                try:
                    self.recorder.append(row, command)
                except BaseException as exc:
                    if primary_error is None:
                        raise
                    if isinstance(exc, KeyboardInterrupt):
                        self.runtime_state.request_emergency_stop()
                    self.recorder.store_error(exc)

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
            and time.monotonic_ns() >= cast(int, self.capture_deadline_ns)
        ):
            self.request_episode_end(
                "max_record_duration", run_end_reason=RunEndReason.DURATION_COMPLETE, failed=False,
            )
            return
        now_ns = time.monotonic_ns()
        if self.next_tick_ns is not None:
            if now_ns < self.next_tick_ns:
                return
            # 绝对网格上跳过错过的时点，不追赶历史控制决策。
            self.next_tick_ns += ((now_ns - self.next_tick_ns) // self.control_dt_ns) * self.control_dt_ns
        row = read_observation(
            self.sensors.channels, self.robot, needs=self.needs,
        )
        issues = self.admit_observation(row)
        if issues:
            self.request_episode_end("control_sources: " + "; ".join(issues))
            return
        stamp = row.observation_timestamp_ns
        if self.next_tick_ns is None:
            self.next_tick_ns = row.control_timestamp_ns
        if stamp >= self.next_tick_ns + self.control_dt_ns:
            # 本次复制已错过提交机会；不映射、不提交，下个未来时点重新读。
            self.record_soft_timeout(
                "observation", deadline_ns=self.next_tick_ns + self.control_dt_ns, observed_ns=stamp,
            )
            self.next_tick_ns = next_tick_ns(self.next_tick_ns, self.control_dt_ns, stamp)
            if self.recorder is not None:
                self.recorder.append(row)
            return
        tick_end_ns = self.next_tick_ns + self.control_dt_ns
        self.execute_control_step(row, tick_end_ns)
        self.next_tick_ns = next_tick_ns(
            self.next_tick_ns, self.control_dt_ns, time.monotonic_ns(),
        )

    def close(self):
        if self.phase is not TeleopPhase.IDLE:
            if self.pending_termination_reason is None:
                self.request_episode_end(
                    "interrupted", run_end_reason=RunEndReason.RUNTIME_SHUTDOWN, failed=False,
                )
            self.end_episode()

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
        self.handle_commands(self.keyboard.poll(timeout=timeout))
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
        """顺序准备、控制、停止与录制收尾；每次 B 都创建独立新段。"""
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
            if self.phase is not TeleopPhase.IDLE:
                if self.pending_termination_reason is None:
                    self.request_episode_end(
                        "interrupted", failed=False,
                        run_end_reason=(
                            RunEndReason.QUIT if self.runtime_state.quit_requested
                            else RunEndReason.RUNTIME_SHUTDOWN
                        ),
                    )
                self.end_episode()

    def run(self):
        while not self.runtime_state.quit_requested:
            self.robot.service_idle()
            if self.runtime_state.estop_request or self.runtime_state.error_state:
                break
            self.handle_commands(self.keyboard.poll(timeout=0.005))

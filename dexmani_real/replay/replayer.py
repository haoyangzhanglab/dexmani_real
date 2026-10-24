"""结合当前反馈，按标称节拍逐行尝试提交 Raw 目标。"""

import logging
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dexmani_real.planning.action import ActionMapper, project_arm_command, project_hand_command
from dexmani_real.planning.paths import wrap_nearest_equivalent
from dexmani_real.replay.capture import ReplayRecorder
from dexmani_real.robot.commands import RobotCommand, OperationCancelled, RobotError
from dexmani_real.runtime.observation import (
    ObservationNeeds, feedback_deadline_ns, read_observation, observation_issues,
    control_deadline_ns, service_feedback,
)
from dexmani_real.runtime.operator_input import OperatorCommand
from dexmani_real.runtime.state import RunEndReason
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)


class ReplayStatus(str, Enum):
    COMPLETED = "completed"
    USER_QUIT = "user_quit"
    REJECTED = "rejected"
    ESTOP = "estop"
    FAULT = "fault"


@dataclass(frozen=True)
class ReplayOutcome:
    """Replay result, including any samples captured before it stopped."""

    status: ReplayStatus
    replay_data: dict[str, np.ndarray] | None = None
    reason: str = ""

    @property
    def successful(self) -> bool:
        return self.status in (ReplayStatus.COMPLETED, ReplayStatus.USER_QUIT)


class TrajectoryReplayer:
    """Own replay admission, command references and the captured physical prefix."""

    def __init__(self, runtime_state, runtime, keyboard, *, robot):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.keyboard = keyboard
        self.robot = robot

    def run(self, trajectory, *, hand_start_duration_s, results=None):
        self.failure_details = []
        self.preparation_submission_count = 0
        self.capture_failed = False
        self.epoch = self.runtime_state.run_id
        self.capture = ReplayRecorder(trajectory.num_frames)
        self.action_mapper = ActionMapper.for_mode(
            "joint", arm=self.runtime.arm, hand=self.runtime.hand,
            motion=self.runtime.motion, environment=self.runtime.environment,
        )
        self.previous_arm = None
        period_ns, _ = sampling_clock_ns(trajectory.fps)
        status, reason = ReplayStatus.COMPLETED, ""
        run_end_reason = RunEndReason.EXECUTOR_BOUNDARY
        try:
            if results is not None:
                results.begin_episode()
            status, reason, row = self.admit_replay(trajectory, hand_start_duration_s)
            if status == ReplayStatus.COMPLETED:
                if results is not None:
                    results.entered(self.epoch)
                if hand_start_duration_s > 0:
                    interrupted = self.warm_up_hand(
                        trajectory, self.epoch, row, hand_start_duration_s
                    )
                    if interrupted is not None:
                        status, reason = interrupted.status, interrupted.reason
                if status == ReplayStatus.COMPLETED:
                    status, reason = self.replay_rows(trajectory, period_ns)
        except (OperationCancelled, TimeoutError) as exc:
            operator_quit = (
                isinstance(exc, OperationCancelled)
                and self.runtime_state.run_ended_id == self.epoch
                and self.runtime_state.run_ended_reason == RunEndReason.QUIT
            )
            status = ReplayStatus.USER_QUIT if operator_quit else ReplayStatus.REJECTED
            if self.runtime_state.estop_request:
                status = ReplayStatus.ESTOP
            reason = str(exc)
        except KeyboardInterrupt:
            self.runtime_state.request_emergency_stop()
            status, reason = ReplayStatus.ESTOP, "KeyboardInterrupt"
            run_end_reason = RunEndReason.ESTOP
        except Exception as exc:
            logger.exception("replay failed")
            status, reason = ReplayStatus.FAULT, str(exc)
            run_end_reason = RunEndReason.HARDWARE_FAULT if isinstance(exc, RobotError) else RunEndReason.POLICY_FAILURE
            self.runtime_state.mark_error()
        finally:
            self.capture_failed = self.capture_failed or status not in {ReplayStatus.COMPLETED, ReplayStatus.USER_QUIT}
            self.capture_reason = reason or status.value
            if status == ReplayStatus.USER_QUIT:
                run_end_reason = RunEndReason.QUIT
            elif status == ReplayStatus.ESTOP:
                run_end_reason = RunEndReason.ESTOP
            self.runtime_state.revoke_motion(reason=run_end_reason)
            try:
                self.robot.stop()
            except Exception as exc:
                self.runtime_state.mark_error()
                logger.exception("replay stop failed")
                self.failure_details.append(f"stop failed: {exc}")
                status, reason = ReplayStatus.FAULT, reason or status.value
        return self.finalize_capture(status, reason)

    def admit_replay(self, trajectory, hand_start_duration_s):
        status, reason = ReplayStatus.COMPLETED, ""
        self.robot.prepare_streaming(self.epoch, service=lambda: service_feedback(self.robot, self.runtime))
        row = read_observation(
            None, self.robot, needs=ObservationNeeds()
        )
        issues = observation_issues(row, self.runtime, ObservationNeeds())
        if not issues:
            self.robot.arm.update_health(
                health_until_ns=control_deadline_ns(row, self.runtime, ObservationNeeds()),
            )
        if issues:
            status, reason = ReplayStatus.REJECTED, "fresh robot feedback unavailable"
        else:
            self.previous_arm = row.arm["qpos"][0].copy()
            # Reposition separately with operator oversight; never synthesize replay rows.
            start_arm = wrap_nearest_equivalent(
                trajectory.arm_qpos[0],
                row.arm["qpos"][0],
                self.runtime.arm.joint_limit_lower,
                self.runtime.arm.joint_limit_upper,
            )
            start_pairs = [
                (row.arm["qpos"][0], start_arm),
                (row.hand["qpos"][0], trajectory.hand_qpos[0]),
            ]
            if hand_start_duration_s > 0:
                start_pairs.append((row.hand["qpos"][0], trajectory.action_hand_joint[0]))
            for measured, recorded in start_pairs:
                if np.max(np.abs(measured - recorded)) > np.deg2rad(10):
                    status, reason = (
                        ReplayStatus.REJECTED,
                        "start posture differs by more than 10 degrees",
                    )
                    break
        if status == ReplayStatus.COMPLETED and hand_start_duration_s > 0:
            target = trajectory.action_hand_joint[0]
            if (
                not np.isfinite(target).all()
                or np.any(target < self.runtime.hand.qpos_min_rad)
                or np.any(target > self.runtime.hand.qpos_max_rad)
            ):
                status, reason = (
                    ReplayStatus.REJECTED,
                    "hand preparation target violates limits",
                )
        if status == ReplayStatus.COMPLETED:
            with self.runtime_state.motion_lock:
                if (
                    self.runtime_state.run_id != self.epoch
                    or not self.runtime_state.begin_motion()
                ):
                    status, reason = ReplayStatus.REJECTED, "motion authority unavailable"
                else:
                    self.epoch = self.runtime_state.run_id
        return status, reason, row

    def replay_rows(self, trajectory, period_ns):
        status, reason = ReplayStatus.COMPLETED, ""
        origin_ns = time.monotonic_ns()
        for index in range(trajectory.num_frames):
            slot_start_ns = origin_ns + index * period_ns
            slot_deadline_ns = slot_start_ns + period_ns
            interrupted = self.wait_replay(self.epoch, slot_start_ns)
            if interrupted is not None:
                status, reason = interrupted.status, interrupted.reason
                break
            if time.monotonic_ns() >= slot_deadline_ns:
                raise TimeoutError(f"replay slot {index} missed before observation")
            row = read_observation(
                None, self.robot, needs=ObservationNeeds()
            )
            issues = observation_issues(row, self.runtime, ObservationNeeds())
            if not issues:
                self.robot.arm.update_health(
                    health_until_ns=control_deadline_ns(row, self.runtime, ObservationNeeds()),
                )
            if issues:
                status, reason = ReplayStatus.REJECTED, "robot feedback stale"
                break
            arm, hand, rejection = self.action_mapper.prepare_joint_target(
                trajectory.action_arm_joint[index],
                trajectory.action_hand_joint[index],
                row.arm["qpos"][0],
                self.previous_arm,
            )
            if rejection is not None:
                self.capture.record(
                    index, row, RobotCommand(self.epoch, arm, hand)
                )
                status, reason = ReplayStatus.REJECTED, f"replay target: {rejection}"
                break
            self.submit_row(index, row, arm, hand, slot_deadline_ns)
            self.previous_arm = arm.copy()
            interrupted = self.wait_replay(self.epoch, slot_deadline_ns)
            if interrupted is not None:
                status, reason = interrupted.status, interrupted.reason
                break
        return status, reason

    def submit_row(self, index, row, arm, hand, slot_deadline_ns):
        failure = None
        submit_started_ns = submit_completed_ns = -1
        valid_until_ns = min(slot_deadline_ns, feedback_deadline_ns(row, self.runtime))
        command = RobotCommand(self.epoch, arm, hand)
        try:
            if time.monotonic_ns() >= valid_until_ns:
                raise TimeoutError(f"replay slot {index} expired before submission")
            if not self.runtime_state.command_may_cross_sdk(run_id=self.epoch):
                raise OperationCancelled("replay cancelled before submission")
            submit_started_ns = time.monotonic_ns()
            try:
                self.robot.send_action(
                    command, valid_until_ns=valid_until_ns,
                    execute_until_ns=row.observation_timestamp_ns
                    + int(self.runtime.execution.max_decision_age_s * 1e9),
                )
            finally:
                submit_completed_ns = time.monotonic_ns()
            if submit_completed_ns >= valid_until_ns:
                raise TimeoutError(f"replay slot {index} submission returned after deadline")
        except BaseException as exc:
            failure = exc
            self.capture_failed = True
            self.runtime_state.revoke_motion(reason=(
                RunEndReason.ESTOP if isinstance(exc, KeyboardInterrupt)
                else RunEndReason.HARDWARE_FAULT if isinstance(exc, RobotError)
                else RunEndReason.EXECUTOR_BOUNDARY
            ))
        try:
            self.capture.record(
                index, row, command, submit_started_ns=submit_started_ns,
                submit_completed_ns=submit_completed_ns,
            )
        except Exception as exc:
            self.failure_details.append(f"capture failed: {type(exc).__name__}: {exc}")
            self.capture_failed = True
            if failure is not None:
                raise failure from exc
            raise
        if failure is not None:
            raise failure

    def finalize_capture(self, status, reason):
        data = self.capture.to_dict()
        if self.failure_details:
            reason = "; ".join([reason or status.value, *self.failure_details])
            status = ReplayStatus.FAULT
        data["preparation_submission_count"] = np.asarray(self.preparation_submission_count)
        data["termination_reason"] = np.asarray(self.capture_reason)
        data["cleanup_errors"] = np.asarray("; ".join(self.failure_details))
        data["technical_status"] = np.asarray("failed" if self.capture_failed else "normal")
        return ReplayOutcome(status, data, reason)

    def wait_replay(self, epoch, deadline_ns):
        """Keep operator input and lifecycle checks active during bounded waits."""
        while True:
            self.robot.check()
            signals = self.keyboard.poll(timeout=0)
            if not self.keyboard.healthy:
                self.runtime_state.request_emergency_stop()
            if self.runtime_state.estop_request:
                return ReplayOutcome(ReplayStatus.ESTOP, reason="operator emergency stop")
            if self.runtime_state.error_state:
                return ReplayOutcome(ReplayStatus.FAULT, reason="runtime failure")
            if self.runtime_state.quit_requested or OperatorCommand.QUIT in signals:
                return ReplayOutcome(ReplayStatus.USER_QUIT, reason="operator quit")
            if not self.runtime_state.running or self.runtime_state.run_id != epoch:
                cause = self.runtime_state.run_ended_reason
                if cause == RunEndReason.QUIT:
                    return ReplayOutcome(ReplayStatus.USER_QUIT, reason="operator quit")
                return ReplayOutcome(
                    ReplayStatus.REJECTED, reason=f"motion interrupted: {cause.name.lower()}"
                )
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return None
            time.sleep(min(0.005, remaining_ns / 1e9))

    def warm_up_hand(self, trajectory, epoch, row, duration_s):
        """Prepare frame zero without adding synthetic rows to the recorded trajectory."""
        arm_hold = project_arm_command(
            row.arm["qpos"][0],
            row.arm["qpos"][0],
            joint_lower_rad=self.runtime.arm.joint_limit_lower,
            joint_upper_rad=self.runtime.arm.joint_limit_upper,
        )
        hand_start = project_hand_command(
            row.hand["qpos"][0],
            qpos_min_rad=self.runtime.hand.qpos_min_rad,
            qpos_max_rad=self.runtime.hand.qpos_max_rad,
        )
        hand_target = project_hand_command(
            trajectory.action_hand_joint[0],
            qpos_min_rad=self.runtime.hand.qpos_min_rad,
            qpos_max_rad=self.runtime.hand.qpos_max_rad,
        )
        steps = max(1, int(np.ceil(duration_s * trajectory.fps)))
        period_ns, _ = sampling_clock_ns(trajectory.fps)
        print(
            f"XHand start: {steps * period_ns / 1e9:g}s ramp to first target, then wait for arrival. "
            "Preparation is outside replay rows; initial hand state differs from Raw. "
            "Q: stop; ESC: emergency stop",
            flush=True,
        )
        step = 0
        deadline_ns = 0
        final_stamp = None
        last_sample = 0
        consecutive = 0
        while True:
            interrupted = self.wait_replay(epoch, deadline_ns)
            if interrupted is not None:
                return interrupted
            row = read_observation(
                None, self.robot, needs=ObservationNeeds()
            )
            issues = observation_issues(row, self.runtime, ObservationNeeds())
            if not issues:
                self.robot.arm.update_health(
                    health_until_ns=control_deadline_ns(row, self.runtime, ObservationNeeds()),
                )
            if issues:
                return ReplayOutcome(ReplayStatus.REJECTED, reason="hand start feedback stale")
            if np.max(np.abs(row.arm["qpos"][0] - arm_hold)) > np.deg2rad(10):
                return ReplayOutcome(ReplayStatus.REJECTED, reason="arm moved during hand start")
            if final_stamp is None:
                progress = step / steps
                weight = progress * progress * (3 - 2 * progress)
                hand = (
                    hand_target
                    if step == steps
                    else hand_start + weight * (hand_target - hand_start)
                )
                rejection = self.action_mapper.target_checker.rejection(
                    arm_hold,
                    row.arm["qpos"][0],
                    arm_hold,
                    collision_hand_qpos=hand,
                )
                if rejection is not None:
                    return ReplayOutcome(
                        ReplayStatus.REJECTED, reason=f"hand start target: {rejection}"
                    )
                command = RobotCommand(epoch, arm_hold, hand)
                self.preparation_submission_count += 1
                self.robot.send_action(
                    command, valid_until_ns=feedback_deadline_ns(row, self.runtime),
                    execute_until_ns=row.observation_timestamp_ns + int(self.runtime.hand.home_timeout_s * 1e9),
                )
                stamp = time.monotonic_ns()
                if step == steps:
                    final_stamp = stamp
                    last_sample = stamp
                    self.robot.end_hand_send_interval()
                step += 1
                deadline_ns = stamp + period_ns
                continue

            if time.monotonic_ns() >= final_stamp + int(self.runtime.hand.home_timeout_s * 1e9):
                return ReplayOutcome(
                    ReplayStatus.REJECTED, reason="hand start convergence timed out"
                )
            sample = int(row.hand["timestamp_ns"][0])
            if sample > last_sample:
                last_sample = sample
                if np.max(np.abs(row.hand["qpos"][0] - hand_target)) <= np.deg2rad(
                    self.runtime.hand.home_tolerance_deg
                ):
                    consecutive += 1
                else:
                    consecutive = 0
                # 只有最终提交完成后不同的新鲜反馈样本才能判定到位。
                if consecutive >= 3:
                    print("XHand start ready; replaying recorded targets.", flush=True)
                    return None
            deadline_ns = time.monotonic_ns() + 10_000_000

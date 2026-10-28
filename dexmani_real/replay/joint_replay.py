"""结合当前反馈，按标称节拍逐行尝试提交 Raw 目标。"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from dexmani_real.control.actions import (
    ActionMapper,
    project_arm_command,
    project_hand_command,
)
from dexmani_real.control.constraints import wrap_nearest_equivalent
from dexmani_real.replay.trajectory import ReplayBuffer
from dexmani_real.robot.commands import OperationCancelled, RobotCommand, RobotError
from dexmani_real.runtime.observation import (
    ObservationNeeds,
    check_observation,
    read_observation,
)
from dexmani_real.runtime.state import RunEndReason
from dexmani_real.utils.timing import sampling_clock_ns

logger = logging.getLogger(__name__)


class ReplayStatus(str, Enum):
    COMPLETED = "completed"
    USER_QUIT = "user_quit"
    REJECTED = "rejected"
    ESTOP = "estop"
    FAULT = "fault"


@dataclass
class ReplayResult:
    status: ReplayStatus
    replay_data: dict[str, np.ndarray] = field(default_factory=lambda: ReplayBuffer(0).to_dict())
    reason: str = ""

    @property
    def successful(self) -> bool:
        return self.status in (ReplayStatus.COMPLETED, ReplayStatus.USER_QUIT)


class JointReplay:
    def __init__(self, runtime_state, runtime, keyboard, *, robot):
        self.runtime_state = runtime_state
        self.runtime = runtime
        self.keyboard = keyboard
        self.robot = robot

    def run(self, trajectory, *, hand_start_duration_s):
        self.epoch = self.runtime_state.run_id
        self.capture = ReplayBuffer(len(trajectory))
        self.action_mapper = ActionMapper.for_mode(
            "joint",
            arm=self.runtime.arm,
            hand=self.runtime.hand,
            motion=self.runtime.motion,
            environment=self.runtime.environment,
        )
        period_ns, _ = sampling_clock_ns(trajectory.fps)
        result = ReplayResult(ReplayStatus.COMPLETED)
        self.decision_age_ns = int(self.runtime.execution.max_decision_age_s * 1e9)
        run_end_reason = RunEndReason.EXECUTOR_BOUNDARY
        try:
            result.status, result.reason, row = self.admit_replay(trajectory, hand_start_duration_s)
            if result.status == ReplayStatus.COMPLETED:
                if hand_start_duration_s > 0:
                    interrupted = self.prepare_hand(
                        trajectory, row, hand_start_duration_s, period_ns
                    )
                    if interrupted is not None:
                        result.status, result.reason = interrupted
                if result.status == ReplayStatus.COMPLETED:
                    result.status, result.reason = self.replay_rows(trajectory, period_ns)
        except (OperationCancelled, TimeoutError) as exc:
            operator_quit = (
                isinstance(exc, OperationCancelled)
                and self.runtime_state.quit_requested
            )
            result.status = ReplayStatus.USER_QUIT if operator_quit else ReplayStatus.REJECTED
            if self.runtime_state.estop_request:
                result.status = ReplayStatus.ESTOP
            result.reason = str(exc)
        except KeyboardInterrupt:
            self.runtime_state.request_emergency_stop()
            result.status, result.reason = ReplayStatus.ESTOP, "KeyboardInterrupt"
            run_end_reason = RunEndReason.ESTOP
        except Exception as exc:
            self.runtime_state.mark_error()
            logger.exception("replay failed")
            result.status, result.reason = ReplayStatus.FAULT, str(exc)
            run_end_reason = (
                RunEndReason.HARDWARE_FAULT
                if isinstance(exc, RobotError)
                else RunEndReason.POLICY_FAILURE
            )
        finally:
            if result.status == ReplayStatus.USER_QUIT:
                run_end_reason = RunEndReason.QUIT
            elif result.status == ReplayStatus.ESTOP:
                run_end_reason = RunEndReason.ESTOP
            self.runtime_state.revoke_motion(reason=run_end_reason)
            try:
                self.robot.stop()
            except Exception as exc:
                self.runtime_state.mark_error()
                logger.exception("replay stop failed")
                result.reason = f"{result.reason}; stop failed: {exc}".lstrip("; ")
                result.status = ReplayStatus.FAULT
        result.replay_data = self.capture.to_dict()
        return result

    def read_feedback(self):
        row = read_observation(None, self.robot)
        issues, deadline_ns = check_observation(row, self.runtime, ObservationNeeds())
        if issues:
            raise TimeoutError("replay feedback: " + "; ".join(issues))
        return row, deadline_ns

    def admit_replay(self, trajectory, hand_start_duration_s):
        self.robot.prepare_streaming(self.epoch)
        row, _ = self.read_feedback()
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
        if any(
            np.max(np.abs(measured - recorded)) > np.deg2rad(10)
            for measured, recorded in start_pairs
        ):
            return ReplayStatus.REJECTED, "start posture differs by more than 10 degrees", row

        with self.runtime_state.motion_lock:
            if self.runtime_state.run_id != self.epoch or not self.runtime_state.begin_motion():
                return ReplayStatus.REJECTED, "motion authority unavailable", row
        return ReplayStatus.COMPLETED, "", row

    def replay_rows(self, trajectory, period_ns):
        status, reason = ReplayStatus.COMPLETED, ""
        origin_ns = time.monotonic_ns()
        for index in range(len(trajectory)):
            slot_start_ns = origin_ns + index * period_ns
            slot_deadline_ns = slot_start_ns + period_ns
            interrupted = self.wait_replay(slot_start_ns)
            if interrupted is not None:
                status, reason = interrupted
                break
            if time.monotonic_ns() >= slot_deadline_ns:
                raise TimeoutError(f"replay slot {index} missed before observation")
            row, feedback_until_ns = self.read_feedback()
            mapped = self.action_mapper.map_joint(
                trajectory.action_arm_joint[index],
                trajectory.action_hand_joint[index],
                row.arm["qpos"][0],
            )
            arm, hand, rejection = mapped.arm_qpos, mapped.hand_qpos, mapped.rejection_reason
            if rejection is not None:
                self.capture.append(row, arm, hand)
                status, reason = ReplayStatus.REJECTED, f"replay target: {rejection}"
                break
            self.submit_row(index, row, arm, hand, min(slot_deadline_ns, feedback_until_ns))
            interrupted = self.wait_replay(slot_deadline_ns)
            if interrupted is not None:
                status, reason = interrupted
                break
        return status, reason

    def submit_row(self, index, row, arm, hand, valid_until_ns):
        captured = self.capture.append(row, arm, hand)
        if time.monotonic_ns() >= valid_until_ns:
            raise TimeoutError(f"replay slot {index} expired before submission")
        # 候选与提交尝试分开记录，send_action 返回不代表机器人已到位。
        self.capture.data["submit_started_ns"][captured] = time.monotonic_ns()
        try:
            self.robot.send_action(
                RobotCommand(self.epoch, arm, hand),
                valid_until_ns=valid_until_ns,
                execute_until_ns=row.observation_timestamp_ns
                + self.decision_age_ns,
            )
        finally:
            completed_ns = time.monotonic_ns()
            self.capture.data["submit_completed_ns"][captured] = completed_ns
        if completed_ns >= valid_until_ns:
            raise TimeoutError(f"replay slot {index} submission returned after deadline")

    def wait_replay(self, deadline_ns):
        while True:
            self.robot.check()
            signals = self.keyboard.poll(timeout=0)
            if not self.keyboard.healthy:
                self.runtime_state.request_emergency_stop()
            if self.runtime_state.estop_request:
                return ReplayStatus.ESTOP, "operator emergency stop"
            if self.runtime_state.error_state:
                return ReplayStatus.FAULT, "runtime failure"
            if self.runtime_state.quit_requested or "q" in signals:
                return ReplayStatus.USER_QUIT, "operator quit"
            if not self.runtime_state.running or self.runtime_state.run_id != self.epoch:
                cause = self.runtime_state.run_ended_reason
                if cause == RunEndReason.QUIT:
                    return ReplayStatus.USER_QUIT, "operator quit"
                return ReplayStatus.REJECTED, f"motion interrupted: {cause.name.lower()}"
            self.robot.service_feedback()
            remaining_ns = deadline_ns - time.monotonic_ns()
            if remaining_ns <= 0:
                return None
            time.sleep(min(0.005, remaining_ns / 1e9))

    def prepare_hand(self, trajectory, row, duration_s, period_ns):
        """首帧手部准备不计入回放采样。"""
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
        print(
            f"XHand start: {steps * period_ns / 1e9:g}s ramp to first target, then wait for arrival. "
            "Preparation is outside replay rows; initial hand state differs from Raw. "
            "Q: stop; ESC: emergency stop",
            flush=True,
        )
        timeout_ns = int(self.runtime.hand.home_timeout_s * 1e9)
        tolerance = np.deg2rad(self.runtime.hand.home_tolerance_deg)
        deadline_ns = 0
        for step in range(steps + 1):
            interrupted = self.wait_replay(deadline_ns)
            if interrupted is not None:
                return interrupted
            row, feedback_until_ns = self.read_feedback()
            if np.max(np.abs(row.arm["qpos"][0] - arm_hold)) > np.deg2rad(10):
                return ReplayStatus.REJECTED, "arm moved during hand start"
            progress = step / steps
            weight = progress * progress * (3 - 2 * progress)
            hand = hand_target if step == steps else hand_start + weight * (hand_target - hand_start)
            rejection = self.action_mapper.constraints.rejection(
                arm_hold, collision_hand_qpos=hand,
            )
            if rejection is not None:
                return ReplayStatus.REJECTED, f"hand start target: {rejection}"
            self.robot.send_action(
                RobotCommand(self.epoch, arm_hold, hand),
                valid_until_ns=feedback_until_ns,
                execute_until_ns=row.observation_timestamp_ns + timeout_ns,
            )
            submitted_ns = time.monotonic_ns()
            deadline_ns = submitted_ns + period_ns

        expires_ns = submitted_ns + timeout_ns
        last_sample_ns = submitted_ns
        consecutive = 0
        while True:
            interrupted = self.wait_replay(min(deadline_ns, expires_ns))
            if interrupted is not None:
                return interrupted
            if time.monotonic_ns() >= expires_ns:
                return ReplayStatus.REJECTED, "hand start convergence timed out"
            row, _ = self.read_feedback()
            if np.max(np.abs(row.arm["qpos"][0] - arm_hold)) > np.deg2rad(10):
                return ReplayStatus.REJECTED, "arm moved during hand start"
            if time.monotonic_ns() >= expires_ns:
                return ReplayStatus.REJECTED, "hand start convergence timed out"
            stamp = int(row.hand["timestamp_ns"][0])
            # 到位判定只计最终提交完成之后、时间不同的新鲜反馈。
            if stamp > last_sample_ns:
                last_sample_ns = stamp
                arrived = np.max(np.abs(row.hand["qpos"][0] - hand_target)) <= tolerance
                consecutive = consecutive + 1 if arrived else 0
                if consecutive >= 3:
                    interrupted = self.wait_replay(time.monotonic_ns())
                    if interrupted is not None:
                        return interrupted
                    if time.monotonic_ns() >= expires_ns:
                        return ReplayStatus.REJECTED, "hand start convergence timed out"
                    print("XHand start ready; replaying recorded targets.")
                    return None
            deadline_ns = time.monotonic_ns() + 10_000_000

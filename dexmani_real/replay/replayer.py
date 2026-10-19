"""Dispatch every raw target once at nominal dt, with current feedback."""

import logging
import json
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dexmani_real.planning.paths import wrap_nearest_equivalent
from dexmani_real.planning.admission import TargetAdmission
from dexmani_real.planning.collision import CollisionModel
from dexmani_real.replay.capture import ReplayRecorder
from dexmani_real.recording.storage.schema import ARM_EXECUTION_FORMAT, validate_execution_boundary
from dexmani_real.robot.action import project_arm_command, project_hand_command
from dexmani_real.robot.commands import (
    RobotCommand,
    DispatchError,
    DispatchInterrupted,
    DispatchResult,
)
from dexmani_real.runtime.observation import feedback_deadline_ns, read_observation
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
        self.epoch = self.runtime_state.run_id
        self.capture = ReplayRecorder(trajectory.num_frames)
        arm_start = self.robot.arm.execution_snapshot()
        control_metadata = self.robot.arm.execution_metadata(1 / trajectory.fps)
        self.admission = TargetAdmission(
            self.runtime, CollisionModel(static_boxes=self.runtime.environment.static_boxes)
        )
        self.previous_arm = None
        period_ns, _ = sampling_clock_ns(trajectory.fps)
        status, reason = ReplayStatus.COMPLETED, ""
        run_end_reason = RunEndReason.EXECUTOR_BOUNDARY
        try:
            if results is not None:
                results.begin_episode()
            self.robot.prepare_streaming(self.epoch)
            row = read_observation(self.runtime_state, self.runtime, self.robot)
            if row is None:
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
            if results is not None and status == ReplayStatus.COMPLETED:
                results.entered(self.epoch)
            if status == ReplayStatus.COMPLETED and hand_start_duration_s > 0:
                interrupted = self.warm_up_hand(trajectory, self.epoch, row, hand_start_duration_s)
                if interrupted is not None:
                    status, reason = interrupted.status, interrupted.reason
            if status == ReplayStatus.COMPLETED:
                origin_ns = time.monotonic_ns()
                for index in range(trajectory.num_frames):
                    slot_start_ns = origin_ns + index * period_ns
                    slot_deadline_ns = slot_start_ns + period_ns
                    interrupted = self.wait_replay(self.epoch, slot_start_ns)
                    if interrupted is not None:
                        status, reason = interrupted.status, interrupted.reason
                        break
                    if time.monotonic_ns() >= slot_deadline_ns:
                        raise DispatchError(
                            f"replay slot {index} missed before observation",
                            DispatchResult(),
                            revoked=True,
                            cause="deadline_expired",
                        )
                    row = read_observation(self.runtime_state, self.runtime, self.robot)
                    if row is None:
                        status, reason = ReplayStatus.REJECTED, "robot feedback stale"
                        break
                    arm = project_arm_command(
                        trajectory.action_arm_joint[index],
                        row.arm["qpos"][0],
                        joint_lower_rad=self.runtime.arm.joint_limit_lower,
                        joint_upper_rad=self.runtime.arm.joint_limit_upper,
                    )
                    hand = project_hand_command(
                        trajectory.action_hand_joint[index],
                        qpos_min_rad=self.runtime.hand.qpos_min_rad,
                        qpos_max_rad=self.runtime.hand.qpos_max_rad,
                    )
                    rejection = self.admission.rejection(
                        arm,
                        row.arm["qpos"][0],
                        self.previous_arm,
                        collision_hand_qpos=hand,
                    )
                    if rejection is not None:
                        self.capture.record(
                            index, row, RobotCommand(self.epoch, arm, hand), DispatchResult()
                        )
                        status, reason = ReplayStatus.REJECTED, f"replay target: {rejection}"
                        break
                    dispatch_error = None
                    submit_started_ns = submit_completed_ns = -1
                    valid_until_ns = min(slot_deadline_ns, feedback_deadline_ns(row, self.runtime))
                    try:
                        command = RobotCommand(self.epoch, arm, hand)
                        if time.monotonic_ns() >= valid_until_ns:
                            raise DispatchError(
                                f"replay slot {index} expired before submission",
                                DispatchResult(),
                                revoked=True,
                                cause="deadline_expired",
                            )
                        submit_started_ns = time.monotonic_ns()
                        try:
                            result = self.robot.send_action(
                                command,
                                valid_until_ns=valid_until_ns,
                                execute_until_ns=row.observation_timestamp_ns
                                + int(self.runtime.execution.max_decision_age_s * 1e9),
                            )
                        finally:
                            submit_completed_ns = time.monotonic_ns()
                        # A blocking SDK call can outlive the last pre-send check, including the final row.
                        if submit_completed_ns >= valid_until_ns:
                            raise DispatchError(
                                f"replay slot {index} dispatch returned after deadline",
                                result,
                                revoked=True,
                                cause="deadline_expired",
                            )
                    except (DispatchError, DispatchInterrupted) as exc:
                        result, dispatch_error = exc.result, exc
                        cancelled = isinstance(exc, DispatchInterrupted)
                        if cancelled:
                            self.runtime_state.request_emergency_stop()
                        self.runtime_state.revoke_motion(
                            reason=(
                                RunEndReason.ESTOP
                                if cancelled
                                else (
                                    RunEndReason.EXECUTOR_BOUNDARY
                                    if exc.revoked
                                    else RunEndReason.HARDWARE_FAULT
                                )
                            ),
                        )
                        try:
                            self.robot.stop()
                        except Exception as exc:
                            self.failure_details.append(
                                f"dispatch stop failed: {type(exc).__name__}: {exc}"
                            )
                            logger.exception("replay stop after dispatch failed")
                            self.runtime_state.mark_error()
                    try:
                        self.capture.record(
                            index,
                            row,
                            command,
                            result,
                            submit_started_ns=submit_started_ns,
                            submit_completed_ns=submit_completed_ns,
                        )
                    except Exception as exc:
                        self.runtime_state.mark_error()
                        self.failure_details.append(f"capture failed: {type(exc).__name__}: {exc}")
                        if dispatch_error is not None:
                            logger.exception("replay recording after dispatch failed")
                            raise dispatch_error from exc
                        raise
                    if dispatch_error is not None:
                        raise dispatch_error
                    self.previous_arm = arm.copy()
                    interrupted = self.wait_replay(self.epoch, slot_deadline_ns)
                    if interrupted is not None:
                        status, reason = interrupted.status, interrupted.reason
                        break
        except DispatchError as exc:
            result = exc.result
            reason = f"{exc}; arm={result.arm.name}, hand={result.hand.name}, command_id={result.command_id}"
            if exc.revoked:
                # A later Q cannot turn a deadline/stop failure into a clean cancellation.
                with self.runtime_state.motion_lock:
                    operator_revocation = (
                        exc.cause == "authority_revoked"
                        and self.runtime_state.run_ended_id == self.epoch
                        and self.runtime_state.run_ended_reason == RunEndReason.QUIT
                    )
                if self.runtime_state.estop_request:
                    status = ReplayStatus.ESTOP
                elif self.runtime_state.error_state:
                    status = ReplayStatus.FAULT
                elif operator_revocation:
                    status = ReplayStatus.USER_QUIT
                else:
                    status = ReplayStatus.REJECTED
            else:
                status = ReplayStatus.FAULT
                self.runtime_state.mark_error()
            run_end_reason = (
                RunEndReason.HARDWARE_FAULT if not exc.revoked else RunEndReason.EXECUTOR_BOUNDARY
            )
        except KeyboardInterrupt as exc:
            self.runtime_state.request_emergency_stop()
            status, reason = ReplayStatus.ESTOP, "KeyboardInterrupt"
            if isinstance(exc, DispatchInterrupted):
                reason += f"; dispatch={exc.result}"
            run_end_reason = RunEndReason.ESTOP
        except Exception as exc:
            logger.exception("replay failed")
            status, reason = ReplayStatus.FAULT, str(exc)
            run_end_reason = RunEndReason.POLICY_FAILURE
            self.runtime_state.mark_error()
        finally:
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
                status, reason = ReplayStatus.FAULT, f"{reason or status.value}; stop failed: {exc}"
        # Deliver physical evidence even if post-motion diagnostics cannot be formed.
        data = self.capture.to_dict()
        arm_end = None
        try:
            arm_end = dict(
                self.robot.arm.execution_snapshot(), robot_stopped=not self.robot.stop_required
            )
            validate_execution_boundary(arm_end, stopped=True)
            if arm_end["fault"]:
                raise RuntimeError(arm_end["fault"])
            if (
                arm_end["sdk_completed"] - arm_start["sdk_completed"]
                != arm_end["sdk_accepted"] - arm_start["sdk_accepted"]
            ):
                raise RuntimeError("replay contains unaccepted arm SDK calls")
            if (
                status == ReplayStatus.COMPLETED
                and arm_end["first_accepted"] - arm_start["first_accepted"]
                != arm_end["submitted"] - arm_start["submitted"]
            ):
                status, reason = (
                    ReplayStatus.REJECTED,
                    "replay target stopped before first SDK acceptance",
                )
        except Exception as exc:
            self.runtime_state.mark_error()
            self.failure_details.append(f"arm execution: {exc}")
        try:
            data["arm_execution"] = np.asarray(
                json.dumps(
                    dict(
                        format=ARM_EXECUTION_FORMAT,
                        scope="replay_including_hand_start",
                        start=arm_start,
                        end=arm_end,
                        control=control_metadata,
                    ),
                    allow_nan=False,
                )
            )
        except Exception as exc:
            self.runtime_state.mark_error()
            self.failure_details.append(
                f"arm diagnostic serialization: {type(exc).__name__}: {exc}"
            )
        if self.failure_details:
            reason = "; ".join([reason or status.value, *self.failure_details])
            status = ReplayStatus.FAULT
        data["termination_reason"] = np.asarray(reason or status.value)
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
            row = read_observation(self.runtime_state, self.runtime, self.robot)
            if row is None:
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
                rejection = self.admission.rejection(
                    arm_hold,
                    row.arm["qpos"][0],
                    arm_hold,
                    collision_hand_qpos=hand,
                )
                if rejection is not None:
                    return ReplayOutcome(
                        ReplayStatus.REJECTED, reason=f"hand start target: {rejection}"
                    )
                stamp = self.robot.send_action(
                    RobotCommand(epoch, arm_hold, hand),
                    valid_until_ns=feedback_deadline_ns(row, self.runtime),
                    execute_until_ns=row.observation_timestamp_ns
                    + int(self.runtime.hand.home_timeout_s * 1e9),
                ).timestamp_ns
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
                # Only distinct fresh samples after the final dispatch establish arrival.
                if consecutive >= 3:
                    print("XHand start ready; replaying recorded targets.", flush=True)
                    return None
            deadline_ns = time.monotonic_ns() + 10_000_000

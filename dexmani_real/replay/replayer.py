"""Dispatch every raw target once at nominal dt, with current feedback."""

import logging
import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dexmani_real.planning.paths import wrap_nearest_equivalent
from dexmani_real.replay.capture import ReplayRecorder
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.projection import project_arm_command, project_hand_command
from dexmani_real.robot.robot import DispatchError, DispatchInterrupted, DispatchResult
from dexmani_real.runtime.observation import feedback_deadline_ns, read_observation
from dexmani_real.runtime.operator_input import OperatorCommand
from dexmani_real.runtime.safety import RunEndReason, begin_motion, revoke_motion
from dexmani_real.utils.control_clock import sampling_clock_ns


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


def _wait_replay(shared, keyboard, epoch, deadline_ns, robot):
    """Keep operator input and lifecycle checks active during bounded waits."""
    while True:
        robot.check()
        signals = keyboard.poll(timeout=0)
        if not keyboard.healthy:
            shared.estop_request = True
        if shared.estop_request:
            return ReplayOutcome(ReplayStatus.ESTOP, reason="operator emergency stop")
        if shared.error_state:
            return ReplayOutcome(ReplayStatus.FAULT, reason="runtime failure")
        if shared.quit_requested or OperatorCommand.QUIT in signals:
            return ReplayOutcome(ReplayStatus.USER_QUIT, reason="operator quit")
        if not shared.running or int(shared.run_id) != epoch:
            cause = RunEndReason(int(shared.run_ended_reason))
            if cause == RunEndReason.QUIT:
                return ReplayOutcome(ReplayStatus.USER_QUIT, reason="operator quit")
            return ReplayOutcome(
                ReplayStatus.REJECTED, reason=f"motion interrupted: {cause.name.lower()}"
            )
        remaining_ns = deadline_ns - time.monotonic_ns()
        if remaining_ns <= 0:
            return None
        time.sleep(min(0.005, remaining_ns / 1e9))


def _warm_up_hand(shared, runtime, trajectory, keyboard, epoch, row, duration_s, robot):
    """Prepare frame zero without adding synthetic rows to the recorded trajectory."""
    arm_hold = project_arm_command(
        row.arm["qpos"][0],
        row.arm["qpos"][0],
        joint_lower_rad=runtime.arm.joint_limit_lower,
        joint_upper_rad=runtime.arm.joint_limit_upper,
    )
    hand_start = project_hand_command(
        row.hand["qpos"][0],
        qpos_min_rad=runtime.hand.qpos_min_rad,
        qpos_max_rad=runtime.hand.qpos_max_rad,
    )
    hand_target = project_hand_command(
        trajectory.action_hand_joint[0],
        qpos_min_rad=runtime.hand.qpos_min_rad,
        qpos_max_rad=runtime.hand.qpos_max_rad,
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
        interrupted = _wait_replay(shared, keyboard, epoch, deadline_ns, robot)
        if interrupted is not None:
            return interrupted
        row = read_observation(shared, runtime, robot)
        if row is None:
            return ReplayOutcome(ReplayStatus.REJECTED, reason="hand start feedback stale")
        if np.max(np.abs(row.arm["qpos"][0] - arm_hold)) > np.deg2rad(10):
            return ReplayOutcome(ReplayStatus.REJECTED, reason="arm moved during hand start")
        if final_stamp is None:
            progress = step / steps
            weight = progress * progress * (3 - 2 * progress)
            hand = (
                hand_target if step == steps else hand_start + weight * (hand_target - hand_start)
            )
            stamp = robot.send_action(
                RobotCommand(epoch, arm_hold, hand),
                valid_until_ns=feedback_deadline_ns(row, runtime),
            ).timestamp_ns
            if step == steps:
                final_stamp = stamp
                last_sample = stamp
                robot.end_hand_send_interval()
            step += 1
            deadline_ns = stamp + period_ns
            continue

        if time.monotonic_ns() >= final_stamp + int(runtime.hand.home_timeout_s * 1e9):
            return ReplayOutcome(ReplayStatus.REJECTED, reason="hand start convergence timed out")
        sample = int(row.hand["timestamp_ns"][0])
        if sample > last_sample:
            last_sample = sample
            if np.max(np.abs(row.hand["qpos"][0] - hand_target)) <= np.deg2rad(
                runtime.hand.home_tolerance_deg
            ):
                consecutive += 1
            else:
                consecutive = 0
            # Only distinct fresh samples after the final dispatch establish arrival.
            if consecutive >= 3:
                print("XHand start ready; replaying recorded targets.", flush=True)
                return None
        deadline_ns = time.monotonic_ns() + 10_000_000


def replay_targets(
    shared, runtime, trajectory, keyboard, *, robot, hand_start_duration_s, results=None
):
    failure_details = []
    epoch = int(shared.run_id)
    capture = ReplayRecorder(trajectory.num_frames)
    period_ns, _ = sampling_clock_ns(trajectory.fps)
    status, reason = ReplayStatus.COMPLETED, ""
    run_end_reason = RunEndReason.EXECUTOR_BOUNDARY
    try:
        if results is not None:
            results.begin_episode()
        robot.prepare_streaming(epoch)
        row = read_observation(shared, runtime, robot)
        if row is None:
            status, reason = ReplayStatus.REJECTED, "fresh robot feedback unavailable"
        else:
            # Reposition separately with operator oversight; never synthesize replay rows.
            start_arm = wrap_nearest_equivalent(
                trajectory.arm_qpos[0],
                row.arm["qpos"][0],
                runtime.arm.joint_limit_lower,
                runtime.arm.joint_limit_upper,
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
                or np.any(target < runtime.hand.qpos_min_rad)
                or np.any(target > runtime.hand.qpos_max_rad)
            ):
                status, reason = ReplayStatus.REJECTED, "hand preparation target violates limits"
        if status == ReplayStatus.COMPLETED:
            with shared.motion_lock:
                if int(shared.run_id) != epoch or not begin_motion(shared):
                    status, reason = ReplayStatus.REJECTED, "motion authority unavailable"
                else:
                    epoch = int(shared.run_id)
        if results is not None and status == ReplayStatus.COMPLETED:
            results.entered(epoch)
        if status == ReplayStatus.COMPLETED and hand_start_duration_s > 0:
            interrupted = _warm_up_hand(
                shared, runtime, trajectory, keyboard, epoch, row, hand_start_duration_s, robot
            )
            if interrupted is not None:
                status, reason = interrupted.status, interrupted.reason
        if status == ReplayStatus.COMPLETED:
            origin_ns = time.monotonic_ns()
            for index in range(trajectory.num_frames):
                slot_start_ns = origin_ns + index * period_ns
                slot_deadline_ns = slot_start_ns + period_ns
                interrupted = _wait_replay(shared, keyboard, epoch, slot_start_ns, robot)
                if interrupted is not None:
                    status, reason = interrupted.status, interrupted.reason
                    break
                if time.monotonic_ns() >= slot_deadline_ns:
                    raise DispatchError(
                        f"replay slot {index} missed before observation", DispatchResult(),
                        revoked=True, cause="deadline_expired",
                    )
                row = read_observation(shared, runtime, robot)
                if row is None:
                    status, reason = ReplayStatus.REJECTED, "robot feedback stale"
                    break
                arm = project_arm_command(
                    trajectory.action_arm_joint[index],
                    row.arm["qpos"][0],
                    joint_lower_rad=runtime.arm.joint_limit_lower,
                    joint_upper_rad=runtime.arm.joint_limit_upper,
                )
                hand = project_hand_command(
                    trajectory.action_hand_joint[index],
                    qpos_min_rad=runtime.hand.qpos_min_rad,
                    qpos_max_rad=runtime.hand.qpos_max_rad,
                )
                dispatch_error = None
                submit_started_ns = submit_completed_ns = -1
                valid_until_ns = min(slot_deadline_ns, feedback_deadline_ns(row, runtime))
                try:
                    command = RobotCommand(epoch, arm, hand)
                    if time.monotonic_ns() >= valid_until_ns:
                        raise DispatchError(
                            f"replay slot {index} expired before submission", DispatchResult(),
                            revoked=True, cause="deadline_expired",
                        )
                    submit_started_ns = time.monotonic_ns()
                    try:
                        result = robot.send_action(command, valid_until_ns=valid_until_ns)
                    finally:
                        submit_completed_ns = time.monotonic_ns()
                    # A blocking SDK call can outlive the last pre-send check, including the final row.
                    if submit_completed_ns >= valid_until_ns:
                        raise DispatchError(
                            f"replay slot {index} dispatch returned after deadline", result,
                            revoked=True, cause="deadline_expired",
                        )
                except (DispatchError, DispatchInterrupted) as exc:
                    result, dispatch_error = exc.result, exc
                    cancelled = isinstance(exc, DispatchInterrupted)
                    if cancelled:
                        shared.estop_request = True
                    revoke_motion(
                        shared,
                        reason=RunEndReason.ESTOP
                        if cancelled
                        else RunEndReason.EXECUTOR_BOUNDARY
                        if exc.revoked
                        else RunEndReason.HARDWARE_FAULT,
                    )
                    try:
                        robot.stop()
                    except Exception as exc:
                        failure_details.append(f"dispatch stop failed: {type(exc).__name__}: {exc}")
                        logger.exception("replay stop after dispatch failed")
                        shared.error_state = True
                try:
                    capture.record(
                        index,
                        row.arm["qpos"][0],
                        arm if result.arm else np.full(7, np.nan),
                        hand if result.hand else np.full(12, np.nan),
                        dispatch_status=(int(result.arm), int(result.hand)),
                        hand_qpos=row.hand["qpos"][0],
                        arm_observation_ns=int(row.arm["timestamp_ns"][0]),
                        hand_observation_ns=int(row.hand["timestamp_ns"][0]),
                        submit_started_ns=submit_started_ns,
                        submit_completed_ns=submit_completed_ns,
                    )
                except Exception as exc:
                    shared.error_state = True
                    failure_details.append(f"capture failed: {type(exc).__name__}: {exc}")
                    if dispatch_error is not None:
                        logger.exception("replay recording after dispatch failed")
                        raise dispatch_error from exc
                    raise
                if dispatch_error is not None:
                    raise dispatch_error
                interrupted = _wait_replay(
                    shared, keyboard, epoch, slot_deadline_ns, robot
                )
                if interrupted is not None:
                    status, reason = interrupted.status, interrupted.reason
                    break
    except DispatchError as exc:
        result = exc.result
        reason = (
            f"{exc}; arm={result.arm.name}, hand={result.hand.name}, arm_code={result.arm_code}"
        )
        if exc.revoked:
            # A later Q cannot turn a deadline/stop failure into a clean cancellation.
            with shared.motion_lock:
                operator_revocation = (
                    exc.cause == "authority_revoked"
                    and int(shared.run_ended_id) == epoch
                    and int(shared.run_ended_reason) == int(RunEndReason.QUIT)
                )
            status = (
                ReplayStatus.ESTOP
                if shared.estop_request
                else ReplayStatus.FAULT
                if shared.error_state
                else ReplayStatus.USER_QUIT
                if operator_revocation
                else ReplayStatus.REJECTED
            )
        else:
            status = ReplayStatus.FAULT
            shared.error_state = True
        run_end_reason = (
            RunEndReason.HARDWARE_FAULT if not exc.revoked else RunEndReason.EXECUTOR_BOUNDARY
        )
    except KeyboardInterrupt as exc:
        shared.estop_request = True
        status, reason = ReplayStatus.ESTOP, "KeyboardInterrupt"
        if isinstance(exc, DispatchInterrupted):
            reason += f"; dispatch={exc.result}"
        run_end_reason = RunEndReason.ESTOP
    except Exception as exc:
        logger.exception("replay failed")
        status, reason = ReplayStatus.FAULT, str(exc)
        run_end_reason = RunEndReason.POLICY_FAILURE
        shared.error_state = True
    finally:
        if status == ReplayStatus.USER_QUIT:
            run_end_reason = RunEndReason.QUIT
        elif status == ReplayStatus.ESTOP:
            run_end_reason = RunEndReason.ESTOP
        revoke_motion(shared, reason=run_end_reason)
        try:
            robot.stop()
        except Exception as exc:
            shared.error_state = True
            logger.exception("replay stop failed")
            status, reason = ReplayStatus.FAULT, f"{reason or status.value}; stop failed: {exc}"
    if failure_details:
        reason = "; ".join([reason or status.value, *failure_details])
        status = ReplayStatus.FAULT
    data = capture.to_dict()
    data["termination_reason"] = np.asarray(reason or status.value)
    return ReplayOutcome(status, data, reason)

"""Dispatch every raw target once at nominal dt, with current feedback."""

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.paths import wrap_nearest_equivalent
from dexmani_real.replay.capture import ReplayRecorder
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.projection import project_arm_command, project_hand_command
from dexmani_real.robot.robot import DispatchError
from dexmani_real.runtime.observation import read_observation
from dexmani_real.runtime.operator_input import OperatorCommand
from dexmani_real.runtime.safety import RunEndReason, begin_motion, revoke_motion
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


class ReplayStatus(str, Enum):
    """Terminal state of one replay attempt."""

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


def _wait_replay(shared, keyboard, epoch, deadline, robot):
    """Keep operator input and lifecycle checks active during bounded waits."""
    while True:
        robot.check()
        signals = keyboard.poll(timeout=0)
        if not keyboard.healthy:
            shared.estop_request.value = True
        if shared.estop_request.value:
            return ReplayOutcome(ReplayStatus.ESTOP, reason="operator emergency stop")
        if shared.error_state.value:
            return ReplayOutcome(ReplayStatus.FAULT, reason="runtime failure")
        if shared.quit_requested.value or OperatorCommand.QUIT in signals:
            return ReplayOutcome(ReplayStatus.USER_QUIT, reason="operator quit")
        if not shared.is_running.value or int(shared.run_id.value) != epoch:
            cause = RunEndReason(int(shared.run_ended_reason.value))
            if cause == RunEndReason.QUIT:
                return ReplayOutcome(ReplayStatus.USER_QUIT, reason="operator quit")
            return ReplayOutcome(
                ReplayStatus.REJECTED, reason=f"motion interrupted: {cause.name.lower()}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(0.005, remaining))


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
    period = 1 / trajectory.fps
    print(
        f"XHand start: {steps * period:g}s ramp to first target, then wait for arrival. "
        "Preparation is outside replay rows; initial hand state differs from Raw. "
        "Q: stop; ESC: emergency stop",
        flush=True,
    )
    step = 0
    deadline = 0.0
    final_stamp = None
    last_sample = 0
    consecutive = 0
    while True:
        interrupted = _wait_replay(shared, keyboard, epoch, deadline, robot)
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
            stamp = robot.send_action(RobotCommand(epoch, arm_hold, hand)).timestamp_ns
            if step == steps:
                final_stamp = stamp
                last_sample = stamp
            step += 1
            deadline = stamp / 1e9 + period
            continue

        if time.monotonic() >= final_stamp / 1e9 + runtime.hand.home_timeout_s:
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
        deadline = time.monotonic() + 0.01


def replay_targets(shared, runtime, trajectory, keyboard, *, robot, hand_start_duration_s):
    capture = ReplayRecorder(trajectory.num_frames)
    status, reason = ReplayStatus.COMPLETED, ""
    run_end_reason = RunEndReason.EXECUTOR_BOUNDARY
    try:
        fk = make_arm_fk()
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
            for measured, recorded in (
                (row.arm["qpos"][0], start_arm),
                (row.hand["qpos"][0], trajectory.hand_qpos[0]),
            ):
                if np.max(np.abs(measured - recorded)) > np.deg2rad(10):
                    status, reason = (
                        ReplayStatus.REJECTED,
                        "start posture differs by more than 10 degrees",
                    )
                    break
        if status == ReplayStatus.COMPLETED and not begin_motion(shared):
            status, reason = ReplayStatus.REJECTED, "motion authority unavailable"
        epoch = int(shared.run_id.value)
        if status == ReplayStatus.COMPLETED and hand_start_duration_s > 0:
            interrupted = _warm_up_hand(
                shared, runtime, trajectory, keyboard, epoch, row, hand_start_duration_s, robot
            )
            if interrupted is not None:
                status, reason = interrupted.status, interrupted.reason
        if status == ReplayStatus.COMPLETED:
            for index in range(trajectory.num_frames):
                interrupted = _wait_replay(shared, keyboard, epoch, 0.0, robot)
                if interrupted is not None:
                    status, reason = interrupted.status, interrupted.reason
                    break
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
                try:
                    result = robot.send_action(RobotCommand(epoch, arm, hand))
                except DispatchError as exc:
                    result, dispatch_error = exc.result, exc
                    # Fence before recording; finally owns the immediate stop on exit.
                    revoke_motion(
                        shared,
                        reason=RunEndReason.EXECUTOR_BOUNDARY
                        if exc.revoked
                        else RunEndReason.HARDWARE_FAULT,
                    )
                stamp = result.timestamp_ns or time.monotonic_ns()
                pos, rot = fk.compute(row.arm["qpos"][0])
                capture.record(
                    index,
                    row.arm["qpos"][0],
                    pos,
                    rot,
                    arm if result.arm else np.full(7, np.nan),
                    hand if result.hand else np.full(12, np.nan),
                    stamp / 1e9,
                    dispatch_status=(int(result.arm), int(result.hand)),
                    hand_qpos=row.hand["qpos"][0],
                    arm_tracking_error=float(np.max(np.abs(arm - row.arm["qpos"][0]))),
                )
                if dispatch_error is not None:
                    raise dispatch_error
                interrupted = _wait_replay(
                    shared, keyboard, epoch, stamp / 1e9 + 1 / trajectory.fps, robot
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
            status = (
                ReplayStatus.ESTOP
                if shared.estop_request.value
                else ReplayStatus.FAULT
                if shared.error_state.value
                else ReplayStatus.USER_QUIT
                if shared.quit_requested.value
                or int(shared.run_ended_reason.value) == int(RunEndReason.QUIT)
                else ReplayStatus.REJECTED
            )
        else:
            status = ReplayStatus.FAULT
            shared.error_state.value = True
        run_end_reason = (
            RunEndReason.HARDWARE_FAULT if not exc.revoked else RunEndReason.EXECUTOR_BOUNDARY
        )
    except KeyboardInterrupt:
        shared.estop_request.value = True
        raise
    except Exception as exc:
        status, reason = ReplayStatus.FAULT, str(exc)
        run_end_reason = RunEndReason.POLICY_FAILURE
        shared.error_state.value = True
    finally:
        if status == ReplayStatus.USER_QUIT:
            run_end_reason = RunEndReason.QUIT
        elif status == ReplayStatus.ESTOP:
            run_end_reason = RunEndReason.ESTOP
        revoke_motion(shared, reason=run_end_reason)
        try:
            robot.stop()
        except Exception as exc:
            shared.error_state.value = True
            logger.exception("replay stop failed")
            status, reason = ReplayStatus.FAULT, f"{reason or status.value}; stop failed: {exc}"
    data = capture.to_dict()
    data["termination_reason"] = np.asarray(reason or status.value)
    return ReplayOutcome(status, data, reason)

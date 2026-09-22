"""Publish every raw target once at nominal dt, with current feedback."""

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.paths import wrap_nearest_equivalent
from dexmani_real.replay.capture import ReplayRecorder
from dexmani_real.robot.commands import RobotCommand, publish_command
from dexmani_real.robot.projection import project_arm_command, project_hand_command
from dexmani_real.runtime.observation import read_observation
from dexmani_real.runtime.operator_input import OperatorCommand
from dexmani_real.runtime.safety import begin_motion, revoke_motion


class ReplayStatus(str, Enum):
    """Terminal state of one replay attempt."""

    COMPLETED = "completed"
    USER_QUIT = "user_quit"
    REJECTED = "rejected"
    ESTOP = "estop"
    FAULT = "fault"
    CLEANUP_FAILED = "cleanup_failed"


@dataclass(frozen=True)
class ReplayOutcome:
    """Replay result, including any samples captured before it stopped."""

    status: ReplayStatus
    replay_data: dict[str, np.ndarray] | None = None
    reason: str = ""

    @property
    def successful(self) -> bool:
        return self.status in (ReplayStatus.COMPLETED, ReplayStatus.USER_QUIT)


def replay_targets(shared, runtime, trajectory, keyboard):
    capture = ReplayRecorder(trajectory.num_frames)
    status, reason = ReplayStatus.COMPLETED, ""
    fk = make_arm_fk()
    row = read_observation(shared, runtime)
    if row is None:
        return ReplayOutcome(ReplayStatus.REJECTED, reason="fresh robot feedback unavailable")
    # Replay starts near the recorded posture; reposition separately with operator oversight.
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
            return ReplayOutcome(
                ReplayStatus.REJECTED, reason="start posture differs by more than 10 degrees"
            )
    if not begin_motion(shared):
        return ReplayOutcome(ReplayStatus.REJECTED, reason="motion authority unavailable")
    epoch = int(shared.run_id.value)
    try:
        for index in range(trajectory.num_frames):
            signals = keyboard.poll(timeout=0)
            if not keyboard.healthy:
                shared.estop_request.value = True
            if shared.estop_request.value:
                status, reason = ReplayStatus.ESTOP, "operator emergency stop"
                break
            if shared.error_state.value or int(shared.run_id.value) != epoch:
                status, reason = ReplayStatus.FAULT, "hardware fault or epoch changed"
                break
            if OperatorCommand.QUIT in signals:
                status = ReplayStatus.USER_QUIT
                break
            row = read_observation(shared, runtime)
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
            stamp = publish_command(shared, RobotCommand(epoch, arm, hand))
            if not stamp:
                status, reason = ReplayStatus.REJECTED, "motion authority revoked"
                break
            pos, rot = fk.compute(row.arm["qpos"][0])
            capture.record(
                index,
                row.arm["qpos"][0],
                pos,
                rot,
                arm,
                hand,
                stamp / 1e9,
                hand_qpos=row.hand["qpos"][0],
                arm_tracking_error=float(np.max(np.abs(arm - row.arm["qpos"][0]))),
            )
            deadline = stamp / 1e9 + 1 / trajectory.fps
            while time.monotonic() < deadline and not shared.estop_request.value:
                time.sleep(min(0.005, max(0, deadline - time.monotonic())))
    finally:
        revoke_motion(shared)
    return ReplayOutcome(status, capture.to_dict(), reason)

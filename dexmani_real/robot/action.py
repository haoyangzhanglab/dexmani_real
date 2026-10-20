"""Map absolute targets to joint commands; downstream execution owns motion authority."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dexmani_real.planning.kinematics.pose import Pose
from dexmani_real.planning.kinematics.ik import (
    IKResult,
    make_online_ik_solver,
)
from dexmani_real.planning.paths import wrap_nearest_equivalent
from dexmani_real.planning.target import TargetChecker


@dataclass(frozen=True)
class ActionIntent:
    """Absolute joint targets or an xArm-base EEF pose in meters with WXYZ rotation."""

    mode: Literal["joint", "eef"]
    arm: np.ndarray | Pose
    hand: np.ndarray | None


@dataclass(frozen=True)
class ActionResult:
    arm_qpos: np.ndarray | None
    hand_qpos: np.ndarray | None
    ik_result: IKResult | None = None
    rejection_reason: str | None = None
    # Projected Cartesian target; Teleop commits its EMA only after dispatch.
    eef_pose: Pose | None = None


class ActionMapper:
    def __init__(self, runtime, *, solver=None, collision_model):
        self.runtime, self.solver = runtime, solver
        self.target_checker = TargetChecker(
            runtime,
            collision_model,
            dynamic_rejection=solver.dynamic_rejection if solver is not None else None,
        )

    @classmethod
    def for_mode(cls, runtime, mode):
        """Online geometry permits table contact; HOME owns its separate path scene."""
        if mode not in ("joint", "eef"):
            raise ValueError("action mode must be joint or eef")
        if mode == "eef":
            solver = make_online_ik_solver(runtime)
            return cls(runtime, solver=solver, collision_model=solver.collision_model)
        from dexmani_real.planning.collision import CollisionModel

        return cls(
            runtime, collision_model=CollisionModel(static_boxes=runtime.environment.static_boxes)
        )

    def joint_rejection(self, target, hand, current, previous):
        physical_hand = hand if hand is not None else np.deg2rad(self.runtime.hand.home_qpos_deg)
        return self.target_checker.rejection(
            target, current, previous, collision_hand_qpos=physical_hand
        )

    def map(self, intent, current_arm_qpos, previous_arm_command_qpos=None):
        if intent.mode not in ("joint", "eef"):
            raise ValueError("action mode must be joint or eef")
        if intent.mode == "joint":
            arm = np.asarray(intent.arm, dtype=np.float64)
            if arm.shape != (7,) or not np.isfinite(arm).all():
                raise ValueError("joint intent must be a finite (7,) vector")
        elif not isinstance(intent.arm, Pose):
            raise TypeError("EEF intent must be an xArm-base Pose")
        current = np.asarray(current_arm_qpos, dtype=np.float64)
        previous = (
            current
            if previous_arm_command_qpos is None
            else np.asarray(previous_arm_command_qpos, dtype=np.float64)
        )
        if any(q.shape != (7,) or not np.isfinite(q).all() for q in (current, previous)):
            raise ValueError("current and previous arm state must be finite (7,) vectors")
        cfg = self.runtime
        if cfg.control.hand_enabled:
            if intent.hand is None:
                raise ValueError("hand-enabled action mapping requires a hand intent")
        elif intent.hand is not None:
            raise ValueError("hand-disabled action mapping must not include a hand intent")
        hand = None
        if intent.hand is not None:
            hand = project_hand_command(
                intent.hand, qpos_min_rad=cfg.hand.qpos_min_rad, qpos_max_rad=cfg.hand.qpos_max_rad
            )
        if intent.mode == "joint":
            target = project_arm_command(
                arm,
                current,
                joint_lower_rad=cfg.arm.joint_limit_lower,
                joint_upper_rad=cfg.arm.joint_limit_upper,
            )
            reason = self.joint_rejection(target, hand, current, previous)
            return ActionResult(
                target if reason is None else None, hand, rejection_reason=reason
            )
        if self.solver is None:
            raise ValueError("EEF mapping requires an online IK solver")
        # Project an owned pose so the source target and teleop EMA reference stay unchanged.
        pose = intent.arm.copy()
        workspace = self.target_checker.workspace
        pose.p = np.clip(pose.p, workspace[:, 0], workspace[:, 1])
        self.solver.collision_model.set_hand_qpos(
            hand if hand is not None else np.deg2rad(cfg.hand.home_qpos_deg)
        )
        result = self.solver.solve(pose, current, previous)
        reason = (
            result.reason
            if not result.success
            else self.joint_rejection(result.qpos, hand, current, previous)
        )
        return ActionResult(
            result.qpos if reason is None else None,
            hand,
            ik_result=result,
            rejection_reason=reason,
            eef_pose=pose,
        )

    def control_from_command(self, command, mode):
        if mode == "joint":
            return np.concatenate((command.arm_qpos, command.hand_qpos))
        from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base

        pose = compute_eef_pose_history_xarm_base(command.arm_qpos[None])[0]
        return np.concatenate((pose, command.hand_qpos))

    def frozen_rejection(self, command, current, previous, mode):
        previous = current if previous is None else previous
        # Reprojection after preparation would change the reserved equivalent branch.
        projected = (
            command.arm_qpos
            if mode == "eef"
            else project_arm_command(
                command.arm_qpos,
                current,
                joint_lower_rad=self.runtime.arm.joint_limit_lower,
                joint_upper_rad=self.runtime.arm.joint_limit_upper,
            )
        )
        if not np.array_equal(projected, command.arm_qpos):
            return "frozen_equivalent_branch_changed"
        return self.joint_rejection(command.arm_qpos, command.hand_qpos, current, previous)


def project_arm_command(target_arm_qpos, reference_arm_qpos, *, joint_lower_rad, joint_upper_rad):
    equivalent = wrap_nearest_equivalent(
        target_arm_qpos, reference_arm_qpos, joint_lower_rad, joint_upper_rad
    )
    return np.clip(equivalent, joint_lower_rad, joint_upper_rad)


def project_hand_command(target_hand_qpos, *, qpos_min_rad, qpos_max_rad):
    target = np.asarray(target_hand_qpos, dtype=np.float64)
    if target.shape != (12,) or not np.isfinite(target).all():
        raise ValueError("hand preparation requires a finite (12,) target")
    return np.clip(target, qpos_min_rad, qpos_max_rad)

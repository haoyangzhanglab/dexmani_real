"""Map absolute targets to joint commands; downstream execution owns motion authority."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dexmani_real.planning.kinematics.ik import (
    IKResult,
    OnlineIKConfig,
    make_online_ik_solver,
)
from dexmani_real.planning.kinematics.pose import Pose
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
    # 投影后的笛卡尔目标；Teleop 仅在公共提交正常返回后更新 EMA。
    eef_pose: Pose | None = None


class ActionMapper:
    def __init__(
        self,
        target_checker,
        *,
        hand_enabled,
        hand_lower_rad,
        hand_upper_rad,
        home_hand_qpos,
        solver=None,
    ):
        self.target_checker, self.solver = target_checker, solver
        self.hand_enabled = hand_enabled
        self.hand_lower = np.array(hand_lower_rad, dtype=np.float64, copy=True)
        self.hand_upper = np.array(hand_upper_rad, dtype=np.float64, copy=True)
        self.home_hand_qpos = np.array(home_hand_qpos, dtype=np.float64, copy=True)
        if any(
            q.shape != (12,) or not np.isfinite(q).all()
            for q in (self.hand_lower, self.hand_upper, self.home_hand_qpos)
        ) or np.any(self.hand_lower > self.hand_upper):
            raise ValueError("action hand limits and home must be finite (12,) vectors")

    @classmethod
    def for_mode(cls, mode, *, arm, hand, motion, environment):
        """Online geometry permits table contact; HOME owns its separate path scene."""
        if mode not in ("joint", "eef"):
            raise ValueError("action mode must be joint or eef")
        from dexmani_real.planning.collision import CollisionModel

        ik_config = OnlineIKConfig(
            max_pose_error_pos_m=motion.ik_max_pose_error_pos_m,
            max_pose_error_rot_rad=motion.ik_max_pose_error_rot_rad,
            operational_joint_lower_rad=tuple(arm.joint_limit_lower),
            operational_joint_upper_rad=tuple(arm.joint_limit_upper),
        )
        static_boxes = environment.static_boxes
        solver = make_online_ik_solver(ik_config, static_boxes=static_boxes) if mode == "eef" else None
        collision = (
            solver.collision_model
            if solver is not None
            else CollisionModel(static_boxes=static_boxes)
        )
        checker = TargetChecker(
            collision,
            joint_lower_rad=ik_config.operational_joint_lower_rad,
            joint_upper_rad=ik_config.operational_joint_upper_rad,
            workspace=environment.workspace.as_array(),
            workspace_margin_m=environment.eef_workspace_margin_m,
            dynamic_rejection=solver.dynamic_rejection if solver is not None else None,
        )
        return cls(
            checker,
            solver=solver,
            hand_enabled=hand.enabled,
            hand_lower_rad=hand.qpos_min_rad,
            hand_upper_rad=hand.qpos_max_rad,
            home_hand_qpos=np.deg2rad(hand.home_qpos_deg),
        )

    def joint_rejection(self, target, hand, current, previous):
        physical_hand = hand if hand is not None else self.home_hand_qpos
        return self.target_checker.rejection(
            target, current, previous, collision_hand_qpos=physical_hand
        )

    def prepare_inputs(self, hand, current_arm_qpos, previous_arm_command_qpos):
        current = np.asarray(current_arm_qpos, dtype=np.float64)
        previous = (
            current
            if previous_arm_command_qpos is None
            else np.asarray(previous_arm_command_qpos, dtype=np.float64)
        )
        if any(q.shape != (7,) or not np.isfinite(q).all() for q in (current, previous)):
            raise ValueError("current and previous arm state must be finite (7,) vectors")
        if self.hand_enabled:
            if hand is None:
                raise ValueError("hand-enabled action mapping requires a hand intent")
        elif hand is not None:
            raise ValueError("hand-disabled action mapping must not include a hand intent")
        if hand is not None:
            hand = project_hand_command(
                hand, qpos_min_rad=self.hand_lower, qpos_max_rad=self.hand_upper
            )
        return current, previous, hand

    def prepare_joint_target(self, arm, hand, current_arm_qpos, previous_arm_command_qpos=None):
        """保留投影候选；拒绝原因必须在发送前处理。"""
        arm = np.asarray(arm, dtype=np.float64)
        if arm.shape != (7,) or not np.isfinite(arm).all():
            raise ValueError("joint intent must be a finite (7,) vector")
        current, previous, hand = self.prepare_inputs(
            hand, current_arm_qpos, previous_arm_command_qpos
        )
        target = project_arm_command(
            arm,
            current,
            joint_lower_rad=self.target_checker.limits[:, 0],
            joint_upper_rad=self.target_checker.limits[:, 1],
        )
        return target, hand, self.joint_rejection(target, hand, current, previous)

    def map(self, intent, current_arm_qpos, previous_arm_command_qpos=None):
        if intent.mode not in ("joint", "eef"):
            raise ValueError("action mode must be joint or eef")
        if intent.mode == "joint":
            target, hand, reason = self.prepare_joint_target(
                intent.arm, intent.hand, current_arm_qpos, previous_arm_command_qpos
            )
            return ActionResult(target if reason is None else None, hand, rejection_reason=reason)
        if not isinstance(intent.arm, Pose):
            raise TypeError("EEF intent must be an xArm-base Pose")
        current, previous, hand = self.prepare_inputs(
            intent.hand, current_arm_qpos, previous_arm_command_qpos
        )
        if self.solver is None:
            raise ValueError("EEF mapping requires an online IK solver")
        # Project an owned pose so the source target and teleop EMA reference stay unchanged.
        pose = intent.arm.copy()
        workspace = self.target_checker.workspace
        pose.p = np.clip(pose.p, workspace[:, 0], workspace[:, 1])
        self.solver.collision_model.set_hand_qpos(
            hand if hand is not None else self.home_hand_qpos
        )
        result = self.solver.solve(
            pose,
            current,
            previous,
            endpoint_rejection=self.target_checker.endpoint_rejection,
        )
        reason = None if result.success else result.reason
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

        pose = compute_eef_pose_history_xarm_base(
            command.arm_qpos[None], arm_fk=self.target_checker.target_fk
        )[0]
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
                joint_lower_rad=self.target_checker.limits[:, 0],
                joint_upper_rad=self.target_checker.limits[:, 1],
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

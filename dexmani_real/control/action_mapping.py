"""动作映射只生成目标，执行授权由运行层管理。"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dexmani_real.control.constraints import MotionConstraints, wrap_nearest_equivalent
from dexmani_real.control.ik import IKSolver
from dexmani_real.geometry.pose import Pose


@dataclass(frozen=True)
class ActionIntent:
    """机械臂绝对关节目标或基座系 EEF 位姿，位置单位 m、旋转为 WXYZ。"""

    mode: Literal["joint", "eef"]
    arm: np.ndarray | Pose
    hand: np.ndarray | None


@dataclass(frozen=True)
class ActionResult:
    arm_qpos: np.ndarray | None
    hand_qpos: np.ndarray | None
    rejection_reason: str | None = None
    # 投影后的笛卡尔目标；Teleop 仅在公共提交正常返回后更新 EMA。
    eef_pose: Pose | None = None


class ActionMapper:
    def __init__(
        self,
        constraints,
        *,
        hand_enabled,
        hand_lower_rad,
        hand_upper_rad,
        home_hand_qpos,
        solver=None,
    ):
        self.constraints, self.solver = constraints, solver
        self.hand_enabled = hand_enabled
        self.hand_lower = np.array(hand_lower_rad, dtype=np.float64, copy=True)
        self.hand_upper = np.array(hand_upper_rad, dtype=np.float64, copy=True)
        self.home_hand_qpos = np.array(home_hand_qpos, dtype=np.float64, copy=True)

    @classmethod
    def for_mode(cls, mode, *, arm, hand, motion, environment):
        """在线允许桌面接触，HOME 使用独立场景。"""
        if mode not in ("joint", "eef"):
            raise ValueError("action mode must be joint or eef")
        from dexmani_real.control.collision import CollisionModel

        collision = CollisionModel(static_boxes=environment.static_boxes)
        constraints = MotionConstraints(
            collision,
            joint_lower_rad=arm.joint_limit_lower,
            joint_upper_rad=arm.joint_limit_upper,
            workspace=environment.workspace.as_array(),
            workspace_margin_m=environment.eef_workspace_margin_m,
        )
        solver = (
            IKSolver(
                constraints,
                position_tolerance=motion.ik_max_pose_error_pos_m,
                rotation_tolerance=motion.ik_max_pose_error_rot_rad,
            )
            if mode == "eef"
            else None
        )
        return cls(
            constraints,
            solver=solver,
            hand_enabled=hand.enabled,
            hand_lower_rad=hand.qpos_min_rad,
            hand_upper_rad=hand.qpos_max_rad,
            home_hand_qpos=np.deg2rad(hand.home_qpos_deg),
        )

    def joint_rejection(self, target, hand, current, previous):
        physical_hand = hand if hand is not None else self.home_hand_qpos
        return self.constraints.rejection(
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
        """保留投影候选，调用方必须在发送前处理拒绝原因。"""
        arm = np.asarray(arm, dtype=np.float64)
        if arm.shape != (7,) or not np.isfinite(arm).all():
            raise ValueError("joint intent must be a finite (7,) vector")
        current, previous, hand = self.prepare_inputs(
            hand, current_arm_qpos, previous_arm_command_qpos
        )
        target = project_arm_command(
            arm,
            current,
            joint_lower_rad=self.constraints.limits[:, 0],
            joint_upper_rad=self.constraints.limits[:, 1],
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
        # 复制后投影，避免改写输入目标及 Teleop 的 EMA 参考。
        pose = intent.arm.copy()
        workspace = self.constraints.workspace
        pose.p = np.clip(pose.p, workspace[:, 0], workspace[:, 1])
        result = self.solver.solve(
            pose,
            current,
            previous,
            hand_qpos=hand if hand is not None else self.home_hand_qpos,
        )
        reason = None if result.success else result.reason
        return ActionResult(
            result.qpos if reason is None else None,
            hand,
            rejection_reason=reason,
            eef_pose=pose,
        )

    def control_from_command(self, command, mode):
        if mode == "joint":
            return np.concatenate((command.arm_qpos, command.hand_qpos))
        position, rotation = self.constraints.target_fk.forward(command.arm_qpos)
        return np.concatenate((position, rotation, command.hand_qpos))

    def frozen_rejection(self, command, current, previous, mode):
        previous = current if previous is None else previous
        # 已准备的目标不能重新选择等价分支。
        projected = (
            command.arm_qpos
            if mode == "eef"
            else project_arm_command(
                command.arm_qpos,
                current,
                joint_lower_rad=self.constraints.limits[:, 0],
                joint_upper_rad=self.constraints.limits[:, 1],
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

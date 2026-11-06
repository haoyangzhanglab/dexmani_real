"""动作映射只生成目标，实际发送由运行层控制。"""

from dataclasses import dataclass

import numpy as np

from dexmani_real.control.constraints import MotionConstraints, wrap_nearest_equivalent
from dexmani_real.control.ik import IKSolver
from dexmani_real.geometry.pose import Pose


class StartPoseRejected(ValueError):
    """实测姿态无法作为本段有界动作的起点；设备会话仍可继续使用。"""


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
        *,
        joint_limits,
        hand_enabled,
        hand_lower_rad,
        hand_upper_rad,
        fixed_hand_qpos=None,
        solver=None,
    ):
        self.solver = solver
        self.joint_limits = joint_limits
        self.hand_enabled = hand_enabled
        self.hand_lower = np.array(hand_lower_rad, dtype=np.float64, copy=True) if hand_enabled else None
        self.hand_upper = np.array(hand_upper_rad, dtype=np.float64, copy=True) if hand_enabled else None
        self.fixed_hand_qpos = (
            None if fixed_hand_qpos is None
            else np.array(fixed_hand_qpos, dtype=np.float64, copy=True)
        )

    @classmethod
    def for_mode(cls, mode, *, arm, hand, motion, environment):
        """joint 只投影关节意图；EEF 使用几何约束。"""
        if mode not in ("joint", "eef"):
            raise ValueError("action mode must be joint or eef")
        solver = None
        fixed_hand_qpos = None
        if mode == "eef":
            if not hand.enabled:
                hand.validate_home_pose()
                fixed_hand_qpos = np.deg2rad(hand.home_qpos_deg)
            solver = IKSolver(
                MotionConstraints(arm, environment),
                target_speed=np.deg2rad(motion.ik_target_speed_deg_s),
                device_speed=arm.max_joint_velocity_rad_per_s,
                max_lead=np.deg2rad(motion.ik_max_lead_deg),
            )
        return cls(
            joint_limits=np.column_stack((arm.joint_limit_lower, arm.joint_limit_upper)),
            solver=solver,
            hand_enabled=hand.enabled,
            hand_lower_rad=hand.qpos_min_rad,
            hand_upper_rad=hand.qpos_max_rad,
            fixed_hand_qpos=fixed_hand_qpos,
        )

    def prepare_hand(self, hand):
        if self.hand_enabled:
            if hand is None:
                raise ValueError("hand-enabled action mapping requires a hand intent")
        elif hand is not None:
            raise ValueError("hand-disabled action mapping must not include a hand intent")
        if hand is not None:
            hand = project_hand_command(
                hand, qpos_min_rad=self.hand_lower, qpos_max_rad=self.hand_upper
            )
        return hand

    def map_joint(self, arm, hand, current_arm_qpos):
        """将关节意图投影到运行范围，不执行几何筛选。"""
        current = np.asarray(current_arm_qpos, dtype=np.float64)
        hand = self.prepare_hand(hand)
        target = project_arm_command(
            arm,
            current,
            joint_lower_rad=self.joint_limits[:, 0],
            joint_upper_rad=self.joint_limits[:, 1],
        )
        return ActionResult(target, hand)

    def map_eef(self, pose: Pose, hand, reference):
        current, previous, anchor, h = reference
        hand = self.prepare_hand(hand)
        if self.solver is None:
            raise ValueError("EEF mapping requires an online IK solver")
        # 两个调用方均交出刚构造的临时 Pose；位置与四元数已在构造时检查。
        workspace = self.solver.constraints.workspace
        pose.p = np.clip(pose.p, workspace[:, 0], workspace[:, 1])
        result = self.solver.solve(
            pose,
            current,
            previous, anchor, h=h,
            hand_qpos=hand if hand is not None else self.fixed_hand_qpos,
        )
        reason = None if result.qpos is not None else result.reason
        return ActionResult(
            result.qpos,
            hand,
            rejection_reason=reason,
            eef_pose=pose,
        )


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

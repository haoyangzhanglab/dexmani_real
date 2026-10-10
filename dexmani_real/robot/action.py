"""Source-neutral online intent to physical target realization.

Motion epochs and SDK authority remain downstream of these operational transforms.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dexmani_real.planning import Pose, XArm7MotionPlanner
from dexmani_real.planning.kinematics.ik import (
    DEFAULT_JUMP_DEG,
    IKResult,
    joint_target_rejection,
    make_online_ik_config,
)
from dexmani_real.planning.kinematics.pose import rot6d_to_quat_wxyz
from dexmani_real.robot.projection import project_arm_command, project_hand_command


@dataclass(frozen=True)
class ActionIntent:
    mode: Literal["joint", "eef"]
    arm: np.ndarray
    hand: np.ndarray | None


@dataclass(frozen=True)
class ActionRealization:
    arm_qpos: np.ndarray | None
    hand_qpos: np.ndarray | None
    ik_result: IKResult | None = None
    rejection_reason: str | None = None
    # Accepted Cartesian target feeds Teleop's explicit EMA only after dispatch.
    eef_pose: np.ndarray | None = None


class ActionRealizer:
    def __init__(self, runtime, planner=None, collision_model=None, *, check_final_target=False):
        for name, lower, upper, size in (
            ("arm", runtime.arm.joint_limit_lower, runtime.arm.joint_limit_upper, 7),
            ("hand", runtime.hand.qpos_min_rad, runtime.hand.qpos_max_rad, 12),
        ):
            lower, upper = np.asarray(lower), np.asarray(upper)
            if (
                lower.shape != (size,)
                or upper.shape != (size,)
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or np.any(lower >= upper)
            ):
                raise ValueError(f"{name} action limits must be finite ordered ({size},) vectors")
        if planner is not None or check_final_target:
            runtime.policy.workspace.validate()
        self.target_fk = None
        if check_final_target:
            from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
            self.target_fk = make_arm_fk()
        self.runtime = runtime
        self.planner = planner
        self.collision_model = collision_model if planner is None else planner.collision_model
        if planner is not None and not runtime.policy.hand_enabled:
            planner.set_hand_qpos(np.deg2rad(runtime.hand.home_qpos_deg))

    @classmethod
    def for_mode(cls, runtime, mode, *, deployment=False):
        runtime.arm.validate()
        runtime.hand.validate()
        if mode not in ("joint", "eef"):
            raise ValueError("action mode must be joint or eef")
        if mode == "eef":
            runtime.policy.workspace.validate()
        # Policy evaluation allows near-table/contact manipulation under operator
        # supervision. Its target checks use static boxes only; HOME owns table
        # path checks through its separate planner.
        environment = {}
        if deployment:
            runtime.environment.validate()
            for box in runtime.environment.static_boxes:
                box.validate()
            environment = dict(static_boxes=runtime.environment.static_boxes)
        planner = (
            XArm7MotionPlanner.create_default(
                online_ik_profile=make_online_ik_config(runtime),
                **environment,
            )
            if mode == "eef"
            else None
        )
        collision = None
        if mode == "joint":
            from dexmani_real.planning.collision import CollisionModel

            collision = CollisionModel(hand_dof=True, **environment)
        return cls(runtime, planner, collision, check_final_target=deployment)

    def _joint_rejection(self, target, hand, current, previous):
        cfg = self.runtime
        reason = (
            self.planner.online_ik_solver.dynamic_rejection(target, current, previous)
            if self.planner is not None
            else joint_target_rejection(
                target,
                current,
                previous,
                np.column_stack((cfg.arm.joint_limit_lower, cfg.arm.joint_limit_upper)),
                np.deg2rad(DEFAULT_JUMP_DEG),
            )
        )
        if reason is not None:
            return reason
        if self.collision_model is None:
            raise RuntimeError("online admission requires a collision model")
        physical_hand = hand if hand is not None else np.deg2rad(cfg.hand.home_qpos_deg)
        if self.target_fk is not None:
            physical_hand = np.asarray(physical_hand)
            if physical_hand.shape != (12,) or not np.isfinite(physical_hand).all():
                return "invalid_hand_target"
            if np.any(physical_hand < cfg.hand.qpos_min_rad) or np.any(physical_hand > cfg.hand.qpos_max_rad):
                return "hand_operational_limits"
        self.collision_model.set_hand_qpos(physical_hand)
        if self.target_fk is not None:
            if not self.collision_model.within_joint_limits(target):
                return "model_joint_limits"
            position, _ = self.target_fk.compute(target)
            bounds = cfg.policy.workspace.as_array()
            margin = cfg.execution.eef_workspace_margin_m
            if np.any(position < bounds[:, 0] - margin) or np.any(position > bounds[:, 1] + margin):
                return "final_eef_workspace"
        if self.collision_model.check_self_collision(target):
            return "self_collision"
        if self.target_fk is not None and self.collision_model.check_environment_collision(target):
            details = self.collision_model.check_environment_collision_details(target)
            contacts = ", ".join(f"{p.link_name1}/{p.link_name2}" for p in details.collision_pairs)
            return f"environment_collision: {contacts or 'contact_details_unavailable'}"
        return None

    def reset_episode(self):
        if self.planner is not None:
            self.planner.reset_episode()

    def realize(self, intent, current_arm_qpos, previous_arm_command_qpos=None):
        if intent.mode not in ("joint", "eef"):
            raise ValueError("action mode must be joint or eef")
        arm = np.asarray(intent.arm, dtype=np.float64)
        current = np.asarray(current_arm_qpos, dtype=np.float64)
        previous = (
            current
            if previous_arm_command_qpos is None
            else np.asarray(previous_arm_command_qpos, dtype=np.float64)
        )
        if arm.shape != ((7,) if intent.mode == "joint" else (9,)) or not np.isfinite(arm).all():
            raise ValueError("arm intent must have the expected shape and finite values")
        if any(q.shape != (7,) or not np.isfinite(q).all() for q in (current, previous)):
            raise ValueError("current and previous arm state must be finite (7,) vectors")
        cfg = self.runtime
        if cfg.policy.hand_enabled:
            if intent.hand is None:
                raise ValueError("hand-enabled action realization requires a hand intent")
        else:
            if intent.hand is not None:
                raise ValueError("hand-disabled action realization must not include a hand intent")
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
            reason = self._joint_rejection(target, hand, current, previous)
            return ActionRealization(
                target if reason is None else None, hand, rejection_reason=reason
            )
        if self.planner is None:
            raise ValueError("EEF realization requires an online IK planner")
        workspace = cfg.policy.workspace.as_array()
        position = np.clip(arm[:3], workspace[:, 0], workspace[:, 1])
        pose = Pose(p=position, q=rot6d_to_quat_wxyz(arm[3:]))
        self.planner.set_hand_qpos(hand if hand is not None else np.deg2rad(cfg.hand.home_qpos_deg))
        result = self.planner.solve_online_ik(pose, current, previous)
        reason = result.reason if not result.success else (
            self._joint_rejection(result.qpos, hand, current, previous)
            if self.target_fk is not None else None
        )
        return ActionRealization(
            result.qpos if reason is None else None,
            hand,
            ik_result=result,
            rejection_reason=reason,
            eef_pose=np.concatenate((position, arm[3:])),
        )

    def control_from_command(self, command, mode):
        if mode == "joint":
            return np.concatenate((command.arm_qpos, command.hand_qpos))
        from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base

        pose = compute_eef_pose_history_xarm_base(command.arm_qpos[None])[0]
        return np.concatenate((pose, command.hand_qpos))

    def frozen_rejection(self, command, current, previous, mode):
        previous = current if previous is None else previous
        # Projection is allowed at preparation only. A changed equivalent branch
        # invalidates the reservation instead of silently changing its condition.
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
        return self._joint_rejection(command.arm_qpos, command.hand_qpos, current, previous)

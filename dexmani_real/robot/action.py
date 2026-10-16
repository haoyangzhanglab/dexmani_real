"""Realize source-neutral targets; motion epochs and SDK authorization stay downstream."""

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
from dexmani_real.robot.projection import project_arm_command, project_hand_command


@dataclass(frozen=True)
class ActionIntent:
    """Absolute joint targets or an xArm-base EEF pose in meters with WXYZ rotation."""

    mode: Literal["joint", "eef"]
    arm: np.ndarray | Pose
    hand: np.ndarray | None


@dataclass(frozen=True)
class ActionRealization:
    arm_qpos: np.ndarray | None
    hand_qpos: np.ndarray | None
    ik_result: IKResult | None = None
    rejection_reason: str | None = None
    # Projected Cartesian target; Teleop commits its EMA only after dispatch.
    eef_pose: Pose | None = None


class ActionRealizer:
    def __init__(self, runtime, planner=None, collision_model=None):
        from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
        self.target_fk = make_arm_fk()
        self.runtime = runtime
        self.planner = planner
        self.collision_model = collision_model if planner is None else planner.collision_model
        if planner is not None and not runtime.policy.hand_enabled:
            planner.set_hand_qpos(np.deg2rad(runtime.hand.home_qpos_deg))

    @classmethod
    def for_mode(cls, runtime, mode):
        """Build geometry after the workflow validates the resolved robot configuration."""
        if mode not in ("joint", "eef"):
            raise ValueError("action mode must be joint or eef")
        runtime.policy.workspace.validate()
        margin = runtime.execution.eef_workspace_margin_m
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("shared final EEF workspace margin must be finite and nonnegative")
        # Online manipulation permits table contact; HOME checks table paths with a separate planner.
        runtime.environment.validate()
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

            collision = CollisionModel(**environment)
        return cls(runtime, planner, collision)

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
        self.collision_model.set_hand_qpos(physical_hand)
        if self.collision_model.check_self_collision(target):
            return "self_collision"
        return self._final_target_rejection(target)

    def _final_target_rejection(self, target):
        """Shared physical endpoint admission; this does not prove swept-path clearance."""
        if not self.collision_model.within_joint_limits(target):
            return "model_joint_limits"
        position, _ = self.target_fk.compute(target)
        bounds = self.runtime.policy.workspace.as_array()
        margin = self.runtime.execution.eef_workspace_margin_m
        if np.any(position < bounds[:, 0] - margin) or np.any(position > bounds[:, 1] + margin):
            return "final_eef_workspace"
        if self.collision_model.check_environment_collision(target):
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
        # Project an owned pose so the source target and teleop EMA reference stay unchanged.
        pose = intent.arm.copy()
        workspace = cfg.policy.workspace.as_array()
        pose.p = np.clip(pose.p, workspace[:, 0], workspace[:, 1])
        self.planner.set_hand_qpos(hand if hand is not None else np.deg2rad(cfg.hand.home_qpos_deg))
        result = self.planner.solve_online_ik(pose, current, previous)
        reason = result.reason if not result.success else (
            self._final_target_rejection(result.qpos)
        )
        return ActionRealization(
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
        return self._joint_rejection(command.arm_qpos, command.hand_qpos, current, previous)

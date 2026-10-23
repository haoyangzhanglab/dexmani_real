"""Physical endpoint checks shared by motion producers in the parent process."""

import numpy as np

from .kinematics.arm_fk import make_arm_fk
from .kinematics.ik import DEFAULT_JUMP_DEG, joint_target_rejection


class TargetChecker:
    """Check a target, not the swept arm/hand path from feedback to that target."""

    def __init__(
        self,
        collision_model,
        *,
        joint_lower_rad,
        joint_upper_rad,
        workspace,
        workspace_margin_m,
        dynamic_rejection=None,
        target_fk=None,
    ):
        self.collision_model = collision_model
        self.target_fk = make_arm_fk() if target_fk is None else target_fk
        self.limits = np.column_stack((joint_lower_rad, joint_upper_rad))
        if self.limits.shape != (7, 2) or not np.isfinite(self.limits).all() or np.any(
            self.limits[:, 0] >= self.limits[:, 1]
        ):
            raise ValueError("target limits must be finite ordered (7,2)")
        self.jump_limit = np.deg2rad(DEFAULT_JUMP_DEG)
        self.workspace = np.array(workspace, dtype=np.float64, copy=True)
        if self.workspace.shape != (3, 2) or not np.isfinite(self.workspace).all() or np.any(
            self.workspace[:, 0] >= self.workspace[:, 1]
        ):
            raise ValueError("target workspace must be finite ordered (3,2)")
        self.margin = workspace_margin_m
        if type(self.margin) not in (int, float) or not np.isfinite(self.margin) or self.margin < 0:
            raise ValueError("final EEF workspace margin must be finite and nonnegative")
        self.dynamic_rejection = dynamic_rejection

    def rejection(self, target, current, previous, *, collision_hand_qpos):
        reason = (
            self.dynamic_rejection(target, current, previous)
            if self.dynamic_rejection is not None
            else joint_target_rejection(
                target, current, previous, self.limits, self.jump_limit
            )
        )
        if reason is not None:
            return reason
        model = self.collision_model
        model.set_hand_qpos(collision_hand_qpos)
        if model.check_self_collision(target):
            return "self_collision"
        return self.endpoint_rejection(target)

    def endpoint_rejection(self, target):
        """Check model limits, FK workspace and environment with the current collision hand pose."""
        model = self.collision_model
        if not model.within_joint_limits(target):
            return "model_joint_limits"
        position, _ = self.target_fk.compute(target)
        if np.any(position < self.workspace[:, 0] - self.margin) or np.any(
            position > self.workspace[:, 1] + self.margin
        ):
            return "final_eef_workspace"
        info = model.check_environment_collision_details(target)
        if info.in_collision:
            contacts = ", ".join(f"{p.link_name1}/{p.link_name2}" for p in info.collision_pairs)
            return f"environment_collision: {contacts or 'contact_details_unavailable'}"
        return None

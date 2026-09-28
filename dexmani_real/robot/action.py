"""Source-neutral online intent to physical target realization.

Motion epochs and SDK authority remain downstream of these operational transforms.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dexmani_real.planning import Pose, XArm7MotionPlanner
from dexmani_real.planning.kinematics.ik import IKResult, make_online_ik_config
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
    workspace_clip_m: float = 0.0
    arm_clip_rad: float = 0.0
    hand_clip_rad: float = 0.0
    ik_result: IKResult | None = None
    # Accepted Cartesian target feeds Teleop's explicit EMA only after publication.
    eef_pose: np.ndarray | None = None


class ActionRealizer:
    def __init__(self, runtime, planner=None):
        self.runtime = runtime
        self.planner = planner
        if planner is not None and not runtime.policy.hand_enabled:
            planner.set_hand_qpos(np.deg2rad(runtime.hand.home_qpos_deg))

    @classmethod
    def for_mode(cls, runtime, mode):
        if mode not in ("joint", "eef"):
            raise ValueError("action mode must be joint or eef")
        planner = (
            XArm7MotionPlanner.create_default(online_ik_profile=make_online_ik_config(runtime))
            if mode == "eef"
            else None
        )
        return cls(runtime, planner)

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
        hand_clip = 0.0
        if intent.hand is not None:
            hand = project_hand_command(
                intent.hand, qpos_min_rad=cfg.hand.qpos_min_rad, qpos_max_rad=cfg.hand.qpos_max_rad
            )
            hand_clip = float(np.max(np.abs(hand - intent.hand)))
        if intent.mode == "joint":
            target = project_arm_command(
                arm,
                current,
                joint_lower_rad=cfg.arm.joint_limit_lower,
                joint_upper_rad=cfg.arm.joint_limit_upper,
            )
            change = target - arm
            periodic = (
                np.asarray(cfg.arm.joint_limit_upper) - np.asarray(cfg.arm.joint_limit_lower)
                >= 2 * np.pi
            )
            change[periodic] = (change[periodic] + np.pi) % (2 * np.pi) - np.pi
            return ActionRealization(
                target, hand, arm_clip_rad=float(np.max(np.abs(change))), hand_clip_rad=hand_clip
            )
        if self.planner is None:
            raise ValueError("EEF realization requires an online IK planner")
        workspace = cfg.policy.workspace.as_array()
        position = np.clip(arm[:3], workspace[:, 0], workspace[:, 1])
        pose = Pose(p=position, q=rot6d_to_quat_wxyz(arm[3:]))
        if hand is not None:
            self.planner.set_hand_qpos(hand)
        result = self.planner.solve_online_ik(pose, current, previous)
        return ActionRealization(
            result.qpos if result.success else None,
            hand,
            workspace_clip_m=float(np.max(np.abs(position - arm[:3]))),
            hand_clip_rad=hand_clip,
            ik_result=result,
            eef_pose=np.concatenate((position, arm[3:])),
        )

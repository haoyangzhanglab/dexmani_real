from __future__ import annotations

from typing import Any

import numpy as np

from .arm_planner import XArm7MotionPlanner
from .planner_types import HandPlanningProfile, PathResult, Pose
from .pose_utils import interpolate_qpos_path, resample_qpos_path


class HierarchicalMotionPlanner:
    """Coordinator for arm path planning plus simple hand interpolation."""

    def __init__(self, arm_planner: XArm7MotionPlanner, hand_profile: HandPlanningProfile | None = None) -> None:
        self.arm_planner = arm_planner
        self.hand_profile = hand_profile or HandPlanningProfile()
        self.arm_dof = arm_planner.dof

    def plan_path(
        self,
        target_eef_pose_world: Pose,
        target_hand_qpos: np.ndarray,
        current_qpos: np.ndarray,
        sync_mode: str | None = None,
    ) -> PathResult:
        current = np.asarray(current_qpos, dtype=np.float64).reshape(-1)
        current_arm = current[: self.arm_dof]
        current_hand = current[self.arm_dof :]
        target_hand = np.asarray(target_hand_qpos, dtype=np.float64).reshape(-1)

        arm_result = self.arm_planner.plan_path(target_eef_pose_world, current_arm)
        if not arm_result.success or arm_result.qpos_path is None:
            return PathResult(
                success=False, qpos_path=None, source="", reason=arm_result.reason, report={"arm": arm_result.report}
            )

        hand_path = interpolate_qpos_path(current_hand, target_hand, self.hand_profile.effective_max_step)
        mode = sync_mode or self.hand_profile.sync_mode
        arm_path, hand_path = self.sync_paths(arm_result.qpos_path, hand_path, mode)
        full_path = np.concatenate([arm_path, hand_path], axis=1)

        hand_report = self.validate_hand_path(hand_path)
        if not hand_report["success"]:
            return PathResult(
                success=False,
                qpos_path=None,
                source="",
                reason=hand_report["reason"],
                report={"arm": arm_result.report, "hand": hand_report},
            )

        report = {
            "arm": arm_result.report,
            "hand": hand_report,
            "sync_mode": mode,
            "num_waypoints": int(len(full_path)),
        }
        return PathResult(success=True, qpos_path=full_path, source=arm_result.source, report=report)

    def sync_paths(self, arm_path: np.ndarray, hand_path: np.ndarray, sync_mode: str) -> tuple[np.ndarray, np.ndarray]:
        arm_len = len(arm_path)
        hand_len = len(hand_path)
        target_len = max(arm_len, hand_len)

        if sync_mode == "interp":
            if hand_len == target_len and arm_len == target_len:
                return arm_path.copy(), hand_path.copy()
            return resample_qpos_path(arm_path, target_len), resample_qpos_path(hand_path, target_len)

        arm = resample_qpos_path(arm_path, target_len)
        if hand_len == target_len:
            hand = hand_path.copy()
        elif sync_mode == "post":
            pad_len = target_len - hand_len
            hand = np.concatenate([hand_path, np.repeat(hand_path[-1:], pad_len, axis=0)], axis=0)
        elif sync_mode == "pre":
            pad_len = target_len - hand_len
            hand = np.concatenate([np.repeat(hand_path[:1], pad_len, axis=0), hand_path], axis=0)
        else:
            raise ValueError("sync_mode must be 'pre', 'post', or 'interp'.")
        return arm, hand

    def validate_hand_path(self, hand_path: np.ndarray) -> dict[str, Any]:
        if len(hand_path) <= 1:
            return {"success": True, "max_hand_delta": 0.0}
        delta = np.diff(hand_path, axis=0)
        max_delta = float(np.max(np.abs(delta)))
        max_allowed = self.hand_profile.effective_max_step
        if max_delta > max_allowed + 1e-12:
            return {
                "success": False,
                "reason": "Hand path waypoint delta too large.",
                "max_hand_delta": max_delta,
                "max_allowed": max_allowed,
            }
        return {"success": True, "max_hand_delta": max_delta, "max_allowed": max_allowed}

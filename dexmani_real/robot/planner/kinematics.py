from __future__ import annotations

from typing import Any

import numpy as np

from .planner_types import Pose, XArm7PlannerConfig
from .pose_utils import compose_pose, compute_pose_error, ensure_qpos, invert_pose


class XArm7Kinematics:
    """FK, Jacobian, pose transforms, and manipulability for xArm7."""

    def __init__(
        self,
        mp_planner: Any,
        pinocchio_model: Any,
        eef_link_id: int,
        dof: int,
        joint_limits: np.ndarray,
        equivalent_joint_mask: np.ndarray,
        base_pose_world: Pose,
        config: XArm7PlannerConfig,
        mp: Any,
    ) -> None:
        self.mp_planner = mp_planner
        self.pinocchio_model = pinocchio_model
        self.eef_link_id = eef_link_id
        self.dof = dof
        self.joint_limits = joint_limits
        self.equivalent_joint_mask = equivalent_joint_mask
        self.base_pose_world = base_pose_world.copy()
        self.base_pose_inverse = invert_pose(self.base_pose_world)
        self.config = config
        self.mp = mp

    def set_base_pose(self, base_pose_world: Pose) -> None:
        self.base_pose_world = base_pose_world.copy()
        self.base_pose_inverse = invert_pose(self.base_pose_world)
        self.mp_planner.set_base_pose(self.to_mplib_pose(self.base_pose_world))

    def world_to_base_pose(self, pose_world: Pose) -> Pose:
        return compose_pose(self.base_pose_inverse, pose_world)

    def base_to_world_pose(self, pose_base: Pose) -> Pose:
        return compose_pose(self.base_pose_world, pose_base)

    def compute_eef_pose_base(self, qpos: np.ndarray) -> Pose:
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        full_qpos = self.mp_planner.pad_move_group_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)
        link_pose = self.pinocchio_model.get_link_pose(self.eef_link_id)
        return Pose(p=np.asarray(link_pose.p, dtype=np.float64), q=np.asarray(link_pose.q, dtype=np.float64))

    def compute_eef_pose_world(self, qpos: np.ndarray) -> Pose:
        return self.base_to_world_pose(self.compute_eef_pose_base(qpos))

    def compute_eef_jacobian(self, qpos: np.ndarray) -> np.ndarray:
        qpos = ensure_qpos(qpos, self.dof, "qpos")
        full_qpos = self.mp_planner.pad_move_group_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)
        jacobian = np.asarray(
            self.pinocchio_model.compute_single_link_jacobian(full_qpos, self.eef_link_id, False), dtype=np.float64
        )
        if jacobian.shape[1] < self.dof:
            raise RuntimeError(f"Jacobian has {jacobian.shape[1]} columns but dof is {self.dof}.")
        return jacobian[:, : self.dof]

    def compute_manipulability(self, qpos: np.ndarray) -> float:
        """Yoshikawa manipulability measure: sqrt(det(J * J^T))."""
        J = self.compute_eef_jacobian(qpos)
        JJT = J @ J.T
        det = float(np.linalg.det(JJT))
        return np.sqrt(max(det, 0.0))

    def compute_world_pose_error(self, target_eef_pose_world: Pose, qpos: np.ndarray) -> tuple[float, float]:
        return compute_pose_error(target_eef_pose_world, self.compute_eef_pose_world(qpos))

    def find_link_id(self, link_name: str) -> int:
        names = list(self.pinocchio_model.get_link_names())
        if link_name not in names:
            raise ValueError(f"Link {link_name!r} not found. Available links: {names}")
        return int(names.index(link_name))

    def to_mplib_pose(self, pose: Pose) -> Any:
        return self.mp.Pose(p=pose.p, q=pose.q)

"""xArm7 kinematics via Pinocchio — FK, Jacobian, pose transforms."""

from __future__ import annotations

from typing import Any

import numpy as np

from .pose_utils import compose_pose, compute_pose_error, invert_pose, quat_wxyz_to_rotmat
from .types import Pose


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
        mplib: Any,
    ) -> None:
        self.mp_planner = mp_planner
        self.pinocchio_model = pinocchio_model
        self.eef_link_id = eef_link_id
        self.dof = dof
        self.joint_limits = joint_limits
        self.equivalent_joint_mask = equivalent_joint_mask
        self.base_pose_world = base_pose_world.copy()
        self.base_pose_inverse = invert_pose(self.base_pose_world)
        self.mplib = mplib

    def set_base_pose(self, base_pose_world: Pose) -> None:
        self.base_pose_world = base_pose_world.copy()
        self.base_pose_inverse = invert_pose(self.base_pose_world)
        self.mp_planner.set_base_pose(self.to_mplib_pose(self.base_pose_world))

    def world_to_base_pose(self, pose_world: Pose) -> Pose:
        return compose_pose(self.base_pose_inverse, pose_world)

    def base_to_world_pose(self, pose_base: Pose) -> Pose:
        return compose_pose(self.base_pose_world, pose_base)

    def compute_eef_pose_base(self, qpos: np.ndarray) -> Pose:
        # Hot-path: ensure_qpos validation is done at entry points (solve / solve_teleop_ik).
        # Skipped here to avoid redundant per-frame checks (ref: P1.1).
        full_qpos = self.mp_planner.pad_move_group_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)
        link_pose = self.pinocchio_model.get_link_pose(self.eef_link_id)
        return Pose(p=np.asarray(link_pose.p, dtype=np.float64), q=np.asarray(link_pose.q, dtype=np.float64))

    def compute_eef_pose_world(self, qpos: np.ndarray) -> Pose:
        pose_base = self.compute_eef_pose_base(qpos)
        pose_world = self.base_to_world_pose(pose_base)
        # Defense-in-depth: NaN guard symmetric with fingertip FK (interface.py:615-616).
        # Pinocchio FK is robust, but guard against model corruption or numerical anomalies.
        if not np.all(np.isfinite(pose_world.p)) or not np.all(np.isfinite(pose_world.q)):
            return Pose(p=np.full(3, np.nan), q=np.full(4, np.nan))
        return pose_world

    def compute_eef_jacobian(self, qpos: np.ndarray) -> np.ndarray:
        # Hot-path: ensure_qpos validation is done at entry points.
        # Skipped here to avoid redundant per-frame checks (ref: P1.1).
        full_qpos = self.mp_planner.pad_move_group_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)
        jacobian = np.asarray(
            self.pinocchio_model.compute_single_link_jacobian(full_qpos, self.eef_link_id, False), dtype=np.float64
        )
        if jacobian.shape[1] < self.dof:
            raise RuntimeError(f"Jacobian has {jacobian.shape[1]} columns but dof is {self.dof}.")
        return jacobian[:, : self.dof]

    def compute_eef_jacobian_and_pose_world(self, qpos: np.ndarray) -> tuple[np.ndarray, "Pose"]:
        """Compute EEF Jacobian and world-frame pose in a **single** FK call.

        Use this in hot paths where both Jacobian and pose are needed
        (e.g. differential IK) to avoid redundant ``compute_forward_kinematics``.

        Returns:
            (jacobian, pose_world) — both in world frame.
            Jacobian columns map joint velocities to world-frame spatial velocity
            [v_world; ω_world].  Pose is the world-frame EEF pose.
        """
        full_qpos = self.mp_planner.pad_move_group_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)

        # Jacobian (base frame) — Pinocchio local_frame=False gives
        # spatial Jacobian expressed in the model's world frame, which
        # for a fixed-base model is the robot base frame.
        jacobian_full = np.asarray(
            self.pinocchio_model.compute_single_link_jacobian(full_qpos, self.eef_link_id, False),
            dtype=np.float64,
        )
        if jacobian_full.shape[1] < self.dof:
            raise RuntimeError(f"Jacobian has {jacobian_full.shape[1]} columns but dof is {self.dof}.")
        jacobian_base = jacobian_full[:, : self.dof]

        # Pose (base frame) — extracted from already-computed FK, no extra FK call.
        link_pose = self.pinocchio_model.get_link_pose(self.eef_link_id)
        pose_base = Pose(
            p=np.asarray(link_pose.p, dtype=np.float64),
            q=np.asarray(link_pose.q, dtype=np.float64),
        )

        # Transform Jacobian from base frame → world frame.
        # Spatial velocity transforms as [v_w; ω_w] = Ad_{R} @ [v_b; ω_b],
        # where Ad_R = block_diag(R_b2w, R_b2w).  When base_pose_world is identity
        # (simulation), this is a no-op.
        R_b2w = quat_wxyz_to_rotmat(self.base_pose_world.q)
        jacobian_world = np.empty_like(jacobian_base)
        jacobian_world[:3, :] = R_b2w @ jacobian_base[:3, :]  # linear part
        jacobian_world[3:, :] = R_b2w @ jacobian_base[3:, :]  # angular part

        pose_world = self.base_to_world_pose(pose_base)

        return jacobian_world, pose_world

    def compute_manipulability(self, qpos: np.ndarray) -> float:
        """Yoshikawa manipulability measure: sqrt(det(J * J^T)).

        If you already have the Jacobian, use :meth:`manipulability_from_jacobian`
        to avoid a redundant FK+Jacobian computation (~0.3 ms).
        """
        J = self.compute_eef_jacobian(qpos)
        return self.manipulability_from_jacobian(J)

    @staticmethod
    def manipulability_from_jacobian(J: np.ndarray) -> float:
        """Yoshikawa manipulability from a pre-computed Jacobian.

        Use this in hot paths where the Jacobian is already available
        (e.g. after ``compute_eef_jacobian``) to avoid redundant FK.

        Args:
            J: 6×dof end-effector Jacobian matrix.

        Returns:
            sqrt(det(J @ Jᵀ)), clamped to ≥ 0.
        """
        JJT = J @ J.T
        det = float(np.linalg.det(JJT))
        return np.sqrt(max(det, 0.0))

    def compute_world_pose_error(self, target_eef_pose_world: Pose, qpos: np.ndarray) -> tuple[float, float]:
        return compute_pose_error(target_eef_pose_world, self.compute_eef_pose_world(qpos))

    def to_mplib_pose(self, pose: Pose) -> Any:
        return self.mplib.Pose(p=pose.p, q=pose.q)

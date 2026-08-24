"""xArm7 kinematics via Pinocchio — FK, Jacobian, pose transforms.

Two FK classes:
  - ``ArmFK`` — standalone Pinocchio FK (no MPlib dependency).
  - ``XArm7Kinematics`` — full kinematics with MPlib integration for planning.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from dexmani_real.robot_spec import ARM_JOINT_SHAPE, XARM7_XHAND_COLLISION_URDF_PATH

from .poses import compose_pose, compute_pose_error, invert_pose, quat_wxyz_to_rotmat
from .types import Pose

_ARM_FK_URDF = str(XARM7_XHAND_COLLISION_URDF_PATH)


@lru_cache(maxsize=1)
def make_arm_fk() -> "ArmFK":
    """Return the shared, URDF-consistent arm FK (cached per process).

    The arm worker no longer computes EEF; any consumer that needs the EEF
    pose derives it from ``qpos`` through this single factory.
    """
    return ArmFK(_ARM_FK_URDF)


class ArmFK:
    """Standalone Pinocchio FK for arm_loop — URDF-consistent, no MPlib dependency.

    Used instead of xArm SDK get_position_aa() because the firmware EEF frame
    differs from the URDF definition.
    """

    def __init__(self, urdf_path: str, eef_frame_name: str = "custom_eef_link") -> None:
        import pinocchio

        self._model = pinocchio.buildModelFromUrdf(str(urdf_path))
        self._data = self._model.createData()
        self._eef_frame_id = self._model.getFrameId(eef_frame_name)
        if self._eef_frame_id >= self._model.nframes:
            raise ValueError(f"URDF is missing EEF frame {eef_frame_name!r}")

    def compute(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute EEF pose in arm base frame. Returns (eef_pos(3), eef_rot6d(6))."""
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(qpos)):
            raise ValueError(f"arm qpos must be finite with shape {ARM_JOINT_SHAPE}")
        import pinocchio

        pinocchio.forwardKinematics(self._model, self._data, qpos)
        pinocchio.updateFramePlacements(self._model, self._data)
        pose = self._data.oMf[self._eef_frame_id]

        R = np.asarray(pose.rotation, dtype=np.float64)  # 3×3
        eef_pos = np.asarray(pose.translation, dtype=np.float64).copy()
        if (
            eef_pos.shape != (3,)
            or R.shape != (3, 3)
            or not np.all(np.isfinite(eef_pos))
            or not np.all(np.isfinite(R))
        ):
            raise RuntimeError("ArmFK produced a malformed or non-finite pose")
        orthogonality_error = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
        determinant = float(np.linalg.det(R))
        if orthogonality_error > 1e-6 or abs(determinant - 1.0) > 1e-6:
            raise RuntimeError(
                "ArmFK produced an invalid rotation "
                f"(orthogonality_error={orthogonality_error:.3g}, det={determinant:.9g})"
            )
        eef_rot6d = np.concatenate([R[:, 0], R[:, 1]]).astype(np.float64)
        return eef_pos, eef_rot6d


class XArm7Kinematics:
    """Full kinematics with MPlib integration — FK, Jacobian, pose transforms, manipulability."""

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
        # Entry points validate qpos and the base pose; this path stays allocation-light.
        if not np.all(np.isfinite(qpos)):
            raise ValueError(f"compute_eef_pose_base: qpos contains NaN or Inf")
        full_qpos = self.mp_planner.pad_move_group_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)
        link_pose = self.pinocchio_model.get_link_pose(self.eef_link_id)
        return Pose(p=np.asarray(link_pose.p, dtype=np.float64), q=np.asarray(link_pose.q, dtype=np.float64))

    def compute_eef_pose_world(self, qpos: np.ndarray) -> Pose:
        pose_base = self.compute_eef_pose_base(qpos)
        pose_world = self.base_to_world_pose(pose_base)
        # Keep a NaN guard for model corruption or numerical anomalies.
        if not np.all(np.isfinite(pose_world.p)) or not np.all(np.isfinite(pose_world.q)):
            return Pose(p=np.full(3, np.nan), q=np.full(4, np.nan))
        return pose_world

    def compute_eef_jacobian(self, qpos: np.ndarray) -> np.ndarray:
        # Entry points validate qpos and the base pose before this hot path.
        if not np.all(np.isfinite(qpos)):
            raise ValueError(f"compute_eef_jacobian: qpos contains NaN or Inf")
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
        # Reject NaN/Inf before Pinocchio FK to protect the C++ engine.
        if not np.all(np.isfinite(qpos)):
            raise ValueError("compute_eef_jacobian_and_pose_world: qpos contains NaN or Inf")
        full_qpos = self.mp_planner.pad_move_group_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)

        # Pinocchio returns this Jacobian in the fixed-base frame.
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

        # Transform spatial velocity from base frame to world frame.
        R_b2w = quat_wxyz_to_rotmat(self.base_pose_world.q)
        jacobian_world = np.empty_like(jacobian_base)
        jacobian_world[:3, :] = R_b2w @ jacobian_base[:3, :]  # linear part
        jacobian_world[3:, :] = R_b2w @ jacobian_base[3:, :]  # angular part

        pose_world = self.base_to_world_pose(pose_base)

        return jacobian_world, pose_world

    def compute_manipulability(self, qpos: np.ndarray) -> float:
        """Yoshikawa manipulability measure: sqrt(det(J * J^T)).

        If you already have the Jacobian, use :meth:`manipulability_from_jacobian`
        to avoid a redundant FK+Jacobian computation.
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
            Returns 0.0 on non-finite Jacobian (NaN/Inf) — zero
            manipulability triggers downstream rejection gates.
        """
        if not np.all(np.isfinite(J)):
            return 0.0
        JJT = J @ J.T
        det = float(np.linalg.det(JJT))
        return np.sqrt(max(det, 0.0))

    def compute_world_pose_error(self, target_eef_pose_world: Pose, qpos: np.ndarray) -> tuple[float, float]:
        return compute_pose_error(target_eef_pose_world, self.compute_eef_pose_world(qpos))

    def to_mplib_pose(self, pose: Pose) -> Any:
        return self.mplib.Pose(p=pose.p, q=pose.q)

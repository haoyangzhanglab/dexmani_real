"""URDF-consistent xArm7 FK and Jacobians in the arm-base frame."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_EQUIVALENT_JOINT_MASK,
    XARM7_XHAND_SRDF_PATH,
)
from ..paths import wrap_nearest_equivalent
from .pose import Pose

def make_arm_kinematics():
    """Construct the seven-joint MPlib backend in xArm-base coordinates, without devices."""
    import mplib

    if not XARM7_XHAND_SRDF_PATH.is_file():
        raise FileNotFoundError(XARM7_XHAND_SRDF_PATH)
    velocity = np.deg2rad(np.full(7, 180.0))
    model = mplib.Planner(
        urdf=str(XARM7_XHAND_COLLISION_URDF_PATH),
        srdf=str(XARM7_XHAND_SRDF_PATH),
        move_group="custom_eef_link",
        use_convex=False,
        joint_vel_limits=velocity.tolist(),
        joint_acc_limits=(velocity * 2).tolist(),
    )
    return XArm7Kinematics(model)


def make_arm_fk() -> "ArmFK":
    """Create an owner-local FK workspace; Pinocchio Data must not cross worker ownership."""
    return ArmFK(str(XARM7_XHAND_COLLISION_URDF_PATH))


class ArmFK:
    """Pinocchio FK in the URDF EEF frame, which differs from the xArm firmware EEF frame."""

    def __init__(self, urdf_path: str, eef_frame_name: str = "custom_eef_link") -> None:
        import pinocchio

        self.model = pinocchio.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.eef_frame_id = self.model.getFrameId(eef_frame_name)
        if self.eef_frame_id >= self.model.nframes:
            raise ValueError(f"URDF is missing EEF frame {eef_frame_name!r}")

    def compute(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute EEF pose in arm base frame. Returns (eef_pos(3), eef_rot6d(6))."""
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != ARM_JOINT_SHAPE or not np.all(np.isfinite(qpos)):
            raise ValueError(f"arm qpos must be finite with shape {ARM_JOINT_SHAPE}")
        import pinocchio

        pinocchio.forwardKinematics(self.model, self.data, qpos)
        pinocchio.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.eef_frame_id]

        R = np.asarray(pose.rotation, dtype=np.float64)
        eef_pos = np.asarray(pose.translation, dtype=np.float64).copy()
        if (
            eef_pos.shape != (3,)
            or R.shape != (3, 3)
            or not np.all(np.isfinite(eef_pos))
            or not np.all(np.isfinite(R))
        ):
            raise RuntimeError("ArmFK produced a malformed or non-finite pose")
        eef_rot6d = np.concatenate([R[:, 0], R[:, 1]])
        return eef_pos, eef_rot6d


def compute_eef_pose_history_xarm_base(
    arm_qpos: np.ndarray,
    *,
    arm_fk: "ArmFK | None" = None,
) -> np.ndarray:
    """Return float64 [T,9] base-frame XYZ+rot6d from measured or commanded qpos."""
    qpos_history = np.asarray(arm_qpos, dtype=np.float64)
    if qpos_history.ndim != 2 or qpos_history.shape[1] != ARM_JOINT_SHAPE[0]:
        raise ValueError(
            f"arm qpos history must have shape (T, {ARM_JOINT_SHAPE[0]}), got {qpos_history.shape}"
        )
    fk = make_arm_fk() if arm_fk is None else arm_fk
    poses = np.empty((len(qpos_history), 9), dtype=np.float64)
    for index, qpos in enumerate(qpos_history):
        eef_pos, eef_rot6d = fk.compute(qpos)
        poses[index, :3] = eef_pos
        poses[index, 3:] = eef_rot6d
    return poses


class XArm7Kinematics:
    """Nominal-URDF FK and Jacobians in the same xArm-base frame as collision geometry."""

    def __init__(self, mp_planner: Any) -> None:
        self.mp_planner = mp_planner
        self.pinocchio_model = mp_planner.pinocchio_model
        self.eef_link_id = list(self.pinocchio_model.get_link_names()).index(mp_planner.move_group)
        self.joint_limits = np.asarray(mp_planner.joint_limits, dtype=np.float64)
        self.dof = len(self.joint_limits)
        self.equivalent_joint_mask = np.asarray(XARM7_EQUIVALENT_JOINT_MASK, dtype=bool)
        mp_planner.set_base_pose(self.to_mplib_pose(Pose.identity()))

    def compute_eef_pose_base(self, qpos: np.ndarray) -> Pose:
        """Return an owned nominal URDF EEF pose in xArm-base coordinates."""
        if not np.all(np.isfinite(qpos)):
            raise ValueError("compute_eef_pose_base: qpos contains NaN or Inf")
        full_qpos = self.mp_planner.pad_move_group_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)
        link_pose = self.pinocchio_model.get_link_pose(self.eef_link_id)
        return Pose(
            p=np.asarray(link_pose.p, dtype=np.float64).copy(),
            q=np.asarray(link_pose.q, dtype=np.float64),
        )

    def compute_eef_jacobian_base(self, qpos: np.ndarray) -> np.ndarray:
        """Return [EEF linear velocity; angular velocity] in xArm-base axes."""
        if not np.all(np.isfinite(qpos)):
            raise ValueError("compute_eef_jacobian_base: qpos contains NaN or Inf")
        full_qpos = self.mp_planner.pad_move_group_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)
        jacobian = np.asarray(
            self.pinocchio_model.compute_single_link_jacobian(full_qpos, self.eef_link_id, False),
            dtype=np.float64,
        )
        if jacobian.ndim != 2 or jacobian.shape[0] != 6 or jacobian.shape[1] < self.dof:
            raise RuntimeError(f"Expected a (6, >= {self.dof}) Jacobian, got {jacobian.shape}")
        jacobian = jacobian[:, : self.dof]
        position = np.asarray(self.pinocchio_model.get_link_pose(self.eef_link_id).p)
        # Pinocchio WORLD uses the base origin; shift its linear velocity to the EEF.
        linear = jacobian[:3] + np.cross(jacobian[3:].T, position).T
        return np.vstack((linear, jacobian[3:]))

    def to_mplib_pose(self, pose: Pose) -> Any:
        import mplib

        return mplib.Pose(p=pose.p, q=pose.q)

    def canonicalize_qpos(
        self,
        qpos: np.ndarray,
        reference_qpos: np.ndarray,
        limits: np.ndarray | None = None,
    ) -> np.ndarray:
        if limits is None:
            limits = self.joint_limits
        return wrap_nearest_equivalent(
            qpos,
            reference_qpos,
            limits[:, 0],
            limits[:, 1],
            equivalent_mask=self.equivalent_joint_mask,
        )

    def compute_qpos_delta(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        """Signed deltas of admitted joint vectors; equivalent axes wrap to [-π, π)."""
        delta = qpos - reference_qpos
        mask = self.equivalent_joint_mask
        delta[mask] = (delta[mask] + np.pi) % (2.0 * np.pi) - np.pi
        return delta

"""URDF-consistent xArm7 FK, Jacobians and pose transforms."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    XARM7_XHAND_COLLISION_URDF_PATH,
    XARM7_EQUIVALENT_JOINT_MASK,
    XARM7_XHAND_SRDF_PATH,
)
from ..paths import wrap_nearest_equivalent
from .pose import Pose, compose_pose, compute_pose_error, invert_pose, quat_wxyz_to_rotmat

_ARM_FK_URDF = str(XARM7_XHAND_COLLISION_URDF_PATH)


def make_arm_kinematics(base_pose_world=None):
    """Construct the existing seven-joint MPlib numerical backend, without devices."""
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
    return XArm7Kinematics(model, Pose.identity() if base_pose_world is None else base_pose_world)


@lru_cache(maxsize=1)
def make_arm_fk() -> "ArmFK":
    """Return cached per-process FK for deriving the URDF EEF pose from arm qpos."""
    return ArmFK(_ARM_FK_URDF)


class ArmFK:
    """Pinocchio FK in the URDF EEF frame, which differs from the xArm firmware EEF frame."""

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

        R = np.asarray(pose.rotation, dtype=np.float64)
        eef_pos = np.asarray(pose.translation, dtype=np.float64).copy()
        if (
            eef_pos.shape != (3,)
            or R.shape != (3, 3)
            or not np.all(np.isfinite(eef_pos))
            or not np.all(np.isfinite(R))
        ):
            raise RuntimeError("ArmFK produced a malformed or non-finite pose")
        eef_rot6d = np.concatenate([R[:, 0], R[:, 1]]).astype(np.float64)
        return eef_pos, eef_rot6d


def compute_eef_pose_history_xarm_base(
    arm_qpos: np.ndarray,
    *,
    arm_fk: "ArmFK | None" = None,
) -> np.ndarray:
    """Return finite float64 [T,9] base-frame XYZ+rot6d from measured/commanded qpos using one URDF FK per row."""
    qpos_history = np.asarray(arm_qpos, dtype=np.float64)
    if qpos_history.ndim != 2 or qpos_history.shape[1] != ARM_JOINT_SHAPE[0]:
        raise ValueError(
            f"arm qpos history must have shape (T, {ARM_JOINT_SHAPE[0]}), got {qpos_history.shape}"
        )
    if not np.all(np.isfinite(qpos_history)):
        raise ValueError("arm qpos history contains non-finite values")
    fk = make_arm_fk() if arm_fk is None else arm_fk
    poses = np.empty((len(qpos_history), 9), dtype=np.float64)
    for index, qpos in enumerate(qpos_history):
        eef_pos, eef_rot6d = fk.compute(qpos)
        poses[index, :3] = eef_pos
        poses[index, 3:] = eef_rot6d
    if not np.all(np.isfinite(poses)):
        raise RuntimeError("EEF pose history contains non-finite values")
    return poses


class XArm7Kinematics:
    """Full kinematics with MPlib integration — FK, Jacobian, pose transforms."""

    def __init__(self, mp_planner: Any, base_pose_world: Pose) -> None:
        self.mp_planner = mp_planner
        self.pinocchio_model = mp_planner.pinocchio_model
        self.eef_link_id = list(self.pinocchio_model.get_link_names()).index(mp_planner.move_group)
        self.joint_limits = np.asarray(mp_planner.joint_limits, dtype=np.float64)
        self.dof = len(self.joint_limits)
        self.equivalent_joint_mask = np.asarray(XARM7_EQUIVALENT_JOINT_MASK, dtype=bool)
        self.base_pose_world = base_pose_world.copy()
        self.base_pose_inverse = invert_pose(self.base_pose_world)
        mp_planner.set_base_pose(self.to_mplib_pose(self.base_pose_world))

    def world_to_base_pose(self, pose_world: Pose) -> Pose:
        return compose_pose(self.base_pose_inverse, pose_world)

    def base_to_world_pose(self, pose_base: Pose) -> Pose:
        return compose_pose(self.base_pose_world, pose_base)

    def compute_eef_pose_base(self, qpos: np.ndarray) -> Pose:
        if not np.all(np.isfinite(qpos)):
            raise ValueError("compute_eef_pose_base: qpos contains NaN or Inf")
        full_qpos = self.mp_planner.pad_move_group_qpos(qpos)
        self.pinocchio_model.compute_forward_kinematics(full_qpos)
        link_pose = self.pinocchio_model.get_link_pose(self.eef_link_id)
        return Pose(
            p=np.asarray(link_pose.p, dtype=np.float64),
            q=np.asarray(link_pose.q, dtype=np.float64),
        )

    def compute_eef_pose_world(self, qpos: np.ndarray) -> Pose:
        pose_base = self.compute_eef_pose_base(qpos)
        return self.base_to_world_pose(pose_base)

    def compute_eef_jacobian_world(self, qpos: np.ndarray) -> np.ndarray:
        """Return [EEF linear velocity; angular velocity] in world axes."""
        if not np.all(np.isfinite(qpos)):
            raise ValueError("compute_eef_jacobian_world: qpos contains NaN or Inf")
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
        # Shift Pinocchio WORLD linear velocity from the base origin to the EEF before rotating axes.
        linear = jacobian[:3] + np.cross(jacobian[3:].T, position).T
        rotation = quat_wxyz_to_rotmat(self.base_pose_world.q)
        return np.vstack((rotation @ linear, rotation @ jacobian[3:]))

    def compute_world_pose_error(
        self, target_eef_pose_world: Pose, qpos: np.ndarray
    ) -> tuple[float, float]:
        return compute_pose_error(target_eef_pose_world, self.compute_eef_pose_world(qpos))

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

    def planning_limits(self, reference_qpos: np.ndarray) -> np.ndarray:
        """Bound HOME's equivalent representations around the measured joint band."""
        limits = self.joint_limits.copy()
        mask = self.equivalent_joint_mask
        limits[mask, 0] = np.maximum(limits[mask, 0], reference_qpos[mask] - 3.0 * np.pi)
        limits[mask, 1] = np.minimum(limits[mask, 1], reference_qpos[mask] + 3.0 * np.pi)
        return limits

    def canonicalize_path(self, path: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        """Choose continuous equivalent angles within model limits, then HOME limits."""
        path = path.copy()
        for limits in (self.joint_limits, self.planning_limits(reference_qpos)):
            previous = reference_qpos
            for index, qpos in enumerate(path):
                path[index] = self.canonicalize_qpos(qpos, previous, limits)
                previous = path[index]
        return path

    def compute_qpos_delta(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        """Signed deltas of admitted joint vectors; equivalent axes wrap to [-π, π)."""
        delta = qpos - reference_qpos
        mask = self.equivalent_joint_mask
        delta[mask] = (delta[mask] + np.pi) % (2.0 * np.pi) - np.pi
        return delta

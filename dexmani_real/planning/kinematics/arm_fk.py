"""Pinocchio xArm7 FK and pose histories in the arm-base frame."""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.model import (
    ARM_JOINT_SHAPE,
    XARM7_XHAND_COLLISION_URDF_PATH,
)


def make_arm_fk() -> ArmFK:
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
    arm_fk: ArmFK | None = None,
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

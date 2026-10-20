"""Squared fingertip errors and analytic joint gradients in the fixed hand-base frame."""

from __future__ import annotations

import numpy as np
import pinocchio as pin


class FingertipObjective:
    def __init__(self, urdf_path: str, fingertip_frame_names: tuple[str, ...]) -> None:
        names = tuple(fingertip_frame_names)
        if len(names) != 5:
            raise ValueError("fingertip_frame_names must contain exactly five entries")
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("fingertip_frame_names entries must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("fingertip_frame_names entries must be unique")

        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.dof = self.model.nv

        missing = [name for name in names if not self.model.existFrame(name)]
        if missing:
            raise ValueError(f"fingertip frames not found in URDF {urdf_path!r}: {missing}")
        self.fingertip_frame_ids = [int(self.model.getFrameId(name)) for name in names]

    def update_kinematics(self, qpos: np.ndarray) -> None:
        pin.computeJointJacobians(self.model, self.data, qpos)
        pin.updateFramePlacements(self.model, self.data)

    def position_value_gradient(
        self, qpos: np.ndarray, target_positions: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return Σ||p_tip(q) − target||² and its joint gradient."""
        self.update_kinematics(qpos)

        gradient = np.zeros(self.dof, dtype=np.float64)
        value = 0.0
        for index, frame_id in enumerate(self.fingertip_frame_ids):
            jacobian = pin.getFrameJacobian(
                self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:3]
            error = self.data.oMf[frame_id].translation - target_positions[index]
            value += float(np.sum(error * error))
            gradient += 2.0 * (error @ jacobian)

        return value, gradient

    def pinch_value_gradient(
        self, qpos: np.ndarray, factors: np.ndarray, base_weight: float
    ) -> tuple[float, np.ndarray]:
        """Return Σ weight·factor²·||p_finger(q) − p_thumb(q)||² and its joint gradient."""
        self.update_kinematics(qpos)
        thumb_frame_id = self.fingertip_frame_ids[0]
        thumb_jacobian = pin.getFrameJacobian(
            self.model, self.data, thumb_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )[:3]
        thumb_position = self.data.oMf[thumb_frame_id].translation

        value = 0.0
        gradient = np.zeros(self.dof, dtype=np.float64)
        for index, frame_id in enumerate(self.fingertip_frame_ids[1:]):
            factor = factors[index]
            if factor < 1e-4:
                continue
            jacobian = pin.getFrameJacobian(
                self.model, self.data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )[:3]
            error = self.data.oMf[frame_id].translation - thumb_position
            weight = base_weight * (factor * factor)
            value += float(weight * np.sum(error * error))
            gradient += weight * 2.0 * (error @ (jacobian - thumb_jacobian))
        return value, gradient


def joint_regularization(
    qpos: np.ndarray, reference: np.ndarray, weight: float
) -> tuple[float, np.ndarray]:
    """Return weight·||q − reference||² and its joint gradient."""
    error = qpos - reference
    return float(weight * np.sum(error * error)), 2.0 * weight * error

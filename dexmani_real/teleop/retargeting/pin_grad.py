"""TAG FreeFlyer gradients: q=[XYZ(3),xyzw(4),joints]; Jacobian[:,6:] omits base velocities."""

from __future__ import annotations

import numpy as np
import pinocchio as pin

_FINGERTIP_ORDER = ("thumb", "index", "mid", "ring", "pinky")


def validate_fingertip_frame_names(
    fingertip_frame_names: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    """Validate the fixed five-finger XHand optimizer contract."""
    names = tuple(fingertip_frame_names)
    if len(names) != len(_FINGERTIP_ORDER):
        raise ValueError("fingertip_frame_names must contain exactly five entries")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("fingertip_frame_names entries must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("fingertip_frame_names entries must be unique")
    for index, (name, expected_finger) in enumerate(zip(names, _FINGERTIP_ORDER)):
        if expected_finger not in name.lower():
            raise ValueError(
                "fingertip_frame_names must be ordered thumb/index/mid/ring/pinky; "
                f"entry {index}={name!r} does not identify {expected_finger}"
            )
    return names


class PinGrad:
    """Hand FK gradients using Pinocchio with a FreeFlyer base."""

    def __init__(self, urdf_path: str, fingertip_frame_names: list[str]) -> None:
        names = validate_fingertip_frame_names(fingertip_frame_names)
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.dof: int = self.model.nv - 6

        # All five semantic fingers are required; a shortened list would shift target/frame associations.
        missing = [name for name in names if not self.model.existFrame(name)]
        if missing:
            raise ValueError(f"fingertip frames not found in URDF {urdf_path!r}: {missing}")
        self.tip_frame_ids = [int(self.model.getFrameId(name)) for name in names]

    def update_kinematics(self, qpos_floating: np.ndarray) -> None:
        """Update FK and Jacobians from (7 + dof,) coordinates with a FreeFlyer base."""
        pin.computeJointJacobians(self.model, self.data, qpos_floating)
        pin.updateFramePlacements(self.model, self.data)

    def compute_position_gradient(
        self, qpos_floating: np.ndarray, target_pos: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Joint-only ∇loss for Σ||FK_tip(q)-target||² in the URDF frame.
        The FreeFlyer base stays fixed; q=[XYZ(3),xyzw(4),joints(12)]."""
        self.update_kinematics(qpos_floating)

        grad = np.zeros(self.dof, dtype=np.float64)
        loss = 0.0
        for i in range(len(self.tip_frame_ids)):
            fid = self.tip_frame_ids[i]
            J_full = pin.getFrameJacobian(
                self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_v = J_full[:3, 6:]  # Skip the six FreeFlyer velocity columns.
            p_tip = self.data.oMf[fid].translation
            diff = p_tip - target_pos[i]
            loss += float(np.sum(diff * diff))
            grad += 2.0 * (diff @ J_v)

        return grad, loss

    @staticmethod
    def compute_smoothness_gradient(
        q: np.ndarray, q_last: np.ndarray, weight: float
    ) -> tuple[np.ndarray, float]:
        """Return ((dof,) gradient, scalar loss) for weight * ||q - q_last||²."""
        diff = q - q_last
        return 2.0 * weight * diff, float(weight * np.sum(diff * diff))

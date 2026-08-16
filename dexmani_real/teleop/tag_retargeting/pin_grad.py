"""Pinocchio analytical gradient engine for hand kinematics optimization.

Ported from TAG/Retargeting/Hand_Retargeting/utils/pin_grad.py.

FreeFlyer convention:
  q = [tx, ty, tz, qx, qy, qz, qw | joint_0 ... joint_N-1]
      0   1   2   3   4   5   6  | 7 ...
  Jacobian[:, 6:] strips the 6 floating-base velocity DOFs.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin

from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

_FINGERTIP_ORDER = ("thumb", "index", "mid", "ring", "pinky")


def validate_fingertip_frame_names(fingertip_frame_names: list[str] | tuple[str, ...]) -> tuple[str, ...]:
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
    """Pinocchio-based analytical gradient engine for hand FK optimization.

    Wraps Pinocchio FK + Jacobian to compute position gradients
    for the hand retargeting optimizer.  Uses JointModelFreeFlyer
    so the generalized coordinate vector includes a floating base.
    """

    def __init__(self, urdf_path: str, fingertip_frame_names: list[str]) -> None:
        """Build Pinocchio model and resolve fingertip frame IDs.

        Args:
            urdf_path: Absolute path to the XHand URDF file.
            fingertip_frame_names: URDF ``<frame>`` names for the 5 fingertips
                (e.g. ``"right_hand_thumb_rota_tip"``).
        """
        names = validate_fingertip_frame_names(fingertip_frame_names)
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.dof: int = self.model.nv - 6  # 12 for XHand
        self._nv: int = self.model.nv

        # The optimizer always receives five targets in semantic finger order;
        # silently shortening this list shifts target-to-frame associations.
        missing = [name for name in names if not self.model.existFrame(name)]
        if missing:
            raise ValueError(f"fingertip frames not found in URDF {urdf_path!r}: {missing}")
        self.tip_frame_ids = [int(self.model.getFrameId(name)) for name in names]

    # ── Kinematics ──────────────────────────────────────────────

    def update_kinematics(self, qpos_floating: np.ndarray) -> None:
        """Run FK, update frame placements, and compute joint Jacobians.

        Args:
            qpos_floating: (7 + dof,) generalized coordinates with FreeFlyer base.
        """
        pin.forwardKinematics(self.model, self.data, qpos_floating)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeJointJacobians(self.model, self.data, qpos_floating)

    # ── Position gradient ───────────────────────────────────────

    def compute_position_gradient(self, qpos_floating: np.ndarray, target_pos: np.ndarray) -> tuple[np.ndarray, float]:
        """Analytic gradient of squared fingertip position error.

        ∂/∂q  Σ_i ||FK_tip_i(q) - target_i||²

        Args:
            qpos_floating: (19,) generalized coords [freeflyer(7) | joints(12)].
            target_pos:     (N, 3) target fingertip positions in URDF frame.

        Returns:
            (grad: (dof,), loss: float) — gradient w.r.t. joint DOFs only.
        """
        self.update_kinematics(qpos_floating)

        grad = np.zeros(self.dof, dtype=np.float64)
        loss = 0.0
        if target_pos.shape != (len(self.tip_frame_ids), 3) or not np.all(np.isfinite(target_pos)):
            raise ValueError(f"target_pos must be a finite ({len(self.tip_frame_ids)}, 3) array")

        for i in range(len(self.tip_frame_ids)):
            fid = self.tip_frame_ids[i]
            J_full = pin.getFrameJacobian(self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            J_v = J_full[:3, 6:]  # strip FreeFlyer columns, keep joint DOFs
            p_tip = self.data.oMf[fid].translation
            diff = p_tip - target_pos[i]
            loss += float(np.sum(diff * diff))
            grad += 2.0 * (diff @ J_v)

        return grad, loss

    # ── Temporal smoothness gradient (static — no Pinocchio needed) ─

    @staticmethod
    def compute_smoothness_gradient(q: np.ndarray, q_last: np.ndarray, weight: float) -> tuple[np.ndarray, float]:
        """Gradient of temporal smoothness penalty:  weight * ||q - q_last||².

        Returns:
            (grad: (dof,), loss: float)
        """
        diff = q - q_last
        return 2.0 * weight * diff, float(weight * np.sum(diff * diff))

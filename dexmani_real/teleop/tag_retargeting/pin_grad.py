"""Pinocchio analytical gradient engine for hand kinematics optimization.

Ported from TAG/Retargeting/Hand_Retargeting/utils/pin_grad.py.
Simplified: rotation gradient methods removed (unused by XHand new method).

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
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.dof: int = self.model.nv - 6  # 12 for XHand
        self._nv: int = self.model.nv

        # Resolve fingertip frame names → frame IDs
        self.tip_frame_ids: list[int] = []
        for name in fingertip_frame_names:
            if not self.model.existFrame(name):
                logger.warning("Fingertip frame %r not found in URDF — skipped", name)
            else:
                self.tip_frame_ids.append(int(self.model.getFrameId(name)))

        if len(self.tip_frame_ids) < 1:
            logger.warning("No fingertip frames resolved from %r", fingertip_frame_names)

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

    def compute_position_gradient(
        self, qpos_floating: np.ndarray, target_pos: np.ndarray
    ) -> tuple[np.ndarray, float]:
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
        n_fingers = min(len(self.tip_frame_ids), len(target_pos))

        for i in range(n_fingers):
            fid = self.tip_frame_ids[i]
            J_full = pin.getFrameJacobian(
                self.model, self.data, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_v = J_full[:3, 6:]  # strip FreeFlyer columns, keep joint DOFs
            p_tip = self.data.oMf[fid].translation
            diff = p_tip - target_pos[i]
            loss += float(np.sum(diff * diff))
            grad += 2.0 * (diff @ J_v)

        return grad, loss

    # ── Temporal smoothness gradient (static — no Pinocchio needed) ─

    @staticmethod
    def compute_smoothness_gradient(
        q: np.ndarray, q_last: np.ndarray, weight: float
    ) -> tuple[np.ndarray, float]:
        """Gradient of temporal smoothness penalty:  weight * ||q - q_last||².

        Returns:
            (grad: (dof,), loss: float)
        """
        diff = q - q_last
        return 2.0 * weight * diff, float(weight * np.sum(diff * diff))

"""Minimal Pinocchio FK for xArm7 — standalone, no MPlib dependency.

Used by arm_loop to replace xArm SDK ``get_position_aa()`` with URDF-consistent
FK.  The xArm firmware uses a different EEF coordinate frame definition than our
URDF; using Pinocchio FK ensures all downstream consumers (IK, recording,
visualisation) share a single coordinate system.

Usage::

    from dexmani_real.planning.arm_fk import ArmFK
    fk = ArmFK(urdf_path)
    eef_pos, eef_rot6d = fk.compute(qpos)  # qpos: (7,) rad → pos (3,) m, rot6d (6,)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class ArmFK:
    """Pinocchio FK for the 7-DOF xArm7 URDF, base-frame output.

    The Pinocchio model + data are constructed once at init (~50 ms) and then
    ``compute()`` is a cheap FK call (~0.05 ms).
    """

    def __init__(self, urdf_path: str, eef_frame_name: str = "custom_eef_link") -> None:
        import pinocchio

        self._model = pinocchio.buildModelFromUrdf(str(urdf_path))
        self._data = self._model.createData()
        self._eef_frame_id = self._model.getFrameId(eef_frame_name)

    # ------------------------------------------------------------------
    def compute(self, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute EEF pose in arm base frame.

        Args:
            qpos: Joint positions (7,) in **radians**.

        Returns:
            (eef_pos, eef_rot6d):
            - *eef_pos*  — (3,)  float64  metres
            - *eef_rot6d* — (6,)  float64  first two columns of rotation matrix
        """
        qpos = np.asarray(qpos, dtype=np.float64).ravel()[:7]
        import pinocchio

        pinocchio.forwardKinematics(self._model, self._data, qpos)
        pinocchio.updateFramePlacements(self._model, self._data)
        pose = self._data.oMf[self._eef_frame_id]

        R = pose.rotation  # 3×3
        eef_pos = np.asarray(pose.translation, dtype=np.float64).copy()
        eef_rot6d = np.concatenate([R[:, 0], R[:, 1]]).astype(np.float64)
        return eef_pos, eef_rot6d

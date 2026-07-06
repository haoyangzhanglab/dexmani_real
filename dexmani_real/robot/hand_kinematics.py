"""Hand FK helper: compute fingertip positions from hand joint positions.

Uses Pinocchio to compute forward kinematics for the hand URDF model,
returning the 3D positions of five fingertips in the hand_base frame.

Chain:
  hand_qpos -> (hand URDF via Pinocchio) -> fingertip positions in hand_base
"""

from __future__ import annotations

import numpy as np

from dexmani_real.utils.array_utils import nan_array


class HandKinematics:
    """Compute fingertip positions in the hand base frame.

    Given hand joint positions (12-dof), computes the 3D positions of up to
    5 fingertip links using Pinocchio forward kinematics on a URDF hand model.
    """

    def __init__(
        self,
        hand_urdf_path: str,
        fingertip_link_names: list[str] | None = None,
    ) -> None:
        self._model = None
        self._data = None
        self._fingertip_link_ids: list[int] = []
        self._fingertip_link_names: list[str] = []
        self._ready = False

        try:
            import pinocchio
        except ImportError:
            return

        try:
            self._model = pinocchio.buildModelFromUrdf(hand_urdf_path)
            self._data = self._model.createData()
        except ImportError:
            return
        except RuntimeError as e:
            logger.warning("HandKinematics: URDF loading failed for %s: %s", hand_urdf_path, e)
            return

        if fingertip_link_names is None:
            fingertip_link_names = [
                "right_hand_thumb_rota_tip",
                "right_hand_index_rota_tip",
                "right_hand_mid_tip",
                "right_hand_ring_tip",
                "right_hand_pinky_tip",
            ]

        all_links = list(self._model.names) if hasattr(self._model, "names") else []
        try:
            from pinocchio import FrameType

            for i, frame in enumerate(self._model.frames):
                if frame.type == FrameType.BODY:
                    all_links.append(frame.name)
        except (ImportError, RuntimeError):
            pass

        for name in fingertip_link_names:
            try:
                idx = self._model.getJointId(name)
                if idx < len(self._model.names):
                    self._fingertip_link_ids.append(idx)
                    self._fingertip_link_names.append(name)
            except (ValueError, RuntimeError):
                pass

        self._ready = len(self._fingertip_link_ids) >= 5

    def is_ready(self) -> bool:
        return self._ready

    def compute_tip_positions_in_handbase(
        self, hand_qpos: np.ndarray
    ) -> np.ndarray:
        """Returns (5, 3) fingertip positions in hand_base frame."""
        if not self._ready:
            return nan_array((5, 3))

        try:
            import pinocchio
        except ImportError:
            return nan_array((5, 3))

        q = np.asarray(hand_qpos, dtype=np.float64).reshape(12)
        pinocchio.forwardKinematics(self._model, self._data, q)
        pinocchio.updateFramePlacements(self._model, self._data)

        tips = np.zeros((5, 3), dtype=np.float64)
        for i, link_id in enumerate(self._fingertip_link_ids[:5]):
            placement = self._data.oMi[link_id]
            tips[i] = placement.translation.copy()
        return tips

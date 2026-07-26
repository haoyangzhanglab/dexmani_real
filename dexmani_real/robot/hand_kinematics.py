"""Hand FK helper: compute fingertip positions from hand joint positions.

Uses Pinocchio to compute forward kinematics for the hand URDF model,
returning the 3D positions of five fingertips in the hand_base frame.

Chain:
  hand_qpos -> (hand URDF via Pinocchio) -> fingertip positions in hand_base
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.utils.array_utils import nan_array
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)


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
        self._model: Any = None
        self._data: Any = None
        self._fingertip_frame_ids: list[int] = []
        self._fingertip_frame_names: list[str] = []
        self._ready = False

        try:
            import pinocchio
        except ImportError:
            return

        try:
            self._model = pinocchio.buildModelFromUrdf(hand_urdf_path)
            self._data = self._model.createData()
        except Exception as e:
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

        # Fingertips are URDF <frame> elements, not joints.
        # Use getFrameId() → oMf (frame placements), NOT getJointId() → oMi (joint placements).
        # desk_safety.py uses the same pattern and is verified correct on hardware.
        EXPECTED_COUNT = 5
        for name in fingertip_link_names:
            try:
                fid = self._model.getFrameId(name)
                self._fingertip_frame_ids.append(fid)
                self._fingertip_frame_names.append(name)
            except (ValueError, RuntimeError):
                logger.warning(
                    "HandKinematics: fingertip frame '%s' not found in URDF — FK will be incomplete",
                    name,
                )

        matched = len(self._fingertip_frame_ids)
        if matched < EXPECTED_COUNT:
            logger.warning(
                "HandKinematics: only %d/%d fingertip frames matched (expected %d) — "
                "fingertip FK is DISABLED. Check URDF frame names against fingertip_link_names.",
                matched,
                EXPECTED_COUNT,
                EXPECTED_COUNT,
            )
        self._ready = matched == EXPECTED_COUNT

    def is_ready(self) -> bool:
        return self._ready

    def compute_tip_positions_in_handbase(self, hand_qpos: np.ndarray) -> np.ndarray:
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
        for i, fid in enumerate(self._fingertip_frame_ids):
            # fid is a frame ID from getFrameId() → use oMf (frame placements),
            # NOT oMi (joint placements).  Same pattern as desk_safety.py:134.
            placement = self._data.oMf[fid]
            tips[i] = placement.translation.copy()
        return tips

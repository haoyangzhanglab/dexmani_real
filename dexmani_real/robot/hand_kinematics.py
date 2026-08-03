"""Hand FK helper: compute fingertip positions from hand joint positions.

Uses Pinocchio to compute forward kinematics for the hand URDF model,
returning the 3D positions of five fingertips in the hand_base frame.

Chain:
  hand_qpos -> (hand URDF via Pinocchio) -> fingertip positions in hand_base

IMPORTANT — Joint ordering:
  The XHand SDK returns qpos in finger-grouped order:
    [thumb(3), index(3), mid(2), ring(2), pinky(2)]
  The standalone hand URDF (xhand_right.urdf) has joints in a different order:
    [index(3), mid(2), pinky(2), ring(2), thumb(3)]
  _SDK_TO_URDF_IDX remaps from SDK order to URDF order before calling Pinocchio FK.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.utils.array_utils import nan_array
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Remap from XHand SDK qpos order to standalone hand URDF joint order.
#
# SDK (JOINT_NAMES from xhand.py):
#   0:thumb_abduction  1:thumb_joint1  2:thumb_joint2
#   3:index_abduction  4:index_joint1  5:index_joint2
#   6:middle_joint1    7:middle_joint2
#   8:ring_joint1      9:ring_joint2
#  10:little_joint1   11:little_joint2
#
# Standalone URDF (xhand_right.urdf, Pinocchio model.names order):
#   0:index_bend   1:index_joint1   2:index_joint2
#   3:mid_joint1   4:mid_joint2
#   5:pinky_joint1 6:pinky_joint2
#   7:ring_joint1  8:ring_joint2
#   9:thumb_bend  10:thumb_rota1   11:thumb_rota2
#
# Verified 2026-07-28 against URDF FK (xarm7_xhand_right.urdf).
_SDK_TO_URDF_IDX = np.array([3, 4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2], dtype=np.intp)


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
        q_urdf = q[_SDK_TO_URDF_IDX]  # remap SDK order → URDF order
        pinocchio.forwardKinematics(self._model, self._data, q_urdf)
        pinocchio.updateFramePlacements(self._model, self._data)

        tips = np.zeros((5, 3), dtype=np.float64)
        for i, fid in enumerate(self._fingertip_frame_ids):
            # fid is a frame ID from getFrameId() → use oMf (frame placements),
            # NOT oMi (joint placements).
            placement = self._data.oMf[fid]
            tips[i] = placement.translation.copy()
        return tips

"""Compute fingertip positions in hand_base from XHand SDK joint positions.

SDK order is thumb/index/mid/ring/pinky; the model's active q order is
index/mid/pinky/ring/thumb. The shared mapping is verified at construction.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.robot.model import (
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_SDK_TO_URDF_IDX,
    XHAND_URDF_JOINT_NAMES,
)
from dexmani_real.utils.log import get_logger

logger = get_logger(__name__)

# Remap from XHand SDK qpos order to Pinocchio model order.
# Defined in robot.model (single source of truth shared with collision.py).
_SDK_TO_URDF_IDX = np.array(HAND_SDK_TO_URDF_IDX, dtype=np.intp)


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

        active = sorted(
            (
                (joint.idx_q, self._model.names[i])
                for i, joint in enumerate(self._model.joints)
                if i != 0 and joint.nq > 0
            ),
        )
        if self._model.nq != 12 or tuple(name for _, name in active) != XHAND_URDF_JOINT_NAMES:
            raise ValueError("hand FK active q order must match XHand URDF joints")

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
                if fid >= self._model.nframes:
                    raise ValueError(f"frame {name!r} is missing")
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
            return np.full(HAND_FINGERTIP_SHAPE, np.nan)

        try:
            import pinocchio
        except ImportError:
            return np.full(HAND_FINGERTIP_SHAPE, np.nan)

        q = np.asarray(hand_qpos, dtype=np.float64).reshape(HAND_JOINT_SHAPE)
        q_urdf = q[_SDK_TO_URDF_IDX]  # remap SDK order → URDF order
        pinocchio.forwardKinematics(self._model, self._data, q_urdf)
        pinocchio.updateFramePlacements(self._model, self._data)

        tips = np.zeros(HAND_FINGERTIP_SHAPE, dtype=np.float64)
        for i, fid in enumerate(self._fingertip_frame_ids):
            # fid is a frame ID from getFrameId() → use oMf (frame placements),
            # NOT oMi (joint placements).
            placement = self._data.oMf[fid]
            tips[i] = placement.translation.copy()
        return tips

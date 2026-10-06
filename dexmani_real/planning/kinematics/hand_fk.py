"""Compute fingertip positions in hand_base from XHand SDK joint positions.

SDK order is thumb/index/mid/ring/pinky; the model's active q order is
index/mid/pinky/ring/thumb. The shared mapping is verified at construction.
"""

from __future__ import annotations

import numpy as np

from dexmani_real.robot.model import (
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_SDK_TO_URDF_IDX,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_URDF_JOINT_NAMES,
)

# Remap from XHand SDK qpos order to Pinocchio model order.
# Defined in robot.model (single source of truth shared with collision.py).
_SDK_TO_URDF_IDX = np.array(HAND_SDK_TO_URDF_IDX, dtype=np.intp)


class HandKinematics:
    """Compute fingertip positions in the hand base frame.

    Maps 12 SDK joint positions to five fingertip links using Pinocchio.
    """

    def __init__(
        self,
        hand_urdf_path: str,
        fingertip_link_names: list[str] | None = None,
    ) -> None:
        self._fingertip_frame_ids: list[int] = []
        import pinocchio

        self._pin = pinocchio
        self._model = pinocchio.buildModelFromUrdf(hand_urdf_path)
        self._data = self._model.createData()

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
            fingertip_link_names = XHAND_FINGERTIP_LINK_NAMES

        # Fingertip links use frame placements (oMf), not joint placements (oMi).
        if len(fingertip_link_names) != 5 or len(set(fingertip_link_names)) != 5:
            raise ValueError("hand FK requires five distinct fingertip frames")
        for name in fingertip_link_names:
            fid = self._model.getFrameId(name)
            if fid >= self._model.nframes:
                raise ValueError(f"hand FK frame {name!r} is missing in {hand_urdf_path}")
            self._fingertip_frame_ids.append(fid)

    def compute_tip_positions_in_handbase(self, hand_qpos: np.ndarray) -> np.ndarray:
        """Returns (5, 3) fingertip positions in hand_base frame."""
        q = np.asarray(hand_qpos, dtype=np.float64).reshape(HAND_JOINT_SHAPE)
        q_urdf = q[_SDK_TO_URDF_IDX]  # remap SDK order → URDF order
        self._pin.forwardKinematics(self._model, self._data, q_urdf)
        self._pin.updateFramePlacements(self._model, self._data)

        tips = np.zeros(HAND_FINGERTIP_SHAPE, dtype=np.float64)
        for i, fid in enumerate(self._fingertip_frame_ids):
            placement = self._data.oMf[fid]
            tips[i] = placement.translation
        return tips

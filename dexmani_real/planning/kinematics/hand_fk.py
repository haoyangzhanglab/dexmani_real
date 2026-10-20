"""Hand-base fingertip FK maps SDK thumb/index/mid/ring/pinky joints to the URDF's active order."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dexmani_real.robot.model import (
    HAND_FINGERTIP_SHAPE,
    HAND_JOINT_SHAPE,
    HAND_SDK_TO_URDF_IDX,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_URDF_JOINT_NAMES,
)

SDK_TO_URDF_IDX = np.array(HAND_SDK_TO_URDF_IDX, dtype=np.intp)


class HandKinematics:
    """Map 12 SDK joint angles to five hand-base fingertip positions with Pinocchio."""

    def __init__(
        self,
        hand_urdf_path: str,
        fingertip_link_names: Sequence[str] | None = None,
    ) -> None:
        self.fingertip_frame_ids: list[int] = []
        import pinocchio

        self.pin = pinocchio
        self.model = pinocchio.buildModelFromUrdf(hand_urdf_path)
        self.data = self.model.createData()

        active = sorted(
            (
                (joint.idx_q, self.model.names[i])
                for i, joint in enumerate(self.model.joints)
                if i != 0 and joint.nq > 0
            ),
        )
        if self.model.nq != 12 or tuple(name for _, name in active) != XHAND_URDF_JOINT_NAMES:
            raise ValueError("hand FK active q order must match XHand URDF joints")

        if fingertip_link_names is None:
            fingertip_link_names = XHAND_FINGERTIP_LINK_NAMES

        # Fingertip links use frame placements (oMf), not joint placements (oMi).
        if len(fingertip_link_names) != 5 or len(set(fingertip_link_names)) != 5:
            raise ValueError("hand FK requires five distinct fingertip frames")
        for name in fingertip_link_names:
            fid = self.model.getFrameId(name)
            if fid >= self.model.nframes:
                raise ValueError(f"hand FK frame {name!r} is missing in {hand_urdf_path}")
            self.fingertip_frame_ids.append(fid)

    def compute_tip_positions_in_handbase(self, hand_qpos: np.ndarray) -> np.ndarray:
        """Returns (5, 3) fingertip positions in hand_base frame."""
        q = np.asarray(hand_qpos, dtype=np.float64).reshape(HAND_JOINT_SHAPE)
        q_urdf = q[SDK_TO_URDF_IDX]
        self.pin.forwardKinematics(self.model, self.data, q_urdf)
        self.pin.updateFramePlacements(self.model, self.data)

        tips = np.empty(HAND_FINGERTIP_SHAPE, dtype=np.float64)
        for i, fid in enumerate(self.fingertip_frame_ids):
            placement = self.data.oMf[fid]
            tips[i] = placement.translation
        return tips

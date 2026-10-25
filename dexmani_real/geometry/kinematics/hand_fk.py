"""Hand-base link FK maps SDK thumb/index/mid/ring/pinky joints to the URDF's active order."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dexmani_real.robot.model import (
    HAND_JOINT_SHAPE,
    HAND_SDK_TO_URDF_IDX,
    XHAND_FINGERTIP_LINK_NAMES,
    XHAND_URDF_JOINT_NAMES,
)

SDK_TO_URDF_IDX = np.array(HAND_SDK_TO_URDF_IDX, dtype=np.intp)


class HandKinematics:
    """Own a Pinocchio workspace for SDK-order hand joints and requested link frames."""

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

    def compute_frame_placements(self, hand_qpos: np.ndarray, frame_ids) -> np.ndarray:
        """Return owned hand-base link transforms after one SDK-order FK update."""
        q = np.asarray(hand_qpos, dtype=np.float64)
        if q.shape != HAND_JOINT_SHAPE or not np.isfinite(q).all():
            raise ValueError("hand qpos must be finite shape (12,)")
        q_urdf = q[SDK_TO_URDF_IDX]
        self.pin.forwardKinematics(self.model, self.data, q_urdf)
        self.pin.updateFramePlacements(self.model, self.data)
        placements = np.stack([self.data.oMf[int(fid)].homogeneous for fid in frame_ids])
        if not np.isfinite(placements).all():
            raise RuntimeError("hand FK produced nonfinite link placements")
        return placements

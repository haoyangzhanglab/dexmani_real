"""Geometric FK-based fingertip-to-desk collision detection.

Uses CollisionModel's Pinocchio FK to compute the world-space Z coordinate of
all five fingertips of the dexterous hand, then compares the minimum against
the table surface height (plus a user-configurable hand-safe margin).

This is the **preferred** detection method — it requires no MPlib point cloud
pollution and avoids the ~47% IK success rate penalty (100% → 53%) observed
with the ``mplib_pointcloud`` environment collision mode.

Key design decisions:
- Uses CollisionModel's shared Pinocchio model (A1) — no duplicate model copy.
- Fingertip link indices are looked up dynamically from the Pinocchio model
  using fingertip_link_names (A3), robust to URDF link ordering changes.
- In 7-DOF mode the hand is fixed at home pose by the collision URDF; in
  19-DOF mode hand joints come from the ``_hand_qpos`` buffer (set via
  ``CollisionModel.set_hand_qpos()``), falling back to zeros when the
  buffer has not been initialized.
- A small epsilon (0.001) is applied to the fingertip threshold to prevent
  floating-point boundary misclassification.
- Path-level checks interpolate waypoints at a configurable step resolution
  (default 0.05 rad, ~2.9°) — coarser than segment collision checks but
  sufficient because the desk is a continuous plane and the hand extends
  7.6 cm below the EEF.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .collision_config import CollisionConfig
    from .collision_model import CollisionModel


class FingertipDeskSafety:
    """Geometric FK-based fingertip-to-desk collision detection.

    Uses CollisionModel's shared Pinocchio model for FK (no duplicate model).
    Always active when ``check_env_collision=True`` in the planning profile,
    independent of the CollisionModel FCL box obstacle layer.
    """

    def __init__(
        self,
        collision_model: "CollisionModel",
        collision_config: "CollisionConfig",
    ) -> None:
        self._cm = collision_model
        self._config: CollisionConfig = collision_config
        self._fingertip_ids: list[int] = self._lookup_fingertip_ids(
            collision_model.pinocchio_model, collision_config.fingertip_link_names
        )
        self._fingertip_names: tuple[str, ...] = collision_config.fingertip_link_names
        self._threshold = collision_config.fingertip_threshold
        self._epsilon = 0.001  # 1mm floating-point tolerance for boundary comparison

    # ── Private helpers ──

    @staticmethod
    def _lookup_fingertip_ids(model, names: tuple[str, ...]) -> list[int]:
        """Look up fingertip **frame** IDs from a Pinocchio model (A3).

        Fingertips are URDF ``<frame>`` elements, not joints.  We use
        ``model.getFrameId()`` to obtain frame indices that index into
        ``data.oMf`` (frame placements), as opposed to ``data.oMi`` (joint
        placements).  The matching logic extracts a finger key from the short
        name (e.g. ``thumb`` from ``thumb_tip``) and searches model frames for
        a name containing the key and ending with ``_tip``.

        Falls back to legacy hardcoded joint IDs if frame lookup fails.
        """
        def _finger_key(short_name: str) -> str:
            return short_name.replace("_tip", "").replace("_rota", "")

        ids: list[int] = []
        for short_name in names:
            fid: int | None = None
            key = _finger_key(short_name)

            for frame in model.frames:
                if key in frame.name and frame.name.endswith("_tip"):
                    fid = model.getFrameId(frame.name)
                    break

            if fid is not None:
                ids.append(fid)
            else:
                from dexmani_real.utils.log import get_logger

                legacy_ids: dict[str, int] = {
                    "thumb_tip": 20,
                    "index_tip": 26,
                    "mid_tip": 31,
                    "ring_tip": 36,
                    "pinky_tip": 41,
                }
                fallback = legacy_ids.get(short_name, -1)
                get_logger(__name__).warning(
                    "Failed to look up fingertip '%s' in Pinocchio frames — "
                    "falling back to hardcoded joint ID %d (for 19-DOF model only).",
                    short_name,
                    fallback,
                )
                ids.append(fallback)
        return ids

    # ── Public API ──

    def min_fingertip_z(self, qpos: np.ndarray) -> tuple[float, str]:
        """Compute the lowest fingertip world Z for a given arm configuration.

        Uses CollisionModel's shared Pinocchio model.  qpos must be 7-DOF arm
        joint angles; internally padded via CollisionModel.pad_arm_for_fk().

        Returns: (min_z, lowest_fingertip_name)
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        full_qpos = self._cm.pad_arm_for_fk(qpos)
        model = self._cm.pinocchio_model
        data = self._cm.pinocchio_data
        import pinocchio as pin
        pin.forwardKinematics(model, data, full_qpos)
        pin.updateFramePlacements(model, data)

        min_z = float("inf")
        min_name = ""
        for fid, name in zip(self._fingertip_ids, self._fingertip_names):
            # fid is a frame ID → use oMf (frame placements), not oMi (joint placements)
            z = float(data.oMf[fid].translation[2])
            if z < min_z:
                min_z = z
                min_name = name
        return min_z, min_name

    def check_hand_desk_clearance(self, qpos: np.ndarray) -> tuple[bool, float, str]:
        """Check if fingertips are above the table for a single configuration.

        Returns: (safe, min_fingertip_z, lowest_fingertip_name)
        """
        min_z, min_name = self.min_fingertip_z(qpos)
        safe = min_z >= self._threshold - self._epsilon
        return safe, min_z, min_name

    def check_path_desk_safety(
        self, path: np.ndarray, step_rad: float | None = None
    ) -> tuple[bool, float, int]:
        """Dense-sampled fingertip desk safety check along a joint path.

        Interpolates between consecutive waypoints at step_rad resolution
        (default from CollisionConfig.desk_safety_step_rad, 0.05 rad ≈ 2.9°)
        and checks fingertip Z at every sample.  This is coarser than the
        segment collision check (0.02 rad) but sufficient for detecting
        hand-through-desk since the hand extends 7.6 cm below EEF and the
        desk is a continuous plane.

        Path should be (N, 7) for arm-only joints.  If padded (N, >7),
        only the first 7 columns are used.

        Returns: (safe, min_fingertip_z_over_path, first_violation_segment_index)
          - violation_segment_index = -1 when all safe.
        """
        if step_rad is None:
            step_rad = self._config.desk_safety_step_rad

        path = np.asarray(path, dtype=np.float64)
        # Extract arm-only columns if padded
        if path.ndim != 2 or path.shape[1] > 7:
            path = path[:, :7]

        if len(path) < 2:
            safe, z, _name = self.check_hand_desk_clearance(path[0])
            return safe, z, 0 if not safe else -1

        min_z = float("inf")
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            dist = float(np.max(np.abs(b - a)))
            n = max(1, int(np.ceil(dist / step_rad)))
            for k in range(n + 1):
                alpha = k / max(n, 1)
                q = a + alpha * (b - a)
                safe, z, _name = self.check_hand_desk_clearance(q)
                if z < min_z:
                    min_z = z
                if not safe:
                    return False, min_z, i
        return True, min_z, -1

    # ── Properties ──

    @property
    def config(self) -> "CollisionConfig":
        return self._config

    @property
    def is_ready(self) -> bool:
        """Always True once constructed."""
        return True

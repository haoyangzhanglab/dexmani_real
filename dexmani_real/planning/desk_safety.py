"""Geometric FK-based fingertip-to-desk collision detection.

Uses Pinocchio forward kinematics to compute the world-space Z coordinate of
all five fingertips of the dexterous hand, then compares the minimum against
the table surface height (plus a user-configurable hand-safe margin).

This is the **preferred** detection method — it requires no MPlib point cloud
pollution and avoids the ~47% IK success rate penalty (100% → 53%) observed
with the ``mplib_pointcloud`` environment collision mode.

Key design decisions:
- The Pinocchio model is the collision URDF with hand joints fixed at home
  pose.  qpos is 7-DOF arm-only; ``pad_move_group_qpos`` fills remaining DOFs.
- A small epsilon (0.001) is applied to the fingertip threshold to prevent
  floating-point boundary misclassification (e.g. pinky_tip Z = 0.03000000 vs
  threshold 0.03000001).
- Path-level checks interpolate waypoints at a configurable step resolution
  (default 0.05 rad, ~2.9°) — coarser than segment collision checks but
  sufficient because the desk is a continuous plane and the hand extends
  7.6 cm below the EEF.
"""

from __future__ import annotations

import warnings

import numpy as np


class FingertipDeskSafety:
    """Geometric FK-based fingertip-to-desk collision detection.

    Uses Pinocchio FK to compute the world Z of all five fingertip links
    and compares the minimum against the table surface height.

    This is the **preferred** detection method — zero-cost, no MPlib point
    cloud pollution, and more accurate than EEF-level Z checks.  The MPlib
    point cloud approach costs ~47% IK success rate (100% → 53%) and is
    only used when env_collision_mode == "mplib_pointcloud".

    Migrated from test_motion_planning_sim.py:817-911 with identical logic.
    """

    def __init__(
        self,
        pinocchio_model,
        mp_planner,
        collision_config: "CollisionConfig",
    ) -> None:

        from .collision_config import CollisionConfig

        self._model = pinocchio_model
        self._mp_planner = mp_planner
        self._config: CollisionConfig = collision_config
        self._fingertip_ids = list(collision_config.fingertip_link_ids)
        self._fingertip_names = list(collision_config.fingertip_link_names)
        self._threshold = collision_config.fingertip_threshold
        # Floating-point tolerance for boundary comparison only.
        # Using 1e-8 (not 0.001) ensures the comparison boundary is tight —
        # the epsilon widens the comparison window, NOT the safety margin.
        self._epsilon = 1e-8
        self._table_z = collision_config.table_z_world
        self._hand_safe_margin = collision_config.hand_safe_margin

    # ── Public API ──

    def min_fingertip_z(self, qpos: np.ndarray) -> tuple[float, str]:
        """Compute the lowest fingertip world Z for a given arm configuration.

        Uses the planner's Pinocchio FK model (collision URDF, hand joints
        fixed at home pose).  qpos must be 7-DOF arm joint angles; internally
        padded to full model dimension via pad_move_group_qpos.

        Returns: (min_z, lowest_fingertip_name)
        """

        qpos = np.asarray(qpos, dtype=np.float64)
        # Must pad through pad_move_group_qpos to fill remaining DOFs
        # (same pattern as kinematics.py compute_eef_pose_base).
        full_qpos = self._mp_planner.pad_move_group_qpos(qpos)
        self._model.compute_forward_kinematics(full_qpos)

        min_z = float("inf")
        min_name = ""
        for lid, name in zip(self._fingertip_ids, self._fingertip_names):
            pose = self._model.get_link_pose(lid)
            z = float(pose.p[2])
            if z < min_z:
                min_z = z
                min_name = name
        return min_z, min_name

    def check_hand_desk_clearance(self, qpos: np.ndarray) -> tuple[bool, float, str]:
        """Check if fingertips are above the table for a single configuration.

        Uses Pinocchio FK to compute five fingertip world Z, compares the
        minimum against the table surface + safe margin.

        Returns: (safe, min_fingertip_z, lowest_fingertip_name)
        """
        min_z, min_name = self.min_fingertip_z(qpos)
        safe = min_z >= self._threshold - self._epsilon
        return safe, min_z, min_name

    def check_path_desk_safety(
        self, path: np.ndarray, step_rad: float = 0.05
    ) -> tuple[bool, float, int]:
        """Dense-sampled fingertip desk safety check along a joint path.

        Interpolates between consecutive waypoints at step_rad resolution
        (default 0.05 rad ≈ 2.9°) and checks fingertip Z at every sample.
        This is coarser than the segment collision check (0.02 rad) but
        sufficient for detecting hand-through-desk since the hand extends
        7.6 cm below EEF and the desk is a continuous plane.

        Path should be (N, 7) for arm-only joints.  If padded (N, >7),
        only the first 7 columns are used.

        Returns: (safe, min_fingertip_z_over_path, first_violation_segment_index)
          - violation_segment_index = -1 when all safe.
        """

        path = np.asarray(path, dtype=np.float64)
        # Extract arm-only columns if padded
        if path.ndim != 2 or path.shape[1] > 7:
            path = path[:, :7]

        if len(path) < 2:
            safe, z, name = self.check_hand_desk_clearance(path[0])
            if not safe:
                eef_z = float("nan")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eef_z = float(path[0][-1] if path.shape[1] > 0 else float("nan"))
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
        """Always True once constructed (construction may fail, handled by caller)."""
        return True

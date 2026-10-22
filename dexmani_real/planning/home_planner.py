"""Geometry for deterministic tabletop HOME candidates and dense path admission."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from dexmani_real.config.environment import StaticCollisionBox, TableCollisionConfig
from .collision import CollisionModel
from .kinematics.arm_fk import make_arm_kinematics
from .paths import WORKSPACE_BOUNDS_TOLERANCE_M, interpolate_waypoints


class HomePlanner:
    """Nominal-URDF arm kinematics plus the 19-DOF arm/hand collision model."""

    def __init__(
        self,
        *,
        workspace_bounds: np.ndarray | None = None,
        static_boxes: Iterable[StaticCollisionBox] = (),
        table: TableCollisionConfig | None = None,
    ) -> None:
        self.workspace_bounds = None
        if workspace_bounds is not None:
            bounds = np.asarray(workspace_bounds, dtype=np.float64)
            if (
                bounds.shape != (3, 2)
                or not np.isfinite(bounds).all()
                or np.any(bounds[:, 0] > bounds[:, 1])
            ):
                raise ValueError("workspace_bounds must be finite shape (3, 2) with lower <= upper")
            self.workspace_bounds = bounds.copy()
        self.kin = make_arm_kinematics()
        self.collision_model = CollisionModel(static_boxes=static_boxes, table=table)

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        self.collision_model.set_hand_qpos(hand_qpos)

    def check_workspace_path(self, path: np.ndarray) -> None:
        """Check EEF workspace at joint intervals of at most 0.02 rad; reject failed FK."""
        if self.workspace_bounds is None:
            return
        for index, qpos in enumerate(interpolate_waypoints(path, max_step=0.02)):
            try:
                position = self.kin.compute_eef_pose_base(qpos).p
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f"workspace FK failed at sample {index}: {exc}") from exc
            if (
                not np.isfinite(position).all()
                or np.any(position < self.workspace_bounds[:, 0] - WORKSPACE_BOUNDS_TOLERANCE_M)
                or np.any(position > self.workspace_bounds[:, 1] + WORKSPACE_BOUNDS_TOLERANCE_M)
            ):
                raise ValueError(f"Path leaves configured workspace at sample {index}: {position}")

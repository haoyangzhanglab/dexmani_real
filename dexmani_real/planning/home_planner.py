"""Geometry for deterministic tabletop HOME candidates and dense path admission."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from dexmani_real.config.environment import StaticCollisionBox, TableCollisionConfig
from dexmani_real.planning.collision import CollisionModel
from dexmani_real.planning.kinematics.arm_fk import make_arm_fk
from dexmani_real.planning.paths import WORKSPACE_BOUNDS_TOLERANCE_M, wrap_nearest_equivalent


class HomePlanner:
    """Nominal-URDF arm kinematics plus the 19-DOF arm/hand collision model."""

    def __init__(
        self,
        *,
        workspace_bounds: np.ndarray | None = None,
        static_boxes: Iterable[StaticCollisionBox] = (),
        table: TableCollisionConfig | None = None,
        table_plane_abcd: tuple[float, float, float, float] | None = None,
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
        self.arm_fk = make_arm_fk()
        self.collision_model = CollisionModel(
            static_boxes=static_boxes, table=table, table_plane_abcd=table_plane_abcd,
        )

    def canonicalize_qpos(self, qpos: np.ndarray, reference_qpos: np.ndarray) -> np.ndarray:
        model = self.arm_fk.model
        return wrap_nearest_equivalent(
            qpos, reference_qpos, model.lowerPositionLimit, model.upperPositionLimit
        )

    def set_hand_qpos(self, hand_qpos: np.ndarray) -> None:
        self.collision_model.set_hand_qpos(hand_qpos)

    def check_workspace_path(self, path: np.ndarray) -> None:
        """Check the owner's dense HOME samples against the EEF workspace."""
        if self.workspace_bounds is None:
            return
        for index, qpos in enumerate(path):
            try:
                position, _ = self.arm_fk.compute(qpos)
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f"workspace FK failed at sample {index}: {exc}") from exc
            if (
                not np.isfinite(position).all()
                or np.any(position < self.workspace_bounds[:, 0] - WORKSPACE_BOUNDS_TOLERANCE_M)
                or np.any(position > self.workspace_bounds[:, 1] + WORKSPACE_BOUNDS_TOLERANCE_M)
            ):
                raise ValueError(f"Path leaves configured workspace at sample {index}: {position}")

"""Geometry for deterministic tabletop HOME candidates and dense path admission."""

from __future__ import annotations

import numpy as np

from dexmani_real.config.environment import EnvironmentConfig
from dexmani_real.planning.collision import CollisionModel
from dexmani_real.planning.kinematics.arm_fk import ArmFK
from dexmani_real.planning.paths import WORKSPACE_BOUNDS_TOLERANCE_M, wrap_nearest_equivalent


class HomePlanner:
    """Nominal-URDF arm kinematics plus the 19-DOF arm/hand collision model."""

    def __init__(
        self,
        environment: EnvironmentConfig,
        *,
        table_plane_abcd: tuple[float, float, float, float] | None,
    ) -> None:
        self.workspace_bounds = environment.workspace.as_array()
        self.arm_fk = ArmFK()
        self.collision_model = CollisionModel(
            static_boxes=environment.static_boxes,
            table=environment.table,
            table_plane_abcd=table_plane_abcd,
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
        for index, qpos in enumerate(path):
            try:
                position, _ = self.arm_fk.compute(qpos)
            except (ValueError, RuntimeError) as exc:
                raise ValueError(f"workspace FK failed at sample {index}: {exc}") from exc
            if (
                np.any(position < self.workspace_bounds[:, 0] - WORKSPACE_BOUNDS_TOLERANCE_M)
                or np.any(position > self.workspace_bounds[:, 1] + WORKSPACE_BOUNDS_TOLERANCE_M)
            ):
                raise ValueError(f"Path leaves configured workspace at sample {index}: {position}")

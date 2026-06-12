from __future__ import annotations

import numpy as np


class WorkspaceSafety:
    """EEF workspace bounds checking and clamping.

    workspace_bounds: (3, 2) array [[x_min, x_max], [y_min, y_max], [z_min, z_max]] in meters.
    """

    def __init__(self, workspace_bounds: np.ndarray) -> None:
        self.bounds = np.asarray(workspace_bounds, dtype=np.float64)
        if self.bounds.shape != (3, 2):
            raise ValueError(f"workspace_bounds must have shape (3, 2), got {self.bounds.shape}.")

    def check(self, eef_pos: np.ndarray) -> bool:
        """Check whether EEF position is within workspace bounds."""
        eef_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
        return bool(
            (eef_pos[0] >= self.bounds[0, 0])
            and (eef_pos[0] <= self.bounds[0, 1])
            and (eef_pos[1] >= self.bounds[1, 0])
            and (eef_pos[1] <= self.bounds[1, 1])
            and (eef_pos[2] >= self.bounds[2, 0])
            and (eef_pos[2] <= self.bounds[2, 1])
        )

    def clamp(self, target_pos: np.ndarray) -> np.ndarray:
        """Clip target position to workspace bounds."""
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3).copy()
        np.clip(target_pos, self.bounds[:, 0], self.bounds[:, 1], out=target_pos)
        return target_pos

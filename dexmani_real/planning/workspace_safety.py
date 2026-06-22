"""Workspace safety bounds checking and clamping for EEF pose."""

from __future__ import annotations

import numpy as np


class WorkspaceSafety:
    """EEF workspace bounds checking and clamping.

    workspace_bounds: (3, 2) array [[x_min, x_max], [y_min, y_max], [z_min, z_max]] in meters.
    orientation_bounds: (3, 2) array [[roll_min, roll_max], [pitch_min, pitch_max], [yaw_min, yaw_max]]
        in radians (Euler XYZ). None disables orientation checking (backward compatible).
    """

    def __init__(
        self,
        workspace_bounds: np.ndarray,
        orientation_bounds: np.ndarray | None = None,
    ) -> None:
        self.bounds = np.asarray(workspace_bounds, dtype=np.float64)
        if self.bounds.shape != (3, 2):
            raise ValueError(f"workspace_bounds must have shape (3, 2), got {self.bounds.shape}.")
        self.ori_bounds = (
            None if orientation_bounds is None
            else np.asarray(orientation_bounds, dtype=np.float64)
        )

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

    # ── Orientation checking (NEW) ──

    def check_orientation(self, eef_quat_wxyz: np.ndarray) -> bool:
        """Check whether EEF orientation (as Euler XYZ) is within orientation bounds.

        Returns True if orientation_bounds is None (backward compatible).
        """
        if self.ori_bounds is None:
            return True
        from scipy.spatial.transform import Rotation

        from dexmani_real.planning.pose_utils import wxyz_to_xyzw

        euler = Rotation.from_quat(wxyz_to_xyzw(eef_quat_wxyz)).as_euler('XYZ')
        return bool(
            np.all(euler >= self.ori_bounds[:, 0])
            and np.all(euler <= self.ori_bounds[:, 1])
        )

    def clamp_orientation(self, eef_quat_wxyz: np.ndarray) -> np.ndarray:
        """Clip EEF orientation to orientation bounds. Returns clamped quat (wxyz).

        Returns input unchanged if orientation_bounds is None.
        """
        if self.ori_bounds is None:
            return np.asarray(eef_quat_wxyz, dtype=np.float64).copy()
        from scipy.spatial.transform import Rotation

        from dexmani_real.planning.pose_utils import wxyz_to_xyzw, xyzw_to_wxyz

        euler = Rotation.from_quat(wxyz_to_xyzw(eef_quat_wxyz)).as_euler('XYZ')
        euler = np.clip(euler, self.ori_bounds[:, 0], self.ori_bounds[:, 1])
        clamped_quat_xyzw = Rotation.from_euler('XYZ', euler).as_quat()
        return xyzw_to_wxyz(clamped_quat_xyzw)

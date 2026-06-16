"""ObservationBuilder — build normalized policy input from RobotState + camera."""

from __future__ import annotations

from typing import Any

import numpy as np

from dexmani_real.robot.robot_interface import RobotState


class ObservationBuilder:
    """Build a normalized observation dict for policy inference.

    Independent of the model — only handles data assembly and normalization.
    """

    def __init__(self, norm_stats: dict) -> None:
        self.norm_stats = norm_stats

    def build(
        self,
        state: RobotState,
        camera_frame: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {}

        # Arm joint positions
        obs["arm_qpos"] = self._normalize(
            state.arm_qpos, "arm_qpos"
        )

        # Hand joint positions
        obs["hand_qpos"] = self._normalize(
            state.hand_qpos, "hand_qpos"
        )

        # EEF pose
        obs["eef_pos"] = state.eef_pos.copy()
        obs["eef_quat"] = state.eef_quat_wxyz.copy()

        # Camera
        if camera_frame is not None:
            rgb = camera_frame.get("rgb")
            if rgb is not None:
                obs["rgb"] = np.asarray(rgb)
            depth = camera_frame.get("depth")
            if depth is not None:
                obs["depth"] = np.asarray(depth)

        return obs

    def _normalize(self, x: np.ndarray, key: str) -> np.ndarray:
        stats = self.norm_stats.get(key)
        if stats is None:
            return np.asarray(x, dtype=np.float64).copy()
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        std = np.where(std < 1e-8, 1.0, std)
        return (np.asarray(x, dtype=np.float64) - mean) / std

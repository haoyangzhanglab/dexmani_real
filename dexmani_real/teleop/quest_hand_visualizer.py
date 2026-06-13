"""Rerun visualizer for QuestHandTracker frames."""

from __future__ import annotations

from typing import Any

import numpy as np
from dexmani_real.robot.planner.pose_utils import normalize_quat_wxyz
from transforms3d.quaternions import quat2mat


FINGER_COLORS = [
    [255, 80, 80],    # thumb  — red
    [80, 255, 80],    # index  — green
    [80, 120, 255],   # middle — blue
    [255, 220, 80],   # ring   — gold
    [200, 80, 255],   # pinky  — purple
]

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Landmark indices per finger (including the wrist anchor at index 0)
FINGER_CHAINS = [
    [0, 1, 2, 3, 4],     # thumb
    [0, 5, 6, 7, 8],     # index
    [0, 9, 10, 11, 12],  # middle
    [0, 13, 14, 15, 16], # ring
    [0, 17, 18, 19, 20], # pinky
]

# Joint type → radius
JOINT_RADII = {
    0: 0.022,                              # wrist
    1: 0.016, 5: 0.016, 9: 0.016, 13: 0.016, 17: 0.016,  # MCP
    2: 0.012, 6: 0.012, 10: 0.012, 14: 0.012, 18: 0.012,  # PIP
    3: 0.009, 7: 0.009, 11: 0.009, 15: 0.009, 19: 0.009,  # DIP
    4: 0.007, 8: 0.007, 12: 0.007, 16: 0.007, 20: 0.007,  # TIP
}


class QuestHandVisualizer:
    """Visualize wrist pose and 21 hand landmarks with per-finger coloring."""

    def __init__(
        self,
        app_id: str = "quest-hand-debug",
        spawn: bool = True,
        show_axes: bool = True,
        point_radius: float = 0.012,
        wrist_radius: float = 0.022,
        axis_length: float = 0.08,
    ) -> None:
        import rerun as rr

        self.rr = rr
        self.show_axes = show_axes
        self.point_radius = point_radius
        self.wrist_radius = wrist_radius
        self.axis_length = axis_length

        self.rr.init(app_id, spawn=spawn)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def log_frame(self, frame: dict[str, Any], path: str = "vr/right_hand") -> None:
        wrist_pos = np.asarray(frame["wrist_pos"], dtype=np.float64)
        wrist_quat = np.asarray(frame["wrist_quat_wxyz"], dtype=np.float64)
        landmarks_local = np.asarray(frame["landmarks"], dtype=np.float64).reshape(21, 3)

        wrist_rot = quat2mat(normalize_quat_wxyz(wrist_quat))
        landmarks_world = wrist_pos + (wrist_rot @ landmarks_local.T).T

        self._log_palm(f"{path}/palm", landmarks_world)
        self._log_fingers(f"{path}/fingers", landmarks_world)
        self._log_joints(f"{path}/joints", landmarks_world)

        if self.show_axes:
            self.log_axes(f"{path}/wrist_axes", wrist_pos, wrist_quat)

    def log_points(self, path: str, points: np.ndarray, radius: float,
                   color: list[int] | None = None) -> None:
        radii = [radius] * len(points)
        colors = None if color is None else [color] * len(points)
        self.rr.log(path, self.rr.Points3D(points.tolist(), radii=radii, colors=colors))

    def log_lines(self, path: str, strips: list, colors: list | None = None) -> None:
        self.rr.log(path, self.rr.LineStrips3D(strips, colors=colors))

    def log_axes(self, path: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        rot = quat2mat(normalize_quat_wxyz(quat_wxyz))
        axes = np.eye(3) * self.axis_length
        axis_colors = [[255, 0, 0], [0, 255, 0], [0, 128, 255]]
        strips = [[pos.tolist(), (pos + rot @ axis).tolist()] for axis in axes]
        self.rr.log(path, self.rr.LineStrips3D(strips, colors=axis_colors))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_fingers(self, path: str, landmarks: np.ndarray) -> None:
        for i, chain in enumerate(FINGER_CHAINS):
            strip = landmarks[chain].tolist()
            self.rr.log(
                f"{path}/{FINGER_NAMES[i]}",
                self.rr.LineStrips3D([strip], colors=[FINGER_COLORS[i]]),
            )

    def _log_joints(self, path: str, landmarks: np.ndarray) -> None:
        for idx, radius in JOINT_RADII.items():
            color = self._joint_color(idx)
            self.rr.log(
                f"{path}/{idx}",
                self.rr.Points3D(
                    [landmarks[idx].tolist()],
                    radii=[radius],
                    colors=[color],
                ),
            )

    def _log_palm(self, path: str, landmarks: np.ndarray) -> None:
        palm_ring = [0, 1, 5, 9, 13, 17, 0]
        strip = landmarks[palm_ring].tolist()
        self.rr.log(
            path,
            self.rr.LineStrips3D([strip], colors=[[180, 150, 130]]),
        )

    @staticmethod
    def _joint_color(idx: int) -> list[int]:
        if idx == 0:
            return [255, 255, 255]  # wrist white
        for fi, chain in enumerate(FINGER_CHAINS):
            if idx in chain[1:]:
                return FINGER_COLORS[fi]
        return [200, 200, 200]

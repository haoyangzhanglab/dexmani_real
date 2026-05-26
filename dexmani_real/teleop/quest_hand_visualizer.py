"""Rerun visualizer for QuestHandTracker frames."""

from __future__ import annotations

from typing import Any

import numpy as np
from transforms3d.quaternions import quat2mat


class QuestHandVisualizer:
    """Visualize wrist pose and 21 hand landmarks."""

    finger_connections = [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
        [5, 9, 13, 17],
    ]

    def __init__(
        self,
        app_id: str = "quest-hand-debug",
        spawn: bool = True,
        show_axes: bool = True,
        point_radius: float = 0.012,
        wrist_radius: float = 0.02,
        axis_length: float = 0.08,
    ) -> None:
        import rerun as rr

        self.rr = rr
        self.show_axes = show_axes
        self.point_radius = point_radius
        self.wrist_radius = wrist_radius
        self.axis_length = axis_length

        self.rr.init(app_id, spawn=spawn)

    def log_frame(self, frame: dict[str, Any], path: str = "vr/right_hand") -> None:
        wrist_pos = np.asarray(frame["wrist_pos"], dtype=np.float64)
        wrist_quat = np.asarray(frame["wrist_quat_wxyz"], dtype=np.float64)
        landmarks = np.asarray(frame["landmarks"], dtype=np.float64).reshape(21, 3)

        self.log_points(f"{path}/wrist", wrist_pos[None], radius=self.wrist_radius)
        self.log_points(f"{path}/landmarks", landmarks, radius=self.point_radius)
        self.log_lines(f"{path}/skeleton", landmarks)

        if self.show_axes:
            self.log_axes(f"{path}/wrist_axes", wrist_pos, wrist_quat)

    def log_points(self, path: str, points: np.ndarray, radius: float) -> None:
        self.rr.log(
            path,
            self.rr.Points3D(
                points.tolist(),
                radii=[radius] * len(points),
            ),
        )

    def log_lines(self, path: str, landmarks: np.ndarray) -> None:
        strips = [landmarks[connection].tolist() for connection in self.finger_connections]
        self.rr.log(path, self.rr.LineStrips3D(strips))

    def log_axes(self, path: str, pos: np.ndarray, quat_wxyz: np.ndarray) -> None:
        rot = quat2mat(self.norm_quat(quat_wxyz))
        axes = np.eye(3) * self.axis_length
        colors = [[255, 0, 0], [0, 255, 0], [0, 128, 255]]
        strips = [[pos.tolist(), (pos + rot @ axis).tolist()] for axis in axes]
        self.rr.log(path, self.rr.LineStrips3D(strips, colors=colors))

    def norm_quat(self, quat_wxyz: np.ndarray) -> np.ndarray:
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
        norm = np.linalg.norm(quat_wxyz)
        if norm < 1e-8:
            return np.array([1.0, 0.0, 0.0, 0.0])
        return quat_wxyz / norm


def example() -> None:
    from quest_hand_tracker import QuestHandTracker

    tracker = QuestHandTracker(output_frame="flu")
    visualizer = QuestHandVisualizer(show_axes=True)

    try:
        with tracker:
            while True:
                frame = tracker.get_latest()
                if frame is not None:
                    visualizer.log_frame(frame)
    except KeyboardInterrupt:
        print("\nStopped")


if __name__ == "__main__":
    example()

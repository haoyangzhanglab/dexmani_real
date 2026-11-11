"""内部米制 RGB-D 与相机几何容器。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    """color-camera 网格的米制 RGB-D。"""

    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    distortion_model: str
    distortion_coeffs: tuple[float, float, float, float, float]

    @classmethod
    def from_camera(cls, camera, camera_row):
        """复用 CameraSnapshot 标定，将运行时相机行转为米制帧。"""
        return cls(
            rgb=camera_row["rgb"],
            depth_m=np.multiply(camera_row["depth"], camera.depth_scale_m, dtype=np.float32),
            K=camera.color.matrix(),
            distortion_model=camera.color.distortion_model,
            distortion_coeffs=camera.color.distortion_coeffs,
        )

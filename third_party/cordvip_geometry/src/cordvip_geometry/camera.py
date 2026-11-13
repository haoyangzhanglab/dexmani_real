"""米制 RGB-D、去畸变与相机射线。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    """彩色相机像素网格上的米制 RGB-D。"""

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
            depth_m=np.multiply(
                camera_row["depth"], camera.depth_scale_m, dtype=np.float32
            ),
            K=camera.color.matrix(),
            distortion_model=camera.color.distortion_model,
            distortion_coeffs=camera.color.distortion_coeffs,
        )


def camera_rays(shape, intrinsics, pixel_stride):
    """固定采样网格；Z=1 使射线参数等于相机深度。"""
    v, u = np.mgrid[0 : shape[0] : pixel_stride, 0 : shape[1] : pixel_stride]
    return np.stack(
        (
            (u - intrinsics[0, 2]) / intrinsics[0, 0],
            (v - intrinsics[1, 2]) / intrinsics[1, 1],
            np.ones_like(u),
        ),
        axis=-1,
    )


class RGBDRectifier:
    """固定相机的去畸变映射，初始化一次。"""

    def __init__(self, shape, intrinsics, distortion):
        self.maps = None
        if np.any(distortion):
            import cv2

            height, width = shape
            self.maps = cv2.initUndistortRectifyMap(
                intrinsics,
                np.asarray(distortion),
                None,
                intrinsics,
                (width, height),
                cv2.CV_32FC1,
            )

    def apply(self, rgb, depth_m):
        if self.maps is None:
            return rgb, depth_m
        import cv2

        map_x, map_y = self.maps
        rgb = cv2.remap(
            rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
        )
        depth_m = cv2.remap(
            depth_m, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT
        )
        return rgb, depth_m

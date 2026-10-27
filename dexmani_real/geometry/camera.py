"""对齐到 color 网格的相机参数与 RGB-D 反投影。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any

import numpy as np


def validate_aligned_depth_distortion(model: str) -> None:
    if model not in {"distortion.none", "none", "distortion.brown_conrady", "brown_conrady"}:
        raise ValueError(f"depth deprojection has no validated path for distortion model {model!r}")


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    distortion_model: str
    distortion_coeffs: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in (self.width, self.height)):
            raise ValueError("camera width and height must be positive integers")
        if (
            not np.isfinite((self.fx, self.fy, self.ppx, self.ppy)).all()
            or min(self.fx, self.fy) <= 0
        ):
            raise ValueError("camera intrinsics must be finite with positive focal lengths")
        coeffs = tuple(float(value) for value in self.distortion_coeffs)
        if len(coeffs) != 5 or not np.all(np.isfinite(coeffs)):
            raise ValueError("distortion_coeffs must contain five finite values")
        object.__setattr__(self, "distortion_coeffs", coeffs)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "distortion_coeffs": list(self.distortion_coeffs)}

    def matrix(self, *, dtype: np.typing.DTypeLike = np.float64) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.ppx], [0.0, self.fy, self.ppy], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )


@dataclass(frozen=True)
class CameraSnapshot:
    """实际相机的静态参数；变换由 color optical frame 到 xArm base，平移单位米。"""

    serial: str
    color: CameraIntrinsics
    depth_scale_m: float
    T_xarm_base_from_color: np.ndarray | None

    def __post_init__(self):
        if not isinstance(self.serial, str) or not self.serial.strip():
            raise ValueError("active camera serial must be nonempty")
        if not np.isfinite(self.depth_scale_m) or self.depth_scale_m <= 0:
            raise ValueError("depth_scale_m must be finite and positive")
        if self.T_xarm_base_from_color is not None:
            transform = np.array(self.T_xarm_base_from_color, dtype=np.float64, copy=True)
            transform.setflags(write=False)
            object.__setattr__(self, "T_xarm_base_from_color", transform)

    def metadata(self):
        color = self.color
        values = {
            "camera_serial": self.serial,
            "camera_payload_mode": "depth_to_color_aligned_rgbd",
            "camera_color_width": color.width,
            "camera_color_height": color.height,
            "camera_color_intrinsics": color.matrix().reshape(-1).tolist(),
            "camera_color_distortion_model": color.distortion_model,
            "camera_color_distortion_coeffs": list(color.distortion_coeffs),
            "depth_scale": self.depth_scale_m,
        }
        if self.T_xarm_base_from_color is not None:
            values["camera_T_xarm_base_from_color"] = self.T_xarm_base_from_color.reshape(
                -1
            ).tolist()
        return values


def camera_parameters_cv(
    color: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """将已校验几何转为 float32 K(9) 和 base→color [R|t](12)，采用 optical RDF 轴。"""
    parameters = {"camera_intrinsic": color.matrix(dtype=np.float32).ravel()}
    if T_xarm_base_from_color is not None:
        rotation = T_xarm_base_from_color[:3, :3].T
        translation = -rotation @ T_xarm_base_from_color[:3, 3]
        parameters["camera_extrinsic"] = (
            np.column_stack((rotation, translation)).astype(np.float32).ravel()
        )
    for values in parameters.values():
        values.setflags(write=False)
    return parameters


def undistorted_depth_coordinates(
    rows: np.ndarray,
    columns: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray]:
    validate_aligned_depth_distortion(intrinsics.distortion_model)
    xd = (np.asarray(columns, dtype=np.float32) - np.float32(intrinsics.ppx)) / np.float32(intrinsics.fx)
    yd = (np.asarray(rows, dtype=np.float32) - np.float32(intrinsics.ppy)) / np.float32(intrinsics.fy)
    if intrinsics.distortion_model in {"distortion.none", "none"}:
        x, y = xd, yd
    else:
        # 与 librealsense 一致，固定点迭代反解 Brown-Conrady 畸变。
        k1, k2, p1, p2, k3 = (np.float32(value) for value in intrinsics.distortion_coeffs)
        x, y = xd.copy(), yd.copy()
        for _ in range(10):
            r2 = x * x + y * y
            radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
            delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
            delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
            x = (xd - delta_x) / radial
            y = (yd - delta_y) / radial
    return x, y


@lru_cache(maxsize=4)
def depth_rays(intrinsics: CameraIntrinsics) -> np.ndarray:
    rows, columns = np.indices((intrinsics.height, intrinsics.width), dtype=np.float32)
    x, y = undistorted_depth_coordinates(rows, columns, intrinsics)
    rays = np.stack((x, y, np.ones_like(x)), axis=-1)
    rays.setflags(write=False)
    return rays


def deproject_depth(
    depth_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(valid)
    z = depth_m[rows, columns]
    rays = depth_rays(intrinsics)[rows, columns]
    return np.ascontiguousarray(rays * z[:, None], dtype=np.float32), rows, columns

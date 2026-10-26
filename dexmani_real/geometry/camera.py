"""Color-grid intrinsics and model camera parameters for aligned RGB-D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

__all__ = [
    "CameraIntrinsics",
    "CameraSnapshot",
    "camera_parameters_cv",
    "validate_aligned_depth_distortion",
]


def validate_aligned_depth_distortion(model: str) -> None:
    """Require a model implemented by canonical color-grid depth deprojection."""
    if model not in {"distortion.none", "none", "distortion.brown_conrady", "brown_conrady"}:
        raise ValueError(f"depth deprojection has no validated path for distortion model {model!r}")


def finite_positive(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def positive_int(value: object, *, name: str) -> int:
    """Validate one dimension without silently truncating a numeric value."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class CameraIntrinsics:
    """One camera stream's immutable active profile calibration."""

    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    distortion_model: str
    distortion_coeffs: tuple[float, float, float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", positive_int(self.width, name="width"))
        object.__setattr__(self, "height", positive_int(self.height, name="height"))
        object.__setattr__(self, "fx", finite_positive(self.fx, name="fx"))
        object.__setattr__(self, "fy", finite_positive(self.fy, name="fy"))
        for name in ("ppx", "ppy"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        model = str(self.distortion_model).strip()
        if not model:
            raise ValueError("distortion_model must be non-empty")
        object.__setattr__(self, "distortion_model", model)
        coeffs = tuple(float(value) for value in self.distortion_coeffs)
        if len(coeffs) != 5 or not np.all(np.isfinite(coeffs)):
            raise ValueError("distortion_coeffs must contain five finite values")
        object.__setattr__(
            self,
            "distortion_coeffs",
            cast(tuple[float, float, float, float, float], coeffs),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "ppx": self.ppx,
            "ppy": self.ppy,
            "distortion_model": self.distortion_model,
            "distortion_coeffs": list(self.distortion_coeffs),
        }

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
        object.__setattr__(self, "depth_scale_m", finite_positive(self.depth_scale_m, name="depth_scale"))
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
            values["camera_T_xarm_base_from_color"] = self.T_xarm_base_from_color.reshape(-1).tolist()
        return values


def camera_parameters_cv(
    color: CameraIntrinsics,
    T_xarm_base_from_color: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """将已校验几何转为 float32 K(9) 和 base→color [R|t](12)，采用 optical RDF 轴。"""
    matrices = {"camera_intrinsic": color.matrix()}
    if T_xarm_base_from_color is not None:
        matrices["camera_extrinsic"] = np.linalg.inv(T_xarm_base_from_color)[:3, :4]
    result = {}
    for name, matrix in matrices.items():
        if not np.isfinite(matrix).all() or np.any(np.abs(matrix) > np.finfo(np.float32).max):
            raise ValueError(f"{name} must be finite and representable as float32")
        values = matrix.astype(np.float32).reshape(-1)
        if name == "camera_intrinsic" and (values[0] <= 0 or values[4] <= 0):
            raise ValueError("camera_intrinsic fx/fy must remain positive in float32")
        values.setflags(write=False)
        result[name] = values
    return result

"""Pure native RGB-D camera calibration contract.

This module deliberately contains no camera SDK, shared-memory, or recording
dependencies.  The driver owns SDK extraction; point-cloud and recording code
consume this serializable snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np

__all__ = ["CameraIntrinsics", "RGBDGeometry"]


def _finite_positive(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _positive_int(value: object, *, name: str) -> int:
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
        object.__setattr__(self, "width", _positive_int(self.width, name="width"))
        object.__setattr__(self, "height", _positive_int(self.height, name="height"))
        object.__setattr__(self, "fx", _finite_positive(self.fx, name="fx"))
        object.__setattr__(self, "fy", _finite_positive(self.fy, name="fy"))
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

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CameraIntrinsics":
        required = {
            "width",
            "height",
            "fx",
            "fy",
            "ppx",
            "ppy",
            "distortion_model",
            "distortion_coeffs",
        }
        unknown = set(value) - required
        missing = required - set(value)
        if unknown or missing:
            raise ValueError(
                f"camera intrinsics keys mismatch: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        coefficients = value["distortion_coeffs"]
        if not isinstance(coefficients, (list, tuple)):
            raise TypeError("distortion_coeffs must be an array")
        return cls(
            width=_positive_int(value["width"], name="width"),
            height=_positive_int(value["height"], name="height"),
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            ppx=float(value["ppx"]),
            ppy=float(value["ppy"]),
            distortion_model=str(value["distortion_model"]),
            distortion_coeffs=cast(
                tuple[float, float, float, float, float],
                tuple(float(item) for item in coefficients),
            ),
        )

    def matrix(self, *, dtype: np.typing.DTypeLike = np.float64) -> np.ndarray:
        """Return the canonical 3x3 pinhole matrix for metadata consumers."""
        return np.array(
            [[self.fx, 0.0, self.ppx], [0.0, self.fy, self.ppy], [0.0, 0.0, 1.0]],
            dtype=dtype,
        )


@dataclass(frozen=True)
class RGBDGeometry:
    """Static depth/color calibration for one active native RGB-D profile."""

    depth: CameraIntrinsics
    color: CameraIntrinsics
    T_color_from_depth: np.ndarray

    def __post_init__(self) -> None:
        transform = np.asarray(self.T_color_from_depth, dtype=np.float64)
        if (
            transform.shape != (4, 4)
            or not np.all(np.isfinite(transform))
            or not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9)
        ):
            raise ValueError(
                "T_color_from_depth must be a finite homogeneous 4x4 matrix"
            )
        rotation = transform[:3, :3]
        if not np.allclose(
            rotation @ rotation.T, np.eye(3), atol=1e-6
        ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
            raise ValueError(
                "T_color_from_depth rotation must be orthonormal with determinant +1"
            )
        transform = transform.copy()
        transform.setflags(write=False)
        object.__setattr__(self, "T_color_from_depth", transform)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth.to_dict(),
            "color": self.color.to_dict(),
            "T_color_from_depth": self.T_color_from_depth.tolist(),
        }

    def aligned_depth_to_color(self) -> "RGBDGeometry":
        """Return geometry for ``rs.align(depth -> color)`` depth samples.

        Librealsense writes the source Z16 measurement onto the color image
        grid. Therefore its depth samples live in the color-camera coordinate
        system: color intrinsics are used for deprojection and no depth-to-color
        extrinsic is applied afterwards.
        """
        return RGBDGeometry(
            depth=self.color,
            color=self.color,
            T_color_from_depth=np.eye(4, dtype=np.float64),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RGBDGeometry":
        required = {"depth", "color", "T_color_from_depth"}
        unknown = set(value) - required
        missing = required - set(value)
        if unknown or missing:
            raise ValueError(
                f"RGB-D geometry keys mismatch: missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        depth = value["depth"]
        color = value["color"]
        if not isinstance(depth, Mapping) or not isinstance(color, Mapping):
            raise TypeError("RGB-D geometry depth/color must be objects")
        return cls(
            depth=CameraIntrinsics.from_dict(depth),
            color=CameraIntrinsics.from_dict(color),
            T_color_from_depth=np.asarray(
                value["T_color_from_depth"], dtype=np.float64
            ),
        )

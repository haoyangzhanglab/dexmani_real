"""用 RANSAC 和最小二乘拟合基座系的向上桌面平面。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TablePlaneFit:
    plane_abcd: tuple[float, float, float, float]
    inlier_points: int
    evaluated_points: int
    inlier_ratio: float
    rms_residual_m: float
    max_abs_residual_m: float
    tilt_deg: float

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        a, b, c, d = values.pop("plane_abcd")
        return {"a": a, "b": b, "c": c, "d": d, "fit": values}


def _least_squares_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.mean(points, axis=0, dtype=np.float64)
    centered = points - center
    covariance = centered.T @ centered / len(points)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    return normal, -float(normal @ center)


def fit_table_plane(points_base: np.ndarray) -> TablePlaneFit:
    distance_threshold_m, max_tilt_deg = 0.006, 20.0
    min_inlier_points, min_inlier_ratio = 1000, 0.20
    max_evaluated_points = 60_000
    points = np.asarray(points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_base must have shape [N,3]")
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < min_inlier_points:
        raise ValueError("not enough finite points for table calibration")

    rng = np.random.default_rng(0)
    if points.shape[0] > max_evaluated_points:
        evaluated = points[rng.choice(points.shape[0], size=max_evaluated_points, replace=False)]
    else:
        evaluated = points

    min_normal_z = float(np.cos(np.deg2rad(max_tilt_deg)))
    best_mask: np.ndarray | None = None
    best_count = 0
    best_mean_residual = np.inf
    for _ in range(500):
        triplet = evaluated[rng.choice(evaluated.shape[0], size=3, replace=False)]
        normal = np.cross(triplet[1] - triplet[0], triplet[2] - triplet[0])
        norm = np.linalg.norm(normal)
        if norm <= 1e-12:
            continue
        normal /= norm
        if normal[2] < 0:
            normal = -normal
        if normal[2] < min_normal_z:
            continue
        offset = -float(normal @ triplet[0])
        residual = np.abs(evaluated @ normal + offset)
        inlier = residual <= distance_threshold_m
        count = int(np.count_nonzero(inlier))
        mean_residual = float(np.mean(residual[inlier])) if count else np.inf
        if count > best_count or (count == best_count and mean_residual < best_mean_residual):
            best_mask = inlier
            best_count = count
            best_mean_residual = mean_residual

    if best_mask is None or best_count < min_inlier_points:
        raise ValueError("no table plane met the minimum inlier count")
    if best_count / evaluated.shape[0] < min_inlier_ratio:
        raise ValueError("dominant horizontal plane has insufficient support")

    normal, offset = _least_squares_plane(evaluated[best_mask])
    for _ in range(2):
        residual = np.abs(evaluated @ normal + offset)
        inlier = residual <= distance_threshold_m
        if int(np.count_nonzero(inlier)) < min_inlier_points:
            raise ValueError("refined table plane lost required support")
        normal, offset = _least_squares_plane(evaluated[inlier])

    residual = np.abs(evaluated @ normal + offset)
    inlier = residual <= distance_threshold_m
    inlier_residual = residual[inlier]
    inlier_count = int(inlier_residual.size)
    if inlier_count < min_inlier_points or inlier_count / evaluated.shape[0] < min_inlier_ratio:
        raise ValueError("final table plane has insufficient support")
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    if tilt_deg > max_tilt_deg:
        raise ValueError("refined table plane exceeds maximum tilt")
    return TablePlaneFit(
        plane_abcd=(
            float(normal[0]),
            float(normal[1]),
            float(normal[2]),
            float(offset),
        ),
        inlier_points=inlier_count,
        evaluated_points=int(evaluated.shape[0]),
        inlier_ratio=inlier_count / evaluated.shape[0],
        rms_residual_m=float(np.sqrt(np.mean(np.square(inlier_residual)))),
        max_abs_residual_m=float(np.max(inlier_residual)),
        tilt_deg=tilt_deg,
    )

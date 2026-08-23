"""Pure table-plane fitting plus explicit transactional calibration publish."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TablePlaneFit:
    """Validated upward table plane and auditable fit quality."""

    plane_abcd: tuple[float, float, float, float]
    inlier_points: int
    evaluated_points: int
    inlier_ratio: float
    rms_residual_m: float
    max_abs_residual_m: float
    tilt_deg: float

    def __post_init__(self) -> None:
        plane = np.asarray(self.plane_abcd, dtype=np.float64)
        metrics = np.asarray(
            (
                self.inlier_ratio,
                self.rms_residual_m,
                self.max_abs_residual_m,
                self.tilt_deg,
            ),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(plane[:3])) if plane.shape == (4,) else 0.0
        if (
            plane.shape != (4,)
            or not np.all(np.isfinite(plane))
            or norm <= 0.0
            or plane[2] / norm <= 0.0
        ):
            raise ValueError("table fit must contain a finite upward plane")
        if (
            self.inlier_points <= 0
            or self.evaluated_points < self.inlier_points
            or not np.all(np.isfinite(metrics))
            or not 0.0 < self.inlier_ratio <= 1.0
            or self.rms_residual_m < 0.0
            or self.max_abs_residual_m < self.rms_residual_m
            or not 0.0 <= self.tilt_deg < 90.0
        ):
            raise ValueError("table fit quality metrics are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        a, b, c, d = self.plane_abcd
        return {
            "schema_version": 1,
            "a": a,
            "b": b,
            "c": c,
            "d": d,
            "fit": {
                "inlier_points": self.inlier_points,
                "evaluated_points": self.evaluated_points,
                "inlier_ratio": self.inlier_ratio,
                "rms_residual_m": self.rms_residual_m,
                "max_abs_residual_m": self.max_abs_residual_m,
                "tilt_deg": self.tilt_deg,
            },
        }


def _normalized_upward_plane(
    normal: np.ndarray, offset: float
) -> tuple[np.ndarray, float]:
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("table plane normal is degenerate")
    normalized = np.asarray(normal, dtype=np.float64) / norm
    normalized_offset = float(offset) / norm
    if normalized[2] < 0.0:
        normalized = -normalized
        normalized_offset = -normalized_offset
    return normalized, normalized_offset


def _least_squares_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = np.mean(points, axis=0, dtype=np.float64)
    centered = points - center
    covariance = centered.T @ centered / max(1, points.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, int(np.argmin(eigenvalues))]
    return _normalized_upward_plane(normal, -float(normal @ center))


def fit_table_plane(
    points_base: np.ndarray,
    *,
    distance_threshold_m: float = 0.006,
    max_iterations: int = 500,
    max_evaluated_points: int = 60_000,
    min_inlier_points: int = 1_000,
    min_inlier_ratio: float = 0.20,
    max_tilt_deg: float = 20.0,
    random_seed: int = 0,
) -> TablePlaneFit:
    """Fit the dominant near-horizontal plane with deterministic RANSAC.

    The tilt constraint prevents a large wall from being accepted as the
    shared table calibration. The final model is refined twice by orthogonal
    least squares over its inliers.
    """
    points = np.asarray(points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_base must have shape [N,3]")
    points = points[np.all(np.isfinite(points), axis=1)]
    if points.shape[0] < max(3, min_inlier_points):
        raise ValueError("not enough finite points for table calibration")
    if not np.isfinite(distance_threshold_m) or distance_threshold_m <= 0.0:
        raise ValueError("distance_threshold_m must be finite and positive")
    if max_iterations <= 0 or max_evaluated_points < 3:
        raise ValueError("RANSAC iteration and point limits must be positive")
    if not 0.0 < min_inlier_ratio <= 1.0:
        raise ValueError("min_inlier_ratio must be in (0,1]")
    if not 0.0 < max_tilt_deg < 90.0:
        raise ValueError("max_tilt_deg must be in (0,90)")

    rng = np.random.default_rng(random_seed)
    if points.shape[0] > max_evaluated_points:
        evaluated = points[
            rng.choice(points.shape[0], size=max_evaluated_points, replace=False)
        ]
    else:
        evaluated = points

    min_normal_z = float(np.cos(np.deg2rad(max_tilt_deg)))
    best_mask: np.ndarray | None = None
    best_count = 0
    best_mean_residual = np.inf
    for _ in range(max_iterations):
        triplet = evaluated[rng.choice(evaluated.shape[0], size=3, replace=False)]
        normal = np.cross(triplet[1] - triplet[0], triplet[2] - triplet[0])
        try:
            normal, offset = _normalized_upward_plane(
                normal, -float(normal @ triplet[0])
            )
        except ValueError:
            continue
        if normal[2] < min_normal_z:
            continue
        residual = np.abs(evaluated @ normal + offset)
        inlier = residual <= distance_threshold_m
        count = int(np.count_nonzero(inlier))
        mean_residual = float(np.mean(residual[inlier])) if count else np.inf
        if count > best_count or (
            count == best_count and mean_residual < best_mean_residual
        ):
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


def publish_table_plane(
    path: str | Path, fit: TablePlaneFit, *, confirmed: bool
) -> Path | None:
    """Atomically publish a confirmed fit and return the backup path, if any."""
    if not confirmed:
        raise PermissionError(
            "table calibration publish requires explicit confirmation"
        )
    target = Path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(
            f"table calibration directory does not exist: {target.parent}"
        )

    backup: Path | None = None
    if target.exists():
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = target.with_suffix(f".json.bak.{timestamp}")
        shutil.copy2(target, backup)

    payload = fit.to_dict()
    payload["calibrated_at_utc"] = datetime.now(timezone.utc).isoformat()
    descriptor, staging_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    staging = Path(staging_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, target)
        with target.open("r", encoding="utf-8") as stream:
            readback = json.load(stream)
        stored = tuple(float(readback[name]) for name in ("a", "b", "c", "d"))
        if not np.allclose(stored, fit.plane_abcd, rtol=0.0, atol=1e-12):
            raise RuntimeError("table plane readback does not match published fit")
    finally:
        staging.unlink(missing_ok=True)
    return backup

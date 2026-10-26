"""桌面标定路径、读取与确认后发布；平面数学位于 geometry.plane。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real import PACKAGE_DIR
from dexmani_real.geometry.plane import TablePlaneFit
from dexmani_real.utils.atomic_io import atomic_json_dump

if TYPE_CHECKING:
    from dexmani_real.control.config import TableCollisionConfig


def resolve_table_plane_path(table: TableCollisionConfig) -> Path:
    """Resolve the configured plane path relative to the repository root."""
    if table.plane_path is None:
        raise RuntimeError("runtime table calibration has no plane_path")
    path = Path(table.plane_path).expanduser()
    if not path.is_absolute():
        path = PACKAGE_DIR.parent / path
    return path.resolve()


def resolve_table_plane(
    table: TableCollisionConfig, *, required=True,
) -> tuple[float, float, float, float] | None:
    """仅在消费者需要时读取一次当前桌面，数值由装配层共享。"""
    if not required:
        return None
    plane_path = resolve_table_plane_path(table)
    with plane_path.open(encoding="utf-8") as stream:
        plane = json.load(stream)
    coefficients = (float(plane["a"]), float(plane["b"]), float(plane["c"]), float(plane["d"]))
    if not np.isfinite(coefficients).all():
        raise ValueError("table plane must be finite")
    norm = np.linalg.norm(coefficients[:3])
    if norm <= 1e-9 or coefficients[2] / norm <= 0:
        raise ValueError("table plane must have an upward non-zero normal")
    return coefficients


def publish_table_plane(path: str | Path, fit: TablePlaneFit, *, confirmed: bool) -> None:
    """Replace current calibration with a validated, explicitly confirmed fit."""
    if not confirmed:
        raise PermissionError("table calibration publish requires explicit confirmation")
    payload = fit.to_dict()
    payload["calibrated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json_dump(payload, path, ensure_ascii=False)

"""桌面标定读写；平面拟合位于 geometry.plane。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dexmani_real import PACKAGE_DIR
from dexmani_real.geometry.plane import TablePlaneFit
from dexmani_real.utils.io import write_json

if TYPE_CHECKING:
    from dexmani_real.config.control import TableCollisionConfig


def resolve_table_plane_path(table: TableCollisionConfig) -> Path:
    if table.plane_path is None:
        raise RuntimeError("runtime table calibration has no plane_path")
    path = Path(table.plane_path).expanduser()
    if not path.is_absolute():
        path = PACKAGE_DIR.parent / path
    return path.resolve()


def load_table_plane(table: TableCollisionConfig) -> tuple[float, float, float, float]:
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


def save_table_plane(path: str | Path, fit: TablePlaneFit) -> None:
    payload = fit.to_dict()
    payload["calibrated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(payload, path)

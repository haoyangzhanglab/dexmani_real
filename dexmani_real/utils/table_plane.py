"""桌面标定读写；平面拟合位于 geometry.plane。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dexmani_real import PACKAGE_DIR
from dexmani_real.geometry.plane import TablePlaneFit
from dexmani_real.utils.io import write_json


def resolve_table_plane_path(value: str | Path | None) -> Path:
    if value is None:
        raise RuntimeError("runtime table calibration has no table_plane_path")
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("table_plane_path must be a non-empty path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PACKAGE_DIR.parent / path
    return path.resolve()


def load_table_plane(path: str | Path | None) -> tuple[float, float, float, float]:
    plane_path = resolve_table_plane_path(path)
    with plane_path.open(encoding="utf-8") as stream:
        plane = json.load(stream)
    coefficients = (float(plane["a"]), float(plane["b"]), float(plane["c"]), float(plane["d"]))
    return normalize_table_plane(coefficients)


def normalize_table_plane(coefficients) -> tuple[float, float, float, float]:
    plane = np.asarray(coefficients, dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError("table plane must contain four finite coefficients")
    norm = np.linalg.norm(plane[:3])
    if norm <= 1e-9 or plane[2] / norm <= 0:
        raise ValueError("table plane must have an upward non-zero normal")
    # JSON 与 Python 入口共用此表示，已归一化的系数保持逐位一致。
    if abs(norm - 1.0) > np.finfo(float).eps:
        plane = plane / norm
    a, b, c, d = plane
    return float(a), float(b), float(c), float(d)


def save_table_plane(path: str | Path, fit: TablePlaneFit) -> None:
    payload = fit.to_dict()
    payload["calibrated_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(payload, path)

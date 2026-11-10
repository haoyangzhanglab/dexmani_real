"""米制物体 Mesh 资产的准备与加载。"""

import json
from numbers import Real
from pathlib import Path

import numpy as np

from .geometry import (
    load_mesh,
    mesh_scale_from_extent,
)
from .io import existing_file, output_path, write_json


def validate_asset_parameters(
    scale_to_meters=None,
    size_axis=None,
    size_meters=None,
    *,
    allow_unscaled=False,
):
    if (size_axis is None) != (size_meters is None):
        raise ValueError("size_axis 与 size_meters 必须同时提供")
    if scale_to_meters is None and size_axis is None and not allow_unscaled:
        raise ValueError("必须提供 scale_to_meters 或完整的 size_axis/size_meters")
    if scale_to_meters is not None and size_axis is not None:
        raise ValueError("缩放因子与实测轴尺寸只能选一种")
    if size_axis is not None and size_axis not in ("x", "y", "z"):
        raise ValueError("size_axis 必须为 x、y 或 z")
    for name, value in (
        ("scale_to_meters", scale_to_meters),
        ("size_meters", size_meters),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} 必须为有限正数")


def prepare_asset(
    mesh_path,
    output,
    *,
    scale_to_meters=None,
    size_axis=None,
    size_meters=None,
):
    validate_asset_parameters(scale_to_meters, size_axis, size_meters)
    source = existing_file(mesh_path)
    output = output_path(output, protected=(source,))
    mesh = load_mesh(source)
    if scale_to_meters is None:
        scale_to_meters = mesh_scale_from_extent(
            mesh, "xyz".index(size_axis), float(size_meters)
        )
    scale_to_meters = float(scale_to_meters)
    # 缩放原点偏移，保留唯一 object frame。
    mesh.apply_scale(scale_to_meters)
    output.mkdir(parents=True, exist_ok=False)
    mesh_file = output / "mesh/mesh_m.obj"
    mesh_file.parent.mkdir()
    mesh.export(mesh_file, file_type="obj")
    metadata = {
        "mesh_file": mesh_file.relative_to(output).as_posix(),
        "frame": "object",
        "units": "m",
    }
    write_json(output / "asset.json", metadata)
    return output


def load_asset(path):
    path = Path(path).expanduser().resolve()
    metadata = json.loads((path / "asset.json").read_text())
    if metadata.get("units") != "m" or metadata.get("frame") != "object":
        raise ValueError("资产必须使用 object frame 和米制单位")
    file = (path / metadata["mesh_file"]).resolve()
    if not file.is_relative_to(path):
        raise ValueError("资产 Mesh 必须位于资产目录内")
    return load_mesh(file), metadata

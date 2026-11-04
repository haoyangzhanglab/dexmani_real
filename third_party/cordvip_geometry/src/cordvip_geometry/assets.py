"""米制物体资产与固定表面采样。"""

import json
from numbers import Integral, Real
from os.path import relpath
from pathlib import Path

import numpy as np

from .geometry import (
    load_mesh,
    load_points,
    mesh_scale_from_extent,
    sample_mesh_surface,
    save_points,
)
from .io import existing_file, output_path, write_json


def validate_asset_parameters(
    scale_to_meters=None,
    size_axis=None,
    size_meters=None,
    num_points=1024,
    seed=0,
    *,
    allow_unscaled=False,
):
    if (
        isinstance(num_points, bool)
        or not isinstance(num_points, Integral)
        or num_points < 1
    ):
        raise ValueError("num_points 必须是正整数")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed 必须是非负整数")
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


def write_asset_samples(directory, mesh, metadata, *, num_points, seed):
    """写固定点与法向，并同步采样元数据。"""
    num_points, seed = int(num_points), int(seed)
    points, normals = sample_mesh_surface(mesh, num_points, seed)
    save_points(directory / "points.npy", points)
    save_points(directory / "normals.npy", normals)
    metadata.update(
        points_file="points.npy",
        normals_file="normals.npy",
        num_points=num_points,
        seed=seed,
        sampling="area_weighted_triangle_barycentric",
        normals_method="mesh_face",
    )
    return points, normals


def prepare_asset(
    mesh_path,
    output,
    *,
    scale_to_meters=None,
    size_axis=None,
    size_meters=None,
    num_points=1024,
    seed=0,
):
    validate_asset_parameters(scale_to_meters, size_axis, size_meters, num_points, seed)
    source = existing_file(mesh_path)
    output = output_path(output, protected=(source,))
    mesh = load_mesh(source)
    measured_extent = None
    if scale_to_meters is None:
        scale_to_meters = mesh_scale_from_extent(
            mesh, "xyz".index(size_axis), float(size_meters)
        )
        measured_extent = {"axis": size_axis, "extent_m": float(size_meters)}
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
        "source_mesh": relpath(source, output),
        "scale_to_meters": scale_to_meters,
        "measured_extent": measured_extent,
        "bounds_m": mesh.bounds.tolist(),
    }
    write_asset_samples(output, mesh, metadata, num_points=num_points, seed=seed)
    write_json(output / "asset.json", metadata)
    return output


def load_asset(path):
    path = Path(path).expanduser().resolve()
    metadata = json.loads((path / "asset.json").read_text())
    if metadata.get("units") != "m" or metadata.get("frame") != "object":
        raise ValueError("资产必须使用 object frame 和米制单位")
    files = {}
    for name in ("mesh", "points", "normals"):
        file = (path / metadata[f"{name}_file"]).resolve()
        if not file.is_relative_to(path):
            raise ValueError("资产文件必须位于资产目录内")
        files[name] = file
    mesh = load_mesh(files["mesh"])
    points = load_points(files["points"])
    normals = np.load(files["normals"], allow_pickle=False)
    if (
        normals.dtype.kind not in "fiu"
        or normals.shape != points.shape
        or not np.isfinite(normals).all()
    ):
        raise ValueError("资产法向必须为与固定点对应的有限实数 (N, 3) 数组")
    if not np.allclose(
        np.linalg.norm(normals.astype(np.float64), axis=1), 1, rtol=0, atol=1e-5
    ):
        raise ValueError("资产法向必须是单位向量")
    normals = normals.astype(np.float32)
    points.setflags(write=False)
    normals.setflags(write=False)
    return mesh, points, normals, metadata

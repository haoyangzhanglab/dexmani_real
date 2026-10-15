#!/usr/bin/env python3
"""将已有 mesh 缩放为米制资产，保存固定表面点与法向；保留原点和朝向。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from ..io import _existing_file, _output_path, _sha256, _write_json
from ._common import _positive_float, _positive_int


def _prepare(args: argparse.Namespace) -> None:
    if (args.size_axis is None) != (args.size_meters is None):
        raise ValueError("--size-axis 与 --size-meters 必须同时提供")
    if args.scale_to_meters is None and args.size_axis is None:
        raise ValueError("必须提供 --scale-to-meters 或 --size-axis 与 --size-meters")
    if args.scale_to_meters is not None and args.size_axis is not None:
        raise ValueError("缩放因子和实测轴尺寸只能选择一种")
    if args.seed < 0:
        raise ValueError("--seed 必须是非负整数")
    source = _existing_file(args.mesh)
    output = _output_path(args.output_dir, workspace=args.workspace)
    import trimesh
    from ..geometry import (
        load_metric_mesh,
        mesh_scale_from_extent,
        sample_mesh_surface,
        save_points,
    )

    mesh = load_metric_mesh(source, scale_to_meters=1.0)
    scale = args.scale_to_meters
    measured_extent = None
    if scale is None:
        axis = "xyz".index(args.size_axis)
        scale = mesh_scale_from_extent(mesh, axis, args.size_meters)
        measured_extent = {"axis": args.size_axis, "extent_m": args.size_meters}
    # 米制缩放也作用于原点偏移；不自动居中，以保留唯一 object frame。
    mesh.apply_scale(scale)
    points, normals = sample_mesh_surface(mesh, args.num_points, args.seed)
    output.mkdir(parents=True, exist_ok=False)
    mesh_file = output / "mesh_m.obj"
    mesh.export(mesh_file, file_type="obj")
    save_points(output / "points.npy", points)
    save_points(output / "normals.npy", normals)
    metadata = {
        "mesh_file": mesh_file.name,
        "points_file": "points.npy",
        "normals_file": "normals.npy",
        "frame": "object",
        "units": "m",
        "source_mesh": str(source),
        "source_mesh_sha256": _sha256(source),
        "mesh_sha256": _sha256(mesh_file),
        "points_sha256": _sha256(output / "points.npy"),
        "normals_sha256": _sha256(output / "normals.npy"),
        "scale_to_meters": scale,
        "measured_extent": measured_extent,
        "origin": "source_mesh_origin_scaled_without_recentering",
        "num_points": args.num_points,
        "seed": args.seed,
        "sampling": "area_weighted_triangle_barycentric",
        "normals_method": "mesh_face",
        "bounds_m": mesh.bounds.tolist(),
        "trimesh_version": trimesh.__version__,
    }
    _write_json(output / "asset.json", metadata)
    print(f"已准备物体资产：{output / 'asset.json'}")
    if args.preview:
        import open3d as o3d

        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.normals = o3d.utility.Vector3dVector(normals)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=float(mesh.extents.max()) * 0.25
        )
        o3d.visualization.draw_geometries([cloud, frame], window_name="物体点云（米）")


def build_parser(*, workspace=None) -> argparse.ArgumentParser:
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--scale-to-meters", type=_positive_float, help="源单位到米的缩放因子；已有米制 mesh 显式传 1")
    parser.add_argument("--size-axis", choices=("x", "y", "z"), help="源 mesh 坐标下用于尺度校准的轴")
    parser.add_argument("--size-meters", type=_positive_float, help="该轴方向的物体实测包围盒长度，单位米")
    parser.add_argument("--num-points", type=_positive_int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True, help="reports/ 下尚不存在的目录")
    parser.add_argument("--preview", action="store_true", help="保存后用 Open3D 查看点云")
    return parser


def main(argv=None, *, workspace=None) -> int:
    parser = build_parser(workspace=workspace)
    args = parser.parse_args(argv)
    try:
        _prepare(args)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""离线重建刚性物体，并准备米制 mesh 与固定表面点。

reconstruct 调用已有 TripoSR 环境，不下载模型或自动去背景。
prepare 仅缩放几何，不改变 mesh 原点和朝向，也不修改源文件。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from ..io import _existing_file, _output_path, _sha256, _write_json
from ._common import _positive_float, _positive_int


def _reconstruct(args: argparse.Namespace) -> None:
    root = args.triposr_root.expanduser().resolve()
    script = _existing_file(root / "run.py")
    weights = args.weights.expanduser().resolve()
    config = _existing_file(weights / "config.yaml")
    checkpoint = _existing_file(weights / "model.ckpt")
    source_image = _existing_file(args.image)
    output = _output_path(args.output_dir, workspace=args.workspace)
    python_command = str(Path(args.python).expanduser())
    if os.path.sep in python_command:
        python_command = str(_existing_file(Path(python_command)))
    python_command = shutil.which(python_command)
    if python_command is None:
        raise ValueError(f"Python 可执行文件不存在或不可执行：{args.python}")
    environment = os.environ.copy()
    environment.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
    )

    # 上游会在缺少 CUDA 时静默回退 CPU；这里要求使用者显式选择 CPU。
    preflight = """
import sys
import torch
from PIL import Image
device = torch.device(sys.argv[1])
if device.type == 'cuda':
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用；如需 CPU 重建，请显式指定 --device cpu')
    index = 0 if device.index is None else device.index
    if index >= torch.cuda.device_count():
        raise RuntimeError('指定的 CUDA device 不存在')
elif device.type != 'cpu':
    raise RuntimeError('本示例仅接受 cuda 或 cpu device')
with Image.open(sys.argv[2]) as image:
    if image.mode != 'RGB':
        raise RuntimeError('输入必须是已经去背景、合成灰底并裁剪好的 RGB 图像')
    image.verify()
"""
    subprocess.run(
        [python_command, "-c", preflight, args.device, str(source_image)],
        cwd=root,
        env=environment,
        check=True,
    )
    command = [
        python_command,
        str(script),
        str(source_image),
        "--pretrained-model-name-or-path",
        str(weights),
        "--no-remove-bg",
        "--device",
        args.device,
        "--mc-resolution",
        str(args.mc_resolution),
        "--model-save-format",
        "obj",
        "--output-dir",
        str(output),
    ]
    provenance = {
        "method": "TripoSR",
        "source_image": str(source_image),
        "source_image_sha256": _sha256(source_image),
        "run_py": str(script),
        "run_py_sha256": _sha256(script),
        "weights_directory": str(weights),
        "config_sha256": _sha256(config),
        "checkpoint_sha256": _sha256(checkpoint),
        "command": command,
        "input_background": "user_preprocessed_gray_rgb",
        "units": "unscaled",
    }
    output.mkdir(parents=True, exist_ok=False)
    # TripoSR 的 --no-remove-bg 分支不创建这个逐图像输出目录。
    (output / "0").mkdir()
    _write_json(output / "reconstruction_input.json", provenance)
    print(shlex.join(command), flush=True)
    result = subprocess.run(command, cwd=root, env=environment, check=False)
    mesh_path = output / "0" / "mesh.obj"
    if result.returncode != 0:
        raise RuntimeError(
            f"TripoSR 退出码 {result.returncode}；保留未完成输出供诊断：{output}"
        )
    _existing_file(mesh_path)
    _write_json(
        output / "reconstruction_result.json",
        {"mesh_file": "0/mesh.obj", "mesh_sha256": _sha256(mesh_path), "units": "unscaled"},
    )
    print(f"已生成未定尺度 mesh：{mesh_path}；请使用 prepare 指定实际尺寸。")


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
    commands = parser.add_subparsers(dest="command", required=True)
    reconstruct = commands.add_parser("reconstruct", help="用本地 TripoSR 从灰底 RGB 图重建 mesh")
    reconstruct.add_argument("--triposr-root", type=Path, required=True, help="已有 TripoSR 源码目录")
    reconstruct.add_argument("--weights", type=Path, required=True, help="含 config.yaml 与 model.ckpt 的本地目录")
    reconstruct.add_argument("--image", type=Path, required=True, help="已去背景、合成灰底并裁剪好的 RGB 图像")
    reconstruct.add_argument("--python", default=sys.executable, help="已有 TripoSR 环境的 Python 可执行文件")
    reconstruct.add_argument("--device", default="cuda:0", help="cuda:0 或显式 cpu；不自动回退")
    reconstruct.add_argument("--mc-resolution", type=_positive_int, default=256)
    reconstruct.add_argument("--output-dir", type=Path, required=True, help="reports/ 下尚不存在的目录")
    reconstruct.set_defaults(run=_reconstruct)
    prepare = commands.add_parser("prepare", help="将 mesh 缩放至米制，保存固定表面点")
    prepare.add_argument("--mesh", type=Path, required=True)
    prepare.add_argument("--scale-to-meters", type=_positive_float, help="源单位到米的缩放因子；已有米制 mesh 显式传 1")
    prepare.add_argument("--size-axis", choices=("x", "y", "z"), help="源 mesh 坐标下用于尺度校准的轴")
    prepare.add_argument("--size-meters", type=_positive_float, help="该轴方向的物体实测包围盒长度，单位米")
    prepare.add_argument("--num-points", type=_positive_int, default=1024)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--output-dir", type=Path, required=True, help="reports/ 下尚不存在的目录")
    prepare.add_argument("--preview", action="store_true", help="保存后用 Open3D 查看点云")
    prepare.set_defaults(run=_prepare)
    return parser


def main(argv=None, *, workspace=None) -> int:
    parser = build_parser(workspace=workspace)
    args = parser.parse_args(argv)
    try:
        args.run(args)
    except (OSError, ValueError, ImportError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

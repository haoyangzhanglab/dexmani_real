#!/usr/bin/env python3
"""单图建模：Grounded-SAM2 选物体、Hunyuan3D-2.1 生成 mesh 与点云、显式定尺度。

这里只读取照片或已保存的 RGB-D，不连接任何设备。segment / select /
generate 可分别运行；build 使用子进程依次执行，阶段结束即释放 GPU。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from ..io import _existing_file, _output_path, _sha256, _write_json
from .._process import module_environment
from ._common import _positive_float, _positive_int


def _display_available() -> None:
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        raise RuntimeError("没有图形会话；请使用 --headless --instance N，或 --list-only 查看候选后 select")


def _load_rgb(args):
    import numpy as np
    from PIL import Image, ImageOps

    path = _existing_file(args.image if args.image is not None else args.rgbd)
    info = {"source_file": str(path), "source_sha256": _sha256(path)}
    if args.image is not None:
        with Image.open(path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
        info.update(input_type="image", exif_orientation="applied_to_saved_rgb")
    else:
        from ..rgbd import load_frame

        frame = load_frame(path)
        rgb = frame.rgb
        info.update(input_type="saved_rgbd", source_id=frame.source_id,
                    timestamp_ns=frame.timestamp_ns, clock=frame.clock,
                    depth_usage="preserved_only; no_automatic_scale_estimation",
                    mask_grid="original_rgb_grid; not_automatically_rectified")
    info["rgb_shape"] = list(rgb.shape)
    return rgb, path, info


def _overlay(rgb, candidate, index):
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = rgb.copy()
    mask = candidate["mask"]
    canvas[mask] = (canvas[mask] * 0.4 + np.array([30, 220, 100]) * 0.6).astype(np.uint8)
    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    draw.rectangle(tuple(float(x) for x in candidate["box_xyxy"]), outline=(255, 180, 0), width=2)
    caption = f"#{index} {candidate['label']}  detection={candidate['score']:.3f}"
    draw.rectangle((0, 0, image.width, 22), fill=(20, 20, 20))
    draw.text((5, 4), caption, fill=(255, 255, 255))
    return image


def _save_candidates(output, rgb, candidates):
    import numpy as np
    from PIL import Image

    np.savez_compressed(output / "candidates.npz",
                        masks=np.stack([c["mask"] for c in candidates]),
                        boxes=np.stack([c["box_xyxy"] for c in candidates]),
                        scores=np.asarray([c["score"] for c in candidates]),
                        mask_scores=np.asarray([c["mask_score"] for c in candidates]),
                        labels=np.asarray([c["label"] for c in candidates], dtype=str))
    width = min(rgb.shape[1], 420)
    height = max(1, round(rgb.shape[0] * width / rgb.shape[1]))
    columns = min(3, len(candidates))
    sheet = Image.new("RGB", (columns * width, ((len(candidates) + columns - 1) // columns) * height))
    for index, candidate in enumerate(candidates):
        thumb = _overlay(rgb, candidate, index).resize((width, height))
        sheet.paste(thumb, ((index % columns) * width, (index // columns) * height))
        print(f"候选 {index}: {candidate['label']}，检测分数 {candidate['score']:.3f}", flush=True)
    sheet.save(output / "candidates.jpg", quality=95)


def _choose(rgb, candidates, instance, *, headless):
    if instance is not None and not 0 <= instance < len(candidates):
        raise ValueError(f"--instance 必须位于 0..{len(candidates) - 1}；索引参见 candidates.jpg")
    if headless:
        if instance is None:
            raise ValueError("headless 必须显式指定 --instance N；不会按检测分数自动选物体")
        return instance
    _display_available()
    import cv2
    import numpy as np

    index = 0 if instance is None else instance
    title = "Grounded-SAM2: n/p candidate; Enter confirm; Esc cancel"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(rgb.shape[1], 1280), min(rgb.shape[0], 850))
    try:
        while True:
            canvas = np.asarray(_overlay(rgb, candidates[index], index))
            cv2.imshow(title, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(50) & 0xFF
            if key in (10, 13):
                return index
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("已取消候选确认；候选文件保留，可通过 select 继续")
            if key in (ord("n"), ord(" "), ord("]")):
                index = (index + 1) % len(candidates)
            elif key in (ord("p"), ord("[")):
                index = (index - 1) % len(candidates)
    finally:
        cv2.destroyWindow(title)


def _save_selection(output, rgb, candidates, index, provenance):
    import numpy as np
    from PIL import Image

    candidate = candidates[index]
    mask = candidate["mask"]
    if not mask.any() or mask.all():
        raise ValueError("生成输入必须包含非空物体和非空背景；请重新分割或拍摄")
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / "mask.png")
    np.save(output / "mask.npy", mask, allow_pickle=False)
    Image.fromarray(np.dstack((rgb, mask.astype(np.uint8) * 255))).save(output / "foreground_rgba.png")
    _overlay(rgb, candidate, index).save(output / "overlay.png")
    _write_json(output / "selection.json", {
        **provenance, "selected_instance": index, "selection": "explicit_headless_instance" if provenance.get("headless") else "gui_confirmed",
        "label": candidate["label"], "box_xyxy": candidate["box_xyxy"].tolist(),
        "detection_score": candidate["score"], "score_kind": "grounding_dino",
        "mask_score": candidate["mask_score"], "mask_score_kind": "sam2_predicted_iou",
        "rgb_sha256": _sha256(output / "rgb.png"),
        "mask_sha256": _sha256(output / "mask.png"),
        "foreground_sha256": _sha256(output / "foreground_rgba.png"),
        "foreground_file": "foreground_rgba.png", "mask_file": "mask.png",
    })
    print(f"已确认候选 {index}；全分辨率 mask、RGBA 与 overlay：{output}", flush=True)


def _segment(args):
    if args.headless and args.instance is None and not args.list_only:
        raise ValueError("headless 须提供 --instance N；未知索引时先使用 --list-only")
    if not args.headless and not args.list_only:
        _display_available()
    rgb, source, provenance = _load_rgb(args)
    output = _output_path(args.output_dir, workspace=args.workspace)
    output.mkdir(parents=True)
    from PIL import Image

    Image.fromarray(rgb).save(output / "rgb.png")
    source_name = "source_rgbd.npz" if args.rgbd is not None else f"source{source.suffix.lower()}"
    shutil.copyfile(source, output / source_name)
    provenance.update(method="Grounded-SAM2 text segmentation", prompt=args.text,
                      sam_root=str(args.sam_root.expanduser().resolve()),
                      sam_python=str(args.sam_python.expanduser().absolute()),
                      sam_checkpoint=str(args.sam_checkpoint.expanduser().resolve()),
                      sam_config=args.sam_config, dino_weights=str(args.dino_weights.expanduser().resolve()),
                      box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                      device='cuda:0', headless=args.headless,
                      saved_source=source_name, candidate_order="descending_grounding_dino_score; zero_based")
    _write_json(output / "segmentation_request.json", provenance)
    from ..segmentation import GroundedSAM2Segmenter
    segmenter = GroundedSAM2Segmenter(args.sam_python, args.sam_root, args.sam_checkpoint,
                                     sam_config=args.sam_config, dino_weights=args.dino_weights,
                                     log_path=output / 'sam2_worker.log')
    try:
        segmenter.start()
        candidates = segmenter.predict(rgb, args.text, box_threshold=args.box_threshold,
                                       text_threshold=args.text_threshold)
        provenance.update(model=segmenter.provenance, performance=segmenter.last_metrics)
    finally:
        segmenter.close()
    if not candidates:
        _write_json(output / "segmentation_result.json", {**provenance, "num_candidates": 0})
        raise RuntimeError(f"没有检测到目标；原图和请求已保留在 {output}，可调整文本或阈值")
    _save_candidates(output, rgb, candidates)
    provenance.update(num_candidates=len(candidates), rgb_sha256=_sha256(output / "rgb.png"),
                      candidates_sha256=_sha256(output / "candidates.npz"))
    _write_json(output / "segmentation_result.json", provenance)
    if args.list_only:
        print(f"候选已保存：{output / 'candidates.jpg'}；使用 select 确认，无需重跑模型。")
        return
    index = _choose(rgb, candidates, args.instance, headless=args.headless)
    _save_selection(output, rgb, candidates, index, provenance)


def _select(args):
    import numpy as np
    from PIL import Image

    source = args.segmentation_dir.expanduser().resolve()
    output = _output_path(args.output_dir, workspace=args.workspace)
    provenance = json.loads(_existing_file(source / "segmentation_result.json").read_text())
    for name, key in (("rgb.png", "rgb_sha256"), ("candidates.npz", "candidates_sha256")):
        if _sha256(_existing_file(source / name)) != provenance[key]:
            raise ValueError(f"{name} 与分割记录的 SHA256 不一致，请重新分割")
    with Image.open(_existing_file(source / "rgb.png")) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with np.load(_existing_file(source / "candidates.npz"), allow_pickle=False) as saved:
        candidates = [dict(mask=mask, box_xyxy=saved["boxes"][i], score=float(saved["scores"][i]),
                           mask_score=float(saved["mask_scores"][i]),
                           label=str(saved["labels"][i]))
                      for i, mask in enumerate(saved["masks"])]
    if not candidates or any(c["mask"].dtype != np.bool_ or c["mask"].shape != rgb.shape[:2] for c in candidates):
        raise ValueError("保存的候选 mask 与 RGB 不一致")
    index = _choose(rgb, candidates, args.instance, headless=args.headless)
    output.mkdir(parents=True)
    for name in ("rgb.png", "candidates.npz", "candidates.jpg", "segmentation_request.json", "segmentation_result.json", provenance["saved_source"]):
        file = (source / name).resolve()
        if not file.is_relative_to(source):
            raise ValueError("候选元数据不能引用输入目录以外的文件")
        shutil.copyfile(_existing_file(file), output / name)
    provenance.update(headless=args.headless, selection_source=str(source))
    _save_selection(output, rgb, candidates, index, provenance)


def _preview_mesh(mesh_path, *, units="unscaled", points=None, normals=None):
    _display_available()
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError("预览 mesh 为空或没有三角面")
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.65, 0.78, 0.87])
    bounds = mesh.get_axis_aligned_bounding_box()
    bounds.color = [0.85, 0.65, 0.15]
    extent = bounds.get_extent()
    print(f"mesh 轴向尺寸 x/y/z ({units}): {extent.tolist()}；红 x、绿 y、蓝 z。", flush=True)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(extent.max()) * 0.35)
    geometries = [mesh, bounds, frame]
    if points is not None:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.normals = o3d.utility.Vector3dVector(normals)
        cloud.paint_uniform_color([0.9, 0.2, 0.1])
        geometries.append(cloud)
    title = "Object mesh + surface points" if points is not None else "Object mesh"
    o3d.visualization.draw_geometries(geometries, window_name=f"{title} ({units}); x=red y=green z=blue", mesh_show_back_face=True)


def _generate(args):
    if args.preview:
        _display_available()
    source = args.segmentation_dir.expanduser().resolve()
    metadata = json.loads(_existing_file(source / "selection.json").read_text())
    image = _existing_file(source / "foreground_rgba.png")
    if _sha256(image) != metadata["foreground_sha256"]:
        raise ValueError("RGBA 与分割记录 SHA256 不一致，请重新确认分割")
    output = _output_path(args.output_dir, workspace=args.workspace)
    output.mkdir(parents=True)
    _write_json(output / "generation_request.json", {
        "method": "Hunyuan3D-2.1 shape_only", "segmentation_directory": str(source),
        "selection_sha256": _sha256(source / "selection.json"),
        "foreground_sha256": metadata["foreground_sha256"],
        "source_root": str(args.hunyuan_root.expanduser().resolve()), "weights_dir": str(args.weights.expanduser().resolve()),
        "seed": args.seed, "steps": args.steps, "octree_resolution": args.octree_resolution,
        "num_chunks": args.num_chunks, "device": args.device, "units": "unscaled",
        "num_points": args.num_points,
    })
    from ..reconstruction import generate_mesh

    mesh_path = output / "mesh_unscaled.obj"
    result = generate_mesh(image_rgba_path=image, source_root=args.hunyuan_root,
                           weights_dir=args.weights, output_mesh_path=mesh_path, seed=args.seed,
                           steps=args.steps, octree_resolution=args.octree_resolution,
                           num_chunks=args.num_chunks, device=args.device)
    _existing_file(mesh_path)
    from ..geometry import load_metric_mesh, sample_mesh_surface, save_points

    # 保留生成模型原点、朝向和任意单位；单张 RGB 不能给出绝对米制尺度。
    mesh = load_metric_mesh(mesh_path, scale_to_meters=1.0)
    points, normals = sample_mesh_surface(mesh, args.num_points, args.seed)
    points_path, normals_path = output / "points_unscaled.npy", output / "normals.npy"
    save_points(points_path, points)
    save_points(normals_path, normals)
    _write_json(output / "generation_result.json", {**result, "mesh_file": mesh_path.name,
                "mesh_sha256": _sha256(mesh_path), "units": "unscaled", "frame": "object",
                "points_file": points_path.name, "normals_file": normals_path.name,
                "points_sha256": _sha256(points_path), "normals_sha256": _sha256(normals_path),
                "num_points": args.num_points, "seed": args.seed,
                "sampling": "area_weighted_triangle_barycentric", "normals_method": "mesh_face",
                "origin": "source_mesh_origin_without_recentering"})
    print(f"未定尺度 mesh：{mesh_path}；表面点云：{points_path}。准备追踪资产前必须提供实测尺寸或明确缩放系数。", flush=True)
    if args.preview:
        _preview_mesh(mesh_path, points=points, normals=normals)


def _add_segment_arguments(parser, *, workspace):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="物体照片；按 EXIF 纠正朝向")
    source.add_argument("--rgbd", type=Path, help="单帧已保存的规范 RGB-D NPZ；不连接相机")
    parser.add_argument("--text", required=True, help="英文物体描述，例如 'red mug.'")
    from ..sam2_config import add_sam_arguments
    add_sam_arguments(parser, workspace)
    parser.add_argument("--instance", type=int, help="零起始候选索引；headless 必须显式选择")
    parser.add_argument("--headless", action="store_true", help="不创建窗口；必须提供 --instance")


def _add_generate_arguments(parser, *, workspace):
    models = workspace / "reports" / "cordvip_models"
    source_root = workspace / "third_party"
    parser.add_argument("--hunyuan-root", type=Path, default=source_root / "Hunyuan3D-2.1")
    parser.add_argument("--weights", type=Path, default=models / "Hunyuan3D-2.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=_positive_int, default=50)
    parser.add_argument("--octree-resolution", type=_positive_int, default=384)
    parser.add_argument("--num-chunks", type=_positive_int, default=8000)
    parser.add_argument("--num-points", type=_positive_int, default=1024, help="固定表面点数；点云与 mesh 使用相同坐标系和单位")
    parser.add_argument("--preview", action="store_true", help="生成后交互查看 mesh、表面点云和坐标轴")


def _build(args):
    if args.headless and args.preview:
        raise ValueError("--headless 与 --preview 不能同时使用")
    if (args.size_axis is None) != (args.size_meters is None):
        raise ValueError("--size-axis 与 --size-meters 必须一起提供")
    if args.scale_to_meters is not None and args.size_axis is not None:
        raise ValueError("--scale-to-meters 与 --size-axis 不能同时使用")
    if args.headless and args.instance is None:
        raise ValueError("headless 必须显式提供 --instance N")
    output = _output_path(args.output_dir, workspace=args.workspace)
    output.mkdir(parents=True)
    command_prefix = [sys.executable, "-m", "cordvip_geometry", "object"]
    workspace_prefix = f"cd {shlex.quote(str(args.workspace))} && "
    segment_command = [*command_prefix, "segment", "--output-dir", str(output / "segmentation")]
    for name in ("image", "rgbd", "text", "sam_python", "sam_root", "sam_checkpoint", "sam_config",
                 "dino_weights", "box_threshold", "text_threshold", "instance"):
        value = getattr(args, name)
        if value is not None:
            if isinstance(value, Path):
                value = value.expanduser().resolve()
            segment_command.extend(["--" + name.replace("_", "-"), str(value)])
    if args.headless:
        segment_command.append("--headless")
    generate_command = [*command_prefix, "generate", "--segmentation-dir", str(output / "segmentation"), "--output-dir", str(output / "generation")]
    for name in ("hunyuan_root", "weights", "seed", "steps", "octree_resolution", "num_chunks", "num_points", "device"):
        value = getattr(args, name)
        if isinstance(value, Path):
            value = value.expanduser().resolve()
        generate_command.extend(["--" + name.replace("_", "-"), str(value)])
    if args.preview:
        generate_command.append("--preview")
    for command in (segment_command, generate_command):
        print(workspace_prefix + shlex.join(command), flush=True)
        subprocess.run(command, check=True, env=module_environment(), cwd=args.workspace)
    mesh = output / "generation" / "mesh_unscaled.obj"
    prepare_command = [sys.executable, "-m", "cordvip_geometry", "prepare", "--mesh", str(mesh), "--output-dir", str(output / "asset"), "--num-points", str(args.num_points), "--seed", str(args.seed)]
    if args.scale_to_meters is not None:
        prepare_command.extend(["--scale-to-meters", str(args.scale_to_meters)])
    elif args.size_axis is not None:
        prepare_command.extend(["--size-axis", args.size_axis, "--size-meters", str(args.size_meters)])
    else:
        print("mesh 与表面点云已生成；尚未指定米制尺度，因此没有生成可用于追踪的资产。")
        print("检查 mesh 坐标轴后运行（替换 AXIS 与 MEASURED_METERS）：")
        print(workspace_prefix + shlex.join(prepare_command) + " --size-axis AXIS --size-meters MEASURED_METERS --preview")
        return
    if args.preview:
        prepare_command.append("--preview")
    subprocess.run(prepare_command, check=True, env=module_environment(), cwd=args.workspace)
    live = [sys.executable, "-m", "cordvip_geometry", "track", "--asset", str(output / "asset"), "--object-prompt", args.text]
    print("资产已准备。以下命令会连接 RealSense，确认相机可用后手动运行：", flush=True)
    print(workspace_prefix + shlex.join(live), flush=True)


def build_parser(*, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(workspace=workspace)
    commands = parser.add_subparsers(dest="command", required=True)
    segment = commands.add_parser("segment", help="Grounded-SAM2 分割并确认物体")
    _add_segment_arguments(segment, workspace=workspace)
    segment.add_argument("--list-only", action="store_true", help="只保存候选和预览图，后续用 select 选择；不创建窗口")
    segment.set_defaults(run=_segment)
    select = commands.add_parser("select", help="从保存的候选继续确认，不再加载模型")
    select.add_argument("--segmentation-dir", type=Path, required=True)
    select.add_argument("--instance", type=int)
    select.add_argument("--headless", action="store_true")
    select.set_defaults(run=_select)
    generate = commands.add_parser("generate", help="从已确认的 RGBA 生成未定尺度 mesh 与表面点云")
    generate.add_argument("--segmentation-dir", type=Path, required=True)
    _add_generate_arguments(generate, workspace=workspace)
    generate.set_defaults(run=_generate)
    build = commands.add_parser("build", help="按分割、生成、可选米制资产准备串行执行")
    _add_segment_arguments(build, workspace=workspace)
    _add_generate_arguments(build, workspace=workspace)
    build.add_argument("--scale-to-meters", type=_positive_float)
    build.add_argument("--size-axis", choices=("x", "y", "z"))
    build.add_argument("--size-meters", type=_positive_float)
    build.set_defaults(run=_build)
    preview = commands.add_parser("preview", help="单独查看 mesh，无需加载任何模型")
    preview.add_argument("--mesh", type=Path, required=True)
    preview.add_argument("--units", choices=("unscaled", "m"), default="unscaled", help="显示标签；不进行缩放")
    preview.set_defaults(run=lambda args: _preview_mesh(_existing_file(args.mesh), units=args.units))
    for command in (segment, select, generate, build):
        command.add_argument("--output-dir", type=Path, required=True, help="reports/ 下尚不存在的目录，拒绝覆盖")
    for command in (generate, build):
        command.add_argument("--device", default="cuda:0", help="Hunyuan 生成阶段使用的 CUDA 设备")
    return parser


def main(argv=None, *, workspace=None):
    args = build_parser(workspace=workspace).parse_args(argv)
    try:
        if hasattr(args, "seed") and args.seed < 0:
            raise ValueError("--seed 必须非负")
        for name in ("box_threshold", "text_threshold"):
            if hasattr(args, name) and not 0 < getattr(args, name) < 1:
                raise ValueError(f"{name} 必须在 (0, 1) 内")
        args.run(args)
    except (OSError, ValueError, ImportError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

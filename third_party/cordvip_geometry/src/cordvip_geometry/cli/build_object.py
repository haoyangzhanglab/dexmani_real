#!/usr/bin/env python3
"""单图分割、重建与显式定尺度；分割/定尺度直接调用，单次生成子进程退出释放 GPU。"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import shlex
import shutil
import sys
from pathlib import Path

from ..io import existing_file, output_path, write_json
from .arguments import parse_object_args
from .object_selection import choose, display_available, save_candidates, save_selection

logger = logging.getLogger(__name__)


def load_rgb(args):
    import numpy as np
    from PIL import Image, ImageOps

    path = existing_file(args.image if args.image is not None else args.rgbd)
    info = {"source_file": str(path)}
    if args.image is not None:
        with Image.open(path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
        info.update(input_type="image", exif_orientation="applied_to_saved_rgb")
    else:
        from ..rgbd import load_frame

        frame = load_frame(path)
        rgb = frame.rgb
        info.update(
            input_type="saved_rgbd",
            source_id=frame.source_id,
            timestamp_ns=frame.timestamp_ns,
            clock=frame.clock,
            depth_usage="preserved_only; no_automatic_scale_estimation",
            mask_grid="original_rgb_grid; not_automatically_rectified",
        )
    info["rgb_shape"] = list(rgb.shape)
    return rgb, path, info


def preview_mesh(mesh_path, *, units="unscaled", points=None, normals=None):
    display_available()
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
    o3d.visualization.draw_geometries(
        geometries, window_name=f"{title} ({units}); x=red y=green z=blue", mesh_show_back_face=True
    )


def segment(args, *, output_dir=None, list_only=False):
    if args.headless and args.instance is None and not list_only:
        raise ValueError("headless 须提供 --instance N；未知索引时先使用 --list-only")
    if not args.headless and not list_only:
        display_available()
    rgb, source, provenance = load_rgb(args)
    output = output_path(output_dir or args.output_dir, workspace=args.workspace)
    output.mkdir(parents=True)
    from PIL import Image

    Image.fromarray(rgb).save(output / "rgb.png")
    source_name = (
        "source_rgbd.npz" if args.rgbd is not None else f"source{source.suffix.lower()}"
    )
    shutil.copyfile(source, output / source_name)
    provenance.update(
        method="Grounded-SAM2 text segmentation",
        prompt=args.text,
        sam_root=str(args.sam_root.expanduser().resolve()),
        sam_python=str(args.sam_python.expanduser().absolute()),
        sam_checkpoint=str(args.sam_checkpoint.expanduser().resolve()),
        sam_config=args.sam_config,
        dino_weights=str(args.dino_weights.expanduser().resolve()),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        device="cuda:0",
        headless=args.headless,
        saved_source=source_name,
        candidate_order="descending_grounding_dino_score; zero_based",
    )
    write_json(output / "segmentation_request.json", provenance)
    from ..segmentation import GroundedSAM2Segmenter

    segmenter = GroundedSAM2Segmenter(
        args.sam_python,
        args.sam_root,
        args.sam_checkpoint,
        sam_config=args.sam_config,
        dino_weights=args.dino_weights,
        log_path=output / "sam2_worker.log",
    )
    try:
        segmenter.start()
        candidates = segmenter.predict(
            rgb,
            args.text,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
        provenance.update(model=segmenter.provenance)
    except BaseException:
        try:
            segmenter.close()
        except BaseException:
            logger.exception("SAM2 cleanup also failed")
        raise
    segmenter.close()
    if not candidates:
        write_json(output / "segmentation_result.json", {**provenance, "num_candidates": 0})
        raise RuntimeError(f"没有检测到目标；原图和请求已保留在 {output}，可调整文本或阈值")
    save_candidates(output, rgb, candidates)
    provenance.update(num_candidates=len(candidates))
    write_json(output / "segmentation_result.json", provenance)
    if list_only:
        print(f"候选已保存：{output / 'candidates.jpg'}；使用 select 确认，无需重跑模型。")
        return
    index = choose(rgb, candidates, args.instance, headless=args.headless)
    save_selection(output, rgb, candidates, index, provenance)


def select(args):
    import numpy as np
    from PIL import Image

    source = args.segmentation_dir.expanduser().resolve()
    output = output_path(
        args.output_dir, workspace=args.workspace, protected=(source,)
    )
    provenance = json.loads(existing_file(source / "segmentation_result.json").read_text())
    with Image.open(existing_file(source / "rgb.png")) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with np.load(existing_file(source / "candidates.npz"), allow_pickle=False) as saved:
        candidates = [
            dict(
                mask=mask,
                box_xyxy=saved["boxes"][i],
                score=float(saved["scores"][i]),
                mask_score=float(saved["mask_scores"][i]),
                label=str(saved["labels"][i]),
            )
            for i, mask in enumerate(saved["masks"])
        ]
    if not candidates or any(
        c["mask"].dtype != np.bool_ or c["mask"].shape != rgb.shape[:2] for c in candidates
    ):
        raise ValueError("保存的候选 mask 与 RGB 不一致")
    index = choose(rgb, candidates, args.instance, headless=args.headless)
    output.mkdir(parents=True)
    for name in (
        "rgb.png",
        "candidates.npz",
        "candidates.jpg",
        "segmentation_request.json",
        "segmentation_result.json",
        provenance["saved_source"],
    ):
        file = (source / name).resolve()
        if not file.is_relative_to(source):
            raise ValueError("候选元数据不能引用输入目录以外的文件")
        shutil.copyfile(existing_file(file), output / name)
    provenance.update(headless=args.headless, selection_source=str(source))
    save_selection(output, rgb, candidates, index, provenance)


def generate(
    segmentation_dir, output_dir, *, workspace, hunyuan_root, weights,
    seed=0, steps=50, octree_resolution=384, num_chunks=8000,
    device="cuda:0", preview=False, num_points=1024,
):
    if preview:
        display_available()
    source = Path(segmentation_dir).expanduser().resolve()
    metadata = json.loads(existing_file(source / "selection.json").read_text())
    if metadata["selection"] not in {"explicit_headless_instance", "gui_confirmed"}:
        raise ValueError("建模需要已明确确认的分割")
    image = existing_file(source / "foreground_rgba.png")
    output = output_path(output_dir, workspace=workspace, protected=(source,))
    output.mkdir(parents=True)
    write_json(output / "generation_request.json", dict(
        method="Hunyuan3D-2.1 shape_only",
        segmentation_directory=str(source), selected_instance=metadata["selected_instance"],
        source_root=str(Path(hunyuan_root).expanduser().resolve()),
        weights_dir=str(Path(weights).expanduser().resolve()),
        seed=seed, steps=steps, octree_resolution=octree_resolution,
        num_chunks=num_chunks, device=device, units="unscaled",
    ))
    from ..reconstruction import generate_mesh

    mesh_path = output / "mesh_unscaled.obj"
    result = generate_mesh(
        image_rgba_path=image, source_root=hunyuan_root, weights_dir=weights,
        output_mesh_path=mesh_path, seed=seed, steps=steps,
        octree_resolution=octree_resolution, num_chunks=num_chunks, device=device,
    )
    existing_file(mesh_path)
    write_json(output / "generation_result.json", {
        **result, "mesh_file": mesh_path.name, "units": "unscaled", "frame": "object",
        "origin": "source_mesh_origin_without_recentering",
    })
    print(f"未定尺度 mesh：{mesh_path}；准备追踪资产前必须提供实测尺寸或明确缩放系数。", flush=True)
    if preview:
        from ..geometry import load_metric_mesh, sample_mesh_surface

        # 预览点只在内存中生成；最终资产在米制尺度上按同一 seed 采样。
        points, normals = sample_mesh_surface(load_metric_mesh(mesh_path), num_points, seed)
        preview_mesh(mesh_path, points=points, normals=normals)


def generation_parameters(args, segmentation_dir, output_dir):
    parameters = dict(
        segmentation_dir=segmentation_dir, output_dir=output_dir,
        workspace=args.workspace, hunyuan_root=args.hunyuan_root, weights=args.weights,
        seed=args.seed, steps=args.steps, octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks, device=args.device,
        preview=args.preview,
    )
    if args.preview:
        parameters["num_points"] = args.num_points
    return parameters


def build(args):
    if args.headless and args.preview:
        raise ValueError("--headless 与 --preview 不能同时使用")
    if (args.size_axis is None) != (args.size_meters is None):
        raise ValueError("--size-axis 与 --size-meters 必须一起提供")
    if args.scale_to_meters is not None and args.size_axis is not None:
        raise ValueError("--scale-to-meters 与 --size-axis 不能同时使用")
    if args.headless and args.instance is None:
        raise ValueError("headless 必须显式提供 --instance N")
    output = output_path(args.output_dir, workspace=args.workspace)
    output.mkdir(parents=True)
    segment(args, output_dir=output / "segmentation")
    # SAM RPC 已在 segment 返回前关闭；只有生成阶段另开进程释放 CUDA。
    process = mp.get_context("spawn").Process(
        target=generate,
        kwargs=generation_parameters(args, output / "segmentation", output / "generation"),
        name="object-generation",
    )
    try:
        process.start()
        process.join()
    except BaseException:
        try:
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
        except BaseException:
            logger.exception("generation process cleanup also failed")
        raise
    if process.exitcode != 0:
        raise RuntimeError(f"mesh generation failed: child exitcode={process.exitcode}")
    mesh = output / "generation" / "mesh_unscaled.obj"
    if args.scale_to_meters is None and args.size_axis is None:
        print("未定尺度 mesh 已生成；尚未指定米制尺度，因此没有生成追踪资产。")
        print("检查 mesh 坐标轴后运行（替换 AXIS 与 MEASURED_METERS）：")
        print(shlex.join([
            sys.executable, "-m", "cordvip_geometry", "prepare",
            "--mesh", str(mesh), "--output-dir", str(output / "asset"),
            "--num-points", str(args.num_points), "--seed", str(args.seed),
            "--size-axis", "AXIS", "--size-meters", "MEASURED_METERS", "--preview",
        ]))
        return
    from .prepare_object import prepare_mesh

    prepare_mesh(
        mesh, output / "asset", workspace=args.workspace,
        scale_to_meters=args.scale_to_meters, size_axis=args.size_axis,
        size_meters=args.size_meters, num_points=args.num_points,
        seed=args.seed, preview=args.preview,
    )
    print(f"米制资产已准备：{output / 'asset'}；可在 Raw 离线追踪中使用。", flush=True)


def main(argv=None, *, workspace=None):
    args = parse_object_args(argv, workspace=workspace)
    if args.command == "segment":
        segment(args, list_only=args.list_only)
    elif args.command == "select":
        select(args)
    elif args.command == "generate":
        generate(**generation_parameters(args, args.segmentation_dir, args.output_dir))
    elif args.command == "build":
        build(args)
    else:
        preview_mesh(existing_file(args.mesh), units=args.units)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

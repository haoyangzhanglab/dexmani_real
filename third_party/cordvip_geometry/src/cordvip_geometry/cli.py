"""物体分割、重建与尺度调整命令；解析不加载模型。"""

import argparse
import math
from pathlib import Path

from .weights import DEFAULT_MODELS_DIR, SAM_MODELS, resolve_sam_model


def probability(value):
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("必须在 (0, 1) 内")
    return number


def add_models_directory(parser, workspace):
    parser.add_argument(
        "--models-dir", type=Path, default=workspace / DEFAULT_MODELS_DIR
    )


def add_segment_arguments(parser, workspace):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="按 EXIF 纠正朝向的物体照片")
    source.add_argument(
        "--rgbd", type=Path, help="含 rgb uint8 [H,W,3] 的 NPZ（仅读取 rgb）"
    )
    parser.add_argument("--prompt", required=True, help="英文物体描述")
    parser.add_argument(
        "--instance", type=int, help="去重后按置信度降序排列的零起始候选索引"
    )
    parser.add_argument("--headless", action="store_true")
    add_models_directory(parser, workspace)
    parser.add_argument(
        "--sam-variant", choices=SAM_MODELS, help="官方 SAM2.1 型号，默认 small"
    )
    parser.add_argument(
        "--sam-checkpoint", type=Path, help="自训权重，须与 --sam-config 配对"
    )
    parser.add_argument("--sam-config", help="自训模型在 SAM2 包内的配置")
    parser.add_argument("--box-threshold", type=probability, default=0.3)
    parser.add_argument("--text-threshold", type=probability, default=0.25)


def add_generate_arguments(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--preview", action="store_true")


def add_scale_arguments(parser, *, required=False):
    scale = parser.add_mutually_exclusive_group(required=required)
    scale.add_argument("--scale-to-meters", type=float)
    scale.add_argument("--size-axis", choices=("x", "y", "z"))
    parser.add_argument("--size-meters", type=float)


def parse_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description="物体分割、重建和米制资产准备。")
    parser.set_defaults(workspace=workspace)
    commands = parser.add_subparsers(dest="command", required=True)

    segment = commands.add_parser("segment", help="分割并确认物体")
    add_segment_arguments(segment, workspace)
    segment.add_argument(
        "--list-only", action="store_true", help="保存候选，稍后 select"
    )

    select = commands.add_parser(
        "select",
        help="从已有候选确认物体",
        description=(
            "从 rgb.png 和 candidates.npz 选择物体，"
            "保存 mask.png、foreground_rgba.png 和 overlay.png。"
        ),
    )
    select.add_argument(
        "--segmentation-dir",
        type=Path,
        required=True,
        help="含 rgb.png 和 candidates.npz 的候选目录",
    )
    select.add_argument("--instance", type=int, help="已保存候选的零起始索引")
    select.add_argument("--headless", action="store_true")

    generate = commands.add_parser("generate", help="从 RGBA 生成未定尺度网格")
    generate.add_argument(
        "--image", type=Path, required=True, help="透明背景 RGBA 文件"
    )
    add_models_directory(generate, workspace)
    add_generate_arguments(generate)

    reconstruct = commands.add_parser(
        "reconstruct", help="分割、生成，并按显式尺度准备资产"
    )
    add_segment_arguments(reconstruct, workspace)
    add_generate_arguments(reconstruct)
    add_scale_arguments(reconstruct)

    scale = commands.add_parser("scale", help="将已有网格缩放为米制网格")
    scale.add_argument("--mesh", type=Path, required=True)
    scale.add_argument("--preview", action="store_true")
    add_scale_arguments(scale, required=True)

    preview = commands.add_parser("preview", help="预览网格")
    preview.add_argument("--mesh", type=Path, required=True)
    preview.add_argument("--units", choices=("unscaled", "m"), default="unscaled")
    for command in (segment, select, generate, reconstruct, scale):
        command.add_argument("--output", type=Path, required=True, help="输出目录")

    download = commands.add_parser("download", help="下载固定版本的模型权重")
    add_models_directory(download, workspace)
    download.add_argument(
        "--models",
        nargs="+",
        choices=("grounding", "sam", "shape"),
        default=["grounding", "sam"],
    )
    download.add_argument("--sam-variant", choices=SAM_MODELS, default="small")

    args = parser.parse_args(argv)
    if args.command == "reconstruct" and args.headless and args.preview:
        parser.error("--headless 与 --preview 不能同时使用")
    if hasattr(args, "models_dir"):
        args.models_dir = args.models_dir.expanduser().resolve()
    if args.command in {"scale", "reconstruct"}:
        scale, axis, size = args.scale_to_meters, args.size_axis, args.size_meters
        if (axis is None) != (size is None):
            parser.error("--size-axis 与 --size-meters 必须同时提供")
        if any(
            value is not None and (not math.isfinite(value) or value <= 0)
            for value in (scale, size)
        ):
            parser.error("尺度必须为有限正数")
    if args.command in {"segment", "reconstruct"}:
        try:
            args.sam_checkpoint, args.sam_config = resolve_sam_model(
                args.models_dir, args.sam_variant, args.sam_checkpoint, args.sam_config
            )
        except ValueError as error:
            parser.error(str(error))
    return args


def main(argv=None):
    args = parse_args(argv)
    command = args.command
    if command == "download":
        from .weights import download_models

        download_models(args.models, args.models_dir, args.sam_variant)
        return 0
    from .mesh import load_mesh, scale_mesh
    from .visualization import choose, preview_mesh, save_candidates, save_selection

    if command == "preview":
        preview_mesh(args.mesh, units=args.units)
        return 0
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if command == "scale":
        mesh = scale_mesh(
            load_mesh(args.mesh),
            scale_to_meters=args.scale_to_meters,
            size_axis=args.size_axis,
            size_meters=args.size_meters,
        )
        mesh.export(output / "mesh_m.obj")
        if args.preview:
            preview_mesh(output / "mesh_m.obj", units="m")
        return 0
    import numpy as np
    from PIL import Image, ImageOps

    if command in {"segment", "reconstruct"}:
        from .segmentation import SAM2Segmenter

        if args.image is not None:
            with Image.open(args.image) as image:
                rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
        else:
            with np.load(args.rgbd, allow_pickle=False) as saved:
                rgb = saved["rgb"]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError("RGB-D NPZ 的 rgb 必须为 uint8 [H,W,3]")
        segmenter = SAM2Segmenter(
            args.workspace / "third_party/Grounded-SAM-2",
            args.sam_checkpoint,
            args.sam_config,
            args.models_dir / "grounding-dino-base",
        )
        candidates = segmenter.predict(
            rgb, args.prompt, args.box_threshold, args.text_threshold
        )
        if not candidates:
            raise ValueError("没有检测到目标，请调整 prompt")
        if command == "segment" and args.list_only:
            Image.fromarray(rgb).save(output / "rgb.png")
            save_candidates(output, rgb, candidates)
            return 0
        index = choose(rgb, candidates, args.instance, headless=args.headless)
        save_selection(output, rgb, candidates, index)
        del segmenter
        if command == "segment":
            return 0
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    elif command == "select":
        with Image.open(args.segmentation_dir / "rgb.png") as image:
            rgb = np.asarray(image.convert("RGB")).copy()
        with np.load(
            args.segmentation_dir / "candidates.npz", allow_pickle=False
        ) as saved:
            candidates = [
                {
                    "mask": mask,
                    "box_xyxy": saved["boxes"][i],
                    "score": float(saved["scores"][i]),
                    "label": str(saved["labels"][i]),
                }
                for i, mask in enumerate(saved["masks"])
            ]
        index = choose(rgb, candidates, args.instance, headless=args.headless)
        save_selection(output, rgb, candidates, index)
        return 0
    from .reconstruction import load_shape_model, reconstruct_mesh

    model = load_shape_model(
        args.workspace / "third_party/Hunyuan3D-2.1",
        args.models_dir / "Hunyuan3D-2.1",
    )
    image_path = (
        output / "foreground_rgba.png" if command == "reconstruct" else args.image
    )
    with Image.open(image_path) as image:
        mesh = reconstruct_mesh(
            model,
            image,
            seed=args.seed,
            steps=args.steps,
            octree_resolution=args.octree_resolution,
            num_chunks=args.num_chunks,
        )
    mesh.export(output / "mesh_unscaled.obj")
    if command == "generate" or (
        args.scale_to_meters is None and args.size_axis is None
    ):
        if args.preview:
            preview_mesh(output / "mesh_unscaled.obj")
        return 0
    mesh = scale_mesh(
        mesh,
        scale_to_meters=args.scale_to_meters,
        size_axis=args.size_axis,
        size_meters=args.size_meters,
    )
    mesh.export(output / "mesh_m.obj")
    if args.preview:
        preview_mesh(output / "mesh_m.obj", units="m")
    return 0

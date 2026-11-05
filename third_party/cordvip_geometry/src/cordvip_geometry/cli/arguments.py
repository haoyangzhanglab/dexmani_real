"""命令行参数与输入约束；解析不加载感知模型。"""

import argparse
import sys
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def probability(value):
    number = float(value)
    if not 0 < number < 1:
        raise argparse.ArgumentTypeError("必须在 (0, 1) 内")
    return number


def add_models_directory(parser, workspace):
    from ..models import DEFAULT_MODELS_DIR

    parser.add_argument(
        "--models-dir", type=Path, default=workspace / DEFAULT_MODELS_DIR
    )


def add_sam_arguments(parser, workspace):
    from ..models import SAM_MODELS

    add_models_directory(parser, workspace)
    parser.add_argument("--sam-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--sam-variant", choices=SAM_MODELS, help="官方 SAM2.1 型号，默认 small"
    )
    parser.add_argument(
        "--sam-checkpoint", type=Path, help="自训权重，须与 --sam-config 配对"
    )
    parser.add_argument("--sam-config", help="自训模型在 SAM2 包内的配置")


def resolve_sam_arguments(args):
    from ..models import resolve_sam_model

    args.models_dir = args.models_dir.expanduser().resolve()
    args.sam_checkpoint, args.sam_config = resolve_sam_model(
        args.models_dir, args.sam_variant, args.sam_checkpoint, args.sam_config
    )
    return args


def add_grounding_arguments(parser):
    parser.add_argument("--box-threshold", type=probability, default=0.3)
    parser.add_argument("--text-threshold", type=probability, default=0.25)


def parse_tracking_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description="确认 Raw 首帧实例，逐行保存位姿和几何。"
    )
    parser.set_defaults(workspace=workspace)
    parser.add_argument(
        "--episode", type=Path, required=True, help="只读 Raw episode 目录"
    )
    parser.add_argument(
        "--scene", type=Path, help="多物体 JSON；对象 id 使用唯一、非空的合法单级目录名"
    )
    parser.add_argument(
        "--asset", type=Path, help="米制物体资产；省略时只处理 mask 和实测点"
    )
    parser.add_argument(
        "--prompt", help="首帧目标文本；未提供 --initial-selection 时必需"
    )
    parser.add_argument(
        "--initial-selection", type=Path,
        help=(
            "当前首帧原始 RGB 网格的 1/L/P PNG（P 按调色板转灰度，非零为前景）、"
            "bool/uint8 单通道 NPY，或含 selection.json 的目录"
        ),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--with-hand", action="store_true", help="附加同行实测关节的 FK 手点云"
    )
    parser.add_argument("--hand-points", type=int, default=1024)
    parser.add_argument("--pixel-stride", type=int, default=2)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.015)
    parser.add_argument("--min-matched-pixels", type=int, default=30)
    parser.add_argument("--min-matched-fraction", type=float, default=0.15)
    parser.add_argument("--max-frames", type=positive_int, help="显式限制处理前缀")
    parser.add_argument("--output", type=Path, help="尚不存在的新输出目录")
    add_sam_arguments(parser, workspace)
    add_grounding_arguments(parser)
    args = parser.parse_args(argv)
    for name in ("episode", "scene", "asset", "initial_selection"):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, (workspace / value.expanduser()).resolve())
    if args.scene and (args.asset or args.initial_selection or args.prompt):
        parser.error("--scene 不能混用单物体参数")
    if args.headless and args.initial_selection is None and args.scene is None:
        parser.error("--headless 需要 --initial-selection 或 --scene")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt 不能为空")
    if args.scene is None and args.initial_selection is None and args.prompt is None:
        parser.error("必须提供 --prompt、--initial-selection 或 --scene")
    try:
        return resolve_sam_arguments(args)
    except ValueError as error:
        parser.error(str(error))


def add_segment_arguments(parser, workspace):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="按 EXIF 纠正朝向的物体照片")
    source.add_argument("--rgbd", type=Path, help="保存的单帧 RGB-D NPZ")
    parser.add_argument("--prompt", required=True, help="英文物体描述")
    parser.add_argument("--instance", type=int, help="零起始候选索引")
    parser.add_argument("--headless", action="store_true")
    add_sam_arguments(parser, workspace)
    add_grounding_arguments(parser)


def add_generate_arguments(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--preview", action="store_true")


def add_scale_arguments(parser):
    parser.add_argument("--scale-to-meters", type=float)
    parser.add_argument("--size-axis", choices=("x", "y", "z"))
    parser.add_argument("--size-meters", type=float)
    parser.add_argument("--num-points", type=int, default=1024)


def parse_object_args(argv=None, *, workspace=None):
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
            "从 rgb.png 和 candidates.npz 选择物体，写出 mask.png、foreground_rgba.png、"
            "overlay.png 和 selection.json。"
            "再次选择仍使用原候选目录。"
        ),
    )
    select.add_argument(
        "--segmentation-dir",
        type=Path,
        required=True,
        help="含 rgb.png 和 candidates.npz 的候选目录；不要求分割日志或预览图",
    )
    select.add_argument("--instance", type=int)
    select.add_argument("--headless", action="store_true")
    generate = commands.add_parser("generate", help="从 RGBA 生成未定尺度 mesh")
    generate.add_argument("--image", type=Path, required=True, help="透明背景 RGBA 文件")
    add_models_directory(generate, workspace)
    add_generate_arguments(generate)
    build = commands.add_parser("build", help="分割、生成，并按显式尺度准备资产")
    add_segment_arguments(build, workspace)
    add_generate_arguments(build)
    add_scale_arguments(build)
    prepare = commands.add_parser("prepare", help="将已有 mesh 校准为米制资产")
    prepare.add_argument("--mesh", type=Path, required=True)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--preview", action="store_true")
    add_scale_arguments(prepare)
    preview = commands.add_parser("preview", help="查看 mesh，无需加载模型")
    preview.add_argument("--mesh", type=Path, required=True)
    preview.add_argument("--units", choices=("unscaled", "m"), default="unscaled")
    for command in (segment, select, generate, build, prepare):
        command.add_argument(
            "--output", type=Path, required=True, help="尚不存在的新输出目录"
        )
    args = parser.parse_args(argv)
    if args.command == "build" and args.headless and args.preview:
        parser.error("--headless 与 --preview 不能同时使用")
    if hasattr(args, "sam_variant"):
        try:
            resolve_sam_arguments(args)
        except ValueError as error:
            parser.error(str(error))
    elif hasattr(args, "models_dir"):
        args.models_dir = args.models_dir.expanduser().resolve()
    return args

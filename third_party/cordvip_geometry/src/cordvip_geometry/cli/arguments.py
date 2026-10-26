"""追踪和物体构建的命令行参数、默认值与输入校验；不加载模型或设备。"""

import argparse
import math
from pathlib import Path


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def parse_tracking_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(description="单物体与多物体 RGB-D 追踪；默认连接 RealSense，离线源只读。")
    parser.set_defaults(workspace=workspace)
    parser.add_argument(
        "--asset", type=Path, help="reconstruct_object.py 或 prepare 生成的米制物体资产目录"
    )
    parser.add_argument(
        "--scene",
        type=Path,
        help="多物体 JSON：objects 中逐个声明 id/role/asset/initial_selection/prompt",
    )
    parser.add_argument(
        "--all-frames", action="store_true", help="离线模式逐帧处理，不按回放速率丢帧或等待"
    )
    parser.add_argument("--max-frames", type=int, help="最多处理的帧数；用于显式的部分验收")
    parser.add_argument(
        "--foundationpose-root", type=Path, default=workspace / "third_party/FoundationPose"
    )
    parser.add_argument(
        "--object-prompt", help='Grounded-SAM2 文本目标，例如 "red mug."；省略时手动圈选'
    )
    from ..sam2_config import add_sam_arguments

    add_sam_arguments(parser, workspace)
    parser.add_argument(
        "--check", action="store_true", help="只检查环境，不加载模型、创建窗口或连接相机"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--episode", type=Path, help="用只读 Raw 替代相机，按 replay-fps 演示异步流程"
    )
    source.add_argument("--rgbd-dir", type=Path, help="用 RGB-D NPZ 替代相机")
    parser.add_argument("--replay-fps", type=float, default=30)
    parser.add_argument("--serial", help="RealSense serial；省略时由 SDK 选择相机")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--no-3d", action="store_true", help="仅显示双 RGB 窗口；仍生成两类物体点云"
    )
    parser.add_argument(
        "--initial-selection", type=Path, help="复用已确认的 select 目录；必须与离线源首帧完全对应"
    )
    parser.add_argument(
        "--headless", action="store_true", help="无窗口离线回放；需要 --initial-selection"
    )
    parser.add_argument(
        "--with-hand", action="store_true", help="读取同一 Raw 行的实测关节，附加 XHand FK 点云"
    )
    parser.add_argument("--experiment-config", type=Path, help="手安装变换使用的当前实验配置 YAML")
    parser.add_argument("--hand-points", type=int, default=1024, help="固定手表面采样点数")
    parser.add_argument("--output", type=Path, help="尚不存在的新输出目录")
    parser.add_argument(
        "--save-inputs", action="store_true", help="保存所有送入推理的源 RGB-D；注册帧总是保存"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="逐次保存实际处理结果到 frames/；不补齐跳过的源帧",
    )
    parser.add_argument(
        "--save-video", action="store_true", help="保存处理帧叠加视频；播放率为 replay-fps"
    )
    parser.add_argument("--max-age-ms", type=float, default=500, help="源帧最大有效年龄，毫秒")
    parser.add_argument("--pixel-stride", type=int, default=2, help="物体实测点云的像素采样步长")
    parser.add_argument("--depth-tolerance-m", type=float, default=0.015)
    parser.add_argument("--min-matched-pixels", type=int, default=30)
    parser.add_argument("--min-matched-fraction", type=float, default=0.15)
    args = parser.parse_args(argv)
    if not args.check and args.asset is None and args.scene is None:
        parser.error("追踪需要 --asset 或 --scene；只检查环境可用 --check")
    if args.scene and (args.asset or args.initial_selection or args.object_prompt):
        parser.error("--scene 内逐个指定资产、选择和 prompt，不能混用单物体参数")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames 必须为正整数")
    if args.serial and (args.episode or args.rgbd_dir):
        parser.error("--serial 只用于实时相机")
    offline = args.episode is not None or args.rgbd_dir is not None
    if args.all_frames and not offline:
        parser.error("--all-frames 仅用于离线源")
    if args.initial_selection is not None and not offline:
        parser.error("--initial-selection 仅用于离线源，不能将历史 mask 用于新的相机帧")
    if args.headless and (not offline or (args.initial_selection is None and args.scene is None)):
        parser.error("--headless 需要离线源和 --initial-selection")
    if args.with_hand and args.episode is None:
        parser.error("--with-hand 需要 --episode，以同一 Raw 行的实测关节生成手点")
    if args.experiment_config is not None and not args.with_hand:
        parser.error("--experiment-config 需要 --with-hand")
    if args.object_prompt is not None and not args.object_prompt.strip():
        parser.error("--object-prompt 不能为空")
    if not offline and min(args.width, args.height, args.fps) < 1:
        parser.error("实时相机 width、height、fps 必须为正整数")
    if args.pixel_stride < 1:
        parser.error("--pixel-stride 必须为正整数")
    if args.with_hand and args.hand_points < 1:
        parser.error("--hand-points 必须为正整数")
    if not math.isfinite(args.max_age_ms) or args.max_age_ms <= 0:
        parser.error("--max-age-ms 必须为有限正数")
    if (args.save_video or (offline and not args.all_frames)) and (
        not math.isfinite(args.replay_fps) or args.replay_fps <= 0
    ):
        parser.error("--replay-fps 必须为有限正数")
    return args


def add_segment_arguments(parser, *, workspace):
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path, help="物体照片；按 EXIF 纠正朝向")
    source.add_argument("--rgbd", type=Path, help="单帧已保存的规范 RGB-D NPZ；不连接相机")
    parser.add_argument("--text", required=True, help="英文物体描述，例如 'red mug.'")
    from ..sam2_config import add_sam_arguments

    add_sam_arguments(parser, workspace)
    parser.add_argument("--instance", type=int, help="零起始候选索引；headless 必须显式选择")
    parser.add_argument("--headless", action="store_true", help="不创建窗口；必须提供 --instance")


def add_generate_arguments(parser, *, workspace):
    models = workspace / "reports" / "cordvip_models"
    source_root = workspace / "third_party"
    parser.add_argument("--hunyuan-root", type=Path, default=source_root / "Hunyuan3D-2.1")
    parser.add_argument("--weights", type=Path, default=models / "Hunyuan3D-2.1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=positive_int, default=50)
    parser.add_argument("--octree-resolution", type=positive_int, default=384)
    parser.add_argument("--num-chunks", type=positive_int, default=8000)
    parser.add_argument(
        "--num-points",
        type=positive_int,
        default=1024,
        help="固定表面点数；点云与 mesh 使用相同坐标系和单位",
    )
    parser.add_argument(
        "--preview", action="store_true", help="生成后交互查看 mesh、表面点云和坐标轴"
    )


def parse_object_args(argv=None, *, workspace=None):
    workspace = Path(workspace or Path.cwd()).expanduser().resolve()
    parser = argparse.ArgumentParser(
        description="单图分割、重建与显式定尺度；build 串行运行子进程，阶段结束释放 GPU。"
    )
    parser.set_defaults(workspace=workspace)
    commands = parser.add_subparsers(dest="command", required=True)
    segment = commands.add_parser("segment", help="Grounded-SAM2 分割并确认物体")
    add_segment_arguments(segment, workspace=workspace)
    segment.add_argument(
        "--list-only",
        action="store_true",
        help="只保存候选和预览图，后续用 select 选择；不创建窗口",
    )
    select = commands.add_parser("select", help="从保存的候选继续确认，不再加载模型")
    select.add_argument("--segmentation-dir", type=Path, required=True)
    select.add_argument("--instance", type=int)
    select.add_argument("--headless", action="store_true")
    generate = commands.add_parser("generate", help="从已确认的 RGBA 生成未定尺度 mesh 与表面点云")
    generate.add_argument("--segmentation-dir", type=Path, required=True)
    add_generate_arguments(generate, workspace=workspace)
    build = commands.add_parser("build", help="按分割、生成、可选米制资产准备串行执行")
    add_segment_arguments(build, workspace=workspace)
    add_generate_arguments(build, workspace=workspace)
    build.add_argument("--scale-to-meters", type=positive_float)
    build.add_argument("--size-axis", choices=("x", "y", "z"))
    build.add_argument("--size-meters", type=positive_float)
    preview = commands.add_parser("preview", help="单独查看 mesh，无需加载任何模型")
    preview.add_argument("--mesh", type=Path, required=True)
    preview.add_argument(
        "--units", choices=("unscaled", "m"), default="unscaled", help="显示标签；不进行缩放"
    )
    for command in (segment, select, generate, build):
        command.add_argument(
            "--output-dir", type=Path, required=True, help="尚不存在的新输出目录，拒绝覆盖"
        )
    for command in (generate, build):
        command.add_argument("--device", default="cuda:0", help="Hunyuan 生成阶段使用的 CUDA 设备")
    args = parser.parse_args(argv)
    if hasattr(args, "seed") and args.seed < 0:
        raise ValueError("--seed 必须非负")
    for name in ("box_threshold", "text_threshold"):
        if hasattr(args, name) and not 0 < getattr(args, name) < 1:
            raise ValueError(f"{name} 必须在 (0, 1) 内")
    return args

"""单图分割、重建与米制资产准备。"""

import logging
import multiprocessing as mp
import shlex
import sys
import time

from ..io import existing_file, output_path
from .arguments import parse_object_args
from .selection import choose, display_available, save_candidates, save_selection

logger = logging.getLogger(__name__)


def load_rgb(args):
    import numpy as np
    from PIL import Image, ImageOps

    path = existing_file(args.image if args.image is not None else args.rgbd)
    if args.image is not None:
        with Image.open(path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
    else:
        from ..rgbd import load_frame

        frame = load_frame(path)
        rgb = frame.rgb
    return rgb, path


def segment(args, *, output=None, list_only=False):
    from PIL import Image

    from ..sam2 import SAM2Client

    if not args.headless and not list_only:
        display_available()
    rgb, source = load_rgb(args)
    output = output_path(output or args.output, protected=(source,))
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(rgb).save(output / "rgb.png")
    print(f"Segmentation: prompt={args.prompt!r}, box_threshold={args.box_threshold}, text_threshold={args.text_threshold}", flush=True)
    started = time.monotonic()
    with SAM2Client(
        args.sam_python,
        args.workspace,
        args.models_dir,
        checkpoint=args.sam_checkpoint,
        config=args.sam_config,
        log_path=output / "sam2_worker.log",
    ) as segmenter:
        candidates = segmenter.predict(
            rgb,
            args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )
    print(f"Candidates={len(candidates)}, elapsed={time.monotonic() - started:.2f}s", flush=True)
    if not candidates:
        raise RuntimeError(
            f"没有检测到目标；rgb.png 已保留在 {output}，可调整文本或阈值"
        )
    save_candidates(output, rgb, candidates)
    if list_only:
        print(f"候选已保存：{output / 'candidates.jpg'}；使用 object select 确认。")
        return
    index = choose(rgb, candidates, args.instance, headless=args.headless)
    save_selection(output, rgb, candidates, index)


def select(args):
    import numpy as np
    from PIL import Image

    source = args.segmentation_dir.expanduser().resolve()
    output = output_path(args.output, protected=(source,))
    with Image.open(existing_file(source / "rgb.png")) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with np.load(existing_file(source / "candidates.npz"), allow_pickle=False) as saved:
        masks = saved["masks"]
        boxes = saved["boxes"]
        scores = saved["scores"]
        mask_scores = saved["mask_scores"]
        labels = saved["labels"]
    if (
        masks.dtype != np.bool_
        or masks.ndim != 3
        or masks.shape[1:] != rgb.shape[:2]
        or len(masks) == 0
    ):
        raise ValueError("保存的候选 mask 与 RGB 不一致")
    count = len(masks)
    if boxes.shape != (count, 4) or any(
        array.shape != (count,) for array in (scores, mask_scores, labels)
    ):
        raise ValueError("候选框、分数和标签必须与 mask 数量及数组形状一致")
    if labels.dtype.kind != "U" or any(
        array.dtype.kind not in "fiu" or not np.isfinite(array).all()
        for array in (boxes, scores, mask_scores)
    ):
        raise ValueError("候选标签必须为字符串，框和分数必须为有限实数")
    candidates = [
        {
            "mask": masks[i],
            "box_xyxy": boxes[i],
            "score": float(scores[i]),
            "mask_score": float(mask_scores[i]),
            "label": str(labels[i]),
        }
        for i in range(count)
    ]
    index = choose(rgb, candidates, args.instance, headless=args.headless)
    output.mkdir(parents=True, exist_ok=False)
    save_selection(output, rgb, candidates, index)


def generate(args, image=None, output=None):
    from ..reconstruction import generate_mesh, validate_generation_parameters

    validate_generation_parameters(
        args.seed, args.steps, args.octree_resolution, args.num_chunks
    )
    if args.preview:
        display_available()
    image = existing_file(image or args.image)
    output = output_path(output or args.output, protected=(image,))
    output.mkdir(parents=True, exist_ok=False)
    root = args.workspace / "third_party/Hunyuan3D-2.1"
    weights = args.models_dir / "Hunyuan3D-2.1"
    print(f"Generation: seed={args.seed}, steps={args.steps}, octree_resolution={args.octree_resolution}, num_chunks={args.num_chunks}", flush=True)
    mesh_path = output / "mesh_unscaled.obj"
    mesh_path = generate_mesh(
        image_rgba_path=image,
        source_root=root,
        weights_dir=weights,
        output_mesh_path=mesh_path,
        seed=args.seed,
        steps=args.steps,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
    )
    print(
        f"未定尺度 mesh：{mesh_path}；追踪前须提供实测尺寸或明确缩放系数。", flush=True
    )
    if args.preview:
        from ..geometry import load_mesh, sample_mesh_surface
        from ..viewer import preview_mesh

        points, normals = sample_mesh_surface(load_mesh(mesh_path), 1024, args.seed)
        preview_mesh(mesh_path, points=points, normals=normals)


def prepare(args, mesh=None, output=None):
    from ..assets import prepare_asset

    asset = prepare_asset(
        mesh or args.mesh,
        output or args.output,
        scale_to_meters=args.scale_to_meters,
        size_axis=args.size_axis,
        size_meters=args.size_meters,
        num_points=args.num_points,
        seed=args.seed,
    )
    print(f"米制资产已准备：{asset}", flush=True)
    if args.preview:
        import numpy as np

        from ..viewer import preview_mesh

        preview_mesh(
            asset / "mesh/mesh_m.obj",
            units="m",
            points=np.load(asset / "points.npy", allow_pickle=False),
            normals=np.load(asset / "normals.npy", allow_pickle=False),
        )


def build(args):
    from ..assets import validate_asset_parameters
    from ..reconstruction import validate_generation_parameters

    validate_asset_parameters(
        args.scale_to_meters,
        args.size_axis,
        args.size_meters,
        args.num_points,
        args.seed,
        allow_unscaled=True,
    )
    validate_generation_parameters(
        args.seed, args.steps, args.octree_resolution, args.num_chunks
    )
    source = args.image if args.image is not None else args.rgbd
    output = output_path(args.output, protected=(source,))
    output.mkdir(parents=True, exist_ok=False)
    segment(args, output=output / "segmentation")
    # 生成进程退出后释放 CUDA，再准备米制资产。
    process = mp.get_context("spawn").Process(
        target=generate,
        args=(
            args,
            output / "segmentation/foreground_rgba.png",
            output / "generation",
        ),
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
    mesh = output / "generation/mesh_unscaled.obj"
    if args.scale_to_meters is None and args.size_axis is None:
        print("未指定米制尺度，保留未定尺度 mesh；检查坐标轴后运行：")
        print(
            shlex.join(
                [
                    sys.executable,
                    "-m",
                    "cordvip_geometry",
                    "object",
                    "prepare",
                    "--mesh",
                    str(mesh),
                    "--output",
                    str(output / "asset"),
                    "--num-points",
                    str(args.num_points),
                    "--seed",
                    str(args.seed),
                    "--size-axis",
                    "AXIS",
                    "--size-meters",
                    "MEASURED_METERS",
                    "--preview",
                ]
            )
        )
        return
    prepare(args, mesh=mesh, output=output / "asset")


def main(argv=None, *, workspace=None):
    args = parse_object_args(argv, workspace=workspace)
    if args.command == "segment":
        segment(args, list_only=args.list_only)
    elif args.command == "select":
        select(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "build":
        build(args)
    elif args.command == "prepare":
        prepare(args)
    else:
        from ..viewer import preview_mesh

        preview_mesh(existing_file(args.mesh), units=args.units)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

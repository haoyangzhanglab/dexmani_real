"""单图分割、重建与米制资产准备。"""

import json
import logging
import multiprocessing as mp
import shlex
import shutil
import sys
from pathlib import Path

from ..io import existing_file, output_path, write_json
from .arguments import parse_object_args
from .selection import choose, display_available, save_candidates, save_selection

logger = logging.getLogger(__name__)


def load_rgb(args):
    import numpy as np
    from PIL import Image, ImageOps

    path = existing_file(args.image if args.image is not None else args.rgbd)
    info = {"source_file": str(path)}
    if args.image is not None:
        with Image.open(path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
        info["input_type"] = "image"
    else:
        from ..rgbd import load_frame

        frame = load_frame(path)
        rgb = frame.rgb
        info.update(
            input_type="saved_rgbd",
            mask_grid="original_rgb_grid; not_automatically_rectified",
        )
    return rgb, path, info


def segment(args, *, output=None, list_only=False):
    from PIL import Image

    from ..sam2 import SAM2Client

    if not args.headless and not list_only:
        display_available()
    rgb, source, provenance = load_rgb(args)
    output = output_path(
        output or args.output, workspace=args.workspace, protected=(source,)
    )
    output.mkdir(parents=True, exist_ok=False)
    Image.fromarray(rgb).save(output / "rgb.png")
    source_name = (
        "source_rgbd.npz" if args.rgbd is not None else f"source{source.suffix.lower()}"
    )
    shutil.copyfile(source, output / source_name)
    provenance.update(
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        headless=args.headless,
        saved_source=source_name,
    )
    write_json(output / "segmentation_request.json", provenance)
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
        provenance["model"] = segmenter.provenance
    provenance["num_candidates"] = len(candidates)
    write_json(output / "segmentation_result.json", provenance)
    if not candidates:
        raise RuntimeError(
            f"没有检测到目标；原图和请求已保留在 {output}，可调整文本或阈值"
        )
    save_candidates(output, rgb, candidates)
    if list_only:
        print(f"候选已保存：{output / 'candidates.jpg'}；使用 object select 确认。")
        return
    index = choose(rgb, candidates, args.instance, headless=args.headless)
    save_selection(output, rgb, candidates, index, provenance)


def select(args):
    import numpy as np
    from PIL import Image

    source = args.segmentation_dir.expanduser().resolve()
    output = output_path(args.output, workspace=args.workspace, protected=(source,))
    provenance = json.loads(
        existing_file(source / "segmentation_result.json").read_text()
    )
    with Image.open(existing_file(source / "rgb.png")) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with np.load(existing_file(source / "candidates.npz"), allow_pickle=False) as saved:
        candidates = [
            {
                "mask": mask,
                "box_xyxy": saved["boxes"][i],
                "score": float(saved["scores"][i]),
                "mask_score": float(saved["mask_scores"][i]),
                "label": str(saved["labels"][i]),
            }
            for i, mask in enumerate(saved["masks"])
        ]
    if not candidates or any(
        item["mask"].dtype != np.bool_ or item["mask"].shape != rgb.shape[:2]
        for item in candidates
    ):
        raise ValueError("保存的候选 mask 与 RGB 不一致")
    index = choose(rgb, candidates, args.instance, headless=args.headless)
    source_name = provenance["saved_source"]
    if Path(source_name).name != source_name:
        raise ValueError("候选元数据不能引用输入目录以外的文件")
    output.mkdir(parents=True, exist_ok=False)
    for name in (
        "rgb.png",
        "candidates.npz",
        "candidates.jpg",
        "segmentation_request.json",
        "segmentation_result.json",
        source_name,
    ):
        shutil.copyfile(existing_file(source / name), output / name)
    provenance.update(headless=args.headless, selection_source=str(source))
    save_selection(output, rgb, candidates, index, provenance)


def generate(args, segmentation_dir=None, output=None):
    from ..reconstruction import generate_mesh, validate_generation_parameters

    validate_generation_parameters(
        args.seed, args.steps, args.octree_resolution, args.num_chunks
    )
    if args.preview:
        display_available()
    source = Path(segmentation_dir or args.segmentation_dir).expanduser().resolve()
    metadata = json.loads(existing_file(source / "selection.json").read_text())
    if metadata["selection"] not in {"explicit_headless_instance", "gui_confirmed"}:
        raise ValueError("建模需要已明确确认的分割")
    image = existing_file(source / "foreground_rgba.png")
    output = output_path(
        output or args.output, workspace=args.workspace, protected=(source,)
    )
    output.mkdir(parents=True, exist_ok=False)
    root = args.workspace / "third_party/Hunyuan3D-2.1"
    weights = args.models_dir / "Hunyuan3D-2.1"
    write_json(
        output / "generation_request.json",
        {
            "segmentation_directory": str(source),
            "selected_instance": metadata["selected_instance"],
            "source_root": str(root),
            "weights_dir": str(weights),
            "seed": args.seed,
            "steps": args.steps,
            "octree_resolution": args.octree_resolution,
            "num_chunks": args.num_chunks,
        },
    )
    mesh_path = output / "mesh_unscaled.obj"
    result = generate_mesh(
        image_rgba_path=image,
        source_root=root,
        weights_dir=weights,
        output_mesh_path=mesh_path,
        seed=args.seed,
        steps=args.steps,
        octree_resolution=args.octree_resolution,
        num_chunks=args.num_chunks,
    )
    write_json(
        output / "generation_result.json",
        {
            **result,
            "mesh_file": mesh_path.name,
            "units": "unscaled",
            "frame": "object",
        },
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
        workspace=args.workspace,
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
    output = output_path(args.output, workspace=args.workspace)
    output.mkdir(parents=True, exist_ok=False)
    segment(args, output=output / "segmentation")
    # 生成进程退出后释放 CUDA，再准备米制资产。
    process = mp.get_context("spawn").Process(
        target=generate,
        args=(args, output / "segmentation", output / "generation"),
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

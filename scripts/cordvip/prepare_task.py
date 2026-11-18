"""复用模型完成任务物体的分割、重建与米制尺度调整。"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]

from cordvip_geometry.mesh import load_mesh, scale_mesh, validate_scale
from cordvip_geometry.weights import DEFAULT_MODELS_DIR, resolve_sam_model

from dexmani_real.geometry.hoi import TASK_MESH_PATH, task_objects


def prepare_task(task):
    task = Path(task).expanduser().resolve()
    objects = []
    for obj in task_objects(task):
        mesh_path = task / obj["name"] / TASK_MESH_PATH
        existing = mesh_path.exists()
        scaling = {
            key: obj[key] for key in ("scale_to_meters", "size_axis", "size_meters") if key in obj
        }
        if not existing or any(value is not None for value in scaling.values()):
            try:
                validate_scale(**scaling)
            except ValueError as error:
                raise ValueError(f"{obj['name']}: {error}") from error
        if existing:
            mesh = load_mesh(mesh_path)
            axis, size = obj.get("size_axis"), obj.get("size_meters")
            if axis is not None and not np.isclose(
                mesh.extents["xyz".index(axis)],
                size,
                rtol=1e-4,
                atol=1e-6,
            ):
                raise ValueError(
                    f"{obj['name']}: 米制 Mesh 的 {axis} 尺寸与实测 {size} m 不一致；"
                    "请核对或重新缩放 Mesh"
                )
            del mesh
        else:
            objects.append(obj)
    if not objects:
        print(f"复用任务 Mesh：{task}")
        return

    import torch
    from cordvip_geometry.reconstruction import load_shape_model, reconstruct_mesh
    from cordvip_geometry.segmentation import SAM2Segmenter
    from cordvip_geometry.visualization import choose, save_selection
    from PIL import Image, ImageOps

    models_dir = ROOT / DEFAULT_MODELS_DIR
    checkpoint, config = resolve_sam_model(models_dir)
    segmenter = SAM2Segmenter(
        ROOT / "third_party/Grounded-SAM-2",
        checkpoint,
        config,
        models_dir / "grounding-dino-base",
    )
    for obj in objects:
        directory = task / obj["name"]
        output = directory / "reconstruction"
        output.mkdir(parents=True, exist_ok=True)
        with Image.open(directory / obj.get("image", "reference.jpg")) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB")).copy()
        candidates = segmenter.predict(rgb, obj["prompt"])
        index = choose(rgb, candidates, obj.get("instance"), headless=True)
        save_selection(output, rgb, candidates, index)

    del segmenter
    gc.collect()
    torch.cuda.empty_cache()

    model = load_shape_model(ROOT / "third_party/Hunyuan3D-2.1", models_dir / "Hunyuan3D-2.1")
    for obj in objects:
        directory = task / obj["name"]
        output = directory / "reconstruction"
        parameters = {
            key: obj[key]
            for key in ("seed", "steps", "octree_resolution", "num_chunks")
            if key in obj
        }
        with Image.open(output / "foreground_rgba.png") as image:
            mesh = reconstruct_mesh(model, image, **parameters)
        mesh.export(output / "mesh_unscaled.obj")
        metric_mesh = scale_mesh(
            mesh,
            **{
                key: obj[key]
                for key in ("scale_to_meters", "size_axis", "size_meters")
                if key in obj
            },
        )
        mesh_path = directory / TASK_MESH_PATH
        mesh_path.parent.mkdir(exist_ok=True)
        metric_mesh.export(mesh_path)
        print(f"已保存米制 Mesh：{directory}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--task-dir", type=Path, required=True, help="任务资产目录；相对 cwd，支持 ~")
    args = parser.parse_args(argv)
    prepare_task(args.task_dir)
    return 0


if __name__ == "__main__":
    main()

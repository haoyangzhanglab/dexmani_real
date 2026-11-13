"""复用模型完成任务物体的分割、重建与米制尺度调整。"""

import argparse
import gc
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]

from cordvip_geometry.mesh import scale_mesh
from cordvip_geometry.weights import DEFAULT_MODELS_DIR, resolve_sam_model

from dexmani_real.geometry.hoi import task_objects


def prepare_task(task):
    task = Path(task)
    objects = [
        obj
        for obj in task_objects(task)
        if not (task / obj["name"] / "mesh/mesh_m.obj").exists()
    ]
    if not objects:
        print(f"复用任务 Mesh：{task}")
        return
    for obj in objects:
        scale, axis, size = (
            obj.get(key) for key in ("scale_to_meters", "size_axis", "size_meters")
        )
        if (axis is None) != (size is None) or (scale is None) == (axis is None):
            raise ValueError(f"{obj['name']}: 必须提供缩放因子或完整实测尺寸，二选一")
        if axis is not None and axis not in ("x", "y", "z"):
            raise ValueError(f"{obj['name']}: 实测轴必须为 x、y 或 z")
        if any(
            value is not None and (not math.isfinite(value) or value <= 0)
            for value in (scale, size)
        ):
            raise ValueError(f"{obj['name']}: 尺度必须为有限正数")

    import numpy as np
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

    model = load_shape_model(
        ROOT / "third_party/Hunyuan3D-2.1", models_dir / "Hunyuan3D-2.1"
    )
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
        (directory / "mesh").mkdir(exist_ok=True)
        metric_mesh.export(directory / "mesh/mesh_m.obj")
        print(f"已保存米制 Mesh：{directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    if Path(args.task).name != args.task or args.task in (".", ".."):
        parser.error("非法任务名")
    prepare_task(ROOT / "assets/pointcloud_template/object" / args.task)


if __name__ == "__main__":
    main()

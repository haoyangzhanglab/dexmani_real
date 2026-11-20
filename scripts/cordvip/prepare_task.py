"""复用模型完成任务物体的分割、重建与米制尺度调整。"""

import argparse
import gc
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "third_party/cordvip_geometry/src"), str(ROOT)]

from cordvip_geometry.mesh import load_mesh, scale_mesh, validate_scale
from cordvip_geometry.reconstruction import validate_reconstruction_parameters
from cordvip_geometry.weights import (
    DEFAULT_MODELS_DIR,
    GROUNDING_WEIGHTS_DIR,
    SHAPE_WEIGHTS_DIR,
    resolve_sam_model,
)

from dexmani_real.geometry.hoi import TASK_MESH_PATH, task_objects
from dexmani_real.utils.io import publish


def prepare_task(task):
    task = Path(task).expanduser().resolve()
    pending = []
    for obj in task_objects(task):
        name = obj["name"]
        directory = task / name
        mesh_path = directory / TASK_MESH_PATH
        existing = mesh_path.exists()
        scaling = {
            key: obj[key] for key in ("scale_to_meters", "size_axis", "size_meters") if key in obj
        }
        if not existing or any(value is not None for value in scaling.values()):
            try:
                validate_scale(**scaling)
            except ValueError as error:
                raise ValueError(f"{name}: {error}") from error
        if existing:
            mesh = load_mesh(mesh_path)
            axis = scaling.get("size_axis")
            if axis is not None:
                size = scaling["size_meters"]
                if not np.isclose(mesh.extents["xyz".index(axis)], size, rtol=1e-4, atol=1e-6):
                    raise ValueError(
                        f"{name}: 米制 Mesh 的 {axis} 尺寸与实测 {size} m 不一致。"
                        "请核对或重新缩放 Mesh"
                    )
            del mesh
            continue

        image = obj.get("image", "reference.jpg")
        if not isinstance(image, str) or not image:
            raise ValueError(f"{name}: image 必须为非空路径字符串")
        image_path = directory / image
        if not image_path.is_file():
            raise FileNotFoundError(f"{name}: 参考图不存在或不是文件：{image_path}")
        instance = obj.get("instance")
        if instance is not None and (type(instance) is not int or instance < 0):
            raise ValueError(f"{name}: instance 必须为非负整数")
        parameters = {
            key: obj[key] for key in ("seed", "steps", "octree_resolution", "num_chunks") if key in obj
        }
        try:
            validate_reconstruction_parameters(**parameters)
        except ValueError as error:
            raise ValueError(f"{name}: {error}") from error

        pending.append((obj, image_path, mesh_path, directory / "reconstruction", scaling, parameters))
    if not pending:
        print(f"复用任务 Mesh：{task}")
        return

    import torch
    from cordvip_geometry.reconstruction import load_shape_model, reconstruct_mesh
    from cordvip_geometry.segmentation import SAM2Segmenter
    from cordvip_geometry.visualization import choose, load_rgb_image, save_selection
    from PIL import Image

    models_dir = ROOT / DEFAULT_MODELS_DIR
    checkpoint, config = resolve_sam_model(models_dir)
    segmenter = SAM2Segmenter(
        ROOT / "third_party/Grounded-SAM-2",
        checkpoint,
        config,
        models_dir / GROUNDING_WEIGHTS_DIR,
    )
    for obj, image_path, _, output, _, _ in pending:
        rgb = load_rgb_image(image_path)
        candidates = segmenter.predict(rgb, obj["prompt"])
        index = choose(rgb, candidates, obj.get("instance"), headless=True)
        output.mkdir(parents=True, exist_ok=True)
        save_selection(output, rgb, candidates, index)

    del segmenter, rgb, candidates
    gc.collect()
    torch.cuda.empty_cache()

    model = load_shape_model(ROOT / "third_party/Hunyuan3D-2.1", models_dir / SHAPE_WEIGHTS_DIR)
    for _, _, mesh_path, output, scaling, parameters in pending:
        with Image.open(output / "foreground_rgba.png") as image:
            mesh = reconstruct_mesh(model, image, **parameters)
        mesh.export(output / "mesh_unscaled.obj")
        metric_mesh = scale_mesh(mesh, **scaling)
        mesh_path.parent.mkdir(exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{mesh_path.name}.", dir=mesh_path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                metric_mesh.export(stream, file_type="obj")
            publish(temporary, mesh_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
        print(f"已保存米制 Mesh：{mesh_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--task-dir", type=Path, required=True, help="任务资产目录；相对 cwd，支持 ~")
    args = parser.parse_args(argv)
    prepare_task(args.task_dir)
    return 0


if __name__ == "__main__":
    main()

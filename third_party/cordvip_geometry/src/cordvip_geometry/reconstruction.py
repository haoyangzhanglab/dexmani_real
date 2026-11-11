"""Hunyuan3D-2.1 单图形状生成；尺度由资产准备阶段确定。"""

import os
import sys
import time
from numbers import Integral
from pathlib import Path

import numpy as np

from .geometry import simplify_mesh
from .io import existing_file, output_path


def validate_generation_parameters(
    seed=0, steps=50, octree_resolution=384, num_chunks=8000
):
    for name, value, minimum in (
        ("seed", seed, 0),
        ("steps", steps, 1),
        ("octree_resolution", octree_resolution, 16),
        ("num_chunks", num_chunks, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(f"{name} 必须是大于等于 {minimum} 的整数")


def generate_mesh(
    image_rgba_path,
    source_root,
    weights_dir,
    output_mesh_path,
    seed=0,
    steps=50,
    octree_resolution=384,
    num_chunks=8000,
):
    validate_generation_parameters(seed, steps, octree_resolution, num_chunks)
    seed, steps, octree_resolution, num_chunks = map(
        int, (seed, steps, octree_resolution, num_chunks)
    )
    image_path = existing_file(image_rgba_path)
    output = output_path(output_mesh_path)
    if output.suffix.lower() != ".obj":
        raise ValueError("Generated mesh output must use .obj")

    from PIL import Image

    with Image.open(image_path) as source_image:
        if source_image.mode != "RGBA":
            raise ValueError(
                "Hunyuan input must be RGBA with the selected object's alpha mask"
            )
        image = source_image.copy()
    alpha = np.asarray(image)[..., 3]
    visible = np.argwhere(alpha > 0)
    if (
        len(visible) < 4
        or np.any(np.ptp(visible, axis=0) < 2)
        or not np.any(alpha == 0)
    ):
        raise ValueError(
            "RGBA input must contain a nonempty foreground and transparent background"
        )
    root = Path(source_root).expanduser().resolve()
    package_parent = root / "hy3dshape"
    existing_file(package_parent / "hy3dshape" / "pipelines.py")
    weights = Path(weights_dir).expanduser().resolve() / "hunyuan3d-dit-v2-1"
    config_path = existing_file(weights / "config.yaml")
    checkpoint_path = existing_file(weights / "model.fp16.ckpt")

    import yaml

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    # 内嵌 DinoV2 配置，禁止 conditioner 隐式下载。
    conditioner = config["conditioner"]["params"]["main_image_encoder"]["kwargs"]
    if conditioner.get("config") is None:
        raise ValueError(
            "Hunyuan conditioner must use embedded model config, without downloads"
        )

    started = time.monotonic()
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1"
    )
    sys.path.insert(0, str(package_parent))
    import torch
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline

    target = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; shape generation was not started")
    torch.cuda.init()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        str(checkpoint_path),
        str(config_path),
        device=str(target),
        dtype=torch.float16,
        use_safetensors=False,
    )
    # eval 还会启用 DiT MoE 的推理分支。
    for component in (pipeline.model, pipeline.vae, pipeline.conditioner):
        component.eval().requires_grad_(False)
    with torch.inference_mode():
        meshes = pipeline(
            image=image,
            generator=torch.Generator(device=target).manual_seed(seed),
            num_inference_steps=steps,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type="trimesh",
        )
    if not isinstance(meshes, list) or len(meshes) != 1 or meshes[0] is None:
        raise RuntimeError("Hunyuan did not return exactly one valid mesh")
    mesh = meshes[0]
    if (
        len(mesh.vertices) < 4
        or len(mesh.faces) < 4
        or not np.isfinite(mesh.vertices).all()
        or np.any(mesh.extents <= 0)
    ):
        raise RuntimeError("Hunyuan returned empty, nonfinite or degenerate geometry")
    mesh = simplify_mesh(mesh)
    payload = mesh.export(file_type="obj").encode("utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(payload)
    elapsed = time.monotonic() - started

    print(f"Mesh: {output}; vertices={len(mesh.vertices)}, faces={len(mesh.faces)}, elapsed={elapsed:.2f}s", flush=True)
    return output

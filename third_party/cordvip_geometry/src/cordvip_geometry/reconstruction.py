"""Hunyuan3D 单图重建；模型在一批物体之间复用。"""

import sys
from pathlib import Path

import numpy as np

from .mesh import simplify_mesh


def load_shape_model(source_root, weights_dir):
    root = Path(source_root).expanduser().resolve()
    weights = Path(weights_dir).expanduser().resolve() / "hunyuan3d-dit-v2-1"
    config_path = weights / "config.yaml"
    checkpoint_path = weights / "model.fp16.ckpt"

    import yaml

    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    # 使用内嵌 DINOv2 配置，避免加载时隐式下载。
    conditioner = config["conditioner"]["params"]["main_image_encoder"]["kwargs"]
    if conditioner.get("config") is None:
        raise ValueError(
            "Hunyuan conditioner must use embedded model config, without downloads"
        )

    sys.path.insert(0, str(root / "hy3dshape"))
    import torch
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; shape generation was not started")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
        str(checkpoint_path),
        str(config_path),
        device="cuda:0",
        dtype=torch.float16,
        use_safetensors=False,
    )
    # eval 还会启用 DiT MoE 的推理分支。
    for component in (pipeline.model, pipeline.vae, pipeline.conditioner):
        component.eval().requires_grad_(False)
    # 仅分层查询表面邻域，保留完整 attention 与原有 Marching Cubes。
    pipeline.vae.enable_flashvdm_decoder(adaptive_kv_selection=False, mc_algo="mc")
    return pipeline


def reconstruct_mesh(
    pipeline, image, *, seed=0, steps=50, octree_resolution=384, num_chunks=8000
):
    import torch

    if image.mode != "RGBA":
        raise ValueError("重建输入必须是带前景 alpha 的 RGBA 图像")
    alpha = np.asarray(image)[..., 3]
    if not np.any(alpha > 0) or not np.any(alpha == 0):
        raise ValueError("重建输入须包含前景和透明背景")
    if seed < 0 or steps < 1 or octree_resolution < 16 or num_chunks < 1:
        raise ValueError("非法重建参数")
    with torch.inference_mode():
        meshes = pipeline(
            image=image,
            generator=torch.Generator(device="cuda:0").manual_seed(seed),
            num_inference_steps=steps,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type="trimesh",
        )
    if not isinstance(meshes, list) or len(meshes) != 1 or meshes[0] is None:
        raise RuntimeError("Hunyuan did not return exactly one valid mesh")
    mesh = meshes[0]
    # 未查询区域经上游 Mesh 转换后可能留下退化面。
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()
    if (
        len(mesh.vertices) < 4
        or len(mesh.faces) < 4
        or not np.isfinite(mesh.vertices).all()
        or np.any(mesh.extents <= 0)
    ):
        raise RuntimeError("Hunyuan returned empty, nonfinite or degenerate geometry")
    return simplify_mesh(mesh)

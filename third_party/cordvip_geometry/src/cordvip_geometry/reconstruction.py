"""Isolated Hunyuan3D-2.1 shape inference; metric scale and object axes need explicit calibration."""

import gc
import importlib.metadata
import os
from pathlib import Path
import sys
import time

import numpy as np

from .io import existing_file


def generate_mesh(
    image_rgba_path,
    source_root,
    weights_dir,
    output_mesh_path,
    seed=0,
    steps=50,
    octree_resolution=384,
    num_chunks=8000,
    device="cuda",
):
    """Generate from local 2.1 weights and retain provenance; no mesh cleanup or metric scaling."""
    for name, value, minimum in (
        ("seed", seed, 0),
        ("steps", steps, 1),
        ("octree_resolution", octree_resolution, 16),
        ("num_chunks", num_chunks, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    image_path = existing_file(image_rgba_path)
    root = Path(source_root).expanduser().resolve()
    package_parent = root / "hy3dshape"
    existing_file(package_parent / "hy3dshape" / "pipelines.py")
    weights = Path(weights_dir).expanduser().resolve()
    if (weights / "hunyuan3d-dit-v2-1").is_dir():
        weights = weights / "hunyuan3d-dit-v2-1"
    config_path = existing_file(weights / "config.yaml")
    checkpoint_path = existing_file(weights / "model.fp16.ckpt")
    output = Path(output_mesh_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite generated mesh: {output}")
    if output.suffix.lower() not in (".obj", ".ply", ".glb"):
        raise ValueError("Generated mesh output must use .obj, .ply or .glb")

    from PIL import Image
    import yaml

    with Image.open(image_path) as source_image:
        if source_image.mode != "RGBA":
            raise ValueError("Hunyuan input must be RGBA with the selected object's alpha mask")
        image = source_image.copy()
    alpha = np.asarray(image)[..., 3]
    visible = np.argwhere(alpha > 0)
    if len(visible) < 4 or np.any(np.ptp(visible, axis=0) < 2) or not np.any(alpha == 0):
        raise ValueError("RGBA input must contain a nonempty foreground and transparent background")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    # Local DinoV2 weights must prevent the conditioner from downloading models implicitly.
    conditioner = config["conditioner"]["params"]["main_image_encoder"]["kwargs"]
    if conditioner.get("config") is None:
        raise ValueError("Hunyuan conditioner must use embedded model config, without downloads")

    loaded = sys.modules.get("hy3dshape")
    if loaded is not None:
        source = getattr(loaded, "__file__", None)
        if source is None or not Path(source).resolve().is_relative_to(package_parent):
            raise RuntimeError("A different hy3dshape package is loaded; use a fresh subprocess")
    previous_path = sys.path.copy()
    offline_keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
    previous_env = {key: os.environ.get(key) for key in offline_keys}
    pipeline = None
    component = None
    started = time.monotonic()
    try:
        os.environ.update({key: "1" for key in offline_keys})
        sys.path.insert(0, str(package_parent))
        import torch
        from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline

        target = torch.device(device)
        if target.type != "cuda":
            raise ValueError("This shape-generation example requires an explicit CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; shape generation was not started")
        if target.index is not None and target.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device does not exist: {target}")
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(target)
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            str(checkpoint_path),
            str(config_path),
            device=str(target),
            dtype=torch.float16,
            use_safetensors=False,
        )
        # inference_mode disables gradients; eval also selects the DiT MoE inference branch.
        for component in (pipeline.model, pipeline.vae, pipeline.conditioner):
            component.eval().requires_grad_(False)
        component = None
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
        payload = mesh.export(file_type=output.suffix[1:].lower())
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(payload)
        elapsed = time.monotonic() - started
        peak_allocated_bytes = int(torch.cuda.max_memory_allocated(target))
        peak_reserved_bytes = int(torch.cuda.max_memory_reserved(target))
    finally:
        pipeline = None
        component = None
        sys.path[:] = previous_path
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        gc.collect()
        torch_module = sys.modules.get("torch")
        if (
            torch_module is not None
            and "target" in locals()
            and target.type == "cuda"
            and torch_module.cuda.is_available()
            and (target.index is None or target.index < torch_module.cuda.device_count())
        ):
            with torch_module.cuda.device(target):
                torch_module.cuda.empty_cache()

    return {
        "method": "Hunyuan3D-2.1 shape-only",
        "source_root": str(root),
        "image_rgba_path": str(image_path),
        "weights_directory": str(weights),
        "mesh_path": str(output),
        "units": "unscaled",
        "object_frame": "generator_output_without_recentering",
        "seed": seed,
        "steps": steps,
        "octree_resolution": octree_resolution,
        "num_chunks": num_chunks,
        "device": str(device),
        "dtype": "float16",
        "model_mode": "eval",
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_unscaled": mesh.bounds.tolist(),
        "generation_seconds": elapsed,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
        "memory_scope": "PyTorch CUDA allocator; excludes external CUDA allocations",
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "transformers",
                "diffusers",
                "trimesh",
            )
        },
    }

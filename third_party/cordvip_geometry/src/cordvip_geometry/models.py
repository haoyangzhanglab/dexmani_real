"""本地感知模型、固定官方 revision 与参数配对。"""

from pathlib import Path
from typing import Any

DEFAULT_MODELS_DIR = Path("third_party/weights")

SAM_MODELS: dict[str, dict[str, str]] = {
    "small": {
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "repo_id": "facebook/sam2.1-hiera-small",
        "revision": "ee5bba1d82bb8749febdf90f45e84b687142ba03",
        "weight": "sam2.1_hiera_small.pt",
    },
    "large": {
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "repo_id": "facebook/sam2.1-hiera-large",
        "revision": "665f8e2ad61cf5f53d65644ff27c8ee525124610",
        "weight": "sam2.1_hiera_large.pt",
    },
}
GROUNDING_MODEL: dict[str, Any] = {
    "repo_id": "IDEA-Research/grounding-dino-base",
    "revision": "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
    "directory": "grounding-dino-base",
    "weight": "model.safetensors",
    "patterns": ["*.json", "model.safetensors", "vocab.txt", "README.md", "LICENSE*"],
}
GROUNDING_FILES = (
    GROUNDING_MODEL["weight"],
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)

SHAPE_MODEL: dict[str, Any] = {
    "repo_id": "tencent/Hunyuan3D-2.1",
    "revision": "0b94677654c57bb9a6b6845cd7b704ccf551d327",
    "directory": "Hunyuan3D-2.1",
    "patterns": [
        "hunyuan3d-dit-v2-1/config.yaml",
        "hunyuan3d-dit-v2-1/model.fp16.ckpt",
        "README.md",
        "LICENSE*",
    ],
    "weight": "hunyuan3d-dit-v2-1/model.fp16.ckpt",
}


def resolve_sam_model(models_dir, variant=None, checkpoint=None, config=None):
    custom = checkpoint is not None or config is not None
    if custom:
        if checkpoint is None or config is None:
            raise ValueError("--sam-checkpoint 与 --sam-config 必须完整配对")
        if variant is not None:
            raise ValueError("自训配对不能同时指定 --sam-variant")
        return Path(checkpoint).expanduser().resolve(), config
    model = SAM_MODELS[variant or "small"]
    return Path(models_dir) / model["weight"], model["config"]

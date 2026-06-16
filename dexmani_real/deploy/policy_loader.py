"""PolicyLoader — load policy model and normalization stats from checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class PolicyLoader:
    """Load a trained policy from a checkpoint directory.

    Directory structure:
        checkpoint/
          config.json          # policy config (obs dims, action dims, chunk, etc.)
          model.safetensors    # model weights
          stats.json           # per-joint normalization statistics

    Returns (model, norm_stats, policy_config) tuple.
    """

    @staticmethod
    def load(checkpoint_dir: str) -> tuple[Any, dict, dict]:
        root = Path(checkpoint_dir)

        config_path = root / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {checkpoint_dir}")
        with open(config_path) as f:
            policy_config = json.load(f)

        stats_path = root / "stats.json"
        norm_stats: dict = {}
        if stats_path.exists():
            with open(stats_path) as f:
                norm_stats = json.load(f)

        model = PolicyLoader._load_model(root, policy_config)

        return model, norm_stats, policy_config

    @staticmethod
    def _load_model(root: Path, policy_config: dict) -> Any:
        """Load model weights. Override for your specific model architecture."""
        safetensors_path = root / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"model.safetensors not found in {root}")

        # Stub: actual model loading depends on your architecture.
        # Import torch / safetensors locally to avoid hard dependency.
        try:
            import torch
            from safetensors.torch import load_file
        except ImportError:
            raise ImportError(
                "torch and safetensors required for model loading. "
                "Install with: pip install torch safetensors"
            )

        state_dict = load_file(str(safetensors_path))
        # Build model from policy_config and load state_dict
        raise NotImplementedError(
            "Model architecture loading not implemented. "
            "Override _load_model() for your specific model."
        )

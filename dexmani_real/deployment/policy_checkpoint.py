"""Real-owned reader for the frozen deployment-v2 Policy checkpoint."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, BinaryIO

import torch

_PAYLOAD_KEYS = frozenset({"_format", "state", "weights"})
_STATE_KEYS = frozenset(
    {
        "epoch",
        "global_step",
        "train_params",
        "inference_config",
        "data_contract",
        "producer",
        "deployment_contract",
    }
)
_WEIGHT_KEYS = frozenset({"model", "ema_model"})
_FORMAT = "dexmani.deployment.v2"


@dataclass(frozen=True)
class LoadedPolicyCheckpoint:
    """Validated deployment-only checkpoint; no Policy class is imported."""

    epoch: int
    global_step: int
    train_params: dict[str, Any]
    inference_config: dict[str, Any]
    data_contract: dict[str, Any]
    producer: dict[str, Any]
    deployment_contract: dict[str, Any]
    model_state: dict[str, torch.Tensor]
    ema_model_state: dict[str, torch.Tensor] | None


def load_deployment_checkpoint_stream(stream: BinaryIO) -> LoadedPolicyCheckpoint:
    """Deserialize exactly one deployment-v2 payload with PyTorch safe loading."""
    payload = torch.load(stream, map_location="cpu", weights_only=True)
    if type(payload) is not dict:
        raise ValueError("deployment checkpoint payload must be a plain dict")
    _require_exact_keys(payload, _PAYLOAD_KEYS, "payload")
    if payload["_format"] != _FORMAT:
        raise ValueError("unsupported deployment checkpoint format")
    state = _require_plain_dict(payload["state"], "state")
    weights = _require_plain_dict(payload["weights"], "weights")
    _require_exact_keys(state, _STATE_KEYS, "state")
    _require_exact_keys(weights, _WEIGHT_KEYS, "weights")
    metadata = {
        name: _require_plain_dict(state[name], name)
        for name in (
            "train_params",
            "inference_config",
            "data_contract",
            "producer",
            "deployment_contract",
        )
    }
    for name, value in metadata.items():
        _validate_plain_metadata(value, name)
    return LoadedPolicyCheckpoint(
        epoch=_require_nonnegative_int(state["epoch"], "epoch"),
        global_step=_require_nonnegative_int(state["global_step"], "global_step"),
        train_params=metadata["train_params"],
        inference_config=metadata["inference_config"],
        data_contract=metadata["data_contract"],
        producer=metadata["producer"],
        deployment_contract=metadata["deployment_contract"],
        model_state=_validate_state_dict(weights["model"], "weights.model"),
        ema_model_state=(
            None
            if weights["ema_model"] is None
            else _validate_state_dict(weights["ema_model"], "weights.ema_model")
        ),
    )


def _require_plain_dict(value: Any, name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"deployment checkpoint {name} must be a plain dict")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], expected: frozenset[str], name: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"deployment checkpoint {name} has an unsupported schema")


def _require_nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"deployment checkpoint {name} must be a non-negative integer")
    return value


def _validate_plain_metadata(value: Any, name: str) -> None:
    if value is None or type(value) in {bool, int, str}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(
                f"deployment checkpoint {name} contains a non-finite float"
            )
        return
    if type(value) is list:
        for item in value:
            _validate_plain_metadata(item, name)
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError(f"deployment checkpoint {name} must use string keys")
            _validate_plain_metadata(item, name)
        return
    raise ValueError(f"deployment checkpoint {name} must contain only plain metadata")


def _validate_state_dict(value: Any, name: str) -> dict[str, torch.Tensor]:
    state_dict = _require_plain_dict(value, name)
    if not state_dict:
        raise ValueError(f"deployment checkpoint {name} must not be empty")
    for key, tensor in state_dict.items():
        if type(key) is not str or not key:
            raise ValueError(
                f"deployment checkpoint {name} must use non-empty string keys"
            )
        if type(tensor) is not torch.Tensor:
            raise ValueError(f"deployment checkpoint {name} values must be tensors")
        if key.startswith("module.") or "_orig_mod." in key:
            raise ValueError(
                f"deployment checkpoint {name} has non-canonical state keys"
            )
    return state_dict

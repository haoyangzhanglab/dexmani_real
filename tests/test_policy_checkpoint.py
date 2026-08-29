"""Pure deployment-v2 checkpoint decoder contracts."""

from __future__ import annotations

import io
import unittest

import torch

from dexmani_real.deployment.policy_checkpoint import load_deployment_checkpoint_stream


def _payload() -> dict:
    metadata = {
        "action_key": "action",
        "action_dim": 19,
        "horizon": 16,
        "n_obs_steps": 2,
        "n_action_steps": 8,
    }
    return {
        "_format": "dexmani.deployment.v2",
        "state": {
            "epoch": 1,
            "global_step": 2,
            "train_params": dict(metadata),
            "inference_config": {**metadata, "eval": {"use_ema": True}},
            "data_contract": {
                "action_key": "action",
                "model_action_dim": 19,
                "horizon": 16,
                "n_obs_steps": 2,
                "n_action_steps": 8,
            },
            "producer": {"commit": "a" * 40},
            "deployment_contract": {"schema_version": 1},
        },
        "weights": {
            "model": {"weight": torch.ones(1)},
            "ema_model": {"weight": torch.ones(1)},
        },
    }


def _load(payload: dict):
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    return load_deployment_checkpoint_stream(stream)


class PolicyCheckpointTest(unittest.TestCase):
    def test_loads_exact_v2_payload_without_policy_import(self) -> None:
        checkpoint = _load(_payload())
        self.assertEqual(checkpoint.epoch, 1)
        self.assertEqual(checkpoint.global_step, 2)
        self.assertEqual(tuple(checkpoint.model_state), ("weight",))
        self.assertEqual(tuple(checkpoint.ema_model_state or {}), ("weight",))

    def test_rejects_extra_schema_key(self) -> None:
        payload = _payload()
        payload["state"]["monitor"] = {}
        with self.assertRaisesRegex(ValueError, "unsupported schema"):
            _load(payload)

    def test_rejects_nonfinite_metadata_and_noncanonical_keys(self) -> None:
        payload = _payload()
        payload["state"]["train_params"]["bad"] = float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _load(payload)

        payload = _payload()
        payload["weights"]["model"] = {"_orig_mod.weight": torch.ones(1)}
        with self.assertRaisesRegex(ValueError, "non-canonical"):
            _load(payload)

    def test_rejects_non_tensor_weight_value(self) -> None:
        payload = _payload()
        payload["weights"]["ema_model"] = {"weight": [1.0]}
        with self.assertRaisesRegex(ValueError, "must be tensors"):
            _load(payload)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

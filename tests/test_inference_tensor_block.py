from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pytest

from dexmani_real.policy.inference_process import InferenceConfig
from dexmani_real.policy.observation import CausalFrame, SnapshotBuilder
from dexmani_real.policy.runtime import ActionSpec, ModalitySpec, ObservationSpec
from dexmani_real.policy.tensor_block import ObservationTensorBlock


def test_observation_tensor_block_round_trip_is_fixed_and_verified() -> None:
    spec = ObservationSpec(
        (
            ModalitySpec("arm", (7,), "float64", history_length=2),
            ModalitySpec("image", (2, 3, 3), "uint8", history_length=1),
        )
    )
    block = ObservationTensorBlock.create(f"tensor_{uuid.uuid4().hex}", spec)
    try:
        anchor = time.monotonic_ns()
        snapshot = SnapshotBuilder(spec, session_generation=4, camera_generation=2).build(
            anchor_monotonic_ns=anchor,
            frames={
                "arm": [CausalFrame(np.arange(7, dtype=np.float64), 1, anchor - 50_000_000, anchor - 40_000_000)],
                "image": [CausalFrame(np.arange(18, dtype=np.uint8).reshape(2, 3, 3), 1, anchor - 1, anchor)],
            },
        )
        sequence = block.write(snapshot)
        result = block.read_latest()

        assert result is not None and result[1] == sequence
        restored = result[0]
        assert restored.observation_id == snapshot.observation_id
        assert restored.session_generation == 4
        assert restored.camera_generation == 2
        np.testing.assert_array_equal(restored.values["arm"], snapshot.values["arm"])
        np.testing.assert_array_equal(restored.values["image"], snapshot.values["image"])
        np.testing.assert_array_equal(restored.valid_history_mask["arm"], snapshot.valid_history_mask["arm"])
        np.testing.assert_array_equal(restored.receive_monotonic_ns["image"], snapshot.receive_monotonic_ns["image"])
        np.testing.assert_allclose(restored.source_age_s["image"], snapshot.source_age_s["image"])
        np.testing.assert_allclose(restored.source_skew_s["image"], snapshot.source_skew_s["image"])
        with pytest.raises(ValueError, match="read-only"):
            restored.values["arm"][0, 0] = 99.0
    finally:
        block.close()
        block.unlink()


def test_snapshot_history_never_duplicates_one_source_frame() -> None:
    spec = ObservationSpec((ModalitySpec("arm", (1,), "float64", history_length=3),), control_hz=10.0)
    anchor = time.monotonic_ns()
    only_frame = CausalFrame(np.array([3.0]), 8, anchor - 50_000_000, anchor - 49_000_000)

    snapshot = SnapshotBuilder(spec, session_generation=1).build(
        anchor_monotonic_ns=anchor,
        frames={"arm": [only_frame]},
    )

    assert np.count_nonzero(snapshot.valid_history_mask["arm"]) == 1
    np.testing.assert_array_equal(snapshot.values["arm"][:2], np.zeros((2, 1)))
    np.testing.assert_array_equal(snapshot.values["arm"][2], np.array([3.0]))


def test_inference_manifest_verifies_resource_hash_and_normalizes_shapes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.bin"
    checkpoint.write_bytes(b"verified-model")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    manifest = {
        "backend_entrypoint": "tests.fake_backend:Backend",
        "modalities": [{"name": "arm_qpos", "shape": [7], "dtype": "float64"}],
        "action": {"arm_shape": [7], "hand_shape": [12], "chunk_length": 2},
        "resources": {"checkpoint": {"path": "model.bin", "sha256": digest}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    config = InferenceConfig.from_manifest(path)

    assert config.observation_spec.modalities[0].shape == (7,)
    assert config.action_spec.chunk_length == 2
    assert config.resource_hashes == (("checkpoint", digest),)

    manifest["resources"]["checkpoint"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        InferenceConfig.from_manifest(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": "object"},
        {"clock": "wall_clock"},
        {"padding": "repeat_oldest"},
    ],
)
def test_modality_spec_rejects_unsupported_runtime_contracts(kwargs: dict[str, str]) -> None:
    values = {"name": "arm_qpos", "shape": (7,), "dtype": "float64", **kwargs}
    with pytest.raises(ValueError):
        ModalitySpec(**values)  # type: ignore[arg-type]


def test_action_spec_rejects_unsupported_runtime_contracts() -> None:
    with pytest.raises(ValueError, match="representation"):
        ActionSpec(units="deg")  # type: ignore[arg-type]

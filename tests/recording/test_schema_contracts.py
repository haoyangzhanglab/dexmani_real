from __future__ import annotations

import hashlib
import json
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from dexmani_real.data.contracts import OutputProfile
from dexmani_real.recording.schema import (
    CAMERA_TIMING_DATASET_SPECS,
    CONDITIONAL_DATASET_SPECS,
    DATASET_SPECS,
    DIAGNOSTIC_TAIL_SHAPES,
    EPISODE_SCHEMA_VERSION,
    normalize_diagnostics,
)

MANIFEST = (
    Path(__file__).parents[1]
    / "fixtures"
    / "contracts"
    / "storage_schema_baseline.json"
)


def group_contract(specs: Mapping[str, Any]) -> dict[str, object]:
    items = [
        {
            "name": name,
            "dtype": spec.dtype.str,
            "tail_shape": list(spec.tail_shape),
        }
        for name, spec in sorted(specs.items())
    ]
    payload = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    return {
        "count": len(items),
        "names": [item["name"] for item in items],
        "layout_sha256": hashlib.sha256(payload).hexdigest(),
    }


class StorageSchemaContractTest(unittest.TestCase):
    def test_raw_processed_and_zarr_contracts_are_frozen(self) -> None:
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
        actual = {
            "raw": {
                "schema_name": "dexmani-real-episode",
                "schema_version": EPISODE_SCHEMA_VERSION,
                "groups": {
                    "datasets": group_contract(DATASET_SPECS),
                    "camera_timing": group_contract(CAMERA_TIMING_DATASET_SPECS),
                    "conditional": group_contract(CONDITIONAL_DATASET_SPECS),
                },
            },
            "processed": {
                "schema_name": "dexmani-real-processed-hdf5",
                "schema_version": 5,
                "profiles": {
                    profile.value: list(profile.dataset_keys)
                    for profile in OutputProfile
                },
            },
            "zarr": {
                "schema_name": "dexmani-real-policy-zarr",
                "schema_version": 3,
            },
        }
        self.assertEqual(actual, expected)

    def test_diagnostic_shapes_match_persisted_schema(self) -> None:
        for name, shape in DIAGNOSTIC_TAIL_SHAPES.items():
            with self.subTest(name=name):
                self.assertIn(name, DATASET_SPECS)
                self.assertEqual(shape, DATASET_SPECS[name].tail_shape)

    def test_head_quaternion_accepts_production_shape(self) -> None:
        quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        normalized = normalize_diagnostics({"head_quat_wxyz": quaternion})
        np.testing.assert_array_equal(normalized["head_quat_wxyz"], quaternion)


if __name__ == "__main__":
    unittest.main()

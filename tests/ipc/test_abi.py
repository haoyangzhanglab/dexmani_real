from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from dexmani_real.ipc import schema

MANIFEST = (
    Path(__file__).parents[1] / "fixtures" / "contracts" / "ipc_abi_baseline.json"
)


def dtype_contract(dtype: np.dtype) -> dict[str, object]:
    dtype_fields = dtype.fields
    assert dtype_fields is not None
    fields = [
        {
            "name": name,
            "dtype": dtype_fields[name][0].base.str,
            "shape": list(dtype_fields[name][0].shape),
            "offset": dtype_fields[name][1],
        }
        for name in dtype.names or ()
    ]
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return {
        "aligned": bool(dtype.isalignedstruct),
        "field_count": len(fields),
        "field_names": [field["name"] for field in fields],
        "itemsize": dtype.itemsize,
        "layout_sha256": hashlib.sha256(payload).hexdigest(),
    }


class IPCABIContractTest(unittest.TestCase):
    def test_fixed_and_factory_dtypes_match_frozen_manifest(self) -> None:
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
        actual = {
            name: dtype_contract(getattr(schema, name))
            for name in (
                "ARM_COMMAND_DTYPE",
                "HAND_COMMAND_DTYPE",
                "POLICY_PLAN_DTYPE",
                "ARM_STATE_DTYPE",
                "HAND_STATE_DTYPE",
                "HAND_TACTILE_DTYPE",
                "VR_FRAME_DTYPE",
                "CAMERA_FRAME_HEADER_DTYPE",
                "RECORD_CONTROL_DTYPE",
                "RECORD_STATUS_DTYPE",
            )
        }
        actual["POINTCLOUD_1024"] = dtype_contract(
            schema.make_pointcloud_frame_dtype(1024)
        )
        actual["RECORD_SAMPLE_2X3"] = dtype_contract(
            schema.make_record_sample_dtype((2, 3, 3), (2, 3))
        )
        self.assertEqual(actual, expected)

    def test_supported_pointcloud_sizes_preserve_payload_shape(self) -> None:
        for count in sorted(schema.SUPPORTED_POINT_CLOUD_COUNTS):
            with self.subTest(count=count):
                dtype = schema.make_pointcloud_frame_dtype(count)
                self.assertEqual(dtype["point_cloud"].shape, (count, 6))
                self.assertEqual(dtype["point_cloud"].base, np.dtype("<f4"))


if __name__ == "__main__":
    unittest.main()

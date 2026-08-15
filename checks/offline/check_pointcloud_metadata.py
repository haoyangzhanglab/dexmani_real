"""A15: pointcloud filter config is persisted into episode START metadata.

``PointCloudProcessorConfig.to_meta_dict`` must be JSON-safe (h5py /meta) and
``_build_start_metadata`` must decode the shared ``camera_pointcloud_config``
text field into ``camera_metadata["camera_pointcloud_config_json"]``, falling
back to ``{}`` when the field is unset.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.sensor.pointcloud_processor import PointCloudProcessorConfig
from dexmani_real.recording.io_process import _build_start_metadata


def _field(value: bytes) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def main() -> int:
    # 1. to_meta_dict is JSON-serializable (h5py-safe) and pc_-prefixed.
    cfg = PointCloudProcessorConfig()
    meta = cfg.to_meta_dict()
    assert meta and all(str(k).startswith("pc_") for k in meta), (
        "to_meta_dict keys must be pc_-prefixed"
    )
    roundtrip = json.loads(json.dumps(meta))  # must not raise on any non-JSON type
    assert roundtrip["pc_dbscan_min_cluster_size"] == cfg.dbscan_min_cluster_size

    # 2. The shared text field is decoded into camera_metadata at the START
    #    boundary.
    pc_json = json.dumps(meta, separators=(",", ":"))
    shared = SimpleNamespace(
        camera_K=[0.0] * 9,
        camera_depth_scale=_field(0.0),
        camera_serial=_field(b"\x00" * 32),
        camera_firmware=_field(b"1.5.0".ljust(64, b"\x00")),
        camera_sdk_version=_field(b"2.55.0".ljust(64, b"\x00")),
        camera_profile=_field(b"{}".ljust(2048, b"\x00")),
        camera_pointcloud_config=_field(pc_json[:2047].ljust(2048, "\x00").encode()),
        arm_device_identity=_field(b"\x00" * 1024),
        hand_device_identity=_field(b"\x00" * 1024),
    )
    result = _build_start_metadata(shared, task_label="t", operator="o")
    got = result["camera_metadata"]["camera_pointcloud_config_json"]
    assert json.loads(got)["pc_dbscan_min_cluster_size"] == cfg.dbscan_min_cluster_size

    # 3. Unset field falls back to "{}".
    shared.camera_pointcloud_config = _field(b"\x00" * 2048)
    result = _build_start_metadata(shared, task_label="t", operator="o")
    assert result["camera_metadata"]["camera_pointcloud_config_json"] == "{}"

    print("check_pointcloud_metadata: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""A15: pointcloud and aligned-camera geometry reach START metadata.

``PointCloudProcessorConfig.to_meta_dict`` must be JSON-safe (h5py /meta) and
``_build_start_metadata`` must decode the shared ``camera_pointcloud_config``
text field into ``camera_metadata["camera_pointcloud_config_json"]``, falling
back to ``{}`` when the field is unset.  The actual camera profile must also
agree with the configured common viewport, output optical frame, and camera K.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import _bootstrap  # noqa: F401  (repo root on sys.path)

from dexmani_real.sensor.pointcloud_processor import PointCloudProcessorConfig
from dexmani_real.recording.io_process import _build_start_metadata


def _field(value: object) -> SimpleNamespace:
    return SimpleNamespace(value=value)


def main() -> int:
    # 1. to_meta_dict is JSON-serializable (h5py-safe) and pc_-prefixed.
    cfg = PointCloudProcessorConfig()
    meta = cfg.to_meta_dict()
    assert meta and all(str(k).startswith("pc_") for k in meta), (
        "to_meta_dict keys must be pc_-prefixed"
    )
    assert "pc_depth_ema_alpha" not in meta, "removed temporal depth EMA must not be persisted"
    roundtrip = json.loads(json.dumps(meta))  # must not raise on any non-JSON type
    assert roundtrip["pc_dbscan_min_cluster_size"] == cfg.dbscan_min_cluster_size

    # 2. The shared text field is decoded into camera_metadata at the START
    #    boundary.
    pc_json = json.dumps(meta, separators=(",", ":"))
    camera_profile_json = json.dumps(
        {
            "align_mode": "depth_to_color",
            "common_viewport": "color",
            "output_optical_frame": "camera_color_optical",
            "output_intrinsics": {
                "width": 640,
                "height": 480,
                "fx": 600.0,
                "fy": 601.0,
                "cx": 320.0,
                "cy": 240.0,
                "model": "none",
                "coeffs": [0.0] * 5,
            },
        },
        separators=(",", ":"),
    )
    shared = SimpleNamespace(
        camera_K=[600.0, 0.0, 320.0, 0.0, 601.0, 240.0, 0.0, 0.0, 1.0],
        camera_depth_scale=_field(0.0),
        camera_serial=_field(b"\x00" * 32),
        camera_firmware=_field(b"1.5.0".ljust(64, b"\x00")),
        camera_sdk_version=_field(b"2.55.0".ljust(64, b"\x00")),
        camera_profile=_field(camera_profile_json.ljust(2048, "\x00").encode()),
        camera_pointcloud_config=_field(pc_json[:2047].ljust(2048, "\x00").encode()),
        arm_device_identity=_field(b"\x00" * 1024),
        hand_device_identity=_field(b"\x00" * 1024),
    )
    result = _build_start_metadata(
        shared,
        task_label="t",
        operator="o",
        align_mode="depth_to_color",
    )
    got = result["camera_metadata"]["camera_pointcloud_config_json"]
    assert json.loads(got)["pc_dbscan_min_cluster_size"] == cfg.dbscan_min_cluster_size
    camera_meta = result["camera_metadata"]
    assert camera_meta["camera_alignment_mode"] == "depth_to_color"
    assert camera_meta["camera_common_viewport"] == "color"
    assert camera_meta["camera_K_optical_frame"] == "camera_color_optical"
    assert camera_meta["camera_output_optical_frame"] == "camera_color_optical"

    # 3. START must not label an actual depth-frame profile as color-frame
    #    geometry merely because RecorderIO was configured for depth_to_color.
    mismatched_profile = json.dumps(
        {
            "align_mode": "color_to_depth",
            "common_viewport": "depth",
            "output_optical_frame": "camera_depth_optical",
        },
        separators=(",", ":"),
    )
    shared.camera_profile = _field(mismatched_profile.ljust(2048, "\x00").encode())
    try:
        _build_start_metadata(
            shared,
            task_label="t",
            operator="o",
            align_mode="depth_to_color",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("mismatched actual camera geometry was accepted")

    # 4. Unset pointcloud config falls back to "{}".
    shared.camera_profile = _field(camera_profile_json.ljust(2048, "\x00").encode())
    shared.camera_pointcloud_config = _field(b"\x00" * 2048)
    result = _build_start_metadata(
        shared,
        task_label="t",
        operator="o",
        align_mode="depth_to_color",
    )
    assert result["camera_metadata"]["camera_pointcloud_config_json"] == "{}"

    print("check_pointcloud_metadata: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

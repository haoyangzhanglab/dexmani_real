from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from dexmani_real.config.camera_calib import CameraCalib, CameraCalibEntry
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.data.contracts import OutputProfile, ProcessingConfig
from dexmani_real.sensor.pointcloud_worker import (
    PointCloudLoopConfig,
    _resolve_base_from_color,
)


def calibration_payload(position: list[float], *, serial: str = "camera-1") -> dict:
    return {
        "camera_0": {
            "serial": serial,
            "type": "eye_to_hand",
            "pose": {
                "position": position,
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
        }
    }


class CameraCalibrationSnapshotTest(unittest.TestCase):
    def test_entry_remains_read_only_after_spawn_pickle_roundtrip(self) -> None:
        transform = np.eye(4, dtype=np.float64)
        entry = CameraCalibEntry(
            serial="camera-1",
            type="eye_to_hand",
            T_world_camera=transform,
        )

        restored = pickle.loads(pickle.dumps(entry))

        assert restored.T_world_camera is not None
        self.assertFalse(restored.T_world_camera.flags.writeable)
        with self.assertRaises(ValueError):
            restored.T_world_camera[0, 3] = 1.0

    def test_loaded_snapshot_does_not_follow_later_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            path.write_text(json.dumps(calibration_payload([1.0, 2.0, 3.0])))
            calibration = CameraCalib(str(path))
            original_hash = calibration.source_sha256
            config = PointCloudLoopConfig(
                pointcloud=PointCloudConfig(num_points=1024),
                camera_calibration=calibration,
            )

            path.write_text(json.dumps(calibration_payload([9.0, 9.0, 9.0])))
            shared = SimpleNamespace(
                camera_serial=SimpleNamespace(value=b"camera-1\x00")
            )
            transform = _resolve_base_from_color(
                cast(Any, shared), config.camera_calibration
            )

        np.testing.assert_allclose(transform[:3, 3], (1.0, 2.0, 3.0))
        self.assertEqual(config.camera_calibration.source_sha256, original_hash)

    def test_duplicate_serial_is_rejected(self) -> None:
        payload = calibration_payload([0.0, 0.0, 0.0])
        payload["camera_1"] = calibration_payload([1.0, 0.0, 0.0], serial="camera-1")[
            "camera_0"
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "appears more than once"):
                CameraCalib(str(path))

    def test_non_string_serial_is_rejected(self) -> None:
        payload = calibration_payload([0.0, 0.0, 0.0])
        payload["camera_0"]["serial"] = 123
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cameras.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "non-empty string"):
                CameraCalib(str(path))


class ProcessingRuntimeProjectionTest(unittest.TestCase):
    def test_runtime_override_reaches_processing_config(self) -> None:
        runtime = resolve_runtime_config(
            data={"arm": {"tracking_error_warn_rad": 0.123}}
        )
        config = ProcessingConfig.from_runtime(
            runtime,
            profile=OutputProfile.JOINT,
        )
        self.assertEqual(config.tracking_error_warn_rad, 0.123)
        self.assertEqual(
            config.arm_joint_limit_lower_rad,
            runtime.arm.joint_limit_lower,
        )
        self.assertEqual(
            config.hand_action_limit_upper_rad,
            runtime.hand.qpos_max_rad,
        )


if __name__ == "__main__":
    unittest.main()

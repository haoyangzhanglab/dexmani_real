"""导出身份、有效前缀与视频保留范围的无设备回归。"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import zarr

from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.data.export import ExportConfig, export_zarr, training_prefix
from dexmani_real.data.raw import RawReader, RawWriter, snapshot_recording_metadata
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import STATUS_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import Observation


class ExportReliabilityTests(unittest.TestCase):
    def make_raw(self, path):
        path.mkdir()
        transform = np.eye(4)
        transform[0, 3] = .4
        camera = CameraSnapshot(
            "test", CameraIntrinsics(32, 32, 200., 200., 16., 16., "none", (0.,) * 5),
            .001, transform,
        )
        metadata = snapshot_recording_metadata(
            camera, tactile_reset_metadata={}, tactile_hold_max_age_s=.2,
        ) | dict(task_label="test", collection_source="policy_rollout")
        writer = RawWriter(path, control_hz=30.)
        arm = np.zeros(1, dtype=STATUS_DTYPE)["frame"]
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        try:
            writer.open(metadata)
            for index in range(3):
                control = 1_000_000_000 + index * round(1e9 / 30)
                if index == 2:
                    control += 9_000_000  # 严格节拍规则应只保留前两行。
                arm["timestamp_ns"] = control
                for name in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
                    hand[name] = control
                row = Observation(
                    arm, hand,
                    dict(rgb=np.full((32, 32, 3), 128, np.uint8),
                         depth=np.full((32, 32), 600, np.uint16), timestamp_ns=control, ring_sequence=index),
                    None, None, control + 1_000_000, control_timestamp_ns=control,
                )
                writer.append_frame(row, RobotCommand(1, np.zeros(7), np.zeros(12)))
            writer.finish("operator", [], expected_frames=3, technical_status="normal")
        finally:
            writer.close()

    def test_retained_video_range_and_new_export_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            self.make_raw(raw)
            original = RawReader.iter_rgb_frames

            def bad_tail(reader):
                frames = original(reader)
                try:
                    yield next(frames)
                    yield next(frames)
                    raise ValueError("damaged discarded tail")
                finally:
                    frames.close()

            processing = ExportConfig(pointcloud=PointCloudConfig(remove_table=False))
            with patch.object(RawReader, "iter_rgb_frames", bad_tail):
                result = export_zarr(raw, root / "first.zarr", processing=processing)
            export_zarr(raw, root / "second.zarr", processing=processing)
            first = zarr.open_group(str(root / "first.zarr"), mode="r")
            second = zarr.open_group(str(root / "second.zarr"), mode="r")
            self.assertNotEqual(first.attrs["source_id"], second.attrs["source_id"])
            self.assertNotIn("processing_source", first.attrs)
            self.assertFalse(first.attrs["pointcloud_config"]["remove_table"])
            self.assertEqual(first.attrs["hand_pointcloud_config"]["num_points"], processing.hand_pointcloud.num_points)
            self.assertIn("<robot", first.attrs["kinematics_urdf"])
            self.assertEqual(len(first.attrs["fingertip_link_names"]), 5)
            self.assertEqual(first.attrs["depth_scale_m_per_unit"], .001)
            self.assertEqual(result["total_frames"], 2)
            self.assertEqual(result["episode_quality"][0]["trim_reason"], "control_grid_drift")
            self.assertEqual(set(first["data"].keys()), set(second["data"].keys()))
            for name in first["data"]:
                np.testing.assert_array_equal(first["data"][name][:], second["data"][name][:])
                self.assertEqual(first["data"][name].dtype, second["data"][name].dtype)
            np.testing.assert_array_equal(first["meta/episode_ends"][:], [2])

            def bad_prefix(reader):
                yield np.zeros((1, 1, 3), np.uint8)

            with patch.object(RawReader, "iter_rgb_frames", bad_prefix):
                with self.assertRaisesRegex(ValueError, "RGB missing or invalid at row 0"):
                    export_zarr(raw, root / "bad.zarr", processing=processing)
            self.assertFalse((root / "bad.zarr").exists())

    def test_failed_tail_and_float_cast_checks_remain(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = Path(directory) / "raw"
            self.make_raw(raw)
            with h5py.File(raw / "data.h5", "r+") as data:
                data["meta"].attrs["technical_status"] = "failed"
                timing = data["timing/rows"][2]
                for name in timing.dtype.names:
                    if name != "camera_sequence":
                        timing[name] -= 9_000_000
                data["timing/rows"][2] = timing
                data["action_arm_joint_target"][1, 0] = 1e300
            with RawReader(raw) as reader:
                quality = training_prefix(reader)
                self.assertEqual(quality["row_stop"], 2)
                self.assertEqual(quality["trim_reason"], "technical_failure_last_row")
            # 有限 float64 转为 float32 溢出仍应失败，不能因原始检查通过而跳过输出检查。
            with self.assertWarnsRegex(RuntimeWarning, "overflow encountered in cast"):
                with self.assertRaisesRegex(ValueError, "canonical action: nonfinite values at row 1"):
                    export_zarr(raw, Path(directory) / "overflow.zarr",
                                processing=ExportConfig(pointcloud=PointCloudConfig(remove_table=False)))


if __name__ == "__main__":
    unittest.main()

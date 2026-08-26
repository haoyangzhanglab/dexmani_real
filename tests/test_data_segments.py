"""Offline checks for source-contiguous policy episode boundaries."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.data.clean import _deployment_action_limit_masks, _quality_summary
from dexmani_real.data.contracts import EpisodeDecision, OutputProfile, ProcessingConfig
from dexmani_real.data.export import (
    PolicyZarrExportConfig,
    export_processed_hdf5_to_zarr,
    preflight_processed_hdf5_to_zarr,
)
from dexmani_real.data.process import PROCESSED_SCHEMA_NAME, PROCESSED_SCHEMA_VERSION
from dexmani_real.ipc.schema import make_record_sample_dtype
from dexmani_real.recording.frame import decode_record_sample
from dexmani_real.recording.schema import (
    SEMANTIC_META_ATTRS,
    required_dataset_names,
    validate_source_frame_keys,
)
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)
from dexmani_real.teleop.action_proposal import limit_hand_target_delta


class DataSegmentTest(unittest.TestCase):
    def test_decision_maps_source_gap_to_compact_segment_end(self) -> None:
        decision = EpisodeDecision(
            source_path=Path("episode/data.h5"),
            source_frames=6,
            profile=OutputProfile.JOINT,
            selected_indices=np.asarray([0, 1, 4, 5], dtype=np.int64),
            keep_mask=np.asarray([1, 1, 0, 0, 1, 1], dtype=bool),
            drop_reason_bits=np.asarray([0, 0, 1, 1, 0, 0], dtype=np.uint64),
            drop_reason_names=("invalid",),
            hard_reason_counts={"invalid": 2},
            boundary_counts={},
            selected_frames=4,
            quality={},
            source_gap_findings=({"source_row_before": 1, "source_row_after": 4},),
        )

        np.testing.assert_array_equal(decision.segment_ends, [2, 4])
        self.assertEqual(decision.to_dict()["selected_segment_ends"], [2, 4])

    def test_full_windows_never_cross_source_segments(self) -> None:
        arrays = {
            "action_arm": np.zeros((6, 7), dtype=np.float64),
            "action_hand": np.zeros((6, 12), dtype=np.float64),
            "tracking_error": np.zeros(6, dtype=np.float64),
        }

        quality = _quality_summary(
            arrays,
            np.asarray([0, 1, 4, 5], dtype=np.int64),
            grid_dt_s=0.1,
            tracking_error_warn_rad=0.2,
            horizon=3,
            segment_ends=np.asarray([2, 4], dtype=np.int64),
        )

        self.assertEqual(quality["full_window_count"], 0)

    def test_quality_summary_accepts_an_empty_retained_stream(self) -> None:
        arrays = {
            "action_arm": np.zeros((2, 7), dtype=np.float64),
            "action_hand": np.zeros((2, 12), dtype=np.float64),
            "tracking_error": np.zeros(2, dtype=np.float64),
        }

        quality = _quality_summary(
            arrays,
            np.empty(0, dtype=np.int64),
            grid_dt_s=0.1,
            tracking_error_warn_rad=0.2,
            horizon=3,
            segment_ends=np.empty(0, dtype=np.int64),
        )

        self.assertEqual(quality["full_window_count"], 0)

    def test_action_limit_uses_the_final_retained_stream(self) -> None:
        arrays = {
            "source_index": np.asarray([0, 1, 2], dtype=np.int64),
            "timestamp": np.asarray([0.0, 0.1, 0.2], dtype=np.float64),
            "control_arm_qpos": np.zeros((3, 7), dtype=np.float64),
            "control_hand_qpos": np.zeros((3, 12), dtype=np.float64),
            "action_arm": np.zeros((3, 7), dtype=np.float64),
            "action_hand": np.asarray(
                [[0.05] * 12, [0.10] * 12, [0.20] * 12], dtype=np.float64
            ),
        }
        config = ProcessingConfig(
            profile=OutputProfile.JOINT,
            arm_max_delta_rad_per_tick=None,
            hand_max_delta_rad_per_tick=0.1,
        )

        invalid, arm_invalid, hand_invalid = _deployment_action_limit_masks(
            arrays,
            np.asarray([True, False, True]),
            config,
            grid_dt_s=0.1,
        )

        np.testing.assert_array_equal(invalid, [False, False, True])
        np.testing.assert_array_equal(arm_invalid, [False, False, False])
        np.testing.assert_array_equal(hand_invalid, [False, False, True])

    def test_current_raw_schema_requires_policy_observation_fields(self) -> None:
        required = required_dataset_names()

        self.assertIn("policy_observation_arm_qpos", required)
        self.assertIn("policy_observation_hand_qpos", required)
        self.assertIn("hand_accepted_target_action_id", required)
        self.assertEqual(
            SEMANTIC_META_ATTRS["camera_payload_mode"],
            "depth_to_color_aligned_rgbd",
        )

    def test_record_sample_decodes_to_the_v23_writer_contract(self) -> None:
        record = np.zeros(
            1,
            dtype=make_record_sample_dtype(rgb_shape=(1, 1, 3), depth_shape=(1, 1)),
        )
        record["vr_wrist_quat_wxyz"][0, 0] = 1.0

        frame = decode_record_sample(record[0], arm_sent_stream=True)

        self.assertEqual(
            validate_source_frame_keys(set(frame.data), arm_sent_stream=True), ()
        )
        self.assertIn("policy_observation_arm_qpos", frame.data)
        self.assertIn("hand_accepted_target_action_id", frame.data)

    def test_hand_target_limiter_matches_policy_endpoint_bound(self) -> None:
        limited = limit_hand_target_delta(
            np.full(12, 0.5, dtype=np.float64),
            np.zeros(12, dtype=np.float64),
            0.1,
        )

        np.testing.assert_allclose(limited, np.full(12, 0.1))


class PolicyZarrPreflightTest(unittest.TestCase):
    @staticmethod
    def _write_deployment_equivalent_pointcloud(path: Path) -> None:
        pointcloud = PointCloudConfig()
        with h5py.File(path, "w") as source:
            source.attrs.update(
                {
                    "schema_name": PROCESSED_SCHEMA_NAME,
                    "schema_version": PROCESSED_SCHEMA_VERSION,
                    "domain": "real",
                    "profile": OutputProfile.POINTCLOUD.value,
                    "episode_steps": 1,
                    "dt": 0.1,
                    "source_contiguity_tolerance_s": 0.005,
                    "task_name": "test_task",
                    "obs_alignment": "obs[t]_before_action[t]",
                    "observation_reference": "camera_source_monotonic_ns",
                    "state_alignment": "camera_source_aligned_state",
                    "max_observation_skew_s": 0.1,
                    "action_semantics": "deployment_grid_rate_limited_target",
                    "arm_max_delta_rad_per_tick": 0.1,
                    "hand_max_delta_rad_per_tick": 0.1,
                    "deployment_equivalent": True,
                    "contact_force_unit": "sdk_scaled",
                    "contact_force_si_verified": False,
                    "contact_force_frame": "xhand_sensor_native_axes_per_finger",
                    "fingertip_points_frame": "xarm_base",
                    "action_ee_frame": "xarm_base",
                    "processing_config_json": json.dumps(
                        {
                            "pointcloud": pointcloud.to_dict(),
                            "table_plane_abcd": None,
                        },
                        separators=(",", ":"),
                    ),
                    "point_cloud_frame": "xarm_base",
                    "point_cloud_color_source": POINT_CLOUD_COLOR_SOURCE,
                    "point_cloud_policy_id": POINT_CLOUD_POLICY_ID,
                    "point_cloud_config_sha256": pointcloud.sha256,
                    "point_cloud_table_plane_abcd_json": "null",
                    "point_cloud_sampling": POINT_CLOUD_SAMPLING,
                    "point_cloud_transform": POINT_CLOUD_TRANSFORM,
                }
            )
            source.create_dataset("joint_state", data=np.zeros((1, 19), np.float32))
            source.create_dataset("action", data=np.zeros((1, 19), np.float32))
            source.create_dataset("action_ee", data=np.zeros((1, 21), np.float32))
            source.create_dataset("contact_force", data=np.zeros((1, 5, 3), np.float32))
            source.create_dataset(
                "fingertip_points", data=np.zeros((1, 5, 3), np.float32)
            )
            source.create_dataset(
                "point_cloud",
                data=np.zeros((1, pointcloud.num_points, 6), np.float32),
            )
            provenance = source.create_group("provenance")
            provenance.create_dataset("source_segment_ends", data=np.asarray([1]))
            provenance.create_dataset("source_row_index", data=np.asarray([0]))
            provenance.create_dataset("source_sample_index", data=np.asarray([0]))
            provenance.create_dataset("source_timestamp_s", data=np.asarray([0.0]))

    def test_preflight_validates_inputs_without_creating_an_output_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)

            report = preflight_processed_hdf5_to_zarr(
                root,
                PolicyZarrExportConfig(expected_task_name="test_task"),
            )

            self.assertEqual(report["source_file_count"], 1)
            self.assertEqual(report["episode_count"], 1)
            self.assertEqual(report["total_frames"], 1)
            self.assertEqual(report["episode_ends"], [1])
            self.assertFalse(any(root.glob("*.zarr")))

    def test_export_cli_dry_run_reports_preflight_without_writing_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "examples/export_policy_zarr.py",
                    "--input-root",
                    str(root),
                    "--task-name",
                    "test_task",
                    "--dry-run",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(json.loads(completed.stdout)["dry_run"])
            self.assertFalse(any(root.glob("*.zarr")))

    def test_export_publishes_the_validated_pointcloud_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            output_path = root / "test_task.zarr"
            self._write_deployment_equivalent_pointcloud(source_path)

            report = export_processed_hdf5_to_zarr(
                root,
                output_path,
                PolicyZarrExportConfig(expected_task_name="test_task"),
            )

            self.assertEqual(report["task_name"], "test_task")
            self.assertEqual(report["total_frames"], 1)
            self.assertTrue(output_path.is_dir())

    def test_preflight_rejects_nonfinite_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)
            with h5py.File(source_path, "r+") as source:
                source["action"][0, 0] = np.nan

            with self.assertRaisesRegex(ValueError, "action contains NaN/Inf"):
                preflight_processed_hdf5_to_zarr(root)

    def test_preflight_rejects_a_non_pointcloud_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)
            with h5py.File(source_path, "r+") as source:
                source.attrs["profile"] = OutputProfile.JOINT.value

            with self.assertRaisesRegex(ValueError, "requires a pointcloud"):
                preflight_processed_hdf5_to_zarr(root)


if __name__ == "__main__":
    unittest.main()

"""Offline checks for source-contiguous policy episode boundaries."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import zarr

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.config.runtime import resolve_runtime_config
from dexmani_real.data.clean import (
    _deployment_action_limit_masks,
    _isolated_tactile_forward_fill_mask,
    _quality_summary,
    _revalidate_camera_duplicates,
    _transient_ik_hold_masks,
    observation_skew_valid_mask,
    recompute_observation_skew_s,
)
from dexmani_real.data.contracts import (
    EpisodeDecision,
    OutputProfile,
    ProcessingConfig,
    QualityPolicy,
    TemporalQualityConfig,
)
from dexmani_real.data.export import (
    PolicyZarrExportConfig,
    export_processed_hdf5_to_zarr,
    preflight_processed_hdf5_to_zarr,
)
from dexmani_real.data.process import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    _invalid_frames_report,
)
from dexmani_real.data.quality import assess_temporal_quality
from dexmani_real.ipc.schema import make_record_sample_dtype
from dexmani_real.planning.poses import (
    rot6d_to_quat_wxyz,
    validate_canonical_rot6d,
    validate_rot6d_geometry,
)
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
    def test_ik_hold_threshold_keeps_up_to_four_frames(self) -> None:
        for length in (1, 2, 3, 4, 5):
            with self.subTest(length=length):
                held = np.ones(length, dtype=bool)
                status = np.full(length, 2, dtype=np.int64)

                transient, persistent = _transient_ik_hold_masks(held, status)

                self.assertEqual(
                    int(np.count_nonzero(transient)), length if length <= 4 else 0
                )
                self.assertEqual(
                    int(np.count_nonzero(persistent)), length if length > 4 else 0
                )

    def test_tactile_forward_fill_repairs_only_one_bracketed_frame(self) -> None:
        np.testing.assert_array_equal(
            _isolated_tactile_forward_fill_mask([True, False, True]),
            [False, True, False],
        )
        np.testing.assert_array_equal(
            _isolated_tactile_forward_fill_mask([True, False, False, True]),
            [False, False, False, False],
        )
        np.testing.assert_array_equal(
            _isolated_tactile_forward_fill_mask([False, True, True]),
            [False, False, False],
        )

    def test_camera_duplicate_revalidation_requires_advancing_trusted_frames(
        self,
    ) -> None:
        arrays = {
            "source_index": np.asarray([10, 11, 12]),
            "timestamp": np.asarray([0.0, 0.1, 0.2]),
            "camera_duplicate": np.asarray([False, True, True]),
            "camera_clock_reset": np.zeros(3, dtype=bool),
            "camera_generation": np.ones(3, dtype=np.int64),
            "camera_depth_frame_number": np.asarray([100, 102, 104]),
            "camera_color_frame_number": np.asarray([200, 202, 204]),
            "camera_source_monotonic_ns": np.asarray([1000, 1100, 1200]),
            "camera_depth_device_timestamp_s": np.asarray([1.0, 1.1, 1.2]),
            "camera_color_device_timestamp_s": np.asarray([1.0, 1.1, 1.2]),
        }

        recovered = _revalidate_camera_duplicates(
            arrays,
            np.asarray([True, False, False]),
            np.ones(3, dtype=bool),
            grid_dt_s=0.1,
            grid_dt_relative_tolerance=0.05,
        )

        np.testing.assert_array_equal(recovered, [False, True, True])
        arrays["camera_depth_frame_number"][1] = 100
        rejected = _revalidate_camera_duplicates(
            arrays,
            np.asarray([True, False, False]),
            np.ones(3, dtype=bool),
            grid_dt_s=0.1,
            grid_dt_relative_tolerance=0.05,
        )
        np.testing.assert_array_equal(rejected, [False, False, False])

    def test_invalid_report_counts_overlapping_reasons_once(self) -> None:
        decision = EpisodeDecision(
            source_path=Path("episode_x"),
            source_frames=3,
            profile=OutputProfile.JOINT,
            selected_indices=np.asarray([0, 2], dtype=np.int64),
            keep_mask=np.asarray([True, False, True]),
            drop_reason_bits=np.asarray([0, 3, 0], dtype=np.uint64),
            drop_reason_names=("reason_a", "reason_b"),
            hard_reason_counts={"reason_a": 1, "reason_b": 1},
            boundary_counts={},
            selected_frames=2,
            quality={},
            hard_invalid_reason_names=("reason_a", "reason_b"),
        )

        report = _invalid_frames_report([decision], task_name="task")

        self.assertEqual(report["episodes"][0]["invalid_frame_count"], 1)
        self.assertEqual(report["episodes"][0]["invalid_ranges"], [[1, 2]])
        self.assertEqual(len(report["episodes"][0]["reasons"]), 2)

    def test_endpoint_tolerance_round_trips_from_runtime_to_processing(self) -> None:
        runtime = resolve_runtime_config(
            data={"policy": {"endpoint_delta_tolerance_rad": 0.0}}
        )
        config = ProcessingConfig.from_runtime(runtime, profile=OutputProfile.JOINT)

        self.assertEqual(config.endpoint_delta_tolerance_rad, 0.0)
        self.assertEqual(config.to_dict()["endpoint_delta_tolerance_rad"], 0.0)
        with self.assertRaises(ValueError):
            resolve_runtime_config(
                data={"policy": {"endpoint_delta_tolerance_rad": -1e-12}}
            )

    def test_observation_skew_recomputes_from_valid_source_timestamps(self) -> None:
        source_timestamps_ns = np.asarray(
            [[100, 200, 0, 300], [0, 0, 0, 0]], dtype=np.int64
        )
        valid_mask = np.asarray(
            [[True, True, False, True], [False, False, False, False]], dtype=bool
        )

        np.testing.assert_allclose(
            recompute_observation_skew_s(source_timestamps_ns, valid_mask),
            [2e-7, 0.0],
        )

    def test_observation_skew_rejects_real_camera_gap_excess(self) -> None:
        valid_mask = np.ones((1, 4), dtype=bool)
        camera_gap_timestamps_ns = np.asarray(
            [[1_000_000_000, 1_000_000_000, 1_000_000_000, 1_200_000_000]],
            dtype=np.int64,
        )
        self.assertFalse(
            observation_skew_valid_mask(
                np.asarray([0.2]),
                camera_gap_timestamps_ns,
                valid_mask,
                max_observation_skew_s=0.1,
            )[0]
        )

        normal_timestamps_ns = camera_gap_timestamps_ns.copy()
        normal_timestamps_ns[0, 3] = 1_050_000_000
        np.testing.assert_array_equal(
            observation_skew_valid_mask(
                np.asarray([0.05]),
                normal_timestamps_ns,
                valid_mask,
                max_observation_skew_s=0.1,
            ),
            [True],
        )
        # A stale aggregate cannot hide the same source-time gap even if it is
        # numerically within the configured bound.
        self.assertFalse(
            observation_skew_valid_mask(
                np.asarray([0.05]),
                camera_gap_timestamps_ns,
                valid_mask,
                max_observation_skew_s=0.1,
            )[0]
        )

    def test_stall_window_uses_inclusive_sample_count_and_last_window(self) -> None:
        window = 4
        config = TemporalQualityConfig(
            policy=QualityPolicy.AUDIT,
            stall_window_frames=window,
        )

        def assess(action_arm: np.ndarray, break_before: np.ndarray | None = None):
            frame_count = len(action_arm)
            return assess_temporal_quality(
                {
                    "action_arm": action_arm,
                    "action_hand": np.zeros((frame_count, 12)),
                    "arm_qpos": np.zeros((frame_count, 7)),
                    "tracking_error": np.ones(frame_count),
                    "arm_last_cmd_seq": np.zeros(frame_count, dtype=np.int64),
                },
                np.ones(frame_count, dtype=bool),
                (
                    np.zeros(frame_count, dtype=bool)
                    if break_before is None
                    else break_before
                ),
                config,
                tracking_error_warn_rad=0.2,
            )

        too_short = assess(np.asarray([[0.0]] * (window - 1)) * np.ones((1, 7)))
        exact = assess(np.asarray([[0.0], [0.0], [0.0], [0.3]]) * np.ones((1, 7)))
        last_window_only = assess(
            np.asarray([[0.0], [0.0], [0.0], [0.0], [0.3]]) * np.ones((1, 7))
        )
        cross_segment = assess(
            np.asarray([[0.0], [0.0], [0.0], [0.3], [0.3], [0.3]]) * np.ones((1, 7)),
            np.asarray([False, False, False, True, False, False]),
        )

        self.assertFalse(np.any(too_short.suspect_masks["arm_feedback_stall"]))
        np.testing.assert_array_equal(
            exact.suspect_masks["arm_feedback_stall"],
            [False, True, True, True],
        )
        np.testing.assert_array_equal(
            last_window_only.suspect_masks["arm_feedback_stall"],
            [False, False, True, True, True],
        )
        self.assertFalse(np.any(cross_segment.suspect_masks["arm_feedback_stall"]))

    def test_stall_window_must_contain_at_least_two_samples(self) -> None:
        with self.assertRaises(ValueError):
            TemporalQualityConfig(stall_window_frames=1)

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
                [[0.05] * 12, [0.10] * 12, [0.40] * 12], dtype=np.float64
            ),
        }
        config = ProcessingConfig(
            profile=OutputProfile.JOINT,
            arm_max_delta_rad_per_tick=None,
            hand_max_delta_rad_per_tick=0.3,
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

    def test_record_sample_decodes_to_the_current_writer_contract(self) -> None:
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
            0.3,
        )

        np.testing.assert_allclose(limited, np.full(12, 0.3))


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
                    "source_frames": 1,
                    "dt": 0.1,
                    "source_contiguity_tolerance_s": 0.005,
                    "source_contiguity": "segment_ends_in_provenance",
                    "task_name": "test_task",
                    "obs_alignment": "obs[t]_before_action[t]",
                    "observation_reference": "camera_source_monotonic_ns",
                    "state_alignment": "camera_source_aligned_state",
                    "max_observation_skew_s": 0.1,
                    "action_semantics": "deployment_grid_rate_limited_target",
                    "arm_max_delta_rad_per_tick": 0.1,
                    "hand_max_delta_rad_per_tick": 0.3,
                    "endpoint_delta_tolerance_rad": 1e-12,
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
                    "source_decision_json": json.dumps(
                        {
                            "accepted": True,
                            "profile": OutputProfile.POINTCLOUD.value,
                            "rejected_reason": None,
                            "source_frames": 1,
                            "selected_frames": 1,
                            "dropped_frames": 0,
                            "hard_invalid_reason_names": [],
                            "selected_source_ranges": [[0, 1]],
                            "selected_segment_ends": [1],
                        },
                        separators=(",", ":"),
                    ),
                    "source_member_sha256_json": json.dumps(
                        {
                            "data.h5": "0" * 64,
                            "depth.h5": "1" * 64,
                            "rgb.mp4": "2" * 64,
                        },
                        separators=(",", ":"),
                    ),
                }
            )
            source.create_dataset("joint_state", data=np.zeros((1, 19), np.float32))
            source.create_dataset("action", data=np.zeros((1, 19), np.float32))
            action_ee = np.zeros((1, 21), np.float32)
            action_ee[0, 3:9] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
            source.create_dataset("action_ee", data=action_ee)
            source.create_dataset("contact_force", data=np.zeros((1, 5, 3), np.float32))
            source.create_dataset(
                "fingertip_points", data=np.zeros((1, 5, 3), np.float32)
            )
            point_cloud = np.zeros((1, pointcloud.num_points, 6), np.float32)
            point_cloud[..., 0] = 0.1
            point_cloud[..., 2] = 0.4
            point_cloud[..., 3:] = 0.5
            source.create_dataset("point_cloud", data=point_cloud)
            provenance = source.create_group("provenance")
            provenance.attrs["drop_reason_bit_names_json"] = "{}"
            provenance.create_dataset(
                "source_segment_ends", data=np.asarray([1], dtype=np.int64)
            )
            provenance.create_dataset(
                "source_row_index", data=np.asarray([0], dtype=np.int64)
            )
            provenance.create_dataset(
                "source_sample_index", data=np.asarray([0], dtype=np.int64)
            )
            provenance.create_dataset(
                "source_timestamp_s", data=np.asarray([0.0], dtype=np.float64)
            )
            provenance.create_dataset(
                "source_keep_mask", data=np.asarray([True], dtype=bool)
            )
            provenance.create_dataset(
                "source_drop_reason_bits", data=np.asarray([0], dtype=np.uint64)
            )

    @classmethod
    def _write_deployment_equivalent_rgbpc(cls, path: Path) -> None:
        cls._write_deployment_equivalent_pointcloud(path)
        with h5py.File(path, "r+") as source:
            decision = json.loads(str(source.attrs["source_decision_json"]))
            decision["profile"] = OutputProfile.RGB_PC.value
            source.attrs.update(
                {
                    "profile": OutputProfile.RGB_PC.value,
                    "source_decision_json": json.dumps(decision, separators=(",", ":")),
                    "depth_scale_m_per_unit": 0.001,
                    "depth_invalid_value": 0,
                    "camera_extrinsic_semantics": (
                        "T_xarm_base_from_color;native_color_optical_to_xarm_base"
                    ),
                }
            )
            source.create_dataset(
                "rgb", data=np.full((1, 2, 3, 3), 128, dtype=np.uint8)
            )
            source.create_dataset(
                "depth", data=np.full((1, 2, 3), 1000, dtype=np.uint16)
            )
            source.create_dataset(
                "camera_intrinsic",
                data=np.asarray(
                    [[1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
            )
            source.create_dataset(
                "camera_extrinsic", data=np.eye(4, dtype=np.float32)[None]
            )

    @staticmethod
    def _mark_one_source_row_removed(path: Path) -> None:
        with h5py.File(path, "r+") as source:
            source.attrs["source_frames"] = 2
            decision = json.loads(str(source.attrs["source_decision_json"]))
            decision.update(
                {
                    "source_frames": 2,
                    "dropped_frames": 1,
                    "hard_invalid_reason_names": ["long_ik_failure_hold"],
                    "hard_invalid_frame_count": 1,
                    "hard_invalid_ranges": [[1, 2]],
                }
            )
            source.attrs["source_decision_json"] = json.dumps(
                decision, separators=(",", ":")
            )
            provenance = source["provenance"]
            provenance.attrs["drop_reason_bit_names_json"] = json.dumps(
                {"0": "long_ik_failure_hold"}, separators=(",", ":")
            )
            del provenance["source_keep_mask"]
            provenance.create_dataset(
                "source_keep_mask", data=np.asarray([True, False], dtype=bool)
            )
            del provenance["source_drop_reason_bits"]
            provenance.create_dataset(
                "source_drop_reason_bits", data=np.asarray([0, 1], dtype=np.uint64)
            )

    @staticmethod
    def _load_processed_visualizer() -> Any:
        visualizer_path = (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "visualize_episode_processed.py"
        )
        spec = importlib.util.spec_from_file_location(
            "dexmani_visualize_episode_processed_test", visualizer_path
        )
        if spec is None or spec.loader is None:
            raise AssertionError("could not load processed visualizer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

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

    def test_preflight_rejects_previous_processed_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)
            with h5py.File(source_path, "r+") as source:
                source.attrs["schema_version"] = 10

            with self.assertRaisesRegex(ValueError, "unsupported processed schema version"):
                preflight_processed_hdf5_to_zarr(root)

    def test_export_rejects_incomplete_episode_without_splitting_others(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_path = root / "episode_valid.h5"
            rejected_path = root / "episode_rejected.h5"
            output_path = root / "test_task.zarr"
            self._write_deployment_equivalent_pointcloud(valid_path)
            self._write_deployment_equivalent_pointcloud(rejected_path)
            self._mark_one_source_row_removed(rejected_path)

            report = export_processed_hdf5_to_zarr(root, output_path)

            self.assertEqual(report["source_file_count"], 2)
            self.assertEqual(report["episode_count"], 1)
            self.assertEqual(report["rejected_episode_count"], 1)
            self.assertEqual(report["episode_ends"], [1])
            self.assertEqual(report["rejected_episodes"][0]["invalid_ranges"], [[1, 2]])
            zarr_root = zarr.open_group(str(output_path), mode="r")
            np.testing.assert_array_equal(zarr_root["meta"]["episode_ends"][:], [1])

    def test_export_cli_reports_and_fails_when_all_episodes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root / "episodes_processed" / "test_task"
            task_root.mkdir(parents=True)
            source_path = task_root / "episode_rejected.h5"
            self._write_deployment_equivalent_pointcloud(source_path)
            self._mark_one_source_row_removed(source_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "examples/export_policy_zarr.py",
                    str(task_root),
                    "--dry-run",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 1, completed.stderr)
            self.assertIn("REJECT episode_rejected", completed.stderr)
            self.assertIn("long_ik_failure_hold", completed.stderr)
            self.assertIn("Exported 0/1 episode(s)", completed.stderr)
            self.assertFalse(any(root.glob("*.zarr")))

    def test_export_cli_dry_run_shows_progress_without_writing_zarr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task_root = root / "episodes_processed" / "test_task"
            task_root.mkdir(parents=True)
            source_path = task_root / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)

            completed = subprocess.run(
                [
                    sys.executable,
                    "examples/export_policy_zarr.py",
                    str(task_root),
                    "--dry-run",
                ],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stdout, "")
            self.assertIn("validate processed episodes", completed.stderr)
            self.assertFalse(any(root.glob("*.zarr")))

    def test_export_publishes_the_validated_pointcloud_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            output_path = root / "test_task.zarr"
            self._write_deployment_equivalent_pointcloud(source_path)
            progress_events: list[tuple[str, int, int]] = []

            report = export_processed_hdf5_to_zarr(
                root,
                output_path,
                PolicyZarrExportConfig(expected_task_name="test_task"),
                progress_callback=lambda phase, completed, total: progress_events.append(
                    (phase, completed, total)
                ),
            )

            self.assertEqual(report["task_name"], "test_task")
            self.assertEqual(report["total_frames"], 1)
            self.assertTrue(output_path.is_dir())
            self.assertEqual(progress_events[0], ("validate", 0, 1))
            self.assertEqual(progress_events[-1][0], "verify")
            self.assertEqual(progress_events[-1][1], progress_events[-1][2])

    def test_preflight_rejects_nonfinite_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)
            with h5py.File(source_path, "r+") as source:
                source["action"][0, 0] = np.nan

            with self.assertRaisesRegex(ValueError, "action contains NaN/Inf"):
                preflight_processed_hdf5_to_zarr(root)

    def test_export_preflight_rejects_missing_rgbd_dataset_before_staging(self) -> None:
        for missing_key in ("rgb", "depth"):
            with (
                self.subTest(missing_key=missing_key),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source_path = root / "episode.h5"
                output_path = root / "out.zarr"
                self._write_deployment_equivalent_rgbpc(source_path)
                with h5py.File(source_path, "r+") as source:
                    del source[missing_key]

                with self.assertRaisesRegex(ValueError, "processed data keys"):
                    export_processed_hdf5_to_zarr(root, output_path)
                self.assertFalse(output_path.exists())
                self.assertFalse(any(root.glob(".out.zarr.tmp-*")))

    def test_export_preflight_rejects_rgbd_shape_mismatch_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            output_path = root / "out.zarr"
            self._write_deployment_equivalent_rgbpc(source_path)
            with h5py.File(source_path, "r+") as source:
                del source["depth"]
                source.create_dataset(
                    "depth", data=np.full((1, 3, 3), 1000, dtype=np.uint16)
                )

            with self.assertRaisesRegex(ValueError, "rgb/depth spatial shape mismatch"):
                export_processed_hdf5_to_zarr(root, output_path)
            self.assertFalse(output_path.exists())
            self.assertFalse(any(root.glob(".out.zarr.tmp-*")))

    def test_processed_info_rejects_malformed_payload_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)
            with h5py.File(source_path, "r+") as source:
                source["action"][0, 0] = np.nan

            visualizer = self._load_processed_visualizer()
            with self.assertRaisesRegex(ValueError, "action contains NaN/Inf"):
                visualizer.print_episode_info(str(source_path))

    def test_processed_info_rejects_malformed_provenance_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)
            with h5py.File(source_path, "r+") as source:
                source.attrs["source_decision_json"] = json.dumps(
                    {"accepted": False}, separators=(",", ":")
                )

            visualizer = self._load_processed_visualizer()
            with self.assertRaisesRegex(ValueError, "source_decision_json"):
                visualizer.print_episode_info(str(source_path))

    def test_preflight_accepts_zero_endpoint_tolerance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)
            with h5py.File(source_path, "r+") as source:
                source.attrs["endpoint_delta_tolerance_rad"] = 0.0

            report = preflight_processed_hdf5_to_zarr(root)

            self.assertEqual(report["total_frames"], 1)

    def test_export_admission_rejects_pointcloud_rgb_and_zero_xyz(self) -> None:
        for field, mutate, message in (
            (
                "rgb",
                lambda source: source["point_cloud"].__setitem__((0, 0, 3), 1.1),
                "point-cloud RGB outside",
            ),
            (
                "xyz",
                lambda source: source["point_cloud"].__setitem__(
                    (0, slice(None), slice(0, 3)), 0.0
                ),
                "all-zero point-cloud frame",
            ),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source_path = root / "episode.h5"
                output_path = root / "out.zarr"
                self._write_deployment_equivalent_pointcloud(source_path)
                with h5py.File(source_path, "r+") as source:
                    mutate(source)

                with self.assertRaisesRegex(ValueError, message):
                    export_processed_hdf5_to_zarr(root, output_path)
                self.assertFalse(output_path.exists())
                self.assertFalse(any(root.glob(".out.zarr.tmp-*")))

    def test_export_admission_rejects_persisted_workspace_escape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)
            with h5py.File(source_path, "r+") as source:
                source["point_cloud"][0, 0, 0] = 0.9

            with self.assertRaisesRegex(ValueError, "persisted workspace"):
                preflight_processed_hdf5_to_zarr(root)

    def test_export_admission_rejects_noncanonical_action_ee(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "episode.h5"
            self._write_deployment_equivalent_pointcloud(source_path)
            with h5py.File(source_path, "r+") as source:
                source["action_ee"][0, 3] = 2.0

            with self.assertRaisesRegex(ValueError, "canonical"):
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


class Rot6DValidationTest(unittest.TestCase):
    def test_model_geometry_accepts_scaled_nonunit_columns(self) -> None:
        values = np.asarray([2.0, 0.0, 0.0, 0.0, 3.0, 0.0])
        validate_rot6d_geometry(values)
        np.testing.assert_allclose(rot6d_to_quat_wxyz(values), [1.0, 0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_canonical_rot6d(values)

    def test_model_geometry_rejects_zero_and_near_collinear_columns(self) -> None:
        for values in (
            np.zeros(6),
            np.asarray([1.0, 0.0, 0.0, 1.0, 1e-8, 0.0]),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    validate_rot6d_geometry(values)
                with self.assertRaises(ValueError):
                    rot6d_to_quat_wxyz(values)


if __name__ == "__main__":
    unittest.main()

"""Focused offline tests for processed multimodal and Policy Zarr boundaries."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.data.clean import align_tactile_sum_rows_to_references
from dexmani_real.data.contracts import OutputProfile, ProcessingConfig
from dexmani_real.data.export import (
    PolicyZarrExportConfig,
    _Artifact,
    _inspect_artifact,
)
from dexmani_real.data.process import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    _validate_processed_output_structure,
    compute_fingertip_history_xarm_base,
    validate_processed_hdf5,
)
from dexmani_real.planning.fingertip import compute_fingertip_points_xarm_base
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)


class _ArmFk:
    def compute(self, qpos):
        return np.asarray(qpos[:3]), np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])


class _HandFk:
    def is_ready(self):
        return True

    def compute_tip_positions_in_handbase(self, qpos):
        return np.broadcast_to(np.asarray(qpos[:3]), (5, 3)).copy()


class OfflineMultimodalTest(unittest.TestCase):
    def test_tactile_alignment_rejects_future_and_over_skew(self) -> None:
        contact = np.zeros((3, 5, 3), dtype=np.float64)
        proven = np.ones(3, dtype=bool)
        unit = np.zeros(3, dtype=np.int64)
        selected = align_tactile_sum_rows_to_references(
            contact,
            np.array([110, 90, 150], dtype=np.int64),
            np.array([110, 90, 150], dtype=np.int64),
            proven,
            proven,
            unit,
            np.array([100, 161, 210], dtype=np.int64),
            max_observation_skew_s=50e-9,
        )
        np.testing.assert_array_equal(selected, [-1, -1, -1])
        mismatch = align_tactile_sum_rows_to_references(
            contact[:1],
            np.array([89], dtype=np.int64),
            np.array([90], dtype=np.int64),
            proven[:1],
            proven[:1],
            unit[:1],
            np.array([100], dtype=np.int64),
            max_observation_skew_s=50e-9,
        )
        np.testing.assert_array_equal(mismatch, [-1])
        wrong_unit = align_tactile_sum_rows_to_references(
            contact[:1],
            np.array([90], dtype=np.int64),
            np.array([90], dtype=np.int64),
            proven[:1],
            proven[:1],
            np.array([1], dtype=np.int64),
            np.array([100], dtype=np.int64),
            max_observation_skew_s=50e-9,
        )
        np.testing.assert_array_equal(wrong_unit, [-1])

    def test_tactile_alignment_handles_nonmonotonic_sources_and_boundaries(
        self,
    ) -> None:
        contact = np.zeros((4, 5, 3), dtype=np.float64)
        source = np.array([80, 60, 75, 90], dtype=np.int64)
        selected = align_tactile_sum_rows_to_references(
            contact,
            source,
            source,
            np.ones(4, dtype=bool),
            np.ones(4, dtype=bool),
            np.zeros(4, dtype=np.int64),
            np.array([80, 70, 80, 100], dtype=np.int64),
            max_observation_skew_s=10e-9,
        )
        np.testing.assert_array_equal(selected, [0, 1, 0, 3])

    def test_forward_fill_never_uses_future_or_out_of_skew_row(self) -> None:
        contact = np.arange(45, dtype=np.float64).reshape(3, 5, 3)
        selected = align_tactile_sum_rows_to_references(
            contact,
            np.array([90, 999, 150], dtype=np.int64),
            np.array([90, 999, 150], dtype=np.int64),
            np.ones(3, dtype=bool),
            np.ones(3, dtype=bool),
            np.zeros(3, dtype=np.int64),
            np.array([100, 160, 210], dtype=np.int64),
            max_observation_skew_s=70e-9,
        )
        np.testing.assert_array_equal(selected, [0, 0, 2])
        selected = align_tactile_sum_rows_to_references(
            contact,
            np.array([90, 999, 150], dtype=np.int64),
            np.array([90, 999, 150], dtype=np.int64),
            np.ones(3, dtype=bool),
            np.ones(3, dtype=bool),
            np.zeros(3, dtype=np.int64),
            np.array([100, 161, 221], dtype=np.int64),
            max_observation_skew_s=70e-9,
        )
        np.testing.assert_array_equal(selected, [0, -1, -1])

    def test_camera_aligned_fingertip_history_matches_shared_helper(self) -> None:
        arm = np.arange(14, dtype=np.float64).reshape(2, 7) / 100.0
        hand = np.arange(24, dtype=np.float64).reshape(2, 12) / 100.0
        kwargs = {
            "arm_fk": _ArmFk(),
            "hand_fk": _HandFk(),
            "handbase_position_eef_m": np.array([0.1, 0.0, 0.0]),
            "handbase_quat_eef_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
        }
        history = compute_fingertip_history_xarm_base(arm, hand, **kwargs)
        expected = np.stack(
            [
                compute_fingertip_points_xarm_base(arm[index], hand[index], **kwargs)
                for index in range(2)
            ]
        ).astype(np.float32)
        np.testing.assert_array_equal(history, expected)

    def test_policy_zarr_v5_admits_all_four_processed_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile in OutputProfile:
                with self.subTest(profile=profile.value):
                    path = root / f"{profile.value}.h5"
                    self._write_processed_fixture(path, profile)
                    with (
                        patch("dexmani_real.data.export.validate_processed_payload"),
                        patch(
                            "dexmani_real.data.export.validate_processed_provenance",
                            return_value=object(),
                        ),
                        patch(
                            "dexmani_real.data.export._whole_episode_rejection",
                            return_value=None,
                        ),
                    ):
                        artifact = _inspect_artifact(path, PolicyZarrExportConfig())
                    self.assertIsInstance(artifact, _Artifact)
                    self.assertEqual(artifact.profile, profile)
                    self.assertEqual(
                        set(artifact.dataset_shapes), set(profile.dataset_keys)
                    )
                    self.assertEqual(
                        artifact.semantic_attrs["fingertip_points_unit"], "m"
                    )

    def test_export_rejects_non_boolean_and_non_integer_semantic_attrs(self) -> None:
        cases = (
            ("deployment_equivalent", "false"),
            ("deployment_equivalent", 0),
            ("contact_force_unit_code", 0.9),
            ("contact_force_si_verified", None),
            ("contact_force_si_verified", True),
            ("contact_force_unit_code", None),
            ("contact_force_unit", "N"),
            ("contact_force_frame", "world"),
            ("fingertip_points_frame", "handbase"),
            ("fingertip_points_unit", "mm"),
            ("action_ee_frame", "world"),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (name, value) in enumerate(cases):
                with self.subTest(name=name, value=value):
                    path = root / f"malformed_export_{index}.h5"
                    self._write_processed_fixture(path, OutputProfile.JOINT)
                    with h5py.File(path, "r+") as output:
                        if value is None:
                            del output.attrs[name]
                        else:
                            output.attrs[name] = value
                    with (
                        patch("dexmani_real.data.export.validate_processed_payload"),
                        patch(
                            "dexmani_real.data.export.validate_processed_provenance",
                            return_value=object(),
                        ),
                        patch(
                            "dexmani_real.data.export._whole_episode_rejection",
                            return_value=None,
                        ),
                    ):
                        with self.assertRaises(ValueError):
                            _inspect_artifact(path, PolicyZarrExportConfig())

    def test_process_rejects_non_boolean_and_non_integer_semantic_attrs(self) -> None:
        cases = (
            ("deployment_equivalent", "false"),
            ("deployment_equivalent", 0),
            ("contact_force_unit_code", 0.9),
            ("contact_force_si_verified", None),
            ("contact_force_si_verified", True),
            ("contact_force_unit_code", None),
            ("contact_force_unit", "N"),
            ("contact_force_frame", "world"),
            ("fingertip_points_frame", "handbase"),
            ("fingertip_points_unit", "mm"),
            ("action_ee_frame", "world"),
        )
        config = ProcessingConfig(
            profile=OutputProfile.JOINT,
            horizon=1,
            min_full_windows=1,
            arm_max_delta_rad_per_tick=0.1,
            hand_max_delta_rad_per_tick=0.3,
            endpoint_delta_tolerance_rad=1e-6,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, (name, value) in enumerate(cases):
                with self.subTest(name=name, value=value):
                    path = root / f"malformed_process_{index}.h5"
                    self._write_processed_fixture(path, OutputProfile.JOINT)
                    with h5py.File(path, "r+") as output:
                        if value is None:
                            del output.attrs[name]
                        else:
                            output.attrs[name] = value
                    with patch("dexmani_real.data.process.validate_processed_payload"):
                        with self.assertRaises(ValueError):
                            validate_processed_hdf5(path, config)

    def test_writer_structural_sanity_skips_payload_scan_but_full_verify_does_not(
        self,
    ) -> None:
        config = ProcessingConfig(
            profile=OutputProfile.JOINT,
            horizon=1,
            min_full_windows=1,
            arm_max_delta_rad_per_tick=0.1,
            hand_max_delta_rad_per_tick=0.3,
            endpoint_delta_tolerance_rad=1e-6,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.h5"
            self._write_processed_fixture(path, OutputProfile.JOINT)
            with h5py.File(path, "r+") as output:
                output["action"][0, 0] = np.nan
            sanity = _validate_processed_output_structure(path, config)
            self.assertEqual(sanity["level"], "structural")
            with self.assertRaisesRegex(ValueError, "action contains NaN/Inf"):
                validate_processed_hdf5(path, config)

    def _write_processed_fixture(self, path: Path, profile: OutputProfile) -> None:
        visual = profile.needs_rgb or profile.needs_pointcloud
        pointcloud = PointCloudConfig()
        with h5py.File(path, "w") as output:
            shapes = {
                "joint_state": ((1, 19), np.float32),
                "action": ((1, 19), np.float32),
                "action_ee": ((1, 21), np.float32),
                "contact_force": ((1, 5, 3), np.float32),
                "fingertip_points": ((1, 5, 3), np.float32),
                "rgb": ((1, 2, 3, 3), np.uint8),
                "depth": ((1, 2, 3), np.uint16),
                "camera_intrinsic": ((1, 9), np.float32),
                "camera_extrinsic": ((1, 4, 4), np.float32),
                "point_cloud": ((1, pointcloud.num_points, 6), np.float32),
            }
            for key in profile.dataset_keys:
                shape, dtype = shapes[key]
                output.create_dataset(
                    key,
                    data=np.zeros(shape, dtype=dtype),
                    compression="gzip",
                    compression_opts=4,
                )
            output.create_group("provenance")
            output.attrs.update(
                {
                    "schema_name": PROCESSED_SCHEMA_NAME,
                    "schema_version": PROCESSED_SCHEMA_VERSION,
                    "domain": "real",
                    "profile": profile.value,
                    "episode_steps": 1,
                    "dt": 0.0625,
                    "task_name": "pick",
                    "obs_alignment": "obs[t]_before_action[t]",
                    "observation_reference": (
                        "camera_source_monotonic_ns"
                        if visual
                        else "grid_anchor_monotonic_ns"
                    ),
                    "state_alignment": (
                        "camera_source_aligned_state"
                        if visual
                        else "control_grid_state"
                    ),
                    "max_observation_skew_s": 0.1,
                    "action_semantics": "deployment_grid_rate_limited_target",
                    "arm_max_delta_rad_per_tick": 0.1,
                    "hand_max_delta_rad_per_tick": 0.3,
                    "endpoint_delta_tolerance_rad": 1e-6,
                    "deployment_equivalent": True,
                    "contact_force_unit": "sdk_scaled_unknown_si",
                    "contact_force_si_verified": False,
                    "contact_force_frame": "xhand_sensor_native_axes_per_finger",
                    "contact_force_source": (
                        "camera_causal_tactile_sum"
                        if visual
                        else "control_grid_tactile_sum"
                    ),
                    "contact_force_alignment": (
                        "newest_source_not_after_camera_within_max_observation_skew"
                        if visual
                        else "newest_source_not_after_grid_within_max_observation_skew"
                    ),
                    "contact_force_fresh_required": True,
                    "contact_force_calibrated_required": True,
                    "contact_force_unit_code": 0,
                    "contact_force_causal_to_reference": True,
                    "contact_force_hand_source_match_required": True,
                    "fingertip_points_frame": "xarm_base",
                    "fingertip_points_unit": "m",
                    "action_ee_frame": "xarm_base",
                }
            )
            if profile.needs_rgb:
                output.attrs.update(
                    {
                        "depth_scale_m_per_unit": 0.001,
                        "depth_invalid_value": 0,
                        "camera_extrinsic_semantics": "T_xarm_base_from_color;native_color_optical_to_xarm_base",
                    }
                )
            if profile.needs_pointcloud:
                processing = {
                    "pointcloud": pointcloud.to_dict(),
                    "table_plane_abcd": None,
                }
                output.attrs.update(
                    {
                        "processing_config_json": json.dumps(processing),
                        "point_cloud_frame": "xarm_base",
                        "point_cloud_color_source": POINT_CLOUD_COLOR_SOURCE,
                        "point_cloud_policy_id": POINT_CLOUD_POLICY_ID,
                        "point_cloud_config_sha256": pointcloud.sha256,
                        "point_cloud_table_plane_abcd_json": "null",
                        "point_cloud_sampling": POINT_CLOUD_SAMPLING,
                        "point_cloud_transform": POINT_CLOUD_TRANSFORM,
                    }
                )


if __name__ == "__main__":
    unittest.main()

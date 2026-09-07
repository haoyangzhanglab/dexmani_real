"""Offline regressions for the frozen Policy Zarr v7 projection boundary.

No hardware.  Exports a processed-v14 fixture and pins the Zarr v7 contract:
schema_version stays 7, the data-key set and root semantic attrs are exactly
the legacy v7 projection, and the processed-v14-only fields (``eef_pose``,
``tactile_force``) plus their attrs never leak into the store.  Run with:

    python -m unittest discover -s tests -p 'test_zarr_v7_projection.py'
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import zarr

from dexmani_real.dataset.contracts import OutputProfile
from dexmani_real.dataset.export import (
    POLICY_ZARR_SCHEMA_NAME,
    POLICY_ZARR_SCHEMA_VERSION,
    _policy_zarr_v7_keys,
    export_processed_hdf5_to_zarr,
)
from dexmani_real.planning.kinematics.fingertip import (
    FINGERTIP_POINTS_DERIVATION,
    FINGERTIP_POLICY_ID,
)
from test_processed_v14 import _process_fixture

_EXPECTED_V7_DATA_KEYS = {
    "joint_state",
    "action",
    "action_ee",
    "contact_force",
    "fingertip_points",
}

_EXPECTED_V7_ATTR_KEYS = {
    "schema_name",
    "schema_version",
    "domain",
    "profile",
    "task_name",
    "dt",
    "episode_start_policy",
    "obs_alignment",
    "observation_reference",
    "state_alignment",
    "max_observation_skew_s",
    "action_semantics",
    "contact_force_unit",
    "contact_force_si_verified",
    "contact_force_frame",
    "contact_force_source",
    "contact_force_alignment",
    "contact_force_fresh_required",
    "contact_force_calibrated_required",
    "contact_force_unit_code",
    "contact_force_causal_to_reference",
    "contact_force_hand_source_match_required",
    "fingertip_points_frame",
    "fingertip_points_unit",
    "fingertip_points_derivation",
    "fingertip_points_policy_id",
    "action_ee_frame",
}


class TestPolicyZarrV7Projection(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        workdir = Path(cls._tmp.name) / "export"
        workdir.mkdir(parents=True)
        out_path, reader, _decision, config = _process_fixture(workdir)
        cls.processed_path = out_path
        cls.config = config
        cls.reader = reader
        cls.zarr_path = workdir / "task.zarr"
        cls.report = export_processed_hdf5_to_zarr(out_path.parent, cls.zarr_path)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.reader.h5f.close()
        finally:
            cls._tmp.cleanup()

    def test_export_published_one_episode(self) -> None:
        self.assertEqual(self.report["episode_count"], 1)
        self.assertEqual(self.report["rejected_episode_count"], 0)
        self.assertEqual(self.report["profile"], "joint")
        self.assertEqual(sorted(self.report["dataset_keys"]), sorted(_EXPECTED_V7_DATA_KEYS))

    def test_schema_version_remains_seven(self) -> None:
        root = zarr.open_group(str(self.zarr_path), mode="r")
        self.assertEqual(int(root.attrs["schema_version"]), 7)
        self.assertEqual(POLICY_ZARR_SCHEMA_VERSION, 7)
        self.assertEqual(str(root.attrs["schema_name"]), POLICY_ZARR_SCHEMA_NAME)

    def test_data_keys_are_exactly_the_legacy_projection(self) -> None:
        root = zarr.open_group(str(self.zarr_path), mode="r")
        self.assertEqual(set(root["data"].array_keys()), _EXPECTED_V7_DATA_KEYS)
        self.assertNotIn("eef_pose", set(root["data"].array_keys()))
        self.assertNotIn("tactile_force", set(root["data"].array_keys()))

    def test_root_attrs_frozen_and_no_new_field_leakage(self) -> None:
        root = zarr.open_group(str(self.zarr_path), mode="r")
        attrs = dict(root.attrs)
        self.assertEqual(set(attrs), _EXPECTED_V7_ATTR_KEYS)
        leaked = [
            key
            for key in attrs
            if key.startswith("eef_pose_") or key.startswith("tactile_force_")
        ]
        self.assertEqual(leaked, [])
        self.assertEqual(attrs["domain"], "real")
        self.assertEqual(attrs["profile"], "joint")
        self.assertEqual(attrs["task_name"], "fixture_task")
        self.assertEqual(attrs["episode_start_policy"], "full_history")
        self.assertEqual(attrs["obs_alignment"], "obs[t]_before_action[t]")
        self.assertEqual(attrs["observation_reference"], "grid_anchor_monotonic_ns")
        self.assertEqual(attrs["state_alignment"], "control_grid_state")
        self.assertEqual(attrs["action_semantics"], "teleop_published_joint_target")
        self.assertEqual(attrs["contact_force_unit"], "sdk_scaled_unknown_si")
        self.assertIs(bool(attrs["contact_force_si_verified"]), False)
        self.assertEqual(
            attrs["contact_force_frame"], "xhand_sensor_native_axes_per_finger"
        )
        self.assertEqual(attrs["contact_force_source"], "control_grid_tactile_sum")
        self.assertEqual(
            attrs["contact_force_alignment"],
            "newest_source_not_after_grid_within_max_observation_skew",
        )
        self.assertIs(bool(attrs["contact_force_fresh_required"]), True)
        self.assertIs(bool(attrs["contact_force_calibrated_required"]), True)
        self.assertEqual(int(attrs["contact_force_unit_code"]), 0)
        self.assertIs(bool(attrs["contact_force_causal_to_reference"]), True)
        self.assertIs(bool(attrs["contact_force_hand_source_match_required"]), True)
        self.assertEqual(attrs["fingertip_points_frame"], "xarm_base")
        self.assertEqual(attrs["fingertip_points_unit"], "m")
        self.assertEqual(
            attrs["fingertip_points_derivation"], FINGERTIP_POINTS_DERIVATION
        )
        self.assertEqual(attrs["fingertip_points_policy_id"], FINGERTIP_POLICY_ID)
        self.assertEqual(attrs["action_ee_frame"], "xarm_base")
        self.assertAlmostEqual(
            float(attrs["max_observation_skew_s"]),
            self.config.max_observation_skew_s,
            places=12,
        )

    def test_projected_payload_matches_processed_source(self) -> None:
        root = zarr.open_group(str(self.zarr_path), mode="r")
        with h5py.File(self.processed_path, "r") as source:
            for key in sorted(_EXPECTED_V7_DATA_KEYS):
                stored = np.asarray(root["data"][key][:])
                original = np.asarray(source[key][:])
                self.assertEqual(stored.shape, original.shape)
                self.assertEqual(np.dtype(stored.dtype), np.dtype(original.dtype))
                np.testing.assert_array_equal(stored, original)
        ends = np.asarray(root["meta"]["episode_ends"][:])
        np.testing.assert_array_equal(ends, np.asarray([24], dtype=np.int64))

    def test_projection_key_helper_excludes_v14_only_fields(self) -> None:
        for profile in OutputProfile:
            keys = _policy_zarr_v7_keys(profile)
            self.assertNotIn("eef_pose", keys)
            self.assertNotIn("tactile_force", keys)
            self.assertTrue(set(keys).issubset(set(profile.dataset_keys)))
        self.assertEqual(
            _policy_zarr_v7_keys(OutputProfile.JOINT),
            (
                "joint_state",
                "action",
                "action_ee",
                "contact_force",
                "fingertip_points",
            ),
        )
        self.assertEqual(
            _policy_zarr_v7_keys(OutputProfile.RGB_PC),
            (
                "joint_state",
                "action",
                "action_ee",
                "contact_force",
                "fingertip_points",
                "rgb",
                "depth",
                "camera_intrinsic",
                "camera_extrinsic",
                "point_cloud",
            ),
        )


class TestZarrV7AdmissionFailClosed(unittest.TestCase):
    """Corrupted v14-only semantics must fail export admission before the v7 projection."""

    def _expect_export_rejects(self, attr_name: str, bad_value: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "corrupt"
            workdir.mkdir(parents=True)
            out_path, reader, _decision, _config = _process_fixture(workdir)
            try:
                with h5py.File(out_path, "r+") as f:
                    f.attrs[attr_name] = bad_value
                with self.assertRaises(ValueError):
                    export_processed_hdf5_to_zarr(
                        out_path.parent, workdir / "out.zarr"
                    )
            finally:
                reader.h5f.close()

    def test_corrupt_eef_pose_semantics_rejected(self) -> None:
        self._expect_export_rejects("eef_pose_frame", "wrong_frame")

    def test_corrupt_eef_algorithm_id_rejected(self) -> None:
        self._expect_export_rejects("eef_pose_algorithm_id", "wrong_fk_v0")

    def test_corrupt_tactile_force_semantics_rejected(self) -> None:
        self._expect_export_rejects(
            "tactile_force_representation", "wrong_representation"
        )


if __name__ == "__main__":
    unittest.main()

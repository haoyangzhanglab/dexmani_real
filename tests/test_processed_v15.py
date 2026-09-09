"""Offline regressions for the processed HDF5 v15 writer and validators.

No hardware.  A synthetic JOINT-profile raw fixture drives ``analyze_episode``
and ``_write_processed_episode`` end to end, then the v15 validators pin keys,
shapes, dtypes, semantic attrs, tactile provenance causality, and fail-closed
behavior, SOURCE-only cleaning and camera/tactile admission. Run with:

    python -m unittest discover -s tests -p 'test_processed_v15.py'
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from dexmani_real.dataset.clean import analyze_episode
from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    OutputProfile,
    ProcessingConfig,
)
from dexmani_real.dataset.processed import (
    PROCESSED_SCHEMA_VERSION,
    _validate_processed_output_structure,
    validate_processed_hdf5,
)
from dexmani_real.dataset.processing import (
    _gather_dataset_rows,
    _processed_joint_state,
    _write_processed_episode,
)
from dexmani_real.config.defaults import hand as hand_defaults
from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base
from dexmani_real.recording.storage.schema import FillReason

_FRAME_COUNT = 24
_GRID_DT_S = 0.05
_T0_NS = 10**15
_DT_NS = int(_GRID_DT_S * 1e9)


def _joint_config() -> ProcessingConfig:
    return ProcessingConfig(profile=OutputProfile.JOINT)


def _write_raw_fixture(
    path: Path,
    *,
    frames: int = _FRAME_COUNT,
    stale_tactile_rows: tuple[int, ...] = (),
    sum_fresh_rows: tuple[int, ...] = (),
    seed: int = 17,
) -> dict[str, np.ndarray]:
    """Create a minimal JOINT-profile raw episode and return its arrays."""
    rng = np.random.default_rng(seed)
    anchor_ns = _T0_NS + np.arange(frames, dtype=np.int64) * _DT_NS
    timestamps_s = anchor_ns.astype(np.float64) / 1e9
    arm_qpos = 0.05 * np.sin(np.arange(frames)[:, None] * 0.3 + np.arange(7)[None, :])
    # Hand action limits have strictly positive minima on several joints; the
    # fixture sits at the action-limit midpoint, inside the mechanical range.
    hand_mid = 0.5 * (
        np.asarray(hand_defaults.qpos_min_rad) + np.asarray(hand_defaults.qpos_max_rad)
    )
    hand_qpos = np.tile(hand_mid, (frames, 1))
    action_arm = arm_qpos + 1e-4
    action_hand = hand_qpos.copy()
    action_arm_ee = np.zeros((frames, 9))
    action_arm_ee[:, 3] = 1.0
    action_arm_ee[:, 7] = 1.0
    hand_contact = rng.normal(scale=0.5, size=(frames, 5, 3))
    hand_tactile_force = rng.normal(scale=0.5, size=(frames, 5, 120, 3))
    fresh = np.ones(frames, dtype=bool)
    for row in stale_tactile_rows:
        fresh[row] = False
    sum_fresh = fresh.copy()
    for row in sum_fresh_rows:
        sum_fresh[row] = False

    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["num_frames"] = frames
        meta.attrs["task_label"] = "fixture_task"

        def put(name: str, values: np.ndarray) -> None:
            f.create_dataset(name, data=values)

        put("timestamp", timestamps_s)
        put("source_sample_index", np.arange(frames, dtype=np.int64))
        put("fill_reason", np.full(frames, int(FillReason.SOURCE), dtype=np.int64))
        put("flag_sample_valid", np.ones(frames, dtype=bool))
        put("flag_action_queued", np.ones(frames, dtype=bool))
        put("observation_valid", np.ones(frames, dtype=bool))
        put("flag_frame_status", np.zeros(frames, dtype=np.int64))
        put("arm_connected", np.ones(frames, dtype=bool))
        put("hand_connected", np.ones(frames, dtype=bool))
        put("hand_qpos_stale", np.zeros(frames, dtype=bool))
        put("arm_qpos", arm_qpos)
        put("hand_qpos", hand_qpos)
        put("action_arm_joint_sent", action_arm)
        put("action_hand_joint", action_hand)
        put("action_arm_ee", action_arm_ee)
        put("hand_contact", hand_contact)
        put("hand_tactile_force", hand_tactile_force)
        put("tracking_error", np.zeros(frames))
        put("arm_last_cmd_seq", np.arange(1, frames + 1, dtype=np.int64))
        put("observation_anchor_monotonic_ns", anchor_ns)
        put("arm_source_monotonic_ns", anchor_ns)
        put("hand_source_monotonic_ns", anchor_ns)
        put("tactile_source_monotonic_ns", anchor_ns)
        put("tactile_sum_fresh", sum_fresh)
        put("tactile_fresh", fresh)
        put("tactile_calibrated", np.ones(frames, dtype=bool))
        put("tactile_unit_code", np.zeros(frames, dtype=np.int64))
    return {
        "anchor_ns": anchor_ns,
        "arm_qpos": arm_qpos,
        "hand_qpos": hand_qpos,
        "hand_contact": hand_contact,
        "hand_tactile_force": hand_tactile_force,
        "tactile_sum_fresh": sum_fresh,
        "tactile_fresh": fresh,
    }


def _fake_reader(raw_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        h5f=h5py.File(raw_path, "r"),
        h5_path=raw_path,
        timing=SimpleNamespace(grid_dt_s=_GRID_DT_S),
    )


def _process_fixture(
    workdir: Path,
    *,
    stale_tactile_rows: tuple[int, ...] = (),
) -> tuple[Path, SimpleNamespace, object, ProcessingConfig]:
    raw_path = workdir / "episode_fixture.h5"
    _write_raw_fixture(
        raw_path,
        stale_tactile_rows=stale_tactile_rows,
    )
    reader = _fake_reader(raw_path)
    config = _joint_config()
    annotation = EpisodeAnnotation(task_name="fixture_task")
    decision = analyze_episode(
        reader, config, annotation, source_already_validated=True
    )
    out_root = workdir / "processed"
    out_root.mkdir(parents=True, exist_ok=True)
    _write_processed_episode(
        reader,
        decision,
        out_root,
        config,
        task_name="fixture_task",
    )
    out_path = out_root / f"{raw_path.name}.h5"
    return out_path, reader, decision, config


class TestVisualProfileEef(unittest.TestCase):
    def test_rgb_geometry_uses_policy_observation_state(self):
        arm_raw = np.zeros((4, 7), dtype=np.float64)
        arm_policy = np.random.default_rng(43).uniform(-0.4, 0.4, (4, 7))
        hand_policy = np.full((4, 12), 0.123456789, dtype=np.float64)
        raw_arm_ee = compute_eef_pose_history_xarm_base(arm_raw)
        selected = np.asarray([0, 2, 3])
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory) / "visual.h5", "w") as raw:
                raw["arm_qpos"] = arm_raw
                raw["hand_qpos"] = np.zeros((4, 12))
                raw["policy_observation_arm_qpos"] = arm_policy
                raw["policy_observation_hand_qpos"] = hand_policy
                joint = _processed_joint_state(
                    SimpleNamespace(h5f=raw),
                    selected,
                    ProcessingConfig(profile=OutputProfile.RGB),
                )
                processed_eef = compute_eef_pose_history_xarm_base(joint[:, :7]).astype(
                    np.float32
                )
                expected_eef = compute_eef_pose_history_xarm_base(
                    arm_policy[selected].astype(np.float32)
                ).astype(np.float32)
                np.testing.assert_array_equal(
                    joint[:, :7], arm_policy[selected].astype(np.float32)
                )
                np.testing.assert_array_equal(
                    joint[:, 7:19], hand_policy[selected].astype(np.float32)
                )
                np.testing.assert_array_equal(processed_eef, expected_eef)
                self.assertFalse(np.allclose(processed_eef, raw_arm_ee[selected]))


class TestGatherDatasetRows(unittest.TestCase):
    def test_duplicate_and_noncontiguous_indices_preserve_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gather.h5"
            values = np.arange(20, dtype=np.float64).reshape(10, 2)
            with h5py.File(path, "w") as f:
                f.create_dataset("rows", data=values)
            with h5py.File(path, "r") as f:
                indices = np.array([3, 1, 3, 9, 0, 3], dtype=np.int64)
                gathered = _gather_dataset_rows(f["rows"], indices)
            np.testing.assert_array_equal(gathered, values[indices])

    def test_all_duplicate_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gather_dup.h5"
            values = np.arange(8, dtype=np.float64).reshape(4, 2)
            with h5py.File(path, "w") as f:
                f.create_dataset("rows", data=values)
            with h5py.File(path, "r") as f:
                gathered = _gather_dataset_rows(
                    f["rows"], np.array([2, 2, 2], dtype=np.int64)
                )
            np.testing.assert_array_equal(gathered, np.repeat(values[2:3], 3, axis=0))


class TestProcessedV15Writer(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        workdir = Path(cls._tmp.name) / "clean"
        workdir.mkdir(parents=True)
        cls.out_path, cls.reader, cls.decision, cls.config = _process_fixture(workdir)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.reader.h5f.close()
        finally:
            cls._tmp.cleanup()

    def test_decision_accepted_all_rows(self) -> None:
        self.assertTrue(self.decision.accepted)
        self.assertIsNone(self.decision.rejected_reason)
        self.assertEqual(self.decision.selected_frames, _FRAME_COUNT)

    def test_validators_pass(self) -> None:
        summary = _validate_processed_output_structure(self.out_path, self.config)
        self.assertEqual(summary["frames"], _FRAME_COUNT)
        validate_processed_hdf5(self.out_path, self.config)

    def test_invalid_task_name_rejected_by_both_validators(self) -> None:
        for invalid_task_name in ("", "unknown", " task ", "task\x7f"):
            with self.subTest(task_name=repr(invalid_task_name)):
                with tempfile.TemporaryDirectory() as tmp:
                    workdir = Path(tmp) / "invalid-task"
                    workdir.mkdir(parents=True)
                    out_path, reader, _decision, config = _process_fixture(workdir)
                    try:
                        with h5py.File(out_path, "r+") as output:
                            output.attrs["task_name"] = invalid_task_name
                        with self.assertRaisesRegex(ValueError, "invalid task_name"):
                            _validate_processed_output_structure(out_path, config)
                        with self.assertRaisesRegex(ValueError, "invalid task_name"):
                            validate_processed_hdf5(out_path, config)
                    finally:
                        reader.h5f.close()

    def test_schema_version_and_keys(self) -> None:
        self.assertEqual(PROCESSED_SCHEMA_VERSION, 15)
        with h5py.File(self.out_path, "r") as f:
            self.assertEqual(int(f.attrs["schema_version"]), 15)
            expected = set(OutputProfile.JOINT.dataset_keys) | {"provenance"}
            self.assertTrue(expected.issubset(set(f.keys())))
            self.assertIn("eef_pose", f)
            self.assertIn("tactile_force", f)

    def test_shapes_and_dtypes(self) -> None:
        with h5py.File(self.out_path, "r") as f:
            self.assertEqual(f["eef_pose"].shape, (_FRAME_COUNT, 9))
            self.assertEqual(f["eef_pose"].dtype, np.dtype(np.float32))
            self.assertEqual(f["tactile_force"].shape, (_FRAME_COUNT, 5, 120, 3))
            self.assertEqual(f["tactile_force"].dtype, np.dtype(np.float32))
            for name in (
                "tactile_source_row_index",
                "observation_reference_monotonic_ns",
                "tactile_source_monotonic_ns",
            ):
                ds = f["provenance"][name]
                self.assertEqual(ds.shape, (_FRAME_COUNT,))
                self.assertEqual(ds.dtype, np.dtype(np.int64))

    def test_eef_pose_equals_fk_of_processed_joint_state(self) -> None:
        with h5py.File(self.out_path, "r") as f:
            joint_state = np.asarray(f["joint_state"][:])
            eef_pose = np.asarray(f["eef_pose"][:])
        expected = compute_eef_pose_history_xarm_base(joint_state[:, :7]).astype(
            np.float32
        )
        np.testing.assert_array_equal(eef_pose, expected)
        self.assertTrue(np.all(np.isfinite(eef_pose)))

    def test_tactile_and_contact_share_one_source_row(self) -> None:
        with h5py.File(self.out_path, "r") as f:
            contact = np.asarray(f["contact_force"][:])
            tactile = np.asarray(f["tactile_force"][:])
            rows = np.asarray(f["provenance/tactile_source_row_index"][:])
            source_rows = np.asarray(f["provenance/source_row_index"][:])
            reference_ns = np.asarray(
                f["provenance/observation_reference_monotonic_ns"][:]
            )
            tactile_ns = np.asarray(f["provenance/tactile_source_monotonic_ns"][:])
        np.testing.assert_array_equal(rows, source_rows)
        np.testing.assert_allclose(
            contact,
            self.reader.h5f["hand_contact"][:][rows].astype(np.float32),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            tactile,
            self.reader.h5f["hand_tactile_force"][:][rows].astype(np.float32),
            rtol=0.0,
            atol=0.0,
        )
        anchor = self.reader.h5f["observation_anchor_monotonic_ns"][:].astype(np.int64)
        np.testing.assert_array_equal(reference_ns, anchor[source_rows])
        np.testing.assert_array_equal(tactile_ns, anchor[rows])
        self.assertTrue(np.all(tactile_ns <= reference_ns))
        skew_ns = reference_ns - tactile_ns
        self.assertTrue(
            np.all(skew_ns <= int(round(self.config.max_observation_skew_s * 1e9)))
        )

    def test_semantic_attrs(self) -> None:
        with h5py.File(self.out_path, "r") as f:
            attrs = dict(f.attrs)

        def _text(name: str) -> str:
            value = attrs[name]
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        self.assertEqual(_text("eef_pose_frame"), "xarm_base")
        self.assertEqual(_text("eef_pose_components"), "position_m(3)+rot6d(6)")
        self.assertEqual(
            _text("eef_pose_derivation"), "canonical_arm_fk_from_aligned_qpos"
        )
        self.assertEqual(
            _text("eef_pose_algorithm_id"), "xarm7_custom_eef_pinocchio_fk_v1"
        )
        self.assertEqual(
            _text("tactile_force_representation"),
            "xhand_sdk_raw_force_fx_fy_fz_bias_corrected",
        )
        self.assertEqual(
            _text("tactile_force_sensor_order"), "xhand_sdk_sensor_data_order"
        )
        self.assertEqual(
            _text("tactile_force_point_order"),
            "xhand_sdk_sensor_data_raw_force_order",
        )
        self.assertEqual(_text("tactile_force_axis_labels"), "fx_fy_fz")
        self.assertEqual(_text("tactile_force_unit"), "xhand_sdk_native_unknown_si")
        self.assertEqual(
            _text("contact_force_representation"),
            "xhand_sdk_calc_force_fx_fy_fz_bias_corrected",
        )
        self.assertEqual(_text("contact_force_unit"), "xhand_sdk_native_unknown_si")
        self.assertIs(bool(attrs["tactile_force_si_verified"]), False)
        self.assertIs(bool(attrs["tactile_force_spatial_geometry_verified"]), False)
        for name in (
            "tactile_force_fresh_required",
            "tactile_force_calibrated_required",
            "tactile_force_causal_to_reference",
            "tactile_force_hand_source_match_required",
        ):
            self.assertIs(bool(attrs[name]), True, name)
        self.assertEqual(int(attrs["tactile_force_unit_code"]), 0)


class TestProcessedV15ForwardFill(unittest.TestCase):
    def test_forward_fill_duplicates_one_source_row(self) -> None:
        # Rows 5-6 are not fresh; their references stay within the 100ms skew
        # cap of row 4, so both forward-fill onto the same raw source row.
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "fill"
            workdir.mkdir(parents=True)
            out_path, reader, decision, config = _process_fixture(
                workdir, stale_tactile_rows=(5, 6)
            )
            try:
                self.assertTrue(decision.accepted)
                self.assertEqual(
                    decision.repair_reason_counts.get("tactile_forward_fill"), 2
                )
                validate_processed_hdf5(out_path, config)
                with h5py.File(out_path, "r") as f:
                    rows = np.asarray(f["provenance/tactile_source_row_index"][:])
                    source_rows = np.asarray(f["provenance/source_row_index"][:])
                    contact = np.asarray(f["contact_force"][:])
                    tactile = np.asarray(f["tactile_force"][:])
                    raw_contact = reader.h5f["hand_contact"][:]
                    raw_tactile = reader.h5f["hand_tactile_force"][:]
                np.testing.assert_array_equal(rows[5:7], np.full(2, 4, dtype=np.int64))
                self.assertTrue(np.all(rows <= source_rows))
                self.assertEqual(len(np.unique(rows)), _FRAME_COUNT - 2)
                np.testing.assert_allclose(
                    contact, raw_contact[rows].astype(np.float32), rtol=0.0, atol=0.0
                )
                np.testing.assert_allclose(
                    tactile, raw_tactile[rows].astype(np.float32), rtol=0.0, atol=0.0
                )
            finally:
                reader.h5f.close()


class TestProcessedV15PairedValidity(unittest.TestCase):
    def test_aggregate_invalid_row_is_not_a_paired_tactile_source(self) -> None:
        # Row 0 has aggregate-invalid (tactile_sum_fresh=False) but dense-valid
        # (tactile_fresh=True) payloads; it must never become the paired source
        # for processed contact_force/tactile_force. Row 1 is fully valid.
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "paired"
            workdir.mkdir(parents=True)
            raw_path = workdir / "episode_fixture.h5"
            _write_raw_fixture(raw_path, sum_fresh_rows=(0,))
            with h5py.File(raw_path, "r+") as raw:
                # A distinctive finite aggregate value that must not reach output.
                raw["hand_contact"][0] = 777.0
            reader = _fake_reader(raw_path)
            config = _joint_config()
            annotation = EpisodeAnnotation(task_name="fixture_task")
            try:
                decision = analyze_episode(
                    reader, config, annotation, source_already_validated=True
                )
                out_root = workdir / "processed"
                out_root.mkdir(parents=True, exist_ok=True)
                _write_processed_episode(
                    reader, decision, out_root, config, task_name="fixture_task"
                )
                out_path = out_root / f"{raw_path.name}.h5"
                self.assertFalse(decision.keep_mask[0])
                self.assertEqual(decision.hard_reason_counts["tactile_invalid"], 1)
                self.assertEqual(decision.selected_frames, _FRAME_COUNT - 1)
                validate_processed_hdf5(out_path, config)
                with h5py.File(out_path, "r") as f:
                    rows = np.asarray(f["provenance/tactile_source_row_index"][:])
                    source_rows = np.asarray(f["provenance/source_row_index"][:])
                    contact = np.asarray(f["contact_force"][:])
                    tactile = np.asarray(f["tactile_force"][:])
                self.assertTrue(np.all(rows != 0))
                np.testing.assert_array_equal(rows, source_rows)
                np.testing.assert_allclose(
                    contact,
                    reader.h5f["hand_contact"][:][rows].astype(np.float32),
                    rtol=0.0,
                    atol=0.0,
                )
                np.testing.assert_allclose(
                    tactile,
                    reader.h5f["hand_tactile_force"][:][rows].astype(np.float32),
                    rtol=0.0,
                    atol=0.0,
                )
                self.assertFalse(np.any(contact == np.float32(777.0)))
            finally:
                reader.h5f.close()


class TestProcessedV15FailClosed(unittest.TestCase):
    def _mutate_and_expect_failure(self, mutate) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "fail"
            workdir.mkdir(parents=True)
            out_path, reader, _decision, config = _process_fixture(workdir)
            try:
                with h5py.File(out_path, "r+") as f:
                    mutate(f)
                with self.assertRaises(ValueError):
                    validate_processed_hdf5(out_path, config)
            finally:
                reader.h5f.close()

    def test_nonfinite_eef_pose_rejected(self) -> None:
        def mutate(f: h5py.File) -> None:
            f["eef_pose"][3, 0] = np.nan

        self._mutate_and_expect_failure(mutate)

    def test_noncanonical_eef_rot6d_rejected(self) -> None:
        def mutate(f: h5py.File) -> None:
            f["eef_pose"][3, 3:9] = np.float32(2.0) * np.asarray(f["eef_pose"][3, 3:9])

        self._mutate_and_expect_failure(mutate)

    def test_nonfinite_tactile_force_rejected(self) -> None:
        def mutate(f: h5py.File) -> None:
            f["tactile_force"][7, 2, 11, 1] = np.inf

        self._mutate_and_expect_failure(mutate)

    def test_tactile_row_after_source_row_rejected(self) -> None:
        def mutate(f: h5py.File) -> None:
            f["provenance/tactile_source_row_index"][2] = (
                int(np.asarray(f["provenance/source_row_index"][2])) + 1
            )

        self._mutate_and_expect_failure(mutate)

    def test_reference_before_source_rejected(self) -> None:
        def mutate(f: h5py.File) -> None:
            f["provenance/tactile_source_monotonic_ns"][1] = (
                int(np.asarray(f["provenance/observation_reference_monotonic_ns"][1]))
                + 1
            )

        self._mutate_and_expect_failure(mutate)

    def test_skew_violation_rejected(self) -> None:
        def mutate(f: h5py.File) -> None:
            f["provenance/observation_reference_monotonic_ns"][0] = (
                int(np.asarray(f["provenance/tactile_source_monotonic_ns"][0])) + 10**12
            )

        self._mutate_and_expect_failure(mutate)


class TestV26Cleaning(unittest.TestCase):
    def _decision(self, mutate, *, visual=False):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "raw.h5"
            _write_raw_fixture(path)
            with h5py.File(path, "r+") as raw:
                if visual:
                    raw["policy_observation_arm_qpos"] = raw["arm_qpos"][:]
                    raw["policy_observation_hand_qpos"] = raw["hand_qpos"][:]
                    raw["policy_observation_valid"] = np.ones(_FRAME_COUNT, dtype=bool)
                    raw["camera_source_monotonic_ns"] = raw[
                        "observation_anchor_monotonic_ns"
                    ][:]
                    raw["flag_camera_fresh"] = np.ones(_FRAME_COUNT, dtype=bool)
                mutate(raw)
            reader = _fake_reader(path)
            try:
                return analyze_episode(
                    reader,
                    ProcessingConfig(
                        profile=OutputProfile.RGB if visual else OutputProfile.JOINT
                    ),
                    source_already_validated=True,
                    depth_valid_mask=(
                        np.ones(_FRAME_COUNT, dtype=bool) if visual else None
                    ),
                )
            finally:
                reader.h5f.close()

    def test_historical_non_source_rows_are_excluded(self):
        def mutate(raw):
            raw["fill_reason"][1:3] = [
                FillReason.CAUSAL_HOLD_LAST,
                FillReason.LEADING_PLACEHOLDER,
            ]
            raw["flag_sample_valid"][1:3] = False

        decision = self._decision(mutate)
        self.assertFalse(np.any(decision.keep_mask[1:3]))
        self.assertEqual(decision.hard_reason_counts["not_source_sample"], 2)

    def test_timestamp_gap_remains_a_segment_boundary(self):
        def mutate(raw):
            raw["timestamp"][12:] += _GRID_DT_S

        decision = self._decision(mutate)
        self.assertEqual(decision.boundary_counts["timestamp_discontinuity"], 1)
        np.testing.assert_array_equal(decision.segment_ends, [12, 24])

    def test_selected_full_tactile_nonfinite_is_not_repaired(self):
        def mutate(raw):
            raw["hand_tactile_force"][4, 0, 0, 0] = np.nan

        decision = self._decision(mutate)
        self.assertFalse(decision.keep_mask[4])
        self.assertEqual(decision.hard_reason_counts["nonfinite_real_modality"], 1)

    def test_frame_status_alone_controls_short_ik_admission(self):
        def mutate(raw):
            raw["flag_frame_status"][2:4] = 2
            raw["flag_frame_status"][10:15] = 2
            raw["flag_frame_status"][18:21] = [1, 3, 4]

        decision = self._decision(mutate)
        self.assertTrue(np.all(decision.keep_mask[2:4]))
        self.assertFalse(np.any(decision.keep_mask[10:15]))
        self.assertFalse(np.any(decision.keep_mask[18:21]))

    def test_observation_valid_is_required(self):
        def mutate(raw):
            raw["observation_valid"][0] = False

        self.assertFalse(self._decision(mutate).keep_mask[0])

    def test_joint_dataset_age_remains_tighter_than_live_stale_threshold(self):
        def mutate(raw):
            raw["arm_source_monotonic_ns"][0] -= 150_000_000

        decision = self._decision(mutate)
        self.assertFalse(decision.keep_mask[0])
        self.assertEqual(
            decision.hard_reason_counts["control_grid_observation_invalid"], 1
        )

    def test_visual_fresh_and_policy_valid_are_required(self):
        def mutate(raw):
            raw["flag_camera_fresh"][0] = False
            raw["policy_observation_valid"][1] = False

        decision = self._decision(mutate, visual=True)
        self.assertFalse(np.any(decision.keep_mask[:2]))
        self.assertTrue(np.all(decision.keep_mask[2:]))

    def test_visual_future_camera_source_is_rejected_without_unsigned_wrap(self):
        def mutate(raw):
            raw["camera_source_monotonic_ns"][0] += _DT_NS

        decision = self._decision(mutate, visual=True)
        self.assertFalse(decision.keep_mask[0])
        self.assertEqual(decision.hard_reason_counts["camera_invalid"], 1)


if __name__ == "__main__":
    unittest.main()

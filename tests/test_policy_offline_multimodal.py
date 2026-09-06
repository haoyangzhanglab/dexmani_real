"""Focused offline tests for processed multimodal and Policy Zarr boundaries."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import zarr

from dexmani_real.config.experiment import resolve_experiment_config
from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.dataset.clean import align_tactile_sum_rows_to_references
from dexmani_real.dataset.contracts import (
    EpisodeAnnotation,
    EpisodeDecision,
    OutputProfile,
    ProcessingConfig,
    QualityPolicy,
)
from dexmani_real.dataset.export import (
    PolicyZarrExportConfig,
    _Artifact,
    _inspect_artifact,
    export_processed_hdf5_to_zarr,
)
from dexmani_real.dataset.processed import (
    PROCESSED_SCHEMA_NAME,
    PROCESSED_SCHEMA_VERSION,
    _validate_processed_output_structure,
    validate_processed_hdf5,
)
from dexmani_real.dataset.processing import (
    _write_processed_episode,
    compute_fingertip_history_xarm_base,
    discover_episode_dirs,
)
from dexmani_real.planning.kinematics.fingertip import (
    FINGERTIP_POINTS_DERIVATION,
    FINGERTIP_POLICY_ID,
    compute_fingertip_geometry_sha256,
    compute_fingertip_points_xarm_base,
)
from dexmani_real.sensor.pointcloud import (
    POINT_CLOUD_COLOR_SOURCE,
    POINT_CLOUD_POLICY_ID,
    POINT_CLOUD_SAMPLING,
    POINT_CLOUD_TRANSFORM,
)
from dexmani_real.sensor.pointcloud_worker import PointCloudLoopConfig
from examples import process_episodes


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

    def test_every_profile_derives_fingertips_from_its_processed_joint_state(
        self,
    ) -> None:
        class _StopAfterFingertipArguments(RuntimeError):
            pass

        control_arm = np.array(
            [
                [0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 0.0],
                [0.04, 0.05, 0.06, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        control_hand = np.array(
            [
                [0.10, 0.11, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.13, 0.14, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        camera_arm = control_arm + 0.20
        camera_hand = control_hand + 0.20
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode"
            episode.mkdir()
            (episode / "rgb.mp4").write_bytes(b"synthetic-rgb")
            with h5py.File(episode / "depth.h5", "w"):
                pass
            source_path = episode / "data.h5"
            with h5py.File(source_path, "w") as source:
                source.create_group("meta").attrs.update(
                    {
                        "task_label": "pick",
                        "resolved_config_sha256": "a" * 64,
                        "depth_scale": 0.001,
                        "camera_depth_intrinsics": np.eye(3).reshape(-1),
                        "camera_depth_distortion_model": "none",
                        "camera_depth_distortion_coeffs": np.zeros(5),
                        "camera_color_distortion_model": "none",
                        "camera_color_distortion_coeffs": np.zeros(5),
                        "camera_T_color_from_depth": np.eye(4),
                    }
                )
                source.create_dataset("arm_qpos", data=control_arm)
                source.create_dataset("hand_qpos", data=control_hand)
                source.create_dataset("policy_observation_arm_qpos", data=camera_arm)
                source.create_dataset("policy_observation_hand_qpos", data=camera_hand)
                source.create_dataset("hand_fingertip", data=np.full((2, 5, 3), -7.0))
                source.create_dataset("action_arm_joint_sent", data=np.zeros((2, 7)))
                source.create_dataset("action_hand_joint", data=np.zeros((2, 12)))
                action_ee = np.zeros((2, 9))
                action_ee[:, 3] = 1.0
                action_ee[:, 7] = 1.0
                source.create_dataset("action_arm_ee", data=action_ee)
                source.create_dataset("hand_contact", data=np.zeros((2, 5, 3)))
                source_times_ns = np.array(
                    [1_000_000_000, 1_062_500_000], dtype=np.int64
                )
                for key in (
                    "hand_source_monotonic_ns",
                    "tactile_source_monotonic_ns",
                    "observation_anchor_monotonic_ns",
                    "camera_source_monotonic_ns",
                ):
                    source.create_dataset(key, data=source_times_ns)
                source.create_dataset("tactile_fresh", data=np.ones(2, dtype=bool))
                source.create_dataset("tactile_calibrated", data=np.ones(2, dtype=bool))
                source.create_dataset("tactile_unit_code", data=np.zeros(2, np.int64))
                source.create_dataset(
                    "source_sample_index", data=np.arange(2, dtype=np.int64)
                )
                source.create_dataset(
                    "timestamp", data=np.array([1.0, 1.0625], dtype=np.float64)
                )

            processed_root = root / "processed"
            processed_root.mkdir()
            with h5py.File(source_path, "r") as source:
                reader = SimpleNamespace(
                    h5_path=episode,
                    h5f=source,
                    timing=SimpleNamespace(grid_dt_s=0.0625),
                )
                for profile in OutputProfile:
                    with self.subTest(profile=profile.value):
                        config = ProcessingConfig(
                            profile=profile,
                            horizon=1,
                            min_full_windows=1,
                        )
                        decision = EpisodeDecision(
                            source_path=episode,
                            source_frames=2,
                            profile=profile,
                            selected_indices=np.arange(2, dtype=np.int64),
                            keep_mask=np.ones(2, dtype=bool),
                            drop_reason_bits=np.zeros(2, dtype=np.uint64),
                            drop_reason_names=(),
                            hard_reason_counts={},
                            boundary_counts={},
                            selected_frames=2,
                            quality={"full_window_count": 1},
                        )
                        expected_state = np.concatenate(
                            (
                                (camera_arm, camera_hand)
                                if profile.needs_rgb or profile.needs_pointcloud
                                else (control_arm, control_hand)
                            ),
                            axis=1,
                        ).astype(np.float32)
                        with (
                            patch(
                                "dexmani_real.dataset.processing.HandKinematics",
                                return_value=_HandFk(),
                            ),
                            patch(
                                "dexmani_real.dataset.processing.make_arm_fk",
                                return_value=_ArmFk(),
                            ),
                            patch(
                                "dexmani_real.dataset.processing.load_raw_episode_camera_model",
                                return_value=object(),
                            ),
                            patch(
                                "dexmani_real.dataset.processing.load_raw_episode_base_from_color",
                                return_value=np.eye(4),
                            ),
                            patch(
                                "dexmani_real.dataset.processing.compute_fingertip_history_xarm_base",
                                side_effect=_StopAfterFingertipArguments,
                            ) as compute_fingertips,
                            self.assertRaises(_StopAfterFingertipArguments),
                        ):
                            _write_processed_episode(
                                reader,
                                decision,
                                processed_root,
                                config,
                                EpisodeAnnotation(),
                            )
                        arm_input, hand_input = compute_fingertips.call_args.args[:2]
                        np.testing.assert_array_equal(arm_input, expected_state[:, :7])
                        np.testing.assert_array_equal(
                            hand_input, expected_state[:, 7:19]
                        )

    def test_fingertip_geometry_sha256_tracks_every_fk_dependency(self) -> None:
        mapping = (3, 4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2)
        links = (
            "thumb_tip",
            "index_tip",
            "mid_tip",
            "ring_tip",
            "pinky_tip",
        )
        values = {
            "arm_fk_urdf_sha256": "a" * 64,
            "arm_eef_frame": "custom_eef_link",
            "hand_fk_urdf_sha256": "b" * 64,
            "hand_sdk_to_urdf_idx": mapping,
            "fingertip_link_names": links,
            "handbase_position_eef_m": np.array([0.01, -0.02, 0.03]),
            "handbase_quat_eef_wxyz": np.array([0.5, 0.5, 0.5, 0.5]),
        }

        reference = compute_fingertip_geometry_sha256(**values)
        self.assertEqual(reference, compute_fingertip_geometry_sha256(**values))
        self.assertEqual(
            reference,
            compute_fingertip_geometry_sha256(
                **{
                    **values,
                    "handbase_quat_eef_wxyz": -values["handbase_quat_eef_wxyz"],
                }
            ),
        )
        self.assertEqual(
            reference,
            compute_fingertip_geometry_sha256(
                **{
                    **values,
                    "handbase_quat_eef_wxyz": 3.0 * values["handbase_quat_eef_wxyz"],
                }
            ),
        )
        for key, value in (
            ("arm_fk_urdf_sha256", "c" * 64),
            ("hand_fk_urdf_sha256", "d" * 64),
            ("hand_sdk_to_urdf_idx", tuple(reversed(mapping))),
            ("fingertip_link_names", tuple(reversed(links))),
            ("handbase_position_eef_m", np.array([0.02, -0.02, 0.03])),
            ("handbase_quat_eef_wxyz", np.array([1.0, 0.0, 0.0, 0.0])),
        ):
            with self.subTest(key=key):
                self.assertNotEqual(
                    reference,
                    compute_fingertip_geometry_sha256(**{**values, key: value}),
                )

    def test_policy_zarr_v7_admits_all_four_processed_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for profile in OutputProfile:
                with self.subTest(profile=profile.value):
                    path = root / f"{profile.value}.h5"
                    self._write_processed_fixture(path, profile)
                    with (
                        patch("dexmani_real.dataset.export.validate_processed_payload"),
                        patch(
                            "dexmani_real.dataset.export.validate_processed_provenance",
                            return_value=object(),
                        ),
                        patch(
                            "dexmani_real.dataset.export._whole_episode_rejection",
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
                    self.assertEqual(
                        artifact.semantic_attrs["action_semantics"],
                        "teleop_published_joint_target",
                    )
                    self.assertNotIn("deployment_equivalent", artifact.semantic_attrs)

    def test_processed_v11_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.h5"
            self._write_processed_fixture(path, OutputProfile.JOINT)
            with h5py.File(path, "r+") as output:
                output.attrs["schema_version"] = PROCESSED_SCHEMA_VERSION - 1
            with self.assertRaisesRegex(
                ValueError, "unsupported processed schema version"
            ):
                _inspect_artifact(path, PolicyZarrExportConfig())

    def test_processing_runtime_timing_defaults_and_cli_overrides(self) -> None:
        runtime = resolve_experiment_config(
            data={
                "camera": {"max_frame_age_s": 0.4},
                "policy": {"max_observation_skew_s": 0.2},
            }
        )
        defaults = ProcessingConfig.from_runtime(runtime, profile=OutputProfile.JOINT)
        self.assertEqual(defaults.max_camera_age_s, 0.4)
        self.assertEqual(defaults.max_observation_skew_s, 0.2)

        parser = process_episodes._parser()
        args = parser.parse_args(["episodes/pick"])
        config = process_episodes._config(
            args, OutputProfile.JOINT, QualityPolicy.AUDIT, runtime
        )
        self.assertEqual(config.max_camera_age_s, 0.4)
        self.assertEqual(config.max_observation_skew_s, 0.2)

        args = parser.parse_args(
            [
                "episodes/pick",
                "--max-camera-age-s",
                "0.3",
                "--max-observation-skew-s",
                "0.1",
            ]
        )
        config = process_episodes._config(
            args, OutputProfile.JOINT, QualityPolicy.AUDIT, runtime
        )
        self.assertEqual(config.max_camera_age_s, 0.3)
        self.assertEqual(config.max_observation_skew_s, 0.1)

    def test_visual_processing_and_realtime_pointcloud_share_freshness_limit(
        self,
    ) -> None:
        for camera_max_age_s, policy_max_age_s, expected_max_age_s in (
            (0.25, 0.15, 0.15),
            (0.10, 0.15, 0.10),
        ):
            runtime = resolve_experiment_config(
                data={
                    "camera": {"max_frame_age_s": camera_max_age_s},
                    "policy": {"max_input_age_s": policy_max_age_s},
                }
            )
            for profile in (
                OutputProfile.RGB,
                OutputProfile.POINTCLOUD,
                OutputProfile.RGB_PC,
            ):
                config = ProcessingConfig.from_runtime(runtime, profile=profile)
                self.assertEqual(config.max_camera_age_s, expected_max_age_s)

            pointcloud_config = PointCloudLoopConfig.from_runtime(
                runtime,
                num_points=1024,
            )
            self.assertEqual(pointcloud_config.max_input_age_s, expected_max_age_s)

    def test_visual_camera_age_override_cannot_exceed_deployment_limit(self) -> None:
        runtime = resolve_experiment_config(
            data={
                "camera": {"max_frame_age_s": 0.25},
                "policy": {"max_input_age_s": 0.15},
            }
        )
        for profile in (
            OutputProfile.RGB,
            OutputProfile.POINTCLOUD,
            OutputProfile.RGB_PC,
        ):
            with self.assertRaisesRegex(ValueError, "max_input_age_s"):
                ProcessingConfig.from_runtime(
                    runtime,
                    profile=profile,
                    max_camera_age_s=0.20,
                )

        joint_config = ProcessingConfig.from_runtime(
            runtime,
            profile=OutputProfile.JOINT,
            max_camera_age_s=0.20,
        )
        self.assertEqual(joint_config.max_camera_age_s, 0.20)

    def test_relative_input_produces_canonical_source_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode"
            episode.mkdir()
            (episode / "data.h5").touch()
            relative_root = Path(os.path.relpath(root, Path.cwd()))

            (source_path,) = discover_episode_dirs(relative_root)
            decision = EpisodeDecision(
                source_path=source_path,
                source_frames=1,
                profile=OutputProfile.JOINT,
                selected_indices=np.asarray([0], dtype=np.int64),
                keep_mask=np.asarray([True]),
                drop_reason_bits=np.asarray([0], dtype=np.uint64),
                drop_reason_names=(),
                hard_reason_counts={},
                boundary_counts={},
                selected_frames=1,
                quality={"full_window_count": 1},
            )

            self.assertEqual(decision.to_dict()["source_path"], str(episode.resolve()))

    def test_joint_writer_recomputes_fingertips_from_processed_joint_state(
        self,
    ) -> None:
        """A JOINT artifact cannot inherit the raw fingertip representation."""
        config = ProcessingConfig(
            profile=OutputProfile.JOINT,
            horizon=2,
            min_full_windows=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = root / "episode"
            episode.mkdir()
            (episode / "rgb.mp4").write_bytes(b"synthetic-rgb")
            with h5py.File(episode / "depth.h5", "w"):
                pass
            source_path = episode / "data.h5"
            with h5py.File(source_path, "w") as source:
                source.create_group("meta").attrs.update(
                    {
                        "task_label": "pick",
                        "resolved_config_sha256": "a" * 64,
                    }
                )
                source.create_dataset(
                    "action_arm_joint_sent",
                    data=np.full((2, 7), 0.1, dtype=np.float64),
                )
                source.create_dataset(
                    "action_hand_joint", data=np.zeros((2, 12), dtype=np.float64)
                )
                action_ee = np.zeros((2, 9), dtype=np.float64)
                action_ee[:, 3] = 1.0
                action_ee[:, 7] = 1.0
                source.create_dataset("action_arm_ee", data=action_ee)
                arm_qpos = np.array(
                    [
                        [0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 0.0],
                        [0.04, 0.05, 0.06, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                )
                hand_qpos = np.array(
                    [
                        [0.10, 0.11, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.13, 0.14, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ],
                    dtype=np.float64,
                )
                source.create_dataset("arm_qpos", data=arm_qpos)
                source.create_dataset("hand_qpos", data=hand_qpos)
                raw_fingertips = np.full((2, 5, 3), -7.0, dtype=np.float64)
                source.create_dataset("hand_fingertip", data=raw_fingertips)
                source.create_dataset("hand_contact", data=np.zeros((2, 5, 3)))
                source_times_ns = np.array(
                    [1_000_000_000, 1_062_500_000], dtype=np.int64
                )
                for key in (
                    "hand_source_monotonic_ns",
                    "tactile_source_monotonic_ns",
                    "observation_anchor_monotonic_ns",
                ):
                    source.create_dataset(key, data=source_times_ns)
                source.create_dataset("tactile_fresh", data=np.ones(2, dtype=bool))
                source.create_dataset("tactile_calibrated", data=np.ones(2, dtype=bool))
                source.create_dataset("tactile_unit_code", data=np.zeros(2, np.int64))
                source.create_dataset(
                    "source_sample_index", data=np.arange(2, dtype=np.int64)
                )
                source.create_dataset(
                    "timestamp", data=np.array([1.0, 1.0625], dtype=np.float64)
                )

            decision = EpisodeDecision(
                source_path=episode,
                source_frames=2,
                profile=OutputProfile.JOINT,
                selected_indices=np.arange(2, dtype=np.int64),
                keep_mask=np.ones(2, dtype=bool),
                drop_reason_bits=np.zeros(2, dtype=np.uint64),
                drop_reason_names=(),
                hard_reason_counts={},
                boundary_counts={},
                selected_frames=2,
                quality={"full_window_count": 1},
            )
            processed_root = root / "processed"
            processed_root.mkdir()
            with h5py.File(source_path, "r") as source:
                reader = SimpleNamespace(
                    h5_path=episode,
                    h5f=source,
                    timing=SimpleNamespace(grid_dt_s=0.0625),
                )
                with (
                    patch(
                        "dexmani_real.dataset.processing.HandKinematics",
                        return_value=_HandFk(),
                    ),
                    patch(
                        "dexmani_real.dataset.processing.make_arm_fk",
                        return_value=_ArmFk(),
                    ),
                ):
                    output = _write_processed_episode(
                        reader,
                        decision,
                        processed_root,
                        config,
                        EpisodeAnnotation(),
                    )

            processed_path = processed_root / output["path"]
            with h5py.File(processed_path, "r") as processed:
                expected_fingertips = compute_fingertip_history_xarm_base(
                    arm_qpos,
                    hand_qpos,
                    arm_fk=_ArmFk(),
                    hand_fk=_HandFk(),
                    handbase_position_eef_m=np.asarray(
                        config.handbase_position_eef_m, dtype=np.float64
                    ),
                    handbase_quat_eef_wxyz=np.asarray(
                        config.handbase_quat_eef_wxyz, dtype=np.float64
                    ),
                )
                np.testing.assert_allclose(
                    processed["fingertip_points"][:], expected_fingertips
                )
                self.assertFalse(
                    np.array_equal(processed["fingertip_points"][:], raw_fingertips)
                )
                self.assertEqual(
                    processed.attrs["fingertip_points_derivation"],
                    FINGERTIP_POINTS_DERIVATION,
                )
                self.assertEqual(
                    processed.attrs["fingertip_points_policy_id"], FINGERTIP_POLICY_ID
                )
                self.assertRegex(
                    processed.attrs["fingertip_points_geometry_sha256"],
                    r"^[0-9a-f]{64}$",
                )
            validate_processed_hdf5(processed_path, config)
            zarr_path = root / "policy.zarr"
            report = export_processed_hdf5_to_zarr(
                processed_root,
                zarr_path,
                PolicyZarrExportConfig(expected_task_name="pick"),
            )
            self.assertEqual(report["episode_count"], 1)
            exported = zarr.open_group(str(zarr_path), mode="r")
            self.assertEqual(
                exported.attrs["action_semantics"],
                "teleop_published_joint_target",
            )
            self.assertNotIn("deployment_equivalent", exported.attrs)
            self.assertEqual(
                exported.attrs["fingertip_points_derivation"],
                FINGERTIP_POINTS_DERIVATION,
            )
            self.assertEqual(
                exported.attrs["fingertip_points_policy_id"], FINGERTIP_POLICY_ID
            )

    def test_export_requires_fingertip_identity_attrs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing_identity.h5"
            self._write_processed_fixture(path, OutputProfile.JOINT)
            with h5py.File(path, "r+") as output:
                for name in (
                    "fingertip_points_derivation",
                    "fingertip_points_policy_id",
                    "fingertip_points_geometry_sha256",
                ):
                    if name in output.attrs:
                        del output.attrs[name]
            with (
                patch("dexmani_real.dataset.export.validate_processed_payload"),
                patch(
                    "dexmani_real.dataset.export.validate_processed_provenance",
                    return_value=object(),
                ),
                patch(
                    "dexmani_real.dataset.export._whole_episode_rejection",
                    return_value=None,
                ),
                self.assertRaisesRegex(ValueError, "fingertip"),
            ):
                _inspect_artifact(path, PolicyZarrExportConfig())

    def test_export_rejects_non_uniform_fingertip_geometry_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.h5"
            second = root / "second.h5"
            self._write_processed_fixture(first, OutputProfile.JOINT)
            self._write_processed_fixture(second, OutputProfile.JOINT)
            with h5py.File(second, "r+") as output:
                output.attrs["fingertip_points_geometry_sha256"] = "b" * 64
            with (
                patch("dexmani_real.dataset.export.validate_processed_payload"),
                patch(
                    "dexmani_real.dataset.export.validate_processed_provenance",
                    return_value=object(),
                ),
                patch(
                    "dexmani_real.dataset.export._whole_episode_rejection",
                    return_value=None,
                ),
                self.assertRaisesRegex(
                    ValueError, "fingertip_points_geometry_sha256 mismatch"
                ),
            ):
                export_processed_hdf5_to_zarr(root, root / "policy.zarr")

    def test_export_rejects_non_boolean_and_non_integer_semantic_attrs(self) -> None:
        cases = (
            ("action_semantics", "deployment_grid_rate_limited_target"),
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
                        patch("dexmani_real.dataset.export.validate_processed_payload"),
                        patch(
                            "dexmani_real.dataset.export.validate_processed_provenance",
                            return_value=object(),
                        ),
                        patch(
                            "dexmani_real.dataset.export._whole_episode_rejection",
                            return_value=None,
                        ),
                    ):
                        with self.assertRaises(ValueError):
                            _inspect_artifact(path, PolicyZarrExportConfig())

    def test_process_rejects_non_boolean_and_non_integer_semantic_attrs(self) -> None:
        cases = (
            ("action_semantics", "deployment_grid_rate_limited_target"),
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
                    with patch(
                        "dexmani_real.dataset.processed.validate_processed_payload"
                    ):
                        with self.assertRaises(ValueError):
                            validate_processed_hdf5(path, config)

    def test_writer_structural_sanity_skips_payload_scan_but_full_verify_does_not(
        self,
    ) -> None:
        config = ProcessingConfig(
            profile=OutputProfile.JOINT,
            horizon=1,
            min_full_windows=1,
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
                    "action_semantics": "teleop_published_joint_target",
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
                    "fingertip_points_derivation": FINGERTIP_POINTS_DERIVATION,
                    "fingertip_points_policy_id": FINGERTIP_POLICY_ID,
                    "fingertip_points_geometry_sha256": "a" * 64,
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

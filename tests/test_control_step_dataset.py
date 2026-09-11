"""Offline end-to-end tests of complete control-step episode processing."""

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from dexmani_real.config.pointcloud import PointCloudConfig
from dexmani_real.dataset.contracts import ProcessingConfig
from dexmani_real.dataset.processed import (
    MULTIMODAL_DATASET_KEYS,
    validate_processed_hdf5,
)
from dexmani_real.dataset.processing import load_annotations, process_episode_root
from dexmani_real.recording.storage.schema import DATASET_SPECS, EPISODE_SCHEMA_VERSION
from dexmani_real.recording.storage.video import VideoEncoder


def permissive_test_config() -> ProcessingConfig:
    """A processing config whose point-cloud policy tolerates the flat synthetic
    plane (16x16, ~25 mm spacing), which the canonical outlier radius would
    reject entirely."""
    return ProcessingConfig(
        pointcloud=replace(
            PointCloudConfig(),
            outlier_min_neighbors=0,
            outlier_min_component_points=1,
        ),
        table_plane_abcd=None,
    )


def write_control_episode(
    root: Path, name: str = "episode_fixture", frames: int = 40
) -> Path:
    """Write current raw with real HDF5/video sidecars, without any hardware."""
    episode = root / name
    episode.mkdir(parents=True)
    anchor = np.uint64(1_025_000_000) + np.arange(frames, dtype=np.uint64) * 62_500_000
    with h5py.File(episode / "data.h5", "w") as source:
        for key, spec in DATASET_SPECS.items():
            source.create_dataset(
                key, data=np.zeros((frames, *spec.tail_shape), dtype=spec.dtype)
            )
        meta = source.create_group("meta")
        meta.attrs.update(
            {
                "schema_version": EPISODE_SCHEMA_VERSION,
                "num_frames": frames,
                "task_label": "fixture",
                "control_hz": 16.0,
                "camera_payload_mode": "depth_to_color_aligned_rgbd",
                "camera_type": "eye_to_hand",
                "depth_scale": 0.001,
                "camera_T_color_from_depth": np.eye(4),
                "camera_T_xarm_base_from_color": np.eye(4),
            }
        )
        for stream in ("color", "depth"):
            meta.attrs.update(
                {
                    f"camera_{stream}_width": 16,
                    f"camera_{stream}_height": 16,
                    f"camera_{stream}_intrinsics": np.array(
                        [20.0, 0.0, 8.0, 0.0, 20.0, 8.0, 0.0, 0.0, 1.0]
                    ),
                    f"camera_{stream}_distortion_model": "none",
                    f"camera_{stream}_distortion_coeffs": np.zeros(5),
                }
            )
        source["timestamp"][:] = anchor / 1e9
        source["source_sample_index"][:] = np.arange(frames)
        source["observation_anchor_monotonic_ns"][:] = anchor
        for sensor in ("arm", "hand", "tactile"):
            source[f"{sensor}_source_monotonic_ns"][:] = anchor - 9_000_000
        source["camera_source_monotonic_ns"][:] = anchor - 25_000_000
        source["hand_contact_source_monotonic_ns"][:] = anchor - 12_000_000
        for field in (
            "tactile_calibrated",
            "tactile_sum_fresh",
            "tactile_fresh",
            "flag_camera_fresh",
        ):
            source[field][:] = True
        source["arm_qpos"][:] = np.arange(frames)[:, None] * 0.001
        source["hand_qpos"][:] = np.arange(12)[None, :] * 0.01
        source["hand_contact"][:] = (
            np.arange(frames)[:, None, None] + np.arange(15).reshape(1, 5, 3) / 100
        )
        source["action_arm_joint_sent"][:] = np.arange(frames)[:, None] * 0.002 + 0.3
        source["action_hand_joint"][:] = np.arange(12)[None, :] * 0.02
        ee = np.zeros((frames, 9))
        ee[:, 3] = 1
        ee[:, 7] = 1
        source["action_arm_ee"][:] = ee
    with h5py.File(episode / "depth.h5", "w") as depth:
        depth.create_dataset(
            "depth", data=np.full((frames, 16, 16), 500, dtype=np.uint16)
        )
    with VideoEncoder(episode / "rgb.mp4", width=16, height=16) as video:
        for row in range(frames):
            video.write_frame(np.full((16, 16, 3), row, dtype=np.uint8))
    return episode


@pytest.mark.parametrize(
    "event",
    ["ordinary", "camera_stale", "newer_tactile", "dense_unavailable", "short_ik"],
)
def test_d1_to_d5_all_rows_and_same_row_contact(tmp_path, event):
    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        if event == "camera_stale":
            raw["flag_camera_fresh"][12] = False
        if event == "dense_unavailable":
            raw["tactile_fresh"][12] = False
            raw["hand_tactile_force"][12] = np.nan
            raw["tactile_source_monotonic_ns"][12] = 0
        if event == "short_ik":
            raw["flag_frame_status"][10:14] = 2
        # The fixture already has camera=1000, tactile=1016, anchor=1025 ms.
        expected_contact = raw["hand_contact"][:].astype(np.float32)
        expected_state = np.concatenate(
            (raw["arm_qpos"][:], raw["hand_qpos"][:]), axis=1
        ).astype(np.float32)
    config = permissive_test_config()
    output = tmp_path / "processed"
    report = process_episode_root(episode, output, config)
    assert report["processed_frame_count"] == 40
    artifact = output / f"{episode.name}.h5"
    validate_processed_hdf5(artifact)
    with h5py.File(artifact, "r") as data:
        assert data.attrs["source_frames"] == data.attrs["episode_steps"] == 40
        assert set(data) == set(MULTIMODAL_DATASET_KEYS)
        np.testing.assert_array_equal(data["contact_force"][:], expected_contact)
        np.testing.assert_array_equal(data["joint_state"][:], expected_state)
        if event == "dense_unavailable":
            assert bool(data["tactile_force_valid"][12]) is False
            assert np.all(np.isnan(data["tactile_force"][12]))


def test_eef_pose_persisted_from_canonical_fk(tmp_path):
    from dexmani_real.planning.kinematics.arm_fk import compute_eef_pose_history_xarm_base

    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r") as raw:
        expected = compute_eef_pose_history_xarm_base(raw["arm_qpos"][:]).astype(
            np.float32
        )
    config = permissive_test_config()
    output = tmp_path / "processed"
    process_episode_root(episode, output, config)
    artifact = output / f"{episode.name}.h5"
    validate_processed_hdf5(artifact)
    with h5py.File(artifact, "r") as data:
        eef = data["eef_pose"]
        assert eef.shape == (40, 9)
        assert eef.dtype == np.float32
        assert np.isfinite(eef[:]).all()
        np.testing.assert_allclose(eef[:], expected, rtol=1e-6, atol=1e-6)
        assert data.attrs["eef_pose_frame"] == "xarm_base"
        assert data.attrs["eef_pose_components"] == "position_m(3)+rot6d(6)"
        # The rot6d tail is two orthonormal columns of a rotation matrix.
        rot6d = eef[0, 3:]
        assert rot6d.shape == (6,)


def test_contact_and_dense_validity_are_independent(tmp_path):
    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        # Dense tactile fully failed at row 5 (stale flag, NaN payload, zero
        # source); the usable aggregate contact must stay valid anyway.
        raw["tactile_fresh"][5] = False
        raw["hand_tactile_force"][5] = np.nan
        raw["tactile_source_monotonic_ns"][5] = 0
    output = tmp_path / "processed"
    process_episode_root(episode, output, permissive_test_config())
    with h5py.File(output / f"{episode.name}.h5", "r") as data:
        assert bool(data["contact_force_valid"][5]) is True
        assert bool(data["tactile_force_valid"][5]) is False


def test_calibrated_native_unit_contact_is_valid(tmp_path):
    """Canonical row: finite payload, positive contact source, successful
    software calibration, and native unit identity is valid."""
    episode = write_control_episode(tmp_path / "raw")
    output = tmp_path / "processed"
    process_episode_root(episode, output, permissive_test_config())
    with h5py.File(output / f"{episode.name}.h5", "r") as data:
        assert bool(np.all(data["tactile_calibrated"][:]))
        assert np.all(data["tactile_unit_code"][:] == 0)
        assert bool(np.all(data["contact_force_valid"][:]))


def test_zero_contact_is_not_intrinsically_invalid(tmp_path):
    """Zero can be a real no-contact reading; only provenance and
    representation flags decide, never payload magnitude."""
    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["hand_contact"][:] = 0.0
    output = tmp_path / "processed"
    process_episode_root(episode, output, permissive_test_config())
    with h5py.File(output / f"{episode.name}.h5", "r") as data:
        assert bool(np.all(data["contact_force_valid"][:]))
        assert np.all(data["contact_force"][:] == 0.0)


def test_uncalibrated_contact_is_invalid_but_payload_preserved(tmp_path):
    """SDK payload validity is not software calibration state: a finite,
    well-sourced row recorded while calibration had failed cannot satisfy the
    declared bias-corrected representation, but its payload stays archived
    unchanged."""
    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["tactile_calibrated"][11] = False
        expected_contact = raw["hand_contact"][11].astype(np.float32)
    output = tmp_path / "processed"
    process_episode_root(episode, output, permissive_test_config())
    with h5py.File(output / f"{episode.name}.h5", "r") as data:
        assert bool(data["contact_force_valid"][11]) is False
        assert bool(data["contact_force_valid"][10]) is True
        np.testing.assert_array_equal(data["contact_force"][11], expected_contact)


def test_wrong_unit_contact_is_invalid(tmp_path):
    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["tactile_unit_code"][13] = 1
    output = tmp_path / "processed"
    process_episode_root(episode, output, permissive_test_config())
    with h5py.File(output / f"{episode.name}.h5", "r") as data:
        assert bool(data["contact_force_valid"][13]) is False
        assert bool(data["contact_force_valid"][12]) is True


def test_d6_persistent_ik_rejects_whole_episode(tmp_path):
    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["flag_frame_status"][10:15] = 2
    output = tmp_path / "processed"
    config = permissive_test_config()
    report = process_episode_root(episode, output, config, dry_run=True)
    assert report["processed_frame_count"] == 0
    assert report["episodes"][0]["rejected_reason"] == "persistent IK failure"
    with pytest.raises(ValueError, match="persistent IK failure"):
        process_episode_root(episode, output, config)
    assert not output.exists()


def test_d7_exact_sent_action(tmp_path):
    episode = write_control_episode(tmp_path / "raw")
    output = tmp_path / "processed"
    process_episode_root(episode, output, permissive_test_config())
    with (
        h5py.File(episode / "data.h5", "r") as raw,
        h5py.File(output / f"{episode.name}.h5", "r") as processed,
    ):
        expected = np.concatenate(
            (raw["action_arm_joint_sent"][:], raw["action_hand_joint"][:]), axis=1
        ).astype(np.float32)
        intent = np.concatenate(
            (raw["action_arm_ee"][:], raw["action_hand_joint"][:]), axis=1
        ).astype(np.float32)
        np.testing.assert_array_equal(processed["action"][:], expected)
        np.testing.assert_array_equal(processed["action_ee"][:], intent)


@pytest.mark.parametrize(
    ("corruption", "match"),
    [
        ("future_source", "source must be positive and causal"),
        ("sample_identity", "raw sample identity"),
        ("nan_action", "NaN/Inf"),
    ],
)
def test_technical_corruption_fails_the_batch(tmp_path, corruption, match):
    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        if corruption == "future_source":
            raw["hand_source_monotonic_ns"][5] = raw["observation_anchor_monotonic_ns"][
                5
            ] + np.uint64(1)
        elif corruption == "sample_identity":
            raw["source_sample_index"][5] = 3
        else:
            raw["action_arm_joint_sent"][5, 0] = np.nan
    output = tmp_path / "processed"
    with pytest.raises(ValueError, match=match):
        process_episode_root(
            episode, output, permissive_test_config()
        )
    assert not output.exists()


def test_missing_required_field_fails_the_batch(tmp_path):
    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        del raw["hand_tactile_force"]
    output = tmp_path / "processed"
    # The v28 reader proves dataset layout completeness before analysis; the
    # missing field must fail the whole batch, never become a silent rejection.
    with pytest.raises(ValueError, match="episode validity"):
        process_episode_root(episode, output, permissive_test_config())
    assert not output.exists()


def test_annotation_ranges_are_not_supported(tmp_path):
    annotation = tmp_path / "annotations.yaml"
    annotation.write_text("episode_fixture:\n  exclude_ranges: [[4, 8]]\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_annotations(annotation)


@pytest.mark.parametrize("key", ["rgb", "depth"])
def test_processed_validator_reads_compressed_image_payload(tmp_path, key):
    episode = write_control_episode(tmp_path / "raw")
    output = tmp_path / "processed"
    process_episode_root(
        episode,
        output,
        permissive_test_config(),
    )
    artifact = output / f"{episode.name}.h5"
    with h5py.File(artifact, "r+") as processed:
        dataset = processed[key]
        # Damage one compressed chunk while leaving schema/shape/dtype intact.
        dataset.id.write_direct_chunk((0,) * dataset.ndim, b"broken zlib payload")
    with pytest.raises(OSError):
        validate_processed_hdf5(artifact)

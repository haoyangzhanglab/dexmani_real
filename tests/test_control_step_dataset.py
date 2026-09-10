"""Offline end-to-end tests of complete control-step episode processing."""

from pathlib import Path

import h5py
import numpy as np
import pytest

from dexmani_real.dataset.contracts import OutputProfile, ProcessingConfig
from dexmani_real.dataset.processed import validate_processed_hdf5
from dexmani_real.dataset.processing import load_annotations, process_episode_root
from dexmani_real.recording.storage.schema import DATASET_SPECS, EPISODE_SCHEMA_VERSION
from dexmani_real.recording.storage.video import VideoEncoder


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


@pytest.mark.parametrize("profile", [OutputProfile.JOINT, OutputProfile.RGB])
@pytest.mark.parametrize(
    "event",
    ["ordinary", "camera_stale", "newer_tactile", "dense_unavailable", "short_ik"],
)
def test_d1_to_d5_all_rows_and_same_row_contact(tmp_path, profile, event):
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
    config = ProcessingConfig(
        profile=profile, target_rgb_height=16, target_rgb_width=16
    )
    output = tmp_path / "processed"
    report = process_episode_root(episode, output, config)
    assert report["processed_frame_count"] == 40
    artifact = output / f"{episode.name}.h5"
    validate_processed_hdf5(artifact)
    with h5py.File(artifact, "r") as data:
        assert data.attrs["source_frames"] == data.attrs["episode_steps"] == 40
        assert set(data) == set(profile.dataset_keys)
        np.testing.assert_array_equal(data["contact_force"][:], expected_contact)
        np.testing.assert_array_equal(data["joint_state"][:], expected_state)


def test_d6_persistent_ik_rejects_whole_episode(tmp_path):
    episode = write_control_episode(tmp_path / "raw")
    with h5py.File(episode / "data.h5", "r+") as raw:
        raw["flag_frame_status"][10:15] = 2
    output = tmp_path / "processed"
    config = ProcessingConfig(profile=OutputProfile.JOINT)
    report = process_episode_root(episode, output, config, dry_run=True)
    assert report["processed_frame_count"] == 0
    assert report["episodes"][0]["rejected_reason"] == "persistent IK failure"
    with pytest.raises(ValueError, match="persistent IK failure"):
        process_episode_root(episode, output, config)
    assert not output.exists()


def test_d7_exact_sent_action(tmp_path):
    episode = write_control_episode(tmp_path / "raw")
    output = tmp_path / "processed"
    process_episode_root(episode, output, ProcessingConfig(profile=OutputProfile.JOINT))
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
    "corruption", ["future_source", "sample_identity", "nan_action"]
)
def test_technical_corruption_has_no_published_artifact(tmp_path, corruption):
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
    with pytest.raises(ValueError, match="no output published"):
        process_episode_root(
            episode, output, ProcessingConfig(profile=OutputProfile.JOINT)
        )
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
        ProcessingConfig(
            profile=OutputProfile.RGB, target_rgb_height=16, target_rgb_width=16
        ),
    )
    artifact = output / f"{episode.name}.h5"
    with h5py.File(artifact, "r+") as processed:
        dataset = processed[key]
        # Damage one compressed chunk while leaving schema/shape/dtype intact.
        dataset.id.write_direct_chunk((0,) * dataset.ndim, b"broken zlib payload")
    with pytest.raises(OSError):
        validate_processed_hdf5(artifact)

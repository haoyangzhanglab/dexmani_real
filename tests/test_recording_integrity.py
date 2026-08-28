"""Focused offline checks for raw-v24 semantic and sidecar integrity boundaries."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import h5py
import numpy as np

from dexmani_real.recording.camera_writer import CameraStreamWriterConfig
from dexmani_real.recording.reader import EpisodeReader, ValidityState
from dexmani_real.recording.recorder import EpisodeRecorder
from dexmani_real.recording.schema import (
    CAMERA_TIMING_DATASET_SPECS,
    DATASET_SPECS,
    validate_camera_metadata_keys,
    validate_raw_member_hashes,
    validate_raw_semantics,
)
from dexmani_real.recording.timeline import TimestampAlignedBuffer
from dexmani_real.recording.video import VideoEncoderConfig
from dexmani_real.robot.types import RobotAction, RobotState


def _semantic_arrays(frame_count: int = 2) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, spec in {**DATASET_SPECS, **CAMERA_TIMING_DATASET_SPECS}.items():
        shape = (frame_count,) + spec.tail_shape
        if np.issubdtype(spec.dtype, np.floating):
            arrays[name] = np.full(shape, np.nan, dtype=spec.dtype)
        elif name == "source_sample_index":
            arrays[name] = np.full(shape, -1, dtype=spec.dtype)
        else:
            arrays[name] = np.zeros(shape, dtype=spec.dtype)

    arrays["timestamp"] = np.arange(frame_count, dtype=np.float64) + 1.0
    arrays["flag_sample_valid"] = np.ones(frame_count, dtype=bool)
    arrays["source_sample_index"] = np.arange(frame_count, dtype=np.int64)
    arrays["source_timestamp"] = arrays["timestamp"].copy()
    arrays["fill_reason"] = np.zeros(frame_count, dtype=np.uint8)
    arrays["observation_anchor_monotonic_ns"] = np.asarray(
        [2_000_000_000, 3_000_000_000], dtype=np.int64
    )
    arrays["observation_id"] = np.arange(frame_count, dtype=np.int64) + 1
    arrays["action_id"] = np.arange(frame_count, dtype=np.int64) + 1
    arrays["action_created_monotonic_ns"] = np.full(frame_count, 100, dtype=np.int64)
    arrays["action_target_monotonic_ns"] = np.full(frame_count, 200, dtype=np.int64)
    arrays["action_valid_until_monotonic_ns"] = np.full(
        frame_count, 300, dtype=np.int64
    )
    arrays["flag_action_queued"] = np.ones(frame_count, dtype=bool)
    arrays["action_arm_ee"][:] = np.asarray(
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    )
    arrays["arm_ee"][:] = arrays["action_arm_ee"]
    for name in (
        "action_arm_joint",
        "action_hand_joint",
        "action_arm_joint_raw",
        "action_hand_joint_raw",
    ):
        arrays[name].fill(0.0)
    return arrays


def _state(timestamp: float) -> RobotState:
    return RobotState(
        arm_qpos=np.zeros(7),
        arm_qvel=np.zeros(7),
        arm_tau=np.zeros(7),
        eef_pos=np.zeros(3),
        eef_quat_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        eef_rot6d=np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
        hand_qpos=np.zeros(12),
        hand_tactile_sum=np.zeros((5, 3)),
        hand_tactile_force=np.zeros((5, 120, 3)),
        hand_tactile_contact=np.zeros(5, dtype=bool),
        hand_tipboard_err=np.zeros(12, dtype=np.int32),
        hand_commboard_err=np.zeros(12, dtype=np.int32),
        hand_jointboard_err=np.zeros(12, dtype=np.int32),
        hand_qpos_stale=False,
        fingertip_pos=np.zeros((5, 3)),
        arm_connected=True,
        hand_connected=True,
        timestamp=timestamp,
        hand_current=np.zeros(12),
    )


def _action(*, canonical: bool = True) -> RobotAction:
    rot6d = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    if not canonical:
        rot6d[0] = 2.0
    return RobotAction(
        arm_qpos_cmd=np.zeros(7),
        hand_qpos_cmd=np.zeros(12),
        target_eef_pos=np.zeros(3),
        target_eef_rot6d=rot6d,
    )


def _vr_frame() -> dict[str, np.ndarray]:
    return {
        "wrist_pos": np.zeros(3),
        "wrist_quat_wxyz": np.asarray([1.0, 0.0, 0.0, 0.0]),
        "landmarks": np.zeros((21, 3)),
    }


def _action_signals() -> dict[str, int | bool]:
    return {
        "observation_id": 1,
        "action_id": 1,
        "action_queued": True,
        "action_created_monotonic_ns": 100,
        "action_target_monotonic_ns": 200,
        "action_valid_until_monotonic_ns": 300,
    }


class RawSemanticValidationTest(unittest.TestCase):
    def test_source_timestamp_tracks_grid_anchor_and_hold(self) -> None:
        buffer = TimestampAlignedBuffer(start_time=0.0, dt=1.0, max_record_steps=8)
        buffer.add({"value": np.asarray(1.0)}, timestamp=0.2)
        buffer.add({"value": np.asarray(2.0)}, timestamp=2.2)
        np.testing.assert_array_equal(buffer.timestamps, [0.0, 1.0, 2.0, 3.0])
        np.testing.assert_array_equal(
            buffer.data["source_timestamp"], [np.nan, 1.0, 1.0, 3.0]
        )

    def test_source_hold_and_rot6d_faults_are_rejected(self) -> None:
        arrays = _semantic_arrays()
        arrays["source_sample_index"][1] = 0
        errors = validate_raw_semantics(arrays, frame_count=2)
        self.assertTrue(any("strictly increase" in error for error in errors))

        arrays = _semantic_arrays()
        arrays["action_arm_ee"][0, 3] = 2.0
        errors = validate_raw_semantics(arrays, frame_count=2)
        self.assertTrue(any("non-canonical rot6d" in error for error in errors))

    def test_camera_metadata_cannot_replace_fixed_semantics(self) -> None:
        errors = validate_camera_metadata_keys(
            {
                "camera_frame_gap_semantics": "caller-definition",
                "camera_frame_gap_admission_policy": "caller-policy",
                "camera_firmware": "firmware-1",
            }
        )
        self.assertEqual(len(errors), 2)


class RawMemberManifestValidationTest(unittest.TestCase):
    @staticmethod
    def _attrs() -> dict[str, object]:
        depth_hash = "a" * 64
        rgb_hash = "b" * 64
        return {
            "raw_manifest_version": 1,
            "depth_sha256": depth_hash,
            "rgb_sha256": rgb_hash,
            "raw_member_sha256_json": json.dumps(
                {"depth.h5": depth_hash, "rgb.mp4": rgb_hash},
                sort_keys=True,
            ),
        }

    @staticmethod
    def _computed_hashes() -> dict[str, str]:
        return {"depth.h5": "a" * 64, "rgb.mp4": "b" * 64}

    def test_exact_manifest_and_dedicated_hashes_are_valid(self) -> None:
        self.assertEqual(
            validate_raw_member_hashes(self._attrs(), self._computed_hashes()), ()
        )

    def test_manifest_requires_json_and_dedicated_attrs(self) -> None:
        attrs = self._attrs()
        del attrs["raw_member_sha256_json"]
        errors = validate_raw_member_hashes(attrs, self._computed_hashes())
        self.assertTrue(any("raw_member_sha256_json" in error for error in errors))

        attrs = self._attrs()
        del attrs["depth_sha256"]
        errors = validate_raw_member_hashes(attrs, self._computed_hashes())
        self.assertTrue(
            any("depth_sha256" in error and "missing" in error for error in errors)
        )

    def test_manifest_rejects_missing_or_extra_json_keys(self) -> None:
        attrs = self._attrs()
        manifest = json.loads(attrs["raw_member_sha256_json"])
        del manifest["depth.h5"]
        attrs["raw_member_sha256_json"] = json.dumps(manifest)
        errors = validate_raw_member_hashes(attrs, self._computed_hashes())
        self.assertTrue(
            any("exactly" in error and "missing" in error for error in errors)
        )

        attrs = self._attrs()
        manifest = json.loads(attrs["raw_member_sha256_json"])
        manifest["unexpected.bin"] = "c" * 64
        attrs["raw_member_sha256_json"] = json.dumps(manifest)
        errors = validate_raw_member_hashes(attrs, self._computed_hashes())
        self.assertTrue(
            any("exactly" in error and "extra" in error for error in errors)
        )

    def test_manifest_rejects_malformed_json(self) -> None:
        attrs = self._attrs()
        attrs["raw_member_sha256_json"] = "{"  # Truncated object.
        errors = validate_raw_member_hashes(attrs, self._computed_hashes())
        self.assertTrue(
            any(
                "raw_member_sha256_json" in error and "invalid" in error
                for error in errors
            )
        )

    def test_manifest_rejects_json_and_computed_hash_mismatches(self) -> None:
        attrs = self._attrs()
        manifest = json.loads(attrs["raw_member_sha256_json"])
        manifest["depth.h5"] = "c" * 64
        attrs["raw_member_sha256_json"] = json.dumps(manifest)
        errors = validate_raw_member_hashes(attrs, self._computed_hashes())
        self.assertTrue(
            any("disagrees with its fixed attr" in error for error in errors)
        )

        errors = validate_raw_member_hashes(
            self._attrs(), {"depth.h5": "c" * 64, "rgb.mp4": "b" * 64}
        )
        self.assertTrue(any("SHA-256 mismatch" in error for error in errors))


class RawFinalizationIntegrityTest(unittest.TestCase):
    @staticmethod
    def _recorder(root: str) -> EpisodeRecorder:
        return EpisodeRecorder(
            root,
            max_frames=4,
            control_hz=1.0,
            min_frames=1,
            camera_writer_config=CameraStreamWriterConfig(
                rgb_shape=(1, 1, 3),
                depth_shape=(1, 1),
                fps=1.0,
                queue_size=2,
                video=VideoEncoderConfig(),
            ),
            resolved_config_hash="a" * 64,
        )

    def test_noncanonical_action_is_faulted_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(root)
            self.assertTrue(recorder.start_episode())
            self.assertTrue(
                recorder.add_frame(
                    _state(time.perf_counter()),
                    _action(canonical=False),
                    _vr_frame(),
                    signals=_action_signals(),
                )
            )
            recorder.stop_episode(success=True)
            self.assertFalse(recorder.join_stop(timeout=30.0))
            self.assertIn("raw semantic validation failed", recorder.stop_error or "")
            self.assertEqual(
                tuple(path for path in Path(root).glob("episode_*") if path.is_dir()),
                (),
            )
            aborted = tuple(Path(root).glob("*.aborted.json"))
            self.assertEqual(len(aborted), 1)
            payload = json.loads(aborted[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "aborted")

    def test_sidecar_content_change_fails_reader_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(root)
            self.assertTrue(recorder.start_episode())
            self.assertTrue(
                recorder.add_frame(
                    _state(time.perf_counter()),
                    _action(),
                    _vr_frame(),
                    signals=_action_signals(),
                )
            )
            path = Path(recorder.stop_episode(success=True) or "")
            self.assertTrue(recorder.join_stop(timeout=30.0))
            with EpisodeReader(path) as reader:
                self.assertEqual(reader.schema_version, 24)
                self.assertIs(reader.validity, ValidityState.VALID)
            with h5py.File(path / "depth.h5", "r+") as depth_h5:
                depth_h5["depth"][0, 0, 0] = 1
            with self.assertRaisesRegex(ValueError, "manifest validation failed"):
                EpisodeReader(path)

            # The HDF5 mutation does not change length, so the failure proves
            # content anchoring rather than ordinary sidecar length checking.
            with h5py.File(path / "depth.h5", "r") as depth_h5:
                self.assertEqual(depth_h5["depth"].shape[0], 1)

    def test_v23_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            recorder = self._recorder(root)
            self.assertTrue(recorder.start_episode())
            self.assertTrue(
                recorder.add_frame(
                    _state(time.perf_counter()),
                    _action(),
                    _vr_frame(),
                    signals=_action_signals(),
                )
            )
            path = Path(recorder.stop_episode(success=True) or "")
            self.assertTrue(recorder.join_stop(timeout=30.0))
            with h5py.File(path / "data.h5", "r+") as data_h5:
                meta = data_h5["meta"]
                meta.attrs["schema_version"] = 23
            with self.assertRaisesRegex(ValueError, "unsupported episode schema v23"):
                EpisodeReader(path)


if __name__ == "__main__":
    unittest.main()

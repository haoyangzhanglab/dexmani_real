"""Offline integrity-boundary tests for transactional raw episode reads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

from dexmani_real.ipc.schema import make_record_sample_dtype
from dexmani_real.recording.client import RecorderClient
from dexmani_real.recording.reader import EpisodeReader, ValidityState
from dexmani_real.recording.schema import EPISODE_SCHEMA_VERSION
from dexmani_real.robot.types import RobotAction, RobotState


class EpisodeReaderIntegrityTest(unittest.TestCase):
    def _write_minimal_episode(self, root: Path) -> Path:
        episode = root / "episode"
        episode.mkdir()
        with h5py.File(episode / "data.h5", "w") as data:
            meta = data.create_group("meta")
            meta.attrs.update(
                {
                    "schema_version": EPISODE_SCHEMA_VERSION,
                    "num_frames": 1,
                    "resolved_config_sha256": "a" * 64,
                    "success": True,
                    "camera_writer_error": "",
                    "camera_encoding_height": 2,
                    "camera_encoding_width": 3,
                }
            )
        with h5py.File(episode / "depth.h5", "w") as depth:
            depth.create_dataset("depth", data=np.zeros((1, 2, 3), dtype=np.uint16))
        (episode / "rgb.mp4").touch()
        return episode

    def test_default_read_checks_structure_without_hashing_or_full_video_decode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            decoder = MagicMock()
            with (
                patch(
                    "dexmani_real.recording.reader.VideoDecoder", return_value=decoder
                ),
                patch(
                    "dexmani_real.recording.reader.validate_data_layout",
                    return_value=(),
                ),
                patch(
                    "dexmani_real.recording.reader.validate_raw_semantics",
                    return_value=(),
                ),
                patch("dexmani_real.recording.reader.sha256_file") as hash_file,
            ):
                with EpisodeReader(
                    self._write_minimal_episode(Path(directory))
                ) as reader:
                    self.assertIs(reader.validity, ValidityState.VALID)
                hash_file.assert_not_called()
                decoder.count_decoded_frames.assert_not_called()

    def test_hash_audit_is_explicit_and_verify_hash_runs_it_at_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            decoder = MagicMock()
            decoder.count_decoded_frames.return_value = 1
            with (
                patch(
                    "dexmani_real.recording.reader.VideoDecoder", return_value=decoder
                ),
                patch(
                    "dexmani_real.recording.reader.validate_data_layout",
                    return_value=(),
                ),
                patch(
                    "dexmani_real.recording.reader.validate_raw_semantics",
                    return_value=(),
                ),
                patch(
                    "dexmani_real.recording.reader.validate_raw_member_hashes",
                    return_value=(),
                ),
                patch(
                    "dexmani_real.recording.reader.sha256_file",
                    return_value="b" * 64,
                ) as hash_file,
            ):
                with EpisodeReader(
                    self._write_minimal_episode(Path(directory)), verify_hash=True
                ):
                    pass
                self.assertEqual(hash_file.call_count, 2)
                decoder.count_decoded_frames.assert_called_once_with()

    def test_default_read_rejects_raw_semantic_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch("dexmani_real.recording.reader.VideoDecoder"),
                patch(
                    "dexmani_real.recording.reader.validate_data_layout",
                    return_value=(),
                ),
                patch(
                    "dexmani_real.recording.reader.validate_raw_semantics",
                    return_value=("invalid source timestamps",),
                ),
            ):
                with self.assertRaises(ValueError):
                    EpisodeReader(self._write_minimal_episode(Path(directory)))

    def test_recorder_sample_uses_explicit_control_generation(self) -> None:
        class SampleRing:
            latest_sequence = 0
            maxlen = 8
            dtype = make_record_sample_dtype((1, 1, 3), (1, 1))

            def __init__(self) -> None:
                self.frame = None

            def write(self, frame) -> None:
                self.frame = frame.copy()

        class Shared:
            record_sample_ring = SampleRing()
            recorder_consumed_sequence = SimpleNamespace(value=0)

            @property
            def run_generation(self):
                raise AssertionError("RecorderClient must not reread global generation")

        state = RobotState(
            arm_qpos=np.zeros(7),
            arm_qvel=np.zeros(7),
            arm_tau=np.zeros(7),
            eef_pos=np.zeros(3),
            eef_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            eef_rot6d=np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
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
            timestamp=1.0,
        )
        action = RobotAction(np.zeros(7), np.zeros(12))
        client = RecorderClient(Shared())
        client._recording = True
        client._generation = 1
        self.assertTrue(
            client.add_frame(
                state,
                action,
                {
                    "wrist_pos": np.zeros(3),
                    "wrist_quat_wxyz": np.array([1.0, 0.0, 0.0, 0.0]),
                    "landmarks": np.zeros((21, 3)),
                },
                control_run_generation=7,
            )
        )
        self.assertEqual(
            int(client.shared.record_sample_ring.frame["control_run_generation"][0]),
            7,
        )


if __name__ == "__main__":
    unittest.main()

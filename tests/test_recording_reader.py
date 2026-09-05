"""Offline integrity-boundary tests for transactional raw episode reads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

from dexmani_real.recording.reader import EpisodeReader, ValidityState
from dexmani_real.recording.schema import EPISODE_SCHEMA_VERSION


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


if __name__ == "__main__":
    unittest.main()

"""Raw 字段、来源时效、重复帧和拒绝时停止的离线重放契约。"""

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
from numpy.testing import assert_array_equal
from test_hand_retargeting import make_retargeter

from dexmani_real.config.real import RealConfig
from dexmani_real.robot.model import XHAND_SDK_JOINT_NAMES
from scripts.tools.evaluate_hand_retargeting import evaluate, read_episode


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        retargeter, self.home, points = make_retargeter()
        self.calibration = retargeter.calibration_metadata()
        self.runtime = replace(RealConfig(), hand_retargeting=retargeter.config)
        self.frames = np.repeat(points[None], 3, axis=0)
        self.stamps = np.array([1_000_000_000, 1_062_500_000, 1_125_000_000])
        self.sequences = np.arange(3)

    def evaluate(self, *, source=None, sequences=None):
        return evaluate(
            self.frames, self.stamps, self.stamps if source is None else source,
            self.sequences if sequences is None else sequences, self.home,
            self.runtime, self.calibration,
        )

    def test_saved_calibration_and_duplicate_frames_are_reused(self):
        result = self.evaluate(source=self.stamps[[0, 0, 2]], sequences=np.array([0, 0, 1]))
        self.assertEqual(result["calibration_source"], "recorded_metadata")
        self.assertEqual(result["cached_frames"], 1)
        self.assertEqual(result["completed_frames"], 3)

    def test_stale_input_and_shape_mismatch_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "stale"):
            self.evaluate(source=np.array([1, 1, 1]))
        with self.assertRaisesRegex(ValueError, "matching"):
            self.evaluate(sequences=self.sequences[:2])

    def test_reused_sequence_number_with_new_source_time_is_a_new_sample(self):
        result = self.evaluate(sequences=np.zeros(3, dtype=np.int64))
        self.assertEqual(result["cached_frames"], 0)
        self.assertEqual(result["completed_frames"], 3)

    def test_first_failure_stops_replay_instead_of_skipping_bad_frames(self):
        self.frames[1] = 0
        result = self.evaluate()
        self.assertEqual(result["completed_frames"], 1)
        self.assertEqual(result["rejected"], 1)
        self.assertTrue(result["rejection_reasons"])

    def test_raw_reader_restores_exact_input_and_morphology(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            with h5py.File(path, "w") as data:
                meta = data.create_group("meta")
                meta.attrs.update(
                    vr_position_units="m", vr_coordinate_axes="FLU", vr_hand_side="right",
                    joint_units="rad", hand_joint_names=json.dumps(XHAND_SDK_JOINT_NAMES),
                    hand_retargeting_calibration_json=json.dumps(self.calibration),
                )
                data["vr_landmarks"] = self.frames
                rows = np.zeros(3, dtype=[("observation_ns", "<i8")])
                rows["observation_ns"] = self.stamps
                data["timing/rows"] = rows
                data["vr_recv_ts_ns"] = self.stamps
                data["vr_sequence_id"] = self.sequences
                data["hand_qpos"] = np.repeat(self.home[None], 3, axis=0)
            frames, stamps, source, sequence, initial, calibration = read_episode(Path(directory))
            assert_array_equal(frames, self.frames)
            assert_array_equal(stamps, self.stamps)
            assert_array_equal(source, self.stamps)
            assert_array_equal(sequence, self.sequences)
            assert_array_equal(initial, self.home)
            self.assertEqual(calibration, self.calibration)
            with h5py.File(path, "a") as data:
                data["meta"].attrs["vr_position_units"] = "mm"
            with self.assertRaisesRegex(ValueError, "vr_position_units"):
                read_episode(path)


if __name__ == "__main__":
    unittest.main()

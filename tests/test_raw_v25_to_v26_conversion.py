"""Offline tests for the raw v25 -> v26 tactile representation migration.

The fixture uses only the published file layout and the frozen v25 dataset
names, mirroring the v24->v25 conversion test.  Pins the exact ``* 10`` scale
correction for direct v25, conservative ``tactile_sum_fresh`` population, the
explicit-scale requirement for v24-derived sources, and transactional cleanup.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np

_CONVERTER_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "convert_raw_v25_to_v26_tactile.py"
)
_SPEC = importlib.util.spec_from_file_location("raw_v25_to_v26", _CONVERTER_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"cannot load converter module from {_CONVERTER_PATH}")
_CONVERTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONVERTER)

_FRAME_COUNT = 3
# Scalar (tail_shape == ()) datasets need a first-dimension only.
_SCALAR_DATASETS = {
    "timestamp",
    "source_sample_index",
    "fill_reason",
    "flag_sample_valid",
    "arm_connected",
    "hand_connected",
    "hand_qpos_stale",
    "tracking_error",
    "arm_last_cmd_seq",
    "flag_action_queued",
    "flag_frame_status",
    "observation_anchor_monotonic_ns",
    "observation_valid",
    "arm_source_monotonic_ns",
    "hand_source_monotonic_ns",
    "tactile_source_monotonic_ns",
    "vr_source_monotonic_ns",
    "camera_source_monotonic_ns",
    "tactile_fresh",
    "tactile_calibrated",
    "tactile_unit_code",
    "flag_camera_fresh",
    "camera_depth_frame_number",
    "camera_color_frame_number",
    "policy_observation_valid",
}
_TAIL_SHAPES = {
    "hand_contact": (5, 3),
    "hand_tactile_force": (5, 120, 3),
    "arm_qpos": (7,),
    "arm_qvel": (7,),
    "arm_tau": (7,),
    "hand_qpos": (12,),
    "hand_current": (12,),
    "action_arm_joint_sent": (7,),
    "action_hand_joint": (12,),
    "action_arm_ee": (9,),
    "policy_observation_arm_qpos": (7,),
    "policy_observation_hand_qpos": (12,),
    "vr_wrist_pos": (3,),
    "vr_wrist_rot6d": (6,),
    "vr_landmarks": (21, 3),
    "head_quat_wxyz": (4,),
}
# Frozen v25 dtype contract: explicit, not derived from the current v26 schema.
V25_DTYPES = {
    "timestamp": np.float64,
    "source_sample_index": np.int64,
    "fill_reason": np.uint8,
    "flag_sample_valid": np.bool_,
    "arm_qpos": np.float64,
    "arm_qvel": np.float64,
    "arm_tau": np.float64,
    "hand_qpos": np.float64,
    "hand_current": np.float64,
    "hand_contact": np.float64,
    "hand_tactile_force": np.float64,
    "arm_connected": np.bool_,
    "hand_connected": np.bool_,
    "hand_qpos_stale": np.bool_,
    "tracking_error": np.float64,
    "arm_last_cmd_seq": np.int64,
    "action_arm_joint_sent": np.float64,
    "action_hand_joint": np.float64,
    "action_arm_ee": np.float64,
    "flag_action_queued": np.bool_,
    "flag_frame_status": np.uint8,
    "observation_anchor_monotonic_ns": np.uint64,
    "observation_valid": np.bool_,
    "arm_source_monotonic_ns": np.uint64,
    "hand_source_monotonic_ns": np.uint64,
    "tactile_source_monotonic_ns": np.uint64,
    "vr_source_monotonic_ns": np.uint64,
    "camera_source_monotonic_ns": np.uint64,
    "tactile_fresh": np.bool_,
    "tactile_calibrated": np.bool_,
    "tactile_unit_code": np.uint8,
    "flag_camera_fresh": np.bool_,
    "camera_depth_frame_number": np.uint64,
    "camera_color_frame_number": np.uint64,
    "policy_observation_arm_qpos": np.float64,
    "policy_observation_hand_qpos": np.float64,
    "policy_observation_valid": np.bool_,
    "vr_wrist_pos": np.float64,
    "vr_wrist_rot6d": np.float64,
    "vr_landmarks": np.float64,
    "head_quat_wxyz": np.float64,
}


def _tail_shape(name: str) -> tuple[int, ...]:
    if name in _TAIL_SHAPES:
        return _TAIL_SHAPES[name]
    return ()


class TestRawV25ToV26Conversion(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_episode(
        self,
        relative: str = "source",
        *,
        converted_from_schema: int | None = None,
        missing: str | None = None,
    ) -> Path:
        episode = self.root / relative
        episode.mkdir(parents=True, exist_ok=True)
        attrs = {
            "schema_version": 25,
            "num_frames": _FRAME_COUNT,
            "task_label": "fixture_task",
            "operator": "fixture_operator",
            "control_hz": 20.0,
            "fps": 20.0,
            "duration": 0.15,
        }
        if converted_from_schema is not None:
            attrs["converted_from_schema"] = converted_from_schema
        with h5py.File(episode / "data.h5", "w") as data:
            meta = data.create_group("meta")
            for key, value in attrs.items():
                meta.attrs[key] = value
            for name in _CONVERTER.V25_DATASETS:
                if name == missing:
                    continue
                shape = (_FRAME_COUNT,) + _tail_shape(name)
                dtype = V25_DTYPES[name]
                if name == "timestamp":
                    values = np.arange(_FRAME_COUNT, dtype=np.float64) + 1.25
                elif name == "hand_contact":
                    values = np.full(shape, 1.0, dtype=dtype)
                elif name == "hand_tactile_force":
                    values = np.full(shape, 2.0, dtype=dtype)
                elif name == "tactile_unit_code":
                    values = np.zeros(_FRAME_COUNT, dtype=dtype)
                elif name == "flag_frame_status":
                    values = np.zeros(_FRAME_COUNT, dtype=dtype)
                elif np.issubdtype(dtype, np.bool_):
                    values = np.array([True, False, True], dtype=dtype)
                elif np.issubdtype(dtype, np.integer):
                    values = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
                else:
                    values = (
                        np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
                        + 0.5
                    ).astype(dtype)
                data.create_dataset(name, data=values)
        with h5py.File(episode / "depth.h5", "w") as depth_file:
            depth_file.create_dataset(
                "depth",
                data=np.arange(_FRAME_COUNT * 4, dtype=np.uint16).reshape(
                    _FRAME_COUNT, 2, 2
                ),
            )
        (episode / "rgb.mp4").write_bytes(b"synthetic-rgb-payload")
        return episode

    def _convert(self, source, destination, scale=None):
        return _CONVERTER.convert_episode(source, destination, scale)

    def test_direct_v25_scales_tactile_by_ten_and_copies_freshness(self) -> None:
        source = self._write_episode()
        destination = self.root / "converted"
        self._convert(source, destination)

        with h5py.File(source / "data.h5", "r") as before, h5py.File(
            destination / "data.h5", "r"
        ) as after:
            self.assertEqual(int(after["meta"].attrs["schema_version"]), 26)
            self.assertEqual(int(after["meta"].attrs["converted_from_schema"]), 25)
            # Tactile payloads are exactly * 10 (representation migration).
            np.testing.assert_array_equal(
                after["hand_contact"][:], before["hand_contact"][:] * 10.0
            )
            np.testing.assert_array_equal(
                after["hand_tactile_force"][:], before["hand_tactile_force"][:] * 10.0
            )
            # Aggregate freshness is copied conservatively from dense freshness.
            np.testing.assert_array_equal(
                after["tactile_sum_fresh"][:], before["tactile_fresh"][:]
            )
            np.testing.assert_array_equal(
                after["tactile_fresh"][:], before["tactile_fresh"][:]
            )
            # All other datasets are bit-identical.
            for name in _CONVERTER.V25_DATASETS:
                if name in ("hand_contact", "hand_tactile_force"):
                    continue
                np.testing.assert_array_equal(after[name][:], before[name][:])
        # The migrated destination is the frozen v26 projection, not the current
        # canonical schema: the converter stays pinned at v26 and never writes
        # the v27 camera-health / camera-aligned-tactile fields.
        with h5py.File(destination / "data.h5", "r") as migrated:
            names = {
                name for name in migrated if isinstance(migrated[name], h5py.Dataset)
            }
        self.assertEqual(names, set(_CONVERTER.V26_DATASETS))
        # Source unchanged and media hard-linked.
        self.assertEqual(
            (source / "rgb.mp4").read_bytes(), (destination / "rgb.mp4").read_bytes()
        )
        self.assertEqual(
            (source / "depth.h5").stat().st_ino,
            (destination / "depth.h5").stat().st_ino,
        )

    def test_v24_derived_refuses_without_explicit_scale(self) -> None:
        source = self._write_episode(converted_from_schema=24)
        with self.assertRaisesRegex(ValueError, "converted-v24-scale"):
            self._convert(source, self.root / "refused")

    def test_v24_derived_legacy_scale_and_native(self) -> None:
        source = self._write_episode("legacy", converted_from_schema=24)
        legacy = self.root / "legacy_out"
        self._convert(source, legacy, "legacy-0.1")
        with h5py.File(legacy / "data.h5", "r") as f:
            np.testing.assert_array_equal(f["hand_contact"][:], np.full((3, 5, 3), 10.0))
            np.testing.assert_array_equal(
                f["hand_tactile_force"][:], np.full((3, 5, 120, 3), 20.0)
            )

        native = self.root / "native_out"
        self._convert(source, native, "native")
        with h5py.File(native / "data.h5", "r") as f:
            np.testing.assert_array_equal(f["hand_contact"][:], np.full((3, 5, 3), 1.0))
            np.testing.assert_array_equal(
                f["hand_tactile_force"][:], np.full((3, 5, 120, 3), 2.0)
            )

    def test_missing_dataset_fails_without_staging_leftover(self) -> None:
        source = self._write_episode("missing", missing="hand_contact")
        destination = self.root / "missing_out"
        with self.assertRaisesRegex(ValueError, "hand_contact"):
            self._convert(source, destination)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.glob(".missing_out.tmp-*")), [])

    def test_source_untouched_and_destination_not_overwritten(self) -> None:
        source = self._write_episode()
        source_bytes = (source / "data.h5").read_bytes()
        destination = self.root / "existing"
        destination.mkdir()
        with self.assertRaises(FileExistsError):
            self._convert(source, destination)
        converted = self._convert(source, self.root / "fresh")
        self.assertEqual(source_bytes, (source / "data.h5").read_bytes())
        self.assertTrue(converted.is_dir())

    def test_exdev_media_copy_is_the_only_link_fallback(self) -> None:
        source = self._write_episode()
        destination = self.root / "exdev_out"
        exdev = getattr(os, "EXDEV", 18)

        def raise_exdev(*args, **kwargs):
            raise OSError(exdev, "cross-device link")

        with mock.patch.object(_CONVERTER.os, "link", side_effect=raise_exdev):
            self._convert(source, destination)
        for filename in ("depth.h5", "rgb.mp4"):
            self.assertEqual(
                (source / filename).read_bytes(), (destination / filename).read_bytes()
            )

    def test_cli_supports_single_episode_and_directory_summary(self) -> None:
        single = self._write_episode("single")
        self.assertEqual(
            _CONVERTER.main([str(single), str(self.root / "single_out")]), 0
        )

        root = self.root / "episodes"
        self._write_episode("episodes/episode_001")
        self._write_episode("episodes/episode_002")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = _CONVERTER.main(
                [str(root), str(self.root / "migrated")]
            )
        self.assertEqual(status, 0)
        self.assertIn("Converted", output.getvalue())


if __name__ == "__main__":
    unittest.main()

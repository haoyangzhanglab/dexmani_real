"""Offline tests for the frozen raw-v24 to raw-v25 projection utility.

The fixture intentionally uses only the published file layout and the frozen
dataset names.  It does not import the runtime schema or an episode reader.
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
    Path(__file__).resolve().parents[1] / "tools" / "convert_raw_v24_to_v25.py"
)
_SPEC = importlib.util.spec_from_file_location("raw_v24_to_v25", _CONVERTER_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - import setup failure
    raise RuntimeError(f"cannot load converter module from {_CONVERTER_PATH}")
_CONVERTER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CONVERTER)


_FRAME_COUNT = 3
_TAIL_SHAPES: dict[str, tuple[int, ...]] = {
    "timestamp": (),
    "source_sample_index": (),
    "fill_reason": (),
    "flag_sample_valid": (),
    "arm_qpos": (7,),
    "arm_qvel": (7,),
    "arm_tau": (7,),
    "hand_qpos": (12,),
    "hand_current": (12,),
    "hand_contact": (5, 3),
    "hand_tactile_force": (5, 120, 3),
    "arm_connected": (),
    "hand_connected": (),
    "hand_qpos_stale": (),
    "tracking_error": (),
    "arm_last_cmd_seq": (),
    "action_arm_joint_sent": (7,),
    "action_hand_joint": (12,),
    "action_arm_ee": (9,),
    "flag_action_queued": (),
    "flag_frame_status": (),
    "observation_anchor_monotonic_ns": (),
    "observation_valid": (),
    "arm_source_monotonic_ns": (),
    "hand_source_monotonic_ns": (),
    "tactile_source_monotonic_ns": (),
    "vr_source_monotonic_ns": (),
    "camera_source_monotonic_ns": (),
    "tactile_fresh": (),
    "tactile_calibrated": (),
    "tactile_unit_code": (),
    "flag_camera_fresh": (),
    "camera_depth_frame_number": (),
    "camera_color_frame_number": (),
    "policy_observation_arm_qpos": (7,),
    "policy_observation_hand_qpos": (12,),
    "policy_observation_valid": (),
    "vr_wrist_pos": (3,),
    "vr_wrist_rot6d": (6,),
    "vr_landmarks": (21, 3),
    "head_quat_wxyz": (4,),
}

_EXPECTED_KEEP_DATASETS = tuple(_TAIL_SHAPES)

_UNSIGNED_DTYPES = {
    "observation_anchor_monotonic_ns": np.dtype(np.uint64),
    "arm_source_monotonic_ns": np.dtype(np.uint64),
    "hand_source_monotonic_ns": np.dtype(np.uint64),
    "tactile_source_monotonic_ns": np.dtype(np.uint64),
    "vr_source_monotonic_ns": np.dtype(np.uint64),
    "camera_source_monotonic_ns": np.dtype(np.uint64),
    "tactile_unit_code": np.dtype(np.uint8),
    "flag_frame_status": np.dtype(np.uint8),
    "camera_depth_frame_number": np.dtype(np.uint64),
    "camera_color_frame_number": np.dtype(np.uint64),
}

_BOOL_DATASETS = {
    "flag_sample_valid",
    "arm_connected",
    "hand_connected",
    "hand_qpos_stale",
    "flag_action_queued",
    "observation_valid",
    "tactile_fresh",
    "tactile_calibrated",
    "flag_camera_fresh",
    "policy_observation_valid",
}
_REMOVED_DATASETS = (
    "arm_ee",
    "hand_fingertip",
    "action_arm_joint",
    "camera_health",
    "policy_observation_reference_monotonic_ns",
    "ik_solve_time_ms",
)


class TestRawV24ToV25Conversion(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _write_episode(
        self,
        relative: str = "source",
        *,
        schema_version: int = 24,
        missing: set[str] | None = None,
        depth_frames: int | None = None,
        rgb_bytes: bytes = b"synthetic-rgb-payload",
        unsigned_overrides: dict[str, np.ndarray] | None = None,
    ) -> Path:
        episode = self.root / relative
        episode.mkdir(parents=True, exist_ok=True)
        missing = missing or set()
        unsigned_overrides = unsigned_overrides or {}
        attrs = {
            "schema_version": schema_version,
            "num_frames": _FRAME_COUNT,
            "task_label": "fixture_task",
            "operator": "fixture_operator",
            "control_hz": 20.0,
            "fps": 20.0,
            "duration": 0.15,
            "wall_fps": 15.0,
            "min_frames_met": False,
            "success": True,
            "truncated": True,
            "stop_reason": "fixture:operator_stop",
            "has_camera": True,
            "has_timestamps": True,
            "camera_stream_frames": _FRAME_COUNT,
            "camera_name": "fixture_camera",
            "camera_serial": "SERIAL-001",
            "camera_type": "synthetic",
            "camera_payload_mode": "depth_to_color_aligned_rgbd",
            "depth_scale": 0.001,
            "camera_depth_width": 2,
            "camera_depth_height": 2,
            "camera_color_width": 2,
            "camera_color_height": 2,
            "camera_depth_intrinsics": np.eye(3).reshape(-1),
            "camera_color_intrinsics": np.eye(3).reshape(-1),
            "camera_T_color_from_depth": np.eye(4).reshape(-1),
            "real_git_commit": "fixture-commit",
            "camera_health_taxonomy_json": "runtime-audit-only",
            "provenance_fixture": "compact-provenance",
            "provenance_seed": np.int64(42),
        }
        with h5py.File(episode / "data.h5", "w") as data:
            meta = data.create_group("meta")
            for key, value in attrs.items():
                meta.attrs[key] = value
            for index, name in enumerate(_EXPECTED_KEEP_DATASETS):
                if name in missing:
                    continue
                shape = (_FRAME_COUNT,) + _TAIL_SHAPES[name]
                if name == "timestamp":
                    values = np.arange(_FRAME_COUNT, dtype=np.float64) + 1.25
                elif name == "source_sample_index":
                    values = np.array([-1, 4, 9], dtype=np.int64)
                elif name == "fill_reason":
                    values = np.array([2, 0, 1], dtype=np.uint8)
                elif name in _BOOL_DATASETS:
                    values = np.array([True, False, True], dtype=bool)
                elif name in unsigned_overrides:
                    values = unsigned_overrides[name]
                elif name in _UNSIGNED_DTYPES:
                    values = np.array([101 + index, 102 + index, 103 + index])
                else:
                    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
                    values += index + 0.5
                    if name == "arm_last_cmd_seq":
                        values = np.array([-3, 0, 7], dtype=np.int64)
                data.create_dataset(name, data=values)
            for name in _REMOVED_DATASETS:
                data.create_dataset(name, data=np.arange(_FRAME_COUNT))

        with h5py.File(episode / "depth.h5", "w") as depth_file:
            frame_count = _FRAME_COUNT if depth_frames is None else depth_frames
            depth_file.create_dataset(
                "depth",
                data=np.arange(frame_count * 4, dtype=np.uint16).reshape(
                    frame_count, 2, 2
                ),
            )
        (episode / "rgb.mp4").write_bytes(rgb_bytes)
        return episode

    def _convert(self, source: Path, destination: Path) -> Path:
        return _CONVERTER.convert_episode(source, destination)

    def test_projection_is_exact_except_declared_unsigned_fields(self) -> None:
        source = self._write_episode()
        destination = self.root / "converted"

        result = self._convert(source, destination)

        self.assertEqual(result, destination)
        with h5py.File(source / "data.h5", "r") as before, h5py.File(
            destination / "data.h5", "r"
        ) as after:
            self.assertEqual(set(_CONVERTER.KEEP_DATASETS), set(_EXPECTED_KEEP_DATASETS))
            self.assertEqual(set(after.keys()), set(_EXPECTED_KEEP_DATASETS) | {"meta"})
            for name in _EXPECTED_KEEP_DATASETS:
                expected = before[name][:]
                if name in _UNSIGNED_DTYPES:
                    expected = expected.astype(_UNSIGNED_DTYPES[name])
                np.testing.assert_array_equal(after[name][:], expected)
                expected_dtype = _UNSIGNED_DTYPES.get(name, before[name].dtype)
                self.assertEqual(after[name].dtype, expected_dtype)
            for name in _REMOVED_DATASETS:
                self.assertNotIn(name, after)
            self.assertEqual(int(after["meta"].attrs["schema_version"]), 25)
            self.assertEqual(int(after["meta"].attrs["converted_from_schema"]), 24)
            self.assertEqual(int(after["meta"].attrs["num_frames"]), _FRAME_COUNT)
            self.assertEqual(after["meta"].attrs["task_label"], "fixture_task")
            self.assertEqual(after["meta"].attrs["operator"], "fixture_operator")
            for name in (
                "duration",
                "fps",
                "wall_fps",
                "min_frames_met",
                "success",
                "truncated",
                "stop_reason",
                "has_camera",
                "has_timestamps",
                "camera_stream_frames",
                "provenance_fixture",
                "provenance_seed",
            ):
                np.testing.assert_array_equal(
                    after["meta"].attrs[name], before["meta"].attrs[name]
                )
            self.assertNotIn("camera_health_taxonomy_json", after["meta"].attrs)
            self.assertEqual(before["meta"].attrs["schema_version"], 24)
            self.assertNotIn("converted_from_schema", before["meta"].attrs)
        self.assertEqual(
            (source / "rgb.mp4").read_bytes(), (destination / "rgb.mp4").read_bytes()
        )
        self.assertEqual(
            (source / "depth.h5").stat().st_ino,
            (destination / "depth.h5").stat().st_ino,
        )
        self.assertEqual(
            (source / "rgb.mp4").stat().st_ino,
            (destination / "rgb.mp4").stat().st_ino,
        )

    def test_every_required_dataset_including_sent_stream_is_required(self) -> None:
        for missing_name in _EXPECTED_KEEP_DATASETS:
            with self.subTest(missing=missing_name):
                source = self._write_episode(
                    f"missing_{missing_name}", missing={missing_name}
                )
                destination = self.root / f"out_{missing_name}"
                with self.assertRaisesRegex(ValueError, missing_name):
                    self._convert(source, destination)
                self.assertFalse(destination.exists())

    def test_missing_optional_metadata_is_not_synthesized(self) -> None:
        source = self._write_episode()
        absent = (
            "min_frames_met",
            "success",
            "truncated",
            "stop_reason",
            "provenance_fixture",
            "provenance_seed",
        )
        with h5py.File(source / "data.h5", "a") as raw:
            for name in absent:
                del raw["meta"].attrs[name]
        destination = self._convert(source, self.root / "converted")
        with h5py.File(destination / "data.h5", "r") as raw:
            for name in absent:
                self.assertNotIn(name, raw["meta"].attrs)

    def test_unsigned_conversion_rejects_negative_and_overflow_without_wrapping(self) -> None:
        negative_name = "observation_anchor_monotonic_ns"
        source = self._write_episode(
            "negative",
            unsigned_overrides={
                negative_name: np.array([10, -1, 12], dtype=np.int64)
            },
        )
        with self.assertRaisesRegex(ValueError, negative_name):
            self._convert(source, self.root / "negative_out")

        overflow_name = "flag_frame_status"
        source = self._write_episode(
            "overflow",
            unsigned_overrides={
                overflow_name: np.array([0, 256, 2], dtype=np.int64)
            },
        )
        with self.assertRaisesRegex(ValueError, overflow_name):
            self._convert(source, self.root / "overflow_out")

    def test_wrong_schema_and_structural_media_mismatch_are_rejected(self) -> None:
        source = self._write_episode("wrong_schema", schema_version=23)
        with self.assertRaisesRegex(ValueError, "schema"):
            self._convert(source, self.root / "wrong_schema_out")

        source = self._write_episode("depth_length", depth_frames=_FRAME_COUNT - 1)
        with self.assertRaisesRegex(ValueError, "depth"):
            self._convert(source, self.root / "depth_length_out")

        source = self._write_episode("depth_missing")
        with h5py.File(source / "depth.h5", "a") as depth_file:
            del depth_file["depth"]
        with self.assertRaisesRegex(ValueError, "depth"):
            self._convert(source, self.root / "depth_missing_out")

        source = self._write_episode("rgb_empty", rgb_bytes=b"")
        with self.assertRaisesRegex(ValueError, "rgb"):
            self._convert(source, self.root / "rgb_empty_out")

    def test_source_is_untouched_and_destination_is_never_overwritten(self) -> None:
        source = self._write_episode()
        source_data_before = (source / "data.h5").read_bytes()
        source_depth_before = (source / "depth.h5").read_bytes()
        source_rgb_before = (source / "rgb.mp4").read_bytes()

        destination = self.root / "existing"
        destination.mkdir()
        sentinel = destination / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._convert(source, destination)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

        converted = self._convert(source, self.root / "fresh")
        self.assertEqual(source_data_before, (source / "data.h5").read_bytes())
        self.assertEqual(source_depth_before, (source / "depth.h5").read_bytes())
        self.assertEqual(source_rgb_before, (source / "rgb.mp4").read_bytes())
        self.assertTrue(converted.is_dir())

    def test_exdev_media_copy_is_the_only_link_fallback(self) -> None:
        source = self._write_episode()
        destination = self.root / "exdev_out"
        exdev = getattr(os, "EXDEV", 18)

        def raise_exdev(*args: object, **kwargs: object) -> None:
            raise OSError(exdev, "cross-device link")

        with mock.patch.object(_CONVERTER.os, "link", side_effect=raise_exdev):
            self._convert(source, destination)

        for filename in ("depth.h5", "rgb.mp4"):
            source_path = source / filename
            output_path = destination / filename
            self.assertEqual(source_path.read_bytes(), output_path.read_bytes())
            self.assertNotEqual(source_path.stat().st_ino, output_path.stat().st_ino)

    def test_cli_supports_single_episode_and_directory_summary(self) -> None:
        single_source = self._write_episode("single")
        single_destination = self.root / "single_out"
        self.assertEqual(
            _CONVERTER.main([str(single_source), str(single_destination)]), 0
        )
        self.assertTrue(single_destination.is_dir())

        source_root = self.root / "episodes"
        self._write_episode("episodes/episode_001")
        self._write_episode("episodes/episode_002")
        destination_root = self.root / "migrated"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = _CONVERTER.main([str(source_root), str(destination_root)])
        self.assertEqual(status, 0)
        self.assertIn("converted", output.getvalue().lower())
        self.assertTrue((destination_root / "episode_001").is_dir())
        self.assertTrue((destination_root / "episode_002").is_dir())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

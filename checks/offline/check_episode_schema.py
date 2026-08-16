"""Focused offline checks for the schema-v16 reader/writer contract.

This check uses synthetic HDF5 sidecars and a decoder fake.  It never starts a
camera, robot SDK, recorder worker, or video encoder.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import _bootstrap  # noqa: F401  (repo root on sys.path)
import h5py
import numpy as np

import dexmani_real.recording.episode_reader as reader_module
import dexmani_real.recording.episode_recorder as recorder_module
from dexmani_real.recording.camera_stream_writer import CameraStreamWriterConfig
from dexmani_real.recording.episode_reader import EpisodeReader, ValidityState
from dexmani_real.recording.episode_recorder import EpisodeRecorder
from dexmani_real.recording.episode_schema import (
    ARM_SENT_DATASET,
    ARM_SENT_MARKER,
    BASE_DATASET_SPECS_V16,
    DIAGNOSTIC_TAIL_SHAPES_V16,
    EPISODE_SCHEMA_VERSION,
    SEMANTIC_META_ATTRS_V16,
    SOURCE_FRAME_DATASET_NAMES_V16,
    expected_source_frame_dataset_names_v16,
    normalize_diagnostics_v16,
    validate_data_layout_v16,
    validate_source_frame_keys_v16,
)
from dexmani_real.recording.recorder_client import _write_sample_metadata
from dexmani_real.recording.timestamp_buffer import FillReason, TimestampAlignedBuffer
from dexmani_real.robot.types import RobotAction
from dexmani_real.utils.schema import (
    ARM_JOINT_SHAPE,
    HAND_JOINT_SHAPE,
    make_record_sample_dtype,
)

_FRAME_COUNT = 4


class _FakeVideoDecoder:
    """Length-only decoder fake for deterministic reader/finalizer checks."""

    def __init__(self, _path: str | Path) -> None:
        self.frame_count = _FRAME_COUNT

    def count_decoded_frames(self) -> int:
        return _FRAME_COUNT

    def close(self) -> None:
        pass

    def __enter__(self) -> "_FakeVideoDecoder":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _base_values() -> dict[str, np.ndarray]:
    values = {
        name: np.zeros((_FRAME_COUNT, *spec.tail_shape), dtype=spec.dtype)
        for name, spec in BASE_DATASET_SPECS_V16.items()
    }
    timestamps = 1.0 + np.arange(_FRAME_COUNT, dtype=np.float64) / 16.0
    values["timestamp"][:] = timestamps
    values["flag_sample_valid"][:] = True
    values["source_sample_index"][:] = np.arange(_FRAME_COUNT, dtype=np.int64)
    values["source_timestamp"][:] = timestamps
    values["fill_reason"][:] = int(FillReason.SOURCE)

    values["observation_id"][:] = np.arange(1, _FRAME_COUNT + 1, dtype=np.int64)
    values["observation_anchor_monotonic_ns"][:] = np.arange(
        1_000_000_000,
        1_000_000_000 + _FRAME_COUNT,
        dtype=np.int64,
    )
    values["action_id"][:] = np.arange(1, _FRAME_COUNT + 1, dtype=np.int64)
    values["action_created_monotonic_ns"][:] = 100
    values["action_target_monotonic_ns"][:] = 200
    values["action_valid_until_monotonic_ns"][:] = 300
    values["flag_action_queued"][:] = True

    # Exercise the conservative raw-action masks independently of the stored
    # numeric values (which intentionally remain finite on every source row).
    values["flag_held"][:] = (False, True, False, False)
    values["flag_ik_ok"][:] = (True, True, False, True)
    values["flag_retarget_ok"][:] = (True, True, True, False)
    return values


def _write_fixture(directory: Path) -> None:
    directory.mkdir()
    with h5py.File(directory / "data.h5", "w") as data_h5:
        meta = data_h5.create_group("meta")
        meta.attrs["schema_version"] = EPISODE_SCHEMA_VERSION
        meta.attrs["num_frames"] = _FRAME_COUNT
        meta.attrs["resolved_config_sha256"] = "0" * 64
        meta.attrs["success"] = True
        meta.attrs["camera_writer_error"] = ""
        meta.attrs["camera_encoding_height"] = 2
        meta.attrs["camera_encoding_width"] = 2
        for name, value in _base_values().items():
            data_h5.create_dataset(name, data=value)
    with h5py.File(directory / "depth.h5", "w") as depth_h5:
        depth_h5.create_dataset(
            "depth", data=np.zeros((_FRAME_COUNT, 2, 2), dtype=np.uint16)
        )
    with h5py.File(directory / "pointcloud.h5", "w") as pointcloud_h5:
        pointcloud_h5.create_dataset(
            "pointcloud",
            data=np.zeros((_FRAME_COUNT, 3, 6), dtype=np.float32),
        )
    (directory / "rgb.mp4").touch()


def _layout_from_specs() -> tuple[dict[str, tuple[int, ...]], dict[str, np.dtype[Any]]]:
    shapes = {
        name: (_FRAME_COUNT, *spec.tail_shape)
        for name, spec in BASE_DATASET_SPECS_V16.items()
    }
    dtypes = {name: spec.dtype for name, spec in BASE_DATASET_SPECS_V16.items()}
    return shapes, dtypes


def _assert_rejected(callable_: Any, expected_fragment: str) -> None:
    try:
        callable_()
    except (RuntimeError, ValueError) as exc:
        assert expected_fragment in str(exc), str(exc)
    else:
        raise AssertionError(f"expected rejection containing {expected_fragment!r}")


def _check_pure_contract() -> None:
    assert len(BASE_DATASET_SPECS_V16) == 96
    assert len(SOURCE_FRAME_DATASET_NAMES_V16) == 91
    assert len(DIAGNOSTIC_TAIL_SHAPES_V16) == 11

    source_keys = set(SOURCE_FRAME_DATASET_NAMES_V16)
    assert not validate_source_frame_keys_v16(source_keys, arm_sent_stream=False)
    assert not validate_source_frame_keys_v16(
        set(expected_source_frame_dataset_names_v16(arm_sent_stream=True)),
        arm_sent_stream=True,
    )
    missing_source_keys = set(source_keys)
    missing_source_keys.remove("arm_qpos")
    assert any(
        "arm_qpos" in error
        for error in validate_source_frame_keys_v16(
            missing_source_keys, arm_sent_stream=False
        )
    )
    unexpected_source_keys = set(source_keys)
    unexpected_source_keys.add("diagnostic_typo")
    assert any(
        "diagnostic_typo" in error
        for error in validate_source_frame_keys_v16(
            unexpected_source_keys, arm_sent_stream=False
        )
    )

    shapes, dtypes = _layout_from_specs()
    assert not validate_data_layout_v16(
        shapes,
        dtypes,
        frame_count=_FRAME_COUNT,
        arm_sent_stream=False,
    )

    # Every one of the 96 base datasets is required, not merely the reader's
    # historical 37-field subset.
    for missing_name in BASE_DATASET_SPECS_V16:
        missing_shapes = dict(shapes)
        del missing_shapes[missing_name]
        errors = validate_data_layout_v16(
            missing_shapes,
            dtypes,
            frame_count=_FRAME_COUNT,
            arm_sent_stream=False,
        )
        assert any(missing_name in error for error in errors), (missing_name, errors)

    marker_errors = validate_data_layout_v16(
        shapes,
        dtypes,
        frame_count=_FRAME_COUNT,
        arm_sent_stream=True,
    )
    assert any(ARM_SENT_DATASET in error for error in marker_errors)

    sent_shapes = dict(shapes, **{ARM_SENT_DATASET: (_FRAME_COUNT, *ARM_JOINT_SHAPE)})
    sent_dtypes = dict(dtypes, **{ARM_SENT_DATASET: np.dtype(np.float64)})
    assert not validate_data_layout_v16(
        sent_shapes,
        sent_dtypes,
        frame_count=_FRAME_COUNT,
        arm_sent_stream=True,
    )
    unmarked_errors = validate_data_layout_v16(
        sent_shapes,
        sent_dtypes,
        frame_count=_FRAME_COUNT,
        arm_sent_stream=False,
    )
    assert any(ARM_SENT_MARKER in error for error in unmarked_errors)

    wrong_shape = dict(shapes, arm_qpos=(_FRAME_COUNT, ARM_JOINT_SHAPE[0] - 1))
    assert any(
        "tail shape" in error
        for error in validate_data_layout_v16(
            wrong_shape,
            dtypes,
            frame_count=_FRAME_COUNT,
            arm_sent_stream=False,
        )
    )
    wrong_dtype = dict(dtypes, arm_qpos=np.dtype(np.float32))
    assert any(
        "dtype" in error
        for error in validate_data_layout_v16(
            shapes,
            wrong_dtype,
            frame_count=_FRAME_COUNT,
            arm_sent_stream=False,
        )
    )

    # Historical custom diagnostics remain readable only while grid-aligned.
    custom_shapes = dict(shapes, historical_custom_metric=(_FRAME_COUNT, 2))
    assert not validate_data_layout_v16(
        custom_shapes,
        dtypes,
        frame_count=_FRAME_COUNT,
        arm_sent_stream=False,
    )
    custom_shapes["historical_custom_metric"] = (_FRAME_COUNT - 1, 2)
    custom_errors = validate_data_layout_v16(
        custom_shapes,
        dtypes,
        frame_count=_FRAME_COUNT,
        arm_sent_stream=False,
    )
    assert any("historical_custom_metric" in error for error in custom_errors)


def _check_diagnostics_boundary() -> None:
    normalized = normalize_diagnostics_v16(
        {
            "tracking_error": 0.25,
            "target_pos_before_clamp": [1.0, 2.0, 3.0],
            "action_hand_joint_raw": np.zeros(HAND_JOINT_SHAPE),
        }
    )
    assert normalized["tracking_error"].shape == ()
    assert normalized["target_pos_before_clamp"].shape == (3,)

    _assert_rejected(
        lambda: normalize_diagnostics_v16({"target_pos_before_clamp": np.zeros(2)}),
        "expected (3,)",
    )
    _assert_rejected(
        lambda: normalize_diagnostics_v16({"diagnostic_typo": 1.0}), "unsupported keys"
    )
    _assert_rejected(
        lambda: normalize_diagnostics_v16({"arm_qpos": np.zeros(ARM_JOINT_SHAPE)}),
        "reserved dataset collisions",
    )

    # The policy-side structured sample boundary must reject the same typo;
    # it may not silently drop a key before RecorderIO sees it.
    frame = np.zeros(1, dtype=make_record_sample_dtype((2, 2, 3), (2, 2), (3, 6)))
    action = RobotAction(np.zeros(ARM_JOINT_SHAPE), np.zeros(HAND_JOINT_SHAPE))
    _assert_rejected(
        lambda: _write_sample_metadata(
            frame,
            action=action,
            camera_frame=None,
            signals=None,
            arm_qpos_sent=None,
            diagnostics={"diagnostic_typo": 1.0},
        ),
        "unsupported keys",
    )


def _check_timestamp_buffer_layout_boundary() -> None:
    buffer = TimestampAlignedBuffer(start_time=1.0, dt=0.1, max_record_steps=4)
    first: dict[str, np.ndarray | float | int] = {
        "vector": np.zeros(2, dtype=np.float64),
        "count": 1,
        "ready": True,
    }
    assert buffer.add(first, timestamp=1.0).source_written
    assert buffer.size == 1

    _assert_rejected(
        lambda: buffer.add({"vector": np.zeros(2), "count": 2}, timestamp=1.1),
        "field keys differ",
    )
    _assert_rejected(
        lambda: buffer.add({**first, "extra": 1.0}, timestamp=1.1),
        "field keys differ",
    )
    _assert_rejected(
        # The previous implementation silently broadcast this scalar into the
        # two-element destination, so final HDF5 layout validation could not
        # detect that the source frame itself was malformed.
        lambda: buffer.add({**first, "vector": 7.0}, timestamp=1.1),
        "shape",
    )
    _assert_rejected(
        lambda: buffer.add(
            {**first, "vector": np.zeros(2, dtype=np.float32)}, timestamp=1.1
        ),
        "dtype",
    )

    # Rejections happen before temporal indices advance: the next valid frame
    # still occupies the immediately following grid slot.
    assert buffer.size == 1
    second = {**first, "count": 2}
    result = buffer.add(second, timestamp=1.1)
    assert result.source_written and result.previous_size == 1 and result.size == 2
    np.testing.assert_array_equal(buffer.data["vector"], np.zeros((2, 2)))


def _reader_state(path: Path) -> ValidityState:
    with EpisodeReader(path) as reader:
        return reader.validity


def _check_reader_and_finalizer(root: Path) -> None:
    episode = root / "episode"
    _write_fixture(episode)

    original_reader_decoder = reader_module.VideoDecoder
    original_recorder_decoder = recorder_module.VideoDecoder
    reader_module.VideoDecoder = _FakeVideoDecoder  # type: ignore[assignment, misc]
    recorder_module.VideoDecoder = _FakeVideoDecoder  # type: ignore[assignment, misc]
    try:
        # The fixture intentionally omits all new additive semantic attrs:
        # historical v16 episodes still read and validate.
        assert _reader_state(episode) is ValidityState.VALID
        with EpisodeReader(episode) as reader:
            np.testing.assert_array_equal(
                reader.action_arm_joint_raw_valid_mask,
                np.array([True, False, False, True]),
            )
            np.testing.assert_array_equal(
                reader.action_hand_joint_raw_valid_mask,
                np.array([True, False, True, False]),
            )

        with h5py.File(episode / "data.h5", "r+") as data_h5:
            del data_h5["arm_qpos"]
        assert _reader_state(episode) is ValidityState.INVALID
        arm_spec = BASE_DATASET_SPECS_V16["arm_qpos"]
        with h5py.File(episode / "data.h5", "r+") as data_h5:
            data_h5.create_dataset(
                "arm_qpos",
                data=np.zeros(
                    (_FRAME_COUNT, *arm_spec.tail_shape), dtype=arm_spec.dtype
                ),
            )

        with h5py.File(episode / "data.h5", "r+") as data_h5:
            data_h5.create_dataset(
                "historical_custom_metric", data=np.zeros((_FRAME_COUNT, 2))
            )
        assert _reader_state(episode) is ValidityState.VALID
        with h5py.File(episode / "data.h5", "r+") as data_h5:
            del data_h5["historical_custom_metric"]
            data_h5.create_dataset(
                "historical_custom_metric", data=np.zeros((_FRAME_COUNT - 1, 2))
            )
        assert _reader_state(episode) is ValidityState.INVALID
        with h5py.File(episode / "data.h5", "r+") as data_h5:
            del data_h5["historical_custom_metric"]

        with h5py.File(episode / "data.h5", "r+") as data_h5:
            data_h5["meta"].attrs[ARM_SENT_MARKER] = True
        assert _reader_state(episode) is ValidityState.INVALID
        with h5py.File(episode / "data.h5", "r+") as data_h5:
            data_h5.create_dataset(
                ARM_SENT_DATASET,
                data=np.zeros((_FRAME_COUNT, *ARM_JOINT_SHAPE), dtype=np.float64),
            )
        assert _reader_state(episode) is ValidityState.VALID
        EpisodeRecorder._validate_and_sync_temp_episode(episode, _FRAME_COUNT)

        with h5py.File(episode / "data.h5", "r+") as data_h5:
            del data_h5["arm_tau"]
        _assert_rejected(
            lambda: EpisodeRecorder._validate_and_sync_temp_episode(
                episode, _FRAME_COUNT
            ),
            "missing required",
        )
    finally:
        reader_module.VideoDecoder = original_reader_decoder  # type: ignore[misc]
        recorder_module.VideoDecoder = original_recorder_decoder  # type: ignore[misc]


def _check_writer_metadata(root: Path) -> None:
    config = CameraStreamWriterConfig(
        rgb_shape=(2, 2, 3),
        depth_shape=(2, 2),
        pointcloud_shape=(3, 6),
        fps=16.0,
        queue_size=1,
    )
    recorder = EpisodeRecorder(
        str(root / "recorder"),
        max_frames=4,
        control_hz=16.0,
        camera_writer_config=config,
        resolved_config_hash="0" * 64,
    )
    recorder._pending_meta = {"skip_initial_frames": 0}
    with h5py.File(root / "metadata.h5", "w") as data_h5:
        meta = data_h5.create_group("meta")
        recorder._write_meta_attrs(meta)
        assert ARM_SENT_MARKER not in meta.attrs
        for name, expected in SEMANTIC_META_ATTRS_V16.items():
            assert name in meta.attrs
            actual = meta.attrs[name]
            if isinstance(expected, bool):
                assert bool(actual) is expected
            else:
                assert actual == expected
        recorder.arm_sent_stream = True
        sent_meta = data_h5.create_group("sent_meta")
        recorder._write_meta_attrs(sent_meta)
        assert bool(sent_meta.attrs[ARM_SENT_MARKER])


def main() -> int:
    _check_pure_contract()
    _check_diagnostics_boundary()
    _check_timestamp_buffer_layout_boundary()
    with tempfile.TemporaryDirectory(prefix="dexmani-episode-schema-") as temp_dir:
        root = Path(temp_dir)
        _check_reader_and_finalizer(root)
        _check_writer_metadata(root)
    print("check_episode_schema: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

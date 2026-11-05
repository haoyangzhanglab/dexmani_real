import logging

import h5py
import numpy as np
import pytest
import zarr
from cordvip_geometry.raw import RawHandCloud, iter_raw_rows

from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.raw import (
    RawReader,
    RawWriter,
    camera_metadata,
    snapshot_recording_metadata,
)
from dexmani_real.geometry.camera import (
    CameraIntrinsics,
    CameraSnapshot,
    deproject_depth,
)
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import Observation


@pytest.fixture
def episode(tmp_path):
    path = tmp_path / "episode_001"
    path.mkdir()
    camera = CameraSnapshot(
        None, CameraIntrinsics(32, 32, 100., 100., 16., 16., "none", (0.,) * 5),
        .001, np.eye(4),
    )
    metadata = snapshot_recording_metadata(camera, tactile_reset_metadata={})
    metadata.update(task_label="test", collection_source="policy_rollout")
    writer = RawWriter(path, control_hz=29.97)
    try:
        writer.open(metadata)
        for index in range(3):
            stamp = index * round(1e9 / 29.97)
            arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
            hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
            arm["timestamp_ns"] = stamp
            for name in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
                hand[name] = stamp
            row = Observation(
                arm=arm, hand=hand, vr=None, point_cloud=None,
                camera={"rgb": np.full((32, 32, 3), 100 + index, np.uint8),
                        "depth": np.full((32, 32), 500, np.uint16),
                        "timestamp_ns": stamp, "ring_sequence": index},
                control_timestamp_ns=stamp, observation_timestamp_ns=stamp,
            )
            writer.append_frame(row, RobotCommand(np.zeros(7), np.zeros(12)))
        writer.finish("operator", [])
    finally:
        writer.close()
    return path


@pytest.mark.parametrize("hz", [None, 0., -1., np.nan, np.inf, -np.inf, "invalid"])
def test_frequency_only_required_by_consumers(episode, tmp_path, hz):
    with h5py.File(episode / "data.h5", "a") as data:
        meta = data["meta"].attrs
        del meta["control_hz"]
        if hz is not None:
            meta["control_hz"] = hz
    with RawReader(episode) as reader:
        frames = list(iter_raw_rows(reader, reader.read_camera()))
        assert len(frames) == 3
        np.testing.assert_allclose(frames[0][0].depth_m, .5)
        assert reader.timestamps_s[0] == 0
        for attribute in ("control_hz", "dt"):
            with pytest.raises((KeyError, ValueError)):
                getattr(reader, attribute)
    assert reader.data_file is None and reader.video_container is None
    with pytest.raises(ValueError):
        export_zarr(episode, tmp_path / "bad.zarr", processing=ExportConfig(), fields=("depth",))


@pytest.mark.parametrize("serial", [None, ""])
def test_geometry_without_device_identity(episode, serial):
    if serial is not None:
        with h5py.File(episode / "data.h5", "a") as data:
            data["meta"].attrs["camera_serial"] = serial
    with RawReader(episode) as reader:
        camera = reader.read_camera()
        assert camera.serial == serial
        frame = next(iter_raw_rows(reader, camera))[0]
        xyz, _, _ = deproject_depth(frame.depth_m, frame.depth_m > 0, camera.color)
        assert xyz.shape == (1024, 3) and np.isfinite(xyz).all()


@pytest.mark.parametrize("name,value", [
    ("depth_scale", 0.), ("depth_scale", np.nan), ("camera_color_width", 0),
    ("camera_color_intrinsics", np.zeros(9)),
    ("camera_T_xarm_base_from_color", np.zeros(16)),
    ("camera_payload_mode", "rgb"),
])
def test_offline_geometry_still_requires_valid_rgbd_parameters(episode, name, value):
    with h5py.File(episode / "data.h5", "a") as data:
        data["meta"].attrs[name] = value
    with RawReader(episode) as reader, pytest.raises(ValueError):
        reader.read_camera()


def test_rgb_metadata_omits_missing_scale_and_identity(tmp_path):
    camera = CameraSnapshot(None, CameraIntrinsics(32, 32, 100., 100., 16., 16., "none", (0.,) * 5),
                            None, None)
    values = camera_metadata(camera)
    assert "camera_serial" not in values and "depth_scale" not in values
    assert values["camera_payload_mode"] == "rgb"
    with h5py.File(tmp_path / "metadata.h5", "w") as data:
        data.attrs.update(values)


@pytest.mark.parametrize("source_ns,available", [(0, True), (-1, False)])
def test_hand_geometry_uses_zero_origin_for_source_times(episode, monkeypatch, source_ns, available):
    from types import SimpleNamespace

    from cordvip_geometry import hand

    model = SimpleNamespace(metadata={}, transform=lambda joints: {})
    monkeypatch.setattr(hand, "HandCloud", lambda *args, **kwargs: model)
    with RawReader(episode) as reader:
        camera = reader.read_camera()
        geometry = RawHandCloud(reader, camera, num_points=16)
        frame, timing = next(iter_raw_rows(reader, camera))
        timing["camera_ns"] = source_ns
        result = geometry.for_row(frame, timing)
    assert result["timing_available"] == available
    assert result["skew_ms"] == (0 if available else None)


def test_zero_clocks_and_repeated_sources_export(episode, tmp_path, caplog):
    with h5py.File(episode / "data.h5", "a") as data:
        timing = data["timing/rows"][:]
        timing["camera_ns"] = 0
        timing["camera_sequence"] = 0
        data["timing/rows"][:] = timing
    with caplog.at_level(logging.WARNING):
        result = export_zarr(episode, tmp_path / "valid.zarr", processing=ExportConfig(),
                             fields=("depth", "joint_state", "action"))
    assert result["total_frames"] == 3
    assert "invalid/regressed" not in caplog.text
    with RawReader(episode) as reader:
        np.testing.assert_array_equal(reader["timing/rows"][:], timing)
    store = zarr.open_group(str(tmp_path / "valid.zarr"), mode="r")
    assert store["data/action"].shape == (3, 19)


@pytest.mark.parametrize("field,values", [
    ("control_ns", [-1, 10, 20]),
    ("observation_ns", [-1, 40_000_000, 80_000_000]),
    ("control_ns", [0, 0, 20]),
    ("control_ns", [0, 20, 10]),
    ("observation_ns", [100_000_000] * 3),
    ("observation_ns", [100_000_000, 120_000_000, 110_000_000]),
    ("observation_ns", [0, 1, 2]),
])
def test_invalid_control_clocks_rejected(episode, tmp_path, field, values):
    with h5py.File(episode / "data.h5", "a") as data:
        timing = data["timing/rows"][:]
        timing[field] = values
        data["timing/rows"][:] = timing
    with pytest.raises(ValueError, match="clock"):
        export_zarr(episode, tmp_path / "bad.zarr", processing=ExportConfig(), fields=("depth",))
    assert not (tmp_path / "bad.zarr").exists()


def test_source_diagnostics_remain_warnings(episode, tmp_path, caplog):
    with h5py.File(episode / "data.h5", "a") as data:
        timing = data["timing/rows"][:]
        timing["camera_ns"] = [0, -1, 999_000_000]
        timing["camera_sequence"] = [0, 2, 1]
        data["timing/rows"][:] = timing
    with caplog.at_level(logging.WARNING):
        result = export_zarr(episode, tmp_path / "diagnostics.zarr", processing=ExportConfig(),
                             fields=("depth",))
    assert result["total_frames"] == 3
    assert "camera_ns invalid/regressed at row 1" in caplog.text
    assert "camera_sequence invalid/regressed at row 2" in caplog.text


@pytest.mark.parametrize("hz,accepted", [(np.float32(29.97), True), (30., False)])
def test_nominal_period_tolerance(episode, tmp_path, hz, accepted):
    import shutil

    second = tmp_path / "episode_002"
    shutil.copytree(episode, second)
    with h5py.File(second / "data.h5", "a") as data:
        data["meta"].attrs["control_hz"] = hz
    target = tmp_path / "combined.zarr"
    if accepted:
        result = export_zarr(tmp_path, target, processing=ExportConfig(), fields=("depth",))
        assert result["total_frames"] == 6
        assert zarr.open_group(str(target), mode="r").attrs["dt"] == 1 / 29.97
    else:
        with pytest.raises(ValueError, match="uniform dt"):
            export_zarr(tmp_path, target, processing=ExportConfig(), fields=("depth",))
        assert not target.exists() and not list(tmp_path.glob(".combined.zarr.tmp-*"))

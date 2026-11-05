import logging
from unittest.mock import MagicMock, Mock, call

import h5py
import numpy as np
import pytest
import zarr
from cordvip_geometry.raw import RawHandCloud, iter_raw_rows

from dexmani_real.data.export import ExportConfig, export_zarr
from dexmani_real.data.raw import (
    RawReader,
    RawWriter,
    _VideoWriter,
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
    assert writer.video is None and writer.data_file is None
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


def test_removed_archive_attributes_are_optional(episode, tmp_path):
    with h5py.File(episode / "data.h5", "a") as data:
        meta = data["meta"].attrs
        for name in ("runtime_config_json", "hand_tactile_hold_policy", "timestamp_source"):
            assert name not in meta
            meta[name] = "unused legacy annotation"
    with RawReader(episode) as reader:
        assert len(list(iter_raw_rows(reader, reader.read_camera()))) == 3
    export_zarr(episode, tmp_path / "legacy.zarr", processing=ExportConfig(), fields=("depth",))


@pytest.mark.parametrize("model", ["small", "large", "custom"])
def test_tracking_cli_resolves_sam_once_and_preserves_raw_rows(episode, tmp_path, monkeypatch, model):
    import json

    from cordvip_geometry import models, sam2
    from cordvip_geometry.cli import track
    from cordvip_geometry.cli.arguments import parse_tracking_args

    mask = np.ones((32, 32), bool)
    selection = tmp_path / "mask.npy"
    np.save(selection, mask)
    masks = {"object": mask}
    segmenter = Mock()
    segmenter.initialize.return_value = segmenter.track.return_value = masks
    sam = MagicMock()
    sam.return_value.__enter__.return_value = segmenter
    monkeypatch.setattr(sam2, "SAM2Client", sam)
    resolve = Mock(wraps=models.resolve_sam_model)
    monkeypatch.setattr(models, "resolve_sam_model", resolve)
    if model == "custom":
        checkpoint, config = tmp_path / "custom.pt", "custom.yaml"
        options = ["--sam-checkpoint", str(checkpoint), "--sam-config", config]
    else:
        checkpoint = tmp_path / "third_party/weights" / models.SAM_MODELS[model]["weight"]
        config = models.SAM_MODELS[model]["config"]
        options = ["--sam-variant", "large"] if model == "large" else []
    output = tmp_path / "tracked"
    args = parse_tracking_args([
        "--episode", str(episode), "--initial-selection", str(selection), "--headless",
        "--output", str(output), "--min-matched-pixels", "0", "--depth-tolerance-m", "nan",
        "--min-matched-fraction", "0", *options,
    ], workspace=tmp_path)
    resolve.assert_not_called()
    assert track.run(args) == 0
    resolve.assert_called_once_with(
        args.models_dir, args.sam_variant, args.sam_checkpoint, args.sam_config,
    )
    assert sam.call_args.kwargs["checkpoint"] == checkpoint
    assert sam.call_args.kwargs["config"] == config
    assert json.loads((output / "scene.json").read_text())["objects"][0]["id"] == "object"
    assert not (output / "model_templates").exists()
    paths = sorted((output / "frames").glob("*.npz"))
    assert len(paths) == 3
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as frame:
            assert frame["row_index"] == index
            assert frame["timestamp_ns"] == index * round(1e9 / 29.97)
            assert frame["object__state"] == "OBSERVED_ONLY"
            assert frame["object__observed_points_camera"].shape == (256, 3)


@pytest.mark.parametrize("options,error,resolutions", [
    (["--min-matched-pixels", "0"], "min_matched_pixels", 0),
    (["--sam-checkpoint", "custom.pt"], "完整配对", 1),
    (["--sam-config", "custom.yaml"], "完整配对", 1),
    (["--sam-checkpoint", "custom.pt", "--sam-config", "custom.yaml",
      "--sam-variant", "large"], "自训配对", 1),
])
def test_tracking_cli_rejects_invalid_setup_before_asset_load(
    episode, tmp_path, monkeypatch, options, error, resolutions,
):
    from cordvip_geometry import assets, models, sam2
    from cordvip_geometry.cli import track
    from cordvip_geometry.cli.arguments import parse_tracking_args

    resolve = Mock(wraps=models.resolve_sam_model)
    load, sam = Mock(), Mock()
    monkeypatch.setattr(models, "resolve_sam_model", resolve)
    monkeypatch.setattr(assets, "load_asset", load)
    monkeypatch.setattr(sam2, "SAM2Client", sam)
    args = parse_tracking_args([
        "--episode", str(episode), "--asset", "missing_asset", "--prompt", "object",
        "--output", str(tmp_path / "tracked"), *options,
    ], workspace=tmp_path)
    resolve.assert_not_called()
    with pytest.raises(ValueError, match=error):
        track.run(args)
    assert resolve.call_count == resolutions
    load.assert_not_called()
    sam.assert_not_called()


@pytest.mark.parametrize("kind", ["reader", "writer"])
@pytest.mark.parametrize("failures", [(), ("video",), ("hdf5",), ("video", "hdf5")])
def test_raw_close_attempts_both_resources_and_retains_failed_handles(tmp_path, kind, failures):
    video_error, data_error = OSError("video close failed"), OSError("hdf5 close failed")
    video, data = Mock(), Mock()
    video.close.side_effect = video_error if "video" in failures else None
    data.close.side_effect = data_error if "hdf5" in failures else None
    calls = Mock()
    calls.attach_mock(video, "video")
    calls.attach_mock(data, "data")
    if kind == "reader":
        resource = RawReader.__new__(RawReader)
        resource.dataset_cache = {"depth": object()}
        resource.video_container, resource.data_file = video, data
        video_attribute = "video_container"
    else:
        resource = RawWriter(tmp_path, control_hz=30)
        resource.video, resource.data_file = video, data
        resource.arrays = {"depth": np.empty((2, 1, 1), np.uint16)}
        resource.timing_rows = np.empty(2)
        resource.buffered = 1
        resource.finish, resource.flush = Mock(), Mock()
        video_attribute = "video"

    if failures:
        with pytest.raises(OSError) as raised:
            resource.close()
        assert raised.value is (data_error if "hdf5" in failures else video_error)
        if len(failures) == 2:
            assert raised.value.__context__ is video_error
    else:
        resource.close()
    assert calls.mock_calls == [call.video.close(), call.data.close()]
    assert getattr(resource, video_attribute) is (video if "video" in failures else None)
    assert resource.data_file is (data if "hdf5" in failures else None)
    if kind == "reader":
        assert not resource.dataset_cache
    else:
        assert not resource.arrays and resource.timing_rows is None
        assert resource.buffered == 1 and resource.frame_count == 0
        resource.finish.assert_not_called()
        resource.flush.assert_not_called()

    video.close.side_effect = data.close.side_effect = None
    resource.close()
    resource.close()
    assert getattr(resource, video_attribute) is None and resource.data_file is None
    assert video.close.call_count == 1 + ("video" in failures)
    assert data.close.call_count == 1 + ("hdf5" in failures)


@pytest.mark.parametrize("flush_failure", [None, "encode", "mux"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_video_close_flushes_once_and_always_closes_container(tmp_path, flush_failure, close_fails):
    flush_error, close_error = OSError("flush failed"), OSError("container close failed")
    stream, container = Mock(), Mock()
    stream.encode.return_value = ["packet"]
    if flush_failure == "encode":
        stream.encode.side_effect = flush_error
    elif flush_failure == "mux":
        container.mux.side_effect = flush_error
    if close_fails:
        container.close.side_effect = close_error
    writer = _VideoWriter(tmp_path / "rgb.mp4", fps=30, width=32, height=32)
    writer.stream, writer.container = stream, container
    if flush_failure or close_fails:
        with pytest.raises(OSError) as raised:
            writer.close()
        assert raised.value is (close_error if close_fails else flush_error)
        if flush_failure and close_fails:
            assert raised.value.__context__ is flush_error
    else:
        writer.close()
    assert writer.stream is None
    assert writer.container is (container if close_fails else None)
    stream.encode.assert_called_once_with(None)
    container.close.assert_called_once_with()
    assert container.mux.call_count == (flush_failure != "encode")

    container.close.side_effect = None
    writer.close()
    writer.close()
    assert writer.container is None
    stream.encode.assert_called_once_with(None)
    assert container.close.call_count == 1 + close_fails


@pytest.mark.parametrize("failure", ["flush", "container", "hdf5", "all"])
def test_recorder_does_not_publish_after_raw_close_failure(tmp_path, monkeypatch, failure):
    from dexmani_real.data import recorder

    video = _VideoWriter(tmp_path / "rgb.mp4", fps=30, width=32, height=32)
    video.stream, video.container = Mock(), Mock()
    video.stream.encode.return_value = []
    data = Mock()
    operations = {
        "flush": video.stream.encode, "container": video.container.close, "hdf5": data.close,
    }
    for name, operation in operations.items():
        if failure in (name, "all"):
            operation.side_effect = OSError(f"{name} failed")

    def open_writer(writer, metadata):
        writer.video, writer.data_file = video, data
        writer.frame_count = 1

    monkeypatch.setattr(RawWriter, "open", open_writer)
    finish = Mock()
    monkeypatch.setattr(RawWriter, "finish", finish)
    publish = Mock()
    monkeypatch.setattr(recorder, "publish", publish)
    recording = recorder.EpisodeRecorder(tmp_path, control_hz=30)
    recording.start(task_label="test", collection_source="policy_rollout", metadata={},
                    episode_name="close_failure")
    recording.request_finish(save=True, reason="operator")
    recording.thread.join(timeout=2)
    assert not recording.writer_pending
    expected_failure = "hdf5" if failure == "all" else failure
    with pytest.raises(recorder.RecordingError, match=f"{expected_failure} failed") as raised:
        recording.poll_finish()
    if failure == "all":
        error = raised.value.__cause__
        for name in ("hdf5", "container", "flush"):
            assert str(error) == f"{name} failed"
            error = error.__context__
    finish.assert_called_once()
    publish.assert_not_called()
    assert recording.result()["status"] == "failed"
    assert recording.published_path is None
    assert recording.staging_path.is_dir()
    assert not (tmp_path / "close_failure").exists()

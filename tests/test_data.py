"""调用真实 Raw/Canonical 与首帧选择链路；设备和感知模型使用替身。"""

import json
from dataclasses import replace
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest
import zarr

from dexmani_real.config.perception import PointCloudConfig
from dexmani_real.data import recorder as recorder_module
from dexmani_real.data.export import (
    ExportConfig,
    canonical_array_specs,
    export_zarr,
    training_prefix,
    validate_export_episode,
)
from dexmani_real.data.raw import RawReader, RawWriter, snapshot_recording_metadata
from dexmani_real.data.recorder import EpisodeRecorder, RecordingError
from dexmani_real.geometry.camera import CameraIntrinsics, CameraSnapshot
from dexmani_real.robot.commands import RobotCommand
from dexmani_real.robot.xarm7.protocol import ARM_STATE_DTYPE
from dexmani_real.robot.xhand.feedback import HAND_STATE_DTYPE
from dexmani_real.runtime.observation import Observation
from dexmani_real.utils import io
from dexmani_real.utils.log import error_detail


@pytest.fixture
def capture():
    camera = CameraSnapshot(
        "synthetic", CameraIntrinsics(32, 32, 200., 200., 15.5, 15.5, "none", (0.,) * 5),
        .001, np.eye(4),
    )
    metadata = snapshot_recording_metadata(
        camera, tactile_reset_metadata={}, tactile_hold_max_age_s=.5,
    )

    def row(index):
        stamp = 1_000_000_000 + index * 100_000_000
        arm = np.zeros(1, dtype=ARM_STATE_DTYPE)
        hand = np.zeros(1, dtype=HAND_STATE_DTYPE)
        arm["timestamp_ns"] = stamp
        for name in ("timestamp_ns", "tactile_aggregate_timestamp_ns", "tactile_dense_timestamp_ns"):
            hand[name] = stamp
        hand["tactile_aggregate"] = np.arange(15).reshape(5, 3)
        hand["tactile_dense"] = np.arange(1800).reshape(5, 120, 3)
        return Observation(
            arm=arm, hand=hand,
            camera={"rgb": np.full((32, 32, 3), 80, np.uint8),
                    "depth": np.full((32, 32), 500, np.uint16),
                    "timestamp_ns": stamp, "ring_sequence": index // 2},
            vr={"wrist_pos": np.zeros(3), "wrist_quat_wxyz": np.array([1., 0., 0., 0.]),
                "landmarks": np.zeros((21, 3)), "recv_ts_ns": stamp, "sequence_id": index},
            point_cloud=None, observation_timestamp_ns=stamp + 1_000_000,
            control_timestamp_ns=stamp,
        )

    return metadata, row, RobotCommand(1, np.zeros(7), np.zeros(12))


def record(path, capture, *, frames=3, save=True, status="normal"):
    metadata, row, command = capture
    recorder = EpisodeRecorder(path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    for index in range(frames):
        recorder.append(row(index), command)
    result = recorder.finish(save=save, reason="test", technical_status=status)
    return recorder, result


@pytest.fixture(params=["mask.npy", "mask.png"])
def initial_selection(tmp_path, capture, monkeypatch, request):
    monkeypatch.syspath_prepend(str(
        Path(__file__).resolve().parents[1] / "third_party/cordvip_geometry/src"
    ))
    from cordvip_geometry.io import write_confirmed_selection
    from cordvip_geometry.raw import iter_raw_rows
    from cordvip_geometry.rgbd import save_frame

    metadata, _, _ = capture
    metadata["camera_color_distortion_model"] = "brown_conrady"
    _, episode = record(tmp_path / "raw", capture, frames=1)
    with RawReader(episode) as reader:
        [(frame, _)] = iter_raw_rows(reader, reader.read_camera())
    mask = np.indices(frame.depth_m.shape).sum(axis=0) % 2 == 0
    directory = tmp_path / "selection"
    directory.mkdir()
    save_frame(directory / "source_rgbd.npz", frame)
    write_confirmed_selection(directory, mask, {
        "selection": "gui_confirmed", "selected_instance": None,
        "mask_grid": "original_rgb_grid; not_automatically_rectified",
    }, mask_file=request.param)
    return episode, directory, frame, mask


def test_initial_selection_reuses_mask_after_raw_move(tmp_path, initial_selection):
    from cordvip_geometry.cli.scene import choose_masks, load_scene
    from cordvip_geometry.geometry import RectificationMaps
    from cordvip_geometry.io import load_initial_selection
    from cordvip_geometry.raw import iter_raw_rows
    from cordvip_geometry.rgbd import load_frame
    from cordvip_geometry.tracking import SceneTracker

    episode, directory, original, mask = initial_selection
    np.testing.assert_array_equal(load_initial_selection(directory, original), mask)
    moved = episode.rename(tmp_path / "moved_episode")
    with RawReader(moved) as reader:
        [(frame, _)] = iter_raw_rows(reader, reader.read_camera())
    assert frame.source_id != original.source_id
    np.testing.assert_array_equal(load_initial_selection(directory, frame), mask)
    assert load_frame(directory / "source_rgbd.npz").source_id == original.source_id

    output = tmp_path / "tracking"
    output.mkdir()
    segmenter = Mock()
    segmenter.initialize.side_effect = lambda rgb, masks: masks
    masks = choose_masks(
        {"object": {"id": "object", "initial_selection": directory}},
        frame, SimpleNamespace(), output, segmenter,
        RectificationMaps(frame.depth_m.shape, frame.K, frame.distortion_coeffs),
    )
    np.testing.assert_array_equal(masks["object"], mask)
    segmenter.predict.assert_not_called()
    scene = load_scene(output / "scene.json", headless=True)
    tracker = SceneTracker(
        segmenter, {}, foundationpose_root=tmp_path, debug_dir=tmp_path,
        initial_selections={name: item["initial_selection"] for name, item in scene.items()},
    )
    result = tracker.step(frame)
    assert result["operation"] == "initialize"
    np.testing.assert_array_equal(result["objects"]["object"]["cloud"].semantic_mask, mask)


@pytest.mark.parametrize("key,change", [
    ("rgb", lambda value: value ^ np.uint8(1)),
    ("depth_m", lambda value: np.nextafter(value, np.float32(np.inf))),
    ("K", lambda value: value + np.diag([1e-12, 0., 0.])),
    ("distortion_model", lambda value: "none"),
    ("distortion_coeffs", lambda value: (1e-12, *value[1:])),
    ("timestamp_ns", lambda value: value + 1),
    ("clock", lambda value: "host_monotonic_observation"),
    ("row_index", lambda value: value + 1),
])
def test_initial_selection_rejects_changed_binding(initial_selection, key, change):
    from cordvip_geometry.io import load_initial_selection

    _, directory, frame, _ = initial_selection
    changed = replace(frame, **{key: change(getattr(frame, key))})
    with pytest.raises(ValueError, match=f"initial-selection {key} differs"):
        load_initial_selection(directory, changed)


@pytest.mark.parametrize("mask_file", ["../outside.npy", "linked.npy"])
def test_initial_selection_rejects_mask_outside_directory(tmp_path, initial_selection, mask_file):
    from cordvip_geometry.io import load_initial_selection

    _, directory, frame, mask = initial_selection
    outside = tmp_path / "outside.npy"
    np.save(outside, mask, allow_pickle=False)
    if mask_file == "linked.npy":
        (directory / mask_file).symlink_to(outside)
    metadata_path = directory / "selection.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["mask_file"] = mask_file
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="mask must remain inside its directory"):
        load_initial_selection(directory, frame)


@pytest.mark.parametrize("frames,save,status", [(3, True, "normal"), (3, True, "failed"),
                                                   (0, True, "normal"), (3, False, "normal")])
def test_raw_publication_and_video(tmp_path, capture, frames, save, status):
    recorder, result = record(tmp_path, capture, frames=frames, save=save, status=status)
    assert not recorder.writer_pending
    assert recorder.written_frames == frames
    if frames and save:
        assert result == recorder.published_path
        with RawReader(result) as reader:
            assert reader.num_frames == frames
            assert reader.meta["technical_status"] == status
            assert not ({"hand_tactile_processing", "hand_tactile_axes", "vr_hand_side",
                         "vr_coordinate_axes", "vr_position_units", "vr_landmark_order",
                         "vr_timestamp_source"} & set(reader.meta))
            assert not ({"clock", "missing_value", "camera_timestamp_source"}
                        & set(reader.data_file["timing"].attrs))
            assert reader.meta["timestamp_source"] == "host_monotonic_ns"
            np.testing.assert_array_equal(reader["depth"][0], 500)
            assert len(list(reader.iter_rgb_frames())) == frames
    else:
        assert result is None
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("case,expected,reason", [
    ("normal", 3, None),
    ("gap", 2, "control_grid_drift"),
    ("nan", 2, "nonfinite:action_arm_joint_target"),
    ("failed", 2, "technical_failure_last_row"),
    ("tactile", 2, "stale_source:tactile_dense_ns"),
    ("readout_between", 3, None),
    ("readout_limit", 2, "observation_crossed_control_period"),
])
def test_real_raw_continuous_training_prefix(tmp_path, capture, case, expected, reason):
    _, path = record(tmp_path, capture, status="failed" if case == "failed" else "normal")
    if case in {"gap", "nan", "tactile", "readout_between", "readout_limit"}:
        with h5py.File(path / "data.h5", "r+") as data:
            if case == "nan":
                data["action_arm_joint_target"][2] = np.full(7, np.nan)
            else:
                rows = data["timing/rows"][:]
                if case == "gap":
                    rows["control_ns"][2] += 100_000_000
                    rows["observation_ns"][2] += 100_000_000
                elif case.startswith("readout"):
                    readout_ns = 30_000_000 if case == "readout_between" else 100_000_000
                    rows["observation_ns"][2] = rows["control_ns"][2] + readout_ns
                else:
                    rows["tactile_dense_ns"] = rows["control_ns"][0]
                    data["meta"].attrs["hand_tactile_hold_max_age_s"] = .15
                data["timing/rows"][:] = rows
    with RawReader(path) as reader:
        quality = training_prefix(reader)
        assert quality["row_stop"] == expected
        assert quality["trim_reason"] == reason
        assert "readout_over_tolerance_rows" not in quality
        if case == "readout_between":
            assert quality["max_readout_ns"] == 30_000_000
        # 重复相机 sequence 合法，不能按帧号删控制行。
        np.testing.assert_array_equal(reader["timing/rows"]["camera_sequence"], [0, 0, 1])


def test_canonical_export_real_fk_and_thirteen_fields(tmp_path, capture):
    _, path = record(tmp_path / "raw", capture)
    output = tmp_path / "canonical.zarr"
    result = export_zarr(path, output, processing=ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
    )))
    assert result["total_frames"] == 3
    root = zarr.open_group(str(output), mode="r")
    specs = canonical_array_specs(32, 32, 32, 1024)
    assert len(specs) == 13
    assert set(root["data"].array_keys()) == set(specs)
    for name, (shape, dtype) in specs.items():
        array = root["data"][name]
        assert array.shape == (3, *shape)
        assert array.dtype == dtype
        np.testing.assert_array_equal(array[0], array[1])
    np.testing.assert_array_equal(root["data/finger_tactile_force"][0], np.arange(15).reshape(5, 3))
    np.testing.assert_array_equal(root["data/taxel_tactile_force"][0], np.arange(1800).reshape(5, 120, 3))
    assert root.attrs["source_id"]
    assert root.attrs["dt"] == .1
    assert "timing_admission" not in root.attrs


@pytest.mark.parametrize("source", ["policy_rollout", "other"])
def test_export_rejects_non_teleop_raw(tmp_path, capture, source):
    _, path = record(tmp_path / "raw", capture)
    with h5py.File(path / "data.h5", "r+") as data:
        data["meta"].attrs["collection_source"] = source
    with RawReader(path) as reader, pytest.raises(ValueError, match=f"got collection_source={source!r}"):
        validate_export_episode(reader)
    output = tmp_path / "canonical.zarr"
    with pytest.raises(ValueError, match=f"got collection_source={source!r}"):
        export_zarr(path, output, processing=ExportConfig())
    assert not output.exists()
    assert not list(tmp_path.glob(".canonical.zarr.tmp-*"))


@pytest.mark.parametrize("recording", [False, True])
def test_soft_timeout_summary_keeps_raw_nanoseconds(tmp_path, capture, caplog, recording):
    from test_interaction import begin, make_loop

    loop = make_loop()
    begin(loop)
    if recording:
        metadata, row, command = capture
        recorder = loop.recorder = EpisodeRecorder(tmp_path, control_hz=10)
        recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
        recorder.append(row(0), command)
    caplog.set_level("INFO", logger="dexmani_real.teleop.loop")
    loop.record_soft_timeout("observation", deadline_ns=1_000_000_000, observed_ns=1_100_000_000)
    loop.record_soft_timeout("compute", deadline_ns=2_000_000_000, observed_ns=2_012_345_000)
    before_end = list(caplog.records)
    last = loop.last_soft_timeout.copy()
    loop.end_episode("operator")
    assert not any(record.getMessage().startswith("Teleop soft timeouts:") for record in before_end)
    summaries = [record for record in caplog.records
                 if record.getMessage().startswith("Teleop soft timeouts:")]
    assert len(summaries) == 1
    assert summaries[0].levelname == "INFO"
    assert summaries[0].getMessage().endswith("last_lateness_ms=12.345")
    expected = {"stage": "soft_timeouts", "count": 2, "last": last}
    details = [item for item in loop.termination_details if item["stage"] == "soft_timeouts"]
    assert details == ([expected] if recording else [])
    assert set(last) == {"stage", "tick", "deadline_ns", "observed_ns"}
    if recording:
        with RawReader(recorder.published_path) as reader:
            persisted = json.loads(reader.data_file["meta/termination_details"][()])
        assert next(item for item in persisted if item["stage"] == "soft_timeouts") == expected
    assert loop.soft_timeout_count == 0
    assert loop.last_soft_timeout is None
    loop.robot.stop.assert_called_once()
    assert [call.args[0] for call in loop.audio.play.call_args_list] == (["save"] if recording else [])


def test_export_prefix_still_checks_raw_field_shape(tmp_path, capture):
    _, path = record(tmp_path, capture)
    with h5py.File(path / "data.h5", "r+") as data:
        del data["action_hand_joint_target"]
        data.create_dataset("action_hand_joint_target", data=np.zeros((3, 11)))
    with RawReader(path) as reader, pytest.raises(ValueError, match="incompatible shape/dtype"):
        validate_export_episode(reader)


@pytest.mark.parametrize("name,value", [("hand_tactile_units", "N"),
                                        ("hand_tactile_finger_order", "pinky,ring,middle,index,thumb"),
                                        ("hand_tactile_hold_max_age_s", float("nan"))])
def test_tactile_numeric_semantics_still_required(tmp_path, capture, name, value):
    _, path = record(tmp_path, capture)
    with h5py.File(path / "data.h5", "r+") as data:
        data["meta"].attrs[name] = value
    with RawReader(path) as reader, pytest.raises(ValueError):
        reader.validate_tactile_metadata()


def test_raw_close_attempts_all_resources_and_raises_first(caplog):
    writer = RawWriter("unused", control_hz=10)
    primary, secondary = OSError("video failed"), OSError("hdf5 failed")
    writer.video = Mock(close=Mock(side_effect=primary))
    writer.data_file = Mock(close=Mock(side_effect=secondary))
    with pytest.raises(OSError) as caught:
        writer.close()
    assert caught.value is primary
    writer.data_file.close.assert_called_once()
    assert "hdf5_close also failed" in caplog.text
    assert not hasattr(primary, "cleanup_errors")


def test_reader_context_preserves_primary_and_reports_close(tmp_path, capture, monkeypatch, caplog):
    _, path = record(tmp_path, capture)
    reader = RawReader(path)
    real_close = reader.close

    def failed_close():
        real_close()
        raise OSError("injected reader close failure")

    monkeypatch.setattr(reader, "close", failed_close)
    primary = ValueError("input failure")
    with pytest.raises(ValueError) as caught:
        with reader:
            raise primary
    assert caught.value is primary
    assert "injected reader close failure" in caplog.text


def test_recording_first_cause_and_secondary_failure_are_flat(caplog):
    recorder = EpisodeRecorder("unused", control_hz=10)
    primary, secondary = OSError("write failure"), OSError("close failure")
    recorder.store_error(primary)
    recorder.store_error(secondary, stage="writer_close")
    with pytest.raises(RecordingError) as caught:
        recorder.check_error()
    assert caught.value.__cause__ is primary
    assert error_detail("recording", caught.value)["message"] == (
        "Recording failed: OSError: write failure"
    )
    with pytest.raises(RecordingError) as repeated:
        recorder.check_error()
    assert repeated.value is caught.value
    assert repeated.value.__cause__ is primary
    assert not hasattr(primary, "cleanup_errors")
    assert "writer_close also failed" in caplog.text


def test_exited_writer_can_publish_after_supervisor_wakes_late(tmp_path, capture, monkeypatch):
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    recorder.append(row(0), command)
    now = [0.0]
    join = recorder.thread.join

    def delayed_join(timeout):
        join(timeout)
        assert not recorder.writer_pending
        now[0] = recorder_module.RECORDER_STOP_TIMEOUT_S + 1

    monkeypatch.setattr(recorder.thread, "join", delayed_join)
    monkeypatch.setattr(recorder_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    path = recorder.finish(save=True, reason="test", technical_status="normal")
    assert recorder.result()["status"] == "published"
    with RawReader(path) as reader:
        assert reader.num_frames == recorder.written_frames == 1
        reader.validate_completion()
        np.testing.assert_array_equal(reader["action_arm_joint_target"][0], command.arm_qpos)
        assert len(list(reader.iter_rgb_frames())) == 1


@pytest.mark.parametrize("failure", ["writer", "queue"])
def test_finish_timeout_stays_failed_after_writer_exits(tmp_path, capture, monkeypatch, failure):
    entered, release = Event(), Event()
    append = RawWriter.append_frame

    def blocked_append(writer, *args):
        entered.set()
        assert release.wait(5)
        return append(writer, *args)

    monkeypatch.setattr(RawWriter, "append_frame", blocked_append)
    monkeypatch.setattr(recorder_module, "RECORDER_STOP_TIMEOUT_S", 0)
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    try:
        recorder.append(row(0), command)
        assert entered.wait(5)
        if failure == "queue":
            for index in range(1, 17):
                recorder.append(row(index), command)
        message = "finish queue timed out" if failure == "queue" else "writer did not exit"
        with pytest.raises(RecordingError, match=message) as caught:
            recorder.finish(save=True, reason="test", technical_status="normal")
        assert recorder.writer_pending
    finally:
        release.set()
        recorder.thread.join(5)
    assert not recorder.writer_pending
    assert recorder.published_path is None and recorder.staging_path.is_dir()
    with pytest.raises(RecordingError) as repeated:
        recorder.finish(save=True, reason="test", technical_status="normal")
    assert repeated.value is caught.value


@pytest.mark.parametrize("overwrite", [False, True])
def test_atomic_write_preserves_publish_error_when_cleanup_fails(
    tmp_path, monkeypatch, caplog, overwrite,
):
    primary = OSError("publication failed")
    cleanup = OSError("temporary cleanup failed")
    boundary = "replace" if overwrite else "publish"
    owner = io.os if overwrite else io
    with monkeypatch.context() as patch:
        patch.setattr(owner, boundary, Mock(side_effect=primary))
        patch.setattr(io.Path, "unlink", Mock(side_effect=cleanup))
        with pytest.raises(OSError) as caught:
            io.write_json({"result": "failed"}, tmp_path / "result.json", overwrite=overwrite)
    assert caught.value is primary
    cleanup_logs = [entry for entry in caplog.records if entry.exc_info and entry.exc_info[1] is cleanup]
    assert len(cleanup_logs) == 1
    assert "Traceback" in caplog.text


@pytest.mark.parametrize("overwrite", [False, True])
def test_successful_atomic_write_consumes_temporary_without_cleanup(tmp_path, monkeypatch, overwrite):
    target = tmp_path / "result.json"
    if overwrite:
        target.write_text("previous result")
    with monkeypatch.context() as patch:
        cleanup = Mock(side_effect=AssertionError("successful publication needs no unlink"))
        patch.setattr(io.Path, "unlink", cleanup)
        assert io.write_json({"result": "saved"}, target, overwrite=overwrite) == target
        cleanup.assert_not_called()
    assert target.read_text() == '{\n  "result": "saved"\n}'
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_publication_refuses_existing_target_and_removes_temporary(tmp_path):
    target = tmp_path / "result.json"
    target.write_text("original result")
    with pytest.raises(FileExistsError):
        io.write_json({"result": "replacement"}, target, overwrite=False)
    assert target.read_text() == "original result"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("failure", ["flush", "count"])
def test_writer_failure_does_not_publish(tmp_path, capture, monkeypatch, failure):
    original_finish = RawWriter.finish

    def fail_finish(self, reason, details, *, expected_frames, technical_status):
        if failure == "flush":
            raise OSError("injected flush failure")
        return original_finish(self, reason, details, expected_frames=expected_frames + 1,
                               technical_status=technical_status)

    if failure == "flush":
        monkeypatch.setattr(RawWriter, "flush", Mock(side_effect=OSError("injected flush failure")))
    else:
        monkeypatch.setattr(RawWriter, "finish", fail_finish)
    with pytest.raises(RecordingError):
        record(tmp_path, capture)
    assert all(path.name.startswith(".tmp_") for path in tmp_path.iterdir())
    assert list(tmp_path.glob(".tmp_*/data.h5"))


def test_queue_full_keeps_staging_and_original_cause(tmp_path, capture, monkeypatch):
    entered, release = Event(), Event()
    original_append = RawWriter.append_frame

    def blocked_append(self, *args):
        entered.set()
        assert release.wait(5)
        return original_append(self, *args)

    monkeypatch.setattr(RawWriter, "append_frame", blocked_append)
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    try:
        recorder.append(row(0), command)
        assert entered.wait(5)
        for index in range(1, 17):
            recorder.append(row(index), command)
        with pytest.raises(RecordingError, match="queue full"):
            recorder.append(row(17), command)
    finally:
        release.set()
        with pytest.raises(RecordingError, match="queue full"):
            recorder.finish(save=True, reason="test")
    assert recorder.published_path is None
    assert recorder.staging_path.is_dir()
    assert not recorder.writer_pending


@pytest.mark.parametrize("write_fails", [False, True])
def test_writer_close_failure_keeps_staging_and_original_cause(
    tmp_path, capture, monkeypatch, caplog, write_fails,
):
    primary, cleanup = OSError("injected write failure"), OSError("injected close failure")
    original_close = RawWriter.close

    def failed_close(writer):
        original_close(writer)
        raise cleanup

    monkeypatch.setattr(RawWriter, "close", failed_close)
    if write_fails:
        monkeypatch.setattr(RawWriter, "append_frame", Mock(side_effect=primary))
    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label="fixture", collection_source="teleop", metadata=metadata)
    recorder.append(row(0), command)
    with pytest.raises(RecordingError) as caught:
        recorder.finish(save=True, reason="test", technical_status="normal")
    assert caught.value.__cause__ is (primary if write_fails else cleanup)
    assert not recorder.writer_pending and recorder.published_path is None
    assert recorder.staging_path.is_dir()
    assert not recorder.writer.data_file.id.valid
    assert recorder.writer.video.container is None
    if write_fails:
        assert any(record.exc_info and record.exc_info[1] is cleanup for record in caplog.records)


@pytest.mark.parametrize('label', [None, 17, '', '   ', '\u3000', *('demo' + chr(c) for c in (*range(32), 127))])
def test_task_label_rejected_before_recording_side_effects(tmp_path, label):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.data.raw import validate_task_label
    from dexmani_real.deployment.run import session_directory
    from dexmani_real.teleop.run import run_teleop

    with pytest.raises(ValueError, match='task_label'):
        validate_task_label(label)
    destination = tmp_path / 'new'
    recorder = EpisodeRecorder(destination, control_hz=10)
    with pytest.raises(RecordingError) as caught:
        recorder.start(task_label=label, collection_source='teleop', metadata={})
    assert isinstance(caught.value.__cause__, ValueError)
    assert not destination.exists() and recorder.thread is None
    info = SimpleNamespace(task_name=label, policy_name='policy', experiment_dir=tmp_path / 'experiment')
    with pytest.raises(ValueError, match='task_label'):
        session_directory(destination, info)
    assert not destination.exists()
    with pytest.raises(ValueError, match='task_label'):
        run_teleop(RealConfig(), record=True, episodes_dir=str(destination),
                   max_duration_s=1, runtime_config_json=None, task_name=label)
    assert not destination.exists()


@pytest.mark.parametrize('label', ['抓取 toy', ' pick/place ', '.natural language', 'a\\b'])
def test_natural_language_labels_round_trip_without_directory_rules(tmp_path, capture, label):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.teleop.run import run_teleop

    metadata, row, command = capture
    recorder = EpisodeRecorder(tmp_path / 'raw', control_hz=10)
    recorder.start(task_label=label, collection_source='teleop', metadata=metadata)
    for index in range(3):
        recorder.append(row(index), command)
    path = recorder.finish(save=True, reason='test', technical_status='normal')
    output = tmp_path / 'export.zarr'
    export_zarr(path, output, processing=ExportConfig(pointcloud=PointCloudConfig(
        remove_table=False, num_points=32, workspace=(-1., -1., 0., 1., 1., 1.),
    )))
    assert zarr.open_group(str(output), mode='r').attrs['task_name'] == label
    if label != '抓取 toy':
        with pytest.raises(ValueError, match='task_name'):
            run_teleop(RealConfig(), record=True, episodes_dir=str(tmp_path),
                       max_duration_s=1, runtime_config_json=None, task_name=label)


@pytest.mark.parametrize('label', [17, '   ', 'demo\x01', 'demo\x1f', 'demo\x7f'])
def test_export_rejects_invalid_stored_task_label(tmp_path, capture, label):
    _, path = record(tmp_path / 'raw', capture)
    # 模拟历史 Raw 的外部输入；只修改测试临时产物。
    with h5py.File(path / 'data.h5', 'r+') as data:
        data['meta'].attrs['task_label'] = label
    with RawReader(path) as reader, pytest.raises(ValueError, match='task_label'):
        validate_export_episode(reader)
    output = tmp_path / 'invalid.zarr'
    with pytest.raises(ValueError, match='task_label'):
        export_zarr(path, output, processing=ExportConfig())
    assert not output.exists()


def test_model_history_build_preserves_recording_input(tmp_path, capture):
    from dexmani_real.config.real import RealConfig
    from dexmani_real.deployment.observation import ObservationBuilder

    metadata, make_row, command = capture
    row = make_row(0)
    fields = ('rgb', 'depth', 'finger_tactile_force', 'taxel_tactile_force', 'joint_state')
    builder = ObservationBuilder(SimpleNamespace(observation_fields=fields), RealConfig())
    history = builder.project(row)
    recorder = EpisodeRecorder(tmp_path, control_hz=10)
    recorder.start(task_label='model and recording', collection_source='teleop', metadata=metadata)
    recorder.append(row, command)
    for value in builder.build((history,)).values():
        value.fill(0)
    path = recorder.finish(save=True, reason='test', technical_status='normal')
    np.testing.assert_array_equal(row.camera['rgb'], 80)
    with RawReader(path) as reader:
        for recorded, expected in (
            ('depth', row.camera['depth']),
            ('hand_contact', row.hand['tactile_aggregate'][0]),
            ('hand_tactile_force', row.hand['tactile_dense'][0]),
            ('arm_qpos', row.arm['qpos'][0]),
        ):
            np.testing.assert_array_equal(reader[recorded][0], expected)
        assert reader['timing/rows'][0]['camera_ns'] == row.camera['timestamp_ns']
